"""Source rows -> questions: quotas, deterministic sampling, split routing, dedup, val.

The source readers (`services/sources/`) own label semantics: what a row means, which
options it has, what its target is. This module owns everything that makes a corpus out
of them, and nothing here reads a label:

- `task_quotas`      a bounded, row-stratified question budget per task;
- `sample_examples`  a deterministic sample of at most that many examples;
- `build`            per task: sample, attach published option definitions, drop
                     unanswerable schemas and Decision Index items, route to a split;
                     then soft targets from published probabilities, cross-split dedup,
                     a per-split cap on any one source row, and the val carve-out;
- `carve_val`        ~5 % of each train task, by group, so twins stay together.

Every random choice is seeded from `seed` and the task's position, so the same
arguments over the same raw store produce the same bytes.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Callable

from lod.corpus.services.decontaminate import Blocklist
from lod.corpus.services.sources.real import descriptions, quality
from lod.corpus.services.sources.real.base import (
    dedup_against_train,
    family_of,
    shape_of,
    split_of,
)
from lod.schema import Example

# where a task is routed before val is carved out of train
ROUTED = ("train", "devreal", "testreal")
ROWS = range(1, 52)
# rows read through the Hub dataset table (`sources/real/table.py`, `hub_sweep.py`)
HUB_ROWS = frozenset({1, 2, 3, 4, 6, 7, 8, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21,
                      22, 23, 24, 25})


def registered_tasks(rows: set[int] | None = None) -> list:
    """Every task the source table offers, or only those of `rows`. The raw store must
    already point at the fetched rows (`raw_store.configure`)."""
    from lod.corpus.services.sources.real import REGISTRY

    tasks = REGISTRY(
        include_hf_table=rows is None or bool(rows & HUB_ROWS),
        include_bigbench=rows is None or 2 in rows,
        rows=rows,
    )
    if rows is not None:
        tasks = [task for task in tasks if task.row in rows]
    return tasks


def task_quotas(tasks, target_questions: int, sample_per_task: int,
                generated_share: float = 0.30) -> dict[str, int]:
    """Allocate a bounded, source-stratified question budget across schemas.

    Rows 1-3 get up to 15 % of `target_questions` each and every other row 10 %, split
    evenly over the row's tasks and capped at `sample_per_task`.

    `generated_share` bounds what fraction of the budget the code-labelled rows may take
    between them; 1.0 is no bound at all, and is what the release corpus uses. A
    code-derived label is exact and answerable from its state by construction, so there
    is no reason to hold generated data to a quota -- but the share is the cheapest A/B
    on whether it pulls its weight, so it stays settable, and `summarize` reports the
    measured share per split either way.
    """
    by_row: dict[int, list] = defaultdict(list)
    for task in tasks:
        by_row[task.row].append(task)
    generated_rows = {row for row, group in by_row.items() if not group[0].real}

    quotas: dict[str, int] = {}
    for row, group in by_row.items():
        row_budget = max(len(group), int(target_questions * (0.15 if row <= 3 else 0.10)))
        each = max(1, row_budget // len(group))
        for task in group:
            # a task that pools the whole row asks for more than one repo's share
            quotas[task.name] = min(sample_per_task, max(each, getattr(task, "min_quota", 0)))

    # then, only if asked, hold the generated rows to a share of the whole, scaling them
    # together so no single generator is privileged by having enumerated more tasks
    gen = sum(quotas[t.name] for t in tasks if t.row in generated_rows)
    real = sum(quotas[t.name] for t in tasks if t.row not in generated_rows)
    allowed = int(real * generated_share / max(1e-9, 1.0 - generated_share))
    # a share of nothing is nothing: a build of one generated row has no real rows, and
    # scaling to a 0-question allowance would hand every task a quota of 1. The group
    # cap only applies where there is a group to take a share of.
    if real and gen > allowed and gen > 0:
        scale = allowed / gen
        for task in tasks:
            if task.row in generated_rows:
                quotas[task.name] = max(1, int(quotas[task.name] * scale))
    return quotas


def sample_examples(task, limit: int, seed: int) -> list[Example]:
    """Load at most limit examples, then sample deterministically without replacement."""
    examples = list(task.load(limit))
    if len(examples) > limit:
        examples = random.Random(seed).sample(examples, limit)
    for example in examples:
        example.task = task.name
    return examples


def soften_meta_p(examples: list[Example]) -> int:
    """Where a source publishes a forecast probability, train on it, not on the draw.

    Rows 26-28 (forecasts, markets) publish a probability in `meta.p` beside the realised
    binary outcome. A settled market that closed at 0.62 is one draw from 0.62, not
    evidence that the answer was always yes; under a proper scoring rule the soft target
    is the one whose optimum is the truth. Trained on the outcome instead, the model
    learns to predict a coin flip and cannot recover the coin's bias.

    Only two-option questions are touched, and only where `meta.p` is present; the
    outcome stays in `meta.outcome` so nothing is lost.
    """
    softened = 0
    for example in examples:
        for question in example.questions:
            p = (question.meta or {}).get("p")
            if p is None or len(question.options) != 2:
                continue
            if isinstance(question.target, list):
                continue
            p = min(max(float(p), 0.0), 1.0)
            question.meta = dict(question.meta or {})
            question.meta["outcome"] = question.target
            question.target = [1.0 - p, p]
            softened += 1
    return softened


def cap_source_share(examples: list[Example], tasks_by_name: dict, seed: int,
                     cap: float = 0.15) -> tuple[list[Example], dict[int, int]]:
    """Hold every source row to `cap` of this split, dropping whole examples at random.

    A corpus-wide budget is not enough, because a whole-family holdout lands entirely in
    one split: a row reserved to testreal at 10 % of the corpus can be half of testreal,
    and every number read off testreal would then be a measurement of that one source.
    Capping within the split is the version of the rule that does what the rule is for.
    Examples are dropped rather than questions, because an example is what a sequence is.
    """
    by_row: dict[int, list[Example]] = defaultdict(list)
    for example in examples:
        task = tasks_by_name.get(example.task)
        by_row[task.row if task else -1].append(example)

    # The share is of the split that *results*, so the limit is a fixed point: capping a
    # row shrinks the split, which lowers the limit, which caps the row again. The loop
    # exits as soon as nothing moves.
    sizes = {row: len(group) for row, group in by_row.items()}
    for _ in range(8):
        total = sum(sizes.values())
        # a cap can never be tighter than an equal split: with 3 rows and cap=0.15 no
        # allocation satisfies it, and iterating without this floor shrinks the split to
        # one example per row
        limit = max(int(total * cap), total // max(len(sizes), 1))
        changed = False
        for row, n in sizes.items():
            if limit > 0 and n > limit:
                sizes[row] = limit
                changed = True
        if not changed:
            break

    dropped: dict[int, int] = {}
    kept: list[Example] = []
    for row, group in by_row.items():
        keep_n = sizes[row]
        if keep_n < len(group):
            rng = random.Random((seed, row).__hash__())
            dropped[row] = len(group) - keep_n
            group = rng.sample(group, keep_n)
        kept.extend(group)
    return kept, dropped


def val_group(example: Example, i: int) -> str:
    """Examples that must land on the same side of the val carve-out.

    Sources emit twins -- the supported and the refuted version of one entity fact, one
    claim over two revisions of its evidence -- and val fits the confidence head, so a
    val example whose twin trains is an answer the model has already seen. A source
    marks twins with `meta.group`; a shared claim groups too.
    """
    for q in example.questions:
        m = q.meta or {}
        if m.get("group") is not None:
            return f"g:{m['group']}"
        if m.get("claim"):
            return f"c:{m['claim']}"
    return f"i:{i}"


def carve_val(train: list[Example], seed: int) -> tuple[list[Example], list[Example]]:
    """Hold out ~5% from each train task, by group, while preserving every task schema."""
    by_task: dict[str, list[Example]] = defaultdict(list)
    for example in train:
        by_task[example.task].append(example)
    kept, val = [], []
    rng = random.Random(seed)
    for examples in by_task.values():
        if len(examples) < 20:
            kept.extend(examples)
            continue
        groups: dict[str, list[int]] = defaultdict(list)
        for i, example in enumerate(examples):
            groups[val_group(example, i)].append(i)
        keys = sorted(groups)
        rng.shuffle(keys)
        want, chosen = max(1, len(examples) // 20), set()
        for k in keys:
            if len(chosen) >= want:
                break
            if len(chosen) + len(groups[k]) > len(examples) - 1:
                continue                     # never move a whole task into val
            chosen.update(groups[k])
        kept.extend(example for i, example in enumerate(examples) if i not in chosen)
        val.extend(example for i, example in enumerate(examples) if i in chosen)
    return kept, val


@dataclass
class Generated:
    """A generated corpus in memory: its splits and what `meta.json` records."""
    splits: dict[str, list[Example]]
    meta: dict
    tasks_by_name: dict = field(repr=False, default_factory=dict)
    softened: int = 0


def build(tasks: list, *, target_questions: int = 400_000, sample_per_task: int = 2000,
          seed: int = 0, describe_share: float = 0.5, generated_share: float = 1.0,
          source_cap: float | None = 0.15, blocklist: Blocklist | None = None,
          quality_filter: bool = True, log: Callable[[str], None] = print,
          verbose: bool = False,
          on_task: Callable[[], None] | None = None) -> Generated:
    """Turn `tasks` into the train / val / devreal / testreal splits.

    Per task, in order: sample its quota; attach published option definitions to
    `describe_share` of the examples; drop the task if its schema is unanswerable (bare
    numeric options with no definition); drop every example whose state carries a
    Decision Index item; route the rest to the task's split. Then, over the whole corpus:
    published probabilities become soft targets, dev/test states that repeat a train
    state are dropped, every split is capped at `source_cap` per source row (None: no
    cap), and val is carved out of train.
    """
    quotas = task_quotas(tasks, target_questions, sample_per_task, generated_share)
    routed: dict[str, list[Example]] = {split: [] for split in ROUTED}
    metadata = []
    rejected: list[dict] = []
    tasks_by_name = {task.name: task for task in tasks}
    for index, task in enumerate(tasks):
        split = task.force_split or split_of(task.name, task.row, seed)
        quota = quotas[task.name]
        try:
            examples = sample_examples(task, quota, seed + index)
            error = None
        except Exception as exc:
            examples = []
            error = f"{type(exc).__name__}: {exc}"
            log(f"FAIL row {task.row} {task.name}: {error}")
        # published option definitions, on a share of the examples
        described = descriptions.attach(examples, seed=seed, share=describe_share)
        # ...then the answerability gate, which sees whatever descriptions exist
        if quality_filter:
            examples, reason = quality.filter_examples(
                examples, task.name, described=bool(described))
            if reason:
                rejected.append({"row": task.row, "name": task.name, "reason": reason})
                if verbose:
                    log(f"drop row {task.row:2} {task.name}: {reason}")
        # the Decision Index suite is evaluation data: no state may carry one of its items
        di_dropped = 0
        if blocklist is not None and examples:
            examples, di_dropped = blocklist.filter(examples)
            if di_dropped and verbose:
                log(f"DI overlap row {task.row:2} {task.name}: dropped {di_dropped}")
        routed[split].extend(examples)
        metadata.append({
            "di_overlap_dropped": di_dropped,
            "row": task.row, "name": task.name, "split": split,
            # the held-out unit: `select`'s devood keeps dev families that never train
            "family": family_of(task),
            "examples": len(examples), "questions": sum(len(e.questions) for e in examples),
            "quota": quota, "error": error, "licence": task.licence,
            "url": task.url, "ordinal": task.ordinal, "soft": task.soft,
            "meta_p": task.meta_p, "notes": task.notes,
            # whether the labels came from people or a real system, or from code;
            # evaluation reports the two apart
            "real": task.real,
            "described": described,
        })
        if verbose and examples:
            log(f"row {task.row:2} {task.name}: {len(examples):,} -> {split}")
        if on_task is not None:
            on_task()

    softened = sum(soften_meta_p(routed[split]) for split in ROUTED)
    # Code-labelled sources are template-generated, so a shared 10-word phrase is the
    # template, not a leak; they still get the exact-match test. Some real sources render
    # a *record* rather than prose (forecast grids, survey rows), and a 10-word window
    # inside one matches for the same reason -- worse, the shingle test would drop those
    # records non-randomly (every all-zero forecast grid, every empty evidence field) and
    # select the eval split rather than thin it. They are exempt too.
    generated = {task.name for task in tasks if not task.real or task.template_state}
    dropped = dedup_against_train(
        routed["train"], {"devreal": routed["devreal"], "testreal": routed["testreal"]},
        shingle_exempt=generated)
    capped: dict = {}
    if source_cap is not None:
        for split in ROUTED:
            routed[split], capped[split] = cap_source_share(
                routed[split], tasks_by_name, seed, source_cap)
    train, val = carve_val(routed["train"], seed)
    splits = {"train": train, "val": val, "devreal": routed["devreal"],
              "testreal": routed["testreal"]}
    meta = {"seed": seed, "dropped": dropped, "rejected": rejected, "capped": capped,
            "tasks": metadata}
    return Generated(splits=splits, meta=meta, tasks_by_name=tasks_by_name,
                     softened=softened)


def summarize(splits: dict[str, list[Example]], tasks_by_name: dict) -> list[str]:
    """Per split: sizes, soft targets, question shapes, rows, and the real share."""
    lines = []
    for split, examples in splits.items():
        questions = sum(len(example.questions) for example in examples)
        rows = Counter(tasks_by_name[e.task].row for e in examples if e.task in tasks_by_name)
        shapes: Counter = Counter()
        soft = 0
        meta_p = 0
        for example in examples:
            task = tasks_by_name.get(example.task)
            for question in example.questions:
                shapes.update(shape_of(question, task.ordinal if task else False))
                soft += isinstance(question.target, list)
                meta_p += (question.meta or {}).get("p") is not None
        real_q = sum(
            len(e.questions) for e in examples
            if (tasks_by_name.get(e.task).real if e.task in tasks_by_name else True))
        share = real_q / questions if questions else 0.0
        lines.append(f"{split}: {len(examples):,} examples, {questions:,} questions; "
                     f"soft={soft:,}, meta.p={meta_p:,}, shapes={dict(shapes)}")
        lines.append(f"  rows: {dict(sorted(rows.items()))}")
        # reported, not enforced (see `task_quotas`)
        lines.append(f"  real questions: {real_q:,}/{questions:,} = {share:.1%} "
                     f"(generated {1 - share:.1%})")
    return lines
