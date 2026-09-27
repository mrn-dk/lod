"""Source-table rows 2 and 3 — bigbench and Super-NaturalInstructions.

Both are large schema sources and neither fits the generic column adapter, for opposite
reasons.

**Row 2, bigbench.** The answer is not in a label column at all: each example carries its
own `multiple_choice_targets` list with `multiple_choice_scores` marking the correct one.
That makes it the corpus's main source of *per-example option sets* — the corpus wants at
least ten schemas of that shape, and a question whose options change from row to row is the
purest test of whether the model reads the option list rather than memorising a label
vocabulary. 167 configs, one schema each.

That option list is *curated*, though, and the curation carries the answer: measured on
an earlier build, the gold option sat at index 0 in 39.4 % of row 2's 22,439 questions
against a 31.4 % chance baseline, and in 20 of the 114 tasks one fixed index was the
answer at least 90 % of the time -- 17 of them at exactly 100 %. `_shuffled` permutes each
example's options under a key drawn from the example's own text, so the order carries
nothing and the permutation survives a rebuild. Rows 36 and 43 already assert the gold
option is not at a fixed index; row 2 was the one that could not.

**Row 3, Super-NaturalInstructions.** The labels are free-text `targets` and the schema
only appears after grouping: a task is a `task_name`, its question is the `definition`,
and its option set is the distinct targets it ever produces. Row 3 keeps tasks with
<= 30 distinct outputs, which is what separates a classification task from a generation
task that happens to have short answers.

A task whose option set is the set of observed answers has a trap worth stating: if a
task appears with only two answered examples, its "option set" is those two answers and
the question is trivially easy. `MIN_EXAMPLES_PER_TASK` guards against that.
"""

from __future__ import annotations

import hashlib
import random
import re
from collections import defaultdict
from typing import Iterator

from lod.schema import Example, Question
from lod.corpus.services.sources.base import CACHE
from lod.corpus.services.sources.real.base import RealTask

BIGBENCH = "tasksource/bigbench"
NI = "Muennighoff/natural-instructions"
LICENCE_BB = "Apache-2.0"
LICENCE_NI = "Apache-2.0"

MAX_OUTPUTS = 30           # row 3
MIN_EXAMPLES_PER_TASK = 40  # below this the observed-answer option set is degenerate
MIN_OPTIONS = 2
NI_SCAN = 400_000          # rows of NI to scan when discovering tasks


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_").lower()[:60]


def _shuffled(options: list[str], target: int, key: str) -> tuple[list[str], int]:
    """Permute one example's options, carrying its target, deterministically.

    bigbench publishes `multiple_choice_targets` in a *curated* order, and many configs
    curate the correct answer into a fixed slot. Measured on an earlier build: the gold
    option sits at index 0 in 39.4 % of all 22,439 row-2 questions against a 31.4 % chance
    baseline, and in 20 of the 114 tasks one fixed index is the answer at least 90 % of
    the time -- 17 of them at exactly 100 % (`bb_which_wiki_edit`, `bb_temporal_sequences`,
    `bb_question_selection`, `bb_undo_permutation`, `bb_hhh_alignment`, ...), 4,142
    questions. On `testreal` "always answer option 0" scored 0.498 against 0.344.

    Training reshuffles options (`data.shuffle_options`), so this never inflated a
    reported number; but the corpus files themselves were solvable by position, and
    every consumer that does not augment -- `scripts/evaluate.py`, the confidence head with
    `--criteria-augment` off, any external reader -- saw that. Row 36 and row 43 both
    assert the gold option is not at a fixed index; row 2 is the only one that could not.

    The key is the example's own text, not its ordinal, so the permutation is stable
    under a different quota, a re-fetch or a resumed stream.
    """
    order = list(range(len(options)))
    seed = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big")
    random.Random(seed).shuffle(order)
    return [options[i] for i in order], order.index(target)


# ---- row 2: bigbench ---------------------------------------------------------

def bigbench_configs() -> list[str]:
    from datasets import get_dataset_config_names
    try:
        return sorted(get_dataset_config_names(BIGBENCH))
    except Exception:
        return []


# Longest state a bigbench example may have. Longer ones are DROPPED, not cut: bigbench
# puts the question at the END of `inputs` ("What convinced Fordney it was murder?"), so
# the old `state[:4000]` cut the question itself off 112 of 113 `minute_mysteries_qa`
# states and 118 of 261 `which_wiki_edit` ones -- a question with no question in it.
# 6,000 characters sits inside the packer's 2,304-token state budget with room to spare.
MAX_BB_STATE_CHARS = 6000

# Configs whose state no text reader can use. `cifar10_classification` is a base64 PNG
# ("iVBORw0KGgo..."): the same opaque-blob state `quality.py` rejects for row 1, which
# its gate cannot see here only because bigbench prefixes the blob with one sentence.
SKIP_CONFIGS = {
    "cifar10_classification": "state is a base64-encoded PNG",
    # The answer is a world fact the state does not contain: memorisation, not reading
    # (domain-4 audit, 2,030 questions). The corpus invariant is that the target is
    # derivable from the state.
    **{c: "world fact not in the state" for c in (
        "anachronisms", "cryobiology_spanish", "emoji_movie", "fact_checker",
        "general_knowledge", "hindu_knowledge", "human_organs_senses",
        "identify_math_theorems", "known_unknowns", "misconceptions",
        "movie_recommendation", "periodic_elements", "physics", "sports_understanding",
        "strategyqa", "english_russian_proverbs", "swahili_english_proverbs",
        "swedish_to_german_proverbs", "persian_idioms")},
}

# Decision Index overlap: bigbench examples whose question is an MMLU *test* question
# (MMLU is in the Decision Index). Matched on the normalised opening words of the
# question; found by token Jaccard >= 0.97 against the DI's MMLU test parquet in a
# domain-4 audit. Both are `physics` word problems lifted from
# high_school_physics.
BENCHMARK_OVERLAP = {
    "physics": (
        "a car starts from rest and uniformly accelerates to a final speed of 20 0 m s in "
        "a time of 15 0 s how far does the car travel during this time",
        "a body moving in the positive x direction passes the origin at time t 0 between "
        "t 0 and t 1 second the body has a constant speed of 24 meters per second",
    ),
}

_CHOICE_LINE = re.compile(r"^[ \t]*choice:[ \t]*(.*?)[ \t]*$", re.M)
# SAN check / mate markers. In `checkmate_in_one` the gold move is written `Qc3#` and no
# distractor carries a `#`, so the mark alone answered 257 of 257 questions.
_SAN_MARKS = re.compile(r"[+#]+$")
OPTION_NORMALISERS = {
    "checkmate_in_one": lambda o: _SAN_MARKS.sub("", o.strip()),
}


def _words(s: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", s.lower()))


def _overlaps_benchmark(config: str, state: str) -> bool:
    text = _words(state)
    return any(q in text for q in BENCHMARK_OVERLAP.get(config, ()))


def _strip_choices(state: str, options: list[str]) -> str:
    """Remove the `choice: X` lines bigbench prints into `inputs`.

    They repeat the option list the question already carries, in bigbench's own order --
    and that order is not always neutral: the gold option is the first `choice:` line in
    80.6 % of `bbq_lite_json` states and 82.8 % of `social_support` ones (3 options,
    chance 33 %), so shuffling the option list did not remove the position cue, it only
    moved it into the state. Only lines naming one of this example's options are removed.
    """
    opts = {o.strip() for o in options}
    kept = [ln for ln in state.split("\n")
            if not ((m := _CHOICE_LINE.fullmatch(ln)) and m.group(1) in opts)]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()


def _bigbench_loader(config: str):
    norm = OPTION_NORMALISERS.get(config)

    def load(n: int) -> Iterator[Example]:
        from datasets import load_dataset

        ds = load_dataset(BIGBENCH, config, split="train", streaming=True)
        made = 0
        seen: set[tuple[str, tuple[str, ...]]] = set()
        for r in ds:
            if made >= n:
                return
            opts = [str(o) for o in (r.get("multiple_choice_targets") or [])]
            scores = list(r.get("multiple_choice_scores") or [])
            state = str(r.get("inputs") or "").strip()
            if len(opts) < MIN_OPTIONS or len(scores) != len(opts) or not state:
                continue
            if sum(1 for s in scores if s) != 1:
                continue          # no single correct answer: not a decision question
            state = _strip_choices(state, opts)
            if norm is not None:
                opts = [norm(o) for o in opts]
            if len({o.strip().casefold() for o in opts}) != len(opts):
                continue          # a repeated option (case aside) makes the target ambiguous
            if not state or len(state) > MAX_BB_STATE_CHARS:
                continue          # dropped, never cut: the question sits at the end
            if _overlaps_benchmark(config, state):
                continue          # a Decision Index test question
            key = (" ".join(state.split()), tuple(sorted(opts)))
            if key in seen:
                continue          # the same question twice is one question
            seen.add(key)
            opts, target = _shuffled(opts, scores.index(max(scores)),
                                     key=f"{config}\x00{state}\x00{'|'.join(opts)}")
            made += 1
            yield Example(
                task=f"bb_{_slug(config)}",
                state=state,
                questions=[Question(id="answer",
                                    question="Which option is correct?",
                                    options=opts,
                                    target=target)],
            )
    return load


def bigbench_tasks(limit: int | None = None) -> list[RealTask]:
    cfgs = [c for c in bigbench_configs() if c not in SKIP_CONFIGS]
    if limit:
        cfgs = cfgs[:limit]
    return [RealTask(row=2, name=f"bb_{_slug(c)}", licence=LICENCE_BB,
                     url=f"https://huggingface.co/datasets/{BIGBENCH}",
                     load=_bigbench_loader(c), per_example_options=True,
                     notes=f"bigbench {c}; options vary per example")
            for c in cfgs]


# ---- row 3: Super-NaturalInstructions ----------------------------------------

_NI_CACHE = CACHE / "ni_tasks.jsonl"
_NI_ROWS = CACHE / "ni_rows.jsonl"
# Read once per process. _ni_loader is called once per task and previously re-streamed
# the whole dataset from the Hub each time -- the fourth place this same repeated-read
# bug appeared, after the network fetch, the raw store and the gharchive cache.
_IN_PROCESS: dict[str, object] = {}


def ni_tasks_index(scan: int = NI_SCAN, refresh: bool = False) -> dict[str, dict]:
    """Group NI by task, keeping those that behave like classification.

    The option set is the distinct `targets` a task produces, so a task is only usable
    once enough of its examples have been seen for that set to be stable -- hence the
    minimum example count as well as the maximum output count.
    """
    import json

    if _NI_CACHE.exists() and not refresh:
        return {r["task"]: r for r in
                (json.loads(l) for l in _NI_CACHE.read_text().splitlines() if l.strip())}

    from datasets import load_dataset

    ds = load_dataset(NI, split="train", streaming=True)
    outputs: dict[str, set] = defaultdict(set)
    counts: dict[str, int] = defaultdict(int)
    definition: dict[str, str] = {}
    for i, r in enumerate(ds):
        if i >= scan:
            break
        task = r.get("task_name")
        tgt = (r.get("targets") or "").strip()
        if not task or not tgt or len(tgt) > 60:
            continue
        counts[task] += 1
        if len(outputs[task]) <= MAX_OUTPUTS + 1:
            outputs[task].add(tgt)
        definition.setdefault(task, (r.get("definition") or "").strip())

    keep = {}
    for task, outs in outputs.items():
        if (MIN_OPTIONS <= len(outs) <= MAX_OUTPUTS
                and counts[task] >= MIN_EXAMPLES_PER_TASK and definition.get(task)):
            keep[task] = {"task": task, "options": sorted(outs),
                          "definition": definition[task], "n": counts[task]}

    CACHE.mkdir(parents=True, exist_ok=True)
    _NI_CACHE.write_text("".join(json.dumps(v) + "\n" for v in keep.values()))
    return keep


def ni_rows_by_task(scan: int = NI_SCAN, refresh: bool = False) -> dict[str, list[dict]]:
    """Every usable NI row, grouped by task, materialised once.

    The loader is called once per task; streaming the dataset inside it meant re-reading
    up to `scan` rows from the Hub for each one. Cached to disk so a rebuild is local,
    and grouped in memory so the build is one pass rather than one pass per task.
    """
    import json

    if "by_task" in _IN_PROCESS:
        return _IN_PROCESS["by_task"]          # type: ignore[return-value]

    if _NI_ROWS.exists() and not refresh:
        rows = [json.loads(l) for l in _NI_ROWS.read_text().splitlines() if l.strip()]
    else:
        from datasets import load_dataset

        keep = set(ni_tasks_index())
        ds = load_dataset(NI, split="train", streaming=True)
        rows = []
        for i, r in enumerate(ds):
            if i >= scan:
                break
            if r.get("task_name") not in keep:
                continue
            rows.append({"task_name": r["task_name"],
                         "inputs": (r.get("inputs") or "")[:4000],
                         "targets": (r.get("targets") or "").strip()})
        CACHE.mkdir(parents=True, exist_ok=True)
        _NI_ROWS.write_text("".join(json.dumps(r) + "\n" for r in rows))

    idx: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        idx[r["task_name"]].append(r)
    _IN_PROCESS["by_task"] = idx
    return idx


# The definition is the question, and it is where NI says what each output means. The old
# 400-character cut removed exactly that from `task043`: "E" is "I don't know", the gold
# on 974 of its 1,742 rows, and the sentence saying so starts at character 380 (domain-1
# audit).
# Cut at a sentence end instead, and late enough to keep every kept task's definition.
MAX_DEFINITION_CHARS = 1200
# An input NI answers two ways is a generation task, or a task whose definition admits
# several right answers: `task082` writes one of several valid questions per passage (944
# of 992 inputs carry two targets), `task046` "may have more than one correct type"
# (3,572 of 6,498). A task this ambiguous is not a classification task.
NI_AMBIGUOUS_SHARE = 0.2
# NI tasks built on a Decision Index benchmark: task043 and task047 pose ARC questions,
# ARC test items among them (17/1,742 and 10/251 states matched ARC test verbatim).
NI_BENCHMARK_OVERLAP = {"task043_essential_terms_answering_incomplete_questions",
                        "task047_miscellaneous_answering_science_questions"}


def _clip_definition(text: str, limit: int = MAX_DEFINITION_CHARS) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    cut = text[:limit]
    end = max(cut.rfind(". "), cut.rfind(".\n"))
    return cut[: end + 1] if end > limit // 2 else cut


def _ni_targets_by_input(task: str) -> dict[str, set[str]]:
    out: dict[str, set[str]] = defaultdict(set)
    for r in ni_rows_by_task().get(task, []):
        state = str(r.get("inputs") or "").strip()
        if state:
            out[state].add((r.get("targets") or "").strip())
    return out


def _ni_loader(task: str, options: list[str], definition: str, scan: int = NI_SCAN):
    def load(n: int) -> Iterator[Example]:
        ambiguous = {s for s, t in _ni_targets_by_input(task).items() if len(t) > 1}
        made = 0
        seen: set[str] = set()
        for r in ni_rows_by_task().get(task, []):
            if made >= n:
                return
            tgt = (r.get("targets") or "").strip()
            state = str(r.get("inputs") or "").strip()
            if tgt not in options or not state or state in ambiguous or state in seen:
                continue
            seen.add(state)
            made += 1
            yield Example(
                task=f"ni_{_slug(task)}",
                state=state[:4000],
                questions=[Question(id="answer", question=_clip_definition(definition),
                                    options=list(options), target=options.index(tgt))],
            )
    return load


def ni_ambiguous_share(task: str) -> float:
    by_input = _ni_targets_by_input(task)
    return sum(1 for t in by_input.values() if len(t) > 1) / max(1, len(by_input))


def ni_real_tasks(limit: int | None = None) -> list[RealTask]:
    idx = ni_tasks_index()
    items = sorted(idx.values(), key=lambda v: -v["n"])
    if limit:
        items = items[:limit]
    items = [v for v in items if ni_ambiguous_share(v["task"]) < NI_AMBIGUOUS_SHARE
             and v["task"] not in NI_BENCHMARK_OVERLAP]
    return [RealTask(row=3, name=f"ni_{_slug(v['task'])}", licence=LICENCE_NI,
                     url=f"https://huggingface.co/datasets/{NI}",
                     load=_ni_loader(v["task"], v["options"], v["definition"]),
                     notes=f"{v['task']}: {len(v['options'])} distinct outputs")
            for v in items]


def tasks() -> list[RealTask]:
    """Both rows. Returns [] when nothing is cached, so the registry stays offline."""
    out: list[RealTask] = []
    if _NI_CACHE.exists():
        out += ni_real_tasks()
    return out
