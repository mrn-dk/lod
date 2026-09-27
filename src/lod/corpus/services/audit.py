"""Acceptance audits of a built corpus: numbers to read before spending a GPU-second.

`sentinel_audit`  what the load-time "none of the above" augmentation actually teaches.
                  It runs in `DecisionDataset`, not at build time, so what the corpus
                  teaches is the JSONL *and* the rates; this replays `augment_criteria`
                  over a split exactly as training would. `ACCEPTANCE` holds the bars.
`domain_counts`   questions per domain and split, counted from the split files -- not
                  from `meta.json`, which records what each task *produced* before the
                  source cap and dedup removed some of it.
`verify`          corpus-level integrity counts; every one should be 0.
"""

from __future__ import annotations

import collections
import json
import random
import re
from collections import Counter, defaultdict
from typing import Iterable

from lod import sentinel
from lod.corpus.domains import DOMAINS, SPLITS

# ---- sentinel audit --------------------------------------------------------------------

#   P(correct | present)  what a model learns the sentinel means; near 0.5 it is a coin
#                         flip, and a model will answer a caller's "None of the above"
#                         with high confidence whatever the state says
#   sentinel share        how much of the corpus is distorted to teach it
#   max exact repeat      one string seen thousands of times is memorisable whatever its
#                         base rate; that is what the sentinel grammar is for
#   three ways to spot a withheld question without reading the state:
#     membership          "the task's usual answer is absent" predicts abstain
#     arity               the option count predicts whether a sentinel is present
#     position            the sentinel is always the last option
ACCEPTANCE = {
    "p_correct_given_present": 0.20,
    "sentinel_share": 0.34,
    "max_exact_repeat": 50,
    "membership_lift": 0.05,           # |P(abstain | modal gold absent) - base|
    "arity_tvd": 0.05,                 # sentinel vs not, among eligible questions
    "sentinel_last_share": 0.35,       # chance for a 3-option question is ~0.33
}
# the (p_withheld, p_decoy) trade-off curve `sentinel_sweep` walks
SWEEP = ((0.150, 0.15), (0.100, 0.30), (0.075, 0.30), (0.075, 0.36), (0.075, 0.40),
         (0.075, 0.45), (0.060, 0.30), (0.060, 0.36), (0.050, 0.25), (0.050, 0.30))


def _sentinel_index(q) -> int | None:
    """Which option, if any, is the sentinel -- by exact membership in the grammar's
    output, not a prefix test: with opaque keys the description is all there is."""
    for i, d in enumerate(q.descriptions or []):
        if sentinel.is_sentinel_description(d):
            return i
    for i, o in enumerate(q.options):
        if sentinel.is_sentinel_key(o):
            return i
    return None


def sentinel_audit(examples, p_withheld: float | None = None, p_decoy: float | None = None,
                   seed: int = 0, epoch: int = 0) -> dict:
    """Replay the augmentation over `examples` and measure what it teaches."""
    from lod.training.data import (
        P_DECOY,
        P_WITHHELD,
        _is_choice,
        augment_criteria,
        sentinel_eligible,
    )

    p_withheld = P_WITHHELD if p_withheld is None else p_withheld
    p_decoy = P_DECOY if p_decoy is None else p_decoy
    n_choice = 0
    n_present = 0
    n_correct = 0
    wordings: collections.Counter = collections.Counter()
    by_arity: dict[int, list[int]] = collections.defaultdict(lambda: [0, 0])
    by_task: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    split_counts: collections.Counter = collections.Counter()
    arity_all: dict[str, collections.Counter] = {
        "sentinel": collections.Counter(), "plain": collections.Counter()}
    # the task's modal gold option key, from the un-augmented corpus: if "the usual
    # answer is missing" predicts abstain, the question can be answered without the state
    gold_counts: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for ex in examples:
        for q in ex.questions:
            p = q.target_probs()
            if _is_choice(q) and p is not None:
                gold_counts[ex.task][str(q.options[max(range(len(p)), key=p.__getitem__)])] += 1
    modal = {t: c.most_common(1)[0][0] for t, c in gold_counts.items() if c}
    n_absent = n_absent_abstain = n_abstain = n_last = 0

    for i, ex in enumerate(examples):
        rng = random.Random((seed, epoch, i).__hash__())
        aug = augment_criteria(ex, rng, p_withheld=p_withheld, p_decoy=p_decoy)
        for orig, q in zip(ex.questions, aug.questions):
            if not _is_choice(q):
                continue
            # the arity comparison is among questions a sentinel COULD have gone on
            eligible = (_is_choice(orig) and sentinel_eligible(orig) is not None
                        # augment_criteria never touches an ordered policy
                        and not (orig.meta or {}).get("ordered_rules"))
            n_choice += 1
            si = _sentinel_index(q)
            probs0 = q.target_probs()
            abstain = (si is not None and probs0 is not None
                       and max(range(len(probs0)), key=probs0.__getitem__) == si)
            n_abstain += abstain
            if ex.task in modal and modal[ex.task] not in {str(o) for o in q.options}:
                n_absent += 1
                n_absent_abstain += abstain
            if si is None:
                if eligible:
                    arity_all["plain"][len(q.options)] += 1
                continue
            n_present += 1
            n_last += si == len(q.options) - 1
            arity_all["sentinel"][len(q.options)] += 1
            probs = q.target_probs()
            hit = probs is not None and max(range(len(probs)),
                                            key=probs.__getitem__) == si
            n_correct += hit
            by_arity[len(q.options)][0] += 1
            by_arity[len(q.options)][1] += hit
            by_task[ex.task][0] += 1
            by_task[ex.task][1] += hit
            d = q.description_for(si)
            if d:
                wordings[d] += 1
                split_counts[sentinel.split_of(d)] += 1
            else:
                split_counts["bare"] += 1

    worst_arity = 0.0
    for n, (tot, ok) in by_arity.items():
        if tot >= 200:
            worst_arity = max(worst_arity, abs(ok / tot - (n_correct / max(n_present, 1))))
    sep_tasks = sorted(
        ((abs(ok / tot - 0.5), name, tot, ok / tot)
         for name, (tot, ok) in by_task.items() if tot >= 100),
        reverse=True)[:5]

    def dist(c):
        tot = sum(c.values())
        return {k: v / tot for k, v in c.items()} if tot else {}

    ds, dp = dist(arity_all["sentinel"]), dist(arity_all["plain"])
    arity_tvd = 0.5 * sum(abs(ds.get(k, 0) - dp.get(k, 0)) for k in set(ds) | set(dp))
    base = n_abstain / n_choice if n_choice else 0.0
    lift = (abs(n_absent_abstain / n_absent - base) if n_absent else 0.0)
    return {
        "p_withheld": p_withheld, "p_decoy": p_decoy,
        "membership_lift": lift,
        "p_abstain": base,
        "p_abstain_given_modal_gold_absent": (n_absent_abstain / n_absent
                                              if n_absent else None),
        "arity_tvd": arity_tvd,
        "sentinel_last_share": n_last / n_present if n_present else 0.0,
        "n_choice_questions": n_choice,
        "n_sentinel": n_present,
        "sentinel_share": n_present / n_choice if n_choice else 0.0,
        "p_correct_given_present": n_correct / n_present if n_present else None,
        "distinct_wordings": len(wordings),
        "max_exact_repeat": wordings.most_common(1)[0][1] if wordings else 0,
        "wordings_over_50": sum(1 for v in wordings.values() if v > 50),
        "wording_split": dict(split_counts),
        "arity_max_deviation": worst_arity,
        "arity_sentinel": dict(sorted(arity_all["sentinel"].items())),
        "arity_plain": dict(sorted(arity_all["plain"].items())),
        "most_separated_tasks": [{"task": t, "n": n, "p_correct": round(p, 3)}
                                 for _, t, n, p in sep_tasks],
    }


def acceptance(result: dict) -> list[tuple[str, object, float, bool]]:
    """-> [(metric, value, bar, value <= bar)] for every `ACCEPTANCE` bar."""
    out = []
    for key, bar in ACCEPTANCE.items():
        v = result[key]
        out.append((key, v, bar, v is not None and v <= bar))
    return out


def sentinel_sweep(examples, grid=SWEEP) -> list[dict]:
    return [sentinel_audit(examples, w, d) for w, d in grid]


# ---- domain counts ---------------------------------------------------------------------

def domain_counts(meta: dict, records_by_split: dict[str, Iterable[dict]]) -> dict:
    """-> {"per": {domain: Counter(split -> questions, "all" -> total)},
           "tasks": {domain: set of tasks}, "total": questions, "unmapped": {row: n}}."""
    row_of = {t["name"]: t["row"] for t in meta["tasks"]}
    row2dom = {r: d for d, (rows, _) in DOMAINS.items() for r in rows}
    per: dict[int, Counter] = defaultdict(Counter)
    tasks: dict[int, set[str]] = defaultdict(set)
    unmapped: Counter = Counter()
    total = 0
    for split, records in records_by_split.items():
        for rec in records:
            n = len(rec["questions"])
            total += n
            dom = row2dom.get(row_of.get(rec["task"], -1))
            if dom is None:
                unmapped[row_of.get(rec["task"], -1)] += n
                continue
            per[dom][split] += n
            per[dom]["all"] += n
            tasks[dom].add(rec["task"])
    return {"per": per, "tasks": tasks, "total": total, "unmapped": dict(unmapped)}


def domain_table(counts: dict) -> list[str]:
    """The domain table as Markdown lines."""
    per, tasks, total = counts["per"], counts["tasks"], counts["total"] or 1
    lines = ["| # | domain | questions | share | train | devreal | testreal | tasks |",
             "|---|---|---|---|---|---|---|---|"]
    for d in sorted(DOMAINS):
        c = per[d]
        n = c["all"]
        lines.append(f"| {d} | {DOMAINS[d][1]} | {n:,} | {n / total:.1%} | "
                     f"{c['train'] + c['val']:,} | {c['devreal']:,} | {c['testreal']:,} | "
                     f"{len(tasks[d]):,} |")
    lines.append(f"\nTotal: {counts['total']:,} questions across "
                 f"{sum(len(v) for v in tasks.values()):,} tasks.")
    return lines


# ---- integrity checks ------------------------------------------------------------------

SLOT = re.compile(r"\{[a-zA-Z_][a-zA-Z0-9_]*\}")
CHECKS = ("malformed:empty_state", "malformed:duplicate_options", "malformed:empty_question",
          "malformed:target_index", "malformed:soft_target", "unfilled_slot",
          "cross_split", "di_overlap", "enrich_rewrote", "wording_predicts")


def _state_str(r) -> str:
    s = r["state"]
    return s if isinstance(s, str) else json.dumps(s, sort_keys=True, ensure_ascii=False)


def _wording_gain(pairs) -> float:
    """Out-of-sample accuracy gain of a wording -> gold lookup over the majority class."""
    a, b = pairs[: len(pairs) // 2], pairs[len(pairs) // 2:]
    maj = Counter(g for _, g in a).most_common(1)[0][0]
    lut: dict = defaultdict(Counter)
    for w, g in a:
        lut[w][g] += 1

    def pred(w):
        return lut[w].most_common(1)[0][0] if w in lut else maj

    return (sum(pred(w) == g for w, g in b) - sum(maj == g for _, g in b)) / len(b)


def verify(corpus, blocklist=None, raw=None) -> tuple[Counter, list]:
    """Integrity counts over a corpus; every count should be 0.

      malformed         target not a valid index / soft target not a distribution /
                        duplicate options / empty state / empty question
      unfilled_slot     a literal {placeholder} left in a question or its instructions
      cross_split       a devreal/testreal state that is also a train/val state, verbatim
      di_overlap        a state the Decision Index blocklist still catches
      enrich_rewrote    (with `raw`, the corpus before enrichment) a protected task
                        whose question differs from the raw build
      wording_predicts  tasks where the question WORDING predicts the target out of
                        sample by > 0.05 over the task's majority, beyond what shuffled
                        wordings reach: phrasing choice must be label-blind

    `corpus` and `raw` are `CorpusStore`s; both are streamed, split by split.
    -> (counts, the flagged wording_predicts tasks).
    """
    from lod.corpus.services.enrich.schemas import PROTECTED_PREFIXES

    bad: Counter = Counter()
    train_states: set[str] = set()
    by_task_q: dict = defaultdict(list)          # (task, qid) -> [(question, gold)]
    for split in corpus.present(SPLITS):
        for r in corpus.iter_records(split):
            s = _state_str(r)
            if not s.strip():
                bad["malformed:empty_state"] += 1
            if split in ("train", "val"):
                train_states.add(s)
            elif s in train_states:
                bad["cross_split"] += 1
            if blocklist is not None and blocklist.blocked(s):
                bad["di_overlap"] += 1
            for q in r["questions"]:
                opts = [str(o) for o in q["options"]]
                if len(set(o.lower().strip() for o in opts)) != len(opts):
                    bad["malformed:duplicate_options"] += 1
                if not str(q.get("question", "")).strip():
                    bad["malformed:empty_question"] += 1
                t = q.get("target")
                if isinstance(t, int):
                    if not 0 <= t < len(opts):
                        bad["malformed:target_index"] += 1
                elif isinstance(t, list):
                    if len(t) != len(opts) or abs(sum(t) - 1) > 1e-3 or min(t) < -1e-9:
                        bad["malformed:soft_target"] += 1
                for fld in ("question", "instructions"):
                    if SLOT.search(str(q.get(fld) or "")):
                        bad["unfilled_slot"] += 1
                if split == "train" and t is not None:
                    gi = t if isinstance(t, int) else max(range(len(t)), key=t.__getitem__)
                    # the gold STRING: where options are shuffled per example, the index
                    # tracks option order, which the reader sees anyway
                    by_task_q[(r["task"], q.get("id"))].append((q["question"], opts[gi]))
    # wording -> target, fitted on one half and scored on the other
    rng = random.Random(0)
    flagged = []
    for (task, qid), items in by_task_q.items():
        if len({w for w, _ in items}) < 2 or len(items) < 200:
            continue
        rng.shuffle(items)
        real = _wording_gain(items)
        # the null: the same wordings, shuffled across items. A lookup over many rare
        # keys "beats" a majority fitted on the other half whenever classes are
        # near-tied, so the bar is what shuffled wordings reach, not zero.
        words = [w for w, _ in items]
        null = []
        for _ in range(40):
            rng.shuffle(words)
            null.append(_wording_gain(list(zip(words, [g for _, g in items]))))
        null.sort()
        if real > 0.05 and real > null[int(0.975 * len(null))]:
            flagged.append((task, qid, round(real, 3), round(null[-1], 3), len(items)))
    bad["wording_predicts"] = len(flagged)
    if raw is not None:
        # enrichment rewrites records in file order, one out per one in
        for split in corpus.present(SPLITS):
            if not raw.has(split):
                continue
            for r, rr in zip(corpus.iter_records(split), raw.iter_records(split)):
                if r["task"] != rr["task"]:
                    bad["enrich_rewrote"] += 1          # order broken: count it
                    continue
                if r["task"].startswith(PROTECTED_PREFIXES):
                    for q, qq in zip(r["questions"], rr["questions"]):
                        if q["question"] != qq["question"]:
                            bad["enrich_rewrote"] += 1
    for k in CHECKS:
        bad[k] += 0
    return bad, flagged


__all__ = ["ACCEPTANCE", "CHECKS", "acceptance", "domain_counts", "domain_table",
           "sentinel_audit", "sentinel_sweep", "verify"]
