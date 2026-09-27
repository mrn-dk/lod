"""The selection sets carved from a built corpus's dev split.

`devreal_sel` (`devsel`) is a task-stratified subsample of the whole of devreal: a fixed
number of examples per *task*, not a fraction, so a task with 2,000 examples and one with
20 weigh the same -- a macro-balanced, cheap in-distribution read. It includes the twins
of trained structures, so it is reported, never selected on.

`devood` (`devood`) is the family-disjoint dev split: only the dev tasks whose *family*
(`sources.real.base.family_of`: the declared held-out unit, else the name without its
twin suffix) has no task in train, and never a family that testreal holds out either --
dev and test hold out disjoint structures, so selecting on one says nothing in advance
about the other. Checkpoints are selected and the temperature is fitted on it.
`devood_sel` is its selection subsample, with per-domain quotas matched to testreal's
domain shares.

Both work on `(task, n_questions, raw line)` rows, so records are written back verbatim.
"""

from __future__ import annotations

import collections
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from types import SimpleNamespace

from lod.corpus.domains import DOMAINS
from lod.corpus.services.sources.real.base import family_of

ROW2DOM = {r: d for d, (rows, _) in DOMAINS.items() for r in rows}

# The release corpus's devreal_sel boosts. At ten per task, the code-labelled families
# with many tasks' worth of structure -- rule application, entity facts, the agentic
# generators -- would be a sliver of the number that picks a checkpoint; these restore
# each to roughly its share of devreal.
DEVSEL_BOOSTS: tuple[tuple[str, int], ...] = (
    ("rule_", 81), ("entity_fact", 240),
    ("fever_", 80), ("vitaminc_", 80),
    ("entityres_", 240), ("toolgate_", 132), ("sensor_", 240),
    ("diff_", 240), ("sched_", 67), ("pdfdocs_", 20),
    ("relational_", 100), ("tabfact_", 40),
    ("sent44_", 60), ("topic_", 60), ("ml_", 15),
)

Row = tuple[str, int, str]          # (task, questions, raw line)


class SplitError(ValueError):
    """The corpus cannot give a family-disjoint dev split as asked."""


def parse_boosts(specs) -> list[tuple[str, int]]:
    """["rule_=81", ...] or [("rule_", 81), ...] -> [("rule_", 81), ...]."""
    out = []
    for spec in specs:
        if isinstance(spec, str):
            prefix, _, n = spec.partition("=")
            out.append((prefix, int(n)))
        else:
            out.append((spec[0], int(spec[1])))
    return out


def rows_of(lines) -> list[Row]:
    return [(rec["task"], len(rec["questions"]), line)
            for line, rec in ((ln, json.loads(ln)) for ln in lines)]


# ---- devreal_sel -----------------------------------------------------------------------

def devsel(examples: list, per_task: int = 10, boosts=DEVSEL_BOOSTS,
           seed: int = 0) -> tuple[list, dict[str, int]]:
    """`per_task` examples of every task (`N` for tasks matching a boost `(prefix, N)`),
    shuffled. Works on anything with a `.task`. -> (subsample, examples taken per task)."""
    boosts = parse_boosts(boosts)
    by: dict[str, list] = collections.defaultdict(list)
    for e in examples:
        by[e.task].append(e)
    rng = random.Random(seed)
    out = []
    taken: dict[str, int] = {}
    for task in sorted(by):
        group = by[task]
        k = next((n for prefix, n in boosts if task.startswith(prefix)), per_task)
        chosen = group if len(group) <= k else rng.sample(group, k)
        taken[task] = len(chosen)
        out += chosen
    rng.shuffle(out)
    return out, taken


# ---- devood ----------------------------------------------------------------------------

def legacy_family(name: str) -> str:
    """The family of a task in a corpus whose meta.json records none."""
    if "__" in name:                          # Hub task: <dataset>__<column>
        return name.split("__", 1)[0]
    return family_of(SimpleNamespace(family=None, name=name))


def task_families(meta: dict) -> tuple[dict[str, str], dict[str, str], dict[str, int], bool]:
    """-> (task -> family, task -> split, task -> row, whether any family was recorded)."""
    fam, split, row = {}, {}, {}
    recorded = False
    for t in meta["tasks"]:
        if t.get("family"):
            recorded = True
            fam[t["name"]] = t["family"]
        else:
            fam[t["name"]] = legacy_family(t["name"])
        split[t["name"]] = t["split"]
        row[t["name"]] = t["row"]
    return fam, split, row, recorded


def by_domain(rows: list[Row], row_of: dict[str, int]) -> Counter:
    c: Counter = Counter()
    for task, q, _ in rows:
        c[ROW2DOM.get(row_of.get(task, -1), 0)] += q
    return c


def select(rows: list[Row], row_of, test_share: dict[int, float], total_q: int,
           per_task: int, boosts: list[tuple[str, int]],
           seed: int) -> tuple[list[Row], dict[int, float]]:
    """Each domain gets its testreal share of `total_q` questions, spread evenly over its
    tasks: `want // n` examples each and one more for `want % n` of them drawn at random,
    never fewer than `per_task`. Rounding every task up instead would hand a domain of
    many small tasks a multiple of its share. -> (rows, mean quota per task by domain)."""
    by_task: dict[str, list] = defaultdict(list)
    for r in rows:
        by_task[r[0]].append(r)
    tasks_in: dict[int, list[str]] = defaultdict(list)
    for t in sorted(by_task):
        tasks_in[ROW2DOM.get(row_of.get(t, -1), 0)].append(t)
    # renormalise over the domains dev can supply: a domain testreal has and devood lacks
    # is reported, not silently given to its neighbours' tasks one by one
    have = sum(test_share.get(d, 0.0) for d in tasks_in) or 1.0
    rng = random.Random(seed)
    quota: dict[str, int] = {}
    k_dom: dict[int, float] = {}
    for d in sorted(tasks_in):
        ts = tasks_in[d]
        q_per_ex = (sum(r[1] for t in ts for r in by_task[t]) /
                    max(1, sum(len(by_task[t]) for t in ts)))
        want = round(test_share.get(d, 0.0) / have * total_q / max(1e-9, q_per_ex))
        base, extra = divmod(want, len(ts))
        plus = set(rng.sample(ts, extra))
        for t in ts:
            quota[t] = max(per_task, base + (t in plus))
        k_dom[d] = round(sum(quota[t] for t in ts) / len(ts), 1)
    out = []
    for t in sorted(by_task):
        group = by_task[t]
        k = next((n for prefix, n in boosts if t.startswith(prefix)), quota[t])
        out += group if len(group) <= k else rng.sample(group, k)
    rng.shuffle(out)
    return out, k_dom


@dataclass
class DevOOD:
    devood: list[Row]
    sel: list[Row]
    summary: dict
    report: list[str] = field(default_factory=list)      # the printable table
    warnings: list[str] = field(default_factory=list)


def devood(meta: dict, dev: list[Row], test: list[Row], *, sel_questions: int = 9_000,
           per_task: int = 1, boosts=(), drop_test_overlap: bool = False,
           seed: int = 0, corpus: str = "") -> DevOOD:
    """The family-disjoint dev split and its selection subsample.

    Raises `SplitError` when a dev family that never trains is also a testreal family --
    a generator routing one structure to both -- unless `drop_test_overlap`, which drops
    such families with a warning instead. `per_task` floors the subsample's examples per
    task; 1 rather than devsel's 10, because a domain of hundreds of small dev tasks
    would otherwise swamp it.
    """
    boosts = parse_boosts(boosts)
    fam, split, row_of, recorded = task_families(meta)
    fams: dict[str, set[str]] = defaultdict(set)
    for t, s in split.items():
        fams[s].add(fam[t])
    report = [f"{corpus}: {len(fam):,} tasks, families "
              f"{'recorded in meta.json' if recorded else 'recomputed from names (legacy)'}"]
    warnings: list[str] = []

    unknown = {t for t, _, _ in dev + test if t not in fam}
    if unknown:
        raise SplitError(f"tasks missing from meta.json: {sorted(unknown)[:10]}")

    dev_fams = {fam[t] for t, _, _ in dev} - fams["train"]
    overlap = dev_fams & fams["testreal"]
    if overlap:
        msg = (f"{len(overlap)} dev families with no train task are ALSO testreal families: "
               f"{sorted(overlap)}")
        if not drop_test_overlap:
            raise SplitError(msg + "\n  a generator routes one structure to dev and test; "
                             "declare its family (or pass drop_test_overlap)")
        warnings.append(f"WARNING (dropped): {msg}")
        dev_fams -= overlap
    ood = [r for r in dev if fam[r[0]] in dev_fams]
    ood_fams = {fam[t] for t, _, _ in ood}
    # the property everything downstream relies on
    assert not ood_fams & fams["train"], sorted(ood_fams & fams["train"])
    assert not ood_fams & fams["testreal"], sorted(ood_fams & fams["testreal"])

    test_ood = [r for r in test if fam[r[0]] not in fams["train"]]
    tq, dq = by_domain(test, row_of), by_domain(ood, row_of)
    toq, devq = by_domain(test_ood, row_of), by_domain(dev, row_of)
    t_total = sum(tq.values()) or 1
    test_share = {d: n / t_total for d, n in tq.items()}
    sel, k_dom = select(ood, row_of, test_share, sel_questions, per_task, boosts, seed)
    sq = by_domain(sel, row_of)

    def tot(rows):
        return sum(r[1] for r in rows)

    report += warnings
    report += [
        f"devreal  {len(dev):,} examples, {tot(dev):,} questions, "
        f"{len({t for t, _, _ in dev}):,} tasks",
        f"devood   {len(ood):,} examples, {tot(ood):,} questions, "
        f"{len({t for t, _, _ in ood}):,} tasks, {len(ood_fams):,} families",
        f"devood_sel {len(sel):,} examples, {tot(sel):,} questions",
        f"testreal {len(test):,} examples, {tot(test):,} questions; family-disjoint "
        f"{len(test_ood):,} examples, {tot(test_ood):,} questions",
        "",
    ]
    # `target` is the domain's testreal share renormalised over the domains devood has,
    # which is what `select` aims `sel` at; `share` beside `test` is over all of testreal
    present = sum(test_share.get(d, 0.0) for d in dq if dq[d]) or 1.0
    report.append(
        f"{'#':>3} {'domain':34} {'devreal':>8} {'devood':>8} {'share':>6} "
        f"{'sel':>6} {'share':>6} {'target':>6} {'k/task':>6} {'test':>8} {'share':>6} "
        f"{'testood':>8} {'share':>6}")
    ds, sel_total, to_total = tot(ood) or 1, tot(sel) or 1, tot(test_ood) or 1
    for d in sorted(set(DOMAINS) | set(dq) | set(tq)):
        title = DOMAINS[d][1][:34] if d in DOMAINS else "(unmapped)"
        report.append(
            f"{d:>3} {title:34} {devq[d]:8,} {dq[d]:8,} {dq[d]/ds:6.1%} "
            f"{sq[d]:6,} {sq[d]/sel_total:6.1%} "
            f"{(test_share.get(d, 0.0) / present if dq[d] else 0.0):6.1%} "
            f"{k_dom.get(d, 0):6} "
            f"{tq[d]:8,} {tq[d]/t_total:6.1%} {toq[d]:8,} {toq[d]/to_total:6.1%}")
    missing = sorted(d for d in DOMAINS if tq[d] and not dq[d])
    if missing:
        report.append(f"\ndomains in testreal with no devood question: {missing}")

    summary = {
        "corpus": corpus, "families_recorded": recorded,
        "dropped_test_overlap": sorted(overlap) if drop_test_overlap else [],
        "devood": {"examples": len(ood), "questions": tot(ood),
                   "families": sorted(ood_fams),
                   "tasks": sorted({t for t, _, _ in ood})},
        "devood_sel": {"examples": len(sel), "questions": tot(sel), "k_per_domain": k_dom},
        "questions_by_domain": {"devreal": dict(devq), "devood": dict(dq),
                                "devood_sel": dict(sq), "testreal": dict(tq),
                                "testreal_ood": dict(toq)},
    }
    return DevOOD(devood=ood, sel=sel, summary=summary, report=report, warnings=warnings)
