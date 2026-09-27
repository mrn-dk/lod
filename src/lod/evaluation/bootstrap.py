"""Bootstrap intervals over eval dumps: one dump's CIs, and paired comparisons of two.

`dump_ci` gives 95 % percentile intervals for one dump's metrics. Micro figures resample
questions; macro figures resample tasks and, within each drawn task, its questions -- a
macro mean over hundreds of tasks whose CI ignored task sampling would be far too narrow.

`compare` gives the interval on a *paired* difference A - B between two dumps. It joins
by question (`key`, else `ref`, else position when the dumps are row-aligned), because
the packer drops questions that do not fit the budget and row i of one dump is then not
row i of another; and it resamples whole tasks by default, because questions within a
task are correlated. A negative difference means A has the lower loss.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

CI_METRICS = ("acc", "nll", "brier", "ece")
LOSSES = ("nll", "brier", "acc")


# ---- one dump -----------------------------------------------------------------------

def as_arrays(records: list[dict]) -> dict[str, np.ndarray]:
    """Per-question columns, so every bootstrap draw is a mean over a resampled index."""
    k = max(len(r["probs"]) for r in records)
    n = len(records)
    probs = np.zeros((n, k), dtype=np.float64)
    target = np.zeros((n, k), dtype=np.float64)
    for i, r in enumerate(records):
        probs[i, : len(r["probs"])] = r["probs"]
        target[i, : len(r["target"])] = r["target"]
    # the dump rounds to 6 dp; clamp as `calibration.metrics` does
    logp = np.log(np.clip(probs, 1e-12, None))
    return {
        "correct": (probs.argmax(axis=1) == target.argmax(axis=1)).astype(np.float64),
        "nll": -(target * logp).sum(axis=1),
        "brier": ((probs - target) ** 2).sum(axis=1),
        "conf": probs.max(axis=1),
        "task": np.array([r["task"] for r in records]),
    }


def _ece(conf: np.ndarray, correct: np.ndarray, n_bins: int) -> float:
    """Equal-width ECE over the top probability; bin edges as in `calibration.metrics`."""
    n = conf.shape[0]
    if n == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        c = int(m.sum())
        if c:
            total += (c / n) * abs(correct[m].mean() - conf[m].mean())
    return total


def _micro(a: dict, idx: np.ndarray, n_bins: int) -> dict[str, float]:
    return {"acc": float(a["correct"][idx].mean()), "nll": float(a["nll"][idx].mean()),
            "brier": float(a["brier"][idx].mean()),
            "ece": _ece(a["conf"][idx], a["correct"][idx], n_bins)}


def _macro(a: dict, groups: list[np.ndarray], n_bins: int) -> dict[str, float]:
    """Mean over tasks of each task's own figure, so one big task cannot swamp it."""
    rows = [_micro(a, g, n_bins) for g in groups if g.size]
    if not rows:
        return {m: float("nan") for m in CI_METRICS}
    return {m: float(np.mean([r[m] for r in rows])) for m in CI_METRICS}


def _ci(draws: list[float]) -> tuple[float, float]:
    finite = [d for d in draws if math.isfinite(d)]
    if not finite:
        return float("nan"), float("nan")
    return float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5))


def dump_ci(records: list[dict], n_boot: int = 2000, n_bins: int = 15,
            seed: int = 0) -> dict:
    """Point estimates plus 95 % CIs, micro (over questions) and macro (over tasks)."""
    a = as_arrays(records)
    rng = np.random.default_rng(seed)
    by: dict[str, list[int]] = defaultdict(list)
    for i, t in enumerate(a["task"]):
        by[t].append(i)
    groups = [np.array(by[t]) for t in sorted(by)]
    n = a["correct"].shape[0]
    all_idx = np.arange(n)

    micro_draws = {k: [] for k in CI_METRICS}
    for _ in range(n_boot):
        m = _micro(a, rng.integers(0, n, n), n_bins)
        for k in CI_METRICS:
            micro_draws[k].append(m[k])
    macro_draws = {k: [] for k in CI_METRICS}
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        m = _macro(a, [groups[j][rng.integers(0, groups[j].size, groups[j].size)]
                       for j in pick], n_bins)
        for k in CI_METRICS:
            macro_draws[k].append(m[k])

    p_micro, p_macro = _micro(a, all_idx, n_bins), _macro(a, groups, n_bins)
    return {"n_questions": n, "n_tasks": len(groups), "n_boot": n_boot,
            "micro": {k: {"value": p_micro[k], "ci95": _ci(micro_draws[k])}
                      for k in CI_METRICS},
            "macro": {k: {"value": p_macro[k], "ci95": _ci(macro_draws[k])}
                      for k in CI_METRICS}}


# ---- two dumps, paired --------------------------------------------------------------

def losses(records: list[dict], metric: str) -> np.ndarray:
    """Per-question loss; lower is better for every metric (acc -> 0/1 error)."""
    out = np.empty(len(records), dtype=np.float64)
    for i, r in enumerate(records):
        p = np.asarray(r["probs"], dtype=np.float64)
        t = np.asarray(r["target"], dtype=np.float64)
        if metric == "nll":
            out[i] = -(t * np.log(np.clip(p, 1e-12, None))).sum()
        elif metric == "brier":
            out[i] = ((p - t) ** 2).sum()
        elif metric == "acc":
            out[i] = 0.0 if r["correct"] else 1.0
        else:
            raise ValueError(metric)
    return out


def _join_kind(dumps: list[list[dict]], how: str) -> str:
    if how != "auto":
        return how
    if all("key" in r for d in dumps for r in d):
        return "key"
    if all("ref" in r for d in dumps for r in d):
        return "ref"
    return "position"


def common_rows(dumps: list[list[dict]], how: str = "auto") -> tuple[list[np.ndarray], str]:
    """-> (for each dump, the row indices of the questions every dump has, in one shared
    order; the join used). A key occurring twice in a file is paired by occurrence."""
    how = _join_kind(dumps, how)
    if how == "position":
        first = dumps[0]
        for d in dumps[1:]:
            if len(d) != len(first) or any(x["task"] != y["task"] or x.get("qid") != y.get("qid")
                                           for x, y in zip(first, d)):
                raise ValueError("the dumps carry no `key`/`ref` and are not row-aligned")
        idx = np.arange(len(first))
        return [idx for _ in dumps], how

    def keyed(recs):
        seen: Counter = Counter()
        out = {}
        for i, r in enumerate(recs):
            k = r["key"] if how == "key" else tuple(r["ref"])
            out[(k, seen[k])] = i
            seen[k] += 1
        return out

    maps = [keyed(d) for d in dumps]
    common = [k for k in maps[0] if all(k in m for m in maps[1:])]
    rows = [np.array([m[k] for k in common], dtype=np.int64) for m in maps]
    if how == "ref":
        # a ref is a question identity within one data file only; catch the easy misuse
        for d, r in zip(dumps[1:], rows[1:]):
            bad = sum(dumps[0][i]["task"] != d[j]["task"] for i, j in zip(rows[0], r))
            if bad:
                raise ValueError(f"join by ref paired {bad} rows of different tasks: the "
                                 "dumps were not made from the same data file")
    return rows, how


def join(a: list[dict], b: list[dict], how: str = "auto") -> tuple[np.ndarray, np.ndarray, str]:
    """-> (row indices into a, row indices into b, the join used)."""
    (ia, ib), used = common_rows([a, b], how)
    return ia, ib, used


def paired_bootstrap(diff: np.ndarray, clusters: np.ndarray | None, n_boot: int = 10000,
                     seed: int = 0, chunk: int = 500) -> dict:
    """Mean of `diff` and its bootstrap interval.

    `clusters` (one label per question) resamples whole clusters; the statistic is then
    sum(diff) / sum(count) over the draw. None resamples questions.
    """
    n = diff.shape[0]
    if n == 0:
        return {"n": 0, "mean_diff": float("nan"), "ci95": [float("nan")] * 2,
                "p_a_better": float("nan"), "n_clusters": 0, "distinguishable": False}
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=np.float64)
    if clusters is not None:
        _, inv = np.unique(clusters, return_inverse=True)
        s = np.bincount(inv, weights=diff)
        c = np.bincount(inv).astype(np.float64)
        t = s.shape[0]
        for lo in range(0, n_boot, chunk):
            m = min(chunk, n_boot - lo)
            w = rng.multinomial(t, np.full(t, 1.0 / t), size=m).astype(np.float64)
            draws[lo:lo + m] = (w @ s) / (w @ c)
        n_clusters = t
    else:
        step = max(1, min(chunk, 20_000_000 // max(n, 1)))
        for lo in range(0, n_boot, step):
            m = min(step, n_boot - lo)
            draws[lo:lo + m] = diff[rng.integers(0, n, size=(m, n))].mean(axis=1)
        n_clusters = n
    lo, hi = np.percentile(draws, [2.5, 97.5])
    if clusters is not None and n_clusters < 2:
        # one task resampled is that task every time: no interval to report
        return {"n": int(n), "n_clusters": int(n_clusters), "mean_diff": float(diff.mean()),
                "ci95": [float("nan")] * 2, "p_a_better": float("nan"),
                "distinguishable": False}
    return {"n": int(n), "n_clusters": int(n_clusters), "mean_diff": float(diff.mean()),
            "ci95": [float(lo), float(hi)],
            # A - B < 0 means A's loss is lower; ties count half
            "p_a_better": float((draws < 0).mean() + 0.5 * (draws == 0).mean()),
            "distinguishable": bool(lo > 0 or hi < 0)}


def domain_of_tasks(corpus: Path) -> dict[str, tuple[int, str]]:
    """Task name -> (domain number, domain name), from the corpus's `meta.json`."""
    from lod.corpus.domains import DOMAINS

    meta = json.loads((Path(corpus) / "meta.json").read_text())
    row2dom = {r: d for d, (rows, _) in DOMAINS.items() for r in rows}
    out = {}
    for t in meta.get("tasks") or []:
        d = row2dom.get(t.get("row"))
        if d is not None:
            out[t["name"]] = (d, DOMAINS[d][1])
    return out


def compare(a: list[dict], b: list[dict], metric: str = "nll", cluster: str = "task",
            n_boot: int = 10000, seed: int = 0, join_how: str = "auto",
            domains: dict[str, tuple[int, str]] | None = None,
            exclude_meta: bool = False) -> dict:
    """Paired A - B on `metric`, joined by question, optionally broken down by domain."""
    if exclude_meta:
        a = [r for r in a if not str(r.get("qid", "")).startswith("meta_")]
        b = [r for r in b if not str(r.get("qid", "")).startswith("meta_")]
    ia, ib, used = join(a, b, join_how)
    la, lb = losses([a[i] for i in ia], metric), losses([b[j] for j in ib], metric)
    diff = la - lb
    tasks = np.array([a[i]["task"] for i in ia])
    res = paired_bootstrap(diff, tasks if cluster == "task" else None, n_boot, seed)
    res.update({"metric": metric, "cluster": cluster, "join": used, "n_boot": n_boot,
                "n_a": len(a), "n_b": len(b), "n_joined": int(ia.shape[0]),
                "dropped_a": len(a) - int(ia.shape[0]),
                "dropped_b": len(b) - int(ib.shape[0]),
                "mean_a": float(la.mean()) if la.size else float("nan"),
                "mean_b": float(lb.mean()) if lb.size else float("nan")})
    if domains is not None:
        by: dict[int, list[int]] = defaultdict(list)
        for k, t in enumerate(tasks):
            by[domains.get(t, (0, "unmapped"))[0]].append(k)
        names = {d: nm for d, nm in domains.values()}
        names[0] = "(task not in corpus meta.json)"
        res["per_domain"] = {}
        for d in sorted(by):
            sel = np.array(by[d])
            r = paired_bootstrap(diff[sel], tasks[sel] if cluster == "task" else None,
                                 max(1000, n_boot // 5), seed)
            r.update({"name": names[d], "mean_a": float(la[sel].mean()),
                      "mean_b": float(lb[sel].mean())})
            res["per_domain"][str(d)] = r
    return res
