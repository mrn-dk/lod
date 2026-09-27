"""Eval dumps broken down: by corpus domain, and by state length x evidence depth.

Both read an `eval-<split>.jsonl` dump only. `by_domain` needs the corpus the dump was
scored on (its `meta.json` maps task -> source row, and `lod.corpus.domains.DOMAINS` maps
row -> domain); `depth_grid` needs questions carrying `meta.gen` (the long-context depth
check), joined from the scored data file when the dump has no `meta`.
"""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path

from lod.evaluation.dumps import accuracy, ece, nll


def _boot_ci(records: list[dict], fn, iters: int = 400, seed: int = 0) -> list[float] | None:
    """Percentile CI over questions; None below 20 questions."""
    if len(records) < 20:
        return None
    rng = random.Random(seed)
    n = len(records)
    vals = sorted(v for v in (fn([records[rng.randrange(n)] for _ in range(n)])
                              for _ in range(iters)) if v is not None)
    if not vals:
        return None
    return [vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals))]]


def by_domain(dump: list[dict], corpus: Path, ci: bool = True) -> dict:
    """Accuracy / NLL / ECE per corpus domain, plus real vs code-labelled."""
    from lod.corpus.domains import DOMAINS

    meta = json.loads((Path(corpus) / "meta.json").read_text())
    row_of = {t["name"]: t["row"] for t in meta["tasks"]}
    real_of = {t["name"]: bool(t.get("real", True)) for t in meta["tasks"]}
    row2dom = {r: d for d, (rows, _) in DOMAINS.items() for r in rows}

    groups: dict[int, list] = defaultdict(list)
    absent = unclaimed = 0
    for r in dump:
        if r["task"] not in row_of:
            absent += 1                    # scored on a different corpus than `corpus`
            continue
        d = row2dom.get(row_of[r["task"]])
        if d is None:
            unclaimed += 1                 # a row no domain claims
            continue
        groups[d].append(r)

    out = {"n": len(dump), "absent_from_corpus": absent, "row_unclaimed": unclaimed,
           "domains": {}}
    for d in sorted(groups):
        recs = groups[d]
        out["domains"][d] = {"name": DOMAINS[d][1], "n": len(recs),
                             "tasks": len({r["task"] for r in recs}),
                             "acc": accuracy(recs), "nll": nll(recs), "ece": ece(recs),
                             "acc_ci": _boot_ci(recs, accuracy) if ci else None}
    for name, keep in (("real", True), ("code_labelled", False)):
        recs = [r for r in dump if real_of.get(r["task"], True) is keep]
        if recs:
            out[name] = {"n": len(recs), "tasks": len({r["task"] for r in recs}),
                         "acc": accuracy(recs), "nll": nll(recs), "ece": ece(recs)}
    return out


def format_by_domain(res: dict) -> list[str]:
    lines = []
    if res["absent_from_corpus"]:
        lines.append(f"!! {res['absent_from_corpus']:,} questions name a task that is not in "
                     "the corpus -- is that the corpus this dump was scored on?")
    if res["row_unclaimed"]:
        lines.append(f"!! {res['row_unclaimed']:,} questions sit in a source row no domain "
                     "claims; they are not in the table")
    head = (f"{'#':>3} {'domain':34} {'n':>7} {'tasks':>6} {'acc':>7} {'nll':>7} {'ece':>7}"
            f"  {'acc 95% CI':>18}")
    lines += [head, "-" * len(head)]
    for d, m in res["domains"].items():
        ci = m.get("acc_ci")
        lines.append(f"{d:>3} {m['name'][:34]:34} {m['n']:7,} {m['tasks']:6,} "
                     f"{m['acc']:7.4f} {m['nll']:7.4f} {m['ece']:7.4f}"
                     + (f"  [{ci[0]:7.4f},{ci[1]:7.4f}]" if ci else ""))
    lines.append("")
    for name in ("real", "code_labelled"):
        if name in res:
            m = res[name]
            lines.append(f"{'':>3} {name.replace('_', '-'):34} {m['n']:7,} {m['tasks']:6,} "
                         f"{m['acc']:7.4f} {m['nll']:7.4f} {m['ece']:7.4f}")
    return lines


# ---- state length x evidence depth --------------------------------------------------

LEN_EDGES = (1500, 3000, 6000, 12000, 24000)          # measured-length buckets
LEN_NAMES = ("1k", "2k", "4k", "8k", "16k", "30k")
LEN_NOMINAL = (1000, 2000, 4000, 8000, 16000, 29500)
DEPTH_NAMES = ("10%", "50%", "90%")


def _len_cell(g: dict) -> str:
    if "cell" in g:
        length = g["cell"]["length"]
        return LEN_NAMES[min(range(len(LEN_NOMINAL)),
                             key=lambda i: abs(LEN_NOMINAL[i] - length))]
    return LEN_NAMES[sum(g["state_tokens"] >= e for e in LEN_EDGES)]


def _depth_cell(g: dict) -> str:
    d = g["cell"]["depth"] if "cell" in g else g["depth_frac"]
    return DEPTH_NAMES[0 if d < 1 / 3 else 1 if d < 2 / 3 else 2]


def depth_rows(dump: list[dict], data: Path | None = None) -> list[dict]:
    """The dump's questions that carry `meta.gen`, joined from `data` by `ref` if needed."""
    if data is not None:
        exs = [json.loads(line) for line in Path(data).open() if line.strip()]
        for r in dump:
            if "meta" not in r and "ref" in r:
                ei, qi = r["ref"]
                r["meta"] = exs[ei]["questions"][qi].get("meta")
    return [r for r in dump if (r.get("meta") or {}).get("gen")]


def depth_grid(rows: list[dict], title: str) -> list[str]:
    """Accuracy (n) per (state length, evidence depth) cell, with chance and NLL per row.

    "The model uses the whole window" reads as a flat table: accuracy at 16k/30k no lower
    than at 2k, and the 10 % / 50 % / 90 % columns equal.
    """
    cell = defaultdict(list)
    for r in rows:
        g = r["meta"]["gen"]
        cell[(_len_cell(g), _depth_cell(g))].append(r)

    def acc_n(rs):
        return f"{accuracy(rs):.3f} ({len(rs):3})" if rs else "-"

    lines = [f"{title}: {len(rows)} questions   accuracy (n) | chance",
             f"{'length':>7} " + " ".join(f"{d:>14}" for d in DEPTH_NAMES)
             + f" {'all':>14} {'chance':>7} {'nll':>6}"]
    for ln in LEN_NAMES:
        rs_all = [r for d in DEPTH_NAMES for r in cell.get((ln, d), [])]
        if not rs_all:
            continue
        chance = sum(1 / len(r["probs"]) for r in rs_all) / len(rs_all)
        row_nll = sum(-math.log(max(1e-12, sum(p * t for p, t in zip(r["probs"], r["target"]))))
                      for r in rs_all) / len(rs_all)
        lines.append(f"{ln:>7} " + " ".join(f"{acc_n(cell.get((ln, d), [])):>14}"
                                             for d in DEPTH_NAMES)
                     + f" {acc_n(rs_all):>14} {chance:7.3f} {row_nll:6.3f}")
    lines.append(f"{'all':>7} " + " ".join(
        f"{acc_n([r for (_, dd), v in cell.items() if dd == d for r in v]):>14}"
        for d in DEPTH_NAMES))
    return lines
