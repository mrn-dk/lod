#!/usr/bin/env python
"""Read eval dumps (`eval-<split>.jsonl`): intervals, comparisons and breakdowns.

    report.py ci        DUMP [--json OUT]           95 % bootstrap CIs, micro and macro
    report.py compare   A B [--metric nll] [--corpus D]
                                                    paired A - B, joined by question and
                                                    resampled by task
    report.py by-domain DUMP --corpus D [--json OUT] per corpus domain, real vs code-labelled
    report.py depth     DUMP [--data FILE]          state length x evidence depth grid

No model and no GPU: everything is computed from the dumps `scripts/evaluate.py` writes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from lod.evaluation import bootstrap, breakdowns
from lod.evaluation.dumps import read_dump


def _write(res: dict, path: Path | None) -> None:
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(res, indent=2) + "\n")
        print(f"\nwrote {path}")


def ci(args) -> None:
    recs = read_dump(args.dump)
    if args.exclude_meta:
        recs = [r for r in recs if not str(r.get("qid", "")).startswith("meta_")]
    if not recs:
        raise SystemExit(f"no records in {args.dump}")
    res = bootstrap.dump_ci(recs, args.n_boot, args.n_bins, args.seed)
    print(f"{args.dump.name}   {res['n_questions']:,} questions over {res['n_tasks']} tasks   "
          f"{res['n_boot']} bootstrap draws\n")
    print("| level | metric | value | 95% CI |\n|---|---|---|---|")
    for level in ("micro", "macro"):
        for k in bootstrap.CI_METRICS:
            v = res[level][k]
            lo, hi = v["ci95"]
            print(f"| {level} | {k} | {v['value']:.4f} | [{lo:.4f}, {hi:.4f}] |")
    _write(res, args.json)


def compare(args) -> None:
    domains = bootstrap.domain_of_tasks(args.corpus) if args.corpus else None
    try:
        res = bootstrap.compare(read_dump(args.a), read_dump(args.b), args.metric,
                                args.cluster, args.n, args.seed, args.join, domains,
                                args.exclude_meta)
    except ValueError as e:
        raise SystemExit(str(e)) from None
    lo, hi = res["ci95"]
    unit = "tasks" if res["cluster"] == "task" else "questions"
    print(f"A = {args.a}\nB = {args.b}")
    print(f"join by {res['join']}: {res['n_joined']:,} questions paired; dropped "
          f"{res['dropped_a']:,} of {res['n_a']:,} from A, {res['dropped_b']:,} of "
          f"{res['n_b']:,} from B")
    print(f"{res['metric']} (lower is better): A {res['mean_a']:.4f}   B {res['mean_b']:.4f}")
    print(f"mean diff A - B {res['mean_diff']:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]   "
          f"P(A better) {res['p_a_better']:.3f}   ({res['n_boot']} draws over "
          f"{res['n_clusters']:,} {unit})")
    print("  -> " + ("distinguishable: the CI excludes 0" if res["distinguishable"]
                     else "not distinguishable: the CI contains 0"))
    for d, r in (res.get("per_domain") or {}).items():
        lo, hi = r["ci95"]
        print(f"{d:>3} {r['name'][:36]:36} {r['n']:7,} {r['mean_a']:8.4f} {r['mean_b']:8.4f} "
              f"{r['mean_diff']:+8.4f} [{lo:+.4f},{hi:+.4f}]"
              + ("" if r["distinguishable"] else "  ~"))
    _write({**res, "a": str(args.a), "b": str(args.b)}, args.json)


def by_domain(args) -> None:
    recs = read_dump(args.dump)
    if not recs:
        raise SystemExit(f"no records in {args.dump}")
    res = breakdowns.by_domain(recs, args.corpus, ci=not args.no_ci)
    print(f"{args.dump}   {len(recs):,} questions\n")
    print("\n".join(breakdowns.format_by_domain(res)))
    _write(res, args.json)


def depth(args) -> None:
    rows = breakdowns.depth_rows(read_dump(args.dump), args.data)
    if not rows:
        sys.exit("no question in the dump carries meta.gen (pass --data to join it)")
    print("\n".join(breakdowns.depth_grid(rows, "all families")))
    if args.by_family:
        for fam in sorted({r["meta"]["gen"]["family"] for r in rows}):
            print()
            print("\n".join(breakdowns.depth_grid(
                [r for r in rows if r["meta"]["gen"]["family"] == fam], fam)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ci", help="bootstrap CIs for one dump")
    p.add_argument("dump", type=Path)
    p.add_argument("--exclude-meta", action="store_true", help="drop meta_* questions")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--n-bins", type=int, default=15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", type=Path, default=None)
    p.set_defaults(fn=ci)

    p = sub.add_parser("compare", help="paired bootstrap of A - B")
    p.add_argument("a", type=Path)
    p.add_argument("b", type=Path)
    p.add_argument("--metric", choices=bootstrap.LOSSES, default="nll")
    p.add_argument("--cluster", choices=["task", "none"], default="task",
                   help="resample whole tasks (default) or single questions")
    p.add_argument("--join", choices=["auto", "key", "ref", "position"], default="auto")
    p.add_argument("--n", type=int, default=10000, help="bootstrap draws")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--corpus", type=Path, default=None,
                   help="corpus dir with meta.json: adds a per-domain breakdown")
    p.add_argument("--exclude-meta", action="store_true", help="drop meta_* questions")
    p.add_argument("--json", type=Path, default=None)
    p.set_defaults(fn=compare)

    p = sub.add_parser("by-domain", help="metrics per corpus domain")
    p.add_argument("dump", type=Path)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--no-ci", action="store_true", help="skip the bootstrap")
    p.add_argument("--json", type=Path, default=None)
    p.set_defaults(fn=by_domain)

    p = sub.add_parser("depth", help="accuracy by state length x evidence depth")
    p.add_argument("dump", type=Path)
    p.add_argument("--data", type=Path, default=None,
                   help="the scored file, to join `meta` into a dump that lacks it")
    p.add_argument("--by-family", action="store_true")
    p.set_defaults(fn=depth)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
