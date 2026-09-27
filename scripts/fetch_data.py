#!/usr/bin/env python
"""Download the raw sources the corpus is built from (stage 1; building is stage 2).

Idempotent and resumable: anything already under the raw root is skipped unless
--refresh. With no selection, every fetcher that is not opt-in runs.

    uv run python scripts/fetch_data.py --dry-run          # what would be fetched
    uv run python scripts/fetch_data.py --inventory        # what is already on disk
    uv run python scripts/fetch_data.py --coverage         # which fetcher serves each row
    uv run python scripts/fetch_data.py                    # everything missing
    uv run python scripts/fetch_data.py --rows 44,45       # just these source-table rows
    uv run python scripts/fetch_data.py --only loghub,spdx --refresh
    uv run python scripts/fetch_data.py --only toolgate_http   # opt-in: rewrite a package seed

Rows 43 and 46 exclude Decision Index items and need the suite's files: --chessbench (the
ChessBench test .bag) and --di-repos (the suite's upstream repos).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _ints(text: str) -> set[int]:
    return {int(x) for x in text.split(",") if x.strip()}


def _names(text: str) -> set[str]:
    return {x.strip() for x in text.split(",") if x.strip()}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-root", type=Path, default=None,
                    help="raw data directory (default: $LOD_RAW_ROOT or ~/.cache/lod-sources)")
    ap.add_argument("--rows", type=_ints, default=None, help="comma-separated table rows")
    ap.add_argument("--only", type=_names, default=None,
                    help="comma-separated fetcher names (see --coverage); opt-in ones too")
    ap.add_argument("--refresh", action="store_true", help="re-download even if present")
    ap.add_argument("--dry-run", action="store_true", help="show the plan, fetch nothing")
    ap.add_argument("--inventory", action="store_true", help="show what is on disk and exit")
    ap.add_argument("--coverage", action="store_true",
                    help="show which fetcher serves each source-table row and exit")
    ap.add_argument("--row1-limit", type=int, default=None,
                    help="cap how many row-1 Hub sweep datasets to fetch")
    ap.add_argument("--chessbench", type=Path, default=None,
                    help="ChessBench test .bag (row 43's Decision Index exclusion)")
    ap.add_argument("--di-repos", type=Path, default=None,
                    help="Decision Index upstream repos (row 46's Unicode re-match)")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    # Source modules bind the raw root when imported, so choose it before importing lod.
    if args.raw_root is not None:
        os.environ["LOD_RAW_ROOT"] = str(args.raw_root.expanduser())

    from lod.corpus.controllers import fetch as ctl

    if args.coverage:
        for row, what in ctl.coverage().items():
            print(f"{row:3}  {'; '.join(what) or 'UNCOVERED'}")
        return 0

    if args.inventory:
        inv = ctl.inventory()
        print(f"raw root {inv['raw_root']}\n")
        print(f"{'fetcher':22} {'rows':18} {'present':>9} {'MB':>10}")
        for f in inv["fetchers"]:
            rows = ",".join(map(str, f["rows"]))
            rows = rows if len(rows) <= 18 else rows[:15] + "..."
            tag = "  (opt-in)" if f["opt_in"] else ""
            print(f"{f['name']:22} {rows:18} {f['have']:>4}/{f['want']:<4} "
                  f"{f['mb']:10,.1f}{tag}")
        print(f"\n{inv['store_keys']} keys in {inv['store']}")
        if inv["unclaimed_keys"]:
            print(f"{len(inv['unclaimed_keys'])} stored keys no fetcher writes: "
                  + ", ".join(inv["unclaimed_keys"]))
        return 0

    try:
        results = ctl.fetch(rows=args.rows, only=args.only, refresh=args.refresh,
                            dry_run=args.dry_run, row1_limit=args.row1_limit,
                            di_repos=args.di_repos, chessbench=args.chessbench)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print()
    for r in results:
        rows = ",".join(map(str, r.rows))
        rows = rows if len(rows) <= 18 else rows[:15] + "..."
        took = f" {r.seconds:6.1f}s" if r.seconds else ""
        print(f"{r.status:8} {r.name:22} {rows:18} {r.have:>4}/{r.want:<4}{took}  {r.detail}")
    return 1 if any(r.status == "failed" for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
