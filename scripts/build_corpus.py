#!/usr/bin/env python
"""Build the Lod corpus from a fetched raw store. One subcommand per pipeline stage.

    all            generate -> enrich -> decontaminate -> select -> probes -> audit
    generate       raw rows -> train / val / devreal / testreal + meta.json
    enrich         LLM question wording and option criteria per schema (cached)
    blocklist      hash the Decision Index suite into the contamination blocklist
    decontaminate  drop records the blocklist catches (in place, or --dst)
    select         devreal_sel, devood, devood_sel
    probes         depth_check.jsonl and knowledge.jsonl
    audit          sentinel acceptance, domain table, integrity, DI overlap
    distill        a teacher's distribution into train as meta.teacher

Defaults are the release settings; locations come from LOD_ROOT, LOD_RAW_ROOT and
LOD_DI_SUITE (see `lod.paths`). The stages are documented in `lod.corpus.controllers.build`.

Examples:
    uv run python scripts/build_corpus.py all --out data/lod-corpus
    uv run python scripts/build_corpus.py generate --out data/scratch/r37 --rows 37 \\
        --sample-per-task 200
    uv run python scripts/build_corpus.py distill --teacher runs/lod-stor-release \\
        --src data/lod-corpus --dst data/lod-corpus-distilled
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from lod.corpus.controllers import build
from lod.paths import DI_BLOCKLIST, DI_SUITE, RAW_ROOT


def rows_arg(value: str) -> list[int]:
    try:
        return [int(part) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("rows must be comma-separated integers") from exc


def add_generate(p: argparse.ArgumentParser, with_out: bool = True) -> None:
    if with_out:
        p.add_argument("--out", type=Path, required=True, help="corpus directory to write")
    p.add_argument("--raw-root", type=Path, default=RAW_ROOT,
                   help=f"fetched raw rows (default: LOD_RAW_ROOT = {RAW_ROOT})")
    p.add_argument("--rows", type=rows_arg, default=None,
                   help="comma-separated source-table rows to build (default: all)")
    p.add_argument("--target-questions", type=int, default=400_000,
                   help="approximate total question budget")
    p.add_argument("--sample-per-task", type=int, default=2000,
                   help="maximum examples per task")
    p.add_argument("--describe-share", type=float, default=0.5,
                   help="share of a describable task's examples that carry published "
                        "option definitions")
    p.add_argument("--generated-share", type=float, default=1.0,
                   help="max share of the budget for code-labelled rows (1.0 = unbounded)")
    p.add_argument("--source-cap", type=float, default=0.15,
                   help="max share of any split from one source row")
    p.add_argument("--no-source-cap", action="store_true")
    p.add_argument("--no-di-filter", action="store_true",
                   help="skip the Decision Index filter (audits only)")
    p.add_argument("--no-quality-filter", action="store_true",
                   help="keep schemas whose options are bare numerals with no definition")
    p.add_argument("--verbose", action="store_true", help="a line per task")


def generate_kw(a: argparse.Namespace) -> dict:
    return dict(raw_root=a.raw_root, rows=a.rows, target_questions=a.target_questions,
                sample_per_task=a.sample_per_task, describe_share=a.describe_share,
                generated_share=a.generated_share,
                source_cap=None if a.no_source_cap else a.source_cap,
                di_filter=not a.no_di_filter, quality_filter=not a.no_quality_filter,
                verbose=a.verbose)


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)

    def common(p, seed=True, blocklist=True):
        if seed:
            p.add_argument("--seed", type=int, default=0)
        if blocklist:
            p.add_argument("--di-blocklist", type=Path, default=DI_BLOCKLIST)

    p = sub.add_parser("all", help=build.build_all.__doc__.split("\n")[0])
    add_generate(p)
    common(p)
    p.add_argument("--raw", type=Path, default=None,
                   help="pre-enrichment corpus (default: <out>-raw)")
    p.add_argument("--cache", type=Path, default=build.SCHEMA_CACHE)
    p.add_argument("--suite", type=Path, default=DI_SUITE)
    p.add_argument("--probes-dir", type=Path, default=build.PROBES)
    p.add_argument("--no-knowledge-probe", action="store_true",
                   help="skip the knowledge probe (it downloads from the Hub)")

    p = sub.add_parser("generate", help=build.generate.__doc__.split("\n")[0])
    add_generate(p)
    common(p)
    p.add_argument("--dry-run", action="store_true", help="report without writing")

    p = sub.add_parser("enrich", help=build.enrich.__doc__.split("\n")[0])
    p.add_argument("--src", type=Path, required=True)
    p.add_argument("--dst", type=Path, required=True)
    p.add_argument("--cache", type=Path, default=build.SCHEMA_CACHE)
    p.add_argument("--model", default=None)
    p.add_argument("--concurrency", type=int, default=64)
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--limit-schemas", type=int, default=None)
    p.add_argument("--enrich-all", action="store_true",
                   help="also rewrite protected tasks, whose criteria are the label")
    p.add_argument("--apply-only", action="store_true", help="make no calls; apply the cache")
    p.add_argument("--dry-run", action="store_true", help="print one prompt and stop")

    p = sub.add_parser("blocklist", help=build.blocklist.__doc__.split("\n")[0])
    p.add_argument("--out", type=Path, default=DI_BLOCKLIST)
    p.add_argument("--suite", type=Path, default=DI_SUITE)
    p.add_argument("--upstream", type=Path, default=None,
                   help="a Decision Index build workspace with the suite's upstream files")

    p = sub.add_parser("decontaminate", help=build.decontaminate.__doc__.split("\n")[0])
    p.add_argument("--src", type=Path, required=True)
    p.add_argument("--dst", type=Path, default=None, help="default: rewrite --src in place")
    p.add_argument("--blocklist", type=Path, default=DI_BLOCKLIST)

    p = sub.add_parser("select", help=build.select.__doc__.split("\n")[0])
    p.add_argument("corpus", type=Path)
    common(p, blocklist=False)
    p.add_argument("--per-task", type=int, default=10, help="devreal_sel examples per task")
    p.add_argument("--boost", action="append", default=None, metavar="PREFIX=N",
                   help="devreal_sel: N per task for task names starting with PREFIX "
                        "(repeatable; replaces the release boosts)")
    p.add_argument("--sel-questions", type=int, default=9_000,
                   help="approximate question count of devood_sel")
    p.add_argument("--devood-per-task", type=int, default=1)
    p.add_argument("--devood-boost", action="append", default=[], metavar="PREFIX=N")
    p.add_argument("--strict", action="store_true",
                   help="fail, rather than drop with a warning, when a held-out dev "
                        "family is also a testreal family")

    p = sub.add_parser("probes", help=build.probes.__doc__.split("\n")[0])
    p.add_argument("--out-dir", type=Path, default=build.PROBES)
    p.add_argument("--per-cell", type=int, default=60)
    p.add_argument("--no-depth", action="store_true")
    p.add_argument("--no-knowledge", action="store_true")
    p.add_argument("--blocklist", type=Path, default=DI_BLOCKLIST)

    p = sub.add_parser("audit", help=build.audit.__doc__.split("\n")[0])
    p.add_argument("corpus", type=Path)
    p.add_argument("--raw", type=Path, default=None,
                   help="the corpus before enrichment, to check protected tasks")
    p.add_argument("--suite", type=Path, default=DI_SUITE)
    p.add_argument("--blocklist", type=Path, default=DI_BLOCKLIST)
    p.add_argument("--limit", type=int, default=0,
                   help="sentinel audit on a random sample of this many train examples")
    p.add_argument("--sweep", action="store_true",
                   help="also walk the sentinel rate trade-off curve")
    p.add_argument("--no-verify", action="store_true")

    p = sub.add_parser("distill", help=build.distill.__doc__.split("\n")[0])
    p.add_argument("--teacher", required=True, help="teacher checkpoint directory")
    p.add_argument("--src", type=Path, required=True)
    p.add_argument("--dst", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--attn", default="flex", choices=["sdpa", "flex"])
    p.add_argument("--max-batch-tokens", type=int, default=98304)
    p.add_argument("--chunk", type=int, default=20_000)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    if a.stage == "all":
        ok = build.build_all(a.out, raw=a.raw, seed=a.seed, cache=a.cache,
                             di_blocklist=a.di_blocklist, suite=a.suite,
                             probes_dir=a.probes_dir,
                             knowledge_probe=not a.no_knowledge_probe, **generate_kw(a))
        return 0 if ok else 1
    if a.stage == "generate":
        build.generate(a.out, seed=a.seed, di_blocklist=a.di_blocklist, dry_run=a.dry_run,
                       **generate_kw(a))
    elif a.stage == "enrich":
        build.enrich(a.src, a.dst, cache=a.cache, model=a.model, concurrency=a.concurrency,
                     retries=a.retries, limit_schemas=a.limit_schemas,
                     enrich_all=a.enrich_all, apply_only=a.apply_only, dry_run=a.dry_run)
    elif a.stage == "blocklist":
        build.blocklist(a.out, suite=a.suite, upstream=a.upstream)
    elif a.stage == "decontaminate":
        build.decontaminate(a.src, a.dst, blocklist=a.blocklist)
    elif a.stage == "select":
        from lod.corpus.services.splits import SplitError
        try:
            build.select(a.corpus, per_task=a.per_task, boosts=a.boost,
                         sel_questions=a.sel_questions, devood_per_task=a.devood_per_task,
                         devood_boosts=a.devood_boost, drop_test_overlap=not a.strict,
                         seed=a.seed)
        except SplitError as e:
            print(e, file=sys.stderr)
            return 1
    elif a.stage == "probes":
        build.probes(a.out_dir, depth_per_cell=a.per_cell, depth=not a.no_depth,
                     knowledge=not a.no_knowledge, blocklist=a.blocklist)
    elif a.stage == "audit":
        ok = build.audit(a.corpus, raw=a.raw, suite=a.suite, blocklist=a.blocklist,
                         sentinel_limit=a.limit, sweep=a.sweep, verify=not a.no_verify)
        return 0 if ok else 1
    elif a.stage == "distill":
        build.distill(a.teacher, a.src, a.dst, batch_size=a.batch_size,
                      num_workers=a.num_workers, attn=a.attn,
                      max_batch_tokens=a.max_batch_tokens, chunk=a.chunk)
    return 0


if __name__ == "__main__":
    sys.exit(main())
