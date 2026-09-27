"""The corpus build, one function per stage. `scripts/build_corpus.py` is a thin CLI over it.

    generate      raw source rows -> questions: quotas, sampling, split routing, dedup, the
                  Decision Index filter, val. Writes the four splits and meta.json (each
                  task's row, split, family, licence).                      services.generate
    enrich        an LLM rewrites each schema's question and option criteria, once per
                  schema, cached; protected tasks are left alone.           services.enrich
    blocklist     hash the Decision Index suite into the contamination blocklist.
                                                                            services.decontaminate
    decontaminate drop every record whose state the blocklist catches.     services.decontaminate
    select        devreal_sel (task-stratified, with boosts), devood (family-disjoint dev)
                  and devood_sel (domain-matched to testreal).               services.splits
    probes        the controlled-depth long-context check and the knowledge-retention
                  probe; read at evaluation, never trained on.              services.probes
    audit         sentinel acceptance test, domain table, integrity checks, and the
                  Decision Index overlap report.                            services.audit
    distill       a teacher checkpoint writes `meta.teacher` into train; the student
                  trains with --distill-alpha.                              services.distill

`build_all` runs generate -> enrich -> decontaminate -> select -> probes -> audit with
the release settings, which is how the release corpus is rebuilt from a fetched raw
store (`scripts/fetch_data.py`) in one pass:

    uv run python scripts/build_corpus.py all --out data/lod-corpus

(The corpus the released models trained on was assembled incrementally -- an earlier
build, re-filtered when the blocklist grew, with the two newest source rows built on
their own and spliced in. Every row is independent of the others once routed, so the
one-pass build is the same recipe. It is not byte-identical to the historical files:
the full blocklist now applies at build time, and the val carve-out and the per-split
source cap draw over the whole corpus at once.)

For `lod-lille`, `distill` then annotates the result with `lod-stor`'s distribution:

    uv run python scripts/build_corpus.py distill --teacher runs/lod-stor-4b/best \\
        --src data/lod-corpus --dst data/lod-corpus-distilled
"""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable

from lod.corpus.repositories.corpus_store import CorpusStore
from lod.paths import ASSETS, DATA, DI_BLOCKLIST, DI_SUITE, RAW_ROOT

Log = Callable[[str], None]
SCHEMA_CACHE = DATA / "schema_cache.json"
# the answers the release corpus was enriched with; a new cache starts from them, so the
# release corpus rebuilds without an API key and only new schemas cost a call
RELEASE_SCHEMAS = ASSETS / "schema_cache.json"
PROBES = DATA / "probes"


def _blocklist(path: Path | None, required: bool = True):
    from lod.corpus.services.decontaminate import Blocklist

    bl = Blocklist.load(path or DI_BLOCKLIST)
    if bl is None and required:
        raise FileNotFoundError(
            f"no Decision Index blocklist at {path or DI_BLOCKLIST}: run "
            "`scripts/build_corpus.py blocklist`, or pass --no-di-filter knowingly")
    return bl


# ---- generate --------------------------------------------------------------------------

def generate(out: Path, *, raw_root: Path = RAW_ROOT, rows: Iterable[int] | None = None,
             target_questions: int = 400_000, sample_per_task: int = 2000, seed: int = 0,
             describe_share: float = 0.5, generated_share: float = 1.0,
             source_cap: float | None = 0.15, di_blocklist: Path | None = DI_BLOCKLIST,
             di_filter: bool = True, quality_filter: bool = True, dry_run: bool = False,
             verbose: bool = False, log: Log = print) -> dict:
    """Build the four splits and meta.json from the raw store at `raw_root`.

    `rows` restricts the build to those source-table rows (1-51). `source_cap` is the
    largest share of any split one source row may take (None: no cap); `generated_share`
    bounds the code-labelled rows' share of the budget (1.0: no bound). Returns the meta.
    """
    from rich.console import Console
    from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

    from lod.corpus.repositories import raw_store
    from lod.corpus.services import generate as gen

    if sample_per_task < 1 or target_questions < 1:
        raise ValueError("sample_per_task and target_questions must be positive")
    rows = None if rows is None else set(rows)
    if rows is not None and (not rows or any(r not in gen.ROWS for r in rows)):
        raise ValueError("rows must be in 1..51")
    raw_store.configure(raw_root)
    tasks = gen.registered_tasks(rows)
    if not tasks:
        raise RuntimeError(f"no source tasks available under {raw_root}; "
                           "fetch the raw rows first (scripts/fetch_data.py)")
    blocklist = _blocklist(di_blocklist) if di_filter else None

    console = Console()
    progress = Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(),
                        "{task.completed}/{task.total}", console=console)
    with progress:
        bar = progress.add_task("Transforming source schemas", total=len(tasks))
        result = gen.build(
            tasks, target_questions=target_questions, sample_per_task=sample_per_task,
            seed=seed, describe_share=describe_share, generated_share=generated_share,
            source_cap=source_cap, blocklist=blocklist, quality_filter=quality_filter,
            log=progress.console.print, verbose=verbose,
            on_task=lambda: progress.advance(bar))

    meta = result.meta
    log(f"meta.p -> soft target on {result.softened:,} questions")
    if source_cap is not None:
        log(f"source cap {source_cap:.0%} per split dropped: {meta['capped']}")
    for line in gen.summarize(result.splits, result.tasks_by_name):
        log(line)
    log(f"dedup dropped: {meta['dropped']}")
    log(f"answerability gate dropped {len(meta['rejected'])} schemas")
    if dry_run:
        log("dry run: no files written")
        return meta
    store = CorpusStore(out)
    for split, examples in result.splits.items():
        log(f"wrote {store.path(split)}: {store.write(split, examples):,} examples")
    store.write_meta(meta)
    return meta


# ---- enrich ----------------------------------------------------------------------------

def enrich(src: Path, dst: Path, *, cache: Path = SCHEMA_CACHE, model: str | None = None,
           concurrency: int = 64, retries: int = 2, limit_schemas: int | None = None,
           enrich_all: bool = False, apply_only: bool = False, dry_run: bool = False,
           log: Log = print) -> dict:
    """Rewrite every unprotected schema's question and criteria, `src` -> `dst`.

    Answers are cached per schema in `cache`, so a rebuild re-pays only for new schemas;
    `apply_only` makes no calls and applies what the cache holds. `enrich_all` also
    rewrites the protected tasks, whose criteria are the label -- almost certainly wrong.
    Needs OPENROUTER_API_KEY (environment or `.env`) unless every schema is cached.
    """
    from lod.corpus.services import enrich as en

    src_store, dst_store = CorpusStore(src), CorpusStore(dst)
    corpus = {split: src_store.read_records(split) for split in src_store.present()}
    if not corpus:
        raise FileNotFoundError(f"no split files under {src}")
    prefixes = () if enrich_all else en.PROTECTED_PREFIXES
    reps, unstable = en.collect_schemas(corpus, prefixes)
    if prefixes:
        kept = sum(1 for rs in corpus.values() for r in rs if en.protected(r["task"], prefixes))
        log(f"{kept:,} records left untouched: their criteria are the label")
    n_records = sum(len(v) for v in corpus.values())
    log(f"{n_records:,} records, {len(reps):,} schemas, "
        f"{len(unstable):,} tasks with per-example options")

    cache = Path(cache)
    seed = cache if cache.exists() else RELEASE_SCHEMAS
    answers = json.loads(seed.read_text()) if seed.exists() else {}
    if dry_run:
        k, rep = next(iter(reps.items()))
        log(en.build_prompt(rep["task"], rep["options"], rep["question"], rep["state"],
                            rep["describe"]))
        return {}

    missing = [(k, r) for k, r in reps.items() if k not in answers]
    if limit_schemas:
        missing = missing[:limit_schemas]
    stats: dict = {}
    if missing and not apply_only:
        def save(c: dict) -> None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(c, ensure_ascii=False))

        stats = asyncio.run(en.enrich_schemas(
            missing, answers, model=model or en.DEFAULT_MODEL, concurrency=concurrency,
            retries=retries, save=save, log=log))
        log(f"schemas enriched: {stats['ok']:,} ok, {stats['failed']:,} failed, "
            f"{stats['tokens_in']:,} in + {stats['tokens_out']:,} out tokens")

    described = total = 0
    for split, records in corpus.items():
        out = []
        for record in records:
            r = en.apply(record, answers, unstable, prefixes)
            for q in r["questions"]:
                total += 1
                described += bool(q.get("descriptions") and any(q["descriptions"]))
            out.append(r)
        log(f"wrote {dst_store.path(split)}: {dst_store.write_records(split, out):,} records")
    dst_store.copy_from(src_store, "meta.json")
    log(f"questions with criteria: {described:,}/{total:,} "
        f"({100 * described / max(total, 1):.1f}%)")
    return {**stats, "questions": total, "described": described}


# ---- blocklist -------------------------------------------------------------------------

def blocklist(out: Path = DI_BLOCKLIST, *, suite: Path = DI_SUITE,
              upstream: Path | None = None, log: Log = print) -> Path:
    """Hash the Decision Index suite into the blocklist at `out`.

    `suite` is a rebuilt suite directory (its hash-verified rows). `upstream`, optional,
    is a Decision Index build workspace holding the suite's upstream source files: its
    whole test sets are hashed too, ahead of the suite rows, which is how the release
    blocklist was built.
    """
    from itertools import chain

    from lod.corpus.services import decontaminate as dc

    items = dc.suite_items(Path(suite))
    if upstream is not None:
        items = chain(dc.upstream_items(Path(upstream)), items)
    bl, per = dc.build_blocklist(items)
    bl.save(out)
    log(str(dict(per)))
    log(f"wrote {out}: {bl.long.size:,} shingles over {len(bl):,} long items, "
        f"{bl.short.size:,} short items")
    return Path(out)


# ---- decontaminate ---------------------------------------------------------------------

def decontaminate(src: Path, dst: Path | None = None, *,
                  blocklist: Path = DI_BLOCKLIST, log: Log = print) -> dict[str, int]:
    """Drop every record whose state the blocklist catches, `src` -> `dst` (default: in
    place). The same filter `generate` applies per task, for a corpus built before the
    blocklist last grew. -> records dropped per split."""
    from lod.corpus.services.decontaminate import Blocklist

    bl = Blocklist.load(blocklist)
    if bl is None:
        raise FileNotFoundError(f"no Decision Index blocklist at {blocklist}")
    src_store = CorpusStore(src)
    dst_store = CorpusStore(dst) if dst is not None else src_store
    if dst_store.root != src_store.root:
        dst_store.copy_from(src_store, "meta.json")
    out = {}
    for split in src_store.present():
        dropped: Counter = Counter()
        kept = dst_store.write_lines(split, bl.filter_lines(src_store.iter_lines(split),
                                                            dropped))
        out[split] = sum(dropped.values())
        log(f"{split}: kept {kept:,}, dropped {out[split]:,} "
            f"{dict(dropped.most_common(6))}")
    return out


# ---- select ----------------------------------------------------------------------------

def select(corpus: Path, *, per_task: int = 10, boosts=None, sel_questions: int = 9_000,
           devood_per_task: int = 1, devood_boosts=(), drop_test_overlap: bool = True,
           seed: int = 0, log: Log = print) -> dict:
    """Write devreal_sel, devood and devood_sel (+ devood.json, devood.txt) into `corpus`.

    `boosts` (default: the release `DEVSEL_BOOSTS`) raise devreal_sel's per-task count
    for task-name prefixes. `drop_test_overlap` drops, with a warning, dev families that
    are also testreal families instead of failing; the release corpus is built with it.
    """
    from lod.corpus.services import splits as sp

    store = CorpusStore(corpus)
    boosts = sp.DEVSEL_BOOSTS if boosts is None else sp.parse_boosts(boosts)

    dev = store.read("devreal")
    sel, taken = sp.devsel(dev, per_task, boosts, seed)
    n = store.write("devreal_sel", sel)
    log(f"{store.path('devreal')} {len(dev):,} examples, {len(taken):,} tasks -> "
        f"{store.path('devreal_sel')} {n:,} examples, "
        f"{sum(len(e.questions) for e in sel):,} questions ({per_task} per task)")
    for prefix, k in boosts:
        hit = {t: v for t, v in taken.items() if t.startswith(prefix)}
        log(f"  boosted {prefix}*: {len(hit)} tasks, {sum(hit.values()):,} examples "
            f"(up to {k} each)")

    result = sp.devood(
        store.read_meta(), sp.rows_of(store.iter_lines("devreal")),
        sp.rows_of(store.iter_lines("testreal")), sel_questions=sel_questions,
        per_task=devood_per_task, boosts=devood_boosts,
        drop_test_overlap=drop_test_overlap, seed=seed, corpus=str(corpus))
    store.write_lines("devood", (line for _, _, line in result.devood))
    store.write_lines("devood_sel", (line for _, _, line in result.sel))
    store.write_json("devood.json", result.summary, default=str)
    report = "\n".join(result.report)
    store.write_text("devood.txt", report + "\n")
    log(report)
    return result.summary


# ---- probes ----------------------------------------------------------------------------

def probes(out_dir: Path = PROBES, *, depth_per_cell: int = 60, depth: bool = True,
           knowledge: bool = True, blocklist: Path = DI_BLOCKLIST,
           log: Log = print) -> dict[str, int]:
    """Write `depth_check.jsonl` and `knowledge.jsonl` into `out_dir`, both filtered
    through the Decision Index blocklist. The knowledge probe downloads from the Hub."""
    from lod.corpus.services import probes as pr

    store = CorpusStore(out_dir)
    bl = _blocklist(blocklist, required=False)
    if bl is None:
        log(f"WARNING: no blocklist at {blocklist}; probes are not DI-filtered")
    written = {}
    if depth:
        examples, cells = pr.depth_check(depth_per_cell)
        if bl is not None:
            examples, dropped = bl.filter(examples)
            log(f"depth check: DI overlap dropped {dropped}")
        written["depth_check"] = store.write("depth_check", examples)
        log(f"wrote {written['depth_check']} examples to {store.path('depth_check')}")
        for line in pr.depth_table(cells):
            log(line)
    if knowledge:
        examples = pr.knowledge_probe()
        n0 = len(examples)
        if bl is not None:
            examples, dropped = bl.filter(examples)
            log(f"knowledge probe: DI blocklist dropped {dropped} of {n0}")
        written["knowledge"] = store.write("knowledge", examples)
        by = Counter(e.task for e in examples)
        log(f"wrote {written['knowledge']} probe questions to "
            f"{store.path('knowledge')}: {dict(by)}")
    return written


# ---- audit -----------------------------------------------------------------------------

def audit(corpus: Path, *, raw: Path | None = None, suite: Path | None = DI_SUITE,
          blocklist: Path = DI_BLOCKLIST, sentinel_limit: int = 0, sweep: bool = False,
          verify: bool = True, log: Log = print) -> bool:
    """Audit `corpus` and write the reports into it:

    sentinel_audit.json  the sentinel acceptance test on train (`sentinel_limit` > 0
                         audits a random sample; `sweep` walks the rate trade-off curve)
    domain_table.md      questions per domain and split
    verify               integrity counts, printed (with `raw`, the corpus before
                         enrichment, also checks protected tasks were left alone)
    di_overlap.json      Decision Index overlap by benchmark and task, if `suite` exists

    -> whether the sentinel acceptance test and the integrity checks all pass.
    """
    import random

    from lod.corpus.services import audit as au

    store = CorpusStore(corpus)
    ok = True

    examples = store.read("train")
    if sentinel_limit and sentinel_limit < len(examples):
        # a random sample: the file is written task by task, so its head is one task
        examples = random.Random(0).sample(examples, sentinel_limit)
    if sweep:
        rows = au.sentinel_sweep(examples)
        log(f"{'p_withheld':>11} {'p_decoy':>8} {'share':>8} {'P(correct|present)':>20}")
        for r in rows:
            log(f"{r['p_withheld']:11.3f} {r['p_decoy']:8.2f} "
                f"{r['sentinel_share'] * 100:7.1f}% {r['p_correct_given_present']:20.3f}")
        store.write_json("sentinel_sweep.json", rows)
    result = au.sentinel_audit(examples)
    del examples
    store.write_json("sentinel_audit.json", result)
    log("sentinel acceptance")
    for key, v, bar, passed in au.acceptance(result):
        ok &= passed
        log(f"  {key:26} {v!s:>10}  <= {bar:<6} {'ok' if passed else 'FAIL'}")

    counts = au.domain_counts(store.read_meta(),
                              {s: store.iter_records(s) for s in store.present()})
    if counts["unmapped"]:
        log(f"  WARNING: questions in rows no domain claims: {counts['unmapped']}")
    table = "\n".join(au.domain_table(counts))
    store.write_text("domain_table.md", table + "\n")
    log(table)

    if verify:
        bl = _blocklist(blocklist, required=False)
        bad, flagged = au.verify(store, bl, CorpusStore(raw) if raw else None)
        log("integrity")
        for k in au.CHECKS:
            log(f"  {k:<32} {bad[k]:>8}")
        for f in flagged[:20]:
            log(f"    wording_predicts: {f}")
        ok &= not any(bad.values())

    if suite is not None and Path(suite).exists():
        from lod.corpus.services.decontaminate import SuiteIndex, overlap_report, suite_items

        index = SuiteIndex(suite_items(Path(suite)))
        log(f"DI overlap index: {index.describe()}")
        report = {}
        for split in store.present(("train", "devreal", "testreal")):
            r = report[split] = overlap_report(index, store.iter_records(split))
            log(f"{split}: {r['overlapping']:,} of {r['records']:,} records overlap the "
                f"Decision Index suite ({r['overlapping'] / max(r['records'], 1):.3%})")
            for fm, c in list(r["by_di_benchmark"].items())[:15]:
                log(f"  {fm:<34} {c:>7,}")
        store.write_json("di_overlap.json", report)
    else:
        log(f"no Decision Index suite at {suite}: overlap report skipped")
    log("PASS" if ok else "FAIL")
    return ok


# ---- distill ---------------------------------------------------------------------------

def distill(teacher: str, src: Path, dst: Path, *, batch_size: int = 32,
            num_workers: int = 8, attn: str = "flex", max_batch_tokens: int = 98304,
            chunk: int = 20_000, log: Log = print) -> int:
    """Annotate `src`'s train split with `teacher`'s distribution into `dst`; every other
    file of `src` is linked unchanged. `chunk` examples are scored per pass, which bounds
    host memory. -> questions annotated."""
    from lod.corpus.services import distill as ds

    t = ds.load_teacher(teacher, attn)
    log(f"teacher {teacher}: {t.describe()}")
    src_store, dst_store = CorpusStore(src), CorpusStore(dst)
    dst_store.link_from(src_store, skip={"train.jsonl"})
    lines = src_store.read_lines("train")
    mixed = kept = 0
    with dst_store.writer("train") as out:
        for c0 in range(0, len(lines), chunk):
            recs = [json.loads(ln) for ln in lines[c0:c0 + chunk]]
            m, too_long = ds.annotate(t, recs, batch_size=batch_size,
                                      num_workers=num_workers,
                                      max_batch_tokens=max_batch_tokens)
            mixed += m
            for rec in recs:
                kept += len(rec["questions"])
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            log(f"  {c0 + len(recs):,}/{len(lines):,} examples, {mixed:,} questions carry "
                f"the teacher ({too_long:,} too long for it in this chunk)")
    log(f"wrote {dst_store.path('train')}: {mixed:,} of {kept:,} questions carry "
        f"meta.teacher; other splits linked unchanged")
    return mixed


# ---- everything ------------------------------------------------------------------------

def build_all(out: Path, *, raw: Path | None = None, raw_root: Path = RAW_ROOT,
              seed: int = 0, cache: Path = SCHEMA_CACHE, di_blocklist: Path = DI_BLOCKLIST,
              suite: Path | None = DI_SUITE, probes_dir: Path = PROBES,
              knowledge_probe: bool = True, log: Log = print, **generate_kw) -> bool:
    """generate -> enrich -> decontaminate -> select -> probes -> audit, release settings.

    The pre-enrichment corpus goes to `raw` (default: `<out>-raw` beside `out`) and is
    kept: `audit` compares against it, and a re-enrichment starts from it. Extra keyword
    arguments go to `generate`. -> whether the audit passed.
    """
    out = Path(out)
    raw = Path(raw) if raw is not None else out.with_name(out.name + "-raw")
    stages = ("generate", "enrich", "decontaminate", "select", "probes", "audit")

    def stage(name: str) -> None:
        log(f"\n==== {name} ({stages.index(name) + 1}/{len(stages)}) ====")

    stage("generate")
    generate(raw, raw_root=raw_root, seed=seed, di_blocklist=di_blocklist, log=log,
             **generate_kw)
    stage("enrich")
    enrich(raw, out, cache=cache, log=log)
    stage("decontaminate")
    decontaminate(out, blocklist=di_blocklist, log=log)
    stage("select")
    select(out, seed=seed, log=log)
    stage("probes")
    probes(probes_dir, knowledge=knowledge_probe, blocklist=di_blocklist, log=log)
    stage("audit")
    return audit(out, raw=raw, suite=suite, blocklist=di_blocklist, log=log)


__all__ = ["audit", "blocklist", "build_all", "decontaminate", "distill",
           "enrich", "generate", "probes", "select"]
