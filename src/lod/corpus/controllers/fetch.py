"""The fetch stage: download every raw source the corpus is built from into one raw root.

Each `Fetcher` names the source-table rows it serves, what it writes, and how to run it;
`FETCHERS` is the whole stage in run order. Fetching is idempotent and resumable -- a
fetcher skips whatever is already on disk unless `refresh` -- so an interrupted run costs
only what it had not finished. Nothing here interprets a row: the build stage reads the
raw root and never touches the network.

Rows that need no fetch are listed too, so every row of the source table is accounted for:
`GENERATED` rows are code-labelled generators, `BUILD_TIME` rows are read from the Hub
while building, and `NO_TASKS` rows are retired or staged without tasks.

Layout under the raw root (`lod.paths.RAW_ROOT`, env `LOD_RAW_ROOT`):

    store/<key>.jsonl, <key>.features.json   raw rows and ClassLabel names (older roots: v4-raw/)
    huggingface/<repo>/                       Hub snapshots read directly (crowd labels)
    loghub/, pdfs/, longdocs/, gefs/          per-source files
    *.jsonl, *.zip, *.xlsx, <sha1 prefix>     per-source caches and cached API responses
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from lod import paths
from lod.corpus.repositories import (
    arxiv_pdfs,
    bespoke,
    chess_pool,
    hub,
    loghub,
    multilingual,
    reallang,
    sentiment,
    toolspecs,
    topic,
)
from lod.corpus.repositories import raw_store as store
from lod.corpus.services.sources.base import CACHE


@dataclass
class Options:
    rows: set[int] | None = None       # restrict Hub specs to these source-table rows
    refresh: bool = False
    row1_limit: int | None = None      # cap the row-1 Hub sweep (smoke runs)
    di_repos: Path | None = None       # Decision Index upstream repos (row 46 re-match)
    chessbench: Path | None = None     # ChessBench test bag (row 43 exclusion)


@dataclass(frozen=True)
class Fetcher:
    name: str
    rows: tuple[int, ...]
    what: str
    run: Callable[[Options], Any]
    outputs: Callable[[Options], list[Path]]
    opt_in: bool = False               # only when asked for by name, and then always run


FETCHERS: tuple[Fetcher, ...] = (
    Fetcher("hub_rows", hub.rows_served(),
            "Hub datasets of the source table, the row-1 sweep and dedicated readers -> store",
            lambda o: hub.fetch_spec_rows(o.rows, o.refresh, o.row1_limit),
            lambda o: hub.row_outputs(o.rows, o.row1_limit)),
    Fetcher("hub_features", hub.rows_served(),
            "published ClassLabel names for every stored Hub spec -> store sidecars",
            lambda o: hub.fetch_features(o.rows, o.refresh, o.row1_limit),
            lambda o: hub.feature_outputs(o.rows, o.row1_limit)),
    Fetcher("natural_instructions", (3,), "Super-NaturalInstructions task index and rows",
            lambda o: bespoke.natural_instructions(o.refresh),
            lambda o: bespoke.ni_outputs()),
    Fetcher("gharchive", (5,), "GH Archive single-label issues",
            lambda o: bespoke.gharchive_issues(o.refresh),
            lambda o: bespoke.gharchive_outputs()),
    Fetcher("arxiv_pdfs", (7,), "CC-licensed arXiv PDFs with outlines -> pdfs/",
            lambda o: arxiv_pdfs.fetch(),
            lambda o: arxiv_pdfs.outputs()),
    Fetcher("nvd", (9,), "NVD CVEs with CVSS v3 metrics, via the API",
            lambda o: bespoke.nvd_cves(o.refresh),
            lambda o: bespoke.nvd_outputs()),
    Fetcher("cwe", (9,), "MITRE CWE catalogue (option descriptions for nvd_cwe)",
            lambda o: bespoke.cwe_catalogue(o.refresh),
            lambda o: bespoke.cwe_outputs()),
    Fetcher("loghub", (14,), "LogHub 2k-line samples of 16 systems -> loghub/",
            lambda o: loghub.fetch(refresh=o.refresh),
            lambda o: loghub.outputs()),
    Fetcher("spdx", (15,), "SPDX licence list and licence texts (URL cache)",
            lambda o: bespoke.spdx_licences(o.refresh),
            lambda o: bespoke.spdx_outputs()),
    Fetcher("crowd", (20, 21), "GoEmotions and Measuring Hate Speech, aggregated per item",
            lambda o: bespoke.crowd_labels(o.refresh),
            lambda o: bespoke.crowd_outputs()),
    Fetcher("chaosnli", (25,), "ChaosNLI release (100 labels per item)",
            lambda o: bespoke.chaosnli_release(o.refresh),
            lambda o: bespoke.chaosnli_outputs()),
    Fetcher("spf", (26,), "Survey of Professional Forecasters microdata, recession rows",
            lambda o: bespoke.spf_recession(o.refresh),
            lambda o: bespoke.spf_outputs()),
    Fetcher("gefs", (27,), "GEFS v12 reforecast members, sampled PoP rows (extra: gefs)",
            lambda o: bespoke.gefs_forecasts(o.refresh),
            lambda o: bespoke.gefs_outputs()),
    Fetcher("manifold", (28,), "Manifold resolved binary markets",
            lambda o: bespoke.manifold_markets(o.refresh),
            lambda o: bespoke.manifold_outputs()),
    Fetcher("chess_pool", (43,), "Lichess puzzle pool disjoint from row 13 and ChessBench",
            lambda o: chess_pool.fetch(o.chessbench, refresh=o.refresh),
            lambda o: chess_pool.outputs()),
    Fetcher("sentiment", (44,), "English sentiment, emotion and stance -> store v14s_*",
            lambda o: sentiment.fetch(refresh=o.refresh),
            lambda o: sentiment.outputs()),
    Fetcher("topic", (45,), "English topic, intent and question type -> store v14_topic_*",
            lambda o: topic.fetch(refresh=o.refresh),
            lambda o: topic.outputs()),
    Fetcher("multilingual", (46,), "multilingual classification -> store ml46_*",
            lambda o: multilingual.fetch(refresh=o.refresh, di_repos=o.di_repos),
            lambda o: multilingual.outputs()),
    Fetcher("reallang", (50,), "broad real-language decisions -> store rl_*",
            lambda o: reallang.fetch(refresh=o.refresh),
            lambda o: reallang.outputs()),
    Fetcher("longdocs", (51,), "QuALITY, CUAD and LexGLUE ECtHR -> longdocs/",
            lambda o: bespoke.long_documents(o.refresh),
            lambda o: bespoke.longdocs_outputs()),
    # opt-in: not part of a normal fetch
    Fetcher("nvd_feeds", (9,), "NVD static yearly feeds -> nvd/ (replace the API cache)",
            lambda o: bespoke.nvd_feeds(o.refresh),
            lambda o: bespoke.nvd_feed_outputs(), opt_in=True),
    Fetcher("toolgate_http", (38,),
            "regenerate the package's frozen toolgate_http.json from apis.guru",
            lambda o: toolspecs.regenerate(openapi=True, scenarios=False),
            lambda o: [toolspecs.toolgate.HTTP_SPECS], opt_in=True),
    Fetcher("toolgate_scenarios", (38,),
            "regenerate the package's frozen toolgate_scenarios.json with an LLM "
            "(OPENROUTER_API_KEY)",
            lambda o: toolspecs.regenerate(openapi=False, scenarios=True),
            lambda o: [toolspecs.toolgate.SCENARIOS], opt_in=True),
)
BY_NAME = {f.name: f for f in FETCHERS}

# rows that need nothing fetched, and why
GENERATED = {
    33: "rule application (synth/rules.py)",
    35: "randomised-attribute entities (synth/entities.py)",
    36: "accessibility trees (synth/browser.py)",
    37: "control and sensors (synth/sensors.py)",
    38: "tool-call gating (synth/toolgate.py; frozen package seeds)",
    39: "entity resolution (synth/entityres.py)",
    40: "code and diffs (synth/diffs.py)",
    41: "scheduling (synth/scheduling.py)",
    42: "multi-hop and ranking (synth/relational.py)",
    47: "compositional rules (synth/compose.py)",
    48: "exact posteriors (synth/posterior.py)",
    49: "many-option matching (synth/catalog.py)",
    51: "generated long contexts (synth/longctx.py)",
}
BUILD_TIME = {2: "BIG-bench configs are streamed from the Hub while building"}
NO_TASKS = {
    11: "clinical trials: staged, but its only labels are a sampling hierarchy",
    29: "GitHub pull requests: removed from the source table",
    30: "record fields: removed from the source table",
}
TABLE_ROWS = range(1, 52)


def coverage() -> dict[int, list[str]]:
    """Source-table row -> what provides it: fetcher names, or a no-fetch reason."""
    out: dict[int, list[str]] = {r: [] for r in TABLE_ROWS}
    for f in FETCHERS:
        for r in f.rows:
            out[r].append(f.name + (" (opt-in)" if f.opt_in else ""))
    for table in (GENERATED, BUILD_TIME, NO_TASKS):
        for r, why in table.items():
            out[r].append(why)
    return out


def _check_root(raw_root: str | Path | None) -> Path:
    """Point the store at `raw_root`, which must be the root the source modules cache in.

    Several readers bind `lod.paths.RAW_ROOT` at import, so a second root cannot be
    honoured halfway; select one with `LOD_RAW_ROOT` before importing `lod`.
    """
    root = Path(raw_root or paths.RAW_ROOT).expanduser()
    if root.resolve() != Path(CACHE).expanduser().resolve():
        raise ValueError(f"raw root {root} differs from lod.paths.RAW_ROOT ({CACHE}); "
                         f"set LOD_RAW_ROOT={root} before importing lod")
    store.configure(root)
    return root


def select(rows: set[int] | None = None, only: set[str] | None = None) -> list[Fetcher]:
    """The fetchers for `rows` and/or `only` (names). Opt-in fetchers need `only`."""
    unknown = (only or set()) - set(BY_NAME)
    if unknown:
        raise ValueError(f"unknown fetchers {sorted(unknown)}; known: {sorted(BY_NAME)}")
    out = []
    for f in FETCHERS:
        if only is not None and f.name not in only:
            continue
        if f.opt_in and (only is None or f.name not in only):
            continue
        if rows is not None and not set(f.rows) & rows:
            continue
        out.append(f)
    return out


def _status(f: Fetcher, o: Options) -> tuple[int, int, float]:
    outs = f.outputs(o)
    have = [p for p in outs if p.exists() and p.stat().st_size > 0]
    return len(have), len(outs), sum(p.stat().st_size for p in have) / 1e6


@dataclass
class Result:
    name: str
    rows: tuple[int, ...]
    status: str                        # planned | skipped | ok | failed
    detail: str = ""
    seconds: float = 0.0
    have: int = 0
    want: int = 0
    extra: dict = field(default_factory=dict)


def fetch(rows: set[int] | None = None, only: set[str] | None = None,
          refresh: bool = False, dry_run: bool = False,
          raw_root: str | Path | None = None, row1_limit: int | None = None,
          di_repos: Path | None = None, chessbench: Path | None = None) -> list[Result]:
    """Run (or with `dry_run`, only plan) the fetchers selected by `rows` / `only`.

    A fetcher whose outputs are all present is skipped unless `refresh` (an opt-in one,
    asked for by name, always runs); a failing fetcher is reported and the rest still run.
    """
    _check_root(raw_root)
    o = Options(rows=rows, refresh=refresh, row1_limit=row1_limit,
                di_repos=di_repos, chessbench=chessbench)
    results: list[Result] = []
    for f in select(rows, only):
        have, want, _ = _status(f, o)
        complete = want > 0 and have == want
        r = Result(f.name, f.rows, "planned", have=have, want=want)
        if complete and not (refresh or f.opt_in):
            r.status, r.detail = "skipped", "present"
        elif dry_run:
            todo = want if (refresh or f.opt_in) else want - have
            r.detail = f"would fetch {todo} of {want}" if want else "would run"
        else:
            print(f"=== {f.name} (rows {', '.join(map(str, f.rows))}): {f.what}", flush=True)
            t0 = time.time()
            try:
                got = f.run(o)
                r.status, r.detail = "ok", "" if got is None else str(got)
            except Exception as e:  # noqa: BLE001 -- one source must not stop the rest
                r.status, r.detail = "failed", f"{type(e).__name__}: {str(e)[:200]}"
                print(f"  FAILED {r.detail}", flush=True)
            r.seconds = time.time() - t0
            r.have, r.want, _ = _status(f, o)
        results.append(r)
    return results


def inventory(raw_root: str | Path | None = None) -> dict[str, Any]:
    """What is on disk, per fetcher, and which stored keys no fetcher accounts for."""
    root = _check_root(raw_root)
    o = Options()
    per = []
    claimed: set[Path] = set()
    for f in FETCHERS:
        have, want, mb = _status(f, o)
        per.append({"name": f.name, "rows": f.rows, "have": have, "want": want,
                    "mb": round(mb, 1), "opt_in": f.opt_in})
        claimed.update(p.resolve() for p in f.outputs(o))
    stored = sorted(store.RAW.glob("*.jsonl")) if store.RAW.is_dir() else []
    unclaimed = [p.stem for p in stored if p.resolve() not in claimed]
    return {"raw_root": str(root), "store": str(store.RAW), "fetchers": per,
            "store_keys": len(stored), "unclaimed_keys": unclaimed}
