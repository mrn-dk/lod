"""Sources whose download lives beside their reader.

These sources are not a table of rows: an API paged under a rate limit (NVD, Manifold),
an hourly event archive filtered on the fly (GH Archive), crowd labels aggregated per item
(GoEmotions, Measuring Hate Speech), a GRIB2 ensemble sampled at gridpoints (GEFS). The
reader module already knows how to download and cache each one, and where its cache
lives, so the fetch stage calls that code rather than keeping a second copy of it. Each
call is a no-op when its cache exists, unless `refresh`.

All of these caches sit directly under the raw root (`lod.paths.RAW_ROOT`).
"""

from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path

from lod.corpus.repositories import hub, web
from lod.corpus.repositories import raw_store as store
from lod.corpus.services.sources.base import CACHE
from lod.corpus.services.sources.real import (
    chaosnli,
    crowd,
    descriptions,
    gefs,
    gharchive,
    instructions,
    longdocs,
    manifold,
    nvd,
    spdx,
    spf,
)


def _all_present(paths: list[Path]) -> bool:
    return all(p.exists() and p.stat().st_size > 0 for p in paths)


# ---- row 3: Super-NaturalInstructions -------------------------------------------------

def ni_outputs() -> list[Path]:
    return [instructions._NI_CACHE, instructions._NI_ROWS]


def natural_instructions(refresh: bool = False) -> str:
    """The classification-shaped NI tasks, and their rows, from one scan of the Hub."""
    idx = instructions.ni_tasks_index(refresh=refresh)
    by_task = instructions.ni_rows_by_task(refresh=refresh)
    return f"{len(idx):,} tasks, {sum(map(len, by_task.values())):,} rows"


# ---- row 5: GH Archive issues ---------------------------------------------------------

def gharchive_outputs() -> list[Path]:
    return [gharchive._CACHE_FILE]


def gharchive_issues(refresh: bool = False) -> str:
    return f"{len(gharchive.collect(refresh=refresh)):,} issues"


# ---- row 9: NVD, plus MITRE's CWE catalogue for option descriptions ------------------

def nvd_outputs() -> list[Path]:
    return [nvd._CACHE_FILE]


def nvd_cves(refresh: bool = False) -> str:
    return f"{len(nvd.download(refresh=refresh)):,} CVEs"


NVD_FEED = "https://nvd.nist.gov/feeds/json/cve/2.0/nvdcve-2.0-{year}.json.gz"
NVD_FIRST_YEAR = 2002


def nvd_feed_outputs() -> list[Path]:
    return [store.RAW_ROOT / "nvd" / f"nvdcve-2.0-{y}.json.gz"
            for y in range(NVD_FIRST_YEAR, date.today().year + 1)]


def nvd_feeds(refresh: bool = False) -> str:
    """NVD's static yearly JSON feeds. Opt-in: when present, `nvd.download` reads them
    *instead of* the API cache, so fetching them changes which CVEs row 9 draws on."""
    for p in nvd_feed_outputs():
        year = p.name.split("-")[-1].split(".")[0]
        web.download(NVD_FEED.format(year=year), p, refresh=refresh, timeout=120, retries=4)
    return f"{len(nvd_feed_outputs())} yearly feeds"


def cwe_outputs() -> list[Path]:
    return [descriptions._CWE_CACHE]


def cwe_catalogue(refresh: bool = False) -> str:
    return f"{len(descriptions.cwe_catalogue(refresh=refresh)):,} CWE entries"


# ---- row 15: SPDX licence texts -------------------------------------------------------

def _url_cache(url: str) -> Path:
    """Where `sources.base.fetch` caches a URL (without creating the directory)."""
    return CACHE / hashlib.sha1(url.encode()).hexdigest()[:20]


def spdx_outputs() -> list[Path]:
    return [_url_cache(f"{spdx.BASE}/licenses.json")]


def spdx_licences(refresh: bool = False) -> str:
    """Warm the URL cache the SPDX reader reads through: the listing and each licence."""
    if refresh:
        for p in spdx_outputs():
            p.unlink(missing_ok=True)
    ids = spdx.licence_ids()
    spdx.published_names()
    return f"{len(ids)} licence texts"


# ---- rows 20, 21: crowd labels --------------------------------------------------------

def crowd_outputs() -> list[Path]:
    return [crowd._GE_CACHE, crowd._HS_CACHE]


def crowd_labels(refresh: bool = False) -> str:
    """Snapshot both Hub datasets, then aggregate their per-annotator rows per item."""
    for repo, patterns in hub.CROWD_SNAPSHOTS:
        hub.snapshot(repo, patterns, refresh=refresh)
    ge = crowd.aggregate_go_emotions(refresh=refresh)
    hs = crowd.aggregate_hate_speech(refresh=refresh)
    return f"{len(ge):,} GoEmotions items, {len(hs):,} Measuring Hate Speech items"


# ---- row 25: ChaosNLI -----------------------------------------------------------------

def chaosnli_outputs() -> list[Path]:
    return [chaosnli._ZIP]


def chaosnli_release(refresh: bool = False) -> str:
    return f"{len(chaosnli.download(refresh=refresh)) / 1e6:.1f} MB zip"


# ---- row 26: Survey of Professional Forecasters ---------------------------------------

SPF_ROWS = CACHE / "spf_recess.jsonl"


def spf_outputs() -> list[Path]:
    return [spf.MICRO, SPF_ROWS]


def spf_recession(refresh: bool = False) -> str:
    """The SPF microdata workbook, and the (forecaster, quarter, horizon) rows built from
    it. The reader only emits tasks once the rows file exists."""
    spf.download(refresh=refresh)
    return f"{len(spf.build(refresh=refresh)):,} forecasts"


# ---- row 27: GEFS reforecast ----------------------------------------------------------

# One cycle every six weeks or so across 2019, so the sample spans the seasons. The order
# matters: gridpoints are drawn from one RNG stream across the cycles in this order.
GEFS_CYCLES = ("2019010100", "2019022800", "2019041500", "2019060100",
               "2019071500", "2019090100", "2019101500", "2019120100")
GEFS_POINTS = 900


def gefs_outputs() -> list[Path]:
    return [gefs._ROWS]


def gefs_forecasts(refresh: bool = False) -> str:
    """Download five ensemble members per cycle (plus verifying analyses) and sample
    gridpoints into (forecast PoP, outcome) rows. Needs the `gefs` extra."""
    return f"{len(gefs.build(GEFS_CYCLES, points=GEFS_POINTS, refresh=refresh)):,} rows"


# ---- row 28: Manifold -----------------------------------------------------------------

def manifold_outputs() -> list[Path]:
    return [manifold._CACHE_FILE]


def manifold_markets(refresh: bool = False) -> str:
    return f"{len(manifold.download(refresh=refresh)):,} resolved markets"


# ---- row 51: long documents -----------------------------------------------------------

def longdocs_outputs() -> list[Path]:
    d = CACHE / "longdocs"
    return ([d / "CUADv1.json"]
            + [d / f"QuALITY.v1.0.1.htmlstripped.{s}" for s in ("train", "dev", "test")]
            + [d / f"ecthr_{s}.jsonl" for s in ("train", "validation", "test")])


def long_documents(refresh: bool = False) -> str:
    """QuALITY and CUAD release archives, and LexGLUE ECtHR A+B joined per case."""
    if _all_present(longdocs_outputs()) and not refresh:
        return "present"
    longdocs.fetch()
    return f"{len(longdocs_outputs())} files"
