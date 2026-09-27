"""Hugging Face Hub datasets: rows into the raw store, label names beside them, and files.

Most of the source table is Hub datasets read through one generic adapter
(`services/sources/real/hf.py`): `table.SPECS` names them, and row 1 is the frozen Hub
sweep (`hub_sweep.specs_from_survey`). A few Hub datasets are read by a dedicated module
instead and are excluded from the sweep on purpose; their rows are fetched here all the
same, under the store key those modules read (`DEDICATED_SPECS`).

Rows are written exactly as the Hub streams them (`raw_store.save`). ClassLabel columns
arrive as integers, so a second, metadata-only pass records the published names in a
`.features.json` sidecar (`fetch_features`); without it a 100-way label reaches the
corpus as the options "0".."99".
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from lod.corpus.repositories import raw_store as store
from lod.corpus.services.sources.real import grounding, kalshi, lichess
from lod.corpus.services.sources.real.hf import HFSpec, fetch_rows
from lod.corpus.services.sources.real.hub_sweep import specs_from_survey
from lod.corpus.services.sources.real.table import SPECS

# Hub datasets a dedicated module reads from the raw store. Row counts are the sample the
# modules were measured on: the first N train rows the Hub streams.
DEDICATED_SPECS: tuple[HFSpec, ...] = (
    HFSpec(13, lichess.PATH, lichess.LICENCE, notes="lichess.py"),
    HFSpec(26, kalshi.PATH, kalshi.LICENCE, max_rows=20_000,
           notes="kalshi.py; the whole dataset (10,016 markets)"),
    HFSpec(34, "copenlu/fever_gold_evidence", grounding.FEVER_LICENCE, max_rows=45_000,
           notes="grounding.py"),
    HFSpec(34, "tals/vitaminc", grounding.VITC_LICENCE, max_rows=45_000,
           notes="grounding.py"),
)

# the Hub snapshots `crowd.py` aggregates per item (one row per annotator upstream)
CROWD_SNAPSHOTS = (
    ("google-research-datasets/go_emotions", ("raw/*.parquet",)),
    ("ucberkeley-dlab/measuring-hate-speech", ("data/*.parquet",)),
)

NUMERIC = re.compile(r"^-?\d+(\.\d+)?$")


def specs(rows: set[int] | None = None, row1_limit: int | None = None) -> list[HFSpec]:
    """Every Hub spec the corpus reads through the raw store, optionally for some rows."""
    out = list(SPECS)
    if rows is None or 1 in rows:
        out += specs_from_survey(limit=row1_limit)
    out += DEDICATED_SPECS
    return [s for s in out if rows is None or s.row in rows]


def rows_served() -> tuple[int, ...]:
    """The source-table rows with at least one Hub spec (row 1 is the sweep)."""
    return tuple(sorted({s.row for s in SPECS} | {1} | {s.row for s in DEDICATED_SPECS}))


def row_outputs(rows: set[int] | None = None, row1_limit: int | None = None) -> list[Path]:
    return [store.path_for(s.key) for s in specs(rows, row1_limit)]


def fetch_spec_rows(rows: set[int] | None = None, refresh: bool = False,
                    row1_limit: int | None = None) -> dict[str, int]:
    """Stream each spec's rows into the store. Skips keys already stored unless `refresh`.

    A failing dataset is reported and skipped rather than fatal: the Hub is hundreds of
    independent repos, and one that moved must not cost the other few hundred.
    """
    got: dict[str, int] = {}
    for spec in specs(rows, row1_limit):
        if store.has(spec.key) and not refresh:
            continue
        t0 = time.time()
        try:
            n = store.save(spec.key, fetch_rows(spec))
        except Exception as e:  # noqa: BLE001 -- report and go on
            print(f"  FAIL  row {spec.row:2} {spec.key[:46]:46} "
                  f"{type(e).__name__}: {str(e)[:60]}", flush=True)
            continue
        got[spec.key] = n
        print(f"  ok    row {spec.row:2} {spec.key[:46]:46} {n:7,} rows "
              f"{time.time() - t0:5.1f}s", flush=True)
    return got


def classlabel_names(spec: HFSpec) -> dict[str, list[str]]:
    """Every ClassLabel column of `spec`'s dataset, as {column: names}.

    Tries the spec's own config first and falls back to discovery, in the same order
    `hf.fetch_rows` does: a sidecar keyed to a different config than the rows would
    silently mislabel every one of them.
    """
    from datasets import get_dataset_config_names, load_dataset_builder

    cfgs = [spec.config] if spec.config else None
    if cfgs is None:
        try:
            cfgs = get_dataset_config_names(spec.path) or [None]
        except Exception:
            cfgs = [None]
    out: dict[str, list[str]] = {}
    for cfg in cfgs[:3]:
        try:
            info = load_dataset_builder(spec.path, cfg).info
        except Exception:
            continue
        for col, feat in (info.features or {}).items():
            names = getattr(feat, "names", None)
            if names is None:    # Sequence(ClassLabel) keeps its names one level down
                names = getattr(getattr(feat, "feature", None), "names", None)
            if names:
                out.setdefault(col, [str(n) for n in names])
        if out:
            break
    return out


def feature_outputs(rows: set[int] | None = None,
                    row1_limit: int | None = None) -> list[Path]:
    """The sidecar of every spec whose rows are stored (only those can be annotated)."""
    return [store.features_path_for(s.key) for s in specs(rows, row1_limit)
            if store.has(s.key)]


def fetch_features(rows: set[int] | None = None, refresh: bool = False,
                   row1_limit: int | None = None) -> int:
    """Write the `.features.json` sidecar for every stored spec that lacks one.

    Metadata only (`load_dataset_builder(...).info`), so a store that already has its
    rows gains names without re-downloading any. Numeric names ("1".."13") are written
    through unchanged; the loader ignores them.
    """
    done = 0
    for spec in specs(rows, row1_limit):
        if not store.has(spec.key):
            continue
        if store.features_path_for(spec.key).exists() and not refresh:
            continue
        try:
            names = classlabel_names(spec)
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL  row {spec.row:2} {spec.key[:46]:46} "
                  f"{type(e).__name__}: {str(e)[:50]}", flush=True)
            continue
        store.save_features(spec.key, names)
        named = sum(1 for v in names.values() if not all(NUMERIC.match(x) for x in v))
        print(f"  ok    row {spec.row:2} {spec.key[:46]:46} "
              f"{len(names):2} ClassLabel col(s), {named:2} with names", flush=True)
        done += 1
    return done


# ---- files -------------------------------------------------------------------------

def file(repo: str, filename: str, revision: str | None = None) -> Path:
    """One file of a dataset repo, through the local Hugging Face cache."""
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(repo, filename, repo_type="dataset", revision=revision))


def parquet_rows(path: Path) -> list[dict]:
    import pyarrow.parquet as pq
    return pq.read_table(path).to_pylist()


def open_parquet(path: str):
    """A remote parquet file (`datasets/<repo>/<file>`), read lazily row group by group."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    return pq.ParquetFile(HfFileSystem().open(path))


def parquet_classlabels(pf) -> dict[str, list[str]]:
    """ClassLabel names from a parquet file's own `huggingface` schema metadata."""
    raw = (pf.schema_arrow.metadata or {}).get(b"huggingface")
    if not raw:
        return {}
    feats = json.loads(raw).get("info", {}).get("features", {})
    return {col: f["names"] for col, f in feats.items()
            if isinstance(f, dict) and f.get("_type") == "ClassLabel" and f.get("names")}


def builder_classlabels(repo: str, config: str | None) -> dict[str, list[str]]:
    from datasets import load_dataset_builder
    feats = load_dataset_builder(repo, config).info.features or {}
    return {k: list(v.names) for k, v in feats.items() if hasattr(v, "names")}


def snapshot_dir(repo: str) -> Path:
    """Where a raw Hub snapshot lives: `<raw root>/huggingface/<repo>`."""
    return store.RAW_ROOT / "huggingface" / repo


def snapshot(repo: str, patterns: tuple[str, ...], refresh: bool = False) -> Path:
    """Mirror the files of `repo` matching `patterns` into `snapshot_dir(repo)`."""
    from huggingface_hub import snapshot_download
    dest = snapshot_dir(repo)
    if dest.is_dir() and any(dest.rglob("*.parquet")) and not refresh:
        return dest
    snapshot_download(repo_id=repo, repo_type="dataset", local_dir=str(dest),
                      allow_patterns=list(patterns))
    return dest
