"""The raw-row store: download once, build the corpus as many times as you like.

Fetching and building have very different costs. A download takes minutes and is bounded
by somebody else's server; turning rows into questions takes milliseconds and is the part
that actually gets iterated on -- option sets, question wording, which column is ordinal,
how a soft target is derived. Tying them together would re-pay the download for every
wording change, and fetch a dataset with seven label columns seven times.

So the fetch stage (`scripts/fetch_data.py`) writes raw rows here and never interprets
them; the loaders read from here and never touch the network.

Format: `<raw root>/store/<key>.jsonl`, one raw row per line, exactly as the source
returned it -- deliberately not a parsed form, so re-interpretation stays free -- plus an
optional `<key>.features.json` sidecar with the published ClassLabel names.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterator

from lod import paths

# Directory names a raw root may use for the rows themselves, newest first. `v4-raw` is
# the name older raw roots use; without it such a root silently resolves to an empty
# store and the build reports "no registered source tasks available".
ROW_DIRS = ("store", "v4-raw")


def _row_dir(root: Path) -> Path:
    return next((root / name for name in ROW_DIRS if (root / name).is_dir()),
                root / ROW_DIRS[0])


RAW_ROOT = paths.RAW_ROOT
RAW = _row_dir(RAW_ROOT)


def configure(root: str | Path) -> Path:
    """Point the store at another raw root (default: `lod.paths.RAW_ROOT`)."""
    global RAW, RAW_ROOT
    RAW_ROOT = Path(root).expanduser()
    RAW = _row_dir(RAW_ROOT)
    clear_cache()
    return RAW


def path_for(key: str) -> Path:
    return RAW / f"{key}.jsonl"


def features_path_for(key: str) -> Path:
    return RAW / f"{key}.features.json"


def has(key: str) -> bool:
    p = path_for(key)
    return p.exists() and p.stat().st_size > 0


def save_features(key: str, names: dict[str, list[str]]) -> int:
    """Record the published option names of each ClassLabel column.

    The store keeps raw rows "exactly as the source returned it", and for a ClassLabel
    column that is an *integer*. The names that integer stands for live in the dataset's
    feature metadata, which streaming rows do not carry -- so without this sidecar a
    100-way label like LexGLUE LEDGAR's clause type reaches the corpus as the option set
    "0".."99", and the model is asked to pick an index with no meaning in it.

    Kept beside the rows rather than mixed into them because it is metadata about the
    column, not data about a row, and because it is fetched separately: builder info is a
    metadata call, so a store that already has its rows can gain names without
    re-downloading anything.
    """
    RAW.mkdir(parents=True, exist_ok=True)
    p = features_path_for(key)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(names, indent=1, default=str))
    tmp.replace(p)
    return len(names)


def load_features(key: str) -> dict[str, list[str]]:
    """Published ClassLabel names per column, or {} when none were recorded."""
    p = features_path_for(key)
    if not p.exists():
        return {}
    try:
        out = json.loads(p.read_text())
    except json.JSONDecodeError:
        return {}
    return {k: [str(x) for x in v] for k, v in out.items() if v}


def save(key: str, rows: list[dict]) -> int:
    """Write rows atomically, so an interrupted fetch cannot leave a half file that
    later reads as a complete one."""
    RAW.mkdir(parents=True, exist_ok=True)
    p = path_for(key)
    tmp = p.with_suffix(".jsonl.tmp")
    with tmp.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
    tmp.replace(p)
    return len(rows)


# Tasks of one source are built consecutively, and each one calls load(). Without a memo
# a dataset with seven label columns re-reads and re-parses its file seven times -- the
# same mistake the fetch/build split fixed for the network, repeated on disk, and it bites
# hardest on the largest sources (one row-1 store is 1.2 GB). A tiny LRU captures almost
# all of it precisely because the access pattern is consecutive.
_MEMO: "OrderedDict[str, list[dict]]" = OrderedDict()
_MEMO_MAX = 2


def _memo_get(key: str) -> list[dict] | None:
    if key in _MEMO:
        _MEMO.move_to_end(key)
        return _MEMO[key]
    return None


def _memo_put(key: str, rows: list[dict]) -> None:
    _MEMO[key] = rows
    _MEMO.move_to_end(key)
    while len(_MEMO) > _MEMO_MAX:
        _MEMO.popitem(last=False)


def clear_cache() -> None:
    _MEMO.clear()


def load(key: str, limit: int | None = None) -> list[dict]:
    cached = _memo_get(key)
    if cached is not None:
        return cached if limit is None else cached[:limit]

    p = path_for(key)
    if not p.exists():
        raise FileNotFoundError(
            f"no raw rows for {key!r} in {RAW}; run scripts/fetch_data.py first "
            f"(fetching and building are separate stages)")
    out = []
    with p.open() as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    if limit is None:
        _memo_put(key, out)
    return out


def stream(key: str) -> Iterator[dict]:
    p = path_for(key)
    if not p.exists():
        return
    with p.open() as f:
        for line in f:
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def inventory() -> dict[str, dict[str, Any]]:
    """What has been fetched, and how big. The answer to "can I build offline yet?"."""
    if not RAW.exists():
        return {}
    out = {}
    for p in sorted(RAW.glob("*.jsonl")):
        out[p.stem] = {"rows": _count_lines(p), "mb": round(p.stat().st_size / 1e6, 1)}
    return out


def _count_lines(p: Path) -> int:
    n = 0
    with p.open("rb") as f:
        while chunk := f.read(1 << 22):
            n += chunk.count(b"\n")
    return n


def total_mb() -> float:
    return round(sum(v["mb"] for v in inventory().values()), 1)
