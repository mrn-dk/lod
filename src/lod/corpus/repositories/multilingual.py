"""Raw rows for multilingual classification (`real/multilingual.py`, row 46).

Only the (language, split) files the reader plans to use are downloaded -- one small file
each, about 120 MB in total. Every row is stored as the file holds it plus `_lang` and
`_split` (the file's own language code and split), because the store keeps rows, not the
file they came from. MASSIVE intent and scenario are two mirrors of the same utterances
and stay two keys, so they can be checked against each other instead of joined on trust.

Decision Index exclusion. The build's row-level contamination filter tokenises with
`[a-z0-9]+`, so an Arabic, Greek or Japanese state has no tokens and can never match. The
DI members whose text is near this row -- iSarcasmEval (Arabic tweets that reuse the
SemEval-2017 tweets behind Tweet Sentiment Multilingual) and ESCI's Spanish and Japanese
queries -- are re-matched here with Unicode word tokens, and the hashes of overlapping
states are written to `ml46_di_exclude.json` in the store, which the reader drops.
"""

from __future__ import annotations

import csv
import gzip
import json
import re
from pathlib import Path

from lod.corpus.repositories import hub
from lod.corpus.repositories import raw_store as store
from lod.corpus.services.sources.real import multilingual as ml

SHINGLE = 8


def _read(path: Path) -> list[dict]:
    name = path.name
    if name.endswith(".json.gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    if name.endswith(".jsonl"):
        with open(path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    if name.endswith(".tsv"):
        # QUOTE_NONE: tweets carry bare double quotes that are text, not field quoting
        with open(path, encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))
    if name.endswith(".parquet"):
        import pandas as pd
        return pd.read_parquet(path).to_dict("records")
    raise ValueError(f"unknown file type: {path}")


def outputs() -> list[Path]:
    return [store.path_for(src.store_key) for src in ml.SOURCES]


def fetch_source(src: "ml.Source", refresh: bool = False) -> int:
    if store.has(src.store_key) and not refresh:
        return 0
    rows: list[dict] = []
    for lang, split in src.files_needed():
        code = src.codes[lang]
        filename = src.file_pattern.format(code=code, split=split)
        got = _read(hub.file(src.hf, filename))
        for r in got:
            r = {k: (v.item() if hasattr(v, "item") else v) for k, v in r.items()}
            r["_lang"] = code
            r["_split"] = split
            rows.append(r)
    n = store.save(src.store_key, rows)
    if src.label_names:
        store.save_features(src.store_key, {src.label_col: list(src.label_names)})
    print(f"  {src.store_key}: {n:,} rows "
          f"({store.path_for(src.store_key).stat().st_size / 1e6:.1f} MB)", flush=True)
    return n


# ---- Decision Index re-match ----------------------------------------------------------

def _utoks(text: str) -> list[str]:
    return re.findall(r"\w+", str(text).lower())


def _di_items(di_repos: Path) -> list[tuple[str, str]]:
    """(family, text) for every DI item near this row, from the suite's upstream repos."""
    out: list[tuple[str, str]] = []
    isar = di_repos / "isarcasm"
    for f in sorted(list((isar / "test").glob("*.csv")) + list((isar / "train").glob("*.csv"))):
        with open(f, encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh):
                for col in ("text", "tweet", "rephrase", "text_0", "text_1"):
                    if r.get(col):
                        out.append((f"iSarcasmEval/{f.name}", r[col]))
    esci = di_repos / "esci/shopping_queries_dataset/shopping_queries_dataset_examples.parquet"
    if esci.exists():
        import pandas as pd
        d = pd.read_parquet(esci, columns=["query", "product_locale"])
        out += [("ESCI es/jp query", q)
                for q in d.loc[d.product_locale != "us", "query"].drop_duplicates()]
    return out


def di_exclusions(di_repos: Path) -> dict[str, str]:
    """state hash -> the DI item family it matches.

    A state matches when a text field equals an item outright, or shares at least three
    eight-word shingles with one item -- the build filter's own thresholds.
    """
    items = _di_items(di_repos)
    if not items:
        return {}
    whole: dict[str, str] = {}
    shingles: dict[str, set[int]] = {}
    for i, (name, text) in enumerate(items):
        t = _utoks(text)
        if len(t) >= 3:
            whole[" ".join(t)] = name
        for j in range(len(t) - SHINGLE + 1):
            shingles.setdefault(" ".join(t[j:j + SHINGLE]), set()).add(i)
    out: dict[str, str] = {}
    for src in ml.SOURCES:
        if not store.has(src.store_key):
            continue
        for r in ml.rows_of(src):
            state = ml.state_of(src, r)
            hit = None
            for part in (r.get(c) for c in src.text_cols):
                t = _utoks(part)
                if len(t) >= 3 and " ".join(t) in whole:
                    hit = whole[" ".join(t)]
                    break
                counts: dict[int, int] = {}
                for j in range(len(t) - SHINGLE + 1):
                    for i in shingles.get(" ".join(t[j:j + SHINGLE]), ()):
                        counts[i] = counts.get(i, 0) + 1
                if counts and max(counts.values()) >= 3:
                    hit = items[max(counts, key=counts.get)][0]
                    break
            if hit and state:
                out[ml.state_hash(state)] = f"{src.store_key}: {hit}"
    return out


def write_exclusions(di_repos: Path | None) -> int | None:
    """Rewrite the exclusion list from `di_repos`. None when the repos are unavailable,
    in which case an existing list is kept rather than replaced by an empty one."""
    path = store.RAW / ml.DI_EXCLUDE
    if di_repos is None or not Path(di_repos).is_dir():
        kept = "keeping the existing list" if path.exists() else "no list written"
        print(f"  Decision Index repos not given or missing ({di_repos}); {kept}",
              flush=True)
        return None
    excl = di_exclusions(Path(di_repos))
    path.write_text(json.dumps(excl, indent=1, sort_keys=True))
    print(f"  Decision Index (Unicode re-match): {len(excl)} states excluded -> {path}",
          flush=True)
    return len(excl)


def fetch(refresh: bool = False, di_repos: Path | None = None) -> int:
    n = sum(fetch_source(src, refresh) for src in ml.SOURCES)
    write_exclusions(di_repos)
    return n
