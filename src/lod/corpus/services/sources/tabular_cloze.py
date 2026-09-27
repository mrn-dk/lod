"""tabular_cloze — masked categorical cells from OpenML tables.

Ground truth is structural: the value really is in the row, and the other columns
are the evidence. Each (dataset, categorical column) is its own schema, which is
what makes this source scale into the thousands.

Source: https://www.openml.org/ — per-dataset licence, filtered to open ones (tier 1).
"""

from __future__ import annotations

import io
import random

from lod.schema import Example, Question
from lod.corpus.services.sources.base import (
    SourceInfo,
    SourceResult,
    fetch,
    fetch_json,
    schema_key,
)

INFO = SourceInfo(
    name="tabular_cloze",
    url="https://www.openml.org/api/v1/json/data/list",
    licence="per-dataset (Public/CC0/CC-BY only)",
    tier=1,
    ground_truth_type="structural",
    state_format="tabular",
    option_type="fixed",
    notes="one row rendered as col: value with a categorical column masked",
)
OPEN_LICENCES = ("public", "cc0", "cc-0", "cc by", "cc-by", "creativecommons", "public domain")
MAX_OPTIONS = 30


def _as_list(x) -> list:
    """OpenML collapses a one-element array to the bare object in its JSON, so
    `feature` and `nominal_value` come back as a dict / a string for some
    datasets. Iterating those yields keys and characters respectively, which is
    how this source died with "'str' object has no attribute 'get'"."""
    if x is None:
        return []
    return x if isinstance(x, list) else [x]


def _is_open(lic: str) -> bool:
    low = (lic or "").strip().lower()
    return bool(low) and any(tok in low for tok in OPEN_LICENCES)


def _render(row: dict, cols: list[str], hide: str, limit: int = 14) -> str:
    out = []
    for c in cols:
        if c == hide or len(out) >= limit:
            continue
        v = row.get(c)
        if v is None:
            continue
        if isinstance(v, float):
            v = f"{v:.4g}"
        out.append(f"{c}: {v}")
    return "\n".join(out)


def generate(n: int, seed: int = 0, n_datasets: int = 120, per_schema: int = 12,
             list_limit: int = 900) -> SourceResult:
    import pyarrow.parquet as pq

    rng = random.Random(seed)
    res = SourceResult(info=INFO)
    listing = fetch_json(f"https://www.openml.org/api/v1/json/data/list/limit/{list_limit}")
    dids = [d["did"] for d in listing["data"]["dataset"] if d.get("status") == "active"]
    rng.shuffle(dids)

    used = 0
    for did in dids:
        if len(res.examples) >= n or used >= n_datasets:
            break
        try:
            desc = fetch_json(f"https://www.openml.org/api/v1/json/data/{did}")["data_set_description"]
            if not _is_open(desc.get("licence", "")) or not desc.get("parquet_url"):
                continue
            feats = fetch_json(f"https://www.openml.org/api/v1/json/data/features/{did}")
            feats = [f for f in _as_list(feats["data_features"]["feature"])
                     if isinstance(f, dict)]
        except Exception:
            continue
        cats = [f for f in feats
                if f.get("data_type") == "nominal"
                and 2 <= len(_as_list(f.get("nominal_value"))) <= MAX_OPTIONS]
        if len(cats) < 2:  # >= 2 categorical columns per table
            continue
        try:
            table = pq.read_table(io.BytesIO(fetch(desc["parquet_url"], timeout=90)))
        except Exception:
            continue
        cols = [str(c) for c in table.column_names]
        rows = table.slice(0, 400).to_pylist()
        if not rows:
            continue
        used += 1
        name = str(desc.get("name", did))[:34]
        for f in cats[:3]:
            col = f["name"]
            if col not in cols or len(res.examples) >= n:
                continue
            options = sorted(str(v) for v in _as_list(f["nominal_value"]))
            task = f"tab_{name}_{col}"[:60]
            picked = [r for r in rng.sample(rows, min(len(rows), per_schema * 3))
                      if r.get(col) is not None and str(r[col]) in options]
            for r in picked[:per_schema]:
                state = _render(r, cols, col)
                if state.count("\n") < 2:
                    continue
                q = Question(col[:24], f"What is the value of '{col}'?", options,
                             options.index(str(r[col])))
                res.examples.append(Example(state=state, questions=[q], task=task))
                res.schemas.add(schema_key(q.question, q.options))
                if len(res.examples) >= n:
                    break
    return res
