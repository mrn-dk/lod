"""Drop any example whose state carries an item of the Decision Index benchmark.

The Decision Index suite is evaluation data ("do not train on it"). Whole datasets that
ARE suite members are removed at the source table; this is the row-level net under that,
for text that reaches us through another dataset. Measured on an earlier build by 8-word
overlap against the suite's own upstream files: Natural Instructions task043/task047
carried ARC test questions (17/1,742 and 10/251), JudgeBench an MMLU-Pro item, SWAG
HellaSwag's ActivityNet contexts, bb_physics 3 MMLU items.

The blocklist is a sorted array of 64-bit hashes -- of every 8-word shingle of every
long suite item, and of every whole short item -- built by `build_blocklist` from a
rebuilt Decision Index suite (`scripts/build_corpus.py blocklist`). Hashes, not text:
the list never redistributes the suite.

A state is dropped when it covers at least half of ONE suite item's shingles (and at
least 3 of them), or when the whole state -- or a line of 6+ words -- equals a short
suite item word for word. Counting raw hits across items was measured and rejected: it
dropped 505 of 572 congressional-bill states for legal boilerplate spread over many
MMLU law questions, while no single question was actually present.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from lod.paths import DI_BLOCKLIST

N = 8
MIN_LONG = 12          # items this long are matched by shingles
MIN_SHORT = 3          # shorter ones (CLINC utterances, ANLI hypotheses) by exact match
MIN_HITS = 3
MIN_COVER = 0.5
MIN_LINE = 6           # a line shorter than this matching a short item is coincidence
DEFAULT_PATH = DI_BLOCKLIST

# Unicode-aware: an ASCII-only tokenizer sees no token at all in Arabic, Greek or CJK
# text, so a non-Latin suite item could never match. Scripts
# written without spaces (CJK, kana, Thai) are split per character.
_NOSPACE = "\u0e00-\u0e7f\u3040-\u30ff\u3400-\u9fff"
WORD = re.compile(f"[{_NOSPACE}]|[^\\W_{_NOSPACE}]+")


def toks(text: str) -> list[str]:
    return WORD.findall(str(text).lower())


def h64(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(),
                          "big", signed=True)


def item_hashes(text: str) -> tuple[list[int], list[int]]:
    """-> (shingle hashes, short-item hashes) contributed by one suite item."""
    t = toks(text)
    if len(t) >= MIN_LONG:
        return [h64(" ".join(t[i:i + N])) for i in range(len(t) - N + 1)], []
    if len(t) >= MIN_SHORT:
        return [], [h64(" ".join(t))]
    return [], []


class Blocklist:
    def __init__(self, long: np.ndarray, item: np.ndarray, size: np.ndarray,
                 short: np.ndarray):
        order = np.argsort(long, kind="stable")
        self.long, self.item = long[order].astype(np.int64), item[order].astype(np.int64)
        self.size = size.astype(np.int64)            # shingles per item
        self.short = np.unique(short.astype(np.int64))

    @classmethod
    def load(cls, path: Path | str | None = None) -> "Blocklist | None":
        p = Path(path) if path else DEFAULT_PATH
        if not p.exists():
            return None
        z = np.load(p)
        return cls(z["long"], z["item"], z["size"], z["short"])

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, long=self.long, item=self.item, size=self.size,
                            short=self.short)
        return path

    def __len__(self) -> int:
        return int(self.size.size)

    def best_cover(self, t: list[str]) -> float:
        """Largest share of any single suite item's shingles found in these tokens."""
        if len(t) < N or self.long.size == 0:
            return 0.0
        q = np.unique(np.asarray([h64(" ".join(t[i:i + N]))
                                  for i in range(len(t) - N + 1)], dtype=np.int64))
        lo = np.searchsorted(self.long, q, "left")
        hi = np.searchsorted(self.long, q, "right")
        items = [self.item[a:b] for a, b in zip(lo, hi) if b > a]
        if not items:
            return 0.0
        ids, counts = np.unique(np.concatenate(items), return_counts=True)
        ok = counts >= MIN_HITS
        if not ok.any():
            return 0.0
        return float((counts[ok] / self.size[ids[ok]]).max())

    def short_match(self, state: str) -> bool:
        t = toks(state)
        cands = {" ".join(t)} if len(t) >= MIN_SHORT else set()
        for line in str(state).split("\n"):
            lt = toks(line)
            if len(lt) >= MIN_LINE:
                cands.add(" ".join(lt))
        if not cands or self.short.size == 0:
            return False
        q = np.asarray([h64(c) for c in cands], dtype=np.int64)
        i = np.searchsorted(self.short, q).clip(0, self.short.size - 1)
        return bool((self.short[i] == q).any())

    def blocked(self, state) -> bool:
        s = state if isinstance(state, str) else str(state)
        return self.best_cover(toks(s)) >= MIN_COVER or self.short_match(s)

    def filter(self, examples: list) -> tuple[list, int]:
        kept = [e for e in examples if not self.blocked(e.state)]
        return kept, len(examples) - len(kept)

    def filter_lines(self, lines: Iterable[str], dropped: Counter) -> Iterator[str]:
        """The JSONL records whose state is not blocked, verbatim; `dropped` counts the
        rest by task. For re-filtering a built corpus when the blocklist grows, without
        a rebuild from source."""
        for line in lines:
            r = json.loads(line)
            if self.blocked(r["state"]):
                dropped[r["task"]] += 1
                continue
            yield line


def build_blocklist(items: Iterable[tuple[str, str]]) -> tuple[Blocklist, Counter]:
    """Hash (benchmark, text) items into a Blocklist. -> (blocklist, items per benchmark)."""
    long, item, size, short, per = [], [], [], [], Counter()
    for bench, text in items:
        lo, sh = item_hashes(text)
        lo = sorted(set(lo))
        if lo:
            long += lo
            item += [len(size)] * len(lo)
            size.append(len(lo))
        short += sh
        per[bench] += 1
    return Blocklist(np.asarray(long, dtype=np.int64), np.asarray(item, dtype=np.int64),
                     np.asarray(size, dtype=np.int64),
                     np.asarray(short, dtype=np.int64)), per


# ---- what goes into the blocklist ------------------------------------------------------

def _text(x) -> str:
    return x if isinstance(x, str) else json.dumps(x, ensure_ascii=False)


def suite_items(suite: Path) -> Iterator[tuple[str, str]]:
    """-> (DI benchmark, text) for every state and instruction of a rebuilt suite
    (`selected-rows.jsonl.gz` + `added-rows.jsonl.gz`, hash-verified rows)."""
    from lod.serving.api import render_state

    for name in ("selected-rows.jsonl.gz", "added-rows.jsonl.gz"):
        with gzip.open(Path(suite) / name, "rt", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                fam = r.get("_evaluation", {}).get("dataset") or r.get("family") or "?"
                st = r.get("state")
                if st:
                    yield fam, render_state(st) if not isinstance(st, str) else st
                for q in (r.get("questions") or {}).values():
                    if q.get("instructions"):
                        yield fam, _text(q["instructions"])


def upstream_items(root: Path) -> Iterator[tuple[str, str]]:
    """-> (DI benchmark, text) from the suite's upstream source files, as a Decision
    Index build workspace (`root`) fetched them.

    Wider than the suite's own rows: whole test sets (MMLU, ARC, GSM8K, ANLI, CLINC150,
    ...) rather than the items the suite sampled, so a corpus cannot carry a sibling of
    a suite item either. What is not on disk cannot be blocked row by row; those suite
    members are removed by name at the source table instead. Needs pandas.
    """
    import io

    import pandas as pd

    root = Path(root)
    s, r = root / "data" / "sources", root / "raw"
    for q in pd.read_parquet(s / "mmlu/all/test-00000-of-00001.parquet").question:
        yield "MMLU", q
    for sub in ("ARC-Challenge", "ARC-Easy"):
        for q in pd.read_parquet(s / f"arc/{sub}/test-00000-of-00001.parquet").question:
            yield "ARC", q
    for q in pd.read_parquet(s / "gsm8k/main/test-00000-of-00001.parquet").question:
        yield "GSM8K", q
    h = pd.read_parquet(s / "hellaswag/data/validation-00000-of-00001.parquet")
    for q in h.ctx:
        yield "HellaSwag", q
    for q in pd.read_parquet(
            s / "winogrande/winogrande_xl/validation-00000-of-00001.parquet").sentence:
        yield "WinoGrande", q
    for rr in ("r1", "r2", "r3"):
        a = pd.read_parquet(r / f"anli/plain_text/test_{rr}-00000-of-00001.parquet")
        for col in ("premise", "hypothesis"):
            for q in a[col]:
                yield "ANLI", q
    cl = json.loads((r / "clinc150/data_full.json").read_text())
    for k in ("test", "oos_test"):
        for t, _ in cl[k]:
            yield "CLINC", t
    repos = root / "artifacts" / "benchmark-suite" / "raw" / "repos"
    # non-English suite members: iSarcasmEval (Arabic and English; train files too, since
    # its Arabic half is ArSarcasm-v2, which other Arabic tweet sets re-use) and the ESCI
    # test queries
    for f in sorted((repos / "isarcasm").rglob("*.csv")):
        df = pd.read_csv(f)
        for col in ("tweet", "text", "Tweet", "sentence", "rephrase", "text_0", "text_1"):
            if col in df.columns:
                for q in df[col].dropna():
                    yield "iSarcasmEval", q
    esci = repos / "esci" / "shopping_queries_dataset" / "shopping_queries_dataset_examples.parquet"
    if esci.exists():
        ex = pd.read_parquet(esci, columns=["query", "split"])
        for q in ex.loc[ex["split"] == "test", "query"].drop_duplicates():
            yield "ESCI", q
    # SATA-Bench is assembled from other datasets (MultiRC, Reuters-21578, toxicity,
    # PubMed, EURLEX, company news); each item is markup + paragraph + question. Hash each
    # sentence-sized chunk on its own, so a state carrying the source paragraph matches.
    sata = repos / "sata" / "src" / "satabench" / "evaluation" / "dataset" / "final_A_multic_choice.csv"
    if sata.exists():
        for t in pd.read_csv(sata)["text"].dropna():
            t = re.sub(r"<[^>]+>", " ", str(t))
            for chunk in re.split(r"Sent \d+:|\n+|Multi Label Question:|Paragraph:", t):
                if len(chunk.split()) >= 12:
                    yield "SATA-Bench", chunk
    sata_json = repos / "sata" / "src" / "satabench" / "methods" / "data" / "sata_bench_final_2025.json"
    if sata_json.exists():
        for line in open(sata_json, encoding="utf-8"):
            if line.strip():
                yield "SATA-Bench", json.loads(line).get("paragraph", "")
    raw = root / "artifacts" / "benchmark-suite" / "raw"
    for f in sorted((raw / "vast" / "data" / "VAST").glob("*.csv")):
        for q in pd.read_csv(f).get("post", pd.Series(dtype=str)).dropna():
            yield "VAST", q
    fe = repos / "finentity" / "data" / "FinEntity.json"
    if fe.exists():
        for rec in json.loads(fe.read_text()):
            yield "FinEntity", rec.get("content", "")
    for f in sorted((repos / "acos" / "data").rglob("*.tsv")):
        for line in open(f, encoding="utf-8"):
            yield "ACOS", line.split("\t")[0]
    hz = root / "raw" / "downloads" / "humicroedit-full.zip"
    if hz.exists():
        with zipfile.ZipFile(hz) as z:
            for n in z.namelist():
                if n.endswith(".csv") and "subtask-1" in n:
                    for q in pd.read_csv(io.BytesIO(z.read(n))).get(
                            "original", pd.Series(dtype=str)).dropna():
                        yield "Humicroedit", re.sub(r"<(.*?)/>", r"\1", q)
    gz = s / "gpqa/dataset.zip"
    if gz.exists():
        try:
            with zipfile.ZipFile(gz) as z:
                for name in z.namelist():
                    if name.endswith(".csv") and "main" in name:
                        # the archive is password-protected upstream; skip if so
                        df = pd.read_csv(z.open(name, pwd=b"deserted-untie-orchid"))
                        for q in df.get("Question", []):
                            yield "GPQA", q
        except Exception as exc:                                  # noqa: BLE001
            print(f"  GPQA skipped: {type(exc).__name__}", file=sys.stderr)


# ---- the overlap report ----------------------------------------------------------------

class SuiteIndex:
    """The same matching rules as `Blocklist`, but remembering which DI benchmark each
    item came from, so an overlap can be reported by benchmark rather than just caught."""

    def __init__(self, items: Iterable[tuple[str, str]]):
        longs, owners, sizes, shorts, fams = [], [], [], {}, []
        for fam, text in items:
            t = toks(text)
            if len(t) >= MIN_LONG:
                hs = {h64(" ".join(t[i:i + N])) for i in range(len(t) - N + 1)}
                iid = len(sizes)
                sizes.append(len(hs))
                fams.append(fam)
                longs.extend(hs)
                owners.extend([iid] * len(hs))
            elif len(t) >= MIN_SHORT:
                shorts.setdefault(h64(" ".join(t)), fam)
        order = np.argsort(np.asarray(longs, dtype=np.int64), kind="stable")
        self.long = np.asarray(longs, dtype=np.int64)[order]
        self.item = np.asarray(owners, dtype=np.int64)[order]
        self.size = np.asarray(sizes, dtype=np.int64)
        self.fam = fams
        self.short = shorts

    def describe(self) -> str:
        return (f"{self.size.size:,} long items, {self.long.size:,} shingles, "
                f"{len(self.short):,} short items")

    def hits(self, text: str) -> set[str]:
        """DI benchmarks whose items this text covers."""
        out: set[str] = set()
        t = toks(text)
        if len(t) >= N and self.long.size:
            q = np.unique(np.asarray([h64(" ".join(t[i:i + N]))
                                      for i in range(len(t) - N + 1)], dtype=np.int64))
            lo = np.searchsorted(self.long, q, "left")
            hi = np.searchsorted(self.long, q, "right")
            parts = [self.item[a:b] for a, b in zip(lo, hi) if b > a]
            if parts:
                ids, counts = np.unique(np.concatenate(parts), return_counts=True)
                ok = (counts >= MIN_HITS) & (counts / self.size[ids] >= MIN_COVER)
                out |= {self.fam[i] for i in ids[ok]}
        cands = {" ".join(t)} if len(t) >= MIN_SHORT else set()
        for line in text.split("\n"):
            lt = toks(line)
            if len(lt) >= MIN_LINE:
                cands.add(" ".join(lt))
        for c in cands:
            f = self.short.get(h64(c))
            if f:
                out.add(f)
        return out


def overlap_report(index: SuiteIndex, records: Iterable[dict]) -> dict:
    """Which DI benchmarks a split's records touch (state or question text), and through
    which tasks -- so a Decision Index score can be read with its contamination beside it."""
    by_fam: Counter = Counter()
    by_task: Counter = Counter()
    pairs: Counter = Counter()
    n = hit = 0
    for r in records:
        n += 1
        texts = [r["state"]] + [_text(q.get("question", "")) for q in r["questions"]]
        fams: set[str] = set()
        for t in texts:
            fams |= index.hits(t if isinstance(t, str) else _text(t))
        if fams:
            hit += 1
            by_task[r["task"]] += 1
            for fm in fams:
                by_fam[fm] += 1
                pairs[(fm, r["task"])] += 1
    return {"records": n, "overlapping": hit,
            "by_di_benchmark": dict(by_fam.most_common()),
            "by_task": dict(by_task.most_common(40)),
            "pairs": [[f, t, c] for (f, t), c in pairs.most_common(60)]}
