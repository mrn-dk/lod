"""Raw rows for English topic, intent and question type (`real/topic.py`, row 45).

Train splits only, and a deterministic sample of each, never a mirror. Large or
class-sorted parquet files are read row group by row group over `HfFileSystem`, so a file
sorted by label costs only the groups read and still yields every class:

| key suffix    | source                                            | kept                              |
|---------------|---------------------------------------------------|-----------------------------------|
| ag_news       | fancyzhx/ag_news train                            | 8,000 rows, hash-ordered          |
| dbpedia_14    | fancyzhx/dbpedia_14 train (sorted by class)       | every 10th row group, 700/class   |
| yahoo_answers | community-datasets/yahoo_answers_topics train     | 12 row groups over both shards   |
| trec          | CogComp/trec train (Hub parquet conversion)       | all 5,452 rows                    |
| mtop_en       | MTOP release zip, dl.fbaipublicfiles.com (CC BY-SA 4.0) | en/train.txt               |
| cfpb          | BEE-spoke-data/consumer-finance-complaints has-text (CC0) | 2018-2022, <= 1,500/product |

Store keys are `v14_topic_<suffix>`, with a `.features.json` sidecar where the parquet
publishes ClassLabel names.
"""

from __future__ import annotations

import hashlib
import io
import zipfile
from collections import defaultdict
from pathlib import Path

from lod.corpus.repositories import hub, web
from lod.corpus.repositories import raw_store as store

PREFIX = "v14_topic_"
MTOP_URL = "https://dl.fbaipublicfiles.com/mtop/mtop.zip"
CFPB_YEARS = ("2018", "2019", "2020", "2021", "2022")   # one product taxonomy throughout
CFPB_PER_PRODUCT = 1500


def _h(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _groups(pf, idx: list[int], columns: list[str] | None = None) -> list[dict]:
    out: list[dict] = []
    for g in idx:
        out += pf.read_row_group(g, columns=columns).to_pylist()
    return out


def fetch_ag_news() -> tuple[list[dict], dict]:
    pf = hub.open_parquet("datasets/fancyzhx/ag_news/data/train-00000-of-00001.parquet")
    rows = pf.read().to_pylist()
    rows.sort(key=lambda r: _h(r["text"]))
    return rows[:8000], hub.parquet_classlabels(pf)


def fetch_dbpedia() -> tuple[list[dict], dict]:
    pf = hub.open_parquet(
        "datasets/fancyzhx/dbpedia_14/dbpedia_14/train-00000-of-00001.parquet")
    rows = _groups(pf, list(range(0, pf.metadata.num_row_groups, 10)))
    by: dict[int, list[dict]] = defaultdict(list)
    for r in sorted(rows, key=lambda r: _h(r["content"])):
        by[r["label"]].append(r)
    return [r for lab in sorted(by) for r in by[lab][:700]], hub.parquet_classlabels(pf)


def fetch_yahoo() -> tuple[list[dict], dict]:
    rows: list[dict] = []
    names: dict = {}
    for shard in ("00000", "00001"):
        pf = hub.open_parquet(
            "datasets/community-datasets/yahoo_answers_topics/yahoo_answers_topics/"
            f"train-{shard}-of-00002.parquet")
        names = names or hub.parquet_classlabels(pf)
        n = pf.metadata.num_row_groups
        rows += _groups(pf, [int(n * k / 6) for k in range(6)])
    return rows, names


def fetch_trec() -> tuple[list[dict], dict]:
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()
    base = "datasets/CogComp/trec@refs%2Fconvert%2Fparquet/default"
    files = sorted(p for p in fs.ls(f"{base}/train", detail=False) if p.endswith(".parquet"))
    rows: list[dict] = []
    names: dict = {}
    for p in files:
        pf = hub.open_parquet(p)
        names = names or hub.parquet_classlabels(pf)
        rows += pf.read().to_pylist()
    return rows, names


def fetch_mtop() -> tuple[list[dict], dict]:
    blob = web.get(MTOP_URL, timeout=300)
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        licence = z.read("mtop/LICENSE.txt").decode("utf-8", "replace")
        if "Attribution-ShareAlike 4.0" not in licence:
            raise RuntimeError("MTOP's licence file changed; re-check it before use")
        text = z.read("mtop/en/train.txt").decode("utf-8")
    zip_sha1 = hashlib.sha1(blob).hexdigest()[:12]
    rows = []
    for line in text.splitlines():
        parts = line.split("\t")
        if len(parts) < 6:
            continue
        rows.append({"id": parts[0], "intent": parts[1], "utterance": parts[3],
                     "domain": parts[4], "locale": parts[5], "zip_sha1": zip_sha1})
    return rows, {}


def fetch_cfpb() -> tuple[list[dict], dict]:
    cols = ["Complaint ID", "Date received", "Product", "Sub-product",
            "Consumer complaint narrative"]
    kept: dict[str, list[dict]] = defaultdict(list)
    for shard in range(3):
        pf = hub.open_parquet("datasets/BEE-spoke-data/consumer-finance-complaints/"
                              f"has-text/train-0000{shard}-of-00003.parquet")
        n = pf.metadata.num_row_groups
        for r in _groups(pf, [int(n * k / 16) for k in range(16)], cols):
            if str(r.get("Date received") or "")[:4] in CFPB_YEARS and r.get("Product"):
                kept[r["Product"]].append(r)
    out = []
    for prod in sorted(kept):
        group = sorted(kept[prod], key=lambda r: _h(str(r["Complaint ID"])))
        out += group[:CFPB_PER_PRODUCT]
    return out, {}


FETCHERS = {"ag_news": fetch_ag_news, "dbpedia_14": fetch_dbpedia,
            "yahoo_answers": fetch_yahoo, "trec": fetch_trec, "mtop_en": fetch_mtop,
            "cfpb": fetch_cfpb}


def outputs() -> list[Path]:
    return [store.path_for(PREFIX + name) for name in FETCHERS]


def fetch(only: tuple[str, ...] = tuple(FETCHERS), refresh: bool = False) -> int:
    done = 0
    for name in only:
        key = PREFIX + name
        if store.has(key) and not refresh:
            continue
        rows, names = FETCHERS[name]()
        store.save(key, rows)
        if names:
            store.save_features(key, names)
        print(f"  {key}: {len(rows):,} rows, "
              f"{store.path_for(key).stat().st_size / 1e6:.1f} MB", flush=True)
        done += 1
    return done
