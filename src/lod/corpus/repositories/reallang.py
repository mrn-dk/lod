"""Raw rows for broad real-language decisions (`real/reallang.py`, row 50).

Rows are stored as each source publishes them under `rl_<name>`; the reader's `SPECS`
record every source's licence and decide what a label means, so this module only knows
where the bytes are. Train splits only, except where a whole dataset is held out for
evaluation and a larger pool helps -- then train plus validation, never a benchmark's
test split.
"""

from __future__ import annotations

import csv
import gzip
import io
import json
from pathlib import Path

from lod.corpus.repositories import hub
from lod.corpus.repositories import raw_store as store

PREFIX = "rl_"
CONVERT = "refs/convert/parquet"   # the Hub's parquet conversion of script-based repos
MAX_ROWS = 60_000                  # a sample, not a mirror: the per-task cap is 2,000


def _parquet(repo: str, filename: str, revision: str | None = None) -> list[dict]:
    return hub.parquet_rows(hub.file(repo, filename, revision))


def _jsonl(repo: str, filename: str) -> list[dict]:
    p = hub.file(repo, filename)
    opener = gzip.open if filename.endswith(".gz") else open
    with opener(p, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _json(repo: str, filename: str) -> list[dict]:
    return json.loads(hub.file(repo, filename).read_text(encoding="utf-8"))


def _csv(repo: str, filename: str) -> list[dict]:
    text = hub.file(repo, filename).read_text(encoding="utf-8")
    return list(csv.DictReader(io.StringIO(text)))


def _socket(task: str, splits: tuple[str, ...] = ("train",)) -> list[dict]:
    """SocKET (Choi et al. 2023): parallel `<split>_text.txt` / `<split>_labels.txt`."""
    def lines(name: str) -> list[str]:
        return hub.file("Blablablab/SOCKET", f"SOCKET_DATA/{task}/{name}").read_text(
            encoding="utf-8").rstrip("\n").split("\n")

    labels = hub.file("Blablablab/SOCKET", f"SOCKET_DATA/{task}/label_list.txt").read_text(
        encoding="utf-8").strip("\n").split("\n")
    out = []
    for split in splits:
        texts, labs = lines(f"{split}_text.txt"), lines(f"{split}_labels.txt")
        if len(texts) != len(labs):
            raise ValueError(f"SocKET {task}/{split}: {len(texts)} texts, {len(labs)} labels")
        for t, y in zip(texts, labs):
            if t.strip() and y.strip():
                out.append({"text": t, "label": y, "label_list": labels, "split": split})
    return out


def _tag(rows: list[dict], split: str) -> list[dict]:
    for r in rows:
        r["_split"] = split
    return rows


FETCHERS = {
    "wanli": lambda: _jsonl("alisawuffles/WANLI", "train.jsonl"),
    "wanli_annotations": lambda: _jsonl("alisawuffles/WANLI", "anonymized_annotations.jsonl"),
    "sick": lambda: _parquet("sick", "default/train/0000.parquet", CONVERT),
    "scitail": lambda: (
        _tag(_parquet("allenai/scitail", "snli_format/train-00000-of-00001.parquet"), "train")
        + _tag(_parquet("allenai/scitail",
                        "snli_format/validation-00000-of-00001.parquet"), "validation")),
    "piqa": lambda: _parquet("ybisk/piqa", "plain_text/train/0000.parquet", CONVERT),
    "siqa": lambda: _parquet("allenai/social_i_qa", "default/train/0000.parquet", CONVERT),
    "cosmosqa": lambda: _tag(
        _parquet("allenai/cosmos_qa", "default/train/0000.parquet", CONVERT), "train"),
    "swag": lambda: _parquet("allenai/swag", "regular/train-00000-of-00001.parquet"),
    "obqa": lambda: _parquet("allenai/openbookqa", "additional/train-00000-of-00001.parquet"),
    "copa": lambda: (_tag(_csv("pkavumba/balanced-copa", "train.csv"), "train")
                     + _tag(_csv("pkavumba/balanced-copa", "test.csv"), "test")),
    "ethics_commonsense": lambda: _csv("hendrycks/ethics", "data/commonsense/train.csv"),
    "ethics_deontology": lambda: _csv("hendrycks/ethics", "data/deontology/train.csv"),
    "ethics_justice": lambda: _csv("hendrycks/ethics", "data/justice/train.csv"),
    "moral_stories": lambda: _jsonl(
        "demelin/moral_stories", "data/classification/action+norm/norm_distance/train.jsonl"),
    "scruples": lambda: _tag(
        _jsonl("metaeval/scruples", "train.scruples-anecdotes.jsonl"), "train"),
    "flute": lambda: _jsonl("ColumbiaNLP/FLUTE", "train.jsonl"),
    "rumoureval": lambda: (
        _tag(_csv("strombergnlp/rumoureval_2019", "rumoureval2019_train.csv"), "train")
        + _tag(_csv("strombergnlp/rumoureval_2019", "rumoureval2019_val.csv"), "validation")),
    "dailydialog": lambda: _parquet("li2017dailydialog/daily_dialog",
                                    "default/train/0000.parquet", CONVERT),
    "multiwoz": lambda: _parquet("pfb30/multi_woz_v22", "v2.2/train/0000.parquet", CONVERT),
    "hwu64": lambda: _parquet("DeepPavlov/hwu64", "data/train-00000-of-00001.parquet"),
    "hwu64_intents": lambda: _parquet("DeepPavlov/hwu64",
                                      "intents/intents-00000-of-00001.parquet"),
    "glaive": lambda: _json("glaiveai/glaive-function-calling-v2",
                            "glaive-function-calling-v2.json"),
    "toolace": lambda: _json("Team-ACE/ToolACE", "data.json"),
    # SocKET subtasks (CC-BY-4.0 aggregate): train files; the held-out ones add val
    "sk_sarc": lambda: _socket("sarc"),
    "sk_is_humor": lambda: _socket("hahackathon#is_humor"),
    "sk_humor_rating": lambda: _socket("hahackathon#humor_rating"),
    "sk_hyperbole": lambda: _socket("hypo-l"),
    "sk_condescension": lambda: _socket("talkdown-pairs"),
    "sk_politeness": lambda: _socket("stanfordpoliteness"),
    "sk_intimacy": lambda: _socket("questionintimacy", ("train", "val")),
    "sk_empathy": lambda: _socket("empathy#empathy_bin", ("train", "val")),
    "sk_person_abuse": lambda: _socket("contextual-abuse#PersonDirectedAbuse"),
    "sk_deception": lambda: _socket("two-to-lie#sender_truth"),
    "sk_complaints": lambda: _socket("complaints", ("train", "val")),
    "sk_brag_achievement": lambda: _socket("bragging#brag_achievement", ("train", "val")),
    "sk_emobank_valence": lambda: _socket("emobank#valence", ("train", "val")),
}


def outputs() -> list[Path]:
    return [store.path_for(PREFIX + name) for name in FETCHERS]


def fetch(only: tuple[str, ...] = tuple(FETCHERS), refresh: bool = False) -> int:
    """Fetch each source not yet stored. A failing source is reported and skipped; the
    reader skips a task whose rows are missing."""
    done = 0
    for name in only:
        key = PREFIX + name
        if store.has(key) and not refresh:
            continue
        try:
            rows = FETCHERS[name]()
        except Exception as e:  # noqa: BLE001
            print(f"  {key}: FAILED {type(e).__name__}: {str(e)[:120]}", flush=True)
            continue
        n = store.save(key, rows[:MAX_ROWS])
        print(f"  {key}: {n:,} rows", flush=True)
        done += 1
    return done
