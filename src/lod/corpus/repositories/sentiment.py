"""Raw rows for English sentiment, emotion and stance (`real/sentiment.py`, row 44).

Train splits only, and samples rather than mirrors: every source is under 25 MB stored.
Rows are kept as published; the reader decides what a label means. Store keys are
`v14s_<name>`, plus a `.features.json` sidecar where the source publishes ClassLabel names.

  sst            Stanford Sentiment Treebank v1.0 with its raw per-annotator scores
                 (nlp.stanford.edu/~socherr/), train sentences only
  fpb            Financial PhraseBank v1.0 (CC BY-NC-SA 3.0), each sentence tagged with
                 the highest annotator-agreement subset it appears in
  amazon         fancyzhx/amazon_polarity (Apache-2.0), leading row groups of shard 0
  imdb           stanfordnlp/imdb plain_text train
  tweeteval_*    cardiffnlp/tweet_eval train: sentiment, offensive, emotion, irony, stance_*
  claim_stance   ibm-research/claim_stance train.csv (CC BY 3.0)
  poem           google-research-datasets/poem_sentiment train (CC BY 4.0)
  tfns           zeroshot/twitter-financial-news-sentiment sent_train.csv (MIT)
"""

from __future__ import annotations

import csv
import html
import io
import zipfile
from pathlib import Path

from lod.corpus.repositories import hub, web
from lod.corpus.repositories import raw_store as store

PREFIX = "v14s_"
SST_MAIN = "https://nlp.stanford.edu/~socherr/stanfordSentimentTreebank.zip"
SST_RAW = "https://nlp.stanford.edu/~socherr/stanfordSentimentTreebankRaw.zip"
TWEETEVAL_CONFIGS = ("sentiment", "offensive", "emotion", "irony", "stance_abortion",
                     "stance_atheism", "stance_climate", "stance_feminist", "stance_hillary")
AMAZON_ROWS = 60_000


# ---- SST ----------------------------------------------------------------------------

def sst_norm(phrase: str) -> str:
    """The key both SST archives agree on.

    The raw-score file writes some symbols as HTML entities, and the main release carries
    PTB brackets and a latin-1 round trip; lower-cased, whitespace-joined tokens after
    undoing both is what matches.
    """
    s = html.unescape(phrase)
    for a, b in (("-LRB-", "("), ("-RRB-", ")"), ("\\/", "/")):
        s = s.replace(a, b)
    try:
        s = s.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    return " ".join(s.lower().split())


def fetch_sst() -> list[dict]:
    main = zipfile.ZipFile(io.BytesIO(web.get(SST_MAIN, timeout=300)))
    raw = zipfile.ZipFile(io.BytesIO(web.get(SST_RAW, timeout=300)))

    def rd(z: zipfile.ZipFile, name: str) -> str:
        return z.read(name).decode("utf-8", errors="replace")

    root = "stanfordSentimentTreebank/"
    sentences = {}
    for line in rd(main, root + "datasetSentences.txt").splitlines()[1:]:
        idx, text = line.split("\t", 1)
        sentences[int(idx)] = text
    split = {}
    for line in rd(main, root + "datasetSplit.txt").splitlines()[1:]:
        idx, lab = line.split(",")
        split[int(idx)] = {"1": "train", "2": "test", "3": "dev"}[lab.strip()]
    phrase_id = {}
    for line in rd(main, root + "dictionary.txt").splitlines():
        text, pid = line.rsplit("|", 1)
        phrase_id.setdefault(sst_norm(text), int(pid))
    value = {}
    for line in rd(main, root + "sentiment_labels.txt").splitlines()[1:]:
        pid, v = line.split("|")
        value[int(pid)] = float(v)
    rroot = "stanfordSentimentTreebankRaw/"
    lex = {}
    for line in rd(raw, rroot + "sentlex_exp12.txt").splitlines():
        idx, text = line.split(",", 1)
        lex.setdefault(sst_norm(text), int(idx))
    scores = {}
    for line in rd(raw, rroot + "rawscores_exp12.txt").splitlines():
        parts = line.strip().split(",")
        if len(parts) >= 2:
            scores[int(parts[0])] = [int(x) for x in parts[1:] if x.strip()]
    out = []
    for idx, text in sorted(sentences.items()):
        if split.get(idx) != "train":
            continue
        key = sst_norm(text)
        pid = phrase_id.get(key)
        li = lex.get(key)
        out.append({"sentence_index": idx, "sentence": text, "split": "train",
                    "phrase_id": pid, "sentiment_value": value.get(pid),
                    "raw_index": li, "raw_scores": scores.get(li) if li is not None else None})
    return out


# ---- Financial PhraseBank -------------------------------------------------------------

def fetch_fpb() -> list[dict]:
    z = zipfile.ZipFile(hub.file("takala/financial_phrasebank",
                                 "data/FinancialPhraseBank-v1.0.zip"))
    level: dict[str, int] = {}
    label: dict[str, str] = {}
    order: list[str] = []
    for agree in (50, 66, 75, 100):
        name = next(n for n in z.namelist()
                    if n.endswith(f"Sentences_{'All' if agree == 100 else agree}Agree.txt"))
        for line in z.read(name).decode("latin-1").splitlines():
            if "@" not in line:
                continue
            text, lab = line.rsplit("@", 1)
            text, lab = text.strip(), lab.strip()
            if text not in level:
                order.append(text)
                label[text] = lab
            elif label[text] != lab:
                label[text] = "CONFLICT"
            level[text] = agree        # the subsets are nested: the last seen is the highest
    return [{"sentence": t, "label": label[t], "agree_min": level[t]} for t in order]


# ---- Hub files ------------------------------------------------------------------------

def fetch_amazon() -> list[dict]:
    pf = hub.open_parquet(
        "datasets/fancyzhx/amazon_polarity/amazon_polarity/train-00000-of-00004.parquet")
    out: list[dict] = []
    for g in range(pf.num_row_groups):
        out += pf.read_row_group(g).to_pylist()
        if len(out) >= AMAZON_ROWS:
            break
    return out[:AMAZON_ROWS]


def fetch_hub(name: str) -> tuple[list[dict], dict[str, list[str]]]:
    if name == "amazon":
        return fetch_amazon(), hub.builder_classlabels("fancyzhx/amazon_polarity", None)
    if name == "imdb":
        rows = hub.parquet_rows(hub.file("stanfordnlp/imdb",
                                         "plain_text/train-00000-of-00001.parquet"))
        return rows, hub.builder_classlabels("stanfordnlp/imdb", "plain_text")
    if name.startswith("tweeteval_"):
        cfg = name[len("tweeteval_"):]
        rows = hub.parquet_rows(hub.file("cardiffnlp/tweet_eval",
                                         f"{cfg}/train-00000-of-00001.parquet"))
        return rows, hub.builder_classlabels("cardiffnlp/tweet_eval", cfg)
    if name == "claim_stance":
        text = hub.file("ibm-research/claim_stance", "train.csv").read_text(encoding="utf-8")
        return list(csv.DictReader(io.StringIO(text))), {}
    if name == "poem":
        rows = hub.parquet_rows(hub.file("google-research-datasets/poem_sentiment",
                                         "data/train-00000-of-00001.parquet"))
        return rows, hub.builder_classlabels("google-research-datasets/poem_sentiment", None)
    if name == "tfns":
        text = hub.file("zeroshot/twitter-financial-news-sentiment",
                        "sent_train.csv").read_text(encoding="utf-8")
        # the dataset card's label map; the csv carries only the integer
        return (list(csv.DictReader(io.StringIO(text))),
                {"label": ["Bearish", "Bullish", "Neutral"]})
    raise KeyError(name)


SOURCES = ("sst", "fpb", "amazon", "imdb", "claim_stance", "poem", "tfns") + tuple(
    f"tweeteval_{c}" for c in TWEETEVAL_CONFIGS)


def outputs() -> list[Path]:
    return [store.path_for(PREFIX + name) for name in SOURCES]


def fetch(only: tuple[str, ...] = SOURCES, refresh: bool = False) -> int:
    done = 0
    for name in only:
        key = PREFIX + name
        if store.has(key) and not refresh:
            continue
        if name == "sst":
            rows, feats = fetch_sst(), {}
        elif name == "fpb":
            rows, feats = fetch_fpb(), {}
        else:
            rows, feats = fetch_hub(name)
        n = store.save(key, rows)
        if feats:
            store.save_features(key, feats)
        print(f"  {key}: {n:,} rows, {store.path_for(key).stat().st_size / 1e6:.1f} MB",
              flush=True)
        done += 1
    return done
