"""Source-table rows 20 and 21 -- crowd annotations, aggregated into soft targets.

Both datasets ship **one row per rater**, not one row per item. GoEmotions `raw` has a
`rater_id` and 28 binary emotion columns; Measuring Hate Speech has an `annotator_id` and
ten ordinal survey items. Read naively -- the way a column adapter reads them -- each
rater's opinion becomes a separate hard-labelled example, which throws away exactly the
thing these datasets exist for: the *disagreement*.

Aggregating over the item turns that disagreement into the soft target the row wants: a
comment eight of ten raters call toxic is 0.8, not a coin flip and not a certainty.
Items with fewer than `MIN_RATERS` raters are dropped -- a "fraction" over one opinion is
a hard label wearing a soft label's clothes.

**Every target here is a share of raters answering the question asked, "yes".**

* GoEmotions is a vote: the target for "Does this comment express anger?" is the share
  of raters who ticked anger. The denominator is the raters who ticked *something*: a
  rater must choose an emotion or `neutral`, or mark the example very unclear, and an
  all-zero row is that last case. Counting it (the earlier cache did) dilutes every emotion
  and `neutral` alike -- 2,730 of 57,349 items.
* Measuring Hate Speech is a survey of five-point scales (Sachdeva et al. 2022, appendix
  Table 1), coded so that the higher value is always the more hateful end. Earlier
  builds used the *mean rating over the scale* -- the paper's "average normalised
  score" -- as p(yes). That is not an answer to a yes/no question. Four items are
  bipolar with a substantive neutral midpoint (sentiment "neutral", respect "neutral",
  status "neither superior nor inferior", attack/defend "neither"), so a comment every
  rater called neutral scored 0.5 on "does this comment state that the group is
  inferior?": 92.8 % of `hatespeech__status` targets were >= 0.5. The target is now the
  share of raters on the hateful side of the item's scale; the midpoint counts as "no"
  where it is a neutral answer and as half where it is an *undecided* one ("neither
  disagree nor agree", hate speech "unclear"). See `HATE_CREDIT`.

The caches hold **counts**, not fractions -- per-emotion vote counts, per-item rating
histograms -- so a change to what the target means is a change here, never a stale file.
`_GE_CACHE` / `_HS_CACHE` carry a version in their name; the earlier files held fractions
computed under the old definitions and are ignored.

Near-constant tasks. Most GoEmotions emotions are ticked by nobody for 90-99 % of
comments (grief: 99.1 % zero), so a contiguous slice of the pool is a task a constant
"no" answers. Each task's slice is therefore filled **positives first** --
`enriched_slices` gives each task up to half its quota from the items where at least
one rater said yes to *its* question, taking turns, then fills from the rest
of the pool. Targets are untouched: each item still carries its own exact rater share.
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Callable, Iterator, Sequence

from lod.phrasings import PhrasingBank
from lod.schema import Example, Question
from lod.corpus.services.sources.base import CACHE
from lod.corpus.repositories import raw_store as store
from lod.corpus.services.sources.real.base import RealTask, disjoint_slice

GO_EMOTIONS = "google-research-datasets/go_emotions"
HATE = "ucberkeley-dlab/measuring-hate-speech"
LICENCE_GE = "Apache-2.0"
LICENCE_HS = "CC-BY-4.0"

MIN_RATERS = 3          # below this a fraction is not a distribution
MAX_ITEMS = 60_000
# a task's slice takes at most this share of its quota from items some rater said yes to
POSITIVE_SHARE = 0.5

GO_EMOTION_COLS = (
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion",
    "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment",
    "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism",
    "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral",
)

HATE_COLS = ("hatespeech", "sentiment", "respect", "insult", "humiliate", "status",
             "dehumanize", "violence", "genocide", "attack_defend")

# How much of a "yes" each rating level is, per item, from the codebook's response
# options (Sachdeva et al. 2022, appendix Table 1) in the Hub release's coding, where the
# higher value is always the more hateful end (checked: each level's mean
# `hate_speech_score` rises with the level, and the paper says so in words).
_BIPOLAR = (0.0, 0.0, 0.0, 1.0, 1.0)   # neutral midpoint: a "no"
_AGREE = (0.0, 0.0, 0.5, 1.0, 1.0)     # "neither disagree nor agree": undecided, half
HATE_CREDIT: dict[str, tuple[float, ...]] = {
    # strongly positive, somewhat positive, neutral, somewhat negative, strongly negative
    "sentiment": _BIPOLAR,
    # strongly respectful, respectful, neutral, disrespectful, strongly disrespectful
    "respect": _BIPOLAR,
    # strongly superior, superior, neither superior nor inferior, inferior, strongly inferior
    "status": _BIPOLAR,
    # strongly defending, defending, neither, attacking, strongly attacking
    "attack_defend": _BIPOLAR,
    # strongly disagree .. strongly agree with "This comment is <item> towards the group"
    "insult": _AGREE,
    "humiliate": _AGREE,
    "dehumanize": _AGREE,
    "violence": _AGREE,
    "genocide": _AGREE,
    # no, unclear, yes
    "hatespeech": (0.0, 0.5, 1.0),
}
HATE_SCALE: dict[str, int] = {c: len(v) - 1 for c, v in HATE_CREDIT.items()}

# The wording is the codebook's own prompt, turned to face the hateful end of the scale,
# because that is the end the target counts. An earlier build derived it from the column
# name and asked "respectful?" and "positive sentiment?" against targets that counted
# disrespect and negativity, and the enrichment cache re-inverted both after a later fix.
# These tasks are therefore in `PROTECTED_PREFIXES` (`services/enrich/schemas.py`): the
# wording is load-bearing, and only a phrasing-bank variant checked against
# `answer_meaning` in lod/assets/phrasing_specs/d03.json may replace it.
HATE_QUESTIONS: dict[str, str] = {
    "hatespeech": "Does this comment contain hate speech?",
    "sentiment": "Is the overall sentiment of this comment negative?",
    "respect": "Is this comment disrespectful toward the group it targets?",
    "insult": "Is this comment insulting toward the group it targets?",
    "humiliate": "Is this comment humiliating toward the group it targets?",
    "status": "Does this comment state that the group it targets is inferior?",
    "dehumanize": "Does this comment dehumanize the group it targets, for example by "
                  "comparing its members to animals?",
    "violence": "Does this comment call for using violence against the group it targets?",
    "genocide": "Does this comment call for the deliberate killing of a large number of "
                "people from the group it targets?",
    "attack_defend": "Does this comment attack the group it targets, rather than defend "
                     "it or do neither?",
}
HATE_TEMPLATE_ID = "d03.hatespeech_{col}"

# GoEmotions asks one yes/no per emotion, and the emotion name is the question. Only
# `neutral` needs saying differently: "Does this comment express neutral?" is not a
# sentence, and the column means the absence of any of the other 27.
GO_EMOTION_QUESTIONS: dict[str, str] = {
    "neutral": "Does this comment express no particular emotion?",
}

_GE_CACHE = CACHE / "go_emotions_counts_v2.jsonl"
_HS_CACHE = CACHE / "hate_speech_counts_v2.jsonl"

_BANK: PhrasingBank | None = None


def _phrasing(template_id: str, template: str, key: str) -> str:
    global _BANK
    if _BANK is None:
        _BANK = PhrasingBank.load()
    return _BANK.pick(template_id, template, key)


def _write(path, rows: list[dict]) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in rows))
    tmp.replace(path)


def _read(path) -> list[dict]:
    key = str(path)
    if key not in _IN_PROCESS:
        _IN_PROCESS[key] = [json.loads(l) for l in path.read_text().splitlines()
                            if l.strip()]
    return _IN_PROCESS[key]


# Read once per process. These are called once per task; without this the build
# re-parses the whole cache for each of them.
_IN_PROCESS: dict[str, list] = {}


def _local_rows(path: str, config: str | None = None):
    """Load downloaded Hub files from the configured raw root, never the network."""
    from datasets import load_dataset

    root = store.RAW_ROOT / "huggingface" / path
    if not root.exists():
        return None
    if path == GO_EMOTIONS:
        files = sorted((root / "raw").glob("*.parquet"))
    else:
        files = sorted((root / "data").glob("*.parquet"))
    if not files:
        return None
    return load_dataset("parquet", data_files=[str(f) for f in files], split="train")


def _norm(text: str) -> str:
    return " ".join((text or "").split())


# ------------------------------------------------------------------ GoEmotions ------

def go_emotion_frac(row: dict) -> dict[str, float]:
    """Share of the item's *labelling* raters who ticked each emotion."""
    n = row["n_labelled"]
    return {c: row["votes"].get(c, 0) / n for c in GO_EMOTION_COLS}


def aggregate_go_emotions(refresh: bool = False, max_items: int = MAX_ITEMS) -> list[dict]:
    """One row per comment text: vote counts, labelling raters, and the derived `frac`.

    Items are keyed on the comment's (whitespace-normalised) text, not its id. The raw
    set has 251 texts under more than one id ("Thank you!", "[NAME]"); keyed by id they
    reached two tasks, and so potentially two splits, as the same state with two
    targets. The state is the text, so the text is the item, and its raters are pooled.
    """
    if not (_GE_CACHE.exists() and not refresh):
        ds = _local_rows(GO_EMOTIONS, "raw")
        if ds is None:
            from datasets import load_dataset
            ds = load_dataset(GO_EMOTIONS, "raw", split="train", streaming=True)
        text: dict[str, str] = {}
        first_id: dict[str, str] = {}
        votes: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        raters: dict[str, int] = defaultdict(int)
        labelled: dict[str, int] = defaultdict(int)
        for r in ds:
            t = (r.get("text") or "").strip()
            key = _norm(t)
            if not key:
                continue
            if key not in text:
                if len(text) >= max_items:
                    continue       # never `break`: a later row may be an admitted item's
                text[key] = t
                first_id[key] = str(r.get("id") or "")
            raters[key] += 1
            ticked = [c for c in GO_EMOTION_COLS if r.get(c)]
            if not ticked:
                continue           # very unclear / no label: not an opinion on anything
            labelled[key] += 1
            for c in ticked:
                votes[key][c] += 1
        # `id` is the first comment id carrying this text; the item itself is the text
        out = [{"id": first_id[k], "text": text[k], "n_raters": raters[k],
                "n_labelled": labelled[k], "votes": dict(votes[k])}
               for k in text if labelled[k] >= MIN_RATERS]
        _write(_GE_CACHE, out)
        _IN_PROCESS.pop(str(_GE_CACHE), None)
    rows = _read(_GE_CACHE)
    for r in rows:
        if "frac" not in r:
            r["frac"] = go_emotion_frac(r)
    return rows


# ------------------------------------------------------- Measuring Hate Speech ------

def hate_frac(hist: dict[str, list[int]]) -> dict[str, float]:
    """Per item: the share of raters answering each item's question "yes".

    `hist[col][k]` is how many raters chose level k. A column answered by fewer than
    `MIN_RATERS` raters is left out, and `_loader` asks no question about it.
    """
    out = {}
    for col, credit in HATE_CREDIT.items():
        h = hist.get(col)
        if not h:
            continue
        n = sum(h)
        if n < MIN_RATERS:
            continue
        out[col] = sum(k * c for k, c in zip(h, credit)) / n
    return out


def aggregate_hate_speech(refresh: bool = False, max_items: int = MAX_ITEMS) -> list[dict]:
    """One row per comment: a rating histogram per survey item, and the derived `frac`."""
    if not (_HS_CACHE.exists() and not refresh):
        ds = _local_rows(HATE)
        if ds is None:
            from datasets import load_dataset
            ds = load_dataset(HATE, split="train", streaming=True)
        text: dict[str, str] = {}
        hist: dict[str, dict[str, list[int]]] = defaultdict(dict)
        n: dict[str, int] = defaultdict(int)
        for r in ds:
            iid = str(r.get("comment_id"))
            if not iid or iid == "None":
                continue
            if iid not in text:
                if len(text) >= max_items:
                    continue
                text[iid] = (r.get("text") or "").strip()
            n[iid] += 1
            for col, credit in HATE_CREDIT.items():
                v = r.get(col)
                if isinstance(v, bool) or not isinstance(v, (int, float)) or v != v:
                    continue           # unanswered: shrinks this item's denominator only
                k = int(round(float(v)))
                if not 0 <= k < len(credit) or abs(float(v) - k) > 1e-9:
                    continue           # off the published scale: not a rating we can read
                hist[iid].setdefault(col, [0] * len(credit))[k] += 1
        out = [{"id": iid, "text": text[iid], "n_raters": count, "hist": hist[iid]}
               for iid, count in n.items() if count >= MIN_RATERS and text.get(iid)]
        _write(_HS_CACHE, out)
        _IN_PROCESS.pop(str(_HS_CACHE), None)
    rows = _read(_HS_CACHE)
    for r in rows:
        if "frac" not in r:
            r["frac"] = hate_frac(r["hist"])
    return rows


# --------------------------------------------------------------- slicing -----------

def enriched_slices(rows: Sequence[dict], cols: Sequence[str],
                    positive: Callable[[dict, str], bool],
                    share: float = POSITIVE_SHARE) -> dict[str, list[dict]]:
    """Disjoint per-task slices of one item pool, each filled positives-first.

    `disjoint_slice` hands task i the i-th contiguous block, so a rare question gets the
    base rate of the whole pool -- 0.3 % for grief -- and is a task a constant answers.
    Here every task has the same quota (pool // tasks). The tasks take turns, rarest
    question first within each round, each claiming its next unclaimed item that
    `positive` says some rater answered yes to for *that* question, until it holds
    `share` of its quota or runs out; then each task, in `cols` order, tops up from the
    unclaimed items in pool order. Turns rather than one task after another because the
    positives overlap -- an insult is also rated toxic -- and whoever claims first would
    otherwise leave the others the remainder of a small pool. A slice is returned in pool
    order, so a prefix of it is as mixed as the whole. No item is in two slices, so the
    whole-task split cannot put one state on both sides.
    """
    cols = list(cols)
    if not cols:
        return {}
    quota = len(rows) // len(cols)
    cap = int(quota * share)
    queues = {c: [i for i, r in enumerate(rows) if positive(r, c)] for c in cols}
    order = sorted(cols, key=lambda c: (len(queues[c]), cols.index(c)))
    cursor = {c: 0 for c in cols}
    taken: set[int] = set()
    mine: dict[str, list[int]] = {c: [] for c in cols}
    live = [c for c in order if cap > 0]
    while live:
        still = []
        for c in live:
            q = queues[c]
            while cursor[c] < len(q) and q[cursor[c]] in taken:
                cursor[c] += 1
            if cursor[c] >= len(q):
                continue
            taken.add(q[cursor[c]])
            mine[c].append(q[cursor[c]])
            if len(mine[c]) < cap:
                still.append(c)
        live = still
    free = (i for i in range(len(rows)) if i not in taken)
    for c in cols:
        while len(mine[c]) < quota:
            i = next(free, None)
            if i is None:
                break
            taken.add(i)
            mine[c].append(i)
    return {c: [rows[i] for i in sorted(mine[c])] for c in cols}


def _has_yes(r: dict, col: str) -> bool:
    p = r.get("frac", {}).get(col)
    return p is not None and p > 0


_SLICES: dict[tuple, dict[str, list[dict]]] = {}


def _slice(rows_fn, col: str, index: int, n_tasks: int, cols: Sequence[str] | None):
    rows = rows_fn()
    if not cols:
        return disjoint_slice(rows, index, n_tasks)
    key = (id(rows), len(rows), tuple(cols))
    if key not in _SLICES:
        _SLICES[key] = enriched_slices(rows, cols, _has_yes)
    return _SLICES[key][col]


def _loader(rows_fn, col: str, task: str, question: str, index: int, n_tasks: int,
            cols: Sequence[str] | None = None,
            phrase: Callable[[dict], str] | None = None):
    def load(n: int) -> Iterator[Example]:
        made = 0
        for r in _slice(rows_fn, col, index, n_tasks, cols):
            if made >= n:
                return
            p = r["frac"].get(col)
            if p is None:
                continue
            made += 1
            yield Example(
                task=task,
                state=r["text"][:4000],
                questions=[Question(id=col,
                                    question=phrase(r) if phrase else question,
                                    options=["no", "yes"],
                                    # the rater share, not a rounded label
                                    target=[1.0 - p, p],
                                    meta={"item": str(r.get("id", "")),
                                          "raters": r.get("n_labelled", r.get("n_raters"))})],
            )
    return load


def _hate_phrase(col: str) -> Callable[[dict], str]:
    tid = HATE_TEMPLATE_ID.format(col=col)
    return lambda r: _phrasing(tid, HATE_QUESTIONS[col], f"{col}:{r.get('id', '')}")


def tasks() -> list[RealTask]:
    """Returns [] until the aggregates can be built, so the registry stays offline."""
    out: list[RealTask] = []

    # Build the count caches from downloaded parquet on first use; only fall back to the
    # Hub when no local snapshot exists (see `_local_rows`).
    if not _GE_CACHE.exists() and (store.RAW_ROOT / "huggingface" / GO_EMOTIONS).exists():
        aggregate_go_emotions()
    if not _HS_CACHE.exists() and (store.RAW_ROOT / "huggingface" / HATE).exists():
        aggregate_hate_speech()

    if _GE_CACHE.exists():
        n = len(GO_EMOTION_COLS)
        for i, col in enumerate(GO_EMOTION_COLS):
            out.append(RealTask(
                row=20, name=f"goemotions__{col}", licence=LICENCE_GE,
                url=f"https://huggingface.co/datasets/{GO_EMOTIONS}",
                load=_loader(aggregate_go_emotions, col, f"goemotions__{col}",
                             GO_EMOTION_QUESTIONS.get(
                                 col, f"Does this comment express {col.replace('_', ' ')}?"),
                             i, n, cols=GO_EMOTION_COLS),
                # one aggregated pool, one column per task: a held-out column is not a
                # held-out dataset (make_devood.py)
                family="goemotions",
                soft=True,
                notes=f"share of labelling raters choosing {col}; >= {MIN_RATERS} raters; "
                      f"slice filled positives-first"))

    if _HS_CACHE.exists():
        n = len(HATE_COLS)
        for i, col in enumerate(HATE_COLS):
            out.append(RealTask(
                row=21, name=f"hatespeech__{col}", licence=LICENCE_HS,
                url=f"https://huggingface.co/datasets/{HATE}",
                load=_loader(aggregate_hate_speech, col, f"hatespeech__{col}",
                             HATE_QUESTIONS[col], i, n, cols=HATE_COLS,
                             phrase=_hate_phrase(col)),
                family="hatespeech",
                soft=True, ordinal=True,
                notes=f"share of raters on the hateful side of the codebook's 0-"
                      f"{HATE_SCALE[col]} {col} scale (undecided midpoint = 1/2, "
                      f"neutral midpoint = no)"))

    return out
