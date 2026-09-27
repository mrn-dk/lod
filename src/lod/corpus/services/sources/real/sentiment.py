"""Source-table row 44 -- English sentiment, emotion and stance (domain 3).

Human-labelled English classification at real volume (up to ~2,000 questions a task),
where the row-1 Hub sweep gave each of these datasets ~89 questions read through a
generic column adapter. Raw rows come from `scripts/fetch_data.py --rows 44`
(`v14s_<name>` in the raw store); this module owns what every label means.

Tasks and where they go (whole tasks, `force_split`; nothing is left to the name hash):

  train     sent44_sst5 (soft), sent44_amazon_polarity, sent44_tweeteval_sentiment,
            sent44_tweeteval_offensive, sent44_tweeteval_emotion,
            sent44_stance_abortion / _atheism / _climate, sent44_claim_stance
  devreal   sent44_poem_sentiment, sent44_tfns, sent44_stance_feminist
  testreal  sent44_imdb, sent44_fpb (soft), sent44_tweeteval_irony, sent44_stance_hillary

Held out whole: two domains of text (long-form film reviews, financial news), one concept
(irony) and one stance target, each with a trained neighbour (SST sentences, financial
tweets in devreal, sentiment, four other stance targets).

Soft targets, only where the publisher gives a rater distribution or agreement level:

* **SST-5.** The treebank's raw release (`stanfordSentimentTreebankRaw.zip`) has each
  phrase's 3 (rarely up to 6) annotator slider values on 1..25. Each annotator's value
  is mapped to [0,1] as (v-1)/24 and binned with the SST's own published cut-offs
  [0,.2], (.2,.4], (.4,.6], (.6,.8], (.8,1] -- the same mapping the release applies to
  the *mean* to make the 5-class label -- and the target is the share of annotators per
  class. A sentence is kept only if the mean of its raw scores reproduces the published
  `sentiment_labels.txt` value within 0.01: 8,156 of 8,455 joined train sentences do,
  and the other 299 (the raw file's README warns of earlier acquisition rounds) are
  dropped rather than trusted.
* **Financial PhraseBank.** Each sentence was labelled by 5-8 annotators; the release
  publishes the label and nested agreement subsets (>= 50 %, >= 66 %, >= 75 %, 100 %).
  The majority label gets the mean feasible majority share of its highest bracket
  (1.0 / 0.82 / 0.69 / 0.57, see `FPB_MAJORITY`); the rest goes to the *adjacent* class
  on negative < neutral < positive -- to neutral for a polar sentence, split evenly for a
  neutral one. The exact split is not published; this is the bracket's estimate.

Both carry the class the source itself publishes in `meta.source_label` (SST: the bin of
the mean score; FPB: the majority label); a tie between rater shares cannot name it.

Everything else is a hard label. Pool rules, all per task: identical texts (compared on
letters and digits only) appear once, a text the source labels two ways is dropped, a
text claimed by an earlier task in `PRIORITY` (eval tasks first) is not reused, a state
over `MAX_STATE_CHARS` is dropped whole rather than cut, and a sample drawn from a pool
larger than the quota caps every class at `MAX_CLASS_SHARE` of it.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

from lod.phrasings import PhrasingBank
from lod.schema import Example, Question
from lod.corpus.repositories import raw_store as store
from lod.corpus.services.sources.real.base import RealTask
from lod.paths import ASSETS

ROW = 44
PREFIX = "sent44_"
TEMPLATE_ID = "v14s.{task}"
MAX_STATE_CHARS = 4000
MAX_CLASS_SHARE = 0.5

SST_CUTS = (0.2, 0.4, 0.6, 0.8, 1.0)
SST_MATCH_TOL = 0.01
# mean feasible majority share inside each Financial PhraseBank agreement bracket, over
# 5-8 annotators: [.75,1) -> {.75,.8,.833,.857,.875}; [.66,.75) -> {.667,.714};
# [.5,.66) -> {.5,.571,.6,.625}. The 100 % subset is unanimous.
FPB_MAJORITY = {100: 1.0, 75: 0.82, 66: 0.69, 50: 0.57}

LIC_TWEETEVAL = ("undefined per the TweetEval card (research release; Twitter ToS "
                 "applies)")
LIC_TWEETEVAL_SENT = "CC-BY-3.0 (SemEval-2017 Task 4, per the TweetEval card)"


@dataclass(frozen=True)
class Spec:
    task: str                         # without PREFIX
    key: str                          # raw-store key
    split: str
    licence: str
    url: str
    question: str
    options: tuple[str, ...]
    criteria: tuple[str | None, ...]  # one per option, published definitions
    state_kind: str
    answer_meaning: str
    ordinal: bool = False
    soft: bool = False
    notes: str = ""
    # raw row -> (state, target) or None. target: option index, or a probability list.
    # A soft reader returns (state, target, published class index) -- the class the
    # source itself assigns, which a tie in the rater shares cannot name.
    read: Callable[[dict], tuple | None] = field(default=lambda r: None)

    @property
    def name(self) -> str:
        return PREFIX + self.task


# ------------------------------------------------------------------ readers ----------

_U_ESCAPE = re.compile(r"\\u([0-9a-fA-F]{4})")


def _clean(text: str) -> str:
    """Whitespace folded; HTML entities and the literal `\\u002c` / `\\u2019` escapes the
    SemEval-2017 tweets carry (11,354 of them in TweetEval sentiment) decoded."""
    text = _U_ESCAPE.sub(lambda m: chr(int(m.group(1), 16)), str(text or ""))
    text = html.unescape(text).replace("\ufeff", "").replace("\ufffd", "")
    return re.sub(r"[ \t]+", " ", text).strip()


_PTB = ((r"-LRB-", "("), (r"-RRB-", ")"), (r"``", '"'), (r"''", '"'), (r"\\/", "/"))


def sst_detok(text: str) -> str:
    """The treebank's PTB tokens back to readable text. A latin-1 round trip in the
    release turned "é" into "Ã©"; that is undone first."""
    try:
        text = text.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    for a, b in _PTB:
        text = re.sub(a, b, text)
    text = re.sub(r" (n't|'s|'re|'ve|'m|'ll|'d)\b", r"\1", text)
    text = re.sub(r" ([.,;:!?%)])", r"\1", text)
    text = re.sub(r"([($]) ", r"\1", text)
    text = re.sub(r'" (.*?) "', r'"\1"', text)
    return _clean(text)


def sst_bin(v: float) -> int:
    for i, cut in enumerate(SST_CUTS):
        if v <= cut + 1e-9:
            return i
    return len(SST_CUTS) - 1


def sst_target(raw_scores: list[int]) -> list[float]:
    """Share of annotators whose own slider value falls in each SST class."""
    votes = [0] * 5
    for s in raw_scores:
        votes[sst_bin((s - 1) / 24)] += 1
    n = sum(votes)
    return [v / n for v in votes]


def _read_sst(r: dict):
    scores, pub = r.get("raw_scores"), r.get("sentiment_value")
    if not scores or pub is None or len(scores) < 3:
        return None
    if abs((statistics.mean(scores) - 1) / 24 - pub) > SST_MATCH_TOL:
        return None                    # the raw scores are not the published label's
    return sst_detok(r["sentence"]), sst_target(scores), sst_bin(pub)


FPB_ORDER = ("negative", "neutral", "positive")


def fpb_target(label: str, agree_min: int) -> list[float]:
    p = FPB_MAJORITY[agree_min]
    i = FPB_ORDER.index(label)
    t = [0.0, 0.0, 0.0]
    t[i] = p
    if i == 1:
        t[0] = t[2] = (1 - p) / 2
    else:
        t[1] = 1 - p
    return t


def _read_fpb(r: dict):
    if r.get("label") not in FPB_ORDER or r.get("agree_min") not in FPB_MAJORITY:
        return None
    return (_clean(r["sentence"]), fpb_target(r["label"], r["agree_min"]),
            FPB_ORDER.index(r["label"]))


def _read_amazon(r: dict):
    # `content` only: the review title is the reviewer's own one-line verdict ("Great
    # product!", "Waste of money") and restates the label the stars gave.
    return _clean(r.get("content")), int(r["label"])


def _read_imdb(r: dict):
    text = re.sub(r"<br\s*/?>", "\n", str(r.get("text") or ""))
    text = re.sub(r"\n{3,}", "\n\n", text)
    return _clean(text), int(r["label"])


SEMST = re.compile(r"\s*#SemST\b", re.I)


def _read_tweet(r: dict):
    # `#SemST` is the SemEval-2016 collection marker appended to every stance tweet
    return _clean(SEMST.sub("", str(r.get("text") or ""))), int(r["label"])


def _read_claim(r: dict):
    lab = {"PRO": 0, "CON": 1}.get(str(r.get("claims.stance")))
    claim = _clean(r.get("claims.claimCorrectedText"))
    if lab is None or not claim:
        return None
    return json.dumps({"motion": _clean(r.get("topicText")), "claim": claim},
                      ensure_ascii=False, indent=2), lab


def _read_poem(r: dict):
    return _clean(r.get("verse_text")), int(r["label"])


# published order Bearish(0) Bullish(1) Neutral(2) -> ordinal Bearish, Neutral, Bullish
_TFNS = {0: 0, 2: 1, 1: 2}


def _read_tfns(r: dict):
    return _clean(r.get("text")), _TFNS[int(r["label"])]


# ------------------------------------------------------------------ the table --------

_TW = "https://huggingface.co/datasets/cardiffnlp/tweet_eval"
_TW_KIND = "one English tweet from the TweetEval benchmark, user mentions anonymised as @user"

STANCE_CRITERIA = (
    "the tweet gives no clue to the author's stance toward the target, or it is neutral",
    "we can infer from the tweet that the author is against the target",
    "we can infer from the tweet that the author supports the target",
)
STANCE_TARGETS = {
    # SemEval-2016 Task 6 target -> how the question names it
    "abortion": ("Legalization of Abortion", "the legalization of abortion"),
    "atheism": ("Atheism", "atheism"),
    "climate": ("Climate Change is a Real Concern",
                "the position that climate change is a real concern"),
    "feminist": ("Feminist Movement", "the feminist movement"),
    "hillary": ("Hillary Clinton", "Hillary Clinton"),
}
STANCE_SPLIT = {"abortion": "train", "atheism": "train", "climate": "train",
                "feminist": "devreal", "hillary": "testreal"}


def _stance(target: str) -> Spec:
    official, phrase = STANCE_TARGETS[target]
    return Spec(
        task=f"stance_{target}", key=f"v14s_tweeteval_stance_{target}",
        split=STANCE_SPLIT[target], licence=LIC_TWEETEVAL, url=_TW,
        question=f"What stance does the author of this tweet take toward {phrase}?",
        options=("none", "against", "favor"), criteria=STANCE_CRITERIA,
        state_kind=_TW_KIND + "; the target need not be named in the tweet",
        answer_meaning=(f"options none / against / favor: the SemEval-2016 Task 6 stance "
                        f"label toward the target '{official}' -- favor = the author "
                        f"supports it, against = the author opposes it, none = neutral "
                        f"or no clue. Stance toward the target, not the tweet's sentiment"),
        read=_read_tweet,
        notes=f"SemEval-2016 Task 6 (Mohammad et al.), target '{official}'; not VAST")


SPECS: tuple[Spec, ...] = (
    # ---- testreal -------------------------------------------------------------------
    Spec(task="imdb", key="v14s_imdb", split="testreal",
         licence="no licence stated (Maas et al. 2011, ai.stanford.edu/~amaas/data/sentiment)",
         url="https://huggingface.co/datasets/stanfordnlp/imdb",
         question="Did the writer of this IMDb movie review rate the film positively or "
                  "negatively?",
         options=("negative", "positive"),
         criteria=("the review's rating was 4 or lower out of 10",
                   "the review's rating was 7 or higher out of 10"),
         state_kind="one full English user review of a film from IMDb, HTML line breaks "
                    "restored as newlines; reviews over 4,000 characters are not used",
         answer_meaning="options negative / positive: the class of the rating the writer "
                        "gave (<= 4/10 negative, >= 7/10 positive; neutral ratings are not "
                        "in the dataset)",
         read=_read_imdb,
         notes="Maas et al. 2011 train split; states over 4,000 chars dropped, not cut"),
    Spec(task="fpb", key="v14s_fpb", split="testreal",
         licence="CC-BY-NC-SA-3.0",
         url="https://huggingface.co/datasets/takala/financial_phrasebank",
         question="From an investor's point of view, is this sentence from financial news "
                  "likely to have a negative, neutral or positive influence on the stock "
                  "price of the company it concerns?",
         options=FPB_ORDER,
         criteria=("the news is likely to push the stock price down",
                   "no clear influence on the stock price",
                   "the news is likely to push the stock price up"),
         state_kind="one English sentence from financial news about a (mostly Finnish, "
                    "OMX Helsinki listed) company",
         answer_meaning="a soft distribution over negative / neutral / positive: the "
                        "annotators' (5-8 finance-trained) majority label, weighted by the "
                        "published agreement bracket, the remainder on the adjacent class",
         ordinal=True, soft=True, read=_read_fpb,
         notes="Malo et al. 2014; soft target from the agreement subset (50/66/75/100 %), "
               "see sentiment.FPB_MAJORITY"),
    Spec(task="tweeteval_irony", key="v14s_tweeteval_irony", split="testreal",
         licence=LIC_TWEETEVAL, url=_TW,
         question="Is this tweet ironic?",
         options=("non_irony", "irony"),
         criteria=("the tweet means what it says; no verbal or situational irony",
                   "verbal irony (saying the opposite of what is meant, or an evaluation "
                   "at odds with the context) or a description of situational irony"),
         state_kind=_TW_KIND,
         answer_meaning="options non_irony / irony: the SemEval-2018 Task 3 subtask A label "
                        "(Van Hee et al.), manually annotated; the #irony/#sarcasm/#not "
                        "query hashtags were removed by the task organisers",
         read=_read_tweet,
         notes="SemEval-2018 Task 3A (Van Hee et al. 2018); not iSarcasmEval"),
    _stance("hillary"),
    # ---- devreal --------------------------------------------------------------------
    Spec(task="poem_sentiment", key="v14s_poem", split="devreal",
         licence="CC-BY-4.0",
         url="https://huggingface.co/datasets/google-research-datasets/poem_sentiment",
         question="What sentiment does this line of poetry convey?",
         options=("negative", "positive", "no_impact", "mixed"),
         criteria=("the line conveys a negative sentiment",
                   "the line conveys a positive sentiment",
                   "the line conveys no particular sentiment",
                   "the line conveys both positive and negative sentiment"),
         state_kind="one verse (a single line) of an English poem from Project Gutenberg",
         answer_meaning="options negative / positive / no_impact / mixed: the Gutenberg Poem "
                        "Dataset label (Sheng & Uthus 2020), expert-annotated per verse",
         read=_read_poem, notes="Sheng & Uthus 2020"),
    Spec(task="tfns", key="v14s_tfns", split="devreal", licence="MIT",
         url="https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment",
         question="Is this finance-related tweet bearish, neutral or bullish about the "
                  "company, asset or market it discusses?",
         options=("Bearish", "Neutral", "Bullish"),
         criteria=("negative for the price or outlook: expecting it to fall",
                   "neither bearish nor bullish",
                   "positive for the price or outlook: expecting it to rise"),
         state_kind="one English finance-news tweet, often with a $TICKER and a t.co link",
         answer_meaning="options Bearish / Neutral / Bullish: the dataset's annotated "
                        "sentiment label (LABEL_0 Bearish, LABEL_1 Bullish, LABEL_2 Neutral)",
         ordinal=True, read=_read_tfns, notes="Twitter Financial News, sent_train.csv"),
    _stance("feminist"),
    # ---- train ----------------------------------------------------------------------
    Spec(task="sst5", key="v14s_sst", split="train",
         licence="no licence stated (Stanford Sentiment Treebank v1.0, nlp.stanford.edu/sentiment)",
         url="https://nlp.stanford.edu/sentiment/",
         question="How negative or positive is the sentiment of this sentence from a "
                  "movie review?",
         options=("very negative", "negative", "neutral", "positive", "very positive"),
         criteria=("positivity 0-0.2 on the treebank's scale",
                   "positivity above 0.2, up to 0.4",
                   "positivity above 0.4, up to 0.6",
                   "positivity above 0.6, up to 0.8",
                   "positivity above 0.8"),
         state_kind="one sentence from a Rotten Tomatoes movie review snippet (Stanford "
                    "Sentiment Treebank train split), detokenised",
         answer_meaning="a soft distribution over the five SST classes: the share of the "
                        "sentence's annotators (usually 3) whose own 25-point slider value "
                        "falls in each class under SST's published cut-offs",
         ordinal=True, soft=True, read=_read_sst,
         notes="Socher et al. 2013; per-annotator raw scores from the raw release, kept "
               "only where their mean reproduces the published label (+-0.01)"),
    Spec(task="amazon_polarity", key="v14s_amazon", split="train",
         licence="Apache-2.0 (per the fancyzhx/amazon_polarity card)",
         url="https://huggingface.co/datasets/fancyzhx/amazon_polarity",
         question="Did the writer of this Amazon review rate the product positively or "
                  "negatively?",
         options=("negative", "positive"),
         criteria=("the review's star rating was 1 or 2", "the review's star rating was 4 or 5"),
         state_kind="the body of one English Amazon product review (the title is removed)",
         answer_meaning="options negative / positive: the class of the writer's star "
                        "rating (1-2 negative, 4-5 positive; 3-star reviews are not in "
                        "the dataset)",
         read=_read_amazon, notes="Zhang et al. 2015 from McAuley & Leskovec 2013; title "
                                  "dropped from the state"),
    Spec(task="tweeteval_sentiment", key="v14s_tweeteval_sentiment", split="train",
         licence=LIC_TWEETEVAL_SENT, url=_TW,
         question="What is the overall sentiment of this tweet?",
         options=("negative", "neutral", "positive"),
         criteria=("the tweet expresses a negative sentiment overall",
                   "the tweet expresses neither a positive nor a negative sentiment",
                   "the tweet expresses a positive sentiment overall"),
         state_kind=_TW_KIND,
         answer_meaning="options negative / neutral / positive: the SemEval-2017 Task 4A "
                        "message-level sentiment label (Rosenthal et al.), crowd-annotated",
         ordinal=True, read=_read_tweet, notes="SemEval-2017 Task 4A"),
    Spec(task="tweeteval_offensive", key="v14s_tweeteval_offensive", split="train",
         licence=LIC_TWEETEVAL, url=_TW,
         question="Does this tweet contain offensive language or a targeted offense?",
         options=("non-offensive", "offensive"),
         criteria=("the tweet contains no offense and no profanity",
                   "the tweet contains non-acceptable language (profanity) or a targeted "
                   "offense, veiled or direct: insults, threats, swear words"),
         state_kind=_TW_KIND + "; URLs written as URL",
         answer_meaning="options non-offensive / offensive: the OLID level-A label "
                        "(OffensEval 2019, Zampieri et al.)",
         read=_read_tweet, notes="SemEval-2019 Task 6 (OLID) level A"),
    Spec(task="tweeteval_emotion", key="v14s_tweeteval_emotion", split="train",
         licence=LIC_TWEETEVAL, url=_TW,
         question="Which emotion best represents the mental state of the author of this "
                  "tweet?",
         options=("anger", "joy", "optimism", "sadness"),
         criteria=("displeasure or antagonism, including annoyance and rage",
                   "pleasure and happiness, including serenity and ecstasy",
                   "hopefulness and confidence about the future",
                   "sorrow or unhappiness, including pensiveness and grief"),
         state_kind=_TW_KIND,
         answer_meaning="options anger / joy / optimism / sadness: the SemEval-2018 Task 1 "
                        "(Affect in Tweets) E-c emotion for tweets carrying one of these "
                        "four, as TweetEval selects them",
         read=_read_tweet, notes="SemEval-2018 Task 1 E-c, four-emotion TweetEval subset"),
    _stance("abortion"),
    _stance("atheism"),
    _stance("climate"),
    Spec(task="claim_stance", key="v14s_claim_stance", split="train",
         licence="CC-BY-3.0", url="https://huggingface.co/datasets/ibm-research/claim_stance",
         question="Does this claim support or contest the debate motion?",
         options=("PRO", "CON"),
         criteria=("the claim supports the motion", "the claim contests the motion"),
         state_kind="a JSON object: a debate motion ('This house ...') and one claim about "
                    "it, a sentence taken from Wikipedia",
         answer_meaning="options PRO / CON: the IBM Debater Claim Stance label (Bar-Haim et "
                        "al. 2017), expert-annotated stance of the claim toward the motion",
         read=_read_claim, notes="IBM Debater Claim Stance Dataset, train topics"),
)
SPEC_BY_NAME = {s.name: s for s in SPECS}
# which task keeps a text found in several pools: eval before train, so the dedup in
# `generate` never has to delete an eval state for being in a train pool
PRIORITY = [s.name for s in SPECS]


# ------------------------------------------------------------------ pools ------------

def norm_key(text: str) -> str:
    """Letters and digits only: "it's" and "its", "@user" spacing, one text."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _rank(task: str, state: str) -> bytes:
    return hashlib.sha1(f"{task}\x00{state}".encode("utf-8")).digest()


Item = tuple[str, object, int]        # (state, target, class for balancing / reporting)


def raw_pool(spec: Spec) -> tuple[list[Item], dict[str, int]]:
    """(state, target, class) items for one task, before the cross-task claim. + drops."""
    items: dict[str, Item] = {}
    labels: dict[str, set] = defaultdict(set)
    drops = defaultdict(int)
    for r in store.load(spec.key):
        got = spec.read(r)
        if got is None:
            drops["unreadable_or_unverified"] += 1
            continue
        state, target = got[0], got[1]
        cls = got[2] if len(got) > 2 else int(target)
        if not state:
            drops["empty"] += 1
            continue
        if len(state) > MAX_STATE_CHARS:
            drops["over_4000_chars"] += 1
            continue
        k = norm_key(state)
        if di_hash(k) in di_excluded():
            drops["decision_index_member"] += 1
            continue
        labels[k].add(json.dumps(target))
        if k in items:
            drops["duplicate_text"] += 1
            continue
        items[k] = (state, target, cls)
    conflicted = {k for k, v in labels.items() if len(v) > 1}
    drops["conflicting_labels"] = len(conflicted)
    return [v for k, v in items.items() if k not in conflicted], dict(drops)


# Pool texts that overlap a Decision Index member the global blocklist does not hash
# (iSarcasmEval, VAST, SATA-Bench, FinEntity, ACOS, Humicroedit), found by an
# audit against the members' raw files.
# Hashes of `norm_key`, never text, so the file redistributes nothing of the suite.
DI_EXCLUDE_FILE = ASSETS / "sentiment_di_exclude.json"
_DI_EXCLUDE: set[str] | None = None


def di_hash(key: str) -> str:
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:20]


def di_excluded() -> set[str]:
    global _DI_EXCLUDE
    if _DI_EXCLUDE is None:
        try:
            _DI_EXCLUDE = set(json.loads(DI_EXCLUDE_FILE.read_text()))
        except (OSError, ValueError):
            _DI_EXCLUDE = set()
    return _DI_EXCLUDE


_POOLS: dict[str, list[Item]] = {}
DROPS: dict[str, dict[str, int]] = {}


def pools() -> dict[str, list[Item]]:
    """Every task's pool, with each text owned by the first task in PRIORITY holding it."""
    if not _POOLS:
        claimed: set[str] = set()
        for name in PRIORITY:
            spec = SPEC_BY_NAME[name]
            items, drops = raw_pool(spec)
            kept = [it for it in items if norm_key(it[0]) not in claimed]
            drops["claimed_by_other_task"] = len(items) - len(kept)
            claimed.update(norm_key(it[0]) for it in kept)
            _POOLS[name] = kept
            DROPS[name] = drops
    return _POOLS


def select(items: list[Item], n: int, task: str,
           cap: float = MAX_CLASS_SHARE) -> list[Item]:
    """n items in a deterministic hashed order. From a pool larger than n, no class
    (the source's own class, for a soft target) takes more than `cap` of the sample while
    another class still has items to give."""
    order = sorted(items, key=lambda it: _rank(task, it[0]))
    if len(order) <= n:
        return order
    # the first n in hashed order is a natural-distribution sample; a class over the cap
    # gives up its last items to the next ones in order from classes still under it
    limit = max(1, int(cap * n))
    taken: list[Item] = []
    count: dict[int, int] = defaultdict(int)
    for it in order:
        c = it[2]
        if count[c] < limit:
            taken.append(it)
            count[c] += 1
            if len(taken) >= n:
                break
    if len(taken) < n:             # the pool cannot satisfy the cap: fill naturally
        have = {id(it) for it in taken}
        taken += [it for it in order if id(it) not in have][: n - len(taken)]
    keep = {id(it) for it in taken}
    return [it for it in order if id(it) in keep]


# ------------------------------------------------------------------ tasks ------------

_BANK: PhrasingBank | None = None


def _phrase(spec: Spec, state: str) -> str:
    global _BANK
    if _BANK is None:
        _BANK = PhrasingBank.load()
    return _BANK.pick(TEMPLATE_ID.format(task=spec.task), spec.question,
                      hashlib.sha1(state.encode("utf-8")).hexdigest()[:16])


def examples_for(spec: Spec, n: int) -> Iterator[Example]:
    for state, target, cls in select(pools()[spec.name], n, spec.name):
        meta = None
        if spec.soft:
            # the class the source publishes for this text (SST: the bin of the mean
            # score; FPB: the majority label), kept beside the rater shares
            meta = {"source_label": spec.options[cls]}
        yield Example(task=spec.name, state=state,
                      questions=[Question(id=spec.task, question=_phrase(spec, state),
                                          options=list(spec.options),
                                          target=list(target) if isinstance(target, list)
                                          else int(target),
                                          meta=meta)])


def criteria_for(task: str, options: list[str]) -> list[str | None] | None:
    """Published option definitions, for `descriptions.for_task`."""
    spec = SPEC_BY_NAME.get(task)
    if spec is None or list(options) != list(spec.options):
        return None
    return list(spec.criteria)


def tasks() -> list[RealTask]:
    """Empty until `scripts/fetch_data.py --rows 44` has stored the rows."""
    out = []
    for spec in SPECS:
        if not store.has(spec.key):
            continue
        out.append(RealTask(
            row=ROW, name=spec.name, licence=spec.licence, url=spec.url,
            load=(lambda s: (lambda n: examples_for(s, n)))(spec),
            ordinal=spec.ordinal, soft=spec.soft, force_split=spec.split,
            notes=spec.notes))
    return out
