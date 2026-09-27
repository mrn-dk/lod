"""Row 46 -- multilingual classification (domain 21).

Human-labelled classification in the source language, asked in English. Six published
datasets, each read per language, one task per (dataset, label, language):

| source | HF id | labels | licence |
|---|---|---|---|
| MASSIVE intent / scenario | `mteb/amazon_massive_{intent,scenario}` | 60 intents, 18 scenarios | CC BY 4.0 (MASSIVE) |
| SIB-200 | `Davlan/sib200` | 7 topics | CC BY-SA 4.0 |
| Tweet Sentiment Multilingual | `cardiffnlp/tweet_sentiment_multilingual` | neg/neu/pos | CC BY 3.0 + Twitter ToS |
| AfriSenti | `masakhane/afrisenti` | neg/neu/pos | CC BY 4.0 |
| PAWS-X | `google-research-datasets/paws-x` | paraphrase y/n | free for any use (Google) |
| XNLI | `facebook/xnli` | NLI 3-way | CC BY-NC 4.0 |

**Splits measure transfer, twice over.**

* Whole *languages* are held out. `TEST_LANGS` (Greek, Thai, Yoruba, Georgian) go to
  `testreal` in every source that has them and train nowhere; `DEV_LANGS` (Armenian) goes
  to `devreal`. Chosen because the rest of the corpus does not contain them either:
  measured on an earlier build's train+val, a state with >= 20 characters of the script
  occurs 5 times for Thai, once for Georgian and Khmer, 0 for Armenian; Greek *words* 4
  times (the 506 Greek-script hits are maths symbols in arXiv/entityres); Yoruba 8 fuzzy
  hits, mostly Vietnamese. Korean was the obvious fourth and is not held out: 476 train
  states are Korean GitHub issues (row 5).
* Whole *datasets* are held out: PAWS-X to `testreal`, XNLI to `devreal`, every language.

**Parallel corpora.** MASSIVE and SIB-200 are translations of one English set, so the
Greek utterance with MASSIVE id 17 *is* the German one with id 17. A held-out language
therefore reads the source's own test (or validation) split, whose ids no training
language has -- otherwise testreal would be translations of sentences trained twenty
times, and English MASSIVE (row 16) already trains the whole English train split. The
training languages read train splits only. This is the only place a test split is read,
and only ever for an eval split (a test split is never read for training). PAWS-X and XNLI
read their *human-translated* test / validation splits: both publish a machine-translated
training set, which is not used anywhere.

**No shared state.** MASSIVE's intent and scenario tasks of one language take disjoint
ids (a hash of the id), a text two tasks could both claim goes to the first in
`task_order()` -- train tasks first, so an eval task never repeats a train text -- and a
text the source labels two ways within one language is dropped. English is never read:
row 16 has MASSIVE English, the row-1 sweep has tweet_eval.

Question wording is a code template (`lod/assets/phrasing_specs/d21.json`), protected
from enrichment by the `ml_` prefix; the option criteria are `criteria_for`, attached to
half the examples by `descriptions.attach`.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterator

from lod.phrasings import PhrasingBank
from lod.schema import Example, Question
from lod.corpus.repositories import raw_store as store
from lod.corpus.services.sources.real.base import RealTask

ROW = 46
PREFIX = "ml_"

# --------------------------------------------------------------------- languages
# code -> (name, script, family). Codes are ISO 639-1 where one exists.
LANGS: dict[str, tuple[str, str, str]] = {
    "ar": ("Arabic", "Arabic", "Afro-Asiatic, Semitic"),
    "am": ("Amharic", "Ge'ez", "Afro-Asiatic, Semitic"),
    "he": ("Hebrew", "Hebrew", "Afro-Asiatic, Semitic"),
    "ha": ("Hausa", "Latin", "Afro-Asiatic, Chadic"),
    "de": ("German", "Latin", "Indo-European, Germanic"),
    "es": ("Spanish", "Latin", "Indo-European, Romance"),
    "fr": ("French", "Latin", "Indo-European, Romance"),
    "pt": ("Portuguese", "Latin", "Indo-European, Romance"),
    "ru": ("Russian", "Cyrillic", "Indo-European, Slavic"),
    "hi": ("Hindi", "Devanagari", "Indo-European, Indo-Aryan"),
    "zh": ("Chinese (Mandarin, simplified)", "Han", "Sino-Tibetan"),
    "ja": ("Japanese", "Kanji/kana", "Japonic"),
    "ko": ("Korean", "Hangul", "Koreanic"),
    "tr": ("Turkish", "Latin", "Turkic"),
    "fi": ("Finnish", "Latin", "Uralic"),
    "vi": ("Vietnamese", "Latin", "Austroasiatic"),
    "id": ("Indonesian", "Latin", "Austronesian"),
    "sw": ("Swahili", "Latin", "Niger-Congo, Bantu"),
    "ig": ("Igbo", "Latin", "Niger-Congo, Igboid"),
    "rw": ("Kinyarwanda", "Latin", "Niger-Congo, Bantu"),
    # held out whole -- never in training, in any source of this row
    "el": ("Greek", "Greek", "Indo-European, Hellenic"),
    "th": ("Thai", "Thai", "Kra-Dai"),
    "yo": ("Yoruba", "Latin", "Niger-Congo, Yoruboid"),
    "ka": ("Georgian", "Georgian", "Kartvelian"),
    "hy": ("Armenian", "Armenian", "Indo-European, Armenian"),
    # XNLI's own languages, read only inside its devreal holdout
    "bg": ("Bulgarian", "Cyrillic", "Indo-European, Slavic"),
    "ur": ("Urdu", "Arabic", "Indo-European, Indo-Aryan"),
}
TEST_LANGS = ("el", "th", "yo", "ka")
DEV_LANGS = ("hy",)


def lang_split(lang: str) -> str:
    if lang in TEST_LANGS:
        return "testreal"
    if lang in DEV_LANGS:
        return "devreal"
    return "train"


# --------------------------------------------------------------------- label sets
MASSIVE_INTENTS: dict[str, str] = {
    "alarm_query": "asks which alarms are set or when one will go off",
    "alarm_remove": "asks to cancel or delete an alarm",
    "alarm_set": "asks to set a new alarm",
    "audio_volume_down": "asks to turn the volume down",
    "audio_volume_mute": "asks to mute the device or be quiet",
    "audio_volume_other": "any other request about the volume, e.g. a specific level",
    "audio_volume_up": "asks to turn the volume up",
    "calendar_query": "asks what is on the calendar, or about a meeting or reminder",
    "calendar_remove": "asks to delete or cancel a calendar event or reminder",
    "calendar_set": "asks to add an event, meeting or reminder to the calendar",
    "cooking_query": "a cooking question that is not a request for a recipe",
    "cooking_recipe": "asks for a recipe or how to cook or prepare something",
    "datetime_convert": "asks to convert a time between time zones or formats",
    "datetime_query": "asks the current time or date, or the date of something",
    "email_addcontact": "asks to add a person or address to the contacts",
    "email_query": "asks to check or read emails",
    "email_querycontact": "asks for a contact's details, such as a number or address",
    "email_sendemail": "asks to write, send or reply to an email",
    "general_greet": "a greeting or asks how the assistant is",
    "general_joke": "asks for a joke",
    "general_quirky": "chit-chat or an off-task request that fits no other intent",
    "iot_cleaning": "asks the robot vacuum or cleaner to clean",
    "iot_coffee": "asks the coffee machine to make coffee",
    "iot_hue_lightchange": "asks to change the colour or scene of the lights",
    "iot_hue_lightdim": "asks to dim the lights",
    "iot_hue_lightoff": "asks to turn the lights off",
    "iot_hue_lighton": "asks to turn the lights on",
    "iot_hue_lightup": "asks to make the lights brighter",
    "iot_wemo_off": "asks to switch off a smart plug or the device on it",
    "iot_wemo_on": "asks to switch on a smart plug or the device on it",
    "lists_createoradd": "asks to create a list or add an item to one",
    "lists_query": "asks what is on a list, or which lists exist",
    "lists_remove": "asks to delete a list or remove an item from one",
    "music_dislikeness": "says the user dislikes a song, artist or genre",
    "music_likeness": "says the user likes a song, artist or genre, or asks to save it",
    "music_query": "asks about music, e.g. what song or artist is playing",
    "music_settings": "asks to change playback settings such as shuffle or repeat",
    "news_query": "asks for news or headlines",
    "play_audiobook": "asks to play or resume an audiobook",
    "play_game": "asks to start or play a game",
    "play_music": "asks to play music, a song, an artist or a playlist",
    "play_podcasts": "asks to play a podcast or an episode",
    "play_radio": "asks to play a radio station",
    "qa_currency": "asks about currency or an exchange rate",
    "qa_definition": "asks what a word means or how it is defined",
    "qa_factoid": "asks a factual question about the world",
    "qa_maths": "asks a calculation or maths question",
    "qa_stock": "asks about a stock or share price",
    "recommendation_events": "asks for events or things to do nearby",
    "recommendation_locations": "asks for a place to go, such as a restaurant or shop",
    "recommendation_movies": "asks for a film recommendation",
    "social_post": "asks to post on social media or send a complaint to a company",
    "social_query": "asks about social-media updates or activity",
    "takeaway_order": "asks to order takeaway food",
    "takeaway_query": "asks about a takeaway order or whether a place delivers",
    "transport_query": "asks about travel, routes or public-transport times",
    "transport_taxi": "asks to book or call a taxi",
    "transport_ticket": "asks to book a train or other travel ticket",
    "transport_traffic": "asks about traffic conditions",
    "weather_query": "asks about the weather",
}
MASSIVE_SCENARIOS: dict[str, str] = {
    "alarm": "alarms: setting, checking or removing them",
    "audio": "the device's volume",
    "calendar": "calendar events, meetings and reminders",
    "cooking": "recipes and cooking",
    "datetime": "the date or time, and time conversions",
    "email": "emails and contacts",
    "general": "greetings, jokes and chit-chat",
    "iot": "smart-home devices: lights, plugs, vacuum cleaner, coffee machine",
    "lists": "to-do and shopping lists",
    "music": "music preferences, information and playback settings (not starting playback)",
    "news": "news and headlines",
    "play": "starting media: music, radio, podcasts, audiobooks or games",
    "qa": "factual questions, definitions, maths, currency and stock prices",
    "recommendation": "recommendations of events, places or films",
    "social": "social media",
    "takeaway": "ordering or asking about takeaway food",
    "transport": "travel, taxis, tickets and traffic",
    "weather": "the weather",
}
SIB_TOPICS: dict[str, str] = {
    "entertainment": "entertainment, media, the arts, celebrities or leisure",
    "geography": "places, countries, landforms, climate or other geographic facts",
    "health": "health, medicine, disease or the body",
    "politics": "politics, government, elections, law or international affairs",
    "science/technology": "science, research, the natural world or technology",
    "sports": "sport, athletes or competitions",
    "travel": "travelling, tourism or advice for travellers",
}
# SemEval-2017 Task 4 / AfriSenti (Mohammad 2016, "A practical guide to sentiment
# annotation"): the sentiment the author expresses, not the topic's.
SENTIMENT: dict[str, str] = {
    "negative": "the author expresses a negative opinion, emotion or attitude "
                "(criticism, anger, sadness, complaint)",
    "neutral": "no positive or negative opinion or emotion is expressed, e.g. a plain "
               "statement of fact or news",
    "positive": "the author expresses a positive opinion, emotion or attitude "
                "(praise, joy, support, gratitude)",
}
# PAWS / PAWS-X (Zhang et al. 2019; Yang et al. 2019): a paraphrase means the same thing;
# the non-paraphrases share nearly every word and differ by a swap.
PARAPHRASE: dict[str, str] = {
    "no": "the two sentences do not mean the same thing -- e.g. roles, places or names "
          "are swapped -- even though they share almost every word",
    "yes": "the two sentences mean the same thing (they are paraphrases), however they "
           "are worded or ordered",
}
# XNLI followed the MultiNLI instructions: the hypothesis is definitely / might be /
# definitely not a true description of the situation in the premise.
NLI: dict[str, str] = {
    "contradiction": "the hypothesis is definitely false given the premise",
    "entailment": "the hypothesis is definitely true given the premise",
    "neutral": "the hypothesis might be true or false; the premise does not settle it",
}


# --------------------------------------------------------------------- sources
@dataclass(frozen=True)
class Source:
    name: str                    # task stem: ml_<name>__<lang>
    store_key: str
    hf: str
    licence: str
    file_pattern: str            # formatted with code= and split=
    codes: dict[str, str]        # corpus language -> the file's language code
    reads: dict[str, tuple[str, ...]]   # corpus split -> the source splits it reads
    text_cols: tuple[str, ...]   # one -> plain text; several -> a JSON object
    label_col: str
    criteria: dict[str, str]     # option -> short criterion; key order is option order
    template_id: str
    question: str
    cap: int                     # questions per task at most
    label_names: tuple[str, ...] = ()   # int label -> name, where the file stores ints
    label_map: dict[str, str] = field(default_factory=dict)   # name -> option
    whole_split: str | None = None      # the whole dataset is held out to this split
    partition: tuple[int, int] | None = None   # (index, n): this task's share of ids
    # A parallel source whose dev and test languages read the same source splits: one id
    # in `dev_of_held` goes to the devreal language, the rest to the testreal ones, so the
    # split that selects checkpoints never holds a translation of a test item. 0 = off
    # (the source's dev and test reads are already disjoint, or it is not parallel).
    dev_of_held: int = 0
    id_col: str = ""
    ordinal: bool = False
    # at most this multiple of the runner-up label, where the raw distribution is extreme
    max_ratio: float | None = None
    notes: str = ""

    @property
    def options(self) -> list[str]:
        return list(self.criteria)

    def split_for(self, lang: str) -> str:
        return self.whole_split or lang_split(lang)

    def files_needed(self) -> list[tuple[str, str]]:
        return [(lang, sp) for lang in self.codes for sp in self.reads[self.split_for(lang)]]

    def task_name(self, lang: str) -> str:
        return f"{PREFIX}{self.name}__{lang}"


def _codes(pairs: str) -> dict[str, str]:
    return dict(p.split("=") for p in pairs.split())


MASSIVE_LANGS = ("ar de es hi ru zh ja ko tr vi id sw am fi he "   # train
                 "el th ka "                                        # testreal
                 "hy")                                              # devreal
MASSIVE_CODES = {lang: ("zh-CN" if lang == "zh" else lang) for lang in MASSIVE_LANGS.split()}
MASSIVE_READS = {"train": ("train",), "devreal": ("validation",), "testreal": ("test",)}
SIB_CODES = _codes("ar=arb_Arab de=deu_Latn es=spa_Latn fr=fra_Latn pt=por_Latn "
                   "hi=hin_Deva ru=rus_Cyrl zh=zho_Hans ja=jpn_Jpan ko=kor_Hang "
                   "tr=tur_Latn vi=vie_Latn id=ind_Latn fi=fin_Latn he=heb_Hebr "
                   "sw=swh_Latn am=amh_Ethi ha=hau_Latn ig=ibo_Latn rw=kin_Latn "
                   "el=ell_Grek th=tha_Thai yo=yor_Latn ka=kat_Geor hy=hye_Armn")

SOURCES: tuple[Source, ...] = (
    Source("massive_intent", "ml46_massive_intent", "mteb/amazon_massive_intent",
           "CC-BY-4.0", "{split}/{code}.json.gz", MASSIVE_CODES, MASSIVE_READS,
           ("text",), "label", MASSIVE_INTENTS, "d21.massive_intent",
           "Which intent does this request to a voice assistant express?",
           cap=400, partition=(0, 2), id_col="id",
           notes="MASSIVE 1.1 (FitzGerald et al. 2022): SLURP's English utterances "
                 "localised by professional translators; intent label from SLURP"),
    Source("massive_scenario", "ml46_massive_scenario", "mteb/amazon_massive_scenario",
           "CC-BY-4.0", "{split}/{code}.json.gz", MASSIVE_CODES, MASSIVE_READS,
           ("text",), "label", MASSIVE_SCENARIOS, "d21.massive_scenario",
           "Which scenario does this request to a voice assistant belong to?",
           cap=250, partition=(1, 2), id_col="id",
           notes="MASSIVE scenario (the intent's domain); ids disjoint from the intent "
                 "task of the same language"),
    Source("sib200_topic", "ml46_sib200", "Davlan/sib200", "CC-BY-SA-4.0",
           "data/{code}/{split}.tsv", SIB_CODES,
           {"train": ("train",), "devreal": ("dev", "test"), "testreal": ("dev", "test")},
           ("text",), "category", SIB_TOPICS, "d21.sib200_topic",
           "What topic is this sentence about?", cap=300, id_col="index_id",
           # dev (Armenian) and test (Greek, Thai, Yoruba, Georgian) both read dev+test,
           # and Flores is parallel: 297 of devreal's 300 Armenian items were
           # translations of the testreal items, with the same labels. One id in four
           # now goes to devreal, the rest to testreal (~225 per test language, was 300).
           dev_of_held=4,
           notes="SIB-200 (Adelani et al. 2024): Flores-200 sentences, topic annotated "
                 "on the English source; human translations; held-out ids split between "
                 "the devreal and testreal languages"),
    Source("tweet_sentiment", "ml46_tweet_sentiment", "cardiffnlp/tweet_sentiment_multilingual",
           "CC-BY-3.0 (SemEval tweets) + Twitter ToS",
           "data/{code}/{split}.jsonl",
           # no Hindi: its 1,839 tweets are romanised Hinglish, and a third read as
           # plain English (audit: 0/500 in Devanagari, LID English 0.33)
           _codes("ar=arabic de=german es=spanish fr=french pt=portuguese"),
           {"train": ("train",)}, ("text",), "label", SENTIMENT, "d21.sentiment",
           "What sentiment does the author of this tweet express?", cap=500,
           label_names=("negative", "neutral", "positive"), ordinal=True,
           notes="UMSAB / XLM-T (Barbieri et al. 2022): per-language human-annotated "
                 "tweet sentiment sets, balanced by the curators"),
    Source("afrisenti", "ml46_afrisenti", "masakhane/afrisenti", "CC-BY-4.0",
           "data/{code}/{split}.tsv",
           _codes("am=amh ha=hau ig=ibo rw=kin sw=swa pt=por yo=yor"),
           {"train": ("train",), "testreal": ("train",)}, ("tweet",), "label", SENTIMENT,
           "d21.sentiment", "What sentiment does the author of this tweet express?",
           cap=500, ordinal=True, max_ratio=2.0,
           notes="AfriSenti-SemEval 2023 (Muhammad et al. 2023): tweets labelled by "
                 "three native-speaker annotators, majority vote; not parallel, so the "
                 "held-out Yoruba reads its train split"),
    Source("pawsx_paraphrase", "ml46_pawsx", "google-research-datasets/paws-x",
           "PAWS-X terms: free for any purpose, acknowledge Google",
           "{code}/{split}-00000-of-00001.parquet",
           _codes("de=de es=es fr=fr ja=ja ko=ko zh=zh"),
           {"testreal": ("test",)}, ("sentence1", "sentence2"), "label", PARAPHRASE,
           "d21.pawsx_paraphrase",
           "Do sentence1 and sentence2 mean the same thing?", cap=350,
           label_names=("no", "yes"), whole_split="testreal", id_col="id",
           notes="PAWS-X (Yang et al. 2019): the human-translated test split; the "
                 "machine-translated train split is not used. Held out whole"),
    Source("xnli", "ml46_xnli", "facebook/xnli", "CC-BY-NC-4.0",
           "{code}/{split}-00000-of-00001.parquet",
           _codes("ar=ar bg=bg de=de es=es fr=fr hi=hi ru=ru sw=sw tr=tr ur=ur vi=vi "
                  "zh=zh"),
           {"devreal": ("validation",)}, ("premise", "hypothesis"), "label", NLI,
           "d21.xnli", "What is the relationship between the premise and the hypothesis?",
           cap=200, label_names=("entailment", "neutral", "contradiction"),
           whole_split="devreal",
           notes="XNLI (Conneau et al. 2018): new MultiNLI-style pairs, human-translated "
                 "validation split; the machine-translated train split is not used. "
                 "Greek and Thai left out (held-out languages). Held out whole"),
)
BY_NAME = {s.name: s for s in SOURCES}


# --------------------------------------------------------------------- reading
def _h(*parts) -> int:
    return int.from_bytes(hashlib.sha1(":".join(map(str, parts)).encode()).digest()[:8], "big")


def _norm(text: str) -> str:
    return " ".join(str(text or "").split())


def state_hash(state: str) -> str:
    return hashlib.sha1(state.encode("utf-8")).hexdigest()[:16]


# Written by scripts/fetch_data.py --rows 46 beside the rows: states that match a Decision
# Index item under Unicode tokenisation, which the build's [a-z0-9]+ filter cannot see.
DI_EXCLUDE = "ml46_di_exclude.json"
_EXCLUDED: set[str] | None = None


def excluded() -> set[str]:
    global _EXCLUDED
    if _EXCLUDED is None:
        p = store.RAW / DI_EXCLUDE
        _EXCLUDED = set(json.loads(p.read_text())) if p.exists() else set()
    return _EXCLUDED


def raw_label(src: Source, row: dict) -> str | None:
    """The row's label as an option string, or None when it is not one."""
    v = row.get(src.label_col)
    if v is None:
        return None
    s = str(v).strip()
    if src.label_names and s.lstrip("-").isdigit():
        i = int(s)
        if not 0 <= i < len(src.label_names):
            return None                  # -1 and friends: no label
        s = src.label_names[i]
    s = src.label_map.get(s, s)
    return s if s in src.criteria else None


def state_of(src: Source, row: dict) -> str:
    if len(src.text_cols) == 1:
        return _norm(row.get(src.text_cols[0]))[:4000]
    fields = {c: _norm(row.get(c))[:4000] for c in src.text_cols}
    if not all(fields.values()):
        return ""
    return json.dumps(fields, ensure_ascii=False, indent=2)


_ROWS: dict[str, list[dict]] = {}


def rows_of(src: Source) -> list[dict]:
    if src.store_key not in _ROWS:
        _ROWS[src.store_key] = store.load(src.store_key)
    return _ROWS[src.store_key]


def candidates(src: Source, lang: str) -> list[tuple[str, str, dict]]:
    """(state, option, raw row) this task may draw from, before claims and the cap.

    The language's rows from the splits its corpus split reads; the task's id partition;
    no text the source labels two ways within the language (over every fetched split);
    no state `excluded()` matched to a Decision Index item.
    """
    code = src.codes[lang]
    reads = set(src.reads[src.split_for(lang)])
    rows = [r for r in rows_of(src) if r.get("_lang") == code]
    labels: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        lab = raw_label(src, r)
        st = state_of(src, r)
        if lab and st:
            labels[st].add(lab)
    drop = excluded()
    out = []
    for r in rows:
        if r.get("_split") not in reads:
            continue
        if src.partition and _h("part", r.get(src.id_col)) % src.partition[1] != src.partition[0]:
            continue
        if src.dev_of_held and src.split_for(lang) != "train" and (
                (_h("held", r.get(src.id_col)) % src.dev_of_held == 0)
                != (src.split_for(lang) == "devreal")):
            continue
        st = state_of(src, r)
        lab = raw_label(src, r)
        if not st or not lab or len(labels[st]) > 1 or state_hash(st) in drop:
            continue
        out.append((st, lab, r))
    return out


def plan() -> list[tuple[Source, str]]:
    """Every (source, language) task this row defines."""
    return [(src, lang) for src in SOURCES for lang in src.codes]


def task_order() -> list[tuple[Source, str]]:
    """Claim order: train, then devreal, then testreal; so eval never repeats train."""
    rank = {"train": 0, "devreal": 1, "testreal": 2}
    return sorted(plan(), key=lambda p: (rank[p[0].split_for(p[1])], p[0].name, p[1]))


_CLAIMS: dict[str, str] | None = None


def claims() -> dict[str, str]:
    """state -> the one task allowed to use it. Computed once over every task's pool."""
    global _CLAIMS
    if _CLAIMS is None:
        owner: dict[str, str] = {}
        for src, lang in task_order():
            if not store.has(src.store_key):
                continue
            for st, _, _ in candidates(src, lang):
                owner.setdefault(st, src.task_name(lang))
        _CLAIMS = owner
    return _CLAIMS


def select(src: Source, lang: str, n: int) -> list[tuple[str, str, dict]]:
    """This task's examples: its own states, hash-ordered, capped, label-balanced."""
    name = src.task_name(lang)
    owner = claims()
    seen: set[str] = set()
    pool = []
    for st, lab, r in candidates(src, lang):
        if owner.get(st) != name or st in seen:
            continue
        seen.add(st)
        pool.append((st, lab, r))
    pool.sort(key=lambda x: _h(name, x[0]))
    limit = min(n, src.cap)
    if src.max_ratio:
        counts = Counter(lab for _, lab, _ in pool)
        ranked = sorted(counts.values(), reverse=True)
        if len(ranked) >= 2:
            # the runner-up share of what will be taken, if the pool held it
            share = ranked[1] / sum(ranked)
            per_label = max(1, int(src.max_ratio * max(share * limit, 1)))
            taken: Counter = Counter()
            kept = []
            for item in pool:
                if taken[item[1]] < per_label:
                    taken[item[1]] += 1
                    kept.append(item)
            pool = kept
    return pool[:limit]


_BANK: PhrasingBank | None = None


def question_text(src: Source, key: str) -> str:
    global _BANK
    if _BANK is None:
        _BANK = PhrasingBank.load()
    return _BANK.pick(src.template_id, src.question, key)


def _loader(src: Source, lang: str):
    def load(n: int) -> Iterator[Example]:
        name = src.task_name(lang)
        for st, lab, r in select(src, lang, n):
            ident = r.get(src.id_col, "") if src.id_col else ""
            yield Example(
                task=name,
                state=st,
                questions=[Question(
                    id=src.name,
                    question=question_text(src, f"{name}:{st}"),
                    options=src.options,
                    target=src.options.index(lab),
                    meta={"lang": lang, "source": src.hf, "source_split": r.get("_split"),
                          **({"source_id": str(ident)} if ident != "" else {})})],
            )
    return load


def criteria_for(task: str, options: list[str]) -> list[str | None] | None:
    """Short option criteria for an `ml_` task, in option order; None if not ours."""
    if not task.startswith(PREFIX):
        return None
    stem = task[len(PREFIX):].split("__", 1)[0]
    src = BY_NAME.get(stem)
    if src is None or list(options) != src.options:
        return None
    return [src.criteria[o] for o in options]


def tasks() -> list[RealTask]:
    """One task per (source, language) whose raw rows are in the store; [] offline."""
    out: list[RealTask] = []
    for src, lang in plan():
        if not store.has(src.store_key):
            continue
        name, script, family = LANGS[lang]
        split = src.split_for(lang)
        out.append(RealTask(
            row=ROW, name=src.task_name(lang), licence=src.licence,
            url=f"https://huggingface.co/datasets/{src.hf}",
            load=_loader(src, lang),
            ordinal=src.ordinal,
            force_split=split,
            min_quota=src.cap,
            notes=f"{name} ({script}; {family}); {src.notes}; "
                  f"reads {'+'.join(src.reads[split])}",
        ))
    return out
