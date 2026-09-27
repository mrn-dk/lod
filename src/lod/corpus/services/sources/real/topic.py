"""Source-table row 45 -- English topic, intent and question-type classification.

Domain 1 had these label sets only through the row-1 Hub sweep, which gives a task about
89 questions (`fancyzhx_ag_news__label`, `cogcomp_trec__*`, `community_datasets_yahoo_
answers_topics__topic`), and read two of them wrongly: DBpedia-14's 2,000 cached rows
were all `Company` (the file is sorted by class), and MTOP's labels arrived as bare
integers and were gated out. This module reads them properly, from a sample of each
*train* split fetched by `scripts/fetch_data.py --rows 45`:

| task | source | options | split |
|---|---|---|---|
| `topic_dbpedia14` | DBpedia-14 (Zhang et al. 2015), Wikipedia abstracts | 14 ontology classes | train |
| `topic_mtop_domain` | MTOP English (Li et al. 2021) | 11 domains | train |
| `topic_mtop_intent` | MTOP English | 81 top-level intents | train |
| `topic_cfpb_product` | CFPB Consumer Complaint Database, 2018-2022 | 9 products | train |
| `topic_agnews` | AG News (Zhang et al. 2015) | 4 sections | devreal |
| `topic_yahoo` | Yahoo! Answers Topics (Zhang et al. 2015) | 10 categories | testreal |
| `topic_trec_coarse` | TREC question classification (Li & Roth 2002) | 6 answer types | testreal |
| `topic_trec_fine` | TREC question classification | 50 answer types | testreal |

Whole datasets are held out: AG News to devreal, Yahoo and both TREC schemas to testreal.
The two TREC tasks and the two MTOP tasks each split their dataset's texts between them,
so no text is asked twice.

What each loader removes, because it states the label rather than being evidence for it:

* DBpedia: the entity **title** is not in the state. It carries Wikipedia's
  disambiguator -- "(film)", "(album)", "(novel)", "(politician)" -- and 1,610 of the
  9,800 stored titles (16.4 %) end in one, which is the answer or most of it in
  brackets. The abstract stays as written (a leading hatnote, 0.4 %, names a different
  sense and is kept).
* AG News: the wire-service attribution. "(Sports Network) -", "(Ticker) --",
  "(SPACE.com) SPACE.com -", "(PC World)", "(Quote, Profile, Research)" are feed
  metadata, and several are one section only (Sports Network 75/75 Sports, SPACE.com
  11/11 Sci/Tech in 8,000 rows). HTML debris (`#39;`, `quot;`, `<b>`) is decoded.
* MTOP: two intents a reader cannot tell from a sibling are dropped with their
  utterances -- `UPDATE_REMINDER` (14 rows; "update reminder for fish market grand
  opening from 5-10 pm" is a date-time update) against `UPDATE_REMINDER_DATE_TIME` /
  `_TODO`, and `GET_DETAILS_NEWS` (13 rows) against `GET_STORIES_NEWS` -- as is
  `PLAY_MEDIA` (10 rows, "find me Ed Sheeran" against `PLAY_MUSIC`). Intents with fewer
  than 10 training utterances (29 of 113, 125 rows together) are not options: ten
  examples are the least that pins down what a label name means.
* CFPB: only complaints received 2018-2022, the window in which the product taxonomy did
  not change (2017 mixes the old and new names, 2023 renames credit reporting), so no
  two options are the same product under two names. Company, sub-product and issue are
  not in the state.

Sampling fills each task's quota class by class, the largest classes capped first
("water-filling"), so CFPB's credit-reporting majority (56 % of 2018-2022 complaints)
and MTOP's long tail do not decide the task. A text the source labels two ways is
dropped; identical texts are kept once.

Questions are code templates (`lod/assets/phrasing_specs/topic.json`) and the tasks
are protected from enrichment (`topic_` in `PROTECTED_PREFIXES`); option criteria are
`CRITERIA` below, attached to half the examples by `descriptions.py`.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Callable, Iterator

from lod.schema import Example, Question
from lod.corpus.repositories import raw_store as store
from lod.corpus.services.sources.real.base import RealTask

ROW = 45
PREFIX = "topic_"
STATE_CHARS = 4000


@dataclass(frozen=True)
class Source:
    task: str
    key: str                  # raw-store key, written by scripts/fetch_data.py
    template_id: str          # phrasing-bank id
    question: str
    licence: str
    url: str
    split: str
    notes: str


SOURCES: tuple[Source, ...] = (
    Source("topic_dbpedia14", "v14_topic_dbpedia_14", "v14_topic.dbpedia14",
           "Which DBpedia ontology class does the entity this Wikipedia abstract "
           "describes belong to?",
           "CC-BY-SA-3.0 (Wikipedia / DBpedia abstracts)",
           "https://huggingface.co/datasets/fancyzhx/dbpedia_14", "train",
           "DBpedia-14; abstract only, the entity title (with its Wikipedia "
           "disambiguator) is not shown"),
    Source("topic_mtop_domain", "v14_topic_mtop_en", "v14_topic.mtop_domain",
           "Which domain of a virtual assistant does this request belong to?",
           "CC-BY-SA-4.0 (MTOP release, LICENSE.txt)",
           "https://dl.fbaipublicfiles.com/mtop/mtop.zip", "train",
           "MTOP English train; utterances disjoint from topic_mtop_intent"),
    Source("topic_mtop_intent", "v14_topic_mtop_en", "v14_topic.mtop_intent",
           "What is the top-level intent of this request to a virtual assistant?",
           "CC-BY-SA-4.0 (MTOP release, LICENSE.txt)",
           "https://dl.fbaipublicfiles.com/mtop/mtop.zip", "train",
           "MTOP English train; intents with >= 10 train utterances, three "
           "indistinguishable ones dropped"),
    Source("topic_cfpb_product", "v14_topic_cfpb", "v14_topic.cfpb_product",
           "Which financial product did the consumer file this complaint about?",
           "public domain (US Government, CFPB Consumer Complaint Database; Hub mirror "
           "CC0-1.0)",
           "https://huggingface.co/datasets/BEE-spoke-data/consumer-finance-complaints",
           "train",
           "complaint narratives received 2018-2022 (one product taxonomy); product "
           "as the consumer filed it; balanced"),
    Source("topic_agnews", "v14_topic_ag_news", "v14_topic.agnews",
           "Which news section does this article belong to?",
           "AG's corpus of news articles: free for non-commercial research use "
           "(Hub card: unknown); devreal only",
           "https://huggingface.co/datasets/fancyzhx/ag_news", "devreal",
           "AG News title + description; wire-service attributions stripped"),
    Source("topic_yahoo", "v14_topic_yahoo_answers", "v14_topic.yahoo",
           "Which Yahoo! Answers category was this question posted in?",
           "Yahoo! Answers Comprehensive Q&A v1.0 (Webscope L6), research use "
           "(Hub card: unknown); testreal only",
           "https://huggingface.co/datasets/community-datasets/yahoo_answers_topics",
           "testreal",
           "question title, details and best answer; top-level category"),
    Source("topic_trec_coarse", "v14_topic_trec", "v14_topic.trec_coarse",
           "What type of answer is this question asking for?",
           "no explicit licence; distributed freely by CogComp for research "
           "(Li & Roth 2002); testreal only",
           "https://huggingface.co/datasets/CogComp/trec", "testreal",
           "TREC train questions, coarse answer type; disjoint from topic_trec_fine"),
    Source("topic_trec_fine", "v14_topic_trec", "v14_topic.trec_fine",
           "What specific type of answer is this question asking for?",
           "no explicit licence; distributed freely by CogComp for research "
           "(Li & Roth 2002); testreal only",
           "https://huggingface.co/datasets/CogComp/trec", "testreal",
           "TREC train questions, fine answer type (50); disjoint from "
           "topic_trec_coarse"),
)
BY_TASK = {s.task: s for s in SOURCES}

# ---- label sets and criteria --------------------------------------------------------

AG_NEWS = ("World", "Sports", "Business", "Sci/Tech")
DBPEDIA = ("Company", "EducationalInstitution", "Artist", "Athlete", "OfficeHolder",
           "MeanOfTransportation", "Building", "NaturalPlace", "Village", "Animal",
           "Plant", "Album", "Film", "WrittenWork")
YAHOO = ("Society & Culture", "Science & Mathematics", "Health", "Education & Reference",
         "Computers & Internet", "Sports", "Business & Finance", "Entertainment & Music",
         "Family & Relationships", "Politics & Government")
TREC_COARSE = ("ABBR", "ENTY", "DESC", "HUM", "LOC", "NUM")
TREC_FINE = (
    "ABBR:abb", "ABBR:exp", "ENTY:animal", "ENTY:body", "ENTY:color", "ENTY:cremat",
    "ENTY:currency", "ENTY:dismed", "ENTY:event", "ENTY:food", "ENTY:instru", "ENTY:lang",
    "ENTY:letter", "ENTY:other", "ENTY:plant", "ENTY:product", "ENTY:religion",
    "ENTY:sport", "ENTY:substance", "ENTY:symbol", "ENTY:techmeth", "ENTY:termeq",
    "ENTY:veh", "ENTY:word", "DESC:def", "DESC:desc", "DESC:manner", "DESC:reason",
    "HUM:gr", "HUM:ind", "HUM:title", "HUM:desc", "LOC:city", "LOC:country", "LOC:mount",
    "LOC:other", "LOC:state", "NUM:code", "NUM:count", "NUM:date", "NUM:dist",
    "NUM:money", "NUM:ord", "NUM:other", "NUM:period", "NUM:perc", "NUM:speed",
    "NUM:temp", "NUM:volsize", "NUM:weight")
MTOP_DOMAINS = ("alarm", "calling", "event", "messaging", "music", "news", "people",
                "recipes", "reminder", "timer", "weather")
CFPB_PRODUCTS = (
    "Checking or savings account",
    "Credit card or prepaid card",
    "Credit reporting, credit repair services, or other personal consumer reports",
    "Debt collection",
    "Money transfer, virtual currency, or money service",
    "Mortgage",
    "Payday loan, title loan, or personal loan",
    "Student loan",
    "Vehicle loan or lease",
)
CFPB_YEARS = ("2018", "2019", "2020", "2021", "2022")

# MTOP: an intent needs this many English train utterances to be an option at all
MIN_INTENT = 10
# ...and these cannot be told from a sibling intent by reading the utterance
DROP_INTENTS = {
    "UPDATE_REMINDER": "generic update; its utterances are date-time or to-do updates",
    "GET_DETAILS_NEWS": "indistinguishable from GET_STORIES_NEWS",
    "PLAY_MEDIA": "'find me <artist>' -- indistinguishable from PLAY_MUSIC",
}
# an intent this rare goes wholly to the intent task's half of the pool
MTOP_RARE = 60

# Li & Roth (2002), "Learning Question Classifiers", Table 1: the published definition of
# every class, as the annotators applied it.
TREC_FINE_CRITERIA = {
    "ABBR:abb": "an abbreviation", "ABBR:exp": "the expression an abbreviation stands for",
    "ENTY:animal": "animals", "ENTY:body": "organs of the body",
    "ENTY:color": "colours",
    "ENTY:cremat": "inventions, books and other creative works",
    "ENTY:currency": "currency names", "ENTY:dismed": "diseases and medicine",
    "ENTY:event": "events", "ENTY:food": "food", "ENTY:instru": "musical instruments",
    "ENTY:lang": "languages", "ENTY:letter": "letters, such as a-z",
    "ENTY:other": "other entities", "ENTY:plant": "plants", "ENTY:product": "products",
    "ENTY:religion": "religions", "ENTY:sport": "sports",
    "ENTY:substance": "elements and substances", "ENTY:symbol": "symbols and signs",
    "ENTY:techmeth": "techniques and methods", "ENTY:termeq": "equivalent terms",
    "ENTY:veh": "vehicles", "ENTY:word": "words with a special property",
    "DESC:def": "the definition of something", "DESC:desc": "a description of something",
    "DESC:manner": "the manner of an action", "DESC:reason": "reasons",
    "HUM:gr": "a group or organisation of persons", "HUM:ind": "an individual",
    "HUM:title": "the title of a person", "HUM:desc": "a description of a person",
    "LOC:city": "cities", "LOC:country": "countries", "LOC:mount": "mountains",
    "LOC:other": "other locations", "LOC:state": "states or provinces",
    "NUM:code": "postcodes or other codes", "NUM:count": "the number of something",
    "NUM:date": "dates", "NUM:dist": "linear measures", "NUM:money": "prices",
    "NUM:ord": "ranks", "NUM:other": "other numbers", "NUM:period": "how long something lasts",
    "NUM:perc": "fractions or percentages", "NUM:speed": "speed", "NUM:temp": "temperature",
    "NUM:volsize": "size, area and volume", "NUM:weight": "weight",
}

CRITERIA: dict[str, dict[str, str]] = {
    "topic_trec_coarse": {
        "ABBR": "an abbreviation, or what one stands for",
        "ENTY": "an entity: an animal, product, food, event, substance, term, colour, "
                "creative work and the like",
        "DESC": "a description or abstract concept: a definition, reason, manner or "
                "explanation",
        "HUM": "a human being: an individual, a group or organisation, a title or a "
               "description of a person",
        "LOC": "a location: a city, country, state, mountain or other place",
        "NUM": "a numeric value: a count, date, distance, price, percentage, period, "
               "speed, temperature, size or weight",
    },
    "topic_trec_fine": TREC_FINE_CRITERIA,
    "topic_agnews": {
        "World": "international and national news: politics, conflict, diplomacy, "
                 "government and world events",
        "Sports": "sport: games, results, teams, athletes and competitions",
        "Business": "business and the economy: companies, markets, earnings, trade, "
                    "oil and jobs",
        "Sci/Tech": "science and technology: computing, the internet, software, "
                    "telecoms, space and scientific research",
    },
    # DBpedia ontology classes, condensed from the ontology's own class hierarchy
    "topic_dbpedia14": {
        "Company": "a business or commercial organisation",
        "EducationalInstitution": "a school, college, university or other place of "
                                  "education",
        "Artist": "a person known for creative work: a musician, painter, actor, "
                  "writer or comedian",
        "Athlete": "a person who competes in a sport",
        "OfficeHolder": "a person who holds or held an office, usually a political or "
                        "public one",
        "MeanOfTransportation": "a vehicle or vehicle type: a ship, aircraft, "
                                "automobile, locomotive or spacecraft",
        "Building": "a building or other built structure",
        "NaturalPlace": "a natural geographic feature: a mountain, river, lake, cave "
                        "or glacier",
        "Village": "a small human settlement",
        "Animal": "an animal species or other animal taxon",
        "Plant": "a plant species or other plant taxon",
        "Album": "a music album",
        "Film": "a film",
        "WrittenWork": "a written work: a book, novel, periodical, play or poem",
    },
    "topic_yahoo": {
        "Society & Culture": "religion, customs, languages, holidays, etiquette and "
                             "cultural questions",
        "Science & Mathematics": "the natural sciences, mathematics, engineering and "
                                 "homework in them",
        "Health": "illness, medicine, fitness, diet and mental health",
        "Education & Reference": "schooling, studying, words and definitions, and "
                                 "general reference questions",
        "Computers & Internet": "computers, software, hardware, the internet and "
                                "email",
        "Sports": "sports, teams, players and outdoor recreation",
        "Business & Finance": "money, jobs, investing, taxes, companies and the economy",
        "Entertainment & Music": "music, films, television, celebrities, books and "
                                 "games as entertainment",
        "Family & Relationships": "dating, marriage, friendship, parenting and family "
                                  "life",
        "Politics & Government": "politics, government, law, the military and "
                                 "elections",
    },
    "topic_mtop_domain": {
        "alarm": "setting, checking, changing or silencing alarms",
        "calling": "making, answering and managing phone or video calls",
        "event": "finding local events and responding to them",
        "messaging": "sending and reading messages",
        "music": "playing and controlling music and playlists",
        "news": "news stories and questions about the news",
        "people": "facts about people and contacts: employer, education, life events",
        "recipes": "recipes, cooking and ingredients",
        "reminder": "creating, checking and changing reminders",
        "timer": "setting and controlling timers",
        "weather": "weather forecasts and conditions, sunrise and sunset",
    },
    # MTOP publishes no intent definitions; the names are the definition. Criteria only
    # where the lexical baseline confuses two intents a reader must still tell apart
    # (an audit printed the pairs), from MTOP's own utterances.
    "topic_mtop_intent": {
        "REPLAY_MUSIC": "play the current song, or a named song, again",
        "PREVIOUS_TRACK_MUSIC": "go back to the song before the current one",
        "QUESTION_MUSIC": "a general question about music: charts, artists, albums",
        "GET_TRACK_INFO_MUSIC": "information about the track that is playing now",
        "GET_CALL": "whether there were calls, or the call history",
        "GET_CALL_TIME": "when a call happened",
        "GET_REMINDER": "which reminders exist",
        "GET_REMINDER_DATE_TIME": "when a reminder is set for",
        "GET_REMINDER_LOCATION": "where a reminder's event takes place",
        "GET_REMINDER_AMOUNT": "how many reminders there are",
        "UPDATE_REMINDER_DATE_TIME": "change when a reminder is set for",
        "UPDATE_REMINDER_TODO": "change what a reminder is about",
        "SILENCE_ALARM": "turn off or silence an alarm",
        "SNOOZE_ALARM": "delay an alarm that is going off",
        "GET_INFO_RECIPES": "information about a dish or recipe: ingredients, time, "
                            "temperature, nutrition",
        "GET_RECIPES": "find or suggest recipes",
        "IS_TRUE_RECIPES": "a yes/no question about a dish or cooking fact",
        "QUESTION_NEWS": "a specific question about something in the news",
        "GET_STORIES_NEWS": "news stories or headlines on a topic or place",
    },
    # CFPB's own product definitions for the complaint form, condensed
    "topic_cfpb_product": {
        "Checking or savings account": "a bank or credit-union deposit account: "
                                       "checking, savings, CDs",
        "Credit card or prepaid card": "a credit card, store card or prepaid card",
        "Credit reporting, credit repair services, or other personal consumer "
        "reports": "a credit report or score, a credit bureau's handling of a "
                   "dispute, or credit repair services",
        "Debt collection": "a creditor or collector trying to collect a debt",
        "Money transfer, virtual currency, or money service": "sending or receiving "
            "money, money orders, check cashing or virtual currency",
        "Mortgage": "a home loan: applying, servicing, payments, modification or "
                    "foreclosure",
        "Payday loan, title loan, or personal loan": "a payday loan, vehicle title loan "
                                                     "or personal installment loan",
        "Student loan": "a federal or private student loan",
        "Vehicle loan or lease": "a car or other vehicle loan or lease",
    },
}


def criteria_for(task: str, options: list[str]) -> list[str | None] | None:
    """Criteria for `descriptions.for_task`, one per option, or None for this task."""
    table = CRITERIA.get(task)
    if not table:
        return None
    got = [table.get(o) for o in options]
    return got if any(got) else None


# ---- state cleaning -------------------------------------------------------------------

_WS = re.compile(r"\s+")


def _squash(text: str) -> str:
    return _WS.sub(" ", text).strip()


# AG News's wire attributions: "(Reuters) Reuters - ", "(Sports Network) - ",
# "(Ticker) -- ", stock-quote link text and "(UpdateN)".
_AG_REPEAT = re.compile(r"\(([^()]{1,40})\)\s+\1\s*(?:--|-|:)\s+")
_AG_TAG_DASH = re.compile(r"\([^()]{1,40}\)\s*(?:--|-)\s+")
_AG_QUOTE = re.compile(r"\([^()]*(?:Quote, Profile, Research|Quote, Chart)[^()]*\)|"
                       r"\(Update\d+\)")


def _codepoint(n: int) -> str:
    # AG's crawl wrote Windows-1252 bytes as numeric references: #151; is an em dash
    if 128 <= n < 160:
        return bytes([n]).decode("cp1252", errors="ignore")
    return chr(n) if n < 0x110000 else ""


def clean_ag(text: str) -> str:
    # entities arrive with their "&" stripped: "It #39;s", " quot;3G quot;", "AT amp;T"
    # ...and an apostrophe's "&" became a space: "Telecom #39;s", "It #39;s"
    t = re.sub(r" &?#(?:39|146|8217);(?=[a-z])", "'", str(text or ""))
    t = re.sub(r"&?#(\d+);", lambda m: _codepoint(int(m.group(1))), t)
    t = re.sub(r"(?<!&)\b(quot|amp|nbsp|hellip|apos|lt|gt);", r"&\1;", t)
    t = re.sub(r"(?<=[a-z])(quot);", r"&\1;", t)
    t = html.unescape(html.unescape(t))
    t = re.sub(r"<[^>]{0,300}>", " ", t)
    t = t.replace("\\", " ")
    t = _AG_REPEAT.sub(" ", t)
    t = _AG_TAG_DASH.sub(" ", t)
    t = _AG_QUOTE.sub(" ", t)
    return _squash(t)


def clean_yahoo(text: str) -> str:
    t = str(text or "").replace("\\n", "\n")
    t = re.sub(r"<br\s*/?>", "\n", t)
    t = html.unescape(t)
    t = re.sub(r"[ \t]+", " ", t)
    return re.sub(r"\n\s*\n+", "\n", t).strip()


def cut(text: str, limit: int) -> str:
    """At most `limit` characters, ending on a word boundary where there is one."""
    if len(text) <= limit:
        return text
    head = text[:limit]
    space = head.rfind(" ")
    return (head[:space] if space > limit // 2 else head).rstrip()


# per-field budgets, so the JSON object is never cut mid-string by STATE_CHARS
YAHOO_BUDGET = {"question": 400, "details": 1200, "best_answer": 2200}


def yahoo_state(row: dict) -> str:
    fields = {"question": clean_yahoo(row.get("question_title")),
              "details": clean_yahoo(row.get("question_content")),
              "best_answer": clean_yahoo(row.get("best_answer"))}
    fields = {k: cut(v, YAHOO_BUDGET[k]) for k, v in fields.items() if v}
    if not fields:
        return ""
    out = json.dumps(fields, ensure_ascii=False, indent=2)
    while len(out) > STATE_CHARS and len(fields.get("best_answer", "")) > 200:
        # escaped newlines and quotes can push the object past the cap
        fields["best_answer"] = cut(fields["best_answer"], len(fields["best_answer"]) - 200)
        out = json.dumps(fields, ensure_ascii=False, indent=2)
    return out


# ---- pools ------------------------------------------------------------------------------

Item = tuple[str, str]           # (state, gold option)


def _h(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def dedup(items: list[tuple]) -> list[tuple]:
    """Each text once; a text the source labels two ways is dropped. Order kept.

    Items are (state, label, ...); only `label` (index 1) and anything after it must
    agree between copies.
    """
    seen: dict[str, tuple] = {}
    bad: set[str] = set()
    order: list[str] = []
    for it in items:
        if not it[0]:
            continue
        k = _norm(it[0])
        if k in seen:
            if seen[k][1:] != it[1:]:
                bad.add(k)
            continue
        seen[k] = it
        order.append(k)
    return [seen[k] for k in order if k not in bad]


def _names(key: str, col: str, default: tuple[str, ...]) -> list[str]:
    got = store.load_features(key).get(col)
    return list(got) if got else list(default)


def _label(value, names: list[str]) -> str | None:
    if isinstance(value, int) and not isinstance(value, bool) and 0 <= value < len(names):
        return names[value]
    return None


def _pool_agnews() -> list[Item]:
    key = BY_TASK["topic_agnews"].key
    names = _names(key, "label", AG_NEWS)
    return dedup([(clean_ag(r.get("text")), _label(r.get("label"), names))
                  for r in store.load(key)])


def _pool_dbpedia() -> list[Item]:
    key = BY_TASK["topic_dbpedia14"].key
    names = _names(key, "label", DBPEDIA)
    return dedup([(_squash(str(r.get("content") or "")), _label(r.get("label"), names))
                  for r in store.load(key)])


def _pool_yahoo() -> list[Item]:
    key = BY_TASK["topic_yahoo"].key
    names = _names(key, "topic", YAHOO)
    return dedup([(yahoo_state(r), _label(r.get("topic"), names))
                  for r in store.load(key)])


def trec_pools() -> dict[str, list[Item]]:
    """TREC train questions split between the coarse and fine tasks by a hash."""
    key = BY_TASK["topic_trec_coarse"].key
    coarse = _names(key, "coarse_label", TREC_COARSE)
    fine = _names(key, "fine_label", TREC_FINE)
    items = dedup([(_squash(str(r.get("text") or "")),
                    _label(r.get("fine_label"), fine), _label(r.get("coarse_label"), coarse))
                   for r in store.load(key)])
    out: dict[str, list[Item]] = {"topic_trec_coarse": [], "topic_trec_fine": []}
    for text, f, c in items:
        if int(_h("trec:" + _norm(text))[:8], 16) % 2:
            out["topic_trec_fine"].append((text, f))
        else:
            out["topic_trec_coarse"].append((text, c))
    return out


def mtop_intents(rows: list[dict]) -> list[str]:
    counts = Counter(str(r.get("intent") or "") for r in rows)
    return sorted(i[3:] for i, c in counts.items()
                  if i.startswith("IN:") and c >= MIN_INTENT and i[3:] not in DROP_INTENTS)


def mtop_pools() -> dict[str, list[Item]]:
    """MTOP English train utterances, split between the domain and intent tasks.

    An utterance whose intent has fewer than `MTOP_RARE` rows goes to the intent task, so
    the long tail is not halved; an utterance whose intent is not an option (too rare, or
    in `DROP_INTENTS`) can still be asked its domain. The rest are split by a hash.
    """
    rows = store.load(BY_TASK["topic_mtop_intent"].key)
    options = set(mtop_intents(rows))
    counts = Counter(str(r.get("intent") or "")[3:] for r in rows)
    items = dedup([(_squash(str(r.get("utterance") or "")), str(r.get("intent") or "")[3:],
                    str(r.get("domain") or "")) for r in rows])
    out: dict[str, list[Item]] = {"topic_mtop_domain": [], "topic_mtop_intent": []}
    for text, intent, domain in items:
        if intent in options and (counts[intent] < MTOP_RARE
                                  or int(_h("mtop:" + _norm(text))[:8], 16) % 2):
            out["topic_mtop_intent"].append((text, intent))
        elif domain in MTOP_DOMAINS:
            out["topic_mtop_domain"].append((text, domain))
    return out


def _pool_cfpb() -> list[Item]:
    rows = store.load(BY_TASK["topic_cfpb_product"].key)
    return dedup([(str(r.get("Consumer complaint narrative") or "").strip(), r.get("Product"))
                  for r in rows
                  if str(r.get("Date received") or "")[:4] in CFPB_YEARS
                  and r.get("Product") in CFPB_PRODUCTS])


def pool(task: str) -> list[Item]:
    """Every (state, gold) this task may draw from, deduped, before sampling."""
    if task in ("topic_trec_coarse", "topic_trec_fine"):
        got = trec_pools()[task]
    elif task in ("topic_mtop_domain", "topic_mtop_intent"):
        got = mtop_pools()[task]
    else:
        got = {"topic_agnews": _pool_agnews, "topic_dbpedia14": _pool_dbpedia,
               "topic_yahoo": _pool_yahoo, "topic_cfpb_product": _pool_cfpb}[task]()
    opts = set(options_for(task))
    return [(s, g) for s, g in got if g in opts]


def options_for(task: str) -> list[str]:
    if task == "topic_mtop_intent":
        return mtop_intents(store.load(BY_TASK[task].key))
    return list({"topic_agnews": AG_NEWS, "topic_dbpedia14": DBPEDIA, "topic_yahoo": YAHOO,
                 "topic_trec_coarse": TREC_COARSE, "topic_trec_fine": TREC_FINE,
                 "topic_mtop_domain": MTOP_DOMAINS,
                 "topic_cfpb_product": CFPB_PRODUCTS}[task])


def waterfill(items: list[Item], n: int) -> list[Item]:
    """At most `n` items, the largest classes capped first, in a class-interleaved order.

    Within a class the order is a hash of the state, so the choice is deterministic and
    unrelated to file order. The cap is the smallest per-class count c with
    sum(min(size, c)) >= n; classes at the cap then give one back each, in hash order of
    the class name, until exactly `n` remain.
    """
    by: dict[str, list[Item]] = defaultdict(list)
    for it in sorted(items, key=lambda it: _h(it[0])):
        by[it[1]].append(it)
    sizes = {k: len(v) for k, v in by.items()}
    if sum(sizes.values()) <= n:
        take = dict(sizes)
    else:
        lo, hi = 0, max(sizes.values())
        while lo < hi:
            mid = (lo + hi) // 2
            if sum(min(s, mid) for s in sizes.values()) >= n:
                hi = mid
            else:
                lo = mid + 1
        take = {k: min(s, lo) for k, s in sizes.items()}
        excess = sum(take.values()) - n
        for k in sorted(take, key=lambda k: _h("cap:" + k)):
            if excess <= 0:
                break
            if take[k] == lo:
                take[k] -= 1
                excess -= 1
    out: list[Item] = []
    rank = 0
    while len(out) < sum(take.values()):
        for k in sorted(by, key=lambda k: _h("order:" + k)):
            if rank < take[k]:
                out.append(by[k][rank])
        rank += 1
    return out


# ---- tasks -----------------------------------------------------------------------------

_BANK = None


def _question(src: Source, state: str) -> str:
    global _BANK
    if _BANK is None:
        from lod.phrasings import PhrasingBank
        _BANK = PhrasingBank.load()
    return _BANK.pick(src.template_id, src.question, _h(state)[:16])


def _loader(src: Source) -> Callable[[int], Iterator[Example]]:
    def load(n: int) -> Iterator[Example]:
        options = options_for(src.task)
        for state, gold in waterfill(pool(src.task), n):
            yield Example(task=src.task, state=cut(state, STATE_CHARS),
                          questions=[Question(id=src.task[len(PREFIX):],
                                              question=_question(src, state),
                                              options=list(options),
                                              target=options.index(gold))])
    return load


def family(src: Source) -> str:
    """The held-out unit: the dataset. MTOP domain and intent read one utterance pool, as
    do TREC coarse and fine, so each pair is one family (both halves of a pair are always
    forced to the same split); every other source is its own dataset and its own task."""
    if sum(s.key == src.key for s in SOURCES) == 1:
        return src.task
    return PREFIX + src.key.removeprefix("v14_topic_").removesuffix("_en")


def tasks() -> list[RealTask]:
    """One RealTask per source whose raw rows are in the store (fetch is stage 1)."""
    out = []
    for src in SOURCES:
        if not store.has(src.key):
            continue
        out.append(RealTask(row=ROW, name=src.task, licence=src.licence, url=src.url,
                            load=_loader(src), force_split=src.split, family=family(src),
                            notes=src.notes))
    return out
