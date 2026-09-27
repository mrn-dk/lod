"""Source-table row 1 — the top `text-classification` datasets on the Hub.

Row 1 is not a fixed list, it is a *query*: the top 500 by downloads that load. So the
source table cannot name it and a survey of the Hub discovers it instead. This
module turns that survey into `HFSpec`s so the same adapter, the same label detection and
the same raw store handle it as everything else.

The survey result is a file rather than a live query on purpose. Which datasets load, and
which columns in them look like labels, is a measurement with a date on it: re-running the
query mid-build would silently change the corpus between the fetch stage and the build
stage. `lod/assets/hub_sweep_survey.json` is that measurement, and the corpus is
reproducible from it.

Measured 2026-09-18 over 500 datasets at low concurrency: 292 usable, 1,357 label columns
by value detection against 52 by typed ClassLabel.
"""

from __future__ import annotations

import json
from pathlib import Path

from lod.paths import ASSETS

from lod.corpus.services.sources.real.hf import HFSpec

LICENCE = "per-dataset; see the Hub repo"
# datasets whose module owns them, and which the sweep must therefore not claim. These
# are not in `table.SPECS` precisely because the generic adapter reads them wrongly:
# left to the sweep, Lichess became "What is the Popularity of this text?" over 82
# integers given a FEN, and Parexel became "What is the tier c member of this text?".
DEDICATED_ADAPTERS = {
    "lichess/chess-puzzles",
    "parexel/clinical-trials-protocols",
    "thomaswmitch/kalshi-prediction-markets-markets",
    # Row 34 (`grounding.py`). The sweep would read these as three-way text classification and
    # lose the one thing they are here for: NOT ENOUGH INFO is an abstention, not a
    # third class to predict.
    "copenlu/fever_gold_evidence",
    "tals/vitaminc",
    # `crowd.py` owns these, and they publish **one row per annotator**. The sweep read
    # them column by column as ordinary hard labels, so an earlier build carried 77 row-1
    # tasks and 1,663 questions asking "Does the text direct hate at people of Asian
    # descent?" over False/True with *one rater's* answer as the target -- the exact thing
    # `crowd.py` aggregates over `comment_id` to avoid, and the opposite of what the
    # provenance manifest claims for this source. 834 of the 1,460 states the sweep drew
    # also reach the corpus under `hatespeech__*` with a rater fraction, so the same
    # comment is asked twice with two different kinds of answer.
    "ucberkeley-dlab/measuring-hate-speech",
    "google-research-datasets/go_emotions",
    # Row 44 (`sentiment.py`) owns English sentiment, emotion and stance. Read by it:
    "stanfordnlp/sst2",            # superseded by SST-5 with per-annotator soft targets on
                                   # the same sentences; GLUE's train is 67k sub-phrases
    "cornell-movie-review-data/rotten_tomatoes",   # the snippets SST was parsed from
    "stanfordnlp/imdb", "fancyzhx/amazon_polarity", "mteb/amazon_polarity",
    "cardiffnlp/tweet_eval",       # the sweep read only `emoji`; row 44 reads 9 configs
    "google-research-datasets/poem_sentiment",
    "zeroshot/twitter-financial-news-sentiment",
    "takala/financial_phrasebank",
    # ...and rejected by it, so the sweep must not train on them either (row-44 audit):
    # labels machine-derived from the author's emotion keyword, "research use only" ...
    "dair-ai/emotion", "mteb/emotion",
    # ... and the Yelp Dataset Agreement limits use to non-profits and educational
    # institutions and forbids displaying the reviews to third parties.
    "Yelp/yelp_review_full", "fancyzhx/yelp_polarity",
    # row 46 (`multilingual.py`) reads these per language with whole languages
    # held out. Left to the sweep they would come back as one first-config slice each
    # (Acehnese SIB-200, Afrikaans MASSIVE scenario, German PAWS-X, Arabic XNLI) under
    # a hash split that ignores the language holdout; the `mteb/` copies are mirrors of
    # the same rows, and two of the SIB-200 copies were each other (an earlier build: 89 + 89).
    "mteb/amazon_massive_scenario", "AmazonScience/massive", "qanastek/MASSIVE",
    "Davlan/sib200", "mteb/sib200",
    "cardiffnlp/tweet_sentiment_multilingual",
    "masakhane/afrisenti", "shmuhammad/AfriSenti-twitter-sentiment",
    "HausaNLP/AfriSenti-Twitter", "mteb/AfriSentiClassification",
    "google-research-datasets/paws-x", "mteb/PawsXPairClassification",
    "facebook/xnli", "mteb/xnli",
    # not read by row 46, withheld from the sweep to protect its holdouts: AfriXNLI is
    # XNLI test translated into 16 African languages, Yoruba among them (held out);
    # LexC-Gen is LLM-generated text carrying SIB-200 labels (brief: no LLM labels).
    # Neither reached an earlier build (0 questions / not built), so nothing is lost.
    "masakhane/afrixnli", "BatsResearch/sib200-LexC-Gen",
    # row 45 (`topic.py`) reads these from a class-spread sample of the train split.
    # The sweep's copies were 89 questions each, DBpedia-14's all `Company` (the file is
    # sorted by class) and MTOP's labels bare integers; sh0416/ag_news is AG News again.
    "fancyzhx/ag_news", "sh0416/ag_news", "fancyzhx/dbpedia_14",
    "community-datasets/yahoo_answers_topics", "CogComp/trec",
    "mteb/MTOPDomainClassification", "mteb/MTOPIntentClassification",
    "mteb/mtop_domain", "mteb/mtop_intent",
    "BEE-spoke-data/consumer-finance-complaints", "CFPB/consumer-finance-complaints",
}
# Decision Index benchmarks the sweep must never train on (domain-1 audit). The DI scores MMLU
# (cais/mmlu test) and Amazon ESCI; `tasksource/mmlu` *is* MMLU's test/validation/dev
# split and `wassname/mmlu_preferences` is MMLU re-cut as preference pairs, and
# `tasksource/esci` is the ESCI shopping-queries set -- read through its product text
# alone, without the query its relevance label is about. ESCI is replaced by Wayfair's
# WANDS (`table.py`, row 1), the same query-product relevance judgement from another
# retailer. CLINC, BANKING77 and ANLI also appear in the DI and are declared, accepted
# overlaps: their train splits are read, the DI scores their test splits.
BENCHMARK_OVERLAP = {
    "tasksource/mmlu",
    "wassname/mmlu_preferences",
    "malhajar/mmlu_tr-v0.2",
    "tasksource/esci",
    # Also Decision Index members, or built from one. These were once "declared, accepted";
    # the brief is not to overlap the suite at all, and the sourcing
    # audit measured their text in our states: ANLI test premises in 19 of 42
    # facebook/anli states, CLINC test utterances verbatim, HellaSwag's ActivityNet
    # contexts in SWAG, an MMLU-Pro item in JudgeBench. The multilingual NLI set
    # contains ANLI.
    "facebook/anli",
    "MoritzLaurer/multilingual-NLI-26lang-2mil7",
    "clinc/clinc_oos",
    "mteb/banking77", "legacy-datasets/banking77", "PolyAI/banking77",
    "allenai/swag",
    "ScalerLab/JudgeBench",
    "yangwang825/reuters-21578",           # a SATA-Bench part
}
# Row-1 datasets whose licence does not clearly allow training (row-44 audit).
LICENCE_UNCLEAR = {
    "mteb/tweet_sentiment_extraction",     # Kaggle competition data, no licence stated
}
# Row-1 datasets whose rows cannot be read as a text-and-label pair at all (domain-1 audit).
UNREADABLE = {
    # the state is the scraped USAJOBS page, 14,000 characters of which the loader keeps
    # the first 4,000 -- mostly site chrome. Measured on 500 cached rows: the gold
    # `appointmentType` sits past the cut on 192, `travelRequirement` on 197, and
    # `hiringAgencyCode` ("DJ03") appears nowhere in the page on 497.
    "abigailhaddad/usajobs-scraping",
}
# Row-1 datasets with no train split whose splits partition *data* -- by month, year,
# licence, genre, or the article corpus itself -- rather than hold out a benchmark's
# evaluation set. Every other split-less dataset is refused by `hf.pick_splits`.
NO_TRAIN_SPLIT_OK = {
    "aisbergpublicorganization/telegram-news-ua-dataset",   # monthly splits
    "almanach/Biomed-Enriched",                             # commercial / noncommercial
    "golfplan18/msi-corpus",                                # news / opinion
    "maastrichtlawtech/bsard",                              # the statute corpus
    "yufan/arxiv-metadata-2020-2026",                       # yearly splits
}
# the frozen survey of the Hub sweep (which datasets loaded and how), shipped with the package
DEFAULT_SURVEY = ASSETS / "hub_sweep_survey.json"


def specs_from_survey(path: Path = DEFAULT_SURVEY, limit: int | None = None,
                      max_rows: int = 2000,
                      exclude_curated: bool = True) -> list[HFSpec]:
    """One HFSpec per dataset the survey found usable, minus the curated ones.

    The sweep is "top text-classification datasets by downloads", so it naturally
    rediscovers datasets the source table already names -- civil_comments, MNLI, SNLI,
    arXiv, MMLU, MASSIVE. The curated entry must win: the table's civil_comments knows
    its seven columns are *soft* rater fractions, where a generic sweep entry would read
    them as hard labels and silently convert the corpus's main soft source into ordinary
    binary questions. Keeping both would also collide in the split hash, since the task
    names are derived from the dataset path.
    """
    if not path.exists():
        return []
    rows = json.loads(path.read_text())
    usable = [r for r in rows if r.get("ok")]
    if exclude_curated:
        from lod.corpus.services.sources.real.table import SPECS
        # lower-cased on both sides: Hub ids are mixed case ("CogComp/trec",
        # "Davlan/sib200") and the survey names are compared lower-cased below
        curated = ({s.path.lower() for s in SPECS}
                   | {d.lower() for d in DEDICATED_ADAPTERS})
        usable = [r for r in usable if r["name"].lower() not in curated]
    banned = {p.lower() for p in BENCHMARK_OVERLAP | UNREADABLE | LICENCE_UNCLEAR}
    usable = [r for r in usable if r["name"].lower() not in banned]
    usable.sort(key=lambda r: -r.get("n_schemas", 0))
    if limit:
        usable = usable[:limit]
    return [HFSpec(row=1, path=r["name"], licence=LICENCE, max_rows=max_rows,
                   no_train_split_ok=r["name"] in NO_TRAIN_SPLIT_OK,
                   notes=f"row-1 sweep; {r.get('n_schemas', 0)} label columns detected")
            for r in usable]


def survey_summary(path: Path = DEFAULT_SURVEY) -> dict:
    if not path.exists():
        return {}
    rows = json.loads(path.read_text())
    return {"probed": len(rows),
            "usable": sum(1 for r in rows if r.get("ok")),
            "strict_schemas": sum(r["shapes"].get("strict", 0) for r in rows),
            "relaxed_schemas": sum(r["shapes"].get("relaxed", 0) for r in rows)}
