"""Row 45 (`sources/real/topic.py`): English topic, intent and question type."""

from __future__ import annotations

import json
from collections import Counter

import pytest

from lod.corpus.services.sources.real import descriptions, topic
from lod.corpus.repositories import raw_store as store
from lod.corpus.domains import DOMAINS
from lod.corpus.services.enrich import protected
from lod.paths import ASSETS, RAW_ROOT

RAW = RAW_ROOT


@pytest.fixture
def fake_store(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "RAW", store.RAW)
    monkeypatch.setattr(store, "RAW_ROOT", store.RAW_ROOT)
    store.configure(tmp_path)
    yield tmp_path
    store.clear_cache()


@pytest.fixture
def real_store(monkeypatch):
    if not (RAW / "v4-raw" / "v14_topic_trec.jsonl").exists():
        pytest.skip("row-45 raw rows not fetched (scripts/fetch_data.py --rows 45)")
    monkeypatch.setattr(store, "RAW", store.RAW)
    monkeypatch.setattr(store, "RAW_ROOT", store.RAW_ROOT)
    store.configure(RAW)
    yield RAW
    store.clear_cache()


# ---- sampling ---------------------------------------------------------------------------

def test_waterfill_caps_the_majority_and_keeps_every_small_class():
    """CFPB 2018-2022 is 56 % credit reporting; the sample must not be."""
    items = ([(f"cr {i}", "cr") for i in range(600)] + [(f"m {i}", "m") for i in range(300)]
             + [(f"s {i}", "s") for i in range(20)])
    got = topic.waterfill(items, 200)
    c = Counter(g for _, g in got)
    assert len(got) == 200 and c["s"] == 20 and abs(c["cr"] - c["m"]) <= 1
    assert got == topic.waterfill(list(reversed(items)), 200)       # file order is ignored
    assert set(Counter(g for _, g in got[:3])) == {"cr", "m", "s"}  # interleaved prefix


def test_waterfill_takes_everything_when_the_pool_is_small():
    items = [("a", "x"), ("b", "y")]
    assert sorted(topic.waterfill(items, 10)) == sorted(items)


def test_dedup_drops_a_text_labelled_two_ways_and_keeps_a_repeat_once():
    got = topic.dedup([("Same text", "a"), ("same  TEXT", "a"), ("Two ways", "a"),
                       ("two ways", "b"), ("", "a")])
    assert got == [("Same text", "a")]


# ---- state cleaning --------------------------------------------------------------------

def test_ag_news_wire_attribution_and_html_debris_are_stripped():
    """'(Sports Network) -' was on 75 of 8,000 rows, every one Sports."""
    raw = ("Yanks win (Sports Network) - New York won. Wall St. Bears (Reuters) Reuters - "
           "Short-sellers are seeing green. It #39;s a quot;deal quot; AT amp;T &lt;b&gt;x&lt;/b&gt; #151; "
           "Oracle (ORCL.O: Quote, Profile, Research) rose (Update2) today\\again")
    got = topic.clean_ag(raw)
    for leak in ("Sports Network", "(Reuters)", "Reuters -", "Quote, Profile", "Update2",
                 "#39;", "quot;", "<b>", "\\", "amp;"):
        assert leak not in got
    assert "It's" in got and "Short-sellers are seeing green" in got


def test_yahoo_state_is_valid_json_under_the_cap_however_long_the_answer():
    row = {"question_title": "Why?", "question_content": "details " * 400,
           "best_answer": ('"quoted"\\n' * 2000)}
    st = topic.yahoo_state(row)
    assert len(st) <= topic.STATE_CHARS
    assert set(json.loads(st)) == {"question", "details", "best_answer"}


def test_dbpedia_state_is_the_abstract_without_the_title(fake_store):
    """16.4 % of stored titles end in a Wikipedia disambiguator ('(film)', '(album)')."""
    store.save("v14_topic_dbpedia_14", [
        {"label": 12, "title": "Her Last Affaire (film)",
         "content": " Her Last Affaire is a 1936 British drama."}])
    assert topic.pool("topic_dbpedia14") == [("Her Last Affaire is a 1936 British drama.",
                                              "Film")]


# ---- pools -------------------------------------------------------------------------------

def _mtop_rows():
    rows = []
    for i in range(80):
        rows.append({"id": str(i), "intent": "IN:GET_WEATHER", "utterance": f"weather {i}",
                     "domain": "weather"})
    for i in range(12):
        rows.append({"id": f"r{i}", "intent": "IN:GET_SUNRISE", "utterance": f"sunrise {i}",
                     "domain": "weather"})
    for i in range(14):
        rows.append({"id": f"u{i}", "intent": "IN:UPDATE_REMINDER", "utterance": f"upd {i}",
                     "domain": "reminder"})
    rows.append({"id": "x", "intent": "IN:GET_GENDER", "utterance": "gender?", "domain": "people"})
    return rows


def test_mtop_pools_are_disjoint_and_drop_indistinguishable_or_rare_intents(fake_store):
    store.save("v14_topic_mtop_en", _mtop_rows())
    opts = topic.options_for("topic_mtop_intent")
    assert opts == ["GET_SUNRISE", "GET_WEATHER"]     # UPDATE_REMINDER dropped, GET_GENDER rare
    pools = topic.mtop_pools()
    intent = {s for s, _ in pools["topic_mtop_intent"]}
    domain = {s for s, _ in pools["topic_mtop_domain"]}
    assert not intent & domain
    # a rare intent goes wholly to the intent task; a dropped one can still be asked its domain
    assert {f"sunrise {i}" for i in range(12)} <= intent
    assert "upd 3" in domain and "upd 3" not in intent
    assert all(g in opts for _, g in pools["topic_mtop_intent"])


def test_trec_pools_split_the_questions_and_coarse_is_the_fine_prefix(real_store):
    pools = topic.trec_pools()
    coarse, fine = pools["topic_trec_coarse"], pools["topic_trec_fine"]
    assert not {s for s, _ in coarse} & {s for s, _ in fine}
    assert abs(len(coarse) - len(fine)) < 200
    assert set(g for _, g in fine) <= set(topic.TREC_FINE)
    assert all(g.split(":")[0] in topic.TREC_COARSE for _, g in fine)


def test_cfpb_keeps_only_the_one_taxonomy_window(fake_store):
    store.save("v14_topic_cfpb", [
        {"Date received": "2019-05-01", "Product": "Mortgage", "Consumer complaint narrative": "a"},
        {"Date received": "2017-03-01", "Product": "Credit reporting",
         "Consumer complaint narrative": "b"},
        {"Date received": "2023-06-01", "Product": "Credit reporting or other personal "
         "consumer reports", "Consumer complaint narrative": "c"},
        {"Date received": "2023-06-01", "Product": "Mortgage", "Consumer complaint narrative": "d"},
    ])
    assert topic.pool("topic_cfpb_product") == [("a", "Mortgage")]


# ---- tasks, splits, criteria, registration ----------------------------------------------

def test_tasks_hold_out_whole_datasets(real_store):
    got = {t.name: t for t in topic.tasks()}
    assert {n: t.force_split for n, t in got.items()} == {
        "topic_dbpedia14": "train", "topic_mtop_domain": "train",
        "topic_mtop_intent": "train", "topic_cfpb_product": "train",
        "topic_agnews": "devreal",
        "topic_yahoo": "testreal", "topic_trec_coarse": "testreal",
        "topic_trec_fine": "testreal"}
    assert all(t.row == 45 and t.licence for t in got.values())


def test_tasks_sharing_a_pool_share_a_family(real_store):
    """TREC coarse/fine and MTOP domain/intent split one pool each: one family apiece, so
    devood's train/test disjointness is judged per dataset, not per label column."""
    from lod.corpus.services.sources.real.base import family_of
    fam = {t.name: family_of(t) for t in topic.tasks()}
    assert fam["topic_trec_coarse"] == fam["topic_trec_fine"] == "topic_trec"
    assert fam["topic_mtop_domain"] == fam["topic_mtop_intent"] == "topic_mtop"
    assert fam["topic_agnews"] == "topic_agnews"
    assert len(set(fam.values())) == 6


def test_tasks_skip_a_source_that_was_never_fetched(fake_store):
    assert topic.tasks() == []


def test_every_task_builds_answerable_distinct_questions(real_store):
    seen: dict[str, str] = {}
    for t in topic.tasks():
        exs = list(t.load(300))
        assert len(exs) == 300, t.name
        opts = exs[0].questions[0].options
        assert len({o.casefold() for o in opts}) == len(opts)
        labs = Counter(opts[e.questions[0].target] for e in exs)
        assert max(labs.values()) / len(exs) <= 0.5, t.name       # never near-constant
        for e in exs:
            assert e.state and len(e.state) <= topic.STATE_CHARS
            assert seen.setdefault(e.state, t.name) == t.name     # no state in two tasks


def test_mtop_intent_keeps_a_large_option_set(real_store):
    assert len(topic.options_for("topic_mtop_intent")) == 81
    assert not set(topic.DROP_INTENTS) & set(topic.options_for("topic_mtop_intent"))


def test_trec_fine_criteria_are_li_and_roth_for_every_class():
    got = descriptions.for_task("topic_trec_fine", list(topic.TREC_FINE))
    assert got is not None and all(got) and len(got) == 50
    assert descriptions.for_task("topic_trec_fine", ["ABBR:abb"]) == ["an abbreviation"]


def test_criteria_never_just_restate_their_option():
    for task, table in topic.CRITERIA.items():
        for opt, text in table.items():
            assert text.lower().strip() != opt.lower().replace("_", " "), (task, opt)
            if task != "topic_mtop_intent":      # MTOP's options come from the store
                assert opt in topic.options_for(task), (task, opt)


def test_questions_are_registered_templates_and_protected():
    spec = json.loads((ASSETS / "phrasing_specs" / "topic.json").read_text())
    for src in topic.SOURCES:
        assert spec[src.template_id]["template"] == src.question
        assert protected(src.task)


def test_row_45_is_domain_1_and_the_sweep_no_longer_claims_it():
    assert 45 in DOMAINS[1][0]
    from lod.corpus.services.sources.real import hub_sweep
    lower = {d.lower() for d in hub_sweep.DEDICATED_ADAPTERS}
    for d in ("fancyzhx/ag_news", "sh0416/ag_news", "fancyzhx/dbpedia_14",
              "community-datasets/yahoo_answers_topics", "cogcomp/trec",
              "mteb/mtopdomainclassification"):
        assert d in lower
    claimed = {s.path.lower() for s in hub_sweep.specs_from_survey()}
    assert "cogcomp/trec" not in claimed and "fancyzhx/ag_news" not in claimed
