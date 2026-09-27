"""Row 44: English sentiment, emotion and stance -- `sources/real/sentiment.py`."""
import json

import pytest

from lod.corpus.services.sources.real import descriptions, hub_sweep, sentiment as s
from lod.corpus.domains import DOMAINS
from lod.corpus.repositories import raw_store as store
from lod.corpus.services.enrich import protected
from lod.paths import ASSETS


@pytest.fixture
def raw(tmp_path, monkeypatch):
    """A private raw store, and fresh pool memos."""
    store.configure(tmp_path)
    (tmp_path / "store").mkdir(exist_ok=True)
    store.configure(tmp_path)
    monkeypatch.setattr(s, "_POOLS", {})
    monkeypatch.setattr(s, "DROPS", {})
    monkeypatch.setattr(s, "_DI_EXCLUDE", set())
    yield tmp_path
    store.clear_cache()


# ---- soft targets -------------------------------------------------------------------

def test_sst_target_bins_each_annotator_with_the_published_cutoffs():
    assert s.sst_target([13, 13, 13]) == [0, 0, 1, 0, 0]
    assert s.sst_target([1, 13, 25]) == pytest.approx([1 / 3, 0, 1 / 3, 0, 1 / 3])
    # (5-1)/24 = .167 very negative; (6-1)/24 = .208 negative; (20-1)/24 = .79 positive
    assert s.sst_target([5, 6, 20]) == pytest.approx([1 / 3, 1 / 3, 0, 1 / 3, 0])
    assert s.sst_bin(0.2) == 0 and s.sst_bin(0.6) == 2 and s.sst_bin(0.61) == 3


def test_sst_row_whose_raw_scores_do_not_reproduce_the_published_label_is_dropped():
    row = {"sentence": "A fine film .", "raw_scores": [20, 20, 20],
           "sentiment_value": (20 - 1) / 24}
    state, target, cls = s._read_sst(row)
    assert state == "A fine film." and target == [0, 0, 0, 1, 0] and cls == 3
    assert s._read_sst(dict(row, sentiment_value=0.5)) is None     # mean .79 != .5
    assert s._read_sst(dict(row, raw_scores=[20, 20])) is None      # < 3 raters


def test_sst_published_class_survives_a_tie_between_rater_shares():
    row = {"sentence": "x y z", "raw_scores": [7, 13, 19], "sentiment_value": 0.5}
    _, target, cls = s._read_sst(row)
    assert target == pytest.approx([0, 1 / 3, 1 / 3, 1 / 3, 0]) and cls == 2


def test_sst_detok():
    assert s.sst_detok("It 's a -LRB- great -RRB- `` film '' , is n't it ?") == \
        "It's a (great) \"film\", isn't it?"
    assert s.sst_detok("Ã©clair") == "éclair"


@pytest.mark.parametrize("label,agree,want", [
    ("positive", 100, [0, 0, 1]),
    ("positive", 50, [0, 0.43, 0.57]),
    ("negative", 75, [0.82, 0.18, 0]),
    ("neutral", 66, [0.155, 0.69, 0.155]),
])
def test_fpb_target_from_agreement_bracket(label, agree, want):
    t = s.fpb_target(label, agree)
    assert t == pytest.approx(want) and sum(t) == pytest.approx(1)
    assert max(range(3), key=lambda i: t[i]) == s.FPB_ORDER.index(label)


# ---- readers --------------------------------------------------------------------------

def test_tfns_label_map_follows_the_card_and_orders_the_options():
    spec = s.SPEC_BY_NAME["sent44_tfns"]
    names = {0: "Bearish", 1: "Bullish", 2: "Neutral"}          # LABEL_0/1/2 per card
    for raw, name in names.items():
        _, idx = s._read_tfns({"text": "$X up", "label": str(raw)})
        assert spec.options[idx] == name
    assert spec.options == ("Bearish", "Neutral", "Bullish") and spec.ordinal


def test_amazon_state_is_the_body_without_the_title():
    state, lab = s._read_amazon({"title": "Great product!", "content": "It works.",
                                 "label": 1})
    assert state == "It works." and "Great" not in state and lab == 1


def test_imdb_breaks_become_newlines():
    state, _ = s._read_imdb({"text": "Good.<br /><br />Really.", "label": 1})
    assert state == "Good.\n\nReally."


def test_semeval_escapes_and_entities_are_decoded():
    state, _ = s._read_tweet({"text": r"fan\u002c friend\u2019s &amp; more", "label": 0})
    assert state == "fan, friend\u2019s & more"


def test_semst_marker_is_stripped():
    state, lab = s._read_tweet({"text": "Life is precious #prolife #SemST", "label": 1})
    assert state == "Life is precious #prolife" and lab == 1


def test_claim_state_carries_motion_and_claim():
    state, lab = s._read_claim({"claims.stance": "CON", "topicText": "This house would X",
                                "claims.claimCorrectedText": "X harms Y"})
    assert json.loads(state) == {"motion": "This house would X", "claim": "X harms Y"}
    assert s.SPEC_BY_NAME["sent44_claim_stance"].options[lab] == "CON"


# ---- pools ----------------------------------------------------------------------------

def _save(key, rows):
    store.save(key, rows)


def test_pool_rules(raw, monkeypatch):
    long = "word " * 900
    _save("v14s_imdb", [{"text": "Shared text here", "label": 1},
                        {"text": long, "label": 0},
                        {"text": "Twice said", "label": 0},
                        {"text": "twice  SAID", "label": 0},
                        {"text": "Split vote", "label": 0},
                        {"text": "split vote", "label": 1}])
    _save("v14s_amazon", [{"title": "t", "content": "Shared text here!", "label": 0},
                          {"title": "t", "content": "Only amazon", "label": 1}])
    monkeypatch.setattr(s, "PRIORITY", ["sent44_imdb", "sent44_amazon_polarity"])
    pools = s.pools()
    imdb = [it[0] for it in pools["sent44_imdb"]]
    assert imdb == ["Shared text here", "Twice said"]
    assert s.DROPS["sent44_imdb"]["over_4000_chars"] == 1        # dropped, never cut
    assert s.DROPS["sent44_imdb"]["conflicting_labels"] == 1
    # the eval task (imdb) owns the shared text; the train task does not reuse it
    assert [it[0] for it in pools["sent44_amazon_polarity"]] == ["Only amazon"]
    assert s.DROPS["sent44_amazon_polarity"]["claimed_by_other_task"] == 1


def test_di_excluded_text_never_enters_a_pool(raw, monkeypatch):
    _save("v14s_amazon", [{"content": "Bad text", "label": 0},
                          {"content": "Good text", "label": 1}])
    monkeypatch.setattr(s, "_DI_EXCLUDE", {s.di_hash(s.norm_key("Bad text"))})
    monkeypatch.setattr(s, "PRIORITY", ["sent44_amazon_polarity"])
    assert [it[0] for it in s.pools()["sent44_amazon_polarity"]] == ["Good text"]


def test_select_caps_a_class_only_when_the_pool_allows():
    items = [(f"t{i}", 0, 0) for i in range(90)] + [(f"u{i}", 1, 1) for i in range(30)]
    got = s.select(items, 40, "task")
    assert len(got) == 40 and sum(1 for it in got if it[2] == 0) == 20
    assert s.select(items, 40, "task") == got                      # deterministic
    small = items[:9] + items[-1:]
    assert sorted(s.select(small, 40, "task")) == sorted(small)    # natural, not thinned
    few = [(f"t{i}", 0, 0) for i in range(90)] + [("u", 1, 1)]
    assert len(s.select(few, 40, "task")) == 40                    # cap unmeetable: fill


def test_examples_carry_soft_target_and_published_class(raw, monkeypatch):
    _save("v14s_fpb", [{"sentence": "Profit rose .", "label": "positive", "agree_min": 66}])
    monkeypatch.setattr(s, "PRIORITY", ["sent44_fpb"])
    (ex,) = s.examples_for(s.SPEC_BY_NAME["sent44_fpb"], 5)
    q = ex.questions[0]
    assert ex.task == "sent44_fpb" and q.options == ["negative", "neutral", "positive"]
    assert q.target == pytest.approx([0, 0.31, 0.69]) and q.meta["source_label"] == "positive"


# ---- the table ------------------------------------------------------------------------

def test_every_spec_is_well_formed():
    splits = {}
    for spec in s.SPECS:
        assert spec.name.startswith(s.PREFIX)
        assert len({o.casefold() for o in spec.options}) == len(spec.options) >= 2
        assert len(spec.criteria) == len(spec.options) and all(spec.criteria)
        assert spec.split in ("train", "devreal", "testreal") and spec.licence
        splits.setdefault(spec.split, []).append(spec.task)
    assert set(splits["testreal"]) == {"imdb", "fpb", "tweeteval_irony", "stance_hillary"}
    assert set(splits["devreal"]) == {"poem_sentiment", "tfns", "stance_feminist"}
    assert s.PRIORITY[:len(splits["testreal"])] == [s.PREFIX + t for t in
                                                     [x.task for x in s.SPECS
                                                      if x.split == "testreal"]]


def test_stance_questions_name_their_own_target():
    for t, (_, phrase) in s.STANCE_TARGETS.items():
        q = s.SPEC_BY_NAME[f"sent44_stance_{t}"].question
        assert phrase in q
        assert not any(p in q for u, (_, p) in s.STANCE_TARGETS.items() if u != t and p != phrase)


def test_phrasing_specs_match_the_templates():
    spec = json.loads((ASSETS / "phrasing_specs" / "sentiment.json").read_text())
    assert spec == {s.TEMPLATE_ID.format(task=x.task):
                    {"template": x.question, "state_kind": x.state_kind,
                     "answer_meaning": x.answer_meaning} for x in s.SPECS}


def test_tasks_are_protected_from_enrichment_and_described():
    assert protected("sent44_poem_sentiment")
    for spec in s.SPECS:
        assert descriptions.for_task(spec.name, list(spec.options)) == list(spec.criteria)
        assert descriptions.for_task(spec.name, ["x", "y"]) is None


def test_row_1_sweep_no_longer_claims_the_datasets_row_44_owns():
    owned = {"stanfordnlp/sst2", "cornell-movie-review-data/rotten_tomatoes",
             "stanfordnlp/imdb", "fancyzhx/amazon_polarity", "cardiffnlp/tweet_eval",
             "dair-ai/emotion", "mteb/emotion", "Yelp/yelp_review_full",
             "google-research-datasets/poem_sentiment",
             "zeroshot/twitter-financial-news-sentiment"}
    assert owned <= hub_sweep.DEDICATED_ADAPTERS


def test_domain_table_maps_row_44_to_affect_and_safety():
    rows, title = DOMAINS[3]
    assert 44 in rows and title == "Affect and safety"


def test_tasks_register_on_row_44_with_their_split(raw):
    _save("v14s_poem", [{"verse_text": "a line", "label": 1}])
    got = s.tasks()
    assert [t.name for t in got] == ["sent44_poem_sentiment"]
    assert got[0].row == 44 and got[0].force_split == "devreal"
