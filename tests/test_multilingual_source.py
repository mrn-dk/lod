"""Row 46, multilingual classification (`sources/real/multilingual.py`).

The plan-level tests need no data. The ones that read the raw store skip when it has not
been fetched (`scripts/fetch_data.py --rows 46`).
"""
from __future__ import annotations

import json
from collections import defaultdict

import pytest

from lod.corpus.services.sources.real import descriptions, hub_sweep
from lod.corpus.repositories import raw_store as store
from lod.corpus.domains import DOMAINS
from lod.corpus.services.enrich import protected
from lod.corpus.services.sources.real import multilingual as ml
from lod.corpus.services.sources.real.base import RealTask
from lod.paths import ASSETS, RAW_ROOT

RAW = RAW_ROOT


# ------------------------------------------------------------------ the plan
def test_held_out_languages_never_train():
    for src, lang in ml.plan():
        split = src.split_for(lang)
        if lang in ml.TEST_LANGS:
            assert split == "testreal", (src.name, lang)
        if lang in ml.DEV_LANGS:
            assert split in ("devreal", "testreal"), (src.name, lang)
    train_langs = {lang for src, lang in ml.plan() if src.split_for(lang) == "train"}
    assert not train_langs & (set(ml.TEST_LANGS) | set(ml.DEV_LANGS))
    assert 12 <= len(train_langs) <= 20, sorted(train_langs)
    assert "en" not in {lang for _, lang in ml.plan()}


def test_whole_datasets_held_out():
    assert ml.BY_NAME["pawsx_paraphrase"].whole_split == "testreal"
    assert ml.BY_NAME["xnli"].whole_split == "devreal"
    for name in ("pawsx_paraphrase", "xnli"):
        src = ml.BY_NAME[name]
        # the machine-translated train split is never read
        assert all("train" not in sp for sps in src.reads.values() for sp in sps)


def test_xnli_leaves_out_the_held_out_languages():
    assert not set(ml.BY_NAME["xnli"].codes) & set(ml.TEST_LANGS)


def test_parallel_corpora_read_disjoint_splits_for_held_out_languages():
    """MASSIVE and SIB-200 are translations of one set: a held-out language must read a
    split whose ids no training language reads."""
    for name in ("massive_intent", "massive_scenario", "sib200_topic"):
        src = ml.BY_NAME[name]
        assert src.reads["train"] == ("train",)
        for split in ("devreal", "testreal"):
            assert "train" not in src.reads[split]


def test_every_task_is_row_46_with_a_forced_split_and_a_licence():
    for src, lang in ml.plan():
        t = RealTask(row=ml.ROW, name=src.task_name(lang), licence=src.licence, url="",
                     load=lambda n: [], force_split=src.split_for(lang))
        assert t.name.startswith(ml.PREFIX)
        assert t.licence
    names = [src.task_name(lang) for src, lang in ml.plan()]
    assert len(names) == len(set(names))


def test_options_have_criteria_and_no_case_duplicates():
    for src in ml.SOURCES:
        opts = src.options
        assert len({o.casefold() for o in opts}) == len(opts)
        got = descriptions.for_task(src.task_name(next(iter(src.codes))), opts)
        assert got == [src.criteria[o] for o in opts]
        assert all(got)
    # a different option set is not described
    assert ml.criteria_for("ml_xnli__de", ["yes", "no"]) is None
    assert ml.criteria_for("goemotions__joy", ["no", "yes"]) is None


def test_massive_label_sets_are_the_published_ones():
    assert len(ml.MASSIVE_INTENTS) == 60
    assert len(ml.MASSIVE_SCENARIOS) == 18
    # every intent is <scenario>_<action>
    assert {i.split("_")[0] for i in ml.MASSIVE_INTENTS} == set(ml.MASSIVE_SCENARIOS)


def test_raw_label_mapping():
    tw = ml.BY_NAME["tweet_sentiment"]
    assert ml.raw_label(tw, {"label": "0"}) == "negative"
    assert ml.raw_label(tw, {"label": "2"}) == "positive"
    assert ml.raw_label(tw, {"label": "-1"}) is None
    px = ml.BY_NAME["pawsx_paraphrase"]
    assert ml.raw_label(px, {"label": 1}) == "yes"
    xn = ml.BY_NAME["xnli"]
    assert ml.raw_label(xn, {"label": 0}) == "entailment"
    assert ml.raw_label(xn, {"label": 2}) == "contradiction"
    af = ml.BY_NAME["afrisenti"]
    assert ml.raw_label(af, {"label": "neutral"}) == "neutral"
    assert ml.raw_label(af, {"label": "mixed"}) is None


def test_pair_states_are_json_with_both_sides():
    px = ml.BY_NAME["pawsx_paraphrase"]
    st = ml.state_of(px, {"sentence1": " a  b ", "sentence2": "c"})
    assert json.loads(st) == {"sentence1": "a b", "sentence2": "c"}
    assert ml.state_of(px, {"sentence1": "a", "sentence2": ""}) == ""


def test_questions_are_registered_templates_and_protected():
    spec = json.loads((ASSETS / "phrasing_specs" / "d21.json").read_text())
    for src in ml.SOURCES:
        assert spec[src.template_id]["template"] == src.question
    assert protected("ml_massive_intent__de")


def test_registered_row_and_domain_and_sweep():
    assert DOMAINS[21] == ({46}, "Multilingual classification")
    for path in ("Davlan/sib200", "mteb/sib200", "mteb/amazon_massive_scenario",
                 "google-research-datasets/paws-x", "mteb/PawsXPairClassification",
                 "mteb/xnli", "cardiffnlp/tweet_sentiment_multilingual",
                 "masakhane/afrisenti"):
        assert path in hub_sweep.DEDICATED_ADAPTERS
    from lod.corpus.services.generate import ROWS
    assert 46 in ROWS


def test_synthetic_pool_claims_ambiguity_and_exclusion(monkeypatch):
    src = ml.BY_NAME["afrisenti"]
    rows = [
        {"tweet": "same text", "label": "positive", "_lang": "hau", "_split": "train"},
        {"tweet": "same text", "label": "negative", "_lang": "hau", "_split": "train"},
        {"tweet": "fine text", "label": "neutral", "_lang": "hau", "_split": "train"},
        {"tweet": "blocked text", "label": "neutral", "_lang": "hau", "_split": "train"},
        {"tweet": "dev only", "label": "neutral", "_lang": "hau", "_split": "dev"},
    ]
    monkeypatch.setitem(ml._ROWS, src.store_key, rows)
    monkeypatch.setattr(ml, "_EXCLUDED", {ml.state_hash("blocked text")})
    got = [st for st, _, _ in ml.candidates(src, "ha")]
    # two labels for one text: dropped; excluded: dropped; wrong split: not read
    assert got == ["fine text"]


# ------------------------------------------------------------------ against the store
needs_store = pytest.mark.skipif(
    not all((RAW / "v4-raw" / f"{s.store_key}.jsonl").exists() for s in ml.SOURCES),
    reason="row 46 not fetched (scripts/fetch_data.py --rows 46)")


@pytest.fixture(scope="module")
def built():
    store.configure(RAW)
    ml._CLAIMS = None
    ml._EXCLUDED = None
    out = {}
    for t in ml.tasks():
        out[t.name] = (t, list(t.load(10_000)))
    return out


@needs_store
def test_store_tasks_and_counts(built):
    assert len(built) == len(ml.plan())
    total = sum(len(ex) for _, ex in built.values())
    assert 25_000 <= total <= 35_000, total
    for name, (task, ex) in built.items():
        src = ml.BY_NAME[name[len(ml.PREFIX):].split("__")[0]]
        assert 0 < len(ex) <= src.cap, name
        assert task.force_split == src.split_for(name.split("__")[1])
        assert task.min_quota == src.cap


@needs_store
def test_store_no_state_in_two_tasks_and_eval_never_repeats_train(built):
    owner: dict[str, str] = {}
    for name, (task, ex) in built.items():
        for e in ex:
            assert owner.setdefault(e.state, name) == name, (e.state[:60], name, owner[e.state])
    train = {e.state for _, (t, ex) in built.items() if t.force_split == "train" for e in ex}
    for _, (t, ex) in built.items():
        if t.force_split != "train":
            assert not {e.state for e in ex} & train


@needs_store
def test_store_targets_valid_and_majority_bounded(built):
    for name, (_, ex) in built.items():
        counts = defaultdict(int)
        for e in ex:
            q = e.questions[0]
            assert 0 <= q.target < len(q.options)
            assert q.meta["lang"] == name.split("__")[1]
            counts[q.target] += 1
        assert max(counts.values()) <= 0.8 * len(ex), name


@needs_store
def test_store_massive_intent_and_scenario_ids_disjoint(built):
    for lang in ml.BY_NAME["massive_intent"].codes:
        a = {e.questions[0].meta["source_id"] for e in built[f"ml_massive_intent__{lang}"][1]}
        b = {e.questions[0].meta["source_id"] for e in built[f"ml_massive_scenario__{lang}"][1]}
        assert not a & b, lang


@needs_store
def test_store_held_out_parallel_items_are_not_translations_of_trained_ones(built):
    for fam in ("massive_intent", "massive_scenario", "sib200_topic"):
        trained, held = set(), set()
        for name, (t, ex) in built.items():
            if not name.startswith(f"ml_{fam}__"):
                continue
            ids = {e.questions[0].meta["source_id"] for e in ex}
            (trained if t.force_split == "train" else held).update(ids)
        assert not trained & held, fam


@needs_store
def test_store_dev_items_are_not_translations_of_test_items(built):
    """Dev selects checkpoints and fits T, so it must not hold a translation of a test
    item either: SIB-200's dev and test languages read the same Flores splits."""
    for fam in ("massive_intent", "massive_scenario", "sib200_topic"):
        side: dict[str, set] = {"devreal": set(), "testreal": set()}
        for name, (t, ex) in built.items():
            if name.startswith(f"ml_{fam}__") and t.force_split in side:
                side[t.force_split].update(e.questions[0].meta["source_id"] for e in ex)
        assert side["devreal"] and side["testreal"], fam
        assert not side["devreal"] & side["testreal"], fam
