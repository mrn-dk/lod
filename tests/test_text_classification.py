"""Domain 1, text classification and taxonomy: rows 1, 3, 6, 8, 10, 16, 18.

The domain is 850 tasks and 67,473 questions of "recognise a published class from a piece
of prose", and its whole claim is *breadth of schema*: an unseen option set is tractable
rather than memorable because the model has seen 850 different ones. Breadth is also what
makes it the easiest domain to fill with questions nobody could answer -- the row-1 sweep
does not read the datasets it claims, it detects a low-cardinality column and asks about
it, and a column can be low-cardinality because it records what crawl the row came from.

Every number quoted here was measured on an earlier shipped corpus build, and every test
asserts that the source modules would not produce it again. Offline throughout: the
loaders need the raw store, the rules are pure functions of rows already in hand.
"""

from __future__ import annotations

import pytest

from lod.schema import Example, Question
from lod.corpus.services.sources.real import hf, quality
from lod.corpus.services.sources.real.table import SPECS


def ex(state: str, gold: int, options=("a", "b"), task: str = "t") -> Example:
    return Example(task=task, state=state,
                   questions=[Question(id="q", question="which?",
                                       options=list(options), target=gold)])


# ---- the answerability gate, on the option set -------------------------------

def test_bare_numeral_options_still_rejected():
    assert quality.rejection("t", ["0", "1", "2"], described=False)
    assert quality.rejection("t", ["0", "1", "2"], described=True) is None


def test_hex_identifier_options_are_rejected_like_numerals():
    """Criteo's anonymised categorical features shipped 13 tasks / 545 questions of
    `2997ef88` against `2ba8d787`, given an 8-hex-digit state. The bare-numeral rule is
    the same argument in a different base and did not cover it."""
    reason = quality.rejection("criteo", ["2997ef88", "2ba8d787", "2ccea557"],
                               described=False)
    assert reason and "hex" in reason
    # words that happen to be hex-shaped are not identifiers
    assert quality.rejection("t", ["dead", "beef", "cafe"], described=False) is None
    assert quality.rejection("t", ["acquired", "crude", "earnings"],
                             described=False) is None


# ---- the answerability gate, on the examples ---------------------------------

def test_constant_answer_is_rejected():
    """215 tasks / 10,126 questions of an earlier build's domain 1 -- 14.3 % of it, and
    22.1 % of its testreal -- give the same answer on every example. A question-only
    lookup scores 1.000 on all of it."""
    items = [ex(f"state {i}", 0) for i in range(40)]
    reason = quality.example_rejection(items)
    assert reason and "every one of 40" in reason
    items[7].questions[0].target = 1
    assert quality.example_rejection(items) is None


def test_constant_answer_needs_enough_examples():
    """Three examples that agree are a small sample, not a degenerate schema."""
    assert quality.example_rejection([ex(f"s{i}", 0) for i in range(3)]) is None


def test_opaque_state_is_rejected():
    """714 questions carry a Fernet blob as their state; 252 carry an image URL and
    nothing else. `tensorshield_reddit_dataset_226__datetime` asks for a calendar date
    given `Z0FBQUFBQm42...`."""
    blob = "Z0FBQUFBQm42STRxOWRzR20xMGlDMWZEOWFORklMcDhacDlnOFlaTGpESGs0aWRfS29"
    items = [ex(blob + str(i), i % 2) for i in range(20)]
    assert "opaque blob" in (quality.example_rejection(items) or "")
    urls = [ex(f"http://images.cocodataset.org/train2017/{i:012d}.jpg", i % 2)
            for i in range(20)]
    assert "bare URL" in (quality.example_rejection(urls) or "")


def test_a_closed_vocabulary_of_short_tokens_is_not_a_state():
    """`nishan-chatterjee/llm_bias_detection` asks which of fourteen languages the text
    is in, and hands the reader the string `numeric`."""
    vocab = ["numeric", "lowercase_start", "uppercase_start", "numeric_zero_index"]
    items = [ex(vocab[i % 4], i % 3, options=("en", "fr", "de")) for i in range(40)]
    assert "short tokens" in (quality.example_rejection(items) or "")


def test_long_states_that_repeat_are_kept():
    """The rule is short *and* repeated. A genuine task may repeat a long passage."""
    items = [ex("a paragraph of real prose " * 5, i % 2) for i in range(40)]
    assert quality.example_rejection(items) is None


def test_short_states_that_do_not_repeat_are_kept():
    """`ni_task092_check_prime_classification` is 2,000 five-digit numbers and is one of
    the better tasks in the domain -- a length rule on its own would delete it."""
    items = [ex(str(60000 + i), i % 2) for i in range(40)]
    assert quality.example_rejection(items) is None


def test_state_that_spells_out_its_own_answer_is_rejected():
    """`finnmok_congressional_bills__docClass` opened with "[H.R. 3870 ...]" on 572 of
    572 states."""
    opts = ("acquisition", "crude", "earnings")
    items = [ex(f"a report about {opts[i % 3]} and nothing else", i % 3, options=opts)
             for i in range(30)]
    assert "spells out its own answer" in (quality.example_rejection(items) or "")


def test_yes_no_options_are_exempt_from_the_leak_test():
    """"no" occurs in ordinary prose; matching it says nothing about the answer."""
    items = [ex(f"there is no way to know, example {i}", i % 2, options=("no", "yes"))
             for i in range(30)]
    assert quality.example_rejection(items) is None


def test_filter_examples_reports_the_example_level_reason():
    items = [ex(f"state {i}", 0) for i in range(40)]
    kept, reason = quality.filter_examples(items, "t", described=True)
    assert kept == [] and reason and "every one of 40" in reason


# ---- which columns the sweep is allowed to ask about -------------------------

@pytest.mark.parametrize("col", [
    "datetime", "date", "published_date", "date_first_available", "fetched_at",
    "curation_date", "subset_date", "dump", "code_version", "git_commit",
    "generated_model", "model", "model2", "quantization", "task_name", "dataset",
    "split", "config", "spark_extractor",
])
def test_provenance_columns_are_not_labels(col):
    """A column recording when, where or with what the row was produced is not a property
    of the text. An earlier build asks "Which Common Crawl crawl snapshot does this web text
    come from?" and "Which quantization precision was used for the language model?"."""
    assert not hf._asks_about_the_text(col, hf.HFSpec(1, "p", "l"))


@pytest.mark.parametrize("col", [
    "language", "lang", "script", "label", "label_text", "category", "intent",
    "sentiment", "topic", "docClass", "publisher", "source", "emotion",
])
def test_content_columns_are_still_labels(col):
    assert hf._asks_about_the_text(col, hf.HFSpec(1, "p", "l"))


def test_an_explicit_label_col_overrides_the_provenance_rule():
    spec = hf.HFSpec(1, "p", "l", label_cols=("model",))
    assert hf._asks_about_the_text("model", spec)


def test_detect_label_cols_skips_provenance():
    rows = [{"text": f"a passage of prose number {i}", "topic": "sport" if i % 2 else "law",
             "crawl_dump": "CC-MAIN-2013-20" if i % 3 else "CC-MAIN-2014-10"}
            for i in range(40)]
    spec = hf.HFSpec(1, "p", "l")
    got = hf.detect_label_cols(rows, spec, "text")
    assert "topic" in got and "crawl_dump" not in got


# ---- which column the sweep reads as the text --------------------------------

def test_detect_text_col_prefers_free_text_over_an_enum():
    """The longest string column is not always the text: `llm_bias_detection`'s is a
    fifteen-token enum, and the corpus asked what language `numeric` is written in."""
    rows = [{"kind": ["numeric", "lowercase_start", "uppercase_start"][i % 3],
             "sentence": f"short {i}"} for i in range(60)]
    assert hf.detect_text_col(rows, hf.HFSpec(1, "p", "l")) == "sentence"


def test_detect_text_col_prefers_prose_even_when_it_repeats():
    rows = [{"passage": "a long piece of prose that runs well past eighty characters "
                        "and then keeps going for a while yet",
             "tag": f"t{i}"} for i in range(40)]
    assert hf.detect_text_col(rows, hf.HFSpec(1, "p", "l")) == "passage"


def test_detect_text_col_ignores_a_timestamp_column():
    rows = [{"Timestamp": f"03/07/2017 08:5{i % 10}:0{i % 10}",
             "note": f"n{i}"} for i in range(40)]
    assert hf.detect_text_col(rows, hf.HFSpec(1, "p", "l")) == "note"


def test_text_cols_still_wins():
    rows = [{"a": "x" * 500, "b": "y"} for _ in range(10)]
    assert hf.detect_text_col(rows, hf.HFSpec(1, "p", "l", text_cols=("b",))) == "b"


# ---- an evenly spread sample, not a prefix -----------------------------------

def test_spread_prefix_covers_a_sorted_pool():
    """`benayas/snips` is sorted by intent, and the first 2,000 of its 4,000 rows are
    92 % AddToPlaylist -- which is exactly the majority baseline an earlier build shipped,
    0.921 over three options, against 0.47 for the pool it was drawn from."""
    rows = [{"i": i, "label": "a" if i < 2000 else "b"} for i in range(4000)]
    prefix = hf._spread(rows)[:2000]
    share = sum(1 for r in prefix if r["label"] == "a") / len(prefix)
    assert 0.45 <= share <= 0.55
    assert sorted(r["i"] for r in hf._spread(rows)) == list(range(4000))


def test_spread_is_deterministic_and_total():
    rows = [{"i": i} for i in range(37)]
    assert hf._spread(rows) == hf._spread(rows)
    assert {r["i"] for r in hf._spread(rows)} == set(range(37))
    assert hf._spread([]) == [] and hf._spread(rows[:1]) == rows[:1]


# ---- the two rows whose states stated their own answer -----------------------

def _spec(path: str) -> hf.HFSpec:
    return next(s for s in SPECS if s.path == path)


def test_arxiv_strips_its_own_category_stamp():
    """An arXiv PDF is stamped with the submission's primary category. Measured on
    an earlier build: 88.5 % of states carried a stamp and it equalled the gold on
    55.4 %."""
    spec = _spec("ccdv/arxiv-classification")
    text = ("Constrained Submodular Maximization\n\n"
            "arXiv:1611.03253v1 [cs.DS] 10 Nov 2016\n\nWe give an algorithm.")
    out = hf.state_for({"text": text}, spec, "text")
    assert "cs.DS" not in out and "arXiv:1611.03253" not in out
    assert "Constrained Submodular Maximization" in out
    assert "We give an algorithm." in out


def test_arxiv_also_redacts_the_gold_category_from_the_body():
    spec = _spec("ccdv/arxiv-classification")
    assert spec.scrub_label
    assert "cs.SY" not in hf._scrub_label("compare against the cs.SY literature", "cs.SY")


def test_congressional_bills_strip_the_gpo_header():
    """572 of 572 states opened with the measure's own designator."""
    spec = _spec("finnmok/congressional_bills")
    text = ("[Congressional Bills 103th Congress]\n"
            "[From the U.S. Government Publishing Office]\n"
            "[H.R. 3870 Placed on Calendar Senate (PCS)]\n\n"
            "103d CONGRESS\n  2d Session\n"
            "                                H. R. 3870\n\n"
            "                                 A BILL\n\n"
            "  To promote the research and development of environmental technologies.\n"
            "                    IN THE HOUSE OF REPRESENTATIVES\n")
    out = hf.state_for({"text": text}, spec, "text")
    assert "H.R. 3870" not in out and "H. R. 3870" not in out
    assert "Publishing Office" not in out
    # what the question is actually about survives
    assert "A BILL" in out and "IN THE HOUSE OF REPRESENTATIVES" in out


def test_openalex_no_longer_asks_for_open_access_status_from_a_title():
    """Row 8's mirror is titles plus metadata. Its only surviving task asked, given a
    title alone, whether the work is gold/green/bronze/hybrid/diamond/closed -- a
    property of the venue and the licence. Naive Bayes on the title scores 0.215 against
    a 0.790 majority, i.e. below answering "closed" every time."""
    assert "oa_status" in _spec("XIfr/Openalex-2005-2025").skip_cols


def test_natural_instructions_task_name_is_not_a_schema():
    """The generic adapter read Super-NaturalInstructions' own `task_name` as a label
    column while `instructions.py` was reading the file properly, and shipped 2,000
    questions of "Which task is this passage an example of?" with a constant answer."""
    spec = next(s for s in SPECS if s.row == 3)
    assert not hf._asks_about_the_text("task_name", spec)


# ---- strip_state_res is off unless a spec asks for it ------------------------

def test_strip_state_res_defaults_to_a_no_op():
    spec = hf.HFSpec(1, "p", "l")
    text = "[Congressional Bills] arXiv:1611.03253v1 [cs.DS]"
    assert hf.state_for({"text": text}, spec, "text") == text


def test_ip_address_options_are_identifiers_too():
    """`rdpahalavan_cic_ids2017__source_ip` asks which of 144 IP addresses a record came
    from, and the enrichment gave all 144 the same gloss."""
    reason = quality.rejection("t", ["10.40.170.2", "10.40.182.1", "149.171.126.0"],
                               described=False)
    assert reason and "hex" in reason


def test_crowd_annotated_sources_are_left_to_their_own_adapter():
    """`crowd.py` aggregates `ucberkeley-dlab/measuring-hate-speech` and GoEmotions over
    the *item*. The sweep read them row by row, and one row is one annotator: 77 tasks
    and 1,663 questions of an earlier build carry a single rater's judgement as a hard
    target for a source whose whole point is the rater fraction."""
    from lod.corpus.services.sources.real.hub_sweep import DEDICATED_ADAPTERS
    assert "ucberkeley-dlab/measuring-hate-speech" in DEDICATED_ADAPTERS
    assert "google-research-datasets/go_emotions" in DEDICATED_ADAPTERS


def test_massive_intent_is_one_task_not_two():
    """`mteb/amazon_massive_intent` ships the intent twice, as `label` and `label_text`,
    with the same 42 strings. The hash put one in train and one in devreal, which is
    1,865 of domain 1's 2,420 eval questions whose option set a trained task already
    has."""
    assert "label" in _spec("mteb/amazon_massive_intent").skip_cols
