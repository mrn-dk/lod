"""Domain 3 (affect and safety): the soft-target invariant `crowd.py` exists for.

The target is the rater fraction, not the majority. Checked by hand for goemotions and
hatespeech records against the raw agg caches (`<LOD_RAW_ROOT>/go_emotions_agg.jsonl`,
`hate_speech_agg.jsonl`): exact match, no rounding, thresholding or majority vote.
"""

from __future__ import annotations

import pytest

from lod.corpus.services.sources.real import crowd


def test_loader_emits_exact_rater_fraction_not_rounded():
    """crowd._loader must reproduce frac[col] as [1-p, p] exactly -- no rounding, no
    thresholding to the nearest class. This is the soft-target backbone's core
    invariant; the documented soft target is false the day this stops holding."""
    rows = [
        {"text": "a", "frac": {"x": 0.0}},
        {"text": "b", "frac": {"x": 1.0}},
        {"text": "c", "frac": {"x": 1 / 3}},
        {"text": "d", "frac": {"x": 0.6}},
    ]
    load = crowd._loader(lambda: rows, "x", "t", "q?", index=0, n_tasks=1)
    examples = list(load(10))
    assert len(examples) == 4
    got = [e.questions[0].target for e in examples]
    expected = [[1.0, 0.0], [0.0, 1.0], [1 - 1 / 3, 1 / 3], [0.4, 0.6]]
    for g, e in zip(got, expected):
        assert g == pytest.approx(e)


def test_loader_skips_columns_missing_from_frac():
    """A column absent from an item's frac dict must be dropped, not defaulted to 0 --
    defaulting would fabricate a fraction over raters who never answered."""
    rows = [{"text": "a", "frac": {"other": 0.5}}]
    load = crowd._loader(lambda: rows, "x", "t", "q?", index=0, n_tasks=1)
    assert list(load(10)) == []


def test_go_emotions_and_hate_speech_tasks_declare_soft():
    """Both crowd.py task families must be tagged `soft`, or downstream soft-share
    accounting silently drops this domain's real contribution to A2's soft bucket."""
    tasks = crowd.tasks()
    if not tasks:
        pytest.skip("no cached go_emotions/hate_speech aggregates")
    by_row = {}
    for t in tasks:
        by_row.setdefault(t.row, []).append(t)
    for row, ts in by_row.items():
        assert row in (20, 21)
        assert all(t.soft for t in ts)
    if 20 in by_row:
        assert len(by_row[20]) == len(crowd.GO_EMOTION_COLS)
    if 21 in by_row:
        assert len(by_row[21]) == len(crowd.HATE_COLS)
        assert all(t.ordinal for t in by_row[21])


# --- polarity: the question has to face the same way as the target -------------------
# Domain-3 audit of an earlier corpus build. Two of the ten Measuring Hate Speech tasks
# shipped with their question inverted relative to their target, because the wording was
# derived from the column name and two of the column names read against the scale.
# Evidence, straight out of that build's `train.jsonl`:
#
#   hatespeech__respect   "Is this comment respectful [...]?"
#       p(yes)=1.000  four homophobic/misogynist slurs, the most confident "yes"es
#       p(yes)=0.000  'God bless you Pastor Owens [...] You are a great role model.'
#   hatespeech__sentiment "Does this comment express positive sentiment [...]?"
#       p(yes)=1.000  'Goys are incapable of thinking.'
#       p(yes)=0.000  'Mad props to the black ladies of AL for fighting the good fight.'
#
# Confirmed against the raw per-annotator rows: every item correlates *positively* with
# the corpus's own published `hate_speech_score`, `sentiment` at +0.81, `respect` at
# +0.85. The scales run towards hate; the questions did not.

_NEGATIVE_DIRECTION = {
    "sentiment": ("negative",),
    "respect": ("disrespectful",),
    "attack_defend": ("attack",),
    "status": ("inferior",),
}


def test_hate_questions_cover_every_column_and_face_the_hateful_end():
    """Each MHS column needs its own wording, pointing at the high end of its scale."""
    assert set(crowd.HATE_QUESTIONS) == set(crowd.HATE_COLS)
    for col, words in _NEGATIVE_DIRECTION.items():
        q = crowd.HATE_QUESTIONS[col].lower()
        assert any(w in q for w in words), f"{col}: {q!r} does not name the high end"
    # the two that were inverted must not ask for the *good* end any more
    assert "positive sentiment" not in crowd.HATE_QUESTIONS["sentiment"].lower()
    respect = crowd.HATE_QUESTIONS["respect"].lower()
    assert "respectful" not in respect.replace("disrespectful", "")
    # `attack_defend` is bipolar: "attacks or defends" is true at both ends while the
    # target is monotone, so the question must be the comparison, not the disjunction
    assert "or defend" not in crowd.HATE_QUESTIONS["attack_defend"].lower()


def test_hate_tasks_use_the_written_question_not_the_column_name():
    tasks = crowd.tasks()
    if not tasks:
        pytest.skip("no cached go_emotions/hate_speech aggregates")
    hate = {t.name: t for t in tasks if t.row == 21}
    for col in crowd.HATE_COLS:
        t = hate[f"hatespeech__{col}"]
        ex = next(iter(t.load(1)), None)
        if ex is None:
            continue
        # the phrasing bank may substitute a checked variant; with no bank entry it is
        # the written template itself
        assert ex.questions[0].question == crowd._phrasing(
            crowd.HATE_TEMPLATE_ID.format(col=col), crowd.HATE_QUESTIONS[col],
            f"{col}:{ex.questions[0].meta['item']}")
        assert "rate this comment high on" not in ex.questions[0].question


def test_goemotions_neutral_question_is_a_sentence():
    assert "express neutral?" not in crowd.GO_EMOTION_QUESTIONS["neutral"]
    assert crowd.GO_EMOTION_QUESTIONS["neutral"].endswith("?")


# --- aggregation: published scale, per-column rater floor ---------------------------

def _hs_rows(values, col="respect", n=4):
    return [{"comment_id": 7, "text": "t", col: v} for v in values][:n]


def test_hate_speech_reads_the_codebook_scale_not_the_observed_max(tmp_path, monkeypatch):
    """A sample whose top rating happens to be 2 must not be stretched to "yes".

    Dividing by the largest value seen rescales a whole task on a sampling accident; the
    codebook's 0-4 (0-2 for `hatespeech`) is a fact about the survey. Under the rater-share
    target (domain-3 audit) respect's midpoint is a neutral "no" and hatespeech's is an
    undecided half.
    """
    monkeypatch.setattr(crowd, "_HS_CACHE", tmp_path / "hs.jsonl")
    monkeypatch.setattr(crowd, "_local_rows", lambda *a, **k: [
        {"comment_id": 1, "text": "t", "respect": 2.0, "hatespeech": 1.0},
        {"comment_id": 1, "text": "t", "respect": 2.0, "hatespeech": 1.0},
        {"comment_id": 1, "text": "t", "respect": 2.0, "hatespeech": 1.0},
    ])
    out = crowd.aggregate_hate_speech(refresh=True)
    assert len(out) == 1
    assert out[0]["frac"]["respect"] == pytest.approx(0.0)
    assert out[0]["frac"]["hatespeech"] == pytest.approx(0.5)


def test_hate_speech_divides_by_the_answers_not_the_annotators(tmp_path, monkeypatch):
    """A skipped survey question must shrink the denominator, not the mean."""
    monkeypatch.setattr(crowd, "_HS_CACHE", tmp_path / "hs.jsonl")
    monkeypatch.setattr(crowd, "_local_rows", lambda *a, **k: [
        {"comment_id": 1, "text": "t", "respect": 4.0, "insult": 4.0},
        {"comment_id": 1, "text": "t", "respect": 4.0, "insult": 4.0},
        {"comment_id": 1, "text": "t", "respect": 4.0, "insult": 4.0},
        {"comment_id": 1, "text": "t", "respect": None, "insult": 4.0},
    ])
    out = crowd.aggregate_hate_speech(refresh=True)
    assert out[0]["n_raters"] == 4
    assert out[0]["frac"]["respect"] == pytest.approx(1.0)
    assert out[0]["frac"]["insult"] == pytest.approx(1.0)


def test_hate_speech_drops_a_column_answered_by_fewer_than_three(tmp_path, monkeypatch):
    """MIN_RATERS is per column, not only per item: two opinions is not a distribution
    however many annotators the *item* had."""
    monkeypatch.setattr(crowd, "_HS_CACHE", tmp_path / "hs.jsonl")
    monkeypatch.setattr(crowd, "_local_rows", lambda *a, **k: [
        {"comment_id": 1, "text": "t", "respect": 4.0, "insult": 4.0},
        {"comment_id": 1, "text": "t", "respect": 4.0, "insult": 4.0},
        {"comment_id": 1, "text": "t", "respect": None, "insult": 4.0},
    ])
    out = crowd.aggregate_hate_speech(refresh=True)
    assert "respect" not in out[0]["frac"]
    assert out[0]["frac"]["insult"] == pytest.approx(1.0)
    # and the loader turns the missing column into no question at all
    load = crowd._loader(lambda: out, "respect", "t", "q?", index=0, n_tasks=1)
    assert list(load(10)) == []


def test_min_raters_matches_the_generic_adapter():
    """`hf.py` documents its floor as "crowd.py's floor"; if they drift the corpus gets
    two different definitions of what counts as a distribution."""
    from lod.corpus.services.sources.real import hf

    assert crowd.MIN_RATERS == hf.MIN_SOFT_RATERS == 3
