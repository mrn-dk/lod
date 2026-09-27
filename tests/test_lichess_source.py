"""Row 13: eight tasks trained (seven themes + rating), eight held out to test, two to dev,
and the things that split can break.

The puzzles are the one domain whose state has no natural-language shortcut, so what the
corpus does with them is worth holding in place with tests rather than comments:

- the eight reserved themes must actually be produced (the theme list is computed from
  whatever rows are cached, so it can move under the holdout);
- no state may be shared between a trained task and a reserved or dev one, or the dedup
  pass deletes the eval side and whatever survives is a leak;
- the dev themes are a second holdout, disjoint from test's, each paired with a trained
  theme on the same axis;
- every theme task must be balanced, because the old loader padded a task out to its
  quota with negatives and shipped `lichess_theme_pin` at 11 % yes, where answering "no"
  scored 89 %;
- the theme and the rating are the labels, so neither may appear in the state;
- a theme whose answer is not *in* the state may not become a task at all -- `master`
  records who played the game, which no FEN carries, and `opening` is printed in the
  state by `OpeningTags`;
- every theme ships its published definition as the option criteria, because a criterion
  that restates the question leaves the theme word as the only thing carrying meaning;
- the gates must name the eight and only the eight, or `reserved_families` counts
  trained themes as reserved and stops measuring what it reports.
"""

from __future__ import annotations

import json
import re

import pytest

from lod.corpus.services.sources.real import lichess
from lod.corpus.repositories import raw_store as store
from lod.paths import RAW_ROOT
from lod.corpus.services.sources.real.base import RESERVED_TESTREAL_ROWS, split_of

_STOP = set("a an the of to in for and or is are it its this that with on by not no yes "
            "does do puzzle chess position solution moves move have theme".split())
TRAINED_PAIRS = [("mateIn1", "mateIn2"), ("short", "long"), ("short", "veryLong"),
                 ("endgame", "rookEndgame"), ("fork", "pin"), ("fork", "sacrifice")]


@pytest.fixture(scope="module")
def tasks():
    """Real cached puzzles, with the store's global state put back afterwards."""
    saved = (store.RAW, store.RAW_ROOT, lichess._ROWS)
    store.configure(RAW_ROOT)
    lichess._ROWS = None
    try:
        out = lichess.tasks()
        if not out:
            pytest.skip("no cached lichess rows")
        yield out
    finally:
        store.RAW, store.RAW_ROOT = saved[0], saved[1]
        lichess._ROWS = saved[2]
        store.clear_cache()


def test_row_13_is_no_longer_reserved_whole():
    """The row-level holdout is gone; the holdout is now per theme, via force_split."""
    assert 13 not in RESERVED_TESTREAL_ROWS
    assert {split_of(f"lichess_theme_{i}", 13) for i in range(200)} != {"testreal"}


def test_every_reserved_theme_is_produced(tasks):
    names = {t.name for t in tasks}
    for theme in lichess.RESERVED_THEMES:
        assert f"lichess_theme_{theme}" in names


def test_a_missing_reserved_theme_fails_loudly(monkeypatch):
    """A cache that stops producing a reserved theme must not quietly move the holdout."""
    monkeypatch.setattr(lichess, "_rows", lambda: [{"FEN": "x", "Themes": "['fork']"}])
    monkeypatch.setattr(lichess, "_top_themes", lambda: ["fork", "mate"])
    with pytest.raises(ValueError, match="reserved themes"):
        lichess.tasks()


def test_eight_trained_eight_reserved_two_dev(tasks):
    by_split = {"train": set(), "devreal": set(), "testreal": set()}
    for t in tasks:
        assert t.row == 13 and t.licence
        assert t.force_split in by_split                 # never left to the hash
        by_split[t.force_split].add(t.name)
    assert len(by_split["testreal"]) == 8
    assert len(by_split["devreal"]) == 2
    assert len(by_split["train"]) == 8                # oneMove duplicates mateIn1
    assert by_split["testreal"] == {f"lichess_theme_{t}" for t in lichess.RESERVED_THEMES}
    assert by_split["devreal"] == {f"lichess_theme_{t}" for t in lichess.DEV_THEMES}
    assert not set(lichess.DEV_THEMES) & set(lichess.RESERVED_THEMES)
    assert "lichess_rating_bucket" in by_split["train"]


def test_each_reserved_theme_is_paired_with_a_trained_one(tasks):
    """The transfer question is depth, length, specialisation and motif -- a held-out
    structure against a trained one, not a held-out sample."""
    trained = {t.name for t in tasks if t.force_split == "train"}
    reserved = {t.name for t in tasks if t.force_split == "testreal"}
    for near, far in TRAINED_PAIRS:
        assert f"lichess_theme_{near}" in trained
        assert f"lichess_theme_{far}" in reserved


def test_each_dev_theme_is_paired_with_a_trained_one(tasks):
    """Dev's held-out themes are the same kind of transfer as test's -- a theme on an axis
    the model trains on, never the axis itself."""
    trained = {t.name for t in tasks if t.force_split == "train"}
    dev = {t.name for t in tasks if t.force_split == "devreal"}
    assert {far for _, far in lichess.DEV_PAIRS} == set(lichess.DEV_THEMES)
    for near, far in lichess.DEV_PAIRS:
        assert f"lichess_theme_{near}" in trained
        assert f"lichess_theme_{far}" in dev


def test_no_state_is_shared_between_trained_and_reserved(tasks):
    """A shared FEN is both a leak and, once dedup_against_train sees it, a deleted
    eval row: 300 devreal and 600 testreal examples went that way on an earlier build."""
    seen: dict[str, set[str]] = {"train": set(), "devreal": set(), "testreal": set()}
    for t in tasks:
        for e in t.load(400):
            seen[t.force_split].add(e.state)
    assert seen["train"] and seen["devreal"] and seen["testreal"]
    assert not (seen["train"] & seen["testreal"])
    assert not (seen["train"] & seen["devreal"])
    assert not (seen["devreal"] & seen["testreal"])


def test_the_dev_pool_leaves_every_testreal_state_where_it_was(tasks):
    """The dev pool is carved from the trained half, so test's held-out states are
    exactly the reserved half they always were."""
    assert all(lichess._reserved_pool(r) for r in lichess._pool(True))
    assert not any(lichess._reserved_pool(r) for r in lichess._pool("devreal"))
    reserved = [r for r in lichess._rows()
                if lichess._reserved_pool(r) and lichess._fully_tagged(r)]
    assert lichess._pool(True) == reserved


def test_theme_tasks_are_balanced(tasks):
    for t in tasks:
        if not t.name.startswith("lichess_theme_"):
            continue
        targets = [q.target for e in t.load(2000) for q in e.questions]
        assert len(targets) >= 200, f"{t.name} is too thin to measure"
        assert sum(targets) * 2 == len(targets), f"{t.name} is not balanced"


def test_the_label_is_not_in_the_state(tasks):
    """Themes and rating are what is being asked for, so the state carries neither."""
    for t in tasks:
        for e in list(t.load(40)):
            state = json.loads(e.state)
            assert set(state) == {"fen", "solution_moves", "opening"}
            assert "theme" not in e.state.lower()
            for q in e.questions:
                if q.id != "rating_bucket":
                    assert q.id.lower() not in e.state.lower()


def test_gates_reserved_prefixes_match_lichess(tasks):
    """The gates read the reserved themes by task prefix. If the prefixes and the routing
    drift, `reserved_families` either counts a trained theme as reserved or drops a
    reserved one, and reports a number for neither."""
    from lod.evaluation.gates import reserved_prefixes

    prefixes = reserved_prefixes()
    for t in tasks:
        reserved = t.force_split == "testreal"
        assert t.name.startswith(prefixes) is reserved, t.name


# ---- a theme must be answerable from the state, and say what it means -------------------

def test_no_task_asks_a_theme_the_state_cannot_answer(tasks):
    """`master` is "the game was played by titled players". That is provenance: it is not
    in the FEN, not in the solution and not in the opening tag, so no reader and no code
    could derive it. An earlier build shipped 588 trained questions of it, on which a
    depth-3 tree over ten shallow state features reached 0.587 against a 0.500
    label-permuted floor -- a confound, not an answer."""
    names = {t.name for t in tasks}
    for theme in lichess.UNANSWERABLE_THEMES:
        assert f"lichess_theme_{theme}" not in names, \
            f"{theme} is provenance, not a property of the position"


def test_no_task_asks_a_theme_the_state_prints(tasks):
    """The mirror failure. `_state` carries `OpeningTags`, and in the cached dump 96.8 %
    of `opening`-themed puzzles have one against 22.3 % of the pool, so the theme
    question would state its own answer. It sits just under the 200-row floor today; the
    exclusion is what stops a larger fetch from shipping it."""
    names = {t.name for t in tasks}
    for theme in lichess.STATED_IN_STATE_THEMES:
        assert f"lichess_theme_{theme}" not in names


def test_an_excluded_theme_cannot_come_back_through_the_counter():
    """The theme list is derived from whatever is cached, so the exclusion has to happen
    where the counting happens, not in a hand-maintained list further down."""
    rows = [{"PuzzleId": str(i), "FEN": f"f{i}", "Moves": "a1a2",
             "Themes": "['master', 'opening', 'fork']"} for i in range(400)]
    saved = lichess._ROWS
    lichess._ROWS = rows
    try:
        assert lichess._top_themes() == ["fork"]
    finally:
        lichess._ROWS = saved


def test_every_theme_carries_its_published_definition(tasks):
    """The criteria are the definition the model applies. Eight of the nineteen themes
    in an earlier build had criteria that only restated the question -- "The
    position and solution do not center on a rook endgame theme" -- which leaves the
    theme word as the only thing carrying what the theme is."""
    for t in tasks:
        if not t.name.startswith("lichess_theme_"):
            continue
        theme = t.name.removeprefix("lichess_theme_")
        for e in list(t.load(4)):
            for q in e.questions:
                assert q.descriptions and len(q.descriptions) == 2, t.name
                # the criterion has to say something the question does not
                asked = set(re.findall(r"[a-z]+", q.question.lower()))
                for d in q.descriptions:
                    novel = set(re.findall(r"[a-z]+", d.lower())) - asked - _STOP
                    assert len(novel) >= 4, f"{t.name}: {d!r} restates the question"
                assert lichess.THEME_CRITERIA[theme] in q.descriptions[1]


def test_a_theme_without_a_definition_fails_loudly(monkeypatch):
    """A new theme reaching the top of the count must not ship with no criteria."""
    monkeypatch.setattr(lichess, "_rows",
                        lambda: [{"FEN": "x", "Themes": "['fork']"}])
    monkeypatch.setattr(lichess, "_top_themes",
                        lambda: list(lichess.RESERVED_THEMES + lichess.DEV_THEMES)
                        + ["zugzwang"])
    with pytest.raises(ValueError, match="no published definition"):
        lichess.tasks()


def test_the_criteria_never_state_which_option_is_right(tasks):
    """Both criteria are the same sentence under a negation, so neither is the tell."""
    for t in tasks:
        if not t.name.startswith("lichess_theme_"):
            continue
        for e in list(t.load(4)):
            for q in e.questions:
                gold = q.options[q.target]
                for d in q.descriptions:
                    assert not re.search(r"(?<![a-z])" + gold + r"(?![a-z])", d.lower()), \
                        f"{t.name}: {d!r} names the gold option {gold!r}"
