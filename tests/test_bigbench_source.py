"""Row 2, bigbench: the option *order* was the answer.

bigbench publishes `multiple_choice_targets` curated rather than shuffled, and a lot of
its configs curate the correct answer into one slot. Measured over an earlier corpus
build's 22,439 row-2 questions: the gold option sat at index 0 in 39.4 % of them against
a 31.4 % chance baseline, and in 20 of the 114 tasks a single fixed index was the answer
at least 90 % of the time -- 17 of those at exactly 100 %, 4,142 questions. On
`testreal`, "always answer option 0" scored 0.498 against a 0.344 chance baseline.

Training reshuffles options, so no reported number was inflated by this; but the corpus
files carried it, and every consumer that does not augment read it. Row 36 and row 43
both assert the gold option is not at a fixed index. These are row 2's.

The loader streams from the Hub, so the tests here drive it against a stub dataset rather
than the network: what is being held in place is the permutation, not the fetch.
"""

from __future__ import annotations

import sys
import types
from collections import Counter

import pytest

from lod.corpus.services.sources.real import instructions as ins


# ---- the permutation itself ------------------------------------------------------

def test_shuffled_is_a_permutation_that_carries_its_target():
    opts = ["alpha", "beta", "gamma", "delta"]
    for gold in range(len(opts)):
        out, target = ins._shuffled(opts, gold, key=f"k{gold}")
        assert sorted(out) == sorted(opts)
        assert len(out) == len(opts)
        assert out[target] == opts[gold], "the target stopped pointing at the gold option"


def test_shuffled_is_stable_under_the_same_key_and_moves_under_a_different_one():
    """The key is the example's own text, so a rebuild, a bigger quota or a resumed
    stream all reproduce the same order -- and two different examples do not share one."""
    opts = list("abcdefgh")
    assert ins._shuffled(opts, 0, "same") == ins._shuffled(opts, 0, "same")
    orders = {tuple(ins._shuffled(opts, 0, f"ex{i}")[0]) for i in range(40)}
    assert len(orders) > 30, "the permutation barely depends on the key"


def test_the_gold_option_no_longer_sits_at_a_fixed_index():
    """The regression this exists for: a config that always lists the answer first."""
    idx = Counter(ins._shuffled(["w", "x", "y", "z"], 0, f"example {i}")[1]
                  for i in range(4000))
    assert set(idx) == {0, 1, 2, 3}
    top = max(idx.values()) / sum(idx.values())
    assert top < 0.30, f"one index is the answer {top:.1%} of the time"


# ---- the loader ------------------------------------------------------------------

def _stub_datasets(rows):
    mod = types.ModuleType("datasets")
    mod.load_dataset = lambda *a, **k: iter(rows)
    mod.get_dataset_config_names = lambda *a, **k: ["stub_config"]
    return mod


@pytest.fixture
def stub(monkeypatch):
    def install(rows):
        monkeypatch.setitem(sys.modules, "datasets", _stub_datasets(rows))
    return install


def _row(i, gold_first=True):
    opts = [f"answer {i}", f"wrong a {i}", f"wrong b {i}", f"wrong c {i}"]
    scores = [1, 0, 0, 0]
    if not gold_first:
        opts = opts[1:] + opts[:1]
        scores = scores[1:] + scores[:1]
    return {"inputs": f"the state of example {i}",
            "multiple_choice_targets": opts, "multiple_choice_scores": scores}


def test_the_loader_shuffles_a_config_that_always_lists_the_answer_first(stub):
    stub([_row(i) for i in range(400)])
    exs = list(ins._bigbench_loader("stub_config")(400))
    assert len(exs) == 400
    idx = Counter(e.questions[0].target for e in exs)
    assert set(idx) == {0, 1, 2, 3}, f"only {sorted(idx)} ever answered"
    assert max(idx.values()) / 400 < 0.35, f"still positional: {idx}"


def test_the_loader_keeps_the_gold_option_pointing_at_the_right_string(stub):
    """A shuffle that loses the target is worse than the leak it fixes."""
    stub([_row(i) for i in range(200)])
    for i, e in enumerate(ins._bigbench_loader("stub_config")(200)):
        q = e.questions[0]
        assert q.options[q.target] == f"answer {i}"
        assert sorted(q.options) == sorted(_row(i)["multiple_choice_targets"])
        assert e.state == f"the state of example {i}"


def test_the_loader_is_deterministic(stub):
    stub([_row(i) for i in range(50)])
    first = [(e.questions[0].options, e.questions[0].target)
             for e in ins._bigbench_loader("stub_config")(50)]
    stub([_row(i) for i in range(50)])
    second = [(e.questions[0].options, e.questions[0].target)
              for e in ins._bigbench_loader("stub_config")(50)]
    assert first == second


def test_a_smaller_quota_is_a_prefix_of_a_larger_one(stub):
    """The key is the example's text, not `made`, so the order a row gets does not move
    when the build asks for fewer of them."""
    stub([_row(i) for i in range(60)])
    many = [(e.questions[0].options, e.questions[0].target)
            for e in ins._bigbench_loader("stub_config")(60)]
    stub([_row(i) for i in range(60)])
    few = [(e.questions[0].options, e.questions[0].target)
           for e in ins._bigbench_loader("stub_config")(10)]
    assert few == many[:10]


def test_an_example_with_a_repeated_option_is_dropped(stub):
    """`options.index` cannot name a target among duplicates, and the packer would see
    two identical keys. Dropped, not disambiguated."""
    dupe = {"inputs": "s", "multiple_choice_targets": ["same", "same", "other"],
            "multiple_choice_scores": [1, 0, 0]}
    stub([dupe, _row(7)])
    exs = list(ins._bigbench_loader("stub_config")(10))
    assert len(exs) == 1 and exs[0].state == "the state of example 7"


def test_an_example_without_exactly_one_correct_answer_is_dropped(stub):
    none_right = {"inputs": "a", "multiple_choice_targets": ["p", "q"],
                  "multiple_choice_scores": [0, 0]}
    two_right = {"inputs": "b", "multiple_choice_targets": ["p", "q"],
                 "multiple_choice_scores": [1, 1]}
    stub([none_right, two_right, _row(3)])
    assert len(list(ins._bigbench_loader("stub_config")(10))) == 1


# ---- row 3 must not move ---------------------------------------------------------

def test_row_3_option_order_is_content_derived_and_untouched():
    """`instructions.py` serves row 3 as well, and NI's option set is `sorted(targets)`:
    an order fixed by the strings themselves, with the target found by lookup. Nothing
    here may reach it -- a shuffle would make the option list task-varying for a source
    whose whole schema is the stable output set."""
    options = ["negative", "neutral", "positive"]
    rows = {"t": [{"task_name": "t", "inputs": f"text {i}",
                   "targets": options[i % 3]} for i in range(30)]}
    ins._IN_PROCESS["by_task"] = rows
    try:
        exs = list(ins._ni_loader("t", options, "Classify the sentiment.")(30))
    finally:
        ins._IN_PROCESS.pop("by_task", None)
    assert len(exs) == 30
    for i, e in enumerate(exs):
        q = e.questions[0]
        assert q.options == options, "row 3's option order moved"
        assert q.options[q.target] == options[i % 3]
    assert Counter(e.questions[0].target for e in exs) == {0: 10, 1: 10, 2: 10}
