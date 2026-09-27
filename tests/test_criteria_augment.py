"""The criteria augmentations (`lod.training.data.augment_criteria`).

Two failure modes shaped them: if every "none of the above" option in training is the
correct answer, the phrase becomes perfectly predictive; and a fixed list of sentinel
wordings is memorised as strings rather than read. These tests pin the sentinel grammar,
the decoy, and the option-count symmetry that keeps the count from being a free label.
"""

from __future__ import annotations

import collections
import random

from lod import sentinel
from lod.training.data import (
    P_WITHHELD,
    augment_criteria,
    decoy_none,
    opaque_keys,
    strip_criteria,
    withhold_criteria,
)
from lod.model.packing import Packer, _is_withheld
from lod.schema import Example, Question


def described(n: int = 4) -> Question:
    return Question("d", "Which team?", [f"key{i}" for i in range(n)], 0, None, None,
                    [f"description of option {i}" for i in range(n)])


def sentinel_index(q) -> int | None:
    for i, d in enumerate(q.descriptions or []):
        if sentinel.is_sentinel_description(d):
            return i
    for i, o in enumerate(q.options):
        if sentinel.is_sentinel_key(o):
            return i
    return None


def test_the_grammar_is_large_and_its_halves_are_disjoint():
    assert sentinel.space_size() > 10_000
    assert len(sentinel.DESCRIPTIONS) == sentinel.space_size()
    rng = random.Random(0)
    train = {sentinel.sample(rng, "train")[1] for _ in range(4000)}
    heldout = {sentinel.sample(rng, "eval")[1] for _ in range(2000)}
    assert not train & heldout, "held-out wordings measure recall, not generalisation, if these meet"
    assert all(sentinel.split_of(d) == "train" for d in train)
    # the fixed legacy wordings are held out: a model may have memorised them, so it
    # must never be scored on them as if they were fresh
    assert all(sentinel.split_of(d) == "eval" for d in sentinel.LEGACY_DESCRIPTIONS)


def test_a_wording_recurs_only_a_handful_of_times():
    rng = random.Random(0)
    counts = collections.Counter(sentinel.sample(rng, "train")[1] for _ in range(40_000))
    assert counts.most_common(1)[0][1] <= 20


def test_opaque_keys_replace_the_key_and_keep_the_description():
    q = opaque_keys(described(3))
    assert q.options == ["opt_0", "opt_1", "opt_2"]
    assert q.target == 0


def test_opaque_keys_is_a_no_op_without_descriptions():
    bare = Question("d", "Which?", ["a", "b", "c"], 0)
    assert opaque_keys(bare) is bare


def test_withheld_drops_the_answer_and_points_at_the_sentinel():
    q = withhold_criteria(described(), random.Random(0))
    # position is deliberately not asserted: the sentinel is shuffled in, so the test
    # checks the property instead of the ordering. See `test_sentinel_is_not_always_last`.
    assert set(q.options) - {q.options[sentinel_index(q)]} == {"key1", "key2", "key3"}
    assert "key0" not in q.options                      # key0 was the answer, now gone
    assert q.target_probs()[sentinel_index(q)] == 1.0
    assert sum(q.target_probs()) == 1.0
    assert _is_withheld(q)
    assert q.meta["abstain"] is True


def test_decoy_drops_a_wrong_option_and_the_sentinel_is_not_the_answer():
    q = decoy_none(described(), random.Random(0))
    si = sentinel_index(q)
    assert q.target_probs()[si] == 0.0
    gold = q.options.index("key0")
    assert q.target_probs()[gold] == 1.0                # the answer is still present
    assert not _is_withheld(q)


def test_sentinel_is_not_always_last():
    """Always appending the sentinel last would make it findable by position.

    Appending it made spotting one free, so nothing ever forced the model to read a
    description to find one -- and a caller who puts "none of these" in the middle of
    their own option list would meet a model that had never seen that.
    """
    seen = {sentinel_index(withhold_criteria(described(5), random.Random(i)))
            for i in range(40)}
    assert len(seen) > 1, "the sentinel landed in the same position every time"
    assert seen - {4}, "the sentinel was always last"


def test_both_paths_produce_the_same_option_count():
    """The arity leak: withholding removed one and appended one, the decoy only appended.

    Per task that separated them perfectly -- `nvd_attackVector` P(correct | present) was
    1.00 at 4 options and 0.00 at 5. Free label, and raising the decoy rate would have
    made it more load-bearing, not less.
    """
    for n in (3, 5, 12):
        rng = random.Random(n)
        assert len(withhold_criteria(described(n), rng).options) == n
        assert len(decoy_none(described(n), rng).options) == n


def test_both_paths_share_one_precondition():
    soft = Question("d", "Which?", ["a", "b", "c"], [0.4, 0.35, 0.25])
    assert withhold_criteria(soft, random.Random(0)) is None
    assert decoy_none(soft, random.Random(0)) is None
    two = Question("d", "Which?", ["a", "b"], 0, None, None, ["x", "y"])
    assert withhold_criteria(two, random.Random(0)) is None
    assert decoy_none(two, random.Random(0)) is None


def test_the_sentinel_matches_its_neighbours_form():
    """Being the one described option in a bare list, or the one readable key among
    `opt_N`, is a tell that survives every wording change."""
    bare = Question("d", "Which?", ["a", "b", "c"], 0)
    q = decoy_none(bare, random.Random(0))
    assert not q.has_descriptions
    opaque = opaque_keys(described())
    q = decoy_none(opaque, random.Random(0))
    assert q.options == ["opt_0", "opt_1", "opt_3", "opt_4"] or all(
        o.startswith("opt_") for o in q.options)


def test_strip_criteria_removes_descriptions_only():
    q = strip_criteria(described(3))
    assert q.descriptions is None
    assert q.options == ["key0", "key1", "key2"]
    assert q.target == 0


def test_rates_match_the_delta_and_the_sentinel_is_usually_wrong():
    example = Example(state="s", questions=[described()], task="t")
    seen = collections.Counter()
    wordings = set()
    n = 8000
    for i in range(n):
        q = augment_criteria(example, random.Random(i)).questions[0]
        si = sentinel_index(q)
        if si is not None:
            wordings.add(q.descriptions[si])
            seen["withheld" if _is_withheld(q) else "decoy"] += 1
        elif q.options[0] == "opt_0":
            seen["opaque"] += 1
        elif not q.has_descriptions:
            seen["strip"] += 1
        else:
            seen["plain"] += 1

    assert abs(seen["withheld"] / n - P_WITHHELD) < 0.015
    assert 0.06 <= seen["strip"] / n <= 0.14                 # p_strip 0.10
    sentinels = seen["withheld"] + seen["decoy"]
    p_correct = seen["withheld"] / sentinels
    assert p_correct <= 0.20, f"the sentinel is correct {p_correct:.3f} of the time"
    assert len(wordings) > 500, "a small wording set is what caused the bug twice"


def test_augmentation_leaves_noul_alone():
    noul = Question("d", "Is it?", ["no", "yes"], 1)
    example = Example(state="s", questions=[noul], task="t")
    for i in range(200):
        q = augment_criteria(example, random.Random(i)).questions[0]
        assert q.options == ["no", "yes"]


def test_a_not_in_context_row_is_abstention_without_a_sentinel():
    """The other half of the delta: a question whose answer is not in the state at all
    supervises the confidence head toward 0 and carries no score target, so it costs no
    choice question its option set."""
    q = Question("g", "Is it?", ["no", "yes"], None, {"abstain": True})
    assert _is_withheld(q)
    packed = Packer.__new__(Packer)          # only _is_withheld is under test here
    assert packed is not None


def test_the_abstention_channel_reaches_the_batch():
    """End to end: a not-in-context row must arrive with no score target and with the
    abstain flag set, or the scoring loss trains on it and the confidence head does not.

    Both halves matter and they are in different files, so this is the only place the
    contract between them is checked.
    """
    from transformers import AutoTokenizer

    from lod.model.packing import collate

    try:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
    except Exception:                                   # no tokenizer cache here
        import pytest as _pytest
        _pytest.skip("tokenizer unavailable")
    packer = Packer(tok, 512, 1024)
    scored = Question("a", "Is it?", ["no", "yes"], [1.0, 0.0])
    abstain = Question("b", "Is it?", ["no", "yes"], None, {"abstain": True})
    withheld = withhold_criteria(described(), random.Random(0))
    ex = Example(state="a record with some fields in it",
                 questions=[scored, abstain, withheld], task="t")
    batch = collate([packer.pack(ex)], packer.pad_id)

    assert batch["has_target"].tolist() == [True, False, True]
    assert batch["withheld"].tolist() == [False, True, True]
    # the scoring loss sees the first and the third, never the second
    assert int((batch["has_target"] & ~batch["withheld"]).sum()) == 1
    # the confidence head sees all three, the last two as negatives
    assert int((batch["has_target"] | batch["withheld"]).sum()) == 3


def test_ordered_rules_are_never_withheld_stripped_churned_or_decoyed():
    """Domain 15's policy is ordered: dropping one rule's text makes the rest undecidable."""
    import random
    from lod.training.data import augment_criteria
    from lod.schema import Example, Question
    opts = ["hold", "raise", "lower", "alarm", "shutdown"]
    q = Question("a", "Which action?", opts, 2, {"ordered_rules": True}, "Apply in order.",
                 [f"rule {i}: ..." for i in range(5)])
    ex = Example(state="s", questions=[q], task="sensor_x")
    for seed in range(300):
        out = augment_criteria(ex, random.Random(seed), p_withheld=0.5, p_strip=0.3,
                               p_decoy=0.9, p_filler=0.9).questions[0]
        assert len(out.options) == 5 and out.target == 2
        assert out.descriptions == q.descriptions


def test_rule_criteria_are_never_stripped():
    """Domain 2: a band's bounds live only in its option descriptions."""
    import random
    from lod.training.data import augment_criteria
    from lod.schema import Example, Question
    q = Question("b", "Which band?", ["low", "mid", "high"], 1, {}, None,
                 ["weight < 10", "10 <= weight < 20", "weight >= 20"])
    for task in ("rule_band", "diff_semver_single", "sched_x"):
        ex = Example(state="s", questions=[q], task=task)
        for seed in range(200):
            out = augment_criteria(ex, random.Random(seed), p_withheld=0.0, p_opaque=0.0,
                                   p_strip=1.0, p_decoy=0.0, p_filler=0.0).questions[0]
            assert out.descriptions == q.descriptions, task
