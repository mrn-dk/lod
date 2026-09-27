"""The proper scoring rule behaves like one."""

import math

import torch

from lod.model.scorer import decision_loss, grouped_logits, probs_from_grouped


def test_ce_with_one_hot_is_minus_log_p():
    grouped = torch.tensor([[2.0, 1.0, 0.5]])
    valid = torch.ones(1, 3, dtype=torch.bool)
    target = torch.tensor([[0.0, 1.0, 0.0]])
    p = torch.softmax(grouped, dim=-1)[0, 1]
    loss = decision_loss(grouped, valid, target, "ce")
    assert math.isclose(loss.item(), -math.log(p.item()), rel_tol=1e-6)


def test_ce_with_soft_target_is_cross_entropy():
    grouped = torch.tensor([[0.3, -1.2, 2.0]])
    valid = torch.ones(1, 3, dtype=torch.bool)
    target = torch.tensor([[0.2, 0.5, 0.3]])
    logp = torch.log_softmax(grouped, dim=-1)
    want = -(target * logp).sum().item()
    assert math.isclose(decision_loss(grouped, valid, target, "ce").item(), want, rel_tol=1e-6)


def test_ce_is_minimised_by_the_target_distribution():
    """A proper scoring rule: predicting the true probability beats any other."""
    target = torch.tensor([[0.7, 0.3]])
    valid = torch.ones(1, 2, dtype=torch.bool)
    true = torch.tensor([[math.log(0.7), math.log(0.3)]])
    best = decision_loss(true, valid, target, "ce").item()
    for shift in (-1.0, -0.3, 0.3, 1.0):
        other = true + torch.tensor([[shift, 0.0]])
        assert decision_loss(other, valid, target, "ce").item() > best
    assert math.isclose(best, -(0.7 * math.log(0.7) + 0.3 * math.log(0.3)), rel_tol=1e-6)


def test_brier_matches_definition_and_is_proper():
    grouped = torch.tensor([[1.0, 0.0]])
    valid = torch.ones(1, 2, dtype=torch.bool)
    target = torch.tensor([[1.0, 0.0]])
    p = torch.softmax(grouped, dim=-1)
    want = ((p - target) ** 2).sum().item()
    assert math.isclose(decision_loss(grouped, valid, target, "brier").item(), want, rel_tol=1e-6)
    true = torch.tensor([[math.log(0.7), math.log(0.3)]])
    soft = torch.tensor([[0.7, 0.3]])
    best = decision_loss(true, valid, soft, "brier").item()
    assert decision_loss(true + torch.tensor([[0.5, 0.0]]), valid, soft, "brier").item() > best


def test_invalid_slots_never_produce_nan():
    grouped = torch.tensor([[2.0, 1.0, float("-inf")], [0.0, float("-inf"), float("-inf")]])
    valid = torch.tensor([[True, True, False], [True, False, False]])
    target = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    for kind in ("ce", "brier"):
        loss = decision_loss(grouped, valid, target, kind)
        assert torch.isfinite(loss), kind
    probs = probs_from_grouped(grouped, valid)
    assert torch.isfinite(probs).all()
    assert torch.allclose(probs.sum(-1), torch.ones(2))
    assert probs[0, 2] == 0.0 and probs[1, 1] == 0.0
    # a single-option question is a certainty, so its log score is 0
    assert math.isclose(decision_loss(grouped[1:], valid[1:], target[1:], "ce").item(), 0.0, abs_tol=1e-6)


def test_invalid_slots_get_no_gradient():
    logits = torch.tensor([2.0, 1.0, 0.5], requires_grad=True)
    valid = torch.tensor([[True, True, False], [True, False, False]])
    grouped = grouped_logits(
        logits, torch.tensor([0, 0, 1]), torch.tensor([0, 1, 0]), valid
    )
    target = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    decision_loss(grouped, valid, target, "ce").backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[2] == 0.0  # the only option of its question


def test_grouped_logits_scatters_and_masks():
    logits = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    group = torch.tensor([0, 0, 0, 1, 1])
    slot = torch.tensor([0, 1, 2, 0, 1])
    valid = torch.tensor([[True, True, True], [True, True, False]])
    g = grouped_logits(logits, group, slot, valid)
    assert g.shape == (2, 3)
    assert g[0].tolist() == [1.0, 2.0, 3.0]
    assert g[1, :2].tolist() == [4.0, 5.0]
    assert g[1, 2] == float("-inf")


def test_questions_without_targets_are_skipped():
    grouped = torch.tensor([[2.0, 1.0], [0.0, 0.0]])
    valid = torch.ones(2, 2, dtype=torch.bool)
    target = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    has = torch.tensor([True, False])
    only = decision_loss(grouped[:1], valid[:1], target[:1], "ce")
    assert torch.allclose(decision_loss(grouped, valid, target, "ce", has_target=has), only)


def test_solved_tracker_hysteresis():
    """Down-weight above 0.99, restore below 0.97, no flapping between."""
    from lod.training.data import SolvedTracker

    tr = SolvedTracker(factor=0.2, window=100, min_seen=50)
    assert tr.enabled and not tr.solved
    tr.update(["a"] * 60, [True] * 60)          # 1.00 -> solved
    assert tr.solved == {"a"}
    assert tr.weights(["a", "b"]) == [0.2, 1.0]
    tr.update(["a"] * 2, [False, False])        # 0.968 -> below off, restored
    assert tr.solved == set()
    # 0.98 is inside the band and it entered unsolved, so it must stay unsolved
    tr.update(["a"] * 40, [True] * 40)
    assert tr.solved == set(), "inside the band, an unsolved task stays unsolved"
    tr.update(["a"] * 100, [True] * 100)        # a clean window -> solved again
    assert tr.solved == {"a"}
    # in the hysteresis band a solved task stays solved
    tr2 = SolvedTracker(factor=0.2, window=100, min_seen=50)
    tr2.update(["c"] * 100, [True] * 100)
    tr2.update(["c"] * 2, [False, True])        # 0.98: between off and on
    assert tr2.solved == {"c"}, "0.98 is inside the band, must stay down-weighted"
    # a task below min_seen is never judged
    tr3 = SolvedTracker(factor=0.2, window=100, min_seen=50)
    tr3.update(["d"] * 10, [True] * 10)
    assert tr3.solved == set()
    assert SolvedTracker(factor=1.0).enabled is False
