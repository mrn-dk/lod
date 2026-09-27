"""Question independence and gradient flow.

Independence is the hard rule for any backbone: a question's logits must not
change when other questions are added, removed or reordered around it.
"""

import torch

from lod.model.scorer import decision_loss, grouped_logits
from lod.model.packing import Packer, collate
from lod.schema import Example, Question

STATE = (
    "Ticket #4412 from a customer: my card was charged twice for the same order "
    "and nobody has replied for three days. This is unacceptable."
)
Q_A = Question("team", "Which team should handle this ticket?",
               ["billing", "technical", "account", "shipping"], 0)
Q_B = Question("angry", "Is the customer angry?", ["no", "yes"], 1)
Q_C = Question("priority", "What priority should this ticket get?", ["low", "medium", "high"], 2)
OTHER = Example(task="other", state="A fair 6-sided die is rolled.",
                questions=[Question("gt3", "Does it show a number greater than 3?", ["no", "yes"], 1)])


def _grouped(model, packer, examples):
    batch = collate([packer.pack(e) for e in examples], packer.pad_id, torch.float32)
    with torch.no_grad():
        logits = model(batch["input_ids"], batch["position_ids"],
                       batch["attention_mask"], batch["option_pos"])
    return grouped_logits(logits, batch["option_group"], batch["option_slot"], batch["valid"])


def _q_a_logits(model, packer, examples, row):
    return _grouped(model, packer, examples)[row]


def test_question_logits_are_independent_of_context(model, capsys):
    packer = Packer(model.tokenizer)
    alone = Example(task="t", state=STATE, questions=[Q_A])
    a_then_b = Example(task="t", state=STATE, questions=[Q_A, Q_B])
    b_then_a = Example(task="t", state=STATE, questions=[Q_B, Q_A])
    crowded = Example(task="t", state=STATE, questions=[Q_B, Q_C, Q_A])

    ref = _q_a_logits(model, packer, [alone], 0)
    cases = {
        "A with B after it": ([a_then_b], 0),
        "A with B before it": ([b_then_a], 1),
        "A last of three": ([crowded], 2),
        "A batched with another example": ([OTHER, a_then_b], 1),
        "A batched, other example first and longer": ([b_then_a, OTHER, alone], 1),
    }
    worst = 0.0
    for name, (examples, row) in cases.items():
        got = _q_a_logits(model, packer, examples, row)
        delta = (got - ref).abs().max().item()
        worst = max(worst, delta)
        print(f"  {name:<42} max|delta| = {delta:.3e}")
    print(f"  INDEPENDENCE max|delta logit| = {worst:.3e}")
    with capsys.disabled():
        print(f"\n  independence max|delta logit| = {worst:.3e} (threshold 1e-4)")
    assert worst < 1e-4, f"independence violated: max|delta| = {worst}"


def test_full_causal_mask_breaks_independence(model):
    """The full-causal ablation must actually break independence."""
    packer = Packer(model.tokenizer, mask_mode="full-causal")
    ref = _q_a_logits(model, packer, [Example(task="t", state=STATE, questions=[Q_A])], 0)
    got = _q_a_logits(model, packer, [Example(task="t", state=STATE, questions=[Q_B, Q_A])], 1)
    assert (got - ref).abs().max().item() > 1e-3


def test_gradient_reaches_head_and_backbone(model):
    packer = Packer(model.tokenizer)
    batch = collate(
        [packer.pack(Example(task="t", state=STATE, questions=[Q_A, Q_B, Q_C]))],
        packer.pad_id,
        torch.float32,
    )
    model.zero_grad(set_to_none=True)
    logits = model(batch["input_ids"], batch["position_ids"], batch["attention_mask"], batch["option_pos"])
    grouped = grouped_logits(logits, batch["option_group"], batch["option_slot"], batch["valid"])
    loss = decision_loss(grouped, batch["valid"], batch["target"], "ce", batch["has_target"])
    loss.backward()

    head_lin = model.head[1]
    assert head_lin.weight.grad is not None and head_lin.weight.grad.abs().sum() > 0
    embed = model.backbone.get_input_embeddings().weight
    assert embed.grad is not None and embed.grad.abs().sum() > 0
    last = [p for n, p in model.backbone.named_parameters() if "layers.29" in n and p.grad is not None]
    assert last and any(p.grad.abs().sum() > 0 for p in last)
    model.zero_grad(set_to_none=True)


def test_probabilities_are_only_over_own_options(model):
    packer = Packer(model.tokenizer)
    grouped = _grouped(model, packer, [Example(task="t", state=STATE, questions=[Q_B, Q_A])])
    valid = torch.tensor([[True, True, False, False], [True, True, True, True]])
    probs = torch.softmax(grouped, dim=-1)
    assert grouped.shape == (2, 4)
    assert torch.isinf(grouped[0, 2:]).all() and (grouped[0, 2:] < 0).all()
    assert torch.allclose(probs.sum(-1), torch.ones(2), atol=1e-6)
    assert probs[0, 2:].sum() == 0.0
    assert valid.shape == grouped.shape


def test_save_load_roundtrip(model, tmp_path):
    packer = Packer(model.tokenizer)
    ex = Example(task="t", state=STATE, questions=[Q_A, Q_B])
    before = _grouped(model, packer, [ex])
    model.temperature.fill_(1.7)
    model.save(tmp_path / "ckpt")
    model.temperature.fill_(1.0)

    from lod.model.scorer import OptionScoringModel

    reloaded = OptionScoringModel.load(tmp_path / "ckpt", torch_dtype=torch.float32)
    reloaded.eval()
    assert abs(float(reloaded.temperature) - 1.7) < 1e-6
    assert reloaded.max_state_tokens == 512 and reloaded.max_total_tokens == 768
    after = _grouped(reloaded, Packer(reloaded.tokenizer), [ex])
    assert torch.allclose(before, after * 1.7, atol=1e-4, equal_nan=True)


def test_save_ships_no_chat_template(model, tmp_path):
    """A shipped chat template advertises an interface this model does not have.

    save_pretrained regenerates it from the backbone tokenizer every time, so deleting
    it from a published repo does not hold unless save() drops it first.
    """
    model.save(tmp_path / "ckpt")
    assert not (tmp_path / "ckpt" / "chat_template.jinja").exists()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(tmp_path / "ckpt"))
    assert getattr(tok, "chat_template", None) is None


def test_independence_holds_with_descriptions(model, tok):
    """The independence check, re-run with option descriptions present.

    Descriptions lengthen every option span and move every scored position, so if the
    block mask or the per-block positions were subtly wrong the extra tokens are exactly
    what would expose it. Same 1e-4 bar.
    """
    packer = Packer(tok)
    q_a = Question("team", "Which team should handle this ticket?",
                   ["billing", "technical", "account", "shipping"], 0,
                   instructions="Pick the team that owns the first action.",
                   descriptions=["payments and refunds", "software faults",
                                 "logins and profiles", "delivery"])
    q_b = Question("angry", "Is the customer angry?", ["no", "yes"], 1,
                   descriptions=["calm or neutral", "annoyed or hostile"])
    q_c = Question("priority", "What priority should this ticket get?",
                   ["low", "medium", "high"], 2,
                   descriptions=["can wait", "this week", "today"])

    alone = _q_a_logits(model, packer, [Example(task="t", state=STATE, questions=[q_a])], 0)
    with_b = _q_a_logits(model, packer,
                         [Example(task="t", state=STATE, questions=[q_a, q_b])], 0)
    with_all = _q_a_logits(model, packer,
                           [Example(task="t", state=STATE, questions=[q_a, q_b, q_c])], 0)
    reordered = _q_a_logits(model, packer,
                            [Example(task="t", state=STATE, questions=[q_c, q_b, q_a])], 2)

    for other in (with_b, with_all, reordered):
        delta = (alone - other).abs().max().item()
        assert delta < 1e-4, f"independence broken with descriptions: {delta:.3e}"
