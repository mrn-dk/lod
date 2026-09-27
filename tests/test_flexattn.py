"""The FlexAttention path must be an exact reformulation, not an approximation.

Two claims, tested separately because they fail for different reasons:

1. the predicate in `flexattn.block_mask_mod` admits exactly the same (i, j) pairs as
   the dense `packing.build_block_mask` -- a pure logic check, no model, no GPU;
2. running a real backbone through FlexAttention gives the same option logits as the
   dense-mask SDPA path, up to the float reassociation of an online softmax.

(1) is the one that catches a wrong transcription. (2) is the one that catches HF
routing the mask somewhere unexpected. Neither is a substitute for the independence
check (test_model.py), which is what catches a semantically-wrong mask that is
self-consistent.
"""

import pytest
import torch

from lod.model.flexattn import BLOCK_SIZE, available, block_mask_mod, round_len
from lod.model.scorer import OptionScoringModel, grouped_logits
from lod.model.packing import Packer, build_block_mask, collate
from lod.schema import Example

from test_model import OTHER, Q_A, Q_B, Q_C, STATE


def _dense_from_mod(block_ids, token_valid):
    """Evaluate the FlexAttention predicate over every (b, i, j) as a bool tensor."""
    b, t = block_ids.shape
    idx = torch.arange(t)
    bb = torch.arange(b)[:, None, None].expand(b, t, t)
    qi = idx[None, :, None].expand(b, t, t)
    kv = idx[None, None, :].expand(b, t, t)
    return block_mask_mod(block_ids, token_valid)(bb, None, qi, kv)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_predicate_matches_dense_mask(seed):
    """The rule, both ways round, on random block layouts including ragged padding."""
    g = torch.Generator().manual_seed(seed)
    b, t = 4, 64
    block_ids = torch.zeros(b, t, dtype=torch.long)
    token_valid = torch.zeros(b, t, dtype=torch.bool)
    for i in range(b):
        n = int(torch.randint(8, t + 1, (1,), generator=g))
        token_valid[i, :n] = True
        state = int(torch.randint(1, max(2, n // 2), (1,), generator=g))
        cur, pos = 1, state
        while pos < n:  # state is block 0, then question blocks 1..k
            width = int(torch.randint(1, 8, (1,), generator=g))
            block_ids[i, pos : pos + width] = cur
            pos += width
            cur += 1

    dense = build_block_mask(block_ids, token_valid, torch.float32)
    allowed_dense = dense[:, 0] == 0.0
    allowed_mod = _dense_from_mod(block_ids, token_valid)
    assert torch.equal(allowed_dense, allowed_mod)


def test_predicate_keeps_the_diagonal():
    """Padding rows must stay attendable to themselves or the softmax produces NaN."""
    block_ids = torch.zeros(2, 32, dtype=torch.long)
    token_valid = torch.zeros(2, 32, dtype=torch.bool)
    token_valid[:, :4] = True
    allowed = _dense_from_mod(block_ids, token_valid)
    assert allowed.diagonal(dim1=1, dim2=2).all()
    assert allowed.any(dim=-1).all()


def test_round_len_is_a_tile_multiple():
    for t in (1, 31, 32, 33, 126, 540):
        r = round_len(t)
        assert r >= t and r % BLOCK_SIZE == 0


def test_collate_pad_to_tile_changes_nothing_but_length(tok):
    packer = Packer(tok)
    items = [packer.pack(e) for e in (Example(task="t", state=STATE, questions=[Q_A, Q_B, Q_C]), OTHER)]
    plain = collate(items, packer.pad_id, torch.float32)
    tiled = collate(items, packer.pad_id, torch.float32, pad_to_tile=BLOCK_SIZE)
    t0 = plain["input_ids"].shape[1]
    assert tiled["input_ids"].shape[1] == round_len(t0)
    # the real tokens, their blocks and their option positions are untouched
    assert torch.equal(plain["input_ids"], tiled["input_ids"][:, :t0])
    assert torch.equal(plain["block_ids"], tiled["block_ids"][:, :t0])
    assert torch.equal(plain["token_valid"], tiled["token_valid"][:, :t0])
    assert not tiled["token_valid"][:, t0:].any()


def test_collate_can_skip_the_dense_mask(tok):
    packer = Packer(tok)
    batch = collate([packer.pack(OTHER)], packer.pad_id, torch.float32, dense_mask=False)
    assert batch["attention_mask"] is None
    assert batch["block_ids"] is not None and batch["token_valid"] is not None


@pytest.mark.skipif(not available(), reason="flex_attention unavailable")
def test_flex_logits_match_sdpa(tok):
    """The claim that matters: same checkpoint, same batch, both paths, same logits."""
    from conftest import BACKBONE

    examples = [Example(task="t", state=STATE, questions=[Q_A, Q_B, Q_C]), OTHER]
    packer = Packer(tok)
    items = [packer.pack(e) for e in examples]

    def run(attn_impl):
        torch.manual_seed(0)
        m = OptionScoringModel(BACKBONE, torch_dtype=torch.float32, lora_r=0,
                           tokenizer=tok, attn_impl=attn_impl)
        m.eval()
        tile = BLOCK_SIZE if attn_impl == "flex" else 1
        batch = collate(items, packer.pad_id, torch.float32, pad_to_tile=tile,
                        dense_mask=(attn_impl == "sdpa"))
        with torch.no_grad():
            logits = m(batch["input_ids"], batch["position_ids"],
                       batch["attention_mask"], batch["option_pos"],
                       block_ids=batch["block_ids"], token_valid=batch["token_valid"])
        return grouped_logits(logits, batch["option_group"], batch["option_slot"],
                              batch["valid"])

    a = run("sdpa")
    try:
        b = run("flex")
    except Exception as e:  # Triton compiles a C shim; without python3-dev it cannot
        if "Python.h" in str(e) or "CalledProcessError" in type(e).__name__:
            pytest.skip(f"triton cannot compile here (install python3-dev): {e}")
        raise
    finite = torch.isfinite(a) & torch.isfinite(b)
    delta = (a[finite] - b[finite]).abs().max().item()
    assert delta < 1e-3, f"flex and sdpa disagree by {delta:.3e}"


@pytest.mark.skipif(not available(), reason="flex_attention unavailable")
def test_flex_matches_sdpa_with_independent_options(tok):
    """The option-level rule, both paths: option_ids must reach the predicate too."""
    from conftest import BACKBONE

    from lod.model.scorer import forward_batch

    examples = [Example(task="t", state=STATE, questions=[Q_A, Q_B, Q_C]), OTHER]
    packer = Packer(tok, option_mode="independent")
    items = [packer.pack(e) for e in examples]

    def run(attn_impl):
        torch.manual_seed(0)
        m = OptionScoringModel(BACKBONE, torch_dtype=torch.float32, tokenizer=tok,
                               attn_impl=attn_impl, option_mode="independent")
        m.eval()
        tile = BLOCK_SIZE if attn_impl == "flex" else 1
        batch = collate(items, packer.pad_id, torch.float32, pad_to_tile=tile,
                        dense_mask=(attn_impl == "sdpa"))
        with torch.no_grad():
            logits = forward_batch(m, batch, "cpu")
        return grouped_logits(logits, batch["option_group"], batch["option_slot"],
                              batch["valid"])

    a = run("sdpa")
    try:
        b = run("flex")
    except Exception as e:
        if "Python.h" in str(e) or "CalledProcessError" in type(e).__name__:
            pytest.skip(f"triton cannot compile here (install python3-dev): {e}")
        raise
    finite = torch.isfinite(a) & torch.isfinite(b)
    delta = (a[finite] - b[finite]).abs().max().item()
    assert delta < 1e-3, f"flex and sdpa disagree by {delta:.3e}"
