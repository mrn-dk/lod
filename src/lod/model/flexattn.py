"""The question-block attention mask expressed as a predicate, for FlexAttention.

`packing.build_block_mask` materialises the rule as a dense additive `[B, 1, T, T]`
tensor. That is correct and portable, but it forces SDPA off its flash path: the kernel
expands the mask to `[B, heads, T, T]` per layer, which at batch 32 is ~600 MB of traffic
*per layer* for a mask that is mostly -inf. Measured consequence: ~11-19 % MFU, and
batch 32 without gradient checkpointing OOMs at 95 GB.

FlexAttention takes the same rule as a *predicate* and skips the tiles that are entirely
masked, so the mask is never materialised. The allowed set is identical, so this is an
exact reformulation and not an approximation -- the only difference in the output is
float reassociation from the online softmax. `tests/test_flexattn.py` asserts that
equivalence against the dense path; `scripts/check_independence.py` remains the semantic
gate.
"""

from __future__ import annotations

import os

import torch

# FlexAttention skips work at tile granularity, so this trades two things off: smaller
# tiles track the question blocks more closely and skip more, but the Triton kernels
# require the tile to divide their own BLOCK_M/BLOCK_N (32 is rejected outright by the
# backward pass) and get less efficient as they shrink. 64 is the smallest that works.
# Overridable for benchmarking.
BLOCK_SIZE = int(os.environ.get("LOD_FLEX_BLOCK", "64"))


def available() -> bool:
    try:
        from torch.nn.attention.flex_attention import create_block_mask  # noqa: F401
        from transformers.integrations.flex_attention import BlockMask  # noqa: F401
    except Exception:
        return False
    return True


def round_len(t: int, block: int = BLOCK_SIZE) -> int:
    """Pad length up to a whole number of tiles.

    Two reasons. `create_block_mask` wants the sequence to tile evenly, and `mask_mod`
    indexes `block_ids[b, kv_idx]`, which must stay in bounds. It also collapses the
    number of distinct shapes `torch.compile` sees, which with length bucketing would
    otherwise be a recompile per batch.
    """
    return ((t + block - 1) // block) * block


def block_mask_mod(block_ids: torch.Tensor, token_valid: torch.Tensor,
                   option_ids: torch.Tensor | None = None):
    """The rule from `packing.build_block_mask`, transcribed as a predicate.

    Token i may attend to token j iff j <= i and j is not padding and
    (block[j] == 0 or (block[j] == block[i] and (opt[j] == 0 or opt[j] == opt[i]))).
    The diagonal is always allowed so that no row is fully masked -- dropping that
    clause makes padding rows produce NaN.
    """
    if option_ids is None:
        def mask_mod(b, h, q_idx, kv_idx):
            causal = kv_idx <= q_idx
            valid = token_valid[b, kv_idx]
            bj = block_ids[b, kv_idx]
            bi = block_ids[b, q_idx]
            same = (bj == 0) | (bj == bi)
            return (causal & valid & same) | (q_idx == kv_idx)
        return mask_mod

    def mask_mod_opt(b, h, q_idx, kv_idx):
        causal = kv_idx <= q_idx
        valid = token_valid[b, kv_idx]
        bj = block_ids[b, kv_idx]
        bi = block_ids[b, q_idx]
        oj = option_ids[b, kv_idx]
        oi = option_ids[b, q_idx]
        same = (bj == 0) | ((bj == bi) & ((oj == 0) | (oj == oi)))
        return (causal & valid & same) | (q_idx == kv_idx)

    return mask_mod_opt


_compiled_create = None


def _create_block_mask(compile_mask: bool):
    """`create_block_mask` is itself worth compiling -- it runs once per batch.

    `dynamic=True`, and that is not a detail. Length bucketing gives a different T per
    batch; rounding to the 64-token tile still leaves 48 distinct values up to the 3,072
    budget. Under `dynamic=False` dynamo recompiles per shape, hits its recompile limit
    of 8, and then falls back to **eager** for every batch after that -- where
    `create_block_mask` materialises a full [B, 1, T, T] mask and reduces it, which is
    the exact cost expressing the rule as a predicate was supposed to avoid.
    """
    from torch.nn.attention.flex_attention import create_block_mask

    if not compile_mask:
        return create_block_mask
    global _compiled_create
    if _compiled_create is None:
        _compiled_create = torch.compile(create_block_mask, dynamic=True)
    return _compiled_create


def build_flex_mask(block_ids: torch.Tensor, token_valid: torch.Tensor,
                    compile_mask: bool = True, option_ids: torch.Tensor | None = None):
    """`BlockMask` for a batch. `block_ids`/`token_valid` are `[B, T]`, T a tile multiple."""
    b, t = block_ids.shape
    if t % BLOCK_SIZE:
        raise ValueError(f"sequence length {t} is not a multiple of {BLOCK_SIZE}; "
                         f"collate(..., pad_to_tile=...) does this")
    return _create_block_mask(compile_mask)(
        block_mask_mod(block_ids, token_valid, option_ids),
        B=b, H=None, Q_LEN=t, KV_LEN=t,
        device=block_ids.device,
        BLOCK_SIZE=BLOCK_SIZE,
    )


def kernel_options() -> dict | None:
    """Triton block sizes, when the tile is smaller than the kernel's own default.

    The kernels insist the tile divide their BLOCK_M/BLOCK_N. Measured on the RTX PRO
    6000 (torch 2.11): tile 128 works out of the box, tile 64 needs BLOCK_M=BLOCK_N=32,
    and tiles of 32 or below are rejected whatever you ask for -- the backward pass has
    no valid kernel. Tile 64 is worth having: against the causal mask flash attention
    already applies for free, it skips 17 % of tiles where 128 skips 8 %.
    """
    if BLOCK_SIZE >= 128:
        return None
    return {"BLOCK_M": 32, "BLOCK_N": 32}


def sparsity(block_ids: torch.Tensor, token_valid: torch.Tensor,
             option_ids: torch.Tensor | None = None) -> float:
    """Fraction of tiles FlexAttention can skip. If this is near 0 there is no win."""
    bm = build_flex_mask(block_ids, token_valid, compile_mask=False, option_ids=option_ids)
    return float(bm.sparsity()) / 100.0
