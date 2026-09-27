"""Scoring past the token budget, with the state encoded once.

A request whose options do not fit beside the state is split into option shards. The
state is prefilled once and its keys and values are cached; every shard is then a short
sequence -- the question's head plus a slice of its options -- that attends to the cached
state, to the head, and to its own option only. Shards are batched along the batch
dimension, so a 2,000-option question is a handful of parallel forwards over the head
and the options, never a re-encoding of the state.

This is exact, not an approximation, for a checkpoint trained with independent options
(`option_mode="independent"`): an option's hidden state is a function of the state, its
question's head and itself, and every option sits at the same position ids whichever
shard carries it. Its stage-1 score is therefore the same number an unsharded pass would
compute, up to floating-point reassociation. Normalisation is one global logsumexp over
every option's logit.

With a comparison layer the options meet once, after all of them are scored: the top
`mixer_topk` by stage-1 score are mixed on the vectors already computed, and the rest
keep their stage-1 mass (`OptionMixer.combine`). That is the same function the model
computes in one pass, so sharded and unsharded agree whenever both can run.

A sequential-option checkpoint is refused: its options see their predecessors, so a shard
boundary changes the answer.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from lod.model.scorer import distribution_features
from lod.model.packing import MIN_STATE_BUDGET, Packer, StateTruncated
from lod.schema import Question


@dataclass
class ShardedResult:
    probs: torch.Tensor           # [N] global distribution
    logits: torch.Tensor          # [N] final logits (temperature applied)
    stage1: torch.Tensor          # [N] independent scores
    confidence: float
    features: dict                # distribution_features of the global distribution
    n_shards: int
    state_tokens: int
    state_truncated: bool
    rows: torch.Tensor | None = None   # [N, H] option vectors, for re-mixing wider


def _state_cache(model, ids: list[int], device):
    """Prefill the state once -> (per-layer (K, V) with batch 1, state length)."""
    x = torch.tensor([ids], dtype=torch.long, device=device)
    pos = torch.arange(len(ids), device=device).unsqueeze(0)
    out = model.backbone(input_ids=x, position_ids=pos, use_cache=True)
    cache = out.past_key_values
    kv = [(layer.keys, layer.values) for layer in cache.layers]
    return kv, len(ids)


def _shards(head_len: int, opt_lens: list[int], budget: int) -> list[list[int]]:
    """Greedy: option indices per shard, head + options <= budget tokens each."""
    out, cur, used = [], [], head_len
    for i, n in enumerate(opt_lens):
        if head_len + n > budget:
            raise ValueError(f"option {i} alone ({n} tokens) does not fit beside the "
                             f"question head in a {budget}-token shard")
        if cur and used + n > budget:
            out.append(cur)
            cur, used = [], head_len
        cur.append(i)
        used += n
    if cur:
        out.append(cur)
    return out


@torch.no_grad()
def score_question(model, packer: Packer, state: str, q: Question, device,
                   shard_budget: int | None = None, shard_batch: int = 16,
                   overflow: str = "refuse", topk: int | None = None,
                   keep_rows: bool = False) -> ShardedResult:
    """Score one question over any number of options, the state encoded once.

    `shard_budget` caps head + options per shard (default: whatever the model's total
    budget leaves after the state). `overflow` is the state policy: `refuse` raises
    `StateTruncated` when the state alone exceeds `max_state_tokens`; `flag` truncates it
    and says so in the result.
    """
    if getattr(model, "option_mode", "sequential") != "independent":
        raise ValueError("sharded scoring is exact only for independent-option checkpoints")
    if getattr(model, "n_heads", 1) != 1:
        raise ValueError("sharded scoring supports single-head checkpoints")
    full = packer.state_ids(state)
    ids = full[: model.max_state_tokens]
    truncated = len(ids) < len(full)
    if truncated and overflow == "refuse":
        raise StateTruncated(len(ids), len(full))
    s = len(ids)
    # Shards are exact at any size, so keep them short: a shard of (window - state) tokens
    # batched 16-wide would build multi-GB masks and copy the cached state per row in every
    # layer. 2,048 is plenty for a head plus a slice of options.
    budget = shard_budget or min(model.max_total_tokens - s, 2048)
    if budget < MIN_STATE_BUDGET:
        raise ValueError("no room for a shard beside the state")

    head_ids, opts = packer._parts(q)
    h = len(head_ids)
    shards = _shards(h, [len(o) for o in opts], budget)
    kv, s = _state_cache(model, ids, device)
    dtype = kv[0][0].dtype
    neg = torch.finfo(dtype).min

    from transformers import DynamicCache

    rows = [None] * len(opts)
    # bound the mask ([B, L, s+L]) and the per-row cache copy to ~0.5 G elements a batch
    per_row = max(1, budget * (s + budget))
    shard_batch = max(1, min(shard_batch, int(5e8 // per_row)))
    for b0 in range(0, len(shards), shard_batch):
        group = shards[b0 : b0 + shard_batch]
        bsz = len(group)
        lens = [h + sum(len(opts[i]) for i in sh) for sh in group]
        L = max(lens)
        inp = torch.full((bsz, L), packer.pad_id, dtype=torch.long)
        pos = torch.zeros((bsz, L), dtype=torch.long)
        oid = torch.zeros((bsz, L), dtype=torch.long)
        valid = torch.zeros((bsz, L), dtype=torch.bool)
        last: list[tuple[int, int, int]] = []    # (batch row, token index, option index)
        for r, sh in enumerate(group):
            inp[r, :h] = torch.tensor(head_ids)
            pos[r, :h] = torch.arange(s, s + h)
            valid[r, :h] = True
            at = h
            for k, i in enumerate(sh):
                n = len(opts[i])
                inp[r, at : at + n] = torch.tensor(opts[i])
                pos[r, at : at + n] = torch.arange(s + h, s + h + n)
                oid[r, at : at + n] = k + 1
                valid[r, at : at + n] = True
                at += n
                last.append((r, at - 1, i))
        # [B, 1, L, s + L]: every real token sees the whole cached state; within the
        # shard the independent-option rule, causal; the diagonal always
        idx = torch.arange(L)
        causal = idx[None, :] <= idx[:, None]
        oj, oi = oid[:, None, :], oid[:, :, None]
        allowed = causal[None] & valid[:, None, :] & ((oj == 0) | (oj == oi))
        allowed |= torch.eye(L, dtype=torch.bool)[None]
        mask = torch.zeros(bsz, 1, L, s + L, dtype=dtype)
        mask[:, 0, :, s:] = torch.where(allowed, torch.zeros((), dtype=dtype),
                                        torch.full((), neg, dtype=dtype))
        cache = DynamicCache()
        for li, (k, v) in enumerate(kv):
            cache.update(k.expand(bsz, -1, -1, -1), v.expand(bsz, -1, -1, -1), li)
        out = model.backbone(input_ids=inp.to(device), position_ids=pos.to(device),
                             attention_mask=mask.to(device), past_key_values=cache,
                             use_cache=False)
        hid = out.last_hidden_state
        for r, t, i in last:
            rows[i] = hid[r, t]
        del cache, out

    R = torch.stack(rows)                                     # [N, H]
    return finish(model, R, len(shards), s, truncated, topk=topk, keep_rows=keep_rows)


@torch.no_grad()
def finish(model, R: torch.Tensor, n_shards: int, s: int, truncated: bool,
           topk: int | None = None, keep_rows: bool = False) -> ShardedResult:
    """Option vectors -> the global distribution and confidence. Separate so that a
    wider shortlist (`topk`) can re-mix vectors already computed."""
    k_mix = topk if topk is not None else model.mixer_topk
    n_opts = R.shape[0]
    with torch.autocast(device_type=R.device.type, enabled=False):
        s1 = model.head(R.float()).squeeze(-1)               # [N]
        valid_n = torch.ones(1, n_opts, dtype=torch.bool, device=s1.device)
        if model.mixer is not None:
            final = model.mixer.combine(R.float()[None], s1[None], valid_n, k_mix)[0]
        else:
            final = s1
        grp = torch.zeros(n_opts, dtype=torch.long, device=final.device)
        final = final / model.temperature_for(grp, final)
        probs = torch.softmax(final, -1)                      # one global logsumexp
        feats = distribution_features(final[None], valid_n)
        conf = float(probs.max())
        if model.confidence_mode == "head" and model.confidence_head is not None:
            k = min(n_opts, k_mix if model.mixer is not None else n_opts)
            short = s1.topk(k).indices
            pooled = R.float()[short].mean(0, keepdim=True)
            conf = float(model.confidence_head(pooled, feats)[0])
    return ShardedResult(probs=probs.cpu(), logits=final.cpu(), stage1=s1.cpu(),
                         confidence=conf, features={k: float(v[0]) for k, v in feats.items()},
                         n_shards=n_shards, state_tokens=s, state_truncated=truncated,
                         rows=R if keep_rows else None)
