"""Score one API request against a checkpoint, under the token-budget policy.

The state is never cut silently. Under `overflow="refuse"` a state that does not fit whole
is a `ValidationError` on `state`; under `"truncate"` it is cut and the result says so.
When the state fits but the questions do not fit beside it, an independent-option
checkpoint scores each question in option shards against the state encoded once
(`lod.model.sharded`), which is exact; a sequential-option checkpoint cannot shard
exactly, so it refuses instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from lod.model import flexattn
from lod.model.packing import Packer, StateTruncated, collate
from lod.schema import Example, Question
from lod.serving import api


@dataclass
class Scored:
    probs: dict[str, list[float]]      # question id -> distribution over its options
    confidence: dict[str, float]       # question id -> the checkpoint's confidence
    input_tokens: int
    state_tokens: int                  # state tokens actually read
    state_truncated: bool


@torch.no_grad()
def score_questions(model, state: str, questions: list[Question], device,
                    overflow: str = "refuse", flex: bool = False) -> Scored:
    """One packed forward pass when everything fits, option shards otherwise.

    `flex` builds the batch for the FlexAttention path (the model must have been loaded
    with `attn_impl="flex"`). Raises `api.ValidationError` for what cannot be served.
    """
    if overflow not in api.STATE_OVERFLOW:
        raise api.ValidationError("state_overflow", f"unknown policy {overflow!r}")
    pk = Packer.for_model(model, overflow="refuse" if overflow == "refuse" else "flag")
    independent = model.option_mode == "independent"
    try:
        packed = pk.pack(Example(state=state, questions=questions),
                         drop_questions_to_fit=False)
    except StateTruncated as e:
        if e.full > model.max_state_tokens:
            raise api.ValidationError(
                "state", f"the state is {e.full} tokens; this model reads at most "
                         f"{model.max_state_tokens}. Shorten it, or send "
                         f"state_overflow='truncate'") from None
        if independent:
            return _score_sharded(model, state, questions, device, overflow)
        raise api.ValidationError(
            "state", f"the state needs {e.full} tokens and these questions leave "
                     f"{e.used}; send fewer questions per request, or "
                     f"state_overflow='truncate'") from None
    if packed is None or (packed.state_truncated and independent
                          and packed.state_tokens_full <= model.max_state_tokens):
        # the state fits on its own; only the questions crowd it out
        if not independent:
            raise api.ValidationError(
                "state", "the state and questions do not fit in the token budget")
        return _score_sharded(model, state, questions, device, overflow)
    batch = (collate([packed], pk.pad_id, model.backbone_dtype,
                     pad_to_tile=flexattn.BLOCK_SIZE, dense_mask=False)
             if flex else collate([packed], pk.pad_id, model.backbone_dtype))
    res = model.score_batch(batch, device)
    probs, conf = res["probs"].cpu(), res["confidence"].cpu()
    return Scored(probs={q.id: probs[qi, :len(q.options)].tolist()
                         for qi, q in enumerate(questions)},
                  confidence={q.id: float(conf[qi]) for qi, q in enumerate(questions)},
                  input_tokens=len(packed), state_tokens=packed.state_tokens,
                  state_truncated=packed.state_truncated)


def _score_sharded(model, state: str, questions: list[Question], device,
                   overflow: str) -> Scored:
    from lod.model.sharded import score_question

    pk = Packer.for_model(model)
    out = Scored({}, {}, 0, 0, False)
    for q in questions:
        try:
            r = score_question(model, pk, state, q, device,
                               overflow="refuse" if overflow == "refuse" else "flag")
        except StateTruncated as e:
            raise api.ValidationError(
                "state", f"the state is {e.full} tokens; this model reads at most "
                         f"{e.used}. Shorten it, or send state_overflow='truncate'") from None
        except ValueError as e:
            raise api.ValidationError("questions", str(e)) from None
        out.probs[q.id] = r.probs.tolist()
        out.confidence[q.id] = float(r.confidence)
        out.state_tokens, out.state_truncated = r.state_tokens, r.state_truncated
        out.input_tokens = max(out.input_tokens, r.state_tokens) + pk.question_tokens(q)
    return out


def answer(model, body: dict, device, flex: bool = False) -> dict:
    """A wire request body -> the wire response. Raises `api.ValidationError` (422)."""
    _model_id, state, parsed = api.parse_request(body)
    overflow = api.overflow_policy(body)
    r = score_questions(model, state, [p.question for p in parsed.values()], device,
                        overflow, flex)
    return api.build_response(parsed, r.probs, r.confidence, r.input_tokens,
                              r.state_tokens, r.state_truncated)
