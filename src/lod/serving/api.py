"""The API wire format, as a pure translation layer.

This module maps a wire request onto `Question`s on the way in and the calibrated
distributions onto wire answers on the way out; `lod.serving.scoring` runs the model and
`scripts/serve.py` is a thin HTTP shell around both. Kept out of the request handler so the
translation -- the part that has to be exactly right -- can be tested without a server.

Three question types, each mapping onto the same option machinery:

    noul    -> options ["no", "yes"],           descriptions [criteria.false, criteria.true]
    choice  -> options = criteria keys in order, descriptions = their values
    score   -> options ["0".."n-1"],            descriptions = the criteria list, low to high

`instructions` becomes `Question.question` in all three.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable

MODELS = ("lod-latest", "lod-1.0.0")
MODEL_ID = "lod-1.0.0"

# No option cap: `collate` sizes the option axis per batch and there is no per-slot
# embedding. The limit is the token budget, and an independent-option checkpoint scores
# past it by sharding (lod/model/sharded.py). Kept as a name so a deployment can set one.
MAX_CHOICE_OPTIONS: int | None = None
MIN_SCORE_LEVELS = 2
MAX_SCORE_LEVELS = 10

KINDS = ("noul", "choice", "score")


class ValidationError(ValueError):
    """422. `field` names the offending field."""

    def __init__(self, field: str, message: str) -> None:
        super().__init__(message)
        self.field = field
        self.message = message

    def as_detail(self) -> dict[str, str]:
        return {"field": self.field, "message": self.message}


# ---- state ------------------------------------------------------------------

def render_state(state: Any) -> str:
    """A string is used verbatim; an object or array becomes indented JSON.

    Key order is preserved rather than sorted: the caller chose an order and a record
    whose fields move between requests is a different state to the model.
    """
    if isinstance(state, str):
        return state
    if isinstance(state, (dict, list)):
        return json.dumps(state, indent=2, ensure_ascii=False, sort_keys=False)
    raise ValidationError("state", "state must be a string, object or array")


# ---- request -> Question -----------------------------------------------------

@dataclass
class Parsed:
    """One wire question, converted, with what the response needs to render it back."""

    kind: str
    question: Any            # lod.schema.Question
    keys: list[str]          # choice: the original keys; score: "0".."n-1"
    legend: dict[str, str]   # score only


def _instructions(qid: str, spec: dict) -> str:
    text = spec.get("instructions")
    if not isinstance(text, str) or not text.strip():
        raise ValidationError(f"questions.{qid}.instructions",
                              "instructions is required and must be a non-empty string")
    return text


def parse_question(qid: str, spec: dict) -> Parsed:
    from lod.schema import Question

    if not isinstance(spec, dict):
        raise ValidationError(f"questions.{qid}", "question must be an object")
    kind = spec.get("type")
    if kind not in KINDS:
        raise ValidationError(f"questions.{qid}.type",
                              f"type must be one of {', '.join(KINDS)}")
    instructions = _instructions(qid, spec)
    criteria = spec.get("criteria")

    if kind == "noul":
        descriptions = None
        if criteria is not None:
            if not isinstance(criteria, dict):
                raise ValidationError(f"questions.{qid}.criteria",
                                      "noul criteria must be an object with true/false")
            # order matters: options are ["no", "yes"], so false's description comes first
            descriptions = [criteria.get("false"), criteria.get("true")]
            if not any(descriptions):
                descriptions = None
        q = Question(id=qid, question=instructions, options=["no", "yes"],
                     descriptions=descriptions)
        return Parsed("noul", q, ["no", "yes"], {})

    if kind == "choice":
        if not isinstance(criteria, dict) or not criteria:
            raise ValidationError(f"questions.{qid}.criteria",
                                  "choice criteria must be a non-empty object")
        keys = [str(k) for k in criteria]
        if MAX_CHOICE_OPTIONS is not None and len(keys) > MAX_CHOICE_OPTIONS:
            raise ValidationError(f"questions.{qid}.criteria",
                                  f"choice supports at most {MAX_CHOICE_OPTIONS} options, "
                                  f"got {len(keys)}")
        if len(keys) < 2:
            raise ValidationError(f"questions.{qid}.criteria",
                                  "choice needs at least 2 options")
        descriptions = [criteria[k] if criteria[k] else None for k in criteria]
        q = Question(id=qid, question=instructions, options=keys,
                     descriptions=descriptions if any(descriptions) else None)
        return Parsed("choice", q, keys, {})

    # score
    if not isinstance(criteria, list):
        raise ValidationError(f"questions.{qid}.criteria",
                              "score criteria must be a list of level descriptions")
    if not MIN_SCORE_LEVELS <= len(criteria) <= MAX_SCORE_LEVELS:
        raise ValidationError(f"questions.{qid}.criteria",
                              f"score supports {MIN_SCORE_LEVELS}-{MAX_SCORE_LEVELS} "
                              f"levels, got {len(criteria)}")
    keys = [str(i) for i in range(len(criteria))]
    descriptions = [c if c else None for c in criteria]
    legend = {k: str(criteria[i]) for i, k in enumerate(keys)}
    q = Question(id=qid, question=instructions, options=keys,
                 descriptions=descriptions if any(descriptions) else None)
    return Parsed("score", q, keys, legend)


STATE_OVERFLOW = ("refuse", "truncate")


def overflow_policy(body: dict) -> str:
    """`state_overflow`: `refuse` (default) answers 422 when the state does not fit whole;
    `truncate` scores what fits and reports `usage.state_truncated` and how many state
    tokens were used. Options never silently cut the state."""
    policy = body.get("state_overflow", "refuse") if isinstance(body, dict) else "refuse"
    if policy not in STATE_OVERFLOW:
        raise ValidationError("state_overflow",
                              f"state_overflow must be one of {', '.join(STATE_OVERFLOW)}")
    return policy


def parse_request(body: dict) -> tuple[str, str, dict[str, Parsed]]:
    """-> (model, rendered state, {id: Parsed}). Raises ValidationError (422)."""
    if not isinstance(body, dict):
        raise ValidationError("body", "request body must be an object")
    model = body.get("model", "lod-latest")
    if model not in MODELS:
        raise ValidationError("model", f"unknown model {model!r}; "
                                       f"expected one of {', '.join(MODELS)}")
    if "state" not in body:
        raise ValidationError("state", "state is required")
    state = render_state(body["state"])

    questions = body.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise ValidationError("questions", "questions must be a non-empty object "
                                           "keyed by question id")
    parsed = {qid: parse_question(qid, spec) for qid, spec in questions.items()}
    return model, state, parsed


# ---- distribution -> wire answer ---------------------------------------------

def build_answer(p: Parsed, probs: Iterable[float], confidence: float | None) -> dict:
    """The wire answer for one question, given its calibrated distribution."""
    probs = [float(x) for x in probs]
    if p.kind == "noul":
        # P(yes) only, and no confidence field on the wire -- eval still computes one
        return {"type": "noul", "noul": probs[1] if len(probs) > 1 else 0.0}

    table = {k: probs[i] if i < len(probs) else 0.0 for i, k in enumerate(p.keys)}
    if p.kind == "choice":
        best = max(table, key=lambda k: table[k]) if table else ""
        out = {"type": "choice", "choice": best, "probabilities": table}
    else:
        expected = sum(i * probs[i] for i in range(min(len(probs), len(p.keys))))
        out = {"type": "score", "score": expected, "legend": p.legend,
               "probabilities": table}
    if confidence is not None:
        out["confidence"] = float(confidence)
    return out


def build_response(parsed: dict[str, Parsed], probs: dict[str, list[float]],
                   confidence: dict[str, float] | None, input_tokens: int,
                   state_tokens: int | None = None, state_truncated: bool = False) -> dict:
    conf = confidence or {}
    usage = {"input_tokens": int(input_tokens), "output_tokens": 0}
    if state_tokens is not None:
        usage["state_tokens"] = int(state_tokens)
        usage["state_truncated"] = bool(state_truncated)
    return {
        "model": MODEL_ID,
        "answers": {qid: build_answer(p, probs.get(qid, []), conf.get(qid))
                    for qid, p in parsed.items()},
        # prefill-only: there are no output tokens, by construction
        "usage": usage,
    }
