"""The API wire format (`lod.serving.api`) and question independence over the wire.

These test the translation layer directly rather than through HTTP: it is the part that
has to be exactly right, and a test that needs a server can only run where one can.
"""

import pytest

from lod.serving.api import (
    MAX_CHOICE_OPTIONS,
    MAX_SCORE_LEVELS,
    MODEL_ID,
    ValidationError,
    build_answer,
    build_response,
    parse_question,
    parse_request,
    render_state,
)

STATE_OBJ = {"ticket": 4412, "text": "charged twice", "days_open": 3}


# ---- state rendering ---------------------------------------------------------

def test_string_state_is_verbatim():
    assert render_state("Ticket #4412: charged twice.") == "Ticket #4412: charged twice."


def test_object_state_is_indented_json_in_given_key_order():
    out = render_state(STATE_OBJ)
    assert out.startswith("{\n  \"ticket\": 4412")
    # keys keep the caller's order rather than being sorted
    assert out.index("ticket") < out.index("text") < out.index("days_open")


def test_array_state_renders():
    assert render_state([1, "a"]).startswith("[\n  1")


def test_bad_state_type_is_422_naming_the_field():
    with pytest.raises(ValidationError) as e:
        render_state(42)
    assert e.value.field == "state"


# ---- the three question types ------------------------------------------------

def test_noul_maps_to_no_yes_with_false_first():
    p = parse_question("angry", {"type": "noul", "instructions": "Is the customer angry?",
                                 "criteria": {"true": "hostile", "false": "calm"}})
    assert p.kind == "noul"
    assert p.question.options == ["no", "yes"]
    # options are ["no", "yes"], so the false description must come first
    assert p.question.descriptions == ["calm", "hostile"]
    assert p.question.question == "Is the customer angry?"


def test_noul_without_criteria_has_no_descriptions():
    p = parse_question("q", {"type": "noul", "instructions": "Rains?"})
    assert p.question.descriptions is None


def test_choice_keeps_key_order_and_maps_values_to_descriptions():
    p = parse_question("team", {"type": "choice", "instructions": "Which team?",
                                "criteria": {"billing": "payments", "technical": None}})
    assert p.question.options == ["billing", "technical"]
    assert p.question.descriptions == ["payments", None]
    assert p.keys == ["billing", "technical"]


def test_score_options_are_indices_and_legend_is_the_criteria():
    p = parse_question("prio", {"type": "score", "instructions": "How urgent?",
                                "criteria": ["can wait", "this week", "today"]})
    assert p.question.options == ["0", "1", "2"]
    assert p.question.descriptions == ["can wait", "this week", "today"]
    assert p.legend == {"0": "can wait", "1": "this week", "2": "today"}


def test_backtick_field_references_pass_through_untouched():
    text = "Look at the `status` field, not `created_at`."
    p = parse_question("q", {"type": "choice", "instructions": text,
                             "criteria": {"a": None, "b": None}})
    assert p.question.question == text


# ---- 422s ---------------------------------------

def test_no_option_cap():
    """No option cap: the token budget is the only limit."""
    assert MAX_CHOICE_OPTIONS is None
    criteria = {f"opt{i}": None for i in range(2000)}
    p = parse_question("q", {"type": "choice", "instructions": "?", "criteria": criteria})
    assert len(p.question.options) == 2000


def test_a_deployment_can_still_set_a_cap(monkeypatch):
    import lod.serving.api as api
    monkeypatch.setattr(api, "MAX_CHOICE_OPTIONS", 255)
    criteria = {f"opt{i}": None for i in range(256)}
    with pytest.raises(ValidationError) as e:
        parse_question("q", {"type": "choice", "instructions": "?", "criteria": criteria})
    assert e.value.field == "questions.q.criteria"


def test_state_overflow_policy():
    from lod.serving.api import overflow_policy
    assert overflow_policy({}) == "refuse"
    assert overflow_policy({"state_overflow": "truncate"}) == "truncate"
    with pytest.raises(ValidationError) as e:
        overflow_policy({"state_overflow": "drop"})
    assert e.value.field == "state_overflow"


@pytest.mark.parametrize("n", [1, MAX_SCORE_LEVELS + 1])
def test_422_on_bad_score_level_count(n):
    with pytest.raises(ValidationError) as e:
        parse_question("q", {"type": "score", "instructions": "?",
                             "criteria": [f"l{i}" for i in range(n)]})
    assert e.value.field == "questions.q.criteria"


@pytest.mark.parametrize("n", [2, MAX_SCORE_LEVELS])
def test_score_level_bounds_are_inclusive(n):
    p = parse_question("q", {"type": "score", "instructions": "?",
                             "criteria": [f"l{i}" for i in range(n)]})
    assert len(p.question.options) == n


def test_422_on_missing_instructions():
    with pytest.raises(ValidationError) as e:
        parse_question("q", {"type": "choice", "criteria": {"a": None, "b": None}})
    assert e.value.field == "questions.q.instructions"


def test_422_on_unknown_model_and_missing_state():
    with pytest.raises(ValidationError) as e:
        parse_request({"model": "gpt-9", "state": "s", "questions": {}})
    assert e.value.field == "model"
    with pytest.raises(ValidationError) as e:
        parse_request({"questions": {}})
    assert e.value.field == "state"


# ---- responses ----------------------------------------------------------------

def test_noul_answer_has_no_confidence_field():
    p = parse_question("q", {"type": "noul", "instructions": "?"})
    a = build_answer(p, [0.23, 0.77], confidence=0.9)
    assert a == {"type": "noul", "noul": 0.77}
    assert "confidence" not in a          # noul answers carry no confidence on the wire


def test_choice_answer_shape():
    p = parse_question("q", {"type": "choice", "instructions": "?",
                             "criteria": {"a": None, "b": None}})
    a = build_answer(p, [0.9, 0.1], confidence=0.88)
    assert a["type"] == "choice" and a["choice"] == "a"
    assert a["probabilities"] == {"a": 0.9, "b": 0.1}
    assert a["confidence"] == 0.88


def test_score_answer_is_the_expectation_not_the_argmax():
    p = parse_question("q", {"type": "score", "instructions": "?",
                             "criteria": ["lo", "mid", "hi"]})
    a = build_answer(p, [0.1, 0.3, 0.6], confidence=0.7)
    assert a["score"] == pytest.approx(0.1 * 0 + 0.3 * 1 + 0.6 * 2)
    assert a["legend"] == {"0": "lo", "1": "mid", "2": "hi"}
    assert a["probabilities"] == {"0": 0.1, "1": 0.3, "2": 0.6}


def test_response_envelope():
    _m, _s, parsed = parse_request({
        "model": "lod-latest", "state": "s",
        "questions": {"q": {"type": "noul", "instructions": "?"}}})
    r = build_response(parsed, {"q": [0.4, 0.6]}, {"q": 0.6}, input_tokens=77)
    assert r["model"] == MODEL_ID
    assert set(r["answers"]) == {"q"}
    # prefill-only: output_tokens is zero by construction, not by rounding
    assert r["usage"] == {"input_tokens": 77, "output_tokens": 0}


# ---- independence over the wire -------------------------------------------

def test_wire_independence(model, tok):
    """A question's answer must be identical alone or alongside the others.

    The wire path is where this could quietly break even though the model-level
    independence check passes: the
    request carries questions in a dict, and anything order- or neighbour-dependent in
    the translation would show up here and nowhere else.
    """
    import torch

    from lod.model.calibration import probabilities as probs_from_grouped
    from lod.model.scorer import grouped_logits
    from lod.model.packing import Packer, collate
    from lod.schema import Example

    packer = Packer(tok)

    def ask(body: dict) -> dict[str, list[float]]:
        _m, state, parsed = parse_request(body)
        questions = [p.question for p in parsed.values()]
        packed = packer.pack(Example(state=state, questions=questions))
        batch = collate([packed], packer.pad_id, torch.float32)
        with torch.no_grad():
            logits = model(batch["input_ids"], batch["position_ids"],
                           batch["attention_mask"], batch["option_pos"])
        grouped = grouped_logits(logits.float(), batch["option_group"],
                                 batch["option_slot"], batch["valid"])
        probs = probs_from_grouped(grouped, batch["valid"])
        return {qid: probs[i, : len(p.question.options)].tolist()
                for i, (qid, p) in enumerate(parsed.items())}

    q_team = {"type": "choice", "instructions": "Which team should handle this ticket?",
              "criteria": {"billing": "payments and refunds", "technical": "software faults",
                           "account": "logins"}}
    q_angry = {"type": "noul", "instructions": "Is the customer angry?",
               "criteria": {"true": "hostile", "false": "calm"}}
    q_prio = {"type": "score", "instructions": "How urgent is this?",
              "criteria": ["can wait", "this week", "today"]}

    alone = ask({"state": STATE_OBJ, "questions": {"team": q_team}})["team"]
    with_others = ask({"state": STATE_OBJ,
                       "questions": {"team": q_team, "angry": q_angry,
                                     "prio": q_prio}})["team"]
    reordered = ask({"state": STATE_OBJ,
                     "questions": {"prio": q_prio, "angry": q_angry,
                                   "team": q_team}})["team"]

    for other in (with_others, reordered):
        delta = max(abs(a - b) for a, b in zip(alone, other))
        assert delta < 1e-4, f"wire independence broken: max |delta p| = {delta:.3e}"
