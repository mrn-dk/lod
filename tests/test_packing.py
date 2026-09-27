"""Packing: the block mask rule, restarting positions, option positions, budget."""

import torch

from lod.model.packing import Packer, collate
from lod.schema import Example, Question


def two_question_example() -> Example:
    return Example(
        task="toy",
        state="the movie was a slog",
        questions=[
            Question("sentiment", "What is the sentiment of the text?", ["negative", "positive"], 0),
            Question("length", "More than 10 words?", ["no", "yes"], 0),
        ],
    )


def one_question_example() -> Example:
    return Example(
        task="toy",
        state="short",
        questions=[Question("q", "Is this short?", ["no", "yes"], 1)],
    )


def test_block_ids_and_positions_restart(tok):
    p = Packer(tok).pack(two_question_example())
    s = p.block_ids.index(1)  # first question token
    assert p.block_ids[:s] == [0] * s
    assert set(p.block_ids[s:]) == {1, 2}
    assert p.position_ids[:s] == list(range(s))
    # every question block restarts at S
    for qi in (1, 2):
        pos = [pp for pp, b in zip(p.position_ids, p.block_ids) if b == qi]
        assert pos == list(range(s, s + len(pos))), f"block {qi} positions must restart at {s}"


def test_option_pos_points_at_last_option_token(tok):
    ex = two_question_example()
    p = Packer(tok).pack(ex)
    assert p.option_group == [0, 0, 1, 1]
    assert p.option_slot == [0, 1, 0, 1]
    assert p.n_options == [2, 2]
    for pos, g, slot in zip(p.option_pos, p.option_group, p.option_slot):
        opt = ex.questions[g].options[slot]
        tail = tok.decode([p.input_ids[pos]])
        assert opt.endswith(tail.strip()), f"{tail!r} is not the tail of {opt!r}"
        # the next token starts the following option (or the sequence ends)
        if pos + 1 < len(p.input_ids):
            assert tok.decode([p.input_ids[pos + 1]]).startswith("\n")


def test_positions_are_independent_of_question_count(tok):
    packer = Packer(tok)
    ex = two_question_example()
    solo = Example(task="toy", state=ex.state, questions=[ex.questions[1]])
    both = packer.pack(ex)
    only = packer.pack(solo)
    q2_pos = [pp for pp, b in zip(both.position_ids, both.block_ids) if b == 2]
    q_pos = [pp for pp, b in zip(only.position_ids, only.block_ids) if b == 1]
    assert q2_pos == q_pos


def test_mask_rule_including_padding(tok):
    packer = Packer(tok)
    items = [packer.pack(two_question_example()), packer.pack(one_question_example())]
    batch = collate(items, packer.pad_id, torch.float32)
    mask = batch["attention_mask"]
    b, _, t, _ = mask.shape
    assert mask.shape == (2, 1, t, t)
    neg = torch.finfo(torch.float32).min
    allowed = mask[:, 0] == 0.0
    blocked = mask[:, 0] == neg
    assert (allowed | blocked).all(), "mask must be exactly 0 or dtype min"

    block_ids = batch["block_ids"]
    valid = batch["token_valid"]
    for bi in range(b):
        for i in range(t):
            for j in range(t):
                rule = (
                    j <= i
                    and bool(valid[bi, j])
                    and (int(block_ids[bi, j]) == 0 or block_ids[bi, j] == block_ids[bi, i])
                )
                want = rule or i == j  # diagonal always on
                assert bool(allowed[bi, i, j]) == want, (bi, i, j)

    # the spelled-out consequences
    ex0 = items[0]
    s = ex0.block_ids.index(1)
    q1 = [i for i, bb in enumerate(ex0.block_ids) if bb == 1]
    q2 = [i for i, bb in enumerate(ex0.block_ids) if bb == 2]
    assert allowed[0][q2[0] :, :s][:, :].all(), "state must be visible to every later token"
    assert not allowed[0][q2[0] :, q1].any(), "question 2 must not see question 1"
    assert not allowed[0][q1, q2[0] :].any(), "question 1 must not see question 2"
    assert not allowed[0][: q1[0], q1[0] :].any(), "state must not see the questions"
    # padding columns are off for every row but the diagonal
    npad = len(items[1])
    if npad < t:
        pad_cols = allowed[1][:, npad:]
        diag = torch.zeros_like(pad_cols)
        for i in range(npad, t):
            diag[i, i - npad] = True
        assert not (pad_cols & ~diag).any()


def test_causal_within_a_block(tok):
    packer = Packer(tok)
    p = packer.pack(two_question_example())
    batch = collate([p], packer.pad_id, torch.float32)
    allowed = batch["attention_mask"][0, 0] == 0.0
    q1 = [i for i, bb in enumerate(p.block_ids) if bb == 1]
    sub = allowed[q1][:, q1]
    assert torch.equal(sub, torch.tril(torch.ones_like(sub)))


def test_collate_targets_and_flat_option_pos(tok):
    packer = Packer(tok)
    items = [packer.pack(two_question_example()), packer.pack(one_question_example())]
    batch = collate(items, packer.pad_id, torch.float32)
    t = batch["input_ids"].shape[1]
    assert batch["valid"].shape == (3, 2)
    assert batch["has_target"].tolist() == [True, True, True]
    assert torch.allclose(batch["target"], torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]))
    assert batch["option_group"].tolist() == [0, 0, 1, 1, 2, 2]  # global across the batch
    assert batch["q_task"] == ["toy"] * 3
    # option_pos is flattened b*T + pos and lands on the recorded tokens
    flat = batch["input_ids"].reshape(-1)
    for idx, (bi, p) in zip(
        batch["option_pos"].tolist(),
        [(0, items[0])] * 4 + [(1, items[1])] * 2,
    ):
        assert flat[idx] == p.input_ids[idx - bi * t]


def test_soft_and_missing_targets(tok):
    packer = Packer(tok)
    ex = Example(
        task="t",
        state="s",
        questions=[
            Question("a", "?", ["x", "y"], [0.3, 0.7]),
            Question("b", "?", ["x", "y", "z"], None),
        ],
    )
    batch = collate([packer.pack(ex)], packer.pad_id, torch.float32)
    assert batch["valid"].tolist() == [[True, True, False], [True, True, True]]
    assert batch["has_target"].tolist() == [True, False]
    assert torch.allclose(batch["target"][0], torch.tensor([0.3, 0.7, 0.0]))
    assert torch.allclose(batch["target"][1], torch.zeros(3))


def test_budget_and_drop_questions(tok):
    long_state = "word " * 4000
    ex = Example(task="t", state=long_state, questions=[
        Question(str(i), "Question number %d, is it fine?" % i, ["no", "yes"], 0) for i in range(8)
    ])
    packer = Packer(tok, max_state_tokens=512, max_total_tokens=768)
    p = packer.pack(ex)
    assert p is not None
    assert len([b for b in set(p.block_ids) if b]) == 8
    assert len(p.input_ids) <= 768
    state_len = p.block_ids.count(0)
    assert state_len == min(512, 768 - (len(p.input_ids) - state_len))

    # too many questions to leave a usable state budget
    fat = Example(task="t", state=long_state, questions=[
        Question(str(i), "Question number %d, is this a fairly long question?" % i,
                 ["option alpha", "option beta", "option gamma"], 0)
        for i in range(60)
    ])
    assert packer.pack(fat) is None
    dropped = packer.pack(fat, drop_questions_to_fit=True)
    assert dropped is not None
    assert 1 <= dropped.n_questions < 60
    assert dropped.block_ids.count(0) >= 16


def test_state_truncation_keeps_head(tok):
    packer = Packer(tok, max_state_tokens=32, max_total_tokens=768)
    ex = Example(task="t", state="alpha beta gamma " * 200, questions=[Question("q", "?", ["a", "b"], 0)])
    p = packer.pack(ex)
    assert p.block_ids.count(0) == 32
    head = packer._enc("State:\n" + ex.state)[:31]
    assert p.input_ids[1:32] == head  # after the bos token


def test_full_causal_mode_is_plain_causal(tok):
    packer = Packer(tok, mask_mode="full-causal")
    p = packer.pack(two_question_example())
    assert set(p.block_ids) == {0}
    assert p.position_ids == list(range(len(p.input_ids)))
    batch = collate([p], packer.pad_id, torch.float32)
    allowed = batch["attention_mask"][0, 0] == 0.0
    assert torch.equal(allowed, torch.tril(torch.ones_like(allowed, dtype=torch.bool)))


# ---- instructions and option descriptions ------------------------

def test_bare_question_packs_as_the_plain_template(tok):
    """The no-description path must be byte-identical to the plain template, or a
    checkpoint trained on it sees a sequence it was not trained on."""
    from lod.model.packing import QUESTION_TEMPLATE, QUESTION_HEAD, OPTIONS_HEAD
    assert QUESTION_TEMPLATE == QUESTION_HEAD + OPTIONS_HEAD
    packer = Packer(tok)
    q = Question("a", "Is the customer angry?", ["no", "yes"], 1)
    plain = packer.pack(Example(task="t", state="S", questions=[q]))
    same = packer.pack(Example(task="t", state="S", questions=[
        Question("a", "Is the customer angry?", ["no", "yes"], 1,
                 instructions=None, descriptions=None)]))
    assert plain.input_ids == same.input_ids
    assert plain.option_pos == same.option_pos


def test_descriptions_extend_the_option_span(tok):
    """The option is scored at the last token of its span; with a description that is
    the last token of the description, so the span grows and the score position moves."""
    packer = Packer(tok)
    bare = packer.pack(Example(task="t", state="S", questions=[
        Question("a", "Which team?", ["billing", "technical"], 0)]))
    desc = packer.pack(Example(task="t", state="S", questions=[
        Question("a", "Which team?", ["billing", "technical"], 0,
                 descriptions=["payments and refunds", "software faults"])]))
    assert len(desc.input_ids) > len(bare.input_ids)
    assert desc.option_pos != bare.option_pos
    # each option's score position is still the final token of its own span
    for pos in desc.option_pos:
        assert 0 <= pos < len(desc.input_ids)
    assert desc.n_options == bare.n_options == [2]


def test_instructions_sit_before_the_options_marker(tok):
    packer = Packer(tok)
    p = packer.pack(Example(task="t", state="S", questions=[
        Question("a", "Which team?", ["billing", "technical"], 0,
                 instructions="Choose the team that owns the first action.")]))
    text = tok.decode(p.input_ids)
    assert "Choose the team that owns the first action." in text
    assert text.index("owns the first action") < text.index("Options:")


def test_partial_descriptions_are_allowed(tok):
    """A source that defines some labels and not others is normal, not an error."""
    packer = Packer(tok)
    q = Question("a", "Which team?", ["billing", "technical", "account"], 0,
                 descriptions=["payments", None])
    p = packer.pack(Example(task="t", state="S", questions=[q]))
    assert len(p.option_pos) == 3
    assert q.description_for(0) == "payments"
    assert q.description_for(1) is None and q.description_for(2) is None


def test_descriptions_round_trip(tok):
    q = Question("a", "Which team?", ["billing", "technical"], 0,
                 instructions="Pick one.", descriptions=["payments", "faults"])
    back = Question.from_dict(q.to_dict())
    assert back.instructions == "Pick one."
    assert back.descriptions == ["payments", "faults"]
    bare = Question.from_dict(Question("b", "?", ["no", "yes"], 1).to_dict())
    assert bare.instructions is None and bare.descriptions is None


def test_question_refs_survive_dropped_questions(tok):
    """The packer drops questions that do not fit, so joining question ids and meta to
    logits by position goes wrong after the first drop. q_ref is the join key."""
    from lod.training.data import DecisionDataset

    long_state = "word " * 4000
    fat = Example(task="fat", state=long_state, questions=[
        Question(f"fat{i}", "Question number %d, is this a fairly long question?" % i,
                 ["option alpha", "option beta", "option gamma"], 0)
        for i in range(60)
    ])
    small = Example(task="small", state="short", questions=[
        Question("s0", "Is it short?", ["no", "yes"], 1),
        Question("s1", "Is it long?", ["no", "yes"], 0)])
    packer = Packer(tok, max_state_tokens=512, max_total_tokens=768)
    ds = DecisionDataset([small, fat, small], packer, augment=False)
    items = [ds[i] for i in range(3)]
    batch = collate(items, tok.pad_token_id or 0, torch.float32)
    refs = batch["q_ref"]
    assert len(refs) == sum(p.n_questions for p in items)
    examples = [small, fat, small]
    ids = [examples[i].questions[j].id for i, j in refs]
    assert ids[:2] == ["s0", "s1"] and ids[-2:] == ["s0", "s1"]
    assert ids[2] == "fat0" and len(ids) < 2 + 60 + 2    # fat lost some questions
    assert [examples[i].task for i, _ in refs] == batch["q_task"]
