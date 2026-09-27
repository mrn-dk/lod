"""Packing: text -> packed ids, positions, block ids, option positions, plus the
block attention mask and a collate function.

One state is prefilled once; every question is its own block that attends to
the state and to itself, never to another question. Each question block's
positions restart right after the state, so a question's output does not depend
on how many other questions precede it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import torch

from lod import sentinel
from lod.schema import Example

STATE_PREFIX = "State:\n"
# The question head and the "Options:" marker are separate so that instructions can sit
# between them; QUESTION_TEMPLATE is their concatenation, the no-instructions layout.
QUESTION_HEAD = "\n\nQuestion: {q}"
OPTIONS_HEAD = "\nOptions:"
QUESTION_TEMPLATE = QUESTION_HEAD + OPTIONS_HEAD
INSTRUCTION_TEMPLATE = "\n{instructions}"
OPTION_TEMPLATE = "\n- {opt}"
OPTION_TEMPLATE_DESC = "\n- {opt}: {desc}"

MIN_STATE_BUDGET = 16
# A generated "none of the above" sentinel is drawn from a grammar (`lod/sentinel.py`),
# so it cannot be recognised by string match; a question that abstains says so in its own
# `meta`. The key set covers the callers that cannot: a request arriving over the wire
# with a hand-written `none_of_the_above`, and older dumps being re-scored.
NONE_KEYS = sentinel.LEGACY_KEYS | sentinel.KEYS
NONE_OF_THE_ABOVE = sentinel.NONE_OF_THE_ABOVE   # the default; kept for the wire contract


def _is_withheld(q) -> bool:
    """True when this question has no correct answer among its own options.

    `meta.abstain` is authoritative: it is set by the augmentation that withheld the
    answer and by the sources that emit a not-in-context row, and it is the only signal
    that survives a sentinel whose wording was generated. The key-match fallback keeps a
    question built by hand, or arriving over the wire with `none_of_the_above` selected,
    treated the same way. That matters because the confidence head is trained to be
    unsure exactly here.
    """
    meta = getattr(q, "meta", None)
    if meta and meta.get("abstain"):
        return True
    probs = q.target_probs() if hasattr(q, "target_probs") else None
    if probs is None:
        return False
    best = max(range(len(probs)), key=probs.__getitem__)
    return probs[best] >= 0.99 and str(q.options[best]) in NONE_KEYS


@dataclass
class Packed:
    input_ids: list[int]
    position_ids: list[int]
    block_ids: list[int]
    option_pos: list[int]  # absolute token index whose hidden state scores the option
    option_group: list[int]  # question index within this example
    option_slot: list[int]  # option index within its question
    n_options: list[int]  # per question
    targets: list[list[float] | None]  # per question
    withheld: list[bool] = field(default_factory=list)  # per question; criteria withheld
    task: str = ""
    # Which pointer head owns this example's loss (K-head ensemble).
    # -1 = unassigned, which is every single-head run and every eval.
    shard: int = -1
    # Position of the source example in its dataset. The packer may drop trailing
    # questions to fit the budget and the dataset may drop whole examples, so a
    # question's row in the collected logits is NOT its row in the file; anything
    # that joins logits back to question ids or `meta` must go through this.
    example_index: int = -1
    # Per token: 0 for the state and every question's head, k+1 for the tokens of option
    # k. Empty means all zeros, which is the sequential layout: an option then sees the
    # options before it. Under `option_mode="independent"` an option attends only to the
    # state, its own question's head and itself, and every option of a question starts
    # at the same position id, so its hidden state -- and its score -- cannot depend on
    # which other options are listed or in what order.
    option_ids: list[int] = field(default_factory=list)
    # The budget policy: the state is never cut without saying so.
    state_tokens: int = 0          # state tokens actually packed (incl. BOS and prefix)
    state_tokens_full: int = 0     # what the whole state encodes to
    state_truncated: bool = False
    n_questions: int = field(init=False)

    def __post_init__(self) -> None:
        self.n_questions = len(self.n_options)

    def __len__(self) -> int:
        return len(self.input_ids)


class StateTruncated(ValueError):
    """The questions leave too little of the budget for the whole state.

    Raised under `overflow="refuse"`. Options must never silently eat the state: a caller
    who adds 200 options to a request and gets an answer computed from the first half of
    the record has been told nothing about it.
    """

    def __init__(self, used: int, full: int) -> None:
        super().__init__(f"state needs {full} tokens but the questions leave {used}")
        self.used = used
        self.full = full


OPTION_MODES = ("sequential", "independent")
OVERFLOW_POLICIES = ("truncate", "flag", "refuse")


class Packer:
    """Turns an `Example` into a `Packed` sequence.

    `mask_mode="block"`: one state, then question blocks that each attend to the state
    and to themselves only, with positions restarting after the state.
    `mask_mode="full-causal"` is an ablation: continuous positions and one block, so
    every question sees the ones before it.

    `option_mode="independent"` goes one level further down: each option is its own
    sub-block of its question, attending to the state, the question head and itself,
    and all of a question's options share a start position. Option order and option
    count then cannot move any option's hidden state; the model's comparison layer is
    the only place options meet.

    `overflow` is what happens when the questions leave less than the whole state:
    `truncate` cuts it (training, where it is recorded on the Packed), `flag` cuts it and
    sets `state_truncated`, `refuse` raises `StateTruncated`.
    """

    def __init__(
        self,
        tokenizer,
        max_state_tokens: int = 512,
        max_total_tokens: int = 768,
        mask_mode: str = "block",
        option_mode: str = "sequential",
        overflow: str = "truncate",
    ) -> None:
        if mask_mode not in ("block", "full-causal"):
            raise ValueError(f"unknown mask_mode {mask_mode!r}")
        if option_mode not in OPTION_MODES:
            raise ValueError(f"unknown option_mode {option_mode!r}")
        if overflow not in OVERFLOW_POLICIES:
            raise ValueError(f"unknown overflow policy {overflow!r}")
        if option_mode == "independent" and mask_mode != "block":
            raise ValueError("independent options need the block mask")
        self.tok = tokenizer
        self.max_state_tokens = max_state_tokens
        self.max_total_tokens = max_total_tokens
        self.mask_mode = mask_mode
        self.option_mode = option_mode
        self.overflow = overflow

    @classmethod
    def for_model(cls, model, **kw) -> "Packer":
        """The packer a checkpoint was trained with: its budgets, mask and option layout."""
        return cls(model.tokenizer, model.max_state_tokens, model.max_total_tokens,
                   getattr(model, "mask_mode", "block"),
                   option_mode=getattr(model, "option_mode", "sequential"), **kw)

    def _enc(self, text: str) -> list[int]:
        return self.tok(text, add_special_tokens=False)["input_ids"]

    @property
    def pad_id(self) -> int:
        for attr in ("pad_token_id", "eos_token_id"):
            v = getattr(self.tok, attr, None)
            if v is not None:
                return int(v)
        return 0

    def _parts(self, q) -> tuple[list[int], list[list[int]]]:
        """Token ids of a question's head and of each option span, encoded separately.

        Instructions sit between the question and "Options:", and an option may carry
        a description. The option is scored at the last token of its span,
        which is simply the last token of the description when one is present.
        """
        head = QUESTION_HEAD.format(q=q.question)
        instructions = getattr(q, "instructions", None)
        if instructions:
            head += INSTRUCTION_TEMPLATE.format(instructions=instructions)
        head_ids = self._enc(head + OPTIONS_HEAD)
        opts: list[list[int]] = []
        for i, opt in enumerate(q.options):
            desc = q.description_for(i) if hasattr(q, "description_for") else None
            if desc:
                opt_ids = self._enc(OPTION_TEMPLATE_DESC.format(opt=opt, desc=desc))
            else:
                opt_ids = self._enc(OPTION_TEMPLATE.format(opt=opt))
            if not opt_ids:  # pathological tokenizer; keep a scorable position
                opt_ids = self._enc(OPTION_TEMPLATE.format(opt=str(opt) + " "))
            opts.append(opt_ids)
        return head_ids, opts

    def _block(self, q) -> tuple[list[int], list[int]]:
        """Token ids of one question block and, per option, the index of the
        option span's last token relative to the block start."""
        head_ids, opts = self._parts(q)
        ids = list(head_ids)
        last: list[int] = []
        for opt_ids in opts:
            ids.extend(opt_ids)
            last.append(len(ids) - 1)
        return ids, last

    def question_tokens(self, q) -> int:
        """Tokens one question adds to a packed sequence (the same in either layout)."""
        head_ids, opts = self._parts(q)
        return len(head_ids) + sum(len(o) for o in opts)

    def state_ids(self, state: str) -> list[int]:
        bos = getattr(self.tok, "bos_token_id", None)
        return ([int(bos)] if bos is not None else []) + self._enc(STATE_PREFIX + state)

    def pack(self, example: Example, drop_questions_to_fit: bool = False) -> Packed | None:
        questions = list(example.questions)
        if not questions:
            return None

        parts = [self._parts(q) for q in questions]

        def q_len(p):
            return len(p[0]) + sum(len(o) for o in p[1])

        budget = min(self.max_state_tokens, self.max_total_tokens - sum(q_len(p) for p in parts))
        if budget < MIN_STATE_BUDGET:
            if not drop_questions_to_fit:
                return None
            while budget < MIN_STATE_BUDGET and len(questions) > 1:
                questions.pop()
                parts.pop()
                budget = min(
                    self.max_state_tokens,
                    self.max_total_tokens - sum(q_len(p) for p in parts),
                )
            if budget < MIN_STATE_BUDGET:
                return None  # a single question already overflows the budget

        full_state = self.state_ids(example.state)
        state_ids = full_state[:budget]
        s = len(state_ids)
        truncated = s < len(full_state)
        if truncated and self.overflow == "refuse":
            raise StateTruncated(s, len(full_state))

        input_ids = list(state_ids)
        position_ids = list(range(s))
        block_ids = [0] * s
        option_ids = [0] * s
        option_pos: list[int] = []
        option_group: list[int] = []
        option_slot: list[int] = []
        n_options: list[int] = []

        full_causal = self.mask_mode == "full-causal"
        independent = self.option_mode == "independent"
        for qi, (q, (head_ids, opts)) in enumerate(zip(questions, parts)):
            start = len(input_ids)
            h = len(head_ids)
            input_ids.extend(head_ids)
            option_ids.extend([0] * h)
            if full_causal:
                position_ids.extend(range(start, start + h))
                block_ids.extend([0] * h)
            else:
                position_ids.extend(range(s, s + h))
                block_ids.extend([qi + 1] * h)
            # sequential: an option's positions continue from the previous option's
            # independent: every option restarts right after the head
            cursor = s + h
            for slot, opt_ids in enumerate(opts):
                o_start = len(input_ids)
                n = len(opt_ids)
                input_ids.extend(opt_ids)
                if full_causal:
                    position_ids.extend(range(o_start, o_start + n))
                    block_ids.extend([0] * n)
                    option_ids.extend([0] * n)
                elif independent:
                    position_ids.extend(range(s + h, s + h + n))
                    block_ids.extend([qi + 1] * n)
                    option_ids.extend([slot + 1] * n)
                else:
                    position_ids.extend(range(cursor, cursor + n))
                    block_ids.extend([qi + 1] * n)
                    option_ids.extend([0] * n)
                    cursor += n
                option_pos.append(o_start + n - 1)
                option_group.append(qi)
                option_slot.append(slot)
            n_options.append(len(q.options))

        return Packed(
            input_ids=input_ids,
            position_ids=position_ids,
            block_ids=block_ids,
            option_pos=option_pos,
            option_group=option_group,
            option_slot=option_slot,
            n_options=n_options,
            targets=[q.target_probs() for q in questions],
            withheld=[_is_withheld(q) for q in questions],
            task=example.task,
            option_ids=option_ids if independent else [],
            state_tokens=s,
            state_tokens_full=len(full_state),
            state_truncated=truncated,
        )


def build_block_mask(
    block_ids: torch.Tensor,  # [B, T] long, padding may be anything
    token_valid: torch.Tensor,  # [B, T] bool, False for padding
    dtype: torch.dtype = torch.float32,
    option_ids: torch.Tensor | None = None,  # [B, T] long, 0 = not inside an option
) -> torch.Tensor:
    """Additive 4-D mask [B, 1, T, T]: 0 where attention is allowed, dtype min
    where it is not.

    Token i may attend to token j iff  j <= i  and j is not padding and
    (block[j] == 0 or (block[j] == block[i] and (opt[j] == 0 or opt[j] == opt[i]))).
    With every opt 0 that is the question-block rule; with independent options it also
    keeps an option from seeing its siblings. The diagonal is always allowed so no row is
    fully masked.
    """
    b, t = block_ids.shape
    device = block_ids.device
    idx = torch.arange(t, device=device)
    causal = idx[None, :] <= idx[:, None]  # [T, T]
    bj = block_ids[:, None, :]  # [B, 1, T]
    bi = block_ids[:, :, None]  # [B, T, 1]
    if option_ids is None:
        same = (bj == 0) | (bj == bi)
    else:
        oj = option_ids[:, None, :]
        oi = option_ids[:, :, None]
        same = (bj == 0) | ((bj == bi) & ((oj == 0) | (oj == oi)))
    allowed = causal[None] & same & token_valid[:, None, :]
    allowed = allowed | torch.eye(t, dtype=torch.bool, device=device)[None]
    mask = torch.zeros(b, 1, t, t, dtype=dtype, device=device)
    return mask.masked_fill_(~allowed.unsqueeze(1), torch.finfo(dtype).min)


def collate(
    items: Sequence[Packed],
    pad_id: int,
    mask_dtype: torch.dtype = torch.float32,
    pad_to_tile: int = 1,
    dense_mask: bool = True,
) -> dict[str, Any]:
    """Batch packed examples. Question indices become global across the batch.

    `dense_mask=False` omits the `[B, 1, T, T]` additive mask, for the FlexAttention
    path which rebuilds the same rule as a predicate on-device (`lod/model/flexattn.py`).
    That also saves shipping the mask over PCIe -- at B=32, T=768, fp32 it is 75 MB a
    batch. `pad_to_tile` rounds T up so FlexAttention's tiles divide it evenly; padding
    is already excluded by `token_valid`, so it changes no result.
    """
    b = len(items)
    t = max(len(p) for p in items)
    if pad_to_tile > 1:
        t = ((t + pad_to_tile - 1) // pad_to_tile) * pad_to_tile
    k = max(max(p.n_options) for p in items)

    input_ids = torch.full((b, t), pad_id, dtype=torch.long)
    position_ids = torch.zeros(b, t, dtype=torch.long)
    block_ids = torch.zeros(b, t, dtype=torch.long)
    token_valid = torch.zeros(b, t, dtype=torch.bool)
    independent = any(p.option_ids for p in items)
    option_ids = torch.zeros(b, t, dtype=torch.long) if independent else None

    option_pos: list[int] = []
    option_group: list[int] = []
    option_slot: list[int] = []
    n_options: list[int] = []
    targets: list[list[float] | None] = []
    withheld: list[bool] = []
    q_task: list[str] = []
    q_shard: list[int] = []
    q_ref: list[tuple[int, int]] = []

    q_offset = 0
    for bi, p in enumerate(items):
        n = len(p)
        input_ids[bi, :n] = torch.tensor(p.input_ids, dtype=torch.long)
        position_ids[bi, :n] = torch.tensor(p.position_ids, dtype=torch.long)
        block_ids[bi, :n] = torch.tensor(p.block_ids, dtype=torch.long)
        token_valid[bi, :n] = True
        if option_ids is not None and p.option_ids:
            option_ids[bi, :n] = torch.tensor(p.option_ids, dtype=torch.long)
        option_pos.extend(bi * t + pos for pos in p.option_pos)
        option_group.extend(q_offset + g for g in p.option_group)
        option_slot.extend(p.option_slot)
        n_options.extend(p.n_options)
        targets.extend(p.targets)
        withheld.extend(p.withheld or [False] * p.n_questions)
        q_task.extend([p.task] * p.n_questions)
        q_shard.extend([p.shard] * p.n_questions)
        # questions are only ever dropped from the end, so kept ones are 0..n-1
        q_ref.extend((p.example_index, j) for j in range(p.n_questions))
        q_offset += p.n_questions

    nq = q_offset
    valid = torch.zeros(nq, k, dtype=torch.bool)
    for qi, n in enumerate(n_options):
        valid[qi, :n] = True
    target = torch.zeros(nq, k, dtype=torch.float32)
    has_target = torch.zeros(nq, dtype=torch.bool)
    for qi, tgt in enumerate(targets):
        if tgt is not None:
            target[qi, : len(tgt)] = torch.tensor(tgt, dtype=torch.float32)
            has_target[qi] = True

    return {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "attention_mask": (build_block_mask(block_ids, token_valid, mask_dtype, option_ids)
                           if dense_mask else None),
        "block_ids": block_ids,
        "option_ids": option_ids,
        "token_valid": token_valid,
        "option_pos": torch.tensor(option_pos, dtype=torch.long),
        "option_group": torch.tensor(option_group, dtype=torch.long),
        "option_slot": torch.tensor(option_slot, dtype=torch.long),
        "valid": valid,
        "target": target,
        "has_target": has_target,
        "withheld": torch.tensor(withheld, dtype=torch.bool),
        "q_task": q_task,
        "q_shard": torch.tensor(q_shard, dtype=torch.long),
        "q_ref": q_ref,
        "n_tokens": int(token_valid.sum()),
        "state_truncated": [p.state_truncated for p in items],
    }
