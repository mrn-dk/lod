"""Token-budget batching and states longer than the short-context budget."""

import numpy as np

from lod.training.data import EpochBatchSampler, token_estimate
from lod.model.packing import Packer
from lod.schema import Example, Question


def test_token_budget_batches_cover_every_example_once_and_respect_the_cap():
    rng = np.random.default_rng(0)
    lengths = np.concatenate([rng.integers(50, 2000, 950), rng.integers(8000, 30000, 50)])
    rng.shuffle(lengths)
    cap = 98_304
    s = EpochBatchSampler(len(lengths), 32, seed=0, epoch=0, lengths=lengths, bucket=16,
                          max_tokens=cap)
    seen = [i for b in s for i in b]
    assert sorted(seen) == list(range(len(lengths)))
    for b in s:
        assert len(b) <= 32
        if len(b) > 1:
            assert len(b) * int(lengths[b].max()) <= cap
    again = EpochBatchSampler(len(lengths), 32, seed=0, epoch=0, lengths=lengths, bucket=16,
                              max_tokens=cap)
    assert [list(b) for b in again] == [list(b) for b in s]
    # long examples really do get small batches
    long_batches = [b for b in s if lengths[b].max() > 8000]
    assert long_batches and max(len(b) for b in long_batches) <= 12


def test_state_past_the_old_budget_packs_whole(tok):
    state = " ".join(f"record {i}: value {i * 7}." for i in range(900))   # ~6-7k tokens
    q = Question("q", "Which record holds value 6293?", ["a", "b"], 0,
                 descriptions=["record 899", "record 12"])
    ex = Example(task="t", state=state, questions=[q])
    old = Packer(tok, 2304, 3072, option_mode="independent", overflow="flag").pack(ex)
    new = Packer(tok, 30000, 32768, option_mode="independent", overflow="refuse").pack(ex)
    assert old.state_truncated
    assert not new.state_truncated and new.state_tokens == new.state_tokens_full > 2304
    assert token_estimate([ex])[0] >= new.state_tokens * 0.6


def test_eval_loader_token_budget_keeps_every_question(tok):
    import torch

    from lod.training.data import DecisionDataset, make_loader

    exs = [Example(task="t", state=("x " * n), questions=[Question("q", "Which?", ["a", "b"], 0)])
           for n in (5, 3000, 40, 900, 12, 2500)]
    pk = Packer(tok, 30000, 32768, option_mode="independent")
    ds = DecisionDataset(exs, pk, augment=False)
    loader = make_loader(ds, pk.pad_id, torch.float32, batch_size=4, num_workers=0,
                         max_tokens=4000)
    refs = [r for b in loader for r in b["q_ref"]]
    assert sorted(refs) == [(i, 0) for i in range(len(exs))]
    for b in loader:
        assert b["input_ids"].shape[0] * b["input_ids"].shape[1] <= 4000 * 1.5 or \
            b["input_ids"].shape[0] == 1
