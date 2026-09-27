"""Independent options, the comparison layer, T(N) and the budget policy.

The acceptance bar is order invariance "to floating-point error": each option attends to
the state, its question's head and itself only, from a shared start position, so no
option's hidden state can depend on which siblings are listed or in what order, and the
comparison layer is permutation-equivariant.
"""

import random

import pytest
import torch

from lod.model.scorer import OptionMixer, OptionScoringModel, forward_batch, grouped_logits
from lod.model.packing import Packer, StateTruncated, build_block_mask, collate
from lod.schema import Example, Question

from conftest import BACKBONE

STATE = ("Order 7731: 3 parcels, 14.2 kg total, destination Oslo, carrier Kestrel, "
         "service express. The customer asked for delivery before Friday.")
OPTS = ["express_air", "ground", "freight", "courier_same_day", "postal"]
DESCS = ["Air, 1-2 days, up to 30 kg", "Road, 3-5 days, up to 70 kg",
         "Pallet freight, 5-9 days, over 50 kg", "Same-day courier, city only",
         "National post, 4-7 days, up to 20 kg"]


def _q(options=OPTS, descs=DESCS, target=0):
    return Question("ship", "Which shipping service fits this order?", list(options),
                    target, descriptions=list(descs))


@pytest.fixture(scope="module")
def imodel():
    torch.manual_seed(0)
    m = OptionScoringModel(BACKBONE, torch_dtype=torch.float32, option_mode="independent",
                           mixer_dim=32, mixer_heads=2, mixer_topk=64)
    # the mixer is zero-initialised; give it weights so the tests exercise it
    with torch.no_grad():
        m.mixer.out.weight.normal_(0, 0.5)
    return m.eval()


def _probs(model, packer, examples):
    batch = collate([packer.pack(e) for e in examples], packer.pad_id, torch.float32)
    with torch.no_grad():
        logits = forward_batch(model, batch, "cpu")
    g = grouped_logits(logits, batch["option_group"], batch["option_slot"], batch["valid"])
    return torch.softmax(g.masked_fill(~batch["valid"], float("-inf")), -1)


def test_independent_layout(tok):
    pk = Packer(tok, option_mode="independent")
    p = pk.pack(Example(task="t", state=STATE, questions=[_q()]))
    s = p.state_tokens
    head_len = sum(1 for b, o in zip(p.block_ids, p.option_ids) if b == 1 and o == 0)
    starts = []
    for slot in range(len(OPTS)):
        idx = [i for i, o in enumerate(p.option_ids) if o == slot + 1]
        assert idx == list(range(idx[0], idx[-1] + 1))
        starts.append(p.position_ids[idx[0]])
        assert p.option_pos[slot] == idx[-1]
    assert set(starts) == {s + head_len}


def test_independent_mask_blocks_siblings(tok):
    pk = Packer(tok, option_mode="independent")
    p = pk.pack(Example(task="t", state=STATE, questions=[_q(), _q()]))
    b = collate([p], pk.pad_id, torch.float32)
    allowed = build_block_mask(b["block_ids"], b["token_valid"], torch.float32,
                               b["option_ids"])[0, 0] == 0
    bl, op = b["block_ids"][0], b["option_ids"][0]
    for i in range(len(p)):
        for j in range(i + 1):
            ok = bool(allowed[i, j])
            want = (i == j) or bl[j] == 0 or (bl[j] == bl[i] and (op[j] == 0 or op[j] == op[i]))
            assert ok == bool(want), (i, j)


def test_sequential_layout_unchanged(tok):
    """The default packer must produce exactly what the shipped checkpoints trained on."""
    pk = Packer(tok)
    p = pk.pack(Example(task="t", state=STATE, questions=[_q(), _q(target=1)]))
    assert p.option_ids == []
    s = p.state_tokens
    # one question block: contiguous positions from s
    blk = [i for i, b in enumerate(p.block_ids) if b == 1]
    assert [p.position_ids[i] for i in blk] == list(range(s, s + len(blk)))


def test_order_invariance_exact(imodel):
    pk = Packer.for_model(imodel)
    ref = _probs(imodel, pk, [Example(task="t", state=STATE, questions=[_q()])])[0]
    rng = random.Random(1)
    worst = 0.0
    for _ in range(4):
        order = list(range(len(OPTS)))
        rng.shuffle(order)
        q = _q([OPTS[i] for i in order], [DESCS[i] for i in order], target=order.index(0))
        got = _probs(imodel, pk, [Example(task="t", state=STATE, questions=[q])])[0]
        unperm = torch.zeros(len(OPTS))
        for new, old in enumerate(order):
            unperm[old] = got[new]
        worst = max(worst, (unperm - ref[: len(OPTS)]).abs().max().item())
    assert worst < 1e-5, worst


def test_stage1_scores_do_not_see_siblings(imodel):
    """Adding options changes no existing option's independent score."""
    pk = Packer.for_model(imodel)

    def stage1(q):
        batch = collate([pk.pack(Example(task="t", state=STATE, questions=[q]))],
                        pk.pad_id, torch.float32)
        with torch.no_grad():
            parts = forward_batch(imodel, batch, "cpu", return_parts=True)
        return parts["stage1"]

    few = stage1(_q(OPTS[:2], DESCS[:2]))
    many = stage1(_q(OPTS, DESCS))
    assert (few - many[:2]).abs().max().item() < 1e-5


def test_mixer_zero_init_is_identity():
    torch.manual_seed(0)
    mix = OptionMixer(16, 8, 2)
    s = torch.randn(3, 5)
    valid = torch.ones(3, 5, dtype=torch.bool)
    valid[2, 3:] = False
    out = mix.combine(torch.randn(3, 5, 16), s, valid, topk=64)
    p_out = torch.softmax(out, -1)
    p_ref = torch.softmax(s.masked_fill(~valid, float("-inf")), -1)
    assert torch.allclose(p_out, p_ref, atol=1e-6)


def test_shortlist_keeps_tail_mass():
    """Options outside the top-k keep their stage-1 probability exactly."""
    torch.manual_seed(0)
    mix = OptionMixer(16, 8, 2)
    with torch.no_grad():
        mix.out.weight.normal_(0, 1.0)
    s = torch.randn(2, 40)
    valid = torch.ones(2, 40, dtype=torch.bool)
    out = mix.combine(torch.randn(2, 40, 16), s, valid, topk=8)
    p1 = torch.softmax(s, -1)
    pf = torch.softmax(out, -1)
    top = s.topk(8, -1).indices
    tail = torch.ones_like(valid).scatter_(1, top, False)
    assert torch.allclose(pf[tail], p1[tail], atol=1e-6)
    assert torch.allclose(pf.sum(-1), torch.ones(2), atol=1e-6)


def test_temperature_logn():
    m = OptionScoringModel.__new__(OptionScoringModel)
    torch.nn.Module.__init__(m)
    m.register_buffer("temperature", torch.tensor(2.0))
    m.temperature_logn = 0.5
    group = torch.tensor([0, 0, 1, 1, 1, 1, 1, 1, 1, 1])
    t = m.temperature_for(group, torch.zeros(10))
    assert torch.allclose(t[:2], torch.tensor(2.0))
    assert torch.allclose(t[2:], torch.tensor(2.0 * (8 / 2) ** 0.5))


def test_overflow_policies(tok):
    long_state = " ".join(["record"] * 400)
    q = _q()
    ex = Example(task="t", state=long_state, questions=[q])
    flag = Packer(tok, max_state_tokens=512, max_total_tokens=200, overflow="flag").pack(ex)
    assert flag.state_truncated and flag.state_tokens < flag.state_tokens_full
    with pytest.raises(StateTruncated):
        Packer(tok, max_state_tokens=512, max_total_tokens=200, overflow="refuse").pack(ex)
    fits = Packer(tok, max_state_tokens=2048, max_total_tokens=4096, overflow="refuse").pack(ex)
    assert not fits.state_truncated and fits.state_tokens == fits.state_tokens_full


def test_save_load_roundtrip(imodel, tmp_path):
    imodel.temperature_logn = 0.25
    imodel.save(tmp_path / "ckpt")
    back = OptionScoringModel.load(tmp_path / "ckpt")
    imodel.temperature_logn = 0.0
    assert back.option_mode == "independent"
    assert back.mixer is not None and back.mixer_topk == 64
    assert back.temperature_logn == 0.25
    for (k, a), (_, b) in zip(imodel.mixer.state_dict().items(), back.mixer.state_dict().items()):
        assert torch.equal(a, b), k


def _unsharded(model, q):
    pk = Packer.for_model(model)
    batch = collate([pk.pack(Example(task="t", state=STATE, questions=[q]))],
                    pk.pad_id, torch.float32)
    res = model.score_batch(batch, "cpu")
    return res["probs"][0, : len(q.options)], float(res["confidence"][0])


@pytest.mark.parametrize("topk", [64, 3])
def test_sharded_matches_one_pass(imodel, topk):
    from lod.model.scorer import ConfidenceHead
    from lod.model.sharded import score_question

    imodel.mixer_topk = topk
    imodel.confidence_head = ConfidenceHead(imodel.head[-1].in_features,
                                            features=("log_n", "max_p", "margin"))
    with torch.no_grad():
        imodel.confidence_head.feat.weight.normal_(0, 1.0)
    imodel.confidence_mode = "head"
    try:
        opts = [f"svc_{i}" for i in range(12)]
        descs = [f"service tier {i}, {3 + i} days, up to {10 * (i + 1)} kg" for i in range(12)]
        q = Question("ship", "Which shipping service fits this order?", opts, 0,
                     descriptions=descs)
        ref_p, ref_c = _unsharded(imodel, q)
        pk = Packer.for_model(imodel)
        res = score_question(imodel, pk, STATE, q, "cpu", shard_budget=60, shard_batch=2)
        assert res.n_shards > 2
        assert (res.probs - ref_p).abs().max().item() < 1e-5
        assert abs(res.confidence - ref_c) < 1e-5
    finally:
        imodel.mixer_topk = 64
        imodel.confidence_head, imodel.confidence_mode = None, "maxprob"


def test_sharded_refuses_sequential(model):
    from lod.model.sharded import score_question

    with pytest.raises(ValueError):
        score_question(model, Packer(model.tokenizer), STATE, _q(), "cpu")


def test_sharded_scores_past_the_budget(imodel):
    """300 options, far past a 768-token budget: scored, normalised, never truncated."""
    from lod.model.sharded import score_question

    opts = [f"item_{i}" for i in range(300)]
    q = Question("pick", "Which catalogue item matches?", opts, 0)
    pk = Packer.for_model(imodel)
    assert pk.pack(Example(task="t", state=STATE, questions=[q])) is None
    res = score_question(imodel, pk, STATE, q, "cpu")
    assert res.probs.shape == (300,) and abs(float(res.probs.sum()) - 1) < 1e-5
    assert not res.state_truncated and res.n_shards >= 2


def test_fit_temperature_logn_recovers_a_slope():
    from lod.model.calibration import fit_temperature_logn

    torch.manual_seed(0)
    rows, valid, target = [], [], []
    for n in (2, 4, 16, 64):
        for _ in range(400):
            z = torch.randn(64) * 3
            t_true = 1.5 * (n / 2) ** 0.3
            p = torch.softmax(z[:n] / t_true, -1)
            y = torch.multinomial(p, 1).item()
            row = torch.full((64,), float("-inf")); row[:n] = z[:n]
            v = torch.zeros(64, dtype=torch.bool); v[:n] = True
            tg = torch.zeros(64); tg[y] = 1
            rows.append(row); valid.append(v); target.append(tg)
    t, slope = fit_temperature_logn(torch.stack(rows), torch.stack(valid), torch.stack(target))
    assert abs(t - 1.5) < 0.25 and abs(slope - 0.3) < 0.1, (t, slope)


def test_distillation_mixes_after_augmentation():
    from lod.training.data import mix_teacher, shuffle_options, teacher_by_option

    q = Question("q", "Which?", ["a", "b", "c"], 0, meta={"teacher": [0.6, 0.3, 0.1]})
    ex = Example(task="t", state="s", questions=[q])
    teachers = [teacher_by_option(q)]
    ex2 = shuffle_options(ex, random.Random(3))
    mixed = mix_teacher(ex2, teachers, 0.5).questions[0]
    want = {"a": 0.8, "b": 0.15, "c": 0.05}
    for o, p in zip(mixed.options, mixed.target):
        assert abs(p - want[o]) < 1e-9
    # an option the teacher never scored: gold kept
    q3 = Question("q", "Which?", ["a", "b", "zzz"], 0)
    kept = mix_teacher(Example(task="t", state="s", questions=[q3]), teachers, 0.5)
    assert kept.questions[0].target == 0
