"""The K pointer-head ensemble over one shared backbone pass.

The claim under test is not "eight heads exist" -- eight copies of the same function
buy nothing. It is that the three decorrelation levers (per-head init, disjoint
content-hashed shards, per-head option order) leave the heads *different*, which is
what the mean pairwise head correlation measures, and that a single-head checkpoint
is still exactly K=1.

CPU-sized throughout: a 2-layer random Llama, hidden 64.
"""

import json
import math
import random

import pytest
import torch
from transformers import AutoModel, LlamaConfig

from lod.model.calibration import (
    aurc,
    fit_temperature,
    fit_temperature_ensemble,
    head_correlation,
    head_disagreement,
    p_wrong_at_confidence,
    probabilities,
)
from lod.training.data import DecisionDataset, example_shard, shuffle_options
from lod.model.scorer import (
    OptionScoringModel,
    decision_loss,
    grouped_head_logits,
    grouped_logits,
    log_mean_prob,
    shard_select,
)
from lod.model.packing import Packer, collate
from lod.schema import Example, Question

K = 8
COLOURS = ["red", "green", "blue", "amber"]


@pytest.fixture(scope="module")
def tiny(tok):
    """A random 2-layer backbone. Nothing here needs a pretrained one."""
    cfg = LlamaConfig(vocab_size=tok.vocab_size, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
                      max_position_embeddings=512)
    torch.manual_seed(0)
    return AutoModel.from_config(cfg)


def make_model(tiny, tok, n_heads, head_seed=0):
    return OptionScoringModel("tiny/random", torch_dtype=torch.float32, n_heads=n_heads,
                              head_seed=head_seed, tokenizer=tok, backbone=tiny,
                              attach_lora=False, max_state_tokens=64, max_total_tokens=192)


def toy_examples(n=192, seed=0):
    """`state` names a colour; the question asks which colour, options are shuffled.

    Answerable from the state, so a linear probe on the option hidden states can learn
    it, and there are four options so a per-question probability vector is worth
    correlating.
    """
    rng = random.Random(seed)
    out = []
    for i in range(n):
        c = rng.randrange(len(COLOURS))
        state = (f"Sensor {i} reported at 09:0{i % 10}. The indicator lamp on the panel "
                 f"is showing {COLOURS[c]} and has been steady for six minutes.")
        q = Question(f"q{i}", "Which colour is the indicator lamp showing?",
                     list(COLOURS), c)
        out.append(Example(task="toy", state=state, questions=[q]))
    return out


# ---- the identity half: K=1 is the model we already ship ---------------------


def test_single_head_forward_shape_and_grouping_are_unchanged(tiny, tok):
    model = make_model(tiny, tok, n_heads=1)
    packer = Packer(tok, 64, 192)
    batch = collate([packer.pack(e) for e in toy_examples(4)], packer.pad_id, torch.float32)
    with torch.no_grad():
        logits = model(batch["input_ids"], batch["position_ids"],
                       batch["attention_mask"], batch["option_pos"])
    assert logits.dim() == 1  # [N_opt] for a single head
    g = grouped_logits(logits, batch["option_group"], batch["option_slot"], batch["valid"])
    assert g.shape == batch["valid"].shape


def test_old_checkpoint_without_n_heads_loads_as_k1(tiny, tok, tmp_path):
    """A config without `n_heads` still loads; K is read off head.pt."""
    make_model(tiny, tok, n_heads=1).save(tmp_path)
    cfg = json.loads((tmp_path / "lod.json").read_text())
    assert cfg["n_heads"] == 1
    del cfg["n_heads"], cfg["head_seed"]  # exactly what a pre-ensemble checkpoint looks like
    (tmp_path / "lod.json").write_text(json.dumps(cfg, indent=2) + "\n")

    loaded = OptionScoringModel.load(tmp_path, torch_dtype=torch.float32)
    assert loaded.n_heads == 1
    assert loaded.head[-1].weight.shape[0] == 1
    packer = Packer(tok, 64, 192)
    batch = collate([packer.pack(e) for e in toy_examples(2)], packer.pad_id, torch.float32)
    with torch.no_grad():
        out = loaded(batch["input_ids"], batch["position_ids"],
                     batch["attention_mask"], batch["option_pos"])
    assert out.dim() == 1


def test_k_head_checkpoint_round_trips(tiny, tok, tmp_path):
    model = make_model(tiny, tok, n_heads=K, head_seed=3)
    model.save(tmp_path)
    loaded = OptionScoringModel.load(tmp_path, torch_dtype=torch.float32)
    assert loaded.n_heads == K and loaded.head_seed == 3
    assert torch.equal(loaded.head[-1].weight, model.head[-1].weight)


# ---- the combination rule ----------------------------------------------------


def test_grouped_logits_of_k_heads_is_log_of_the_mean_probability():
    torch.manual_seed(0)
    nq, k, h = 5, 4, K
    hg = torch.randn(nq, k, h)
    valid = torch.ones(nq, k, dtype=torch.bool)
    valid[2, 3] = False
    hg = hg.masked_fill(~valid.unsqueeze(-1), float("-inf"))

    combined = log_mean_prob(hg, valid)
    want = torch.stack([torch.softmax(hg[:, :, i].masked_fill(~valid, float("-inf")), dim=-1)
                        for i in range(h)]).mean(0)
    assert torch.allclose(probabilities(combined, valid), want, atol=1e-6)
    # it is a distribution over each question's own options, and nothing leaks to a
    # slot that has no option
    assert torch.allclose(probabilities(combined, valid).sum(-1), torch.ones(nq), atol=1e-6)
    assert probabilities(combined, valid)[2, 3] == 0.0


def test_mean_probability_is_never_more_confident_than_its_boldest_head():
    """Why p̄ and not mean-of-logits: one head cannot veto the rest."""
    # head 0 is undecided over three options; head 1 is certain option 0 is wrong
    hg = torch.tensor([[[0.0, -20.0], [0.0, 0.0], [0.0, 0.0]]])  # [1, 3 options, 2 heads]
    valid = torch.ones(1, 3, dtype=torch.bool)
    p_mean = probabilities(log_mean_prob(hg, valid), valid)
    p_logit_mean = probabilities(hg.mean(-1), valid)
    per_head = torch.stack([probabilities(hg[:, :, i], valid) for i in range(2)])
    # the arithmetic mean cannot out-confident its boldest member
    assert p_mean.max() <= per_head.max() + 1e-6
    # and one head is not allowed to veto an option the other still rates 1/3
    assert p_mean[0, 0] > 0.15 and p_logit_mean[0, 0] < 1e-4


def test_shard_select_reads_each_question_off_its_own_head():
    torch.manual_seed(0)
    hg = torch.randn(6, 3, K)
    shard = torch.tensor([0, 7, 3, 3, 1, 6])
    got = shard_select(hg, shard)
    for i, s in enumerate(shard.tolist()):
        assert torch.equal(got[i], hg[i, :, s])


def test_only_the_owning_head_gets_gradient_but_the_backbone_gets_all_of_it(tiny, tok):
    model = make_model(tiny, tok, n_heads=K)
    for p in model.backbone.parameters():
        p.requires_grad_(True)
    packer = Packer(tok, 64, 192)
    ds = DecisionDataset(toy_examples(16), packer, augment=True, seed=0, n_heads=K)
    batch = collate([ds[i] for i in range(16)], packer.pad_id, torch.float32)
    logits = model(batch["input_ids"], batch["position_ids"],
                   batch["attention_mask"], batch["option_pos"])
    hg = grouped_head_logits(logits, batch["option_group"], batch["option_slot"],
                             batch["valid"])
    loss = decision_loss(shard_select(hg, batch["q_shard"]), batch["valid"],
                         batch["target"], "ce", batch["has_target"])
    loss.backward()

    used = sorted(set(batch["q_shard"].tolist()))
    grad = model.head[-1].weight.grad
    for k in range(K):
        touched = grad[k].abs().sum().item() > 0
        assert touched == (k in used), f"head {k}: grad={touched}, in batch={k in used}"
    emb = model.backbone.get_input_embeddings().weight.grad
    assert emb is not None and emb.abs().sum().item() > 0


# ---- the decorrelation levers ------------------------------------------------


def test_shards_are_a_stable_balanced_hash_of_the_content():
    exs = toy_examples(4000, seed=1)
    counts = [0] * K
    for e in exs:
        s = example_shard(e, K)
        assert s == example_shard(e, K)  # deterministic
        counts[s] += 1
    assert min(counts) > 0.7 * len(exs) / K, counts
    assert all(example_shard(e, 1) == -1 for e in exs[:10])  # K=1 routes nothing


def test_option_order_is_per_head_per_example_and_targets_follow(tok):
    """The order differs example by example, and the target still names the same option."""
    packer = Packer(tok, 64, 192)
    exs = toy_examples(64)
    plain = DecisionDataset(exs, packer, augment=True, seed=0, n_heads=1)
    sharded = DecisionDataset(exs, packer, augment=True, seed=0, n_heads=K)
    differ = sum(1 for i in range(len(exs))
                 if plain[i].input_ids != sharded[i].input_ids)
    assert differ > len(exs) // 4, differ  # the shard is in the permutation key

    # per example, not one fixed permutation per head: two examples on the same head
    # get different orders
    orders = set()
    for i, e in enumerate(exs):
        s = example_shard(e, K)
        r = random.Random((0, 0, i, s).__hash__())
        shuffled = shuffle_options(e, r)
        q, q0 = shuffled.questions[0], e.questions[0]
        assert q.options[q.target] == q0.options[q0.target]  # logits and target agree
        orders.add(tuple(q.options))
    assert len(orders) > 1, "one fixed permutation per head is not what was asked for"


def test_per_head_init_is_seeded_and_distinct(tiny, tok):
    a = make_model(tiny, tok, n_heads=K, head_seed=11).head[-1].weight.detach()
    b = make_model(tiny, tok, n_heads=K, head_seed=11).head[-1].weight.detach()
    c = make_model(tiny, tok, n_heads=K, head_seed=12).head[-1].weight.detach()
    assert torch.equal(a, b)                       # reproducible
    assert not torch.allclose(a, c)                # head_seed moves it
    for i in range(K):
        for j in range(i + 1, K):
            assert not torch.allclose(a[i], a[j])  # and the heads start apart


# ---- the headline: eight heads train to eight different functions ------------


def _train_probe(model, ds, packer, steps=600, sharded=True, lr=0.05):
    """Train the head(s) on a frozen random backbone. Linear probes, so CPU-cheap."""
    for p in model.backbone.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(model.head.parameters(), lr=lr)
    items = [ds[i] for i in range(len(ds))]
    batch = collate(items, packer.pad_id, torch.float32)
    with torch.no_grad():
        feats = model.backbone(input_ids=batch["input_ids"],
                               position_ids=batch["position_ids"],
                               attention_mask=batch["attention_mask"],
                               use_cache=False).last_hidden_state
    rows = feats.reshape(-1, feats.shape[-1]).index_select(0, batch["option_pos"])
    for _ in range(steps):
        opt.zero_grad()
        logits = model.head(rows)
        if model.n_heads == 1:
            logits = logits.squeeze(-1)
        hg = grouped_head_logits(logits, batch["option_group"], batch["option_slot"],
                                 batch["valid"]) if model.n_heads > 1 else None
        if hg is None:
            g = grouped_logits(logits, batch["option_group"], batch["option_slot"],
                               batch["valid"])
            loss = decision_loss(g, batch["valid"], batch["target"], "ce")
        elif sharded:
            loss = decision_loss(shard_select(hg, batch["q_shard"]), batch["valid"],
                                 batch["target"], "ce")
        else:
            # the control: every head sees every example
            loss = sum(decision_loss(hg[:, :, k], batch["valid"], batch["target"], "ce")
                       for k in range(model.n_heads)) / model.n_heads
        loss.backward()
        opt.step()
    with torch.no_grad():
        logits = model.head(rows)
        if model.n_heads == 1:
            logits = logits.squeeze(-1)
        hg = grouped_head_logits(logits, batch["option_group"], batch["option_slot"],
                                 batch["valid"])
    return hg, batch


def test_eight_heads_learn_different_functions(tiny, tok, capsys):
    packer = Packer(tok, 64, 192)
    exs = toy_examples(192)

    model = make_model(tiny, tok, n_heads=K, head_seed=7)
    ds = DecisionDataset(exs, packer, augment=True, seed=0, n_heads=K)
    hg, batch = _train_probe(model, ds, packer, sharded=True)
    corr = head_correlation(hg, batch["valid"])
    dis = head_disagreement(hg, batch["valid"])

    # the control: identical inits, every head trained on every example. Same
    # architecture, levers off -- if this does not come out at 1.0 the metric is
    # measuring noise rather than decorrelation.
    ctrl = make_model(tiny, tok, n_heads=K, head_seed=7)
    with torch.no_grad():
        ctrl.head[-1].weight.copy_(ctrl.head[-1].weight[:1].expand(K, -1))
        ctrl.head[-1].bias.copy_(ctrl.head[-1].bias[:1].expand(K))
    ctrl_ds = DecisionDataset(exs, packer, augment=True, seed=0, n_heads=1)
    ctrl_hg, ctrl_batch = _train_probe(ctrl, ctrl_ds, packer, sharded=False)
    ctrl_corr = head_correlation(ctrl_hg, ctrl_batch["valid"])

    p = probabilities(log_mean_prob(hg, batch["valid"]), batch["valid"])
    ens_acc = (p.argmax(-1) == batch["target"].argmax(-1)).float().mean().item()
    accs = [(probabilities(hg[:, :, k], batch["valid"]).argmax(-1)
             == batch["target"].argmax(-1)).float().mean().item() for k in range(K)]

    with capsys.disabled():
        print(f"\n  K={K} mean pairwise head correlation = {corr['mean']:.4f} "
              f"(min {corr['min']:.4f}, max {corr['max']:.4f})")
        print(f"  mean pairwise argmax disagreement     = {dis:.4f}")
        print(f"  control (same init, no shards)        = {ctrl_corr['mean']:.4f}")
        print(f"  per-head acc {min(accs):.3f}..{max(accs):.3f}   ensemble acc {ens_acc:.3f}")

    assert ctrl_corr["mean"] > 0.999, "the control heads should be one function"
    assert corr["mean"] < 0.95, corr
    assert dis > 0.0
    # and the point of paying for eight of them: each head saw an eighth of the data,
    # and the mean of their distributions is better than any of them
    assert ens_acc > max(accs), (ens_acc, accs)


# ---- the three reported numbers ----------------------------------------------


def test_aurc_matches_the_dump_definition():
    from lod.evaluation import dumps

    torch.manual_seed(0)
    logits = torch.randn(200, 4)
    valid = torch.ones(200, 4, dtype=torch.bool)
    target = torch.zeros(200, 4)
    target[torch.arange(200), torch.randint(0, 4, (200,))] = 1.0
    probs = probabilities(logits, valid)
    records = [{"confidence": float(probs[i].max()),
                "correct": bool(probs[i].argmax() == target[i].argmax())}
               for i in range(200)]
    assert math.isclose(aurc(logits, valid, target), dumps.aurc(records), rel_tol=1e-6)


def test_p_wrong_at_high_confidence_counts_the_right_set():
    logits = torch.tensor([[5.0, 0.0], [5.0, 0.0], [0.2, 0.0]])
    valid = torch.ones(3, 2, dtype=torch.bool)
    target = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    out = p_wrong_at_confidence(logits, valid, target, 0.9)
    assert out["n"] == 2 and math.isclose(out["p_wrong"], 0.5)
    assert p_wrong_at_confidence(logits, valid, target, 0.999)["n"] == 0


def test_one_temperature_is_fitted_on_p_bar_not_per_head():
    torch.manual_seed(0)
    nq, k = 400, 3
    truth = torch.randint(0, k, (nq,))
    target = torch.zeros(nq, k)
    target[torch.arange(nq), truth] = 1.0
    valid = torch.ones(nq, k, dtype=torch.bool)
    base = torch.randn(nq, k) * 0.5
    base[torch.arange(nq), truth] += 1.0
    hg = (base.unsqueeze(-1) * 4.0 + torch.randn(nq, k, K) * 0.3)  # overconfident heads

    t = fit_temperature_ensemble(hg, valid, target)
    assert t > 1.0, t  # overconfident => flatten
    before = log_mean_prob(hg, valid)
    after = log_mean_prob(hg / t, valid)

    def nll(g):
        return float((-(target * torch.log(probabilities(g, valid) + 1e-12)).sum(-1)).mean())
    assert nll(after) < nll(before)

    # per-head T then average is a different object: its NLL on p̄ is not better
    per_head = [fit_temperature(hg[:, :, i], valid, target) for i in range(K)]
    mixed = log_mean_prob(torch.stack([hg[:, :, i] / per_head[i] for i in range(K)], -1), valid)
    assert nll(after) <= nll(mixed) + 1e-6

    # K=1 degenerates to the scalar fit we already ship
    one = fit_temperature_ensemble(hg[:, :, :1], valid, target)
    assert math.isclose(one, fit_temperature(hg[:, :, 0], valid, target), rel_tol=1e-3)
