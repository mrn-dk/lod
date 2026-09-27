"""Row 48 -- exact posteriors (`sources/synth/posterior.py`).

The target is the whole point of this row, so it is checked twice by routes that share no
code with the generator's arithmetic: every target against a Monte Carlo simulation of the
stated story (`simulate`, which reads only the story's raw parameters -- compositions,
rates, reports -- and draws balls, tosses coins and opens doors), and every rendered option
against the target of the option text it carries.
"""

from __future__ import annotations

import copy
import math
import re
from collections import Counter, defaultdict
from fractions import Fraction as F

import numpy as np
import pytest

from lod.corpus.services.sources.real.base import family_of
from lod.corpus.services.sources.synth import posterior as P

SEEDS = {"train": P.SEED_TRAIN, "devreal": P.SEED_DEV, "testreal": P.SEED_TEST}


def _seed(v):
    return SEEDS[P.VARIANTS[v][1]]


@pytest.fixture(scope="module")
def sample():
    """Per variant, 80 (draft, info, example) triples at the variant's own task seed."""
    return {v: list(P.generate(v, _seed(v), 80)) for v in P.VARIANTS}


@pytest.fixture(scope="module")
def tasks():
    return P.tasks()


# ---- the registry and the splits ------------------------------------------------------

def test_every_task_is_row_48_generated_soft_and_split_explicitly(tasks):
    assert len(tasks) == len(P.TRAINED) * 2 + len(P.DEV) + len(P.TEST)
    for t in tasks:
        assert t.row == 48 and not t.real and t.soft and t.per_example_options
        assert t.name.startswith("post_")
        assert t.force_split in ("train", "devreal", "testreal"), t.name
        assert t.family, t.name


def test_held_out_families_are_disjoint_across_splits(tasks):
    """dev and test hold out whole variants, disjoint from train and from each other.
    The `_eval` twins are the only devreal tasks whose family trains."""
    train = {family_of(t) for t in tasks if t.force_split == "train"}
    test = {family_of(t) for t in tasks if t.force_split == "testreal"}
    dev_all = [t for t in tasks if t.force_split == "devreal"]
    twins = {family_of(t) for t in dev_all if t.name.endswith("_eval")}
    dev = {family_of(t) for t in dev_all if not t.name.endswith("_eval")}
    assert not train & test and not train & dev and not dev & test
    assert twins == train, "every trained variant has exactly one in-structure twin"
    assert dev and test


def test_every_concept_family_trains_and_is_held_out_in_both_eval_splits():
    by = defaultdict(set)
    for fam, split in P.VARIANTS.values():
        by[fam].add(split)
    assert set(by) == set(P.FAMILIES)
    for fam, splits in by.items():
        assert splits == {"train", "devreal", "testreal"}, fam


def test_no_state_is_shared_between_a_trained_variant_and_its_twin():
    for v in P.TRAINED[:6]:
        a = {ex.state for _, _, ex in P.generate(v, P.SEED_TRAIN, 150)}
        b = {ex.state for _, _, ex in P.generate(v, P.SEED_TWIN, 60)}
        assert not a & b, v


def test_every_loader_yields_what_it_is_asked_for(tasks):
    for t in tasks[:6]:
        got = list(t.load(12))
        assert len(got) == 12, t.name


# ---- determinism ----------------------------------------------------------------------

def test_generation_is_seeded():
    for v in ("urn_which_norepl", "sensor_mixed", "sources_relay", "chain_host",
              "missing_two", "coin_nextcount"):
        a = [ex.to_dict() for _, _, ex in P.generate(v, 11, 25)]
        b = [ex.to_dict() for _, _, ex in P.generate(v, 11, 25)]
        c = [ex.to_dict() for _, _, ex in P.generate(v, 12, 25)]
        assert a == b, v
        assert a != c, v


# ---- targets --------------------------------------------------------------------------

def test_targets_are_well_formed(sample):
    for v, triples in sample.items():
        for d, info, ex in triples:
            (q,) = ex.questions
            assert 2 <= len(q.options) <= 12, v
            assert len(set(q.options)) == len(q.options), (v, q.options)
            assert isinstance(q.target, list) and len(q.target) == len(q.options)
            assert all(t >= 0 for t in q.target)
            assert abs(sum(q.target) - 1) < 1e-9, (v, sum(q.target))
            if q.descriptions is not None:
                assert len(q.descriptions) == len(q.options)
            assert sum(info["exact"]) == 1, "the exact target is a Fraction vector"


def test_each_option_carries_the_target_of_its_own_outcome(sample):
    """Re-render each draft from its seed and check the shuffled options by their text:
    option `o` must carry the exact probability of the outcome whose text is `o`."""
    for v, triples in sample.items():
        for d, info, ex in triples[:20]:
            (q,) = ex.questions
            i = int(q.id.rsplit("_", 1)[1])
            rng = P._rng(_seed(v), f"{v}:{i}")
            d2 = P.DRAW[v](rng, v)
            opts = d2.render(copy.deepcopy(rng))[3]
            exact = P.exact_target(d2)
            offered = d2.keep if d2.keep is not None else list(range(len(opts)))
            by_text = {opts[o]: float(exact[j]) for j, o in enumerate(offered)}
            for o, t in zip(q.options, q.target):
                assert abs(by_text[o] - t) < 1e-12, (v, o)


def _sim(n):
    return max(20_000, min(2_000_000, n))


def simulate(v: str, p: dict, n_out: int, rng: np.random.Generator, budget: int):
    """Monte Carlo of the stated story from its raw parameters. -> (counts over outcomes
    in natural outcome order, accepted draws)."""
    fam = P.VARIANTS[v][0]
    counts = np.zeros(n_out)

    def weights(ws):
        a = np.array([float(x) for x in ws], dtype=float)
        return a / a.sum()

    if v.startswith("urn"):
        comps, draws, mode = p["comps"], p["draws"], p["mode"]
        n = len(draws)
        hs = rng.choice(len(comps), size=budget, p=weights(p["prior"]))
        for h in range(len(comps)):
            N = int((hs == h).sum())
            comp = np.array(comps[h])
            tot = int(comp.sum())
            if mode == "repl":
                seq = rng.choice(len(comp), size=(N, n + 1), p=comp / tot)
            elif mode == "norepl":
                balls = np.repeat(np.arange(len(comp)), comp)
                if tot < n + 1:
                    seq = np.full((N, n + 1), -1)
                else:
                    perm = np.argsort(rng.random((N, tot)), axis=1)[:, : n + 1]
                    seq = balls[perm]
            else:  # polya
                cur = np.tile(comp.astype(float), (N, 1))
                seq = np.zeros((N, n), dtype=int)
                for j in range(n):
                    cum = np.cumsum(cur / cur.sum(1, keepdims=True), axis=1)
                    c = (rng.random((N, 1)) > cum).sum(1)
                    seq[:, j] = c
                    cur[np.arange(N), c] += 1
            ok = (seq[:, :n] == np.array(draws)).all(1)
            if p["query"] == "which":
                counts[h] += ok.sum()
            else:
                np.add.at(counts, seq[ok, n], 1)
        return counts, counts.sum()

    if v.startswith("coin"):
        b = np.array([float(x) for x in p["biases"]])
        tosses = np.array(p["tosses"])
        n = len(tosses)
        hs = rng.choice(len(b), size=budget, p=weights(p["prior"]))
        extra = 1 if p["query"] == "next" else p["m"] if p["query"] == "count" else 0
        seq = (rng.random((budget, n + extra)) < b[hs][:, None]).astype(int)
        ok = (seq[:, :n] == tosses).all(1)
        if p["query"] == "which":
            np.add.at(counts, hs[ok], 1)
        elif p["query"] == "next":
            np.add.at(counts, 1 - seq[ok, n], 1)          # outcomes: heads, tails
        else:
            np.add.at(counts, seq[ok, n:].sum(1), 1)
        return counts, counts.sum()

    if v.startswith("die"):
        faces = [weights(f) for f in p["faces"]]
        rolls = p["rolls"]
        n = len(rolls)
        hs = rng.choice(len(faces), size=budget, p=weights(p["prior"]))
        for h, fp in enumerate(faces):
            N = int((hs == h).sum())
            seq = rng.choice(len(fp), size=(N, n + 1), p=fp)
            ok = (seq[:, :n] == np.array(rolls)).all(1)
            if p["query"] == "which":
                counts[h] += ok.sum()
            else:
                np.add.at(counts, seq[ok, n], 1)
        return counts, counts.sum()

    if v in ("sensor_single", "sensor_repeat", "sensor_mixed"):
        truth = rng.random(budget) < float(p["base"])
        ok = np.ones(budget, bool)
        for t, pos in p["readings"]:
            se, sp = (float(x) for x in p["tests"][t])
            says_pos = rng.random(budget) < np.where(truth, se, 1 - sp)
            ok &= says_pos == pos
        counts[0] = (truth & ok).sum()
        counts[1] = (~truth & ok).sum()
        return counts, counts.sum()

    if v.startswith("sensor_multi"):
        m = len(p["mats"][0])
        truth = rng.choice(m, size=budget, p=weights(p["prior"]))
        ok = np.ones(budget, bool)
        for dv, rep in p["readings"]:
            mat = np.array([[float(x) for x in row] for row in p["mats"][dv]])
            cum = np.cumsum(mat[truth], axis=1)
            said = (rng.random((budget, 1)) > cum).sum(1)
            ok &= said == rep
        np.add.at(counts, truth[ok], 1)
        return counts, counts.sum()

    if fam == "sources":
        m = p["m"]
        truth = rng.choice(m, size=budget, p=weights(p["prior"]))

        def noisy(v_, r):
            keep = rng.random(budget) < float(r)
            shift = rng.integers(1, m, size=budget)          # uniform over the others
            return np.where(keep, v_, (v_ + shift) % m)

        ok = np.ones(budget, bool)
        for r, q, rep in zip(p["rels"], p["relays"], p["reports"]):
            got = noisy(truth, r)
            if q is not None:
                got = noisy(got, q)
            ok &= got == rep
        np.add.at(counts, truth[ok], 1)
        return counts, counts.sum()

    if fam == "missing":
        if "counts" in p:                   # draw a record from the population table itself
            tab = np.array(p["counts"], dtype=float)
            cell = rng.choice(tab.size, size=budget, p=(tab / tab.sum()).ravel())
            a, b = np.divmod(cell, tab.shape[1])
            ok = np.isin(a, p["evidence"][0])
            np.add.at(counts, b[ok], 1)
            return counts, counts.sum()
        b = rng.choice(n_out, size=budget, p=weights(p["prior"]))
        ok = np.ones(budget, bool)
        for cond, ev in zip(p["conds"], p["evidence"]):
            mat = np.array([[float(x) for x in row] for row in cond])      # [a][b]
            cum = np.cumsum(mat[:, b].T, axis=1)
            a = (rng.random((budget, 1)) > cum).sum(1)
            ok &= np.isin(a, ev)
        np.add.at(counts, b[ok], 1)
        return counts, counts.sum()

    if v == "chain_two_step":
        pA = weights(p["pA"])
        pBA = np.array([[float(x) for x in r] for r in p["pBA"]])
        pCB = np.array([[float(x) for x in r] for r in p["pCB"]])
        a = rng.choice(len(pA), size=budget, p=pA)
        b = (rng.random((budget, 1)) > np.cumsum(pBA[a], axis=1)).sum(1)
        c = (rng.random((budget, 1)) > np.cumsum(pCB[b], axis=1)).sum(1)
        ok = c == p["c"]
        np.add.at(counts, (a if p["query"] == "A" else b)[ok], 1)
        return counts, counts.sum()

    if v == "chain_host":
        nd = len(p["prior"])
        prize = rng.choice(nd, size=budget, p=weights(p["prior"]))
        opened = np.zeros(budget, int)
        for d in range(nd):
            idx = np.where(prize == d)[0]
            allowed = [x for x in range(nd) if x != p["pick"] and x != d]
            if p["rule"] == "lowest":
                opened[idx] = min(allowed)
            elif p["rule"] == "highest":
                opened[idx] = max(allowed)
            else:
                opened[idx] = rng.choice(allowed, size=len(idx),
                                         p=weights([p["hw"][x] for x in allowed]))
        ok = opened == p["opened"]
        np.add.at(counts, prize[ok], 1)
        return counts, counts.sum()

    if v == "chain_explain":
        c1 = rng.random(budget) < float(p["p1"])
        c2 = rng.random(budget) < float(p["p2"])
        pe = np.select([c1 & c2, c1 & ~c2, ~c1 & c2],
                       [float(p["cpt"][(1, 1)]), float(p["cpt"][(1, 0)]),
                        float(p["cpt"][(0, 1)])], float(p["cpt"][(0, 0)]))
        e = rng.random(budget) < pe
        ok = e == bool(p["e"])
        if p["c2"] is not None:
            ok &= c2 == bool(p["c2"])
        if p["query"] == "joint":            # hypotheses (1,1) (1,0) (0,1) (0,0)
            idx = np.where(c1, np.where(c2, 0, 1), np.where(c2, 2, 3))
        else:
            idx = np.where(c1, 0, 1)
        np.add.at(counts, idx[ok], 1)
        return counts, counts.sum()
    raise AssertionError(v)


def test_every_target_agrees_with_a_monte_carlo_of_the_stated_story():
    """Four drafts per variant, simulated to ~40,000 accepted draws each. Tolerance: 5
    standard errors plus 0.004."""
    rng = np.random.default_rng(0)
    checked = Counter()
    for v in P.VARIANTS:
        for d, info, _ in P.generate(v, _seed(v), 40):
            if checked[v] >= 4:
                break
            # acceptance rate, only to size the simulation budget
            post_mass = float(sum(a * math.prod(lk[h] for lk in d.liks)
                                  for h, a in enumerate(d.prior)))
            if post_mass < 0.004:
                continue
            n_out = len(d.pred[0]) if d.pred is not None else len(d.prior)
            counts, acc = simulate(v, d.params, n_out, rng, _sim(int(40_000 / post_mass)))
            assert acc > 2000, (v, acc)
            mc = counts / acc
            if d.keep is not None:
                assert mc[[o for o in range(n_out) if o not in d.keep]].sum() == 0
                mc = mc[d.keep]
            for est, t in zip(mc, info["target"]):
                se = math.sqrt(max(t * (1 - t), 1e-6) / acc)
                assert abs(est - t) < 5 * se + 0.004, (v, list(mc), info["target"])
            checked[v] += 1
    assert all(checked[v] >= 3 for v in P.VARIANTS), checked


def test_posterior_matches_a_second_exact_route():
    """Batch Bayes (prior x product of likelihoods) against sequential updating, both in
    Fractions: order-free and exact."""
    for v in P.VARIANTS:
        for d, info, _ in P.generate(v, _seed(v), 20):
            w = list(d.prior)
            for lk in d.liks:
                w = [a * b for a, b in zip(w, lk)]
                s = sum(w)
                w = [x / s for x in w]
            assert w == P.posterior(d.prior, d.liks), v


def test_count_based_urn_likelihoods_match_the_hypergeometric():
    """Without replacement, P(sequence | urn) is a ratio of falling factorials: an
    independent closed form for the sequential likelihood the generator multiplies."""
    for d, _, _ in P.generate("urn_which_norepl", 0, 60):
        comps, draws = d.params["comps"], d.params["draws"]
        for h, comp in enumerate(comps):
            tot, n = sum(comp), len(draws)
            num = math.prod(math.perm(comp[c], k) for c, k in Counter(draws).items())
            closed = F(num, math.perm(tot, n))
            assert closed == math.prod(lk[h] for lk in d.liks)


# ---- the distribution of what is asked -------------------------------------------------

def test_max_posterior_covers_the_range_in_every_family(sample):
    by = defaultdict(Counter)
    for v, triples in sample.items():
        for _, info, _ in triples:
            by[P.VARIANTS[v][0]][info["bin"]] += 1
    for fam, c in by.items():
        n = sum(c.values())
        shares = [c[b] / n for b in range(P.N_BINS)]
        assert min(shares) >= 0.06, (fam, shares)
        assert max(shares) <= 0.35, (fam, shares)


def test_the_shallow_readings_stay_far_from_the_exact_answer(sample):
    """Prior-only, likelihood-only, last observation, majority vote and the
    variant-specific pickers: none may come near the exact argmax in any family."""
    by = defaultdict(lambda: defaultdict(list))
    for v, triples in sample.items():
        for _, info, _ in triples:
            for h, a in info["agree"].items():
                by[P.VARIANTS[v][0]][h].append(a)
    for fam, hs in by.items():
        for h, accs in hs.items():
            assert sum(accs) / len(accs) <= 0.80, (fam, h, sum(accs) / len(accs))


def test_the_answer_is_not_the_commonest_label(sample):
    for v, triples in sample.items():
        golds = Counter()
        for _, info, ex in triples:
            q = ex.questions[0]
            golds[q.options[max(range(len(q.options)), key=lambda i: q.target[i])]] += 1
        assert golds.most_common(1)[0][1] / len(triples) <= 0.62, (v, golds.most_common(3))


def test_meta_records_the_generator_parameters(sample):
    need = {"family", "variant", "n_options", "max_post", "conf_bin", "n_obs", "prior", "fmt"}
    fmts = Counter()
    for v, triples in sample.items():
        for _, _, ex in triples:
            g = ex.questions[0].meta["gen"]
            assert need <= set(g), (v, need - set(g))
            assert g["variant"] == v and g["family"] == P.VARIANTS[v][0]
            fmts[g["fmt"]] += 1
    assert set(fmts) == set(P.FORMATS)


def test_no_hard_coded_abstention_option(sample):
    pat = re.compile(r"none of|not listed|other value|something else", re.I)
    for v, triples in sample.items():
        for _, _, ex in triples:
            assert not any(pat.search(o) for o in ex.questions[0].options), v


# ---- token budget (Qwen3 tokenizer) ------------------------------------------------------

@pytest.fixture(scope="module")
def qwen():
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
    except Exception as exc:                     # no cache on this machine
        pytest.skip(f"tokenizer unavailable: {exc}")


def test_token_lengths_fit_the_training_budget(qwen, sample):
    """95 % of states <= 1,200 tokens and of packed examples <= 3,072 (the training
    budget); every state >= 40 (under 40 is a shortcut, not a task)."""
    from lod.model.packing import Packer

    packer = Packer(qwen, 100_000, 100_000)
    states, totals = [], []
    for v, triples in sample.items():
        for _, _, ex in triples[:40]:
            pk = packer.pack(ex)
            s = pk.block_ids.count(0)
            states.append(s)
            totals.append(len(pk))
            assert s >= 40, (v, s)
    states.sort()
    totals.sort()
    assert states[int(0.95 * len(states))] <= 1200
    assert totals[int(0.95 * len(totals))] <= 3072
