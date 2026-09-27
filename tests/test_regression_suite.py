"""lod/evaluation/regression.py: exact posteriors, safe nonce insertion, un-permuting, buckets,
and every probe end to end on the tiny backbone (a handful of examples each)."""

import argparse
import json
import math
import random
import re
import sys
from fractions import Fraction as F
from pathlib import Path

import pytest

from lod.evaluation import regression as rg
from lod.schema import Example, Question

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


# ---- exact posteriors ----------------------------------------------------------------

def test_posteriors_match_hand_computed_values():
    fair = [F(1, 6)] * 6
    six = [F(1, 10)] * 5 + [F(1, 2)]
    # two sixes: fair 1/36, loaded 1/4 -> 1/10 vs 9/10
    assert rg.dice_posterior([fair, six], [F(1, 2), F(1, 2)], [6, 6]) == [F(1, 10), F(9, 10)]
    # without replacement, two reds are impossible from an urn holding one red
    assert rg.urn_posterior([[1, 1], [2, 0]], [F(1, 2)] * 2, [0, 0], False) == [0, 1]
    # with replacement: 1/4 against 1
    assert rg.urn_posterior([[1, 1], [2, 0]], [F(1, 2)] * 2, [0, 0], True) == [F(1, 5), F(4, 5)]
    assert rg.sensor_posterior(F(1, 2), F(1, 10), F(1, 10), [True]) == F(9, 10)
    # base rate 1/100, 5% FPR, no misses: one positive -> 1/(1 + 99*0.05)
    assert rg.sensor_posterior(F(1, 100), F(1, 20), F(0), [True]) == F(20, 119)
    # 0.9*0.4 against 0.1*0.6
    assert rg.sources_posterior(F(1, 2), [F(9, 10), F(6, 10)], [True, False]) == F(6, 7)


def test_posterior_probes_are_deterministic_normalised_and_labelled():
    a, b = rg.posterior_probes(24, seed=3), rg.posterior_probes(24, seed=3)
    assert [e.to_dict() for e in a] == [e.to_dict() for e in b]
    assert [e.to_dict() for e in a] != [e.to_dict() for e in rg.posterior_probes(24, seed=4)]
    fams = set()
    for e in a:
        q = e.questions[0]
        assert e.task.startswith("probe_posterior_")
        assert abs(sum(q.target) - 1) < 1e-9 and len(q.target) == len(q.options)
        assert q.meta["probe"] is True
        fams.add(q.meta["family"])
    assert fams == set(rg.FAMILIES)


# ---- nonce insertion -----------------------------------------------------------------

def _ex(state, question="Is the order urgent?", options=("no", "yes")):
    return Example(state=state, task="t", questions=[Question("q", question, list(options), 0)])


def test_json_key_insertion_keeps_every_original_field_and_format():
    obj = {"customer": "Dana", "order_id": "ORD-77", "items": [{"sku": "A1", "qty": 2}],
           "note": "please hurry"}
    for kw in ({"indent": 2}, {}):
        state = json.dumps(obj, **kw)
        vs = rg.nonce_variants(_ex(state), random.Random(0))
        new = json.loads(vs["json_key"].state)
        added = [k for k in new if k not in obj]
        assert len(added) == 1 and re.fullmatch(r"[0-9a-f-]{36}", new[added[0]])
        assert {k: v for k, v in new.items() if k in obj} == obj
        assert vs["json_key"].state == json.dumps(new, **kw)
        # the order id occurs once and not in the question: it may be replaced
        rep = json.loads(vs["id_replace"].state)
        assert rep["order_id"] != "ORD-77" and rep["customer"] == "Dana"


def test_id_replace_leaves_cross_referenced_and_quoted_ids_alone():
    state = json.dumps({"id": 5, "parent_id": 5, "user_id": "u-19", "text": "x"}, indent=2)
    vs = rg.nonce_variants(_ex(state), random.Random(0))
    rep = json.loads(vs["id_replace"].state)
    assert rep["id"] == 5 and rep["parent_id"] == 5 and rep["user_id"] != "u-19"
    quoted = _ex(state, "Is u-19 a new user?")
    rep = json.loads(rg.nonce_variants(quoted, random.Random(0)).get(
        "id_replace", Example(state=state)).state)
    assert rep["user_id"] == "u-19"


def test_unsafe_examples_are_skipped():
    assert rg.nonce_variants(_ex('{"a": 1}', "How many fields are set?"), random.Random(0)) == {}
    assert rg.nonce_variants(_ex("line one\n\nline two", "Which line mentions a refund?"),
                             random.Random(0)) == {}
    # a JSON state that does not round-trip byte for byte gets no JSON insertion
    assert rg.nonce_variants(_ex('{"price": 1.0,  "x": 2}'), random.Random(0)) == {}


def test_text_line_is_its_own_paragraph():
    state = "Hello team.\n\nThe printer is on fire again."
    v = rg.nonce_variants(_ex(state), random.Random(1))["text_line"].state
    paras = v.split("\n\n")
    assert len(paras) == 3 and [p for p in paras if ": " in p and len(p) > 36]
    assert [p for p in paras if p in state.split("\n\n")] == state.split("\n\n")


# ---- permutation bookkeeping, with fake scorers --------------------------------------

class TextScorer:
    """Log p depends only on each option's text: exactly order-invariant."""

    def score(self, examples):
        out = []
        for e in examples:
            row = []
            for q in e.questions:
                w = [((sum(map(ord, o)) * 7919) % 97) / 10.0 for o in q.options]
                z = math.log(sum(math.exp(x) for x in w))
                row.append([x - z for x in w])
            out.append(row)
        return out


class FirstScorer(TextScorer):
    """Always prefers whatever comes first."""

    def score(self, examples):
        return [[[math.log(0.9)] + [math.log(0.1 / (len(q.options) - 1))] * (len(q.options) - 1)
                 for q in e.questions] for e in examples]


def _perm_examples():
    q1 = Question("a", "Pick one.", ["alpha", "beta", "gamma", "delta"], 2,
                  descriptions=["first", None, "third", "fourth"])
    q2 = Question("b", "Yes?", ["no", "yes"], [0.3, 0.7])
    return [Example(state="s", task="t", questions=[q1, q2]),
            Example(state="s2", task="t", questions=[q2])]


def test_permute_question_moves_descriptions_and_target_with_options():
    q = _perm_examples()[0].questions[0]
    p = rg.permute_question(q, [3, 0, 2, 1])
    assert p.options == ["delta", "alpha", "gamma", "beta"]
    assert p.descriptions == ["fourth", "first", "third", None]
    assert p.options[p.target] == "gamma"


def test_unpermuting_an_order_invariant_scorer_gives_zero():
    r = rg.run_permutation(TextScorer(), _perm_examples(), k=3)
    assert r["max_abs_dp"] == 0.0 and r["flip_rate"] == 0.0 and r["n_questions"] == 3
    biased = rg.run_permutation(FirstScorer(), _perm_examples(), k=3)
    # a non-identity permutation may keep the first option first, so not every draw flips
    assert biased["flip_rate"] > 0.5 and biased["max_abs_dp"] > 0.7


def test_nonce_with_an_invariant_scorer_is_zero():
    exs = [_ex(json.dumps({"order_id": "X1", "body": "late"}, indent=2)), _ex("Plain text.")]
    r = rg.run_nonce(TextScorer(), exs)
    assert r["n_variants"] == 3 and r["max_abs_dlogp"] == 0.0 and r["flip_rate"] == 0.0
    assert set(r["by_type"]) == {"json_key", "id_replace", "text_line"}


# ---- option count ---------------------------------------------------------------------

def test_option_count_buckets():
    recs = ([{"probs": [0.8, 0.2], "target": [1, 0]}] * 3
            + [{"probs": [0.1] * 10, "target": [0] * 9 + [1]}])
    b = {r["bucket"]: r for r in rg.option_count_buckets(recs)}
    assert b["2"]["n"] == 3 and b["2"]["acc"] == 1.0 and abs(b["2"]["ece"] - 0.2) < 1e-9
    assert b["9-16"]["n"] == 1 and abs(b["9-16"]["nll"] - math.log(10)) < 1e-9
    assert b["129+"]["n"] == 0 and len(b) == len(rg.BUCKETS)


def test_lookup_example_answer_is_in_the_state():
    for n in (2, 16, 256):
        e = rg.lookup_example(n, random.Random(n))
        q = e.questions[0]
        assert len(q.options) == n == len(set(q.options))
        bin_ = re.search(r"bin (\d+)\?", q.question).group(1)
        assert f"{q.options[q.target]}: bin {bin_}\n" in e.state + "\n"


# ---- end to end on the tiny backbone -------------------------------------------------

@pytest.fixture(scope="module")
def scorer(model):
    return rg.Scorer(model, "cpu", batch_size=2)


def test_fits_rejects_a_truncated_state(scorer):
    short = rg.lookup_example(4, random.Random(0))
    assert scorer.fits(short)
    # questions fit, so the packer would keep it -- by cutting the state
    long = Example(state="word " * 3000, task="t", questions=short.questions)
    assert scorer.n_tokens(long) is not None and not scorer.fits(long)


def test_scorer_maps_rows_back_to_examples(scorer):
    exs = [rg.lookup_example(4, random.Random(0)), _ex("Short state."),
           Example(state="x", task="t", questions=[Question("u", "Unlabelled?", ["a", "b", "c"])])]
    fwd = scorer.score(exs)
    rev = scorer.score(exs[::-1])[::-1]
    for a, b in zip(fwd, rev):
        for x, y in zip(a, b):
            assert x is not None and len(x) == len(y)
            assert max(abs(u - v) for u, v in zip(x, y)) < 1e-4
            assert abs(sum(math.exp(u) for u in x) - 1) < 1e-6


def test_all_probes_run_on_the_tiny_model(scorer):
    post = rg.run_posterior(scorer, n=4)
    assert post["n"] == 4 and set(post["per_family"]) == set(rg.FAMILIES)
    assert post["kl"] >= 0 and 0 <= post["tv"] <= 1 and post["calib_slope"] is not None
    exs = [_ex(json.dumps({"ticket_id": "T-9", "text": "refund please"}, indent=2)),
           _ex("Customer writes: where is my parcel?")]
    nonce = rg.run_nonce(scorer, exs)
    assert nonce["n_variants"] == 3 and nonce["mean_abs_dlogp"] >= 0
    perm = rg.run_permutation(scorer, _perm_examples(), k=2)
    assert perm["n_questions"] == 3 and perm["max_abs_dp"] >= 0
    sweep = rg.run_sweep(scorer, ns=(2, 4, 512), reps=2)
    rows = {r["N"]: r for r in sweep["rows"]}
    assert rows[2]["fits"] and rows[2]["n"] == 2 and 0 <= rows[2]["acc"] <= 1
    assert not rows[512]["fits"] and sweep["tested_up_to"] == 4


def test_suite_run_writes_every_section(scorer, tmp_path):
    import regression_suite

    data = tmp_path / "dev.jsonl"
    rows = [_ex(json.dumps({"order_id": f"O{i}", "text": "late parcel"}, indent=2))
            for i in range(3)] + [_ex(f"Plain message {i}.") for i in range(3)]
    data.write_text("\n".join(json.dumps(e.to_dict()) for e in rows) + "\n")
    args = argparse.Namespace(
        ckpt=tmp_path, device="cpu", batch_size=2, tests="posterior,nonce,perm,count",
        seed=0, n_posterior=4, n_nonce=4, n_perm=2, k_perm=2, max_tokens=512, per_task=10,
        data=data, corpus=None, dump=None, sweep_ns="2,4", sweep_reps=2)
    out = regression_suite.run(scorer.model, args)
    assert {"known_posterior", "nonce", "permutation", "option_count"} <= set(out)
    assert out["sample"]["n_examples"] == 4
    assert out["option_count"]["tested_up_to"] == 4
    assert sum(b["n"] for b in out["option_count"]["buckets"]) == 4
    json.dumps(out)
