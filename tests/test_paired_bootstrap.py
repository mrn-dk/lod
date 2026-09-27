"""Paired comparison of two eval dumps: join by question, not by row; resample tasks."""

import numpy as np
import pytest

from lod.evaluation import bootstrap as pb
from lod.evaluation.dumps import question_key
from lod.schema import Question


def _rec(task, key, p_right, correct=True):
    return {"task": task, "key": key, "probs": [p_right, 1 - p_right], "target": [1.0, 0.0],
            "correct": correct}


def test_question_key_ignores_option_order_but_not_state():
    q = Question("q", "Which?", ["a", "b", "c"], 0)
    r = Question("q", "Which?", ["c", "a", "b"], 1)
    assert question_key("t", "s", q) == question_key("t", "s", r)
    assert question_key("t", "s", q) != question_key("t", "s2", q)


def test_join_by_key_survives_dropped_and_reordered_rows():
    a = [_rec("t1", "k1", 0.9), _rec("t1", "k2", 0.8), _rec("t2", "k3", 0.7)]
    b = [_rec("t2", "k3", 0.6), _rec("t1", "k1", 0.5)]          # k2 dropped, order changed
    ia, ib, how = pb.join(a, b)
    assert how == "key" and len(ia) == 2
    assert [a[i]["key"] for i in ia] == [b[j]["key"] for j in ib]
    res = pb.compare(a, b, "nll", "none", n_boot=200)
    assert res["n_joined"] == 2 and res["dropped_a"] == 1 and res["dropped_b"] == 0
    expect = np.mean([-np.log(0.9) + np.log(0.5), -np.log(0.7) + np.log(0.6)])
    assert abs(res["mean_diff"] - expect) < 1e-12
    assert res["p_a_better"] == 1.0 and res["distinguishable"]


def test_duplicate_keys_pair_by_occurrence():
    a = [_rec("t", "k", 0.9), _rec("t", "k", 0.9)]
    b = [_rec("t", "k", 0.5)]
    ia, ib, _ = pb.join(a, b)
    assert list(ia) == [0] and list(ib) == [0]


def test_task_clustering_widens_the_interval_for_correlated_questions():
    rng = np.random.default_rng(0)
    tasks = np.repeat(np.arange(20), 50)
    diff = rng.normal(0, 1, 20)[tasks] * 0.5 + rng.normal(0, 0.1, tasks.size)
    per_q = pb.paired_bootstrap(diff, None, 2000)
    per_t = pb.paired_bootstrap(diff, tasks, 2000)
    assert per_q["mean_diff"] == per_t["mean_diff"]
    width = lambda r: r["ci95"][1] - r["ci95"][0]
    assert width(per_t) > 3 * width(per_q)
    one = pb.paired_bootstrap(diff[:50], tasks[:50], 100)
    assert np.isnan(one["ci95"][0]) and not one["distinguishable"]


def test_position_join_refuses_misaligned_dumps():
    a = [{"task": "x", "probs": [1.0], "target": [1.0], "correct": True}]
    b = [{"task": "y", "probs": [1.0], "target": [1.0], "correct": True}]
    with pytest.raises(ValueError, match="not row-aligned"):
        pb.join(a, b)
