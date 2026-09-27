"""The report layer: metrics read off eval dumps agree with the tensor metrics, and the
gates are computed from a report directory without a model."""

import json

import torch

from lod.evaluation import bootstrap, breakdowns, dumps, gates
from lod.evaluation.engine import per_question_records
from lod.model.calibration import metrics


def _records(n=300, k=4, seed=0):
    torch.manual_seed(seed)
    logits = torch.randn(n, k) * 2
    valid = torch.ones(n, k, dtype=torch.bool)
    valid[: n // 3, 3] = False                        # some binary-ish, some wider
    logits = logits.masked_fill(~valid, float("-inf"))
    target = torch.zeros(n, k)
    target[torch.arange(n), torch.randint(0, 3, (n,))] = 1.0
    recs = per_question_records(logits, valid, target, [f"task{i % 7}" for i in range(n)])
    return logits, valid, target, recs


def test_dump_metrics_match_tensor_metrics():
    logits, valid, target, recs = _records()
    m = metrics(logits, valid, target)
    assert abs(dumps.accuracy(recs) - m["acc"]) < 1e-6        # float32 mean
    assert abs(dumps.nll(recs) - m["nll"]) < 1e-4       # the dump rounds to 6 dp
    assert abs(dumps.ece(recs) - m["ece"]) < 1e-4


def test_dump_ci_brackets_the_point_estimate():
    *_, recs = _records()
    res = bootstrap.dump_ci(recs, n_boot=200)
    for level in ("micro", "macro"):
        for k in bootstrap.CI_METRICS:
            lo, hi = res[level][k]["ci95"]
            assert lo <= res[level][k]["value"] <= hi, (level, k)
    assert res["n_tasks"] == 7


def test_gates_from_a_report_directory(tmp_path):
    *_, recs = _records()
    report, ckpt = tmp_path / "report", tmp_path / "ckpt"
    report.mkdir()
    ckpt.mkdir()
    with open(report / "eval-testreal.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    # the per-case lines come first; the gate reads the final maximum, not the first number
    (report / "independence.txt").write_text(
        "  A alone (reference)    max|delta logit| = 0.000e+00\n"
        "  A last of three        max|delta logit| = 3.000e-06\n\n"
        "max |delta logit| = 3.000e-06   threshold 1e-04   PASS\n")
    (ckpt / "lod.json").write_text(json.dumps({"temperature": 1.0}))
    (ckpt / "confidence_summary.json").write_text(json.dumps(
        {"conf_gap": 0.3, "aurc_ratio": 0.9, "head_aurc": 0.1, "maxprob_aurc": 0.11}))
    res = gates.compute(report, ckpt)
    g = res["gates"]
    assert g["independence"]["max_abs_delta_logit"] == 3e-6 and g["independence"]["pass"]
    assert g["abstention"]["pass"] and g["confidence_beats_maxprob"]["pass"]
    assert g["score_coherence"]["pass"]
    assert g["probability_recovery"]["pass"] is False          # no meta.p questions
    assert g["rule_transfer"]["pass"] is None and "missing" in g["rule_transfer"]
    assert res["blocking_failed"] == [k for k in gates.BLOCKING if not g[k]["pass"]]
    assert all(g[k]["blocking"] for k in gates.BLOCKING)
    json.dumps(res)


def test_depth_grid_renders_cells():
    rows = [{"correct": i % 2 == 0, "probs": [0.6, 0.4], "target": [1.0, 0.0],
             "meta": {"gen": {"cell": {"length": L, "depth": d}, "family": "f"}}}
            for i, (L, d) in enumerate([(1000, 0.1), (2000, 0.5), (16000, 0.9)] * 4)]
    lines = breakdowns.depth_grid(rows, "all")
    assert lines[0].startswith("all: 12 questions")
    assert any(line.strip().startswith("16k") for line in lines)
