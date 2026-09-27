#!/usr/bin/env python
"""Regression suite: known posteriors, nonce robustness, option order, option count.

The probes live in `lod/evaluation/regression.py` (importable, unit-tested on a tiny
model); this runs all four against a checkpoint and writes one JSON. Every number is a
later model's baseline, so the defaults are small enough for a CPU (about half an hour for
a 0.6B model on 8 cores) and the sample is fixed by `--seed`.

    regression_suite.py --ckpt runs/<run>-release --corpus data/<corpus> \
        --json runs/<run>-release/report/regression.json

The option-count buckets read an existing eval dump (`--dump`, default the checkpoint's
`report/eval-testreal.jsonl`, then `eval-devreal.jsonl`); without one they bucket the
sampled examples instead, which is a much smaller sample.
"""

from __future__ import annotations

import os

os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")  # must precede `import torch`

import argparse
import json
import math
import time
from pathlib import Path

import torch

from lod.evaluation import regression as rg
from lod.model.scorer import OptionScoringModel

TESTS = ("posterior", "nonce", "perm", "count")


def default_data(corpus: Path | None) -> Path | None:
    if corpus is not None and (corpus / "devreal.jsonl").exists():
        return corpus / "devreal.jsonl"
    return None


def default_dump(ckpt: Path) -> Path | None:
    for name in ("eval-testreal.jsonl", "eval-devreal.jsonl"):
        for d in (ckpt / "report", ckpt):
            if (d / name).exists():
                return d / name
    return None


def run(model, args) -> dict:
    scorer = rg.Scorer(model, args.device, args.batch_size)
    tests = set(args.tests.split(","))
    out: dict = {"ckpt": str(args.ckpt), "device": str(args.device), "seed": args.seed,
                 "temperature": float(model.temperature), "timing_s": {}}

    if "posterior" in tests:
        t = time.time()
        out["known_posterior"] = rg.run_posterior(scorer, args.n_posterior, args.seed)
        out["timing_s"]["known_posterior"] = round(time.time() - t, 1)
        _say("known posterior", out["known_posterior"])

    base = sample = None
    if tests & {"nonce", "perm"} or ("count" in tests and args.dump is None):
        data = args.data or default_data(args.corpus)
        if data is None:
            raise SystemExit("no --data and no devreal.jsonl under --corpus")
        t = time.time()
        sample = rg.sample_examples(data, max(args.n_nonce, args.n_perm), scorer, args.seed,
                                    args.max_tokens, args.per_task)
        base = scorer.score(sample)
        out["sample"] = {"data": str(data), "n_examples": len(sample),
                         "n_questions": sum(len(e.questions) for e in sample),
                         "max_tokens": args.max_tokens, "per_task": args.per_task,
                         "n_tasks": len({e.task for e in sample})}
        out["timing_s"]["sample_and_base"] = round(time.time() - t, 1)

    if "nonce" in tests:
        t = time.time()
        k = min(args.n_nonce, len(sample))
        out["nonce"] = rg.run_nonce(scorer, sample[:k], args.seed, base[:k])
        out["timing_s"]["nonce"] = round(time.time() - t, 1)
        _say("nonce", out["nonce"])

    if "perm" in tests:
        t = time.time()
        k = min(args.n_perm, len(sample))
        out["permutation"] = rg.run_permutation(scorer, sample[:k], args.k_perm, args.seed,
                                                base[:k])
        out["timing_s"]["permutation"] = round(time.time() - t, 1)
        _say("permutation", out["permutation"])

    if "count" in tests:
        t = time.time()
        dump = args.dump or default_dump(args.ckpt)
        if dump is not None:
            recs = [json.loads(ln) for ln in open(dump, encoding="utf-8") if ln.strip()]
            src = str(dump)
        else:
            recs = []
            for e, lp in zip(sample, base):
                for q, l in zip(e.questions, lp):
                    if l is not None and q.target_probs() is not None:
                        recs.append({"probs": [math.exp(x) for x in l],
                                     "target": q.target_probs()})
            src = f"sampled examples from {out['sample']['data']}"
        sweep = rg.run_sweep(scorer, [int(x) for x in args.sweep_ns.split(",")],
                             args.sweep_reps, args.seed)
        out["option_count"] = {"buckets_source": src, "n_questions": len(recs),
                               "buckets": rg.option_count_buckets(recs),
                               "sweep": sweep["rows"], "sweep_reps": sweep["reps"],
                               "tested_up_to": sweep["tested_up_to"]}
        out["timing_s"]["option_count"] = round(time.time() - t, 1)
        _say_count(out["option_count"])
    return out


def _f(x, d: int = 4) -> str:
    return "  n/a " if x is None else f"{x:.{d}f}"


def _say(name: str, r: dict) -> None:
    print(f"\n== {name}")
    if name == "known posterior":
        print(f"  n {r['n']}   KL(true||model) {_f(r.get('kl'))}   TV {_f(r.get('tv'))}   "
              f"slope {_f(r.get('calib_slope'))} (intercept {_f(r.get('calib_intercept'))})   "
              f"ECE(soft) {_f(r.get('ece_soft'))}   argmax agree {_f(r.get('argmax_agree'))}")
        for f, m in r["per_family"].items():
            print(f"    {f:8} n {m['n']:4}  KL {_f(m.get('kl'))}  TV {_f(m.get('tv'))}  "
                  f"slope {_f(m.get('calib_slope'))}  ECE {_f(m.get('ece_soft'))}  "
                  f"agree {_f(m.get('argmax_agree'))}")
    elif name == "nonce":
        print(f"  {r['n_examples']} examples ({r['n_examples_skipped']} skipped as unsafe), "
              f"{r['n_variants']} variants, {r.get('n_questions', 0)} question pairs")
        rows = [("all", r)] + list(r["by_type"].items())
        for k, m in rows:
            if not m.get("n_questions"):
                continue
            print(f"    {k:10} n {m['n_questions']:4}  mean|dlogp| {_f(m['mean_abs_dlogp'])}  "
                  f"max|dlogp| {_f(m['max_abs_dlogp'])}  TV {_f(m['mean_tv'])}  "
                  f"flips {_f(m['flip_rate'])}")
    elif name == "permutation":
        mx = r["max_abs_dp"]
        print(f"  k {r['k']}  {r['n_questions']} questions  mean|dp| {_f(r['mean_abs_dp'])}  "
              f"mean max|dp| {_f(r['mean_max_abs_dp'])}  max|dp| raw {mx!r}  "
              f"flip rate {_f(r['flip_rate'])}  any-flip {_f(r['any_flip_rate'])}")


def _say_count(r: dict) -> None:
    print(f"\n== option count (buckets from {r['buckets_source']})")
    print(f"  {'N':>8} {'n':>7} {'acc':>7} {'nll':>7} {'ece':>7} {'max-p':>7}")
    for b in r["buckets"]:
        print(f"  {b['bucket']:>8} {b['n']:7,} {_f(b.get('acc'), 3):>7} "
              f"{_f(b.get('nll'), 3):>7} {_f(b.get('ece'), 3):>7} {_f(b.get('mean_maxp'), 3):>7}")
    print(f"  synthetic lookup sweep ({r['sweep_reps']} per N):")
    print(f"  {'N':>8} {'n':>4} {'tokens':>7} {'acc':>7} {'ece':>7} {'p(corr)':>8} {'chance':>7}")
    for s in r["sweep"]:
        if not s["fits"]:
            print(f"  {s['N']:>8}  does not fit the token budget")
            continue
        print(f"  {s['N']:>8} {s['n']:4} {_f(s.get('mean_tokens'), 0):>7} {_f(s.get('acc'), 3):>7} "
              f"{_f(s.get('ece'), 3):>7} {_f(s.get('mean_p_correct'), 3):>8} {s['chance']:7.3f}")
    print(f"  tested up to N = {r['tested_up_to']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--weights-dtype", choices=["fp32", "bf16", "fp16"], default=None,
                    help="default bf16 on cuda, fp32 on cpu")
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--corpus", type=Path, default=None,
                    help="corpus dir; --data defaults to its devreal.jsonl")
    ap.add_argument("--data", type=Path, default=None,
                    help="examples for the nonce and permutation tests")
    ap.add_argument("--dump", type=Path, default=None, help="eval dump for the option-count buckets")
    ap.add_argument("--tests", default=",".join(TESTS))
    ap.add_argument("--n-posterior", type=int, default=160)
    ap.add_argument("--n-nonce", type=int, default=80)
    ap.add_argument("--n-perm", type=int, default=40)
    ap.add_argument("--k-perm", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=768,
                    help="packed length cap for sampled real examples (CPU cost)")
    ap.add_argument("--per-task", type=int, default=2)
    ap.add_argument("--sweep-ns", default=",".join(map(str, rg.SWEEP_NS)))
    ap.add_argument("--sweep-reps", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}.get(
        args.weights_dtype, torch.bfloat16 if device.type == "cuda" else torch.float32)
    t0 = time.time()
    model = OptionScoringModel.load(args.ckpt, torch_dtype=dtype)
    model.to(device).eval()
    print(f"{args.ckpt}: device {device}, dtype {dtype}, T {float(model.temperature):.4f}")
    out = run(model, args)
    out["dtype"] = str(dtype)
    out["timing_s"]["total"] = round(time.time() - t0, 1)
    print(f"\ntotal {out['timing_s']['total']:.0f} s")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=2) + "\n")
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
