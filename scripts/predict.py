#!/usr/bin/env python
"""Score one request from the command line: one prefill, a calibrated answer per question.

    predict.py --ckpt runs/<run>-release --state "..." --q "Which team? | billing | tech"
    predict.py --ckpt runs/<run>-release --json request.json [--repeat 20]

`--q` builds a `choice` question from "question | option | option ...". `--json` reads a
request in the API wire format (`{"state": ..., "questions": {...}}`, see
`lod.serving.api`) and prints the wire response. `--repeat` times the request.
"""

from __future__ import annotations

import os

os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")  # must precede `import torch`

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from lod.model.scorer import OptionScoringModel
from lod.serving import api
from lod.serving.scoring import answer


def parse_q(spec: str) -> dict:
    parts = [s.strip() for s in spec.split("|")]
    if len(parts) < 3:
        raise SystemExit(f"--q needs a question and at least 2 options: {spec!r}")
    return {"type": "choice", "instructions": parts[0],
            "criteria": {o: None for o in parts[1:]}}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--state", default=None)
    p.add_argument("--state-file", type=Path, default=None)
    p.add_argument("--q", action="append", default=[], metavar='"Question | opt1 | opt2"')
    p.add_argument("--json", type=Path, default=None, help="a request in the wire format")
    p.add_argument("--state-overflow", choices=api.STATE_OVERFLOW, default="refuse")
    p.add_argument("--repeat", type=int, default=1, help="timed passes; reports p50/p95")
    p.add_argument("--weights-dtype", choices=["fp32", "bf16", "fp16"], default=None,
                   help="default bf16 on cuda, fp32 on cpu")
    args = p.parse_args()

    if args.json:
        body = json.loads(args.json.read_text())
    else:
        state = (args.state_file.read_text() if args.state_file else args.state)
        if state is None:
            raise SystemExit("pass --state, --state-file or --json")
        if not args.q:
            raise SystemExit("no questions given (--q)")
        body = {"state": state, "questions": {f"q{i}": parse_q(s) for i, s in enumerate(args.q)}}
    body.setdefault("state_overflow", args.state_overflow)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}.get(
        args.weights_dtype, torch.bfloat16 if device.type == "cuda" else torch.float32)
    model = OptionScoringModel.load(args.ckpt, torch_dtype=dtype).to(device).eval()

    timings = []
    for _ in range(max(1, args.repeat)):
        t0 = time.perf_counter()
        try:
            resp = answer(model, body, device)
        except api.ValidationError as e:
            raise SystemExit(f"invalid request: {e.field}: {e.message}") from None
        if device.type == "cuda":
            torch.cuda.synchronize()
        timings.append((time.perf_counter() - t0) * 1000.0)

    print(json.dumps(resp, indent=2))
    ordered = sorted(timings)
    print(f"\nlatency {timings[-1]:.1f} ms on {device} ({dtype}), model load excluded"
          + (f"; over {len(timings)} runs p50 {statistics.median(timings):.1f} ms, "
             f"p95 {ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]:.1f} ms"
             if len(timings) > 1 else ""))


if __name__ == "__main__":
    main()
