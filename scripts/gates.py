#!/usr/bin/env python
"""Release gates.

    gates.py probes  --ckpt C --json R/probes.json   score the generated probe sets
                                                      (needs the model; GPU if available)
    gates.py summary --ckpt C --report R [--corpus D] > R/gates.json
                                                      every gate, from files on disk

`summary` reads the report directory the release recipe writes (`eval-testreal.jsonl`,
`eval-testreal.txt`, `independence.txt`, `probes.json`) plus the checkpoint's config and
`confidence_summary.json`; see `lod.evaluation.gates` for each gate and its bar. It exits
non-zero when a blocking gate fails.
"""

from __future__ import annotations

import os

os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")  # must precede `import torch`

import argparse
import json
import sys
from pathlib import Path


def probes(args) -> None:
    import torch

    from lod.evaluation.probes import run_all
    from lod.model.scorer import OptionScoringModel

    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = OptionScoringModel.load(args.ckpt, torch_dtype=dtype).to(device).eval()
    out = {"ckpt": str(args.ckpt), **run_all(model, device, args.n_entities, args.per_family)}
    for key in ("unseen_entity_abstention", "rule_transfer", "sentinel_wordings",
                "reefer_probe"):
        g = out[key]
        print(f"{key}: {'PASS' if g['pass'] else 'FAIL'}")
        for k, v in g.items():
            if k not in ("pass", "detail", "note", "acc_per_family", "forms"):
                print(f"    {k:28} {v}")
    for form, qs in out["card_probes"].items():
        print(f"\ncard probes, {form.replace('_', ' ')}:")
        for qid, r in qs.items():
            dist = "  ".join(f"{o}={p:.3f}" for o, p in zip(r["options"], r["probs"]))
            print(f"  {qid:16} -> {r['argmax']:24} conf {r['confidence']:.3f}   {dist}")
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {args.json}")


def summary(args) -> None:
    from lod.evaluation.gates import compute

    res = compute(args.report, args.ckpt, args.corpus, args.probes)
    print(json.dumps(res, indent=2))
    if res["blocking_failed"]:
        print(f"blocking gates failed: {', '.join(res['blocking_failed'])}", file=sys.stderr)
        raise SystemExit(1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("probes", help="score the generated probe sets")
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--json", type=Path, required=True)
    p.add_argument("--device", default=None)
    p.add_argument("--n-entities", type=int, default=600)
    p.add_argument("--per-family", type=int, default=150)
    s = sub.add_parser("summary", help="every gate, as JSON on stdout")
    s.add_argument("--ckpt", type=Path, required=True)
    s.add_argument("--report", type=Path, required=True)
    s.add_argument("--corpus", type=Path, default=None,
                   help="the corpus scored; its meta.json says which tasks are real")
    s.add_argument("--probes", type=Path, default=None,
                   help="probe results (default <report>/probes.json)")
    args = ap.parse_args()
    if args.cmd == "probes":
        if args.device is None:
            import torch
            args.device = "cuda" if torch.cuda.is_available() else "cpu"
        probes(args)
    else:
        summary(args)


if __name__ == "__main__":
    main()
