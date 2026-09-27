#!/usr/bin/env python
"""Evaluate a checkpoint on one file; optionally fit a temperature.

Prints a per-task table (n, acc, nll, brier, ece) with micro and macro averages, and
writes, next to each other:

    <out>                  one record per question (lod.evaluation.dumps), the input to
                           scripts/report.py, scripts/gates.py and the model card
    <out stem>-summary.json  overall metrics, the temperature used, selective prediction

`--out` defaults to <ckpt>/eval-<split>.jsonl. The checkpoint's own temperature is used
unless `--fit-temperature FILE` fits one on FILE (a dev split, never the test split);
`--save-temperature` then writes it into the checkpoint's config.

    scripts/evaluate.py --ckpt runs/<run>-release --data data/<corpus>/testreal.jsonl \\
        --attn flex --max-batch-tokens 98304 --oracle-temperature \\
        --out runs/<run>-release/report/eval-testreal.jsonl
"""

from __future__ import annotations

import os

os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")  # must precede `import torch`

import argparse
import json
from pathlib import Path

import torch
from tqdm.auto import tqdm

from lod.evaluation.dumps import question_key
from lod.evaluation.engine import collect_logits, per_question_records, per_task_metrics
from lod.model.calibration import (
    aurc,
    fit_temperature,
    fit_temperature_ensemble,
    fit_temperature_logn,
    head_correlation,
    head_disagreement,
    metrics,
    p_wrong_at_confidence,
    probabilities,
    temperature_per_question,
)
from lod.model.flexattn import BLOCK_SIZE as FLEX_BLOCK
from lod.model.packing import Packer
from lod.model.scorer import OptionScoringModel, log_mean_prob
from lod.paths import config_path
from lod.schema import read_jsonl
from lod.training.data import DecisionDataset, make_loader

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", required=True, help="jsonl file to evaluate")
    p.add_argument("--out", default=None, help="the per-question dump (default "
                                                "<ckpt>/eval-<split>.jsonl)")
    p.add_argument("--fit-temperature", default=None, metavar="FITFILE",
                   help="fit a temperature on this file and report before/after")
    p.add_argument("--temperature-logn", action="store_true",
                   help="fit T(N) = T * (N/2)^slope jointly instead of one scalar")
    p.add_argument("--save-temperature", action="store_true",
                   help="write the fitted temperature into the checkpoint's config")
    p.add_argument("--oracle-temperature", action="store_true",
                   help="also fit T on THIS file and report it (never saved): how far the "
                        "shipped temperature is from the best one for this split")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-batch-tokens", type=int, default=0,
                   help="cap padded tokens per batch (long states); 0 = batch-size only")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--weights-dtype", choices=list(DTYPES), default=None,
                   help="default bf16 on cuda, fp32 on cpu")
    p.add_argument("--attn", choices=["sdpa", "flex"], default="sdpa",
                   help="attention kernel path; identical semantics")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--n-bins", type=int, default=15)
    p.add_argument("--no-progress", action="store_true")
    return p.parse_args(argv)


def run_file(model, path: Path, args, device, mask_dtype) -> tuple:
    examples = read_jsonl(path)
    if args.limit:
        examples = examples[: args.limit]
    packer = Packer.for_model(model)
    ds = DecisionDataset(examples, packer, augment=False)
    flex_kw = {"pad_to_tile": FLEX_BLOCK, "dense_mask": False} if args.attn == "flex" else {}
    loader = make_loader(ds, packer.pad_id, mask_dtype, batch_size=args.batch_size,
                         num_workers=args.num_workers, max_tokens=args.max_batch_tokens,
                         **flex_kw)
    if not args.no_progress:
        loader = tqdm(loader, desc=path.name, unit="batch", dynamic_ncols=True, leave=False)
    return collect_logits(model, loader, device, return_heads=True, return_refs=True)


def at_temperature(logits, head_logits, valid, temperature: float, logn: float = 0.0):
    """The shipped distribution's logits at T(N).

    Single head: `logits / T(N)`. K heads: every head's logits are divided by the same
    T(N) and the mean of the per-head softmaxes is retaken, because p̄ is what ships.
    """
    if abs(temperature - 1.0) <= 1e-9 and logn == 0.0:
        return logits
    div = temperature_per_question(valid, temperature, logn)
    if head_logits is None:
        return logits / div
    return log_mean_prob(head_logits / div.unsqueeze(-1), valid)


def print_table(title: str, overall: dict, per_task: dict[str, dict], max_rows: int = 40) -> None:
    """Per-task rows plus both averages: micro over questions, macro over tasks."""
    print(f"\n{title}")
    print(f"{'task':<28} {'n':>7} {'acc':>8} {'nll':>8} {'brier':>8} {'ece':>8}")
    print("-" * 70)
    rows = sorted(per_task.items())
    for task, m in rows[:max_rows]:
        print(f"{task:<28} {m['n']:>7} {m['acc']:>8.4f} {m['nll']:>8.4f} "
              f"{m['brier']:>8.4f} {m['ece']:>8.4f}")
    if len(rows) > max_rows:
        print(f"... {len(rows) - max_rows} more tasks (see the per-question dump)")
    print("-" * 70)
    print(f"{'MICRO (over questions)':<28} {overall['n']:>7} {overall['acc']:>8.4f} "
          f"{overall['nll']:>8.4f} {overall['brier']:>8.4f} {overall['ece']:>8.4f}")
    if rows:
        def mac(k):
            return sum(m[k] for _, m in rows) / len(rows)
        print(f"{'MACRO (over tasks)':<28} {len(rows):>7} {mac('acc'):>8.4f} {mac('nll'):>8.4f} "
              f"{mac('brier'):>8.4f} {mac('ece'):>8.4f}")


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = DTYPES[args.weights_dtype] if args.weights_dtype else (
        torch.bfloat16 if device.type == "cuda" else torch.float32)
    ckpt = Path(args.ckpt)
    model = OptionScoringModel.load(ckpt, torch_dtype=dtype, attn_impl=args.attn)
    temperature = float(model.temperature)
    logn = float(model.temperature_logn)
    model.temperature.fill_(1.0)  # collect raw logits; the temperature is applied below
    model.temperature_logn = 0.0
    model.to(device).eval()

    data_path = Path(args.data)
    logits, valid, target, tasks, head_logits, refs = run_file(model, data_path, args,
                                                               device, dtype)
    if logits.shape[0] == 0:
        raise SystemExit(f"no labelled questions in {data_path}")
    n_heads = int(model.n_heads)
    if n_heads > 1:
        print(f"K-head ensemble: {n_heads} heads, predicting with p̄ = mean_k softmax(logits_k)")

    if args.fit_temperature:
        fit_path = Path(args.fit_temperature)
        if fit_path == data_path:
            f_logits, f_valid, f_target, f_heads = logits, valid, target, head_logits
        else:
            f_logits, f_valid, f_target, _, f_heads, _ = run_file(model, fit_path, args,
                                                                  device, dtype)
        logn = 0.0
        if args.temperature_logn and f_heads is None:
            temperature, logn = fit_temperature_logn(f_logits, f_valid, f_target)
            print(f"fitted T(N) = {temperature:.4f} * (N/2)^{logn:+.4f}")
        else:
            temperature = (fit_temperature_ensemble(f_heads, f_valid, f_target)
                           if f_heads is not None
                           else fit_temperature(f_logits, f_valid, f_target))
        print(f"fitted temperature T = {temperature:.4f} on {fit_path} "
              f"({f_logits.shape[0]} questions)")

    split = data_path.stem
    # the question behind each logits row, by ref: the packer drops questions that do not
    # fit, so a positional join would pair later rows with the wrong questions
    examples = read_jsonl(data_path)
    if args.limit:
        examples = examples[: args.limit]
    qmeta = [examples[i].questions[j] for i, j in refs]

    before = metrics(logits, valid, target, args.n_bins)
    print(f"\ncheckpoint: {ckpt}   data: {data_path}   device: {device}   dtype: {dtype}")
    print_table(f"{split}: before temperature (T = 1.0)", before,
                per_task_metrics(logits, valid, target, tasks, args.n_bins))
    final_logits, final = logits, before
    if abs(temperature - 1.0) > 1e-6 or logn != 0.0:
        final_logits = at_temperature(logits, head_logits, valid, temperature, logn)
        final = metrics(final_logits, valid, target, args.n_bins)
        print_table(f"{split}: after temperature (T = {temperature:.4f}, "
                    f"slope {logn:+.4f})", final,
                    per_task_metrics(final_logits, valid, target, tasks, args.n_bins))
        print(f"\noverall ECE {before['ece']:.4f} -> {final['ece']:.4f}   "
              f"NLL {before['nll']:.4f} -> {final['nll']:.4f}")

    summary = {"ckpt": str(ckpt), "data": str(data_path), "n_heads": n_heads,
               "temperature": temperature, "temperature_logn": logn,
               "metrics_T1": before, "metrics": final}

    if args.oracle_temperature:
        t_extra = fit_temperature(final_logits, valid, target)
        om = metrics(final_logits / t_extra, valid, target, args.n_bins)
        summary["oracle_temperature"] = temperature * t_extra
        print(f"\nORACLE temperature (fitted on {data_path} itself, NOT saved): "
              f"T_total = {temperature * t_extra:.4f}")
        print(f"  ECE {final['ece']:.4f} -> {om['ece']:.4f}   NLL {om['nll']:.4f}")

    # probability recovery, for questions whose target is a known probability
    ps = [(i, q.meta["p"]) for i, q in enumerate(qmeta) if q.meta and "p" in q.meta]
    if ps:
        idx = torch.tensor([i for i, _ in ps])
        truth = torch.tensor([float(v) for _, v in ps])
        p_yes = probabilities(final_logits[idx], valid[idx])[:, 1]
        err = p_yes - truth
        r = torch.corrcoef(torch.stack([p_yes, truth]))[0, 1].item()
        summary["probability_recovery"] = {"n": len(ps), "mae": float(err.abs().mean()),
                                           "bias": float(err.mean()), "r": r}
        print(f"\nprobability recovery on {len(ps)} questions carrying meta.p:")
        print(f"  MAE {err.abs().mean():.4f}   bias {err.mean():+.4f}   r {r:.4f}   "
              f"(MAE if it always said 0.5: {(0.5 - truth).abs().mean():.4f})")

    cw = p_wrong_at_confidence(final_logits, valid, target, 0.9)
    summary.update({"aurc": aurc(final_logits, valid, target), "confidently_wrong": cw})
    print(f"\nselective prediction on {split}: AURC {summary['aurc']:.4f} (lower is better)")
    print(f"  P(wrong | p >= 0.9) = {cw['p_wrong']:.4f} over {cw['n']} of "
          f"{final_logits.shape[0]} questions ({cw['share']:.1%} of the split)")
    if head_logits is not None:
        hl = head_logits / temperature
        hc = head_correlation(hl, valid)
        summary.update({"head_correlation": hc,
                        "head_disagreement": head_disagreement(hl, valid),
                        "per_head": [metrics(hl[:, :, k], valid, target, args.n_bins)
                                     for k in range(n_heads)]})
        print(f"  mean pairwise head correlation {hc['mean']:.4f} "
              f"(min {hc['min']:.4f}, max {hc['max']:.4f})")

    out_path = Path(args.out) if args.out else ckpt / f"eval-{split}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records = per_question_records(final_logits, valid, target, tasks)
    with open(out_path, "w") as f:
        for rec, q, (ei, qi) in zip(records, qmeta, refs):
            rec["qid"] = q.id
            rec["ref"] = [int(ei), int(qi)]
            rec["key"] = question_key(examples[ei].task, examples[ei].state, q)
            if q.meta:
                rec["meta"] = q.meta
            f.write(json.dumps(rec) + "\n")
    summary_path = out_path.with_name(out_path.stem + "-summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nwrote {len(records)} per-question records to {out_path} and {summary_path.name}")

    if args.save_temperature:
        cfg_path = config_path(ckpt)
        cfg = json.loads(cfg_path.read_text())
        cfg["temperature"], cfg["temperature_logn"] = temperature, logn
        cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")
        print(f"saved temperature {temperature:.4f} (slope {logn:+.4f}) to {cfg_path}")


if __name__ == "__main__":
    main()
