#!/usr/bin/env python
"""Train an OptionScoringModel on typed decisions.

Prefill-only supervised training on a proper scoring rule: no generation, no sampling,
just the log (or Brier) score of each question's distribution over its own options.

Checkpoints are written atomically; `--resume auto` continues from the newest `step-*`
with the optimiser, scheduler, RNG and data position restored, so a preempted run
resumes exactly. `best/` is chosen on dev-OOD NLL (`--devood`, default
<data>/devood.jsonl), `last/` is the final step. Every event lands in `<out>/log.jsonl`.

    scripts/train.py --data data/<corpus> --out runs/<run> --backbone Qwen/Qwen3-0.6B-Base \
        --lora-r 32 --lora-alpha 64 ...        (see scripts/recipes/ for the release set)
"""

from __future__ import annotations

import os

os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")  # must precede `import torch`

import argparse
import json
import math
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from lod.evaluation.engine import evaluate
from lod.model.flexattn import BLOCK_SIZE as FLEX_BLOCK
from lod.model.packing import Packer
from lod.model.scorer import (
    OptionScoringModel,
    decision_loss,
    forward_batch,
    grouped_head_logits,
    grouped_logits,
    log_mean_prob,
    shard_select,
)
from lod.schema import read_jsonl
from lod.training.data import (
    DecisionDataset,
    EpochBatchSampler,
    SolvedTracker,
    length_key,
    make_loader,
    token_lengths,
)

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
# args a resume may legitimately differ in
RESUME_MUTABLE = {"resume", "out", "max_steps", "epochs", "eval_every", "save_every",
                  "log_every", "num_workers"}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backbone", default="HuggingFaceTB/SmolLM2-135M")
    p.add_argument("--data", required=True,
                   help="corpus directory with train.jsonl, val.jsonl and devood.jsonl")
    p.add_argument("--out", required=True, help="run directory")
    p.add_argument("--lora-r", type=int, default=0)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--weights-dtype", choices=list(DTYPES), default="fp32")
    p.add_argument("--amp", choices=["none", "fp16", "bf16"], default="fp16")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=None, help="default 2e-5 full FT, 2e-4 LoRA")
    p.add_argument("--lr-head", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--max-state-tokens", type=int, default=512)
    p.add_argument("--max-tokens", type=int, default=768)
    p.add_argument("--grad-ckpt", action="store_true",
                   help="gradient checkpointing: recompute the forward on the backward")
    p.add_argument("--compile", dest="compile", action="store_true",
                   help="torch.compile the backbone (dynamic shapes: length bucketing "
                        "varies the sequence length per batch)")
    p.add_argument("--loss", choices=["ce", "brier"], default="ce")
    p.add_argument("--mask", choices=["block", "full-causal"], default="block",
                   help="'full-causal' is an ablation in which questions see each other")
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--eval-limit", type=int, default=2000)
    p.add_argument("--devood", default=None, metavar="FILE",
                   help="dev-OOD file: families held out of train. Default "
                        "<data>/devood.jsonl if it exists. Drives best/ selection and "
                        "early stopping")
    p.add_argument("--patience", type=int, default=3,
                   help="evals without dev-OOD NLL improvement before stopping; 0 disables")
    p.add_argument("--min-delta", type=float, default=0.0,
                   help="how much dev-OOD NLL must improve to reset the patience counter; "
                        "best/ still tracks the lowest NLL seen, whatever the delta")
    p.add_argument("--solved-downweight", type=float, default=0.2,
                   help="loss factor for tasks above 0.99 running train acc; 1.0 disables")
    p.add_argument("--head-weight-decay", type=float, default=0.01)
    p.add_argument("--heads", type=int, default=1,
                   help="K pointer heads over the one shared backbone pass. Each\n"
                        "training example is routed to exactly one head's loss by a\n"
                        "stable hash, so the heads see disjoint shards and different\n"
                        "option orders; the backbone still gets gradient from all of\n"
                        "them. Prediction is the mean of the per-head softmaxes.\n"
                        "K=1 is the single-head model, unchanged.")
    p.add_argument("--head-seed", type=int, default=0,
                   help="per-head init seed; head k is initialised from head_seed + k")
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--keep-last", type=int, default=2)
    p.add_argument("--resume", default=None, help="'auto' or a checkpoint dir")
    p.add_argument("--train-limit", type=int, default=None)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--attn", choices=["sdpa", "flex"], default="sdpa",
                   help="attention kernel path. 'flex' expresses the block mask as a "
                        "predicate instead of a dense [B,1,T,T] tensor; identical "
                        "semantics (tests/test_flexattn.py), different speed and memory")
    p.add_argument("--bucket", type=int, default=0, metavar="N",
                   help="length-bucketed batching: sort each window of N*batch_size examples "
                        "by length before cutting batches (0 = off); cuts padding")
    p.add_argument("--criteria-augment", action="store_true",
                   help="criteria augmentations on train (lod.training.data."
                        "augment_criteria): opaque keys, withheld correct options, stripped "
                        "descriptions, decoy sentinels, churned option sets")
    p.add_argument("--no-progress", action="store_true", help="suppress the tqdm progress bar")
    # option layout and the comparison layer
    p.add_argument("--option-mode", choices=["sequential", "independent"], default="sequential",
                   help="independent: each option attends to the state, its question's head "
                        "and itself only, from a shared start position (order-invariant)")
    p.add_argument("--mixer-dim", type=int, default=0,
                   help="width of the comparison layer over option vectors; 0 = none")
    p.add_argument("--mixer-heads", type=int, default=4)
    p.add_argument("--mixer-topk", type=int, default=64,
                   help="the comparison layer mixes the top-k options by stage-1 score")
    p.add_argument("--stage1-weight", type=float, default=0.25,
                   help="weight of the auxiliary loss on the independent stage-1 scores, "
                        "which shortlist and shard; only with a comparison layer")
    p.add_argument("--max-batch-tokens", type=int, default=0,
                   help="cap each batch's padded tokens (members x longest); with long states "
                        "a batch shrinks to fit instead of running out of memory. Needs --bucket")
    p.add_argument("--distill-alpha", type=float, default=0.0,
                   help="train on (1-a)*gold + a*meta.teacher where a question carries a "
                        "teacher distribution")
    return p


def parse_args(argv=None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if args.lr is None:
        args.lr = 2e-4 if args.lora_r > 0 else 2e-5
    return args


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


class Logger:
    """Every event goes to log.jsonl and, as the same line, to stdout."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, **event) -> None:
        event.setdefault("time", now())
        line = json.dumps(event)
        with open(self.path, "a") as f:
            f.write(line + "\n")
        tqdm.write(line, file=sys.stdout)  # keeps the progress bar intact
        sys.stdout.flush()


def set_rng_state(state: dict) -> None:
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    random.setstate(state["random"])
    np.random.set_state(state["numpy"])


def get_rng_state() -> dict:
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "random": random.getstate(),
        "numpy": np.random.get_state(),
    }


def lr_lambda_factory(warmup: int, total: int):
    """Linear warmup, then cosine to 10 % of peak."""

    def fn(step: int) -> float:
        if warmup > 0 and step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total - warmup)
        progress = min(max(progress, 0.0), 1.0)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    return fn


def step_dirs(out: Path) -> list[Path]:
    return sorted([d for d in out.glob("step-*") if d.is_dir() and not d.name.endswith(".tmp")])


def clear_stale_tmp(out: Path) -> None:
    """A save interrupted mid-write leaves step-NNNNNN.tmp. step_dirs() already
    ignores it, so it is never mistaken for a checkpoint; drop them at startup."""
    for d in out.glob("step-*.tmp"):
        if d.is_dir():
            shutil.rmtree(d, ignore_errors=True)
            print(f"removed stale partial checkpoint {d}", flush=True)


def save_checkpoint(out: Path, name: str, model: OptionScoringModel, trainer: dict | None,
                    log=None) -> Path:
    """Atomic: fill a .tmp directory, then rename it into place."""
    final = out / name
    tmp = out / (name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)
    model.save(tmp)
    if trainer is not None:
        torch.save(trainer, tmp / "trainer.pt")
    shutil.rmtree(final, ignore_errors=True)
    tmp.rename(final)
    if log is not None and trainer is not None:
        log(kind="save", step=trainer["step"], path=str(final))
    return final


def prune_checkpoints(out: Path, keep_last: int) -> None:
    for d in step_dirs(out)[:-keep_last] if keep_last > 0 else step_dirs(out):
        shutil.rmtree(d, ignore_errors=True)


def check_resume_args(saved: dict, current: dict, defaults: dict | None = None) -> None:
    """Flags added to the parser after a checkpoint was written are fine as long
    as they still hold their default; anything else must match."""
    defaults = defaults or {}
    diffs = []
    for k in set(saved) | set(current):
        if k in RESUME_MUTABLE:
            continue
        if k not in saved and current.get(k) == defaults.get(k):
            continue
        if saved.get(k) != current.get(k):
            diffs.append(k)
    if diffs:
        detail = ", ".join(f"{k}: {saved.get(k)!r} -> {current.get(k)!r}" for k in sorted(diffs))
        raise SystemExit(f"--resume: incompatible args ({detail})")


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = Logger(out / "log.jsonl")
    clear_stale_tmp(out)

    if args.lora_r == 0 and args.weights_dtype == "fp16":
        raise SystemExit("full fine-tuning with fp16 weights is unstable; use --weights-dtype fp32 "
                         "(fp16 weights are allowed only with LoRA, where the base is frozen)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = None if args.amp == "none" else DTYPES[args.amp]
    if amp_dtype == torch.bfloat16 and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise SystemExit("--amp bf16 is not supported on this GPU (compute capability < 8.0)")
    weights_dtype = DTYPES[args.weights_dtype]
    # the mask is added to attention scores, so it must match what attention runs in
    mask_dtype = amp_dtype if amp_dtype is not None else weights_dtype
    # under flex the dense mask is never built: the model rebuilds the rule on-device
    flex_kw = ({"pad_to_tile": FLEX_BLOCK, "dense_mask": False} if args.attn == "flex"
               else {})

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    resume_dir: Path | None = None
    if args.resume:
        if args.resume == "auto":
            candidates = step_dirs(out)
            resume_dir = candidates[-1] if candidates else None
            if resume_dir is None:
                print(f"--resume auto: no step-* checkpoint under {out}, starting fresh")
        else:
            resume_dir = Path(args.resume)
            if not resume_dir.exists():
                raise SystemExit(f"--resume: {resume_dir} does not exist")

    # ---- model ----------------------------------------------------------------
    if resume_dir is not None:
        model = OptionScoringModel.load(resume_dir, torch_dtype=weights_dtype,
                                    merge_lora=False, attn_impl=args.attn)
    else:
        model = OptionScoringModel(
            args.backbone,
            option_mode=args.option_mode,
            mixer_dim=args.mixer_dim,
            mixer_heads=args.mixer_heads,
            mixer_topk=args.mixer_topk,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            torch_dtype=weights_dtype,
            max_state_tokens=args.max_state_tokens,
            max_total_tokens=args.max_tokens,
            mask_mode=args.mask,
            attn_impl=args.attn,
            n_heads=args.heads,
            head_seed=args.head_seed,
        )
    model.to(device)
    if model.n_heads > 1:
        print(f"K-head ensemble: {model.n_heads} heads, disjoint content-hashed shards, "
              f"per-head option order, init seeds {args.head_seed}+k", flush=True)
    model.head.float()
    if args.grad_ckpt:
        model.enable_gradient_checkpointing()
    if args.compile:
        # the backbone only. The head is three ops on a [N, H] slice and compiling it
        # buys nothing; `create_block_mask` is compiled separately in lod/model/flexattn.py.
        # dynamic=True because length bucketing gives a different T per batch and
        # static compilation would recompile on each one.
        model.backbone.forward = torch.compile(model.backbone.forward, dynamic=True)

    # ---- data -----------------------------------------------------------------
    packer = Packer(model.tokenizer, args.max_state_tokens, args.max_tokens, args.mask,
                    option_mode=model.option_mode)
    train_examples = read_jsonl(Path(args.data) / "train.jsonl")
    if args.train_limit:
        train_examples = train_examples[: args.train_limit]
    val_examples = read_jsonl(Path(args.data) / "val.jsonl")[: args.eval_limit]
    devood_path = Path(args.devood) if args.devood else Path(args.data) / "devood.jsonl"
    devood_examples = read_jsonl(devood_path) if devood_path.exists() else []
    train_lengths = ((token_lengths(train_examples, model.tokenizer) if args.max_batch_tokens > 0
                      else length_key(train_examples)) if args.bucket > 1 else None)
    train_ds = DecisionDataset(train_examples, packer, augment=True, seed=args.seed,
                               criteria_augment=args.criteria_augment,
                               n_heads=model.n_heads, distill_alpha=args.distill_alpha)
    val_ds = DecisionDataset(val_examples, packer, augment=False, seed=args.seed)
    val_loader = make_loader(val_ds, packer.pad_id, mask_dtype, batch_size=args.batch_size,
                             num_workers=args.num_workers, max_tokens=args.max_batch_tokens, **flex_kw)
    devood_loader = None
    if devood_examples:
        devood_ds = DecisionDataset(devood_examples, packer, augment=False, seed=args.seed)
        devood_loader = make_loader(devood_ds, packer.pad_id, mask_dtype,
                                    batch_size=args.batch_size,
                                    num_workers=args.num_workers, max_tokens=args.max_batch_tokens, **flex_kw)
        print(f"dev-OOD: {len(devood_ds)} examples from {devood_path} "
              f"-> best/ selection, early stopping, temperature", flush=True)
    else:
        print(f"WARNING: no dev-OOD file at {devood_path}. best/ will be selected on "
              f"in-distribution val NLL, which can pick a checkpoint that transfers worst "
              f"to unseen tasks.", flush=True)

    batches_per_epoch = math.ceil(len(train_ds) / args.batch_size)
    steps_per_epoch = max(1, batches_per_epoch // args.grad_accum)
    total_steps = args.max_steps if args.max_steps > 0 else max(1, int(args.epochs * steps_per_epoch))

    # ---- optimiser ------------------------------------------------------------
    body = [p for p in model.backbone.parameters() if p.requires_grad]
    head = list(model.head.parameters()) + \
        (list(model.mixer.parameters()) if model.mixer is not None else [])
    optimizer = torch.optim.AdamW([
        {"params": body, "lr": args.lr, "weight_decay": args.weight_decay},
        {"params": head, "lr": args.lr_head, "weight_decay": args.head_weight_decay},
    ])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda_factory(args.warmup, total_steps))
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16 and device.type == "cuda"))

    step = 0
    micro_step = 0
    start_epoch = 0
    batches_consumed = 0
    best_val_nll = float("inf")   # dev-OOD NLL when devood.jsonl exists, else val NLL
    bad_evals = 0
    last_eval_step = -1
    elapsed_base = 0.0

    if resume_dir is not None:
        state = torch.load(resume_dir / "trainer.pt", map_location="cpu", weights_only=False)
        parser = build_parser()
        check_resume_args(state["args"], vars(args),
                          {k: parser.get_default(k) for k in vars(args)})
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        set_rng_state(state["rng"])
        step = state["step"]
        micro_step = state["micro_step"]
        start_epoch = state["epoch"]
        batches_consumed = state["batches_consumed_in_epoch"]
        best_val_nll = state["best_val_nll"]
        bad_evals = state.get("bad_evals", 0)
        elapsed_base = state.get("elapsed_s", 0.0)
        log(kind="resume", **{"from": str(resume_dir)}, step=step)

    (out / "args.json").write_text(json.dumps({
        "args": vars(args), "git_commit": git_commit(), "start_time": now(),
        "device": str(device), "total_steps": total_steps,
        "n_train": len(train_ds), "n_val": len(val_ds),
        "trainable_params": sum(p.numel() for p in body + head),
    }, indent=2) + "\n")

    n_trainable = sum(p.numel() for p in body + head)
    print(f"device={device} backbone={model.backbone_name} trainable={n_trainable/1e6:.1f}M "
          f"train={len(train_ds)} val={len(val_ds)} total_steps={total_steps} "
          f"steps/epoch={steps_per_epoch} amp={args.amp} weights={args.weights_dtype}", flush=True)

    # solved-task down-weighting (see lod.training.data.SolvedTracker for the why)
    tracker = SolvedTracker(factor=args.solved_downweight)

    run_start = time.time()

    def elapsed() -> float:
        return elapsed_base + (time.time() - run_start)

    def trainer_state(epoch: int, consumed: int) -> dict:
        return {
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
            "micro_step": micro_step,
            "epoch": epoch,
            "batches_consumed_in_epoch": consumed,
            "best_val_nll": best_val_nll,
            "bad_evals": bad_evals,
            "rng": get_rng_state(),
            "args": vars(args),
            "elapsed_s": elapsed(),
        }

    def _eval_split(name: str, base_loader, epoch: int) -> dict:
        loader = base_loader if args.no_progress else tqdm(
            base_loader, desc=f"{name} @ step {step}", leave=False, unit="batch",
            dynamic_ncols=True)
        res = evaluate(model, loader, device, amp_dtype)
        per_task = {k: {kk: round(vv, 6) if isinstance(vv, float) else vv for kk, vv in v.items()}
                    for k, v in res["per_task"].items()}
        log(kind="eval", step=step, split=name, acc=round(res["acc"], 6), nll=round(res["nll"], 6),
            brier=round(res["brier"], 6), ece=round(res["ece"], 6), n=res["n"],
            per_task=per_task, elapsed_s=round(elapsed(), 1))
        return res

    def run_eval(epoch: int) -> dict:
        """-> the metrics best/ and early stopping are judged on."""
        nonlocal last_eval_step
        last_eval_step = step
        res_val = _eval_split("val", val_loader, epoch)
        res_dev = _eval_split("devood", devood_loader, epoch) if devood_loader is not None else None
        if tracker.solved:
            log(kind="solved", step=step, tasks=sorted(tracker.solved), factor=tracker.factor)
        return res_dev if res_dev is not None else res_val

    bar = tqdm(total=total_steps, initial=step, unit="step", dynamic_ncols=True,
               disable=args.no_progress, desc=out.name, smoothing=0.05)

    stop = False
    # the running loss and the solved-tracker's correctness stay on the GPU and are
    # drained once per --log-every window: reading them per micro-step would synchronise
    window_loss_t = torch.zeros((), device=device)
    pending: list[tuple[list[str], torch.Tensor]] = []
    window_loss, window_n, window_tokens = 0.0, 0, 0
    window_start = time.time()
    interrupted = False
    epoch = start_epoch

    try:
        while step < total_steps:
            train_ds.set_epoch(epoch)
            sampler = EpochBatchSampler(len(train_ds), args.batch_size, args.seed, epoch,
                                        skip_batches=batches_consumed,
                                        lengths=train_lengths, bucket=args.bucket,
                                        max_tokens=args.max_batch_tokens)
            loader = make_loader(train_ds, packer.pad_id, mask_dtype, batch_sampler=sampler,
                                 num_workers=args.num_workers, **flex_kw)
            model.train()
            for batch in loader:
                batches_consumed += 1
                if batch is None:
                    continue
                aux = model.mixer is not None and args.stage1_weight > 0
                with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                    enabled=amp_dtype is not None):
                    out_b = forward_batch(model, batch, device, return_parts=aux)
                logits = out_b["logits"] if aux else out_b
                q_valid = batch["valid"].to(device)
                q_target = batch["target"].to(device)
                q_has = batch["has_target"].to(device)
                if model.n_heads > 1:
                    head_grouped = grouped_head_logits(
                        logits.float(),
                        batch["option_group"].to(device),
                        batch["option_slot"].to(device),
                        q_valid,
                    )
                    # each example's loss is read off its own head only; the backbone
                    # sees every example, routed through whichever head owns it
                    loss_logits = shard_select(head_grouped, batch["q_shard"].to(device))
                    # what the ensemble would actually answer, for the solved-tracker
                    grouped = log_mean_prob(head_grouped, q_valid)
                else:
                    grouped = grouped_logits(
                        logits.float(),
                        batch["option_group"].to(device),
                        batch["option_slot"].to(device),
                        q_valid,
                    )
                    loss_logits = grouped
                q_weight = None
                if tracker.enabled and tracker.solved:
                    q_weight = tracker.weight_tensor(batch["q_task"], device)
                loss = decision_loss(loss_logits, q_valid, q_target, args.loss, q_has, q_weight)
                if aux:
                    # the independent scores must stay a good ranking on their own: they
                    # pick the shortlist the comparison layer sees, and past the token
                    # budget they are the only thing a shard can compute
                    s1 = grouped_logits(out_b["stage1"].float(),
                                        batch["option_group"].to(device),
                                        batch["option_slot"].to(device), q_valid)
                    loss = loss + args.stage1_weight * decision_loss(
                        s1, q_valid, q_target, args.loss, q_has, q_weight)
                if tracker.enabled:
                    with torch.no_grad():
                        ok = (grouped.detach().argmax(dim=-1) == q_target.argmax(dim=-1))
                        # kept on the GPU: a .cpu() here would synchronise every
                        # micro-step. Drained in one transfer per --log-every window.
                        pending.append((batch["q_task"], torch.stack([ok, q_has])))
                scaler.scale(loss / args.grad_accum).backward()

                window_loss_t += loss.detach()
                window_n += 1
                window_tokens += batch["n_tokens"]
                micro_step += 1

                if micro_step % args.grad_accum:
                    continue

                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(body + head, 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                step += 1
                bar.update(1)
                if not args.no_progress:
                    # reading the running loss is a synchronisation; a run with the bar
                    # off never pays it
                    bar.set_postfix(loss=f"{float(window_loss_t) / max(1, window_n):.3f}",
                                    lr=f"{scheduler.get_last_lr()[0]:.2e}",
                                    mem=f"{torch.cuda.max_memory_allocated() / 1e9:.1f}G"
                                    if device.type == "cuda" else "cpu",
                                    refresh=False)

                if step % args.log_every == 0 or step == total_steps:
                    dt = max(time.time() - window_start, 1e-6)
                    # the one synchronisation per logging window, not one per micro-step
                    window_loss = float(window_loss_t)
                    if pending:
                        flat = torch.cat([f for _, f in pending], dim=1).cpu()
                        at = 0
                        for tasks_b, f in pending:
                            n = f.shape[1]
                            ok_b = flat[0, at:at + n].tolist()
                            has_b = flat[1, at:at + n].tolist()
                            at += n
                            tracker.update([t for t, k in zip(tasks_b, has_b) if k],
                                           [bool(x) for x, k in zip(ok_b, has_b) if k])
                        pending.clear()
                    log(kind="train", step=step,
                        epoch=round(epoch + batches_consumed / max(1, sampler.n_batches_total), 4),
                        loss=round(window_loss / max(1, window_n), 6),
                        lr=scheduler.get_last_lr()[0],
                        grad_norm=round(float(grad_norm), 4),
                        tokens_per_s=round(window_tokens / dt, 1),
                        examples_seen=micro_step * args.batch_size,
                        max_mem_gb=round(torch.cuda.max_memory_allocated() / 1e9, 3)
                        if device.type == "cuda" else 0.0,
                        elapsed_s=round(elapsed(), 1))
                    window_loss_t = torch.zeros((), device=device)
                    window_loss, window_n, window_tokens = 0.0, 0, 0
                    window_start = time.time()

                if args.eval_every > 0 and step % args.eval_every == 0 and step < total_steps:
                    res = run_eval(epoch)
                    # two thresholds, deliberately: `best/` keeps the lowest NLL seen at
                    # all, while patience only resets on an improvement worth waiting for
                    improved = res["nll"] < best_val_nll - args.min_delta
                    if res["nll"] < best_val_nll:
                        best_val_nll = res["nll"]
                        save_checkpoint(out, "best", model, None)
                    if improved:
                        bad_evals = 0
                    else:
                        bad_evals += 1
                        if args.patience > 0 and bad_evals >= args.patience:
                            log(kind="earlystop", step=step, patience=args.patience,
                                best_nll=round(best_val_nll, 6),
                                selection="devood" if devood_loader is not None else "val")
                            stop = True
                            break
                    model.train()
                    window_start = time.time()

                if args.save_every > 0 and step % args.save_every == 0 and step < total_steps:
                    save_checkpoint(out, f"step-{step:06d}", model,
                                    trainer_state(epoch, batches_consumed), log)
                    prune_checkpoints(out, args.keep_last)
                    window_start = time.time()

                if step >= total_steps:
                    stop = True
                    break
            del loader
            if stop:
                break
            epoch += 1
            batches_consumed = 0
    except KeyboardInterrupt:
        bar.close()
        interrupted = True
        print("\ninterrupted: saving a checkpoint before exiting", flush=True)
        save_checkpoint(out, f"step-{step:06d}", model, trainer_state(epoch, batches_consumed), log)
        prune_checkpoints(out, args.keep_last)

    bar.close()
    if not interrupted:
        if step != last_eval_step:  # early stopping already evaluated this step
            res = run_eval(epoch)
            if res["nll"] < best_val_nll:
                best_val_nll = res["nll"]
                save_checkpoint(out, "best", model, None)
        save_checkpoint(out, "last", model, None)
        save_checkpoint(out, f"step-{step:06d}", model, trainer_state(epoch, batches_consumed), log)
        prune_checkpoints(out, args.keep_last)
        sel = "dev-OOD" if devood_loader is not None else "val"
        print(f"done: {step} steps, best {sel} NLL {best_val_nll:.4f}, "
              f"{elapsed():.0f}s, peak {torch.cuda.max_memory_allocated()/1e9:.2f} GB"
              if device.type == "cuda" else f"done: {step} steps", flush=True)

    if interrupted:
        raise SystemExit(130)


if __name__ == "__main__":
    main()
