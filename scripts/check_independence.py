#!/usr/bin/env python
"""The independence gate: a question's logits must not move when other questions are
added, removed or reordered around it, or when other examples share its batch.

Run it on a trained checkpoint (the release gate reads its last line), on a new backbone
before adopting it (`--backbone`), or with `--mask full-causal` to confirm the ablation
does break it. fp32 on CPU is the gate; `--attn flex` checks the FlexAttention path.
"""

from __future__ import annotations

import os

os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")  # must precede `import torch`

import argparse

import torch

from lod.model.flexattn import BLOCK_SIZE as FLEX_BLOCK
from lod.model.packing import Packer, collate
from lod.model.scorer import OptionScoringModel, forward_batch, grouped_logits
from lod.schema import Example, Question

STATE = ("Ticket #4412 from a customer: my card was charged twice for the same order "
         "and nobody has replied for three days. This is unacceptable.")
Q_A = Question("team", "Which team should handle this ticket?",
               ["billing", "technical", "account", "shipping"])
Q_B = Question("angry", "Is the customer angry?", ["no", "yes"])
Q_C = Question("priority", "What priority should this ticket get?", ["low", "medium", "high"])
OTHER = Example(task="other", state="A fair 6-sided die is rolled.",
                questions=[Question("gt3", "Does it show a number greater than 3?", ["no", "yes"])])


def q_a_logits(model, packer, examples, row, device):
    flex = model.attn_impl == "flex"
    batch = collate([packer.pack(e) for e in examples], packer.pad_id, torch.float32,
                    pad_to_tile=FLEX_BLOCK if flex else 1, dense_mask=not flex)
    with torch.no_grad():
        logits = forward_batch(model, batch, device)
    return grouped_logits(logits.float(), batch["option_group"].to(device),
                          batch["option_slot"].to(device), batch["valid"].to(device))[row].cpu()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default=None, help="a trained checkpoint directory")
    p.add_argument("--backbone", default=None, help="a fresh HF backbone instead")
    p.add_argument("--mask", choices=["block", "full-causal"], default=None,
                   help="override the checkpoint's mask mode")
    p.add_argument("--attn", choices=["sdpa", "flex"], default="sdpa",
                   help="which attention path to gate; a new kernel path must pass this "
                        "before it is used for anything")
    p.add_argument("--threshold", type=float, default=1e-4)
    p.add_argument("--device", default="cpu", help="cpu keeps it in fp32, which is the gate")
    args = p.parse_args()
    if not (args.ckpt or args.backbone):
        raise SystemExit("pass --ckpt or --backbone")

    device = torch.device(args.device)
    if args.ckpt:
        model = OptionScoringModel.load(args.ckpt, torch_dtype=torch.float32, attn_impl=args.attn)
        label = args.ckpt
    else:
        model = OptionScoringModel(args.backbone, torch_dtype=torch.float32, attn_impl=args.attn)
        label = args.backbone
    model.to(device).eval()
    mask_mode = args.mask or model.mask_mode
    packer = Packer(model.tokenizer, model.max_state_tokens, model.max_total_tokens, mask_mode,
                    option_mode=getattr(model, "option_mode", "sequential"))

    alone = Example(task="t", state=STATE, questions=[Q_A])
    a_then_b = Example(task="t", state=STATE, questions=[Q_A, Q_B])
    b_then_a = Example(task="t", state=STATE, questions=[Q_B, Q_A])
    crowded = Example(task="t", state=STATE, questions=[Q_B, Q_C, Q_A])

    ref = q_a_logits(model, packer, [alone], 0, device)
    cases = {
        "A alone (reference)": ([alone], 0),
        "A with B after it": ([a_then_b], 0),
        "A with B before it": ([b_then_a], 1),
        "A last of three": ([crowded], 2),
        "A batched with another example": ([OTHER, a_then_b], 1),
        "A batched, other example first": ([b_then_a, OTHER, alone], 1),
    }
    print(f"model: {label}\nmask: {mask_mode}   dtype: fp32   device: {device}\n")
    worst = 0.0
    for name, (examples, row) in cases.items():
        delta = (q_a_logits(model, packer, examples, row, device) - ref).abs().max().item()
        worst = max(worst, delta)
        print(f"  {name:<34} max|delta logit| = {delta:.3e}")
    ok = worst < args.threshold
    print(f"\nmax |delta logit| = {worst:.3e}   threshold {args.threshold:.0e}   "
          f"{'PASS - questions are independent' if ok else 'FAIL - questions leak into each other'}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
