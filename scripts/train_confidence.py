#!/usr/bin/env python
"""Train the confidence head against a frozen scoring model.

`confidence` is P(the model's argmax is right and the right answer is among the options)
-- a separate output from the distribution. The head is a Linear on the mean of a
question's (shortlisted) option hidden states plus, with `--features`, a Linear on the
shape of its distribution; it is trained with BCE on `val` + `devreal` (plus any `--ood`
files) against whether the frozen model was right.

Everything upstream is frozen: the backbone, the scoring head, the comparison layer and
the temperature do not move, so the head reads exactly the distribution that ships. Train
it last, on the final checkpoint.

Why a head, when max-prob is free: it sees the hidden states, so it can notice that a
question is unlike anything in training even when the distribution looks confident, and
it is trained to be unsure when the correct option was withheld -- which max-prob, being
a function of the distribution only, cannot express.

    train_confidence.py --ckpt runs/<run>/best --data data/<corpus> \
        --criteria-augment --ood data/<corpus>/devood.jsonl --out runs/<run>-release
"""

from __future__ import annotations

import os

os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")  # must precede `import torch`

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from lod.model.packing import Packer
from lod.model.scorer import OptionScoringModel
from lod.schema import read_jsonl
from lod.training.data import DecisionDataset, make_loader


@torch.no_grad()
def collect(model, loader, device, amp_dtype=None, feature_names=()):
    """-> (pooled hidden [N, H], correct [N], maxprob [N], tasks, withheld [N]).

    One pass over the frozen model. Both the head's input and its target come from here,
    so they cannot drift apart.

    `correct` is the head's training target and it is **not** simply "the argmax matched
    the gold index". On a withheld-criteria question the right answer is not in the
    option list at all, and the gold index points at the appended sentinel; scoring that
    as correct would train the head to be confident exactly where it should be unsure.
    So the target is: does the supplied option set contain the right answer, and did the
    model pick it? -- which is 0 on every abstaining question, however the distribution
    fell.

    A not-in-context row (a rule naming a field the record does not carry, a claim the
    evidence neither states nor contradicts) carries no score target at all, so it never
    reaches the scoring loss; it is included here with target 0.
    """
    model.eval()
    H, Y, P, tasks, W = [], [], [], [], []
    F: dict[str, list] = {k: [] for k in feature_names}
    for batch in loader:
        if batch is None:
            continue
        # the serving path: pooled over the comparison layer's shortlist, and the
        # global distribution's features at the checkpoint's temperature
        res = model.score_batch(batch, device, amp_dtype)
        valid = res["valid"]
        grouped = res["grouped"]
        pooled = res["pooled"]

        probs = torch.softmax(grouped.masked_fill(~valid, float("-inf")), dim=-1)
        pred = probs.argmax(dim=-1)
        target = batch["target"].to(device)
        gold = target.argmax(dim=-1)
        has = batch["has_target"].to(device)

        withheld = batch.get("withheld")
        withheld = (withheld.to(device) if withheld is not None
                    else torch.zeros_like(has))
        hit = (pred == gold) & has & ~withheld
        # a question with no score target still trains the head, as a negative: that is
        # the whole abstention channel
        use = has | withheld

        H.append(pooled[use].cpu())
        for k in feature_names:
            F[k].append(res["features"][k][use].float().cpu())
        Y.append(hit[use].float().cpu())
        P.append(probs[use].max(dim=-1).values.cpu())
        W.append(withheld[use].cpu())
        tasks += [t for t, k in zip(batch["q_task"], use.tolist()) if k]
    if not H:
        raise SystemExit("no labelled questions found")
    feats = {k: torch.cat(v) for k, v in F.items()}
    return torch.cat(H), torch.cat(Y), torch.cat(P), tasks, torch.cat(W), feats


def aurc(conf: torch.Tensor, correct: torch.Tensor) -> float:
    """Area under the risk-coverage curve. Lower is better.

    The head ships if it beats max-prob on this (`aurc_ratio` <= 0.95). It measures
    ranking, not calibration: a confidence that orders questions well has low AURC even
    if its absolute values are off, which is the right property for an abstention signal.
    """
    order = torch.argsort(conf, descending=True)
    err = (1.0 - correct[order])
    cum = torch.cumsum(err, dim=0) / torch.arange(1, len(err) + 1, dtype=torch.float32)
    return float(cum.mean())


def ece_against_correctness(conf: torch.Tensor, correct: torch.Tensor,
                            bins: int = 15) -> float:
    edges = torch.linspace(0, 1, bins + 1)
    total = 0.0
    for i in range(bins):
        m = (conf >= edges[i]) & (conf < edges[i + 1] if i < bins - 1 else conf <= 1.0)
        if m.sum() == 0:
            continue
        total += float(m.float().mean()) * abs(float(conf[m].mean()) - float(correct[m].mean()))
    return total


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="the frozen scoring checkpoint")
    ap.add_argument("--data", type=Path, required=True,
                    help="corpus directory; val.jsonl and devreal.jsonl are read from it")
    ap.add_argument("--val", type=Path, default=None)
    ap.add_argument("--devreal", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--criteria-augment", action="store_true",
                    help="apply the criteria augmentations while collecting, so the head "
                         "sees withheld-criteria questions. Without them it is never shown "
                         "an unanswerable option set")
    ap.add_argument("--features", default="",
                    help="comma list of distribution features the head also reads "
                         "(log_n,max_p,entropy_ratio,margin,logit_gap)")
    ap.add_argument("--ood", type=Path, action="append", default=[],
                    help="outlier exposure: a file of families the scorer never trained on "
                         "(e.g. devood.jsonl). Repeatable.")
    ap.add_argument("--max-batch-tokens", type=int, default=0,
                    help="cap padded tokens per batch (long states)")
    ap.add_argument("--holdout-frac", type=float, default=0.2,
                    help="share of TASKS kept out of head training and scored after, so the "
                         "AURC printed is not the head's own training fit")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = OptionScoringModel.load(args.ckpt, torch_dtype=dtype)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)          # frozen: the head must not move the scorer

    packer = Packer.for_model(model)
    val_path = args.val or args.data / "val.jsonl"
    dev_path = args.devreal or args.data / "devreal.jsonl"
    examples = read_jsonl(val_path) + read_jsonl(dev_path)
    for f in args.ood:
        examples += read_jsonl(f)
    print(f"{len(examples):,} examples from {val_path.name} + {dev_path.name}"
          + "".join(f" + {f.name} (outlier exposure)" for f in args.ood))
    feature_names = tuple(x for x in args.features.split(",") if x)

    ds = DecisionDataset(examples, packer, augment=args.criteria_augment,
                         seed=args.seed, criteria_augment=args.criteria_augment)
    loader = make_loader(ds, packer.pad_id,
                         dtype if device.type == "cuda" else torch.float32,
                         batch_size=args.batch_size, num_workers=args.num_workers,
                         max_tokens=args.max_batch_tokens)
    hidden, correct, maxprob, tasks, withheld, feats = collect(
        model, loader, device, torch.bfloat16 if device.type == "cuda" else None,
        feature_names)
    # task-disjoint holdout for an honest read of the head
    import hashlib
    held = torch.tensor([int(hashlib.sha1(t.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
                         < args.holdout_frac for t in tasks])
    fit = ~held
    print(f"{len(correct):,} labelled questions; base accuracy {correct.mean():.4f}; "
          f"withheld criteria on {int(withheld.sum()):,} ({100*float(withheld.float().mean()):.1f}%)")
    print(f"max-prob baseline: AURC {aurc(maxprob, correct):.4f}  "
          f"ECE {ece_against_correctness(maxprob, correct):.4f}")

    head = model.attach_confidence_head(features=feature_names)
    head.train()
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr)
    lossf = nn.BCELoss()

    def run_head(idx=None):
        sel = slice(None) if idx is None else idx
        f = {k: v[sel].to(device) for k, v in feats.items()} if feature_names else None
        return head(hidden[sel].to(device), f)

    fit_idx = fit.nonzero().squeeze(-1)
    n = len(fit_idx)
    for epoch in range(args.epochs):
        perm = fit_idx[torch.randperm(n)]
        total = 0.0
        for i in range(0, n, 256):
            idx = perm[i : i + 256]
            opt.zero_grad(set_to_none=True)
            pred = run_head(idx)
            loss = lossf(pred, correct[idx].to(device))
            loss.backward()
            opt.step()
            total += float(loss) * len(idx)
        head.eval()
        with torch.no_grad():
            conf = run_head().cpu()
        msg = f"  epoch {epoch+1}: loss {total/max(n,1):.4f}  AURC {aurc(conf[fit], correct[fit]):.4f}"
        if held.any():
            msg += (f"  | held-out tasks AURC {aurc(conf[held], correct[held]):.4f} "
                    f"(max-prob {aurc(maxprob[held], correct[held]):.4f})")
        print(msg)
        head.train()

    head.eval()
    with torch.no_grad():
        conf = run_head().cpu()
    intact = ~withheld
    gap = (float(conf[intact].mean()) - float(conf[withheld].mean())
           if int(withheld.sum()) and int(intact.sum()) else float("nan"))
    summary = {
        "n_questions": int(n),
        "accuracy": float(correct.mean()),
        "n_withheld": int(withheld.sum()),
        "conf_withheld": float(conf[withheld].mean()) if int(withheld.sum()) else None,
        "conf_intact": float(conf[intact].mean()) if int(intact.sum()) else None,
        # abstention: mean confidence on withheld criteria below intact by > 0.15
        "conf_gap": gap,
        "gap_pass": bool(gap > 0.15) if gap == gap else None,
        "head_aurc": aurc(conf, correct),
        "maxprob_aurc": aurc(maxprob, correct),
        "head_ece": ece_against_correctness(conf, correct),
        "maxprob_ece": ece_against_correctness(maxprob, correct),
        # the head must beat max-prob by at least 5 % on AURC
        "aurc_ratio": aurc(conf, correct) / max(aurc(maxprob, correct), 1e-9),
        "features": list(feature_names),
        "ood_files": [str(f) for f in args.ood],
        "heldout_tasks_n": int(held.sum()),
        "heldout_head_aurc": aurc(conf[held], correct[held]) if held.any() else None,
        "heldout_maxprob_aurc": aurc(maxprob[held], correct[held]) if held.any() else None,
    }
    print("\n" + json.dumps(summary, indent=2))
    print(f"AURC ratio head / max-prob (<= 0.95): {summary['aurc_ratio']:.4f} "
          f"{'PASS' if summary['aurc_ratio'] <= 0.95 else 'FAIL'}")
    if summary["conf_gap"] == summary["conf_gap"] and summary["conf_gap"] is not None:
        print(f"abstention gap (> 0.15): {summary['conf_gap']:.4f} "
              f"{'PASS' if summary['gap_pass'] else 'FAIL'}")

    args.out.mkdir(parents=True, exist_ok=True)
    model.confidence_mode = "head"
    model.save(args.out)
    (args.out / "confidence_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {args.out} (confidence_mode=head)")


if __name__ == "__main__":
    main()
