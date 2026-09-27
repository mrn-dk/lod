"""Run the model over a loader and turn the result into metrics and per-question records."""

from __future__ import annotations

from collections import defaultdict

import torch

from lod.model.calibration import metrics, probabilities
from lod.model.scorer import forward_batch, grouped_head_logits, grouped_logits, log_mean_prob


@torch.no_grad()
def collect_logits(
    model,
    loader,
    device: torch.device | str = "cpu",
    amp_dtype: torch.dtype | None = None,
    max_questions: int | None = None,
    return_heads: bool = False,
    return_refs: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    """-> (grouped logits [Nq, K], valid [Nq, K], target [Nq, K], task per question).

    Questions without a target are dropped: metrics need one.

    For a K-head ensemble the grouped logits are the ensemble's log p̄, i.e. the
    distribution that ships. `return_heads=True` appends the raw per-head grouped
    logits [Nq, K, H] as a fifth element (None for a single-head checkpoint); that is
    what the ensemble temperature fit and the head-correlation diagnostic read.

    `return_refs=True` appends, last, the (example index, question index) of every row.
    Questions that do not fit the budget are dropped by the packer, so row i is not the
    i-th question of the file: anything that joins rows back to question ids or `meta`
    must go through these refs, never by position.
    """
    was_training = model.training
    model.eval()
    chunks: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]] = []
    tasks: list[str] = []
    refs: list[tuple[int, int]] = []
    for batch in loader:
        if batch is None:
            continue
        logits = (forward_batch(model, batch, device, non_blocking=True)
                  if amp_dtype is None else _amp_forward(model, batch, device, amp_dtype))
        group = batch["option_group"].to(logits.device)
        slot = batch["option_slot"].to(logits.device)
        v = batch["valid"].to(logits.device)
        heads = None
        if logits.dim() == 2:
            hg = grouped_head_logits(logits.float(), group, slot, v)
            grouped = log_mean_prob(hg, v).cpu()
            heads = hg.cpu() if return_heads else None
        else:
            grouped = grouped_logits(logits.float(), group, slot, v).cpu()
        keep = batch["has_target"]
        chunks.append((grouped[keep], batch["valid"][keep], batch["target"][keep],
                       heads[keep] if heads is not None else None))
        tasks += [t for t, k in zip(batch["q_task"], keep.tolist()) if k]
        refs += [r for r, k in zip(batch.get("q_ref", ()), keep.tolist()) if k]
        if max_questions is not None and sum(c[0].shape[0] for c in chunks) >= max_questions:
            break
    if was_training:
        model.train()

    if not chunks:
        empty = torch.zeros(0, 1)
        out = (empty, empty.bool(), empty, [])
        out = out + (None,) if return_heads else out
        return out + ([],) if return_refs else out

    k = max(c[0].shape[1] for c in chunks)
    def pad(x, fill):
        if x.shape[1] == k:
            return x
        shape = (x.shape[0], k - x.shape[1]) + tuple(x.shape[2:])
        return torch.cat([x, torch.full(shape, fill, dtype=x.dtype)], dim=1)

    logits = torch.cat([pad(c[0], float("-inf")) for c in chunks])
    valid = torch.cat([pad(c[1], False) for c in chunks])
    target = torch.cat([pad(c[2], 0.0) for c in chunks])
    head_logits = (torch.cat([pad(c[3], float("-inf")) for c in chunks])
                   if chunks[0][3] is not None else None)
    if max_questions is not None:
        logits, valid, target, tasks = logits[:max_questions], valid[:max_questions], target[:max_questions], tasks[:max_questions]
        refs = refs[:max_questions]
        if head_logits is not None:
            head_logits = head_logits[:max_questions]
    out = (logits, valid, target, tasks)
    if return_heads:
        out = out + (head_logits,)
    if return_refs:
        if len(refs) != logits.shape[0]:
            raise RuntimeError(f"{len(refs)} question refs for {logits.shape[0]} rows: "
                               "the collate function did not carry q_ref")
        out = out + (refs,)
    return out


def _amp_forward(model, batch, device, amp_dtype):
    with torch.autocast(device_type=torch.device(device).type, dtype=amp_dtype):
        return forward_batch(model, batch, device, non_blocking=True)


def evaluate(
    model,
    loader,
    device: torch.device | str = "cpu",
    amp_dtype: torch.dtype | None = None,
    max_questions: int | None = None,
    n_bins: int = 15,
    temperature: float = 1.0,
) -> dict:
    """Overall + per-task metrics. `temperature` divides the logits first."""
    logits, valid, target, tasks = collect_logits(model, loader, device, amp_dtype, max_questions)
    if temperature != 1.0:
        logits = logits / temperature
    out = dict(metrics(logits, valid, target, n_bins))
    out["per_task"] = per_task_metrics(logits, valid, target, tasks, n_bins)
    return out


def per_task_metrics(logits, valid, target, tasks: list[str], n_bins: int = 15) -> dict[str, dict]:
    idx: dict[str, list[int]] = defaultdict(list)
    for i, t in enumerate(tasks):
        idx[t].append(i)
    out = {}
    for task, rows in sorted(idx.items()):
        sel = torch.tensor(rows, dtype=torch.long)
        out[task] = metrics(logits[sel], valid[sel], target[sel], n_bins)
    return out


def per_question_records(logits, valid, target, tasks: list[str]) -> list[dict]:
    """One record per question for `eval-<split>.jsonl`."""
    probs = probabilities(logits, valid)
    label = target.argmax(dim=-1)
    pred = probs.argmax(dim=-1)
    conf = probs.max(dim=-1).values
    records = []
    for i, task in enumerate(tasks):
        n = int(valid[i].sum())
        records.append({
            "task": task,
            "id": i,
            "confidence": round(float(conf[i]), 6),
            "correct": bool(pred[i] == label[i]),
            "probs": [round(float(x), 6) for x in probs[i, :n]],
            "target": [round(float(x), 6) for x in target[i, :n]],
        })
    return records
