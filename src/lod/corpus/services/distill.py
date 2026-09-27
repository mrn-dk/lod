"""Teacher targets: a larger scorer's distribution over each train question's options.

Every train question the teacher can read gets `meta.teacher`: the teacher's distribution
at its own fitted temperature. Training with `--distill-alpha a` then fits
(1 - a) * gold + a * teacher -- cross-entropy against that mixture is exactly
(1 - a) * CE(gold) + a * CE(teacher), the usual distillation objective. The mixing
happens in `DecisionDataset` after augmentation, matched by option key: mixing into
`target` here would push most questions below `sentinel_eligible`'s confidence bar and
silently change the withheld-criteria and decoy rates the corpus was tuned for.

Only train is annotated. Dev and test measure the student against the world, not
against the teacher. States longer than the teacher's state budget, and questions with
no target (abstentions), keep what they had.

Needs torch and a teacher checkpoint; a GPU in practice.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Teacher:
    model: object
    packer: object
    device: object
    dtype: object
    flex_kw: dict

    def describe(self) -> str:
        return (f"T = {float(self.model.temperature):.4f}, "
                f"option_mode {self.model.option_mode}, {self.device}")


def load_teacher(ckpt: str, attn: str = "flex") -> Teacher:
    """The checkpoint on the best device available; flex attention needs CUDA."""
    import torch

    from lod.model.flexattn import BLOCK_SIZE as FLEX_BLOCK
    from lod.model.packing import Packer
    from lod.model.scorer import OptionScoringModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    attn = attn if device.type == "cuda" else "sdpa"
    model = OptionScoringModel.load(ckpt, torch_dtype=dtype, attn_impl=attn)
    model.to(device).eval()
    flex_kw = {"pad_to_tile": FLEX_BLOCK, "dense_mask": False} if attn == "flex" else {}
    return Teacher(model, Packer.for_model(model), device, dtype, flex_kw)


def annotate(teacher: Teacher, records: list[dict], *, batch_size: int = 32,
             num_workers: int = 8, max_batch_tokens: int = 98304) -> tuple[int, int]:
    """Add `meta.teacher` to every question of `records` (plain dicts, edited in place)
    the teacher can read. -> (questions annotated, examples too long for the teacher)."""
    from lod.model.calibration import probabilities
    from lod.schema import Example
    from lod.evaluation.engine import collect_logits
    from lod.training.data import DecisionDataset, make_loader

    model, packer, device, dtype = teacher.model, teacher.packer, teacher.device, teacher.dtype
    examples = [Example.from_dict(r) for r in records]
    # the teacher reads states only up to the budget it was trained with; a longer state
    # would be truncated or out of its depth, so those keep gold labels
    fits = [len(packer.state_ids(e.state)) <= model.max_state_tokens for e in examples]
    idx = [i for i, f in enumerate(fits) if f]
    ds = DecisionDataset([examples[i] for i in idx], packer, augment=False)
    loader = make_loader(ds, packer.pad_id, dtype, batch_size=batch_size,
                         num_workers=num_workers, max_tokens=max_batch_tokens,
                         **teacher.flex_kw)
    logits, valid, _target, _tasks, refs = collect_logits(
        model, loader, device, amp_dtype=dtype if device.type == "cuda" else None,
        return_refs=True)
    probs = probabilities(logits, valid)
    mixed = 0
    for row, (ii, j) in enumerate(refs):
        i = idx[ii]
        q = records[i]["questions"][j]
        n = len(q["options"])
        if examples[i].questions[j].target_probs() is None:
            continue
        q["meta"] = dict(q.get("meta") or {},
                         teacher=[round(float(x), 6) for x in probs[row, :n]])
        mixed += 1
    return mixed, len(fits) - len(idx)
