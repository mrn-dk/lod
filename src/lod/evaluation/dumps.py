"""Per-question eval dumps (`eval-<split>.jsonl`) and the metrics read off them.

`scripts/evaluate.py` writes one record per question: `task`, `qid`, `probs` and `target`
over the question's own options, `confidence` (max-prob), `correct`, `ref` (example and
question index in the data file), `key` (a content hash, see `question_key`) and `meta`.
Everything here reads those records only -- no model, no GPU -- and reproduces
`lod.model.calibration.metrics` on the same questions.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

N_BINS = 15

# Task-name prefixes of the code-labelled (generated) sources, used only when the corpus's
# own `meta.json` is not to hand; `real_map` reads the flag off the build instead.
GENERATED_PREFIXES = ("rule_", "entity_fact", "browser_", "sensor_", "toolgate_",
                      "entityres_", "diff_", "sched_")


def read_dump(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with open(path, encoding="utf-8", newline="") as fh:
        return [json.loads(line) for line in fh.read().split("\n") if line.strip()]


def question_key(task: str, state: str, q) -> str:
    """Content hash of one question in its example, written to the dump as `key`.

    The option *set* is hashed, sorted, so the same question scored with its options in
    another order still pairs; the losses compared are order-invariant anyway. The state
    is in the key because a question id alone repeats across every example of a task.
    """
    parts = [task, state, str(q.id), q.question, q.instructions or ""]
    parts += sorted(str(o) for o in q.options)
    return hashlib.blake2b("\x1f".join(parts).encode("utf-8"), digest_size=10).hexdigest()


# ---- metrics over records -----------------------------------------------------------

def accuracy(records: list[dict]) -> float | None:
    return sum(bool(r["correct"]) for r in records) / len(records) if records else None


def nll(records: list[dict]) -> float | None:
    """Mean log score against each question's target distribution."""
    if not records:
        return None
    return sum(-sum(t * math.log(max(p, 1e-12)) for p, t in zip(r["probs"], r["target"]))
               for r in records) / len(records)


def ece(records: list[dict], bins: int = N_BINS) -> float | None:
    """Equal-width ECE of the top probability (`confidence`) against correctness."""
    if not records:
        return None
    total, n = 0.0, len(records)
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        sel = [r for r in records if r["confidence"] >= lo
               and (r["confidence"] < hi or (i == bins - 1 and r["confidence"] <= 1.0))]
        if sel:
            conf = sum(r["confidence"] for r in sel) / len(sel)
            total += (len(sel) / n) * abs(conf - accuracy(sel))
    return total


def aurc(records: list[dict]) -> float | None:
    """Area under the risk-coverage curve, most confident first. Lower is better."""
    if not records:
        return None
    run = total = 0.0
    for i, r in enumerate(sorted(records, key=lambda r: -r["confidence"]), start=1):
        run += 0.0 if r["correct"] else 1.0
        total += run / i
    return total / len(records)


def p_wrong_at(records: list[dict], floor: float = 0.9) -> dict:
    """P(wrong | top probability >= floor): the confidently-wrong rate, with its `n`."""
    sel = [r for r in records if r["confidence"] >= floor]
    if not sel:
        return {"floor": floor, "n": 0, "share": 0.0, "p_wrong": None}
    return {"floor": floor, "n": len(sel), "share": len(sel) / len(records),
            "p_wrong": sum(0.0 if r["correct"] else 1.0 for r in sel) / len(sel)}


def summarise(records: list[dict]) -> dict:
    if not records:
        return {"n": 0}
    return {"n": len(records), "nll": nll(records), "ece": ece(records),
            "accuracy": accuracy(records), "tasks": len({r.get("task", "") for r in records})}


# ---- real labels vs code-labelled ---------------------------------------------------

def real_map(corpus: Path | None) -> dict[str, bool]:
    """Task name -> whether its labels are real (people, real systems), from the corpus's
    `meta.json`. Empty when it is unavailable."""
    if corpus is None or not (Path(corpus) / "meta.json").exists():
        return {}
    got = json.loads((Path(corpus) / "meta.json").read_text())
    return {t["name"]: bool(t.get("real", True)) for t in got.get("tasks", []) if "real" in t}


def is_real(record: dict, mapping: dict[str, bool]) -> bool:
    task = record.get("task", "")
    if task in mapping:
        return mapping[task]
    return not task.startswith(GENERATED_PREFIXES)


def split_real(records: list[dict], mapping: dict[str, bool]) -> tuple[list[dict], list[dict]]:
    return ([r for r in records if is_real(r, mapping)],
            [r for r in records if not is_real(r, mapping)])
