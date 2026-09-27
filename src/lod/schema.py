"""Typed-decision data model and JSONL I/O.

Everything is an enum. A question carries its own option list; the model's
distribution is only ever over that list, which is what makes the output
type-safe by construction.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

# `target` is an option index (hard), a probability vector over the options
# (soft), or absent at inference time.
Target = int | list[float] | None


@dataclass
class Question:
    id: str
    question: str
    options: list[str]
    target: Target = None
    meta: dict[str, Any] | None = None  # passed through untouched; probes use it (e.g. {"p": 0.73})
    # Optional instructions and per-option descriptions. Absent or None is exactly the
    # behaviour of a question without them.
    instructions: str | None = None       # free text after the question, before "Options:"
    descriptions: list[str | None] | None = None  # one per option, same order

    def description_for(self, i: int) -> str | None:
        """The description of option `i`, or None. Tolerates a short or ragged list:
        a source that publishes definitions for some labels and not others is normal."""
        if not self.descriptions or i >= len(self.descriptions):
            return None
        d = self.descriptions[i]
        return d if d else None

    @property
    def has_descriptions(self) -> bool:
        return bool(self.descriptions) and any(self.descriptions)

    def target_probs(self) -> list[float] | None:
        """Target as a probability vector over `options`, or None if unlabelled."""
        t = self.target
        if t is None:
            return None
        if isinstance(t, bool):  # bool is an int subclass; treat as index 0/1
            t = int(t)
        if isinstance(t, int):
            if not 0 <= t < len(self.options):
                raise ValueError(f"target index {t} out of range for {len(self.options)} options")
            return [1.0 if i == t else 0.0 for i in range(len(self.options))]
        probs = [float(x) for x in t]
        if len(probs) != len(self.options):
            raise ValueError(f"soft target of length {len(probs)} for {len(self.options)} options")
        return probs

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"id": self.id, "question": self.question, "options": list(self.options)}
        if self.target is not None:
            d["target"] = self.target
        if self.meta:
            d["meta"] = self.meta
        if self.instructions:
            d["instructions"] = self.instructions
        if self.has_descriptions:
            d["descriptions"] = list(self.descriptions)
        return d

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Question":
        return Question(
            id=str(d.get("id", "")),
            question=d["question"],
            options=[str(o) for o in d["options"]],
            target=d.get("target"),
            meta=d.get("meta"),
            instructions=d.get("instructions"),
            descriptions=d.get("descriptions"),
        )


@dataclass
class Example:
    state: str
    questions: list[Question] = field(default_factory=list)
    task: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "state": self.state,
            "questions": [q.to_dict() for q in self.questions],
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Example":
        return Example(
            state=d["state"],
            questions=[Question.from_dict(q) for q in d["questions"]],
            task=str(d.get("task", "")),
        )


def typed_to_enum(d: dict[str, Any]) -> Question:
    """Map an API-level typed question onto a plain enum Question."""
    kind = d.get("type", "enum")
    qid = str(d.get("id", ""))
    text = d["question"]
    if kind == "bool":
        options = ["no", "yes"]
    elif kind == "score":
        lo, hi = int(d["min"]), int(d["max"])
        if hi < lo:
            raise ValueError(f"score question {qid!r}: max {hi} < min {lo}")
        options = [str(i) for i in range(lo, hi + 1)]
    elif kind == "enum":
        options = [str(o) for o in d["options"]]
        if len(options) < 2:
            raise ValueError(f"enum question {qid!r} needs at least 2 options")
    else:
        raise ValueError(f"unknown question type {kind!r}")
    return Question(id=qid, question=text, options=options, target=d.get("target"),
                    instructions=d.get("instructions"),
                    descriptions=d.get("descriptions"))


def read_jsonl(path: str | Path) -> list[Example]:
    out: list[Example] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(Example.from_dict(json.loads(line)))
    return out


def write_jsonl(path: str | Path, examples: Iterable[Example]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex.to_dict(), ensure_ascii=False) + "\n")
            n += 1
    return n
