"""Evaluation probes built beside the corpus. Probes are read, never trained on.

`depth_check`  controlled-depth long-context questions: states of 1k to 30k tokens with
               the one deciding item at 10 %, 50 % or 90 % of the state. Every question
               comes from a *devreal* structure of the long-context generator (row 51) --
               the `order` lookup domain, the `bracket` log format and the `markdown`
               policy format -- so no structure in it trains, and its seeds are its own,
               so no state is one of devreal's either. `meta.gen.cell` names the cell;
               `meta.gen` also carries the measured `state_tokens` and `depth_frac`.
               Dev only: never fit a temperature on it.

`knowledge`    does fine-tuning erode what the backbone knew? Public multiple-choice
               knowledge benchmarks outside the corpus's domains, posed as choice
               questions: the question is the state, the answers are the options'
               criteria, the keys are letters. OpenBookQA test, SciQ test, QASC
               validation -- none is in the corpus and none overlaps the Decision Index.
               MMLU, ARC, WinoGrande and HellaSwag were tried and dropped: they are
               Decision Index members, and must not steer any decision.

Both are passed through the Decision Index blocklist by the caller.
"""

from __future__ import annotations

import random
from collections import defaultdict

from lod.schema import Example, Question

# ---- depth check -----------------------------------------------------------------------

LENGTHS = (1000, 2000, 4000, 8000, 16000, 29500)       # 29.5k: the 30k cell, under the cap
DEPTHS = (0.10, 0.50, 0.90)
# single-evidence devreal structures: the depth is where *the* item is
FAMILIES = (("records", "lookup", "order"), ("doc", "log", "bracket"),
            ("doc", "policy", "markdown"))


def depth_check(per_cell: int = 60) -> tuple[list[Example], dict]:
    """-> (examples, {(length, depth): [meta.gen, ...]}), one question per example."""
    from lod.corpus.services.sources.synth import longctx as L

    specs = {(s.kind, s.fam, s.variant): s for s in L.specs()}
    out, cells = [], defaultdict(list)
    for length in LENGTHS:
        for depth in DEPTHS:
            for i in range(per_cell):
                kind, fam, var = FAMILIES[i % len(FAMILIES)]
                spec = specs[(kind, fam, var)]
                assert spec.split == "devreal"
                rng = random.Random(L._seed("depth_check", length, depth, i))
                ex = L.make(spec, rng, lengths=[length], depths=[depth], n_q=1)
                ex.task = f"depthcheck_{fam}"
                for q in ex.questions:
                    q.id = f"L{length}_D{int(depth * 100)}_{i}"
                    q.meta["gen"]["cell"] = {"length": length, "depth": depth}
                    cells[(length, depth)].append(q.meta["gen"])
                out.append(ex)
    return out, cells


def depth_table(cells: dict) -> list[str]:
    """Per cell: count, measured state tokens and where the deciding item landed."""
    lines = [f"{'cell':>14} {'n':>4} {'state_tokens (min/mean/max)':>30} "
             f"{'depth_frac (mean, min-max)':>28}"]
    for (length, depth), gs in sorted(cells.items()):
        st = [g["state_tokens"] for g in gs]
        df = [g["depth_frac"] for g in gs]
        lines.append(f"{length:>7}@{int(depth * 100):>3}% {len(gs):4} "
                     f"{min(st):>9} {sum(st) / len(st):>9.0f} {max(st):>9}   "
                     f"{sum(df) / len(df):>8.3f} ({min(df):.3f}-{max(df):.3f})")
    return lines


# ---- knowledge probe -------------------------------------------------------------------

LETTERS = "ABCDEFGH"
INSTRUCTIONS = "Pick the correct answer to the question in the state."


def _ex(task: str, stem: str, answers: list[str], gold: int, qid: str) -> Example:
    keys = list(LETTERS[: len(answers)])
    q = Question(id=qid, question="Which answer is correct?", options=keys, target=gold,
                 descriptions=[str(a) for a in answers], instructions=INSTRUCTIONS)
    return Example(state=stem.strip(), questions=[q], task=task)


def sciq() -> list[Example]:
    from datasets import load_dataset
    ds = load_dataset("allenai/sciq", split="test")
    out = []
    for i, r in enumerate(ds):
        a = [r["distractor1"], r["distractor2"], r["distractor3"], r["correct_answer"]]
        order = list(range(4))
        random.Random(i).shuffle(order)      # the source lists the answer last
        out.append(_ex("knowledge_sciq", r["question"], [a[j] for j in order],
                       order.index(3), f"sciq{i}"))
    return out


def qasc() -> list[Example]:
    from datasets import load_dataset
    ds = load_dataset("allenai/qasc", split="validation")
    return [_ex("knowledge_qasc", r["question"], r["choices"]["text"],
                list(r["choices"]["label"]).index(r["answerKey"]), r["id"])
            for r in ds]


def obqa() -> list[Example]:
    from datasets import load_dataset
    ds = load_dataset("allenai/openbookqa", "main", split="test")
    out = []
    for r in ds:
        labels = list(r["choices"]["label"])
        out.append(_ex("knowledge_openbookqa", r["question_stem"], r["choices"]["text"],
                       labels.index(r["answerKey"]), r["id"]))
    return out


def knowledge_probe() -> list[Example]:
    """OpenBookQA, SciQ and QASC as choice questions (downloads from the Hub)."""
    return obqa() + sciq() + qasc()
