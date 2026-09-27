"""Source-table row 34 — claim verification against supplied evidence.

The corpus's `noul` questions are almost all "does this text carry this property":
sentiment, toxicity, emotion, intent. None of them is "is this statement true of the
record in front of you", which is the shape a caller actually sends when the answer is
supposed to come out of the state rather than out of the weights. FEVER and VitaminC are
that shape, with human labels, and both carry an explicit **not enough information** class
— so they supply the abstention channel without a choice question having to be mutilated
to produce one.

    SUPPORTS           -> yes
    REFUTES            -> no
    NOT ENOUGH INFO    -> not in context: no score target, confidence supervised to 0

VitaminC is the more valuable of the two here. It is built by pairing a claim with two
*revisions* of the same Wikipedia sentence, one that supports it and one that does not, so
it is a contrastive pair with a human label on both halves. `vitaminc_contrast` puts both
revisions in one state and asks both of the case's claims of each revision (four
questions, each revision supporting one claim and not the other), which is the only way
to be certain both halves reach the model under one prefill.

Licences. FEVER: the claims are CC BY-SA 3.0 and the evidence is Wikipedia text, likewise
CC BY-SA 3.0. VitaminC: CC BY-SA 3.0, Wikipedia-derived. Both are share-alike, which is
recorded here and in the build's `meta.json` so the release audit can see it; neither is
a permissive licence and neither may be relicensed.
"""

from __future__ import annotations

import hashlib
import re
from typing import Iterator

from lod.phrasings import PhrasingBank
from lod.schema import Example, Question
from lod.corpus.services.sources.real.base import RealTask, disjoint_slice
from lod.corpus.repositories import raw_store as store

ROW = 34
FEVER_KEY = "copenlu_fever_gold_evidence"
FEVER_LICENCE = "CC BY-SA 3.0 (FEVER claims; Wikipedia evidence) — share-alike"
FEVER_URL = "https://huggingface.co/datasets/copenlu/fever_gold_evidence"
VITC_KEY = "tals_vitaminc"
VITC_LICENCE = "CC BY-SA 3.0 (VitaminC; Wikipedia-derived) — share-alike"
VITC_URL = "https://huggingface.co/datasets/tals/vitaminc"

CRITERIA = ["The evidence contradicts the claim, or shows it to be false.",
            "The evidence states the claim, or entails it."]
INSTRUCTION = ("Answer from the evidence above and nothing else. If the evidence neither "
               "states nor contradicts the claim, the claim is not in context.")

# Question templates, registered in `lod/assets/phrasing_specs/d06.json`. The wording is
# drawn per record from the phrasing bank; with an empty bank it is the template itself.
Q_CLAIM_ID = "d06.claim_supported"
Q_CLAIM = "Is this claim supported by the evidence: {claim}"
Q_CONTRAST_ID = "d06.vitaminc_contrast"
Q_CONTRAST = "Reading revision {letter} only — Is this claim supported by the evidence: {claim}"
_BANK: PhrasingBank | None = None


def _phrasing(template_id: str, template: str, key: str) -> str:
    global _BANK
    if _BANK is None:
        _BANK = PhrasingBank.load()
    return _BANK.pick(template_id, template, key)


_BRACKETS = {"-LRB-": "(", "-RRB-": ")", "-LSB-": "[", "-RSB-": "]",
             "-LCB-": "{", "-RCB-": "}", "``": '"', "''": '"'}
_SPACE = re.compile(r"\s+")


def clean(text: str) -> str:
    """FEVER ships Penn-Treebank bracket tokens and loose spacing; a caller would not."""
    for a, b in _BRACKETS.items():
        text = text.replace(a, b)
    text = _SPACE.sub(" ", text).strip()
    text = re.sub(r" ([,.;:!?%)])", r"\1", text)
    text = re.sub(r"([(]) ", r"\1", text)
    return text.replace("_", " ")


def _question(claim: str, label: str, qid: str = "claim", key: str = "",
              letter: str | None = None) -> Question:
    claim = clean(claim)
    if letter is None:
        text = _phrasing(Q_CLAIM_ID, Q_CLAIM, key or claim).format(claim=claim)
    else:
        text = _phrasing(Q_CONTRAST_ID, Q_CONTRAST, key or claim).format(
            letter=letter, claim=claim)
    if label == "NOT ENOUGH INFO":
        return Question(qid, text, ["no", "yes"], None,
                        {"abstain": True, "reason": "not_enough_information",
                         "case": "not_in_context", "claim": claim},
                        INSTRUCTION, list(CRITERIA))
    target = [0.0, 1.0] if label == "SUPPORTS" else [1.0, 0.0]
    return Question(qid, text, ["no", "yes"], target,
                    {"case": "supported" if label == "SUPPORTS" else "refuted",
                     "claim": claim},
                    INSTRUCTION, list(CRITERIA))


def _conflicts(pairs) -> set:
    """Keys the raw rows give more than one label.

    The same claim over the same evidence, labelled SUPPORTS by one annotator and
    REFUTES or NOT ENOUGH INFO by another, is a question with two answers; neither label
    is ours to prefer, so the input is dropped. VitaminC has 21 such (claim, evidence)
    pairs among 45,000 rows, 6 of them inside the 2,000 `vitaminc_claim` examples.
    """
    seen: dict = {}
    for key, label in pairs:
        seen.setdefault(key, set()).add(label)
    return {k for k, v in seen.items() if len(v) > 1}


def _fever_state(row: dict) -> str:
    seen, out = set(), []
    for item in row.get("evidence") or []:
        if isinstance(item, (list, tuple)) and len(item) >= 3:
            page, _, text = item[0], item[1], item[2]
            text = clean(str(text))
            if text and text not in seen:
                seen.add(text)
                out.append(f"{clean(str(page))}: {text}")
    return "\n".join(out)


def fever(n: int, index: int = 0, n_tasks: int = 1) -> Iterator[Example]:
    rows = disjoint_slice(store.load(FEVER_KEY), index, n_tasks)
    keyed = [((clean(str(row["claim"])), _fever_state(row)), row) for row in rows]
    bad = _conflicts((k, row["label"]) for k, row in keyed)
    seen: set = set()
    made = 0
    for key, row in keyed:
        if made >= n:
            return
        state = key[1]
        # an exact repeat of (claim, evidence) is one question, not two
        if len(state) < 40 or key in bad or key in seen:
            continue
        seen.add(key)
        yield Example(state=state,
                      questions=[_question(row["claim"], row["label"],
                                           key=str(row.get("id") or key[0]))],
                      task="fever_claim")
        made += 1


def vitaminc(n: int, index: int = 0, n_tasks: int = 1) -> Iterator[Example]:
    every = store.load(VITC_KEY)
    rows = disjoint_slice(every, index, n_tasks)
    keyed = [((clean(str(row["claim"])), clean(str(row.get("evidence") or ""))), row)
             for row in rows]
    # conflicts are judged over the whole store, not the slice: a clashing pair can sit
    # one row either side of the slice boundary, or anywhere else in the file
    bad = _conflicts(((clean(str(r["claim"])), clean(str(r.get("evidence") or ""))),
                      r["label"]) for r in every)
    seen: set = set()
    made = 0
    for key, row in keyed:
        if made >= n:
            return
        state = key[1]
        if len(state) < 40 or key in bad or key in seen:
            continue
        seen.add(key)
        yield Example(state=f"{clean(str(row.get('page') or ''))}: {state}",
                      questions=[_question(row["claim"], row["label"],
                                           key=str(row.get("unique_id") or key))],
                      task="vitaminc_claim")
        made += 1


def _ab_order(case: str, bit: int = 0) -> bool:
    """Whether to swap a pair (bit 0: the revisions; bit 1: the claims).

    Deterministic in the `case_id` alone, so the generator stays reproducible and two
    builds of the same slice agree, but not a function of the labels -- which is the
    whole point. See `vitaminc_contrast`.
    """
    h = hashlib.sha256(f"vitaminc_contrast:{case}".encode()).digest()
    return bool((h[0] >> bit) & 1)


def vitaminc_contrast(n: int, index: int = 0, n_tasks: int = 1) -> Iterator[Example]:
    """Two revisions of one sentence in one state, every claim of the case asked of each.

    A VitaminC `case_id` is a grid: one or two claims, each labelled by a human against
    each of two revisions of the same Wikipedia sentence. The labels are not ours; the
    construction only decides which cells go into one sequence.

    Two leaks shaped this, both measured with the state (or the claim) ignored:

    * **Position.** The first version picked "the SUPPORTS row, then another" in a fixed
      order, so revision A was `supported` in 2,000 of 2,000 examples in an earlier build
      and "yes to A, no to B" scored **1.0000 over 2,989 scored questions**. The revisions
      (and the claims) are now ordered by a hash of the `case_id`.
    * **Which revision is the supporting one.** A two-row case is one claim over two
      revisions, SUPPORTS on one and NOT ENOUGH INFO on the other in 3,102 of 3,112
      cases -- and the NEI revision is the one with the fact deleted, so the supported
      revision is the *longer* one in 97 % of them. After the position fix, "the longer
      revision is the supported one" still scored **0.688 of 2,000 pairs** (0.908 on the
      supported/NEI half). So a case is used only if the answer is not a function of the
      revision at all: every claim is asked of both revisions, and each revision
      supports one claim and fails another. That is VitaminC's two-claim grid (4,045 of
      the 7,159 two-revision cases in this half), where "longer revision -> yes" scores
      0.497 per question, i.e. nothing.
    """
    rows = disjoint_slice(store.load(VITC_KEY), index, n_tasks)
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(str(row.get("case_id")), []).append(row)
    made = 0
    for case, members in groups.items():
        if made >= n:
            return
        cell: dict[tuple[str, str], dict] = {}
        clash = False
        for m in members:
            key = (clean(str(m["claim"])), clean(str(m.get("evidence") or "")))
            if key in cell and cell[key]["label"] != m["label"]:
                clash = True
            cell.setdefault(key, m)
        if clash:
            continue
        claims = sorted({c for c, _ in cell})
        revs = sorted({e for _, e in cell})
        if len(revs) != 2 or len(claims) < 2 or min(len(r) for r in revs) < 40:
            continue
        if any((c, r) not in cell for c in claims for r in revs):
            continue
        if any("SUPPORTS" not in {cell[(c, r)]["label"] for c in claims}
               or {cell[(c, r)]["label"] for c in claims} == {"SUPPORTS"}
               for r in revs):
            continue
        if _ab_order(case, 0):
            revs.reverse()
        if _ab_order(case, 1):
            claims.reverse()
        page = clean(str(members[0].get("page") or ""))
        state = (f"Revision A of {page}: {revs[0]}\n\n"
                 f"Revision B of {page}: {revs[1]}")
        qs = []
        for ci, c in enumerate(claims, 1):
            for letter, r in zip("AB", revs):
                m = cell[(c, r)]
                qs.append(_question(m["claim"], m["label"],
                                    f"claim{ci}_{letter.lower()}",
                                    key=str(m.get("unique_id") or f"{case}:{ci}{letter}"),
                                    letter=letter))
        yield Example(state=state, questions=qs, task="vitaminc_contrast")
        made += 1


def tasks() -> list[RealTask]:
    out = []
    if store.has(FEVER_KEY):
        out.append(RealTask(row=ROW, name="fever_claim", licence=FEVER_LICENCE,
                            url=FEVER_URL, load=fever,
                            notes="SUPPORTS/REFUTES/NOT ENOUGH INFO -> yes/no/abstain"))
    if store.has(VITC_KEY):
        out.append(RealTask(row=ROW, name="vitaminc_claim", licence=VITC_LICENCE,
                            url=VITC_URL,
                            load=lambda n: vitaminc(n, 0, 2), family="vitaminc",
                            notes="SUPPORTS/REFUTES/NOT ENOUGH INFO -> yes/no/abstain"))
        out.append(RealTask(row=ROW, name="vitaminc_contrast", licence=VITC_LICENCE,
                            url=VITC_URL,
                            load=lambda n: vitaminc_contrast(n, 1, 2),
                            family="vitaminc",
                            notes="two revisions of one sentence, both claims of the "
                                  "case asked of each, in one sequence"))
    return out
