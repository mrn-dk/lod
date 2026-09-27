"""Row 17, LexGLUE UNFAIR-ToS -- which kind of unfair term a flagged sentence is.

UNFAIR-ToS (Lippi et al. 2019, "CLAUDETTE"; packaged by LexGLUE) is 50 online terms of
service split into sentences, and a sentence carries a label only when lawyers judged it
**potentially or clearly unfair to the consumer** under EU consumer law, in one of eight
categories. The label is not the sentence's topic. A choice-of-law sentence naming the
consumer's own country's law is fair and carries no label; so does the fifth sentence of
an arbitration clause whose operative sentence carries `Arbitration`.

The generic adapter read the list column as eight yes/no tasks, "does this text carry the
label X", and enrichment then rewrote each into a *topic* question ("Does the text
specify which jurisdiction's law governs the agreement?") while the target stayed "was
this sentence flagged as an unfair X term". Measured on the 4,000 stored sentences
(domain-10 audit): of the sentences a topic
reading would call yes, 57-89 % per category carry no label -- 144 of 161 arbitration
sentences, 116 of 141 governing-law sentences. And each yes/no task saw 1/8 of the rows
with a 0.6-4.4 % positive rate, so its majority baseline was 0.956-0.994: the domain's
eval was a constant.

Neither half of that can be repaired from the sentence alone. Whether an *unlabelled*
sentence is fair, or an unflagged continuation of a flagged clause, is decided by
neighbouring sentences and by the annotation guideline's segmentation, so "none" is not a
class a reader of one sentence can recover. What the label *does* determine, given that a
sentence was flagged, is which of the eight kinds of unfairness it is. That is the one
question asked here:

* state: one flagged sentence (lower-cased and tokenised, as LexGLUE ships it);
* options: the eight published categories, in the dataset's own order, with their
  criteria condensed from the CLAUDETTE guideline in `descriptions.py`;
* target: the sentence's single label. The 46 sentences with two or three labels are
  dropped -- one option cannot be the answer -- and unlabelled sentences are not used.
"""

from __future__ import annotations

import hashlib
from typing import Iterator

from lod.schema import Example, Question
from lod.corpus.services.sources.real.base import RealTask

TASK = "lexglue_unfair_tos__type"
LICENCE = "CC-BY-4.0"
URL = "https://huggingface.co/datasets/coastalcph/lex_glue"
TEMPLATE = ("Annotators flagged this sentence from an online service's terms of service "
            "as potentially unfair to consumers under EU consumer law. Which type of "
            "unfair term is it?")

# LexGLUE's published ClassLabel order, used when the store has no features sidecar
CATEGORIES = ("Limitation of liability", "Unilateral termination", "Unilateral change",
              "Content removal", "Contract by using", "Choice of law", "Jurisdiction",
              "Arbitration")

# Condensed from the category definitions in Lippi et al. (2019), as the
# annotators applied them. Each says what makes the term *unfair*, not only its topic.
CRITERIA = {
    "Limitation of liability": "the provider excludes or limits its liability for harm "
                               "or losses the consumer may suffer from the service",
    "Unilateral termination": "the provider may suspend or terminate the service or the "
                              "contract at its own discretion",
    "Unilateral change": "the provider may change the terms or the service itself "
                         "without the consumer's agreement",
    "Content removal": "the provider may delete or modify content the consumer has "
                       "posted or uploaded",
    "Contract by using": "the consumer is bound by the terms merely by using the "
                         "service, without explicitly accepting them",
    "Choice of law": "disputes are governed by a law other than that of the "
                     "consumer's country of residence",
    "Jurisdiction": "disputes must be brought before courts other than those where "
                    "the consumer lives",
    "Arbitration": "disputes are to be resolved by arbitration rather than in a court, "
                   "other than at the consumer's free choice",
}

_BANK = None


def _question(key: str) -> str:
    global _BANK
    if _BANK is None:
        from lod.phrasings import PhrasingBank
        _BANK = PhrasingBank.load()
    return _BANK.pick("d10.unfair_tos_type", TEMPLATE, key)


def _spec():
    from lod.corpus.services.sources.real.table import SPECS
    return next(s for s in SPECS if s.path == "coastalcph/lex_glue"
                and s.config == "unfair_tos")


def flagged_sentences(rows: list[dict], names: list[str]) -> list[tuple[str, str]]:
    """(sentence, its one label) for every sentence flagged with exactly one category.

    A sentence repeated in the source (boilerplate shared by two services) is kept once,
    and dropped if its copies disagree.
    """
    labels_of: dict[str, set[tuple[str, ...]]] = {}
    order: list[str] = []
    for r in rows:
        text = str(r.get("text") or "").strip()
        labs = r.get("labels")
        if not text or not isinstance(labs, list):
            continue
        key = tuple(sorted(names[i] if isinstance(i, int) and 0 <= i < len(names) else str(i)
                           for i in labs))
        if text not in labels_of:
            labels_of[text] = set()
            order.append(text)
        labels_of[text].add(key)
    out = []
    for text in order:
        sets = labels_of[text]
        if len(sets) != 1:
            continue
        (labs,) = sets
        if len(labs) == 1 and labs[0] in CATEGORIES:
            out.append((text, labs[0]))
    return out


def _load(n: int) -> Iterator[Example]:
    from lod.corpus.services.sources.real import hf
    spec = _spec()
    rows = hf.load_rows(spec)
    names = hf.class_names(spec).get("labels") or list(CATEGORIES)
    options = list(CATEGORIES)
    made = 0
    # rows are stored in document order; spread so a short quota is not one service
    for text, label in flagged_sentences(hf._spread(rows), names):
        if made >= n:
            return
        made += 1
        key = hashlib.sha1(text.encode()).hexdigest()[:16]
        yield Example(
            task=TASK,
            state=text[:4000],
            questions=[Question(id="unfair_type", question=_question(key),
                                options=options, target=options.index(label))],
        )


def tasks() -> list[RealTask]:
    return [RealTask(row=17, name=TASK, licence=LICENCE, url=URL, load=_load,
                     notes="UNFAIR-ToS sentences flagged with exactly one unfair-term "
                           "category; which category (8-way). Unlabelled and "
                           "multi-label sentences are not used")]
