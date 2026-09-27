"""Attach published option definitions to the corpus.

`Question` carries a `descriptions` list, `packing` renders it as
`- <option>: <definition>`, and `api.py` fills it from the caller's `criteria` on every
`bool`, `choice` and `score` request. So descriptions are the shape the deployed model is
asked about most -- and with none in the corpus the model has never seen one. That is the
gap this closes.

The rule: for a row whose source publishes label definitions, attach them
to **half** that task's examples, chosen by a seed-0 hash, so the model learns both the
bare and the described form. `attach` is applied once in `generate.build`, where a
task's full example list is in hand, which is what lets the half be exact rather than
merely expected.

Only definitions that are genuinely *published* go in here:

- **row 9, CWE** -- names and descriptions straight from MITRE's CWE catalogue CSV, fetched
  and cached. The publisher's own text, not a paraphrase.
- **row 9, CVSS v3.1 metrics** -- the metric-value definitions from the CVSS v3.1
  specification (§2.1-2.3, §5). Condensed to one clause each, because a whole
  specification paragraph per option would dominate the state it is meant to annotate.
- **row 20, GoEmotions** -- the emotion definitions from the GoEmotions taxonomy
  (Demszky et al. 2020), as given to its raters.

Rows that were *not* done, and why, so the gap is legible rather than silent:

- **LexGLUE SCOTUS** publishes its issue areas, but its ClassLabel names are "1".."13" and
  the index-to-issue-area mapping is not published anywhere machine-readable. Guessing it
  would put wrong definitions on right labels, which is worse than none.
- **LEDGAR, arXiv, CLINC, MASSIVE, SNIPS, PubMed** need nothing here: their published
  *names* are the definition, and recovering those was the ClassLabel fix in `hf.py`.
- **row 5 (GitHub labels)** does publish a description per label, but GH Archive streams
  are not cached and re-reading the 14-day window to recover them is ~23 h of download.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import urllib.request
import zipfile
from typing import Iterable

from lod.corpus.services.sources.base import CACHE

CWE_CSV = "https://cwe.mitre.org/data/csv/1000.csv.zip"
_CWE_CACHE = CACHE / "cwe_catalogue.json"
UA = {"User-Agent": "lod/1.0 (corpus build; contact via repo)"}


# ---- row 9: CVSS v3.1 -------------------------------------------------------------
# CVSS v3.1 specification, "Base Metrics". One clause per value: enough to define the
# distinction the label turns on, short enough to sit beside 40 other options.
CVSS: dict[str, dict[str, str]] = {
    "attackVector": {
        "NETWORK": "exploitable remotely across a network, at OSI layer 3 or above",
        "ADJACENT_NETWORK": "exploitable only from the same shared physical or logical "
                            "network, such as the same subnet or Bluetooth range",
        "LOCAL": "not reachable over a network; needs local read, write or execute "
                 "access, or a user who runs something",
        "PHYSICAL": "needs the attacker to physically touch or manipulate the component",
    },
    "attackComplexity": {
        "LOW": "no special conditions; an attacker can expect repeatable success",
        "HIGH": "success depends on a condition outside the attacker's control, or on "
                "measurable preparation such as reconnaissance",
    },
    "privilegesRequired": {
        "NONE": "no authorisation needed before the attack",
        "LOW": "needs ordinary user privileges over the attacker's own resources",
        "HIGH": "needs administrative or otherwise significant control of the component",
    },
    "userInteraction": {
        "NONE": "exploitable with no participation by any user",
        "REQUIRED": "a user other than the attacker must do something first",
    },
    "confidentialityImpact": {
        "NONE": "no information is disclosed",
        "LOW": "some information is disclosed, but the attacker does not choose what",
        "HIGH": "total loss of confidentiality, or disclosure of directly serious "
                "information such as credentials",
    },
    "integrityImpact": {
        "NONE": "nothing can be modified",
        "LOW": "data can be modified, but not chosen or not enough to matter seriously",
        "HIGH": "total loss of integrity: the attacker can modify anything, with "
                "serious consequence",
    },
    "availabilityImpact": {
        "NONE": "availability is unaffected",
        "LOW": "reduced performance or intermittent interruption of the resource",
        "HIGH": "total or sustained loss of availability of the resource",
    },
    "baseSeverity": {
        "NONE": "CVSS base score 0.0",
        "LOW": "CVSS base score 0.1 to 3.9",
        "MEDIUM": "CVSS base score 4.0 to 6.9",
        "HIGH": "CVSS base score 7.0 to 8.9",
        "CRITICAL": "CVSS base score 9.0 to 10.0",
    },
}


# ---- row 20: GoEmotions ----------------------------------------------------------
# The GoEmotions taxonomy definitions (Demszky et al., 2020), as given to raters. The
# task asks "Does this comment express X?" over no/yes, so the definition belongs on
# "yes" -- which is also exactly where api.py puts `criteria.true`.
GOEMOTIONS: dict[str, str] = {
    "admiration": "finding something impressive or worthy of respect",
    "amusement": "finding something funny or being entertained",
    "anger": "a strong feeling of displeasure or antagonism",
    "annoyance": "mild anger, irritation",
    "approval": "having or expressing a favourable opinion",
    "caring": "displaying kindness and concern for others",
    "confusion": "lack of understanding, uncertainty",
    "curiosity": "a strong desire to know or learn something",
    "desire": "a strong feeling of wanting something, or wishing for something to happen",
    "disappointment": "sadness or displeasure caused by hopes or expectations not being met",
    "disapproval": "having or expressing an unfavourable opinion",
    "disgust": "revulsion or strong disapproval aroused by something unpleasant",
    "embarrassment": "self-consciousness, shame or awkwardness",
    "excitement": "feeling of great enthusiasm and eagerness",
    "fear": "being afraid or worried",
    "gratitude": "a feeling of thankfulness and appreciation",
    "grief": "intense sorrow, especially caused by someone's death",
    "joy": "a feeling of pleasure and happiness",
    "love": "a strong positive emotion of regard and affection",
    "nervousness": "apprehension, worry, anxiety",
    "optimism": "hopefulness and confidence about the future or about success",
    "pride": "pleasure or satisfaction from one's own achievements, or those of people "
             "one is close to",
    "realization": "becoming aware of something",
    "relief": "reassurance and relaxation following release from anxiety or distress",
    "remorse": "regret or guilty feeling",
    "sadness": "emotional pain, sorrow",
    "surprise": "feeling astonished, startled by something unexpected",
}


def cwe_catalogue(refresh: bool = False) -> dict[str, str]:
    """{"CWE-79": "<MITRE's description>"}, cached to disk.

    MITRE's own text. `Description` is the one-line summary; `Extended Description` runs
    to paragraphs and is not what belongs next to an option.
    """
    if _CWE_CACHE.exists() and not refresh:
        try:
            return json.loads(_CWE_CACHE.read_text())
        except json.JSONDecodeError:
            pass
    req = urllib.request.Request(CWE_CSV, headers=UA)
    blob = urllib.request.urlopen(req, timeout=120).read()
    out: dict[str, str] = {}
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        with z.open(z.namelist()[0]) as fh:
            for r in csv.DictReader(io.TextIOWrapper(fh, encoding="utf-8")):
                cid, name = r.get("CWE-ID"), (r.get("Name") or "").strip()
                desc = (r.get("Description") or "").strip()
                if not cid or not name:
                    continue
                text = name if not desc else f"{name} — {desc}"
                out[f"CWE-{cid}"] = re.sub(r"\s+", " ", text)[:400]
    CACHE.mkdir(parents=True, exist_ok=True)
    _CWE_CACHE.write_text(json.dumps(out, indent=0))
    return out


def for_task(task: str, options: list[str]) -> list[str | None] | None:
    """Published definitions for this task's option set, or None if the source has none.

    Returns one entry per option in the given order, `None` where the publisher defines
    nothing for that option -- `Question.description_for` is built to tolerate exactly
    that, because a source that documents some of its labels is the normal case.
    """
    if task.startswith("goemotions__"):
        emotion = task.split("__", 1)[1]
        text = GOEMOTIONS.get(emotion)
        if not text or [o.lower() for o in options] != ["no", "yes"]:
            return None
        return [None, text]

    if task == "nvd_cwe":
        cat = cwe_catalogue()
        got = [cat.get(o) for o in options]
        return got if any(got) else None

    if task.startswith("nvd_"):
        table = CVSS.get(task[len("nvd_"):])
        if not table:
            return None
        got = [table.get(o) for o in options]
        return got if any(got) else None

    # row 15: SPDX publishes a name for every licence id; row 17: UNFAIR-ToS's category
    # definitions (Lippi et al. 2019). Both tasks are protected from LLM enrichment, so
    # these are their only criteria.
    if task == "spdx_licence":
        from lod.corpus.services.sources.real.spdx import published_names
        names = published_names()
        got = [names.get(o) or None for o in options]
        return got if any(got) else None

    # row 46: short criteria per option, from the source's own annotation guidance
    # where it publishes one (see `multilingual.py`); the tasks are protected (`ml_`)
    if task.startswith("ml_"):
        from lod.corpus.services.sources.real.multilingual import criteria_for
        return criteria_for(task, list(options))

    # row 44: each source's published label definitions (SemEval task
    # descriptions, OLID, SST cut-offs, Malo et al.); protected from enrichment (`sent44_`)
    if task.startswith("sent44_"):
        from lod.corpus.services.sources.real.sentiment import criteria_for
        return criteria_for(task, list(options))

    # row 45: TREC's published class definitions (Li & Roth 2002), DBpedia's
    # ontology, CFPB's product definitions, short criteria elsewhere; protected (`topic_`)
    if task.startswith("topic_"):
        from lod.corpus.services.sources.real.topic import criteria_for
        return criteria_for(task, list(options))

    if task.startswith("lexglue_unfair_tos"):
        from lod.corpus.services.sources.real.unfair_tos import CRITERIA
        got = [CRITERIA.get(o) for o in options]
        return got if any(got) else None

    return None


def _rank(task: str, state: str, seed: int) -> bytes:
    return hashlib.sha1(f"{seed}:{task}:{state}".encode()).digest()


def attach(examples: Iterable, seed: int = 0, share: float = 0.5) -> int:
    """Attach descriptions to `share` of the examples whose task has any. -> how many.

    The rule wants *exactly* half, so the half is taken by ranking the task's examples
    on a seed-0 hash and describing the lowest-ranked half, rather than by thresholding
    each example's hash independently -- which would only give half in expectation and
    would wobble by a percent or so per task.
    """
    items = list(examples)
    if not items:
        return 0
    by_task: dict[str, list] = {}
    for e in items:
        by_task.setdefault(e.task, []).append(e)

    described = 0
    for task, group in by_task.items():
        opts = group[0].questions[0].options if group[0].questions else []
        texts = for_task(task, list(opts))
        if texts is None:
            continue
        order = sorted(group, key=lambda e: _rank(e.task, e.state, seed))
        for e in order[: int(len(order) * share)]:
            for q in e.questions:
                # only where the option set really is the one the table describes
                if list(q.options) == list(opts):
                    q.descriptions = list(texts)
                    described += 1
    return described


def coverage(tasks: Iterable[tuple[str, list[str]]]) -> dict[str, int]:
    """How many of these (task, options) pairs the tables can describe. For A7."""
    n = sum(1 for t, o in tasks if for_task(t, list(o)) is not None)
    return {"describable": n}
