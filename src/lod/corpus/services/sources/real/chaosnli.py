"""Source-table row 25 — ChaosNLI: NLI relations with 100 annotators each.

Reserved whole-family to `testreal`, and the most demanding
soft targets in the corpus: 100 opinions per pair rather than the five that MNLI and SNLI
ship. Where row 24 gives a distribution that is mostly one label with a dissenter or two,
this gives genuine shape -- 12/68/20 over entailment/neutral/contradiction -- which is
what makes it a real test of whether a calibrated model tracks human disagreement rather
than just the majority.

Getting the data is the awkward part and worth recording. ChaosNLI has no Hub mirror and
no GitHub release; its repo carries only baseline predictions, and its sole published URL
is a Dropbox `/s/` link that is now deprecated and serves a landing page. Kaggle carries
the original archive and serves it unauthenticated, so that is the route.

`label_dist` is ordered [entailment, neutral, contradiction], which `label_counter`
confirms: {'c': 20, 'e': 12, 'n': 68} against [0.12, 0.68, 0.2]. Reading it positionally
without that check would silently invert entailment and contradiction on every example.
"""

from __future__ import annotations

import io
import json
import urllib.request
import zipfile
from typing import Iterator

from lod.schema import Example, Question
from lod.corpus.services.sources.base import CACHE, UA
from lod.corpus.services.sources.real.base import RealTask

KAGGLE = "https://www.kaggle.com/api/v1/datasets/download/curiosityquotient/chaos-nli"
LICENCE = "CC-BY-SA-4.0 (ChaosNLI; Nie et al. 2020)"
URL = "https://github.com/easonnie/ChaosNLI"
_ZIP = CACHE / "chaosnli_v1.0.zip"

# label_dist order, verified against label_counter rather than assumed
NLI_OPTIONS = ["entailment", "neutral", "contradiction"]
FILES = {
    "snli": "chaosNLI_v1.0/chaosNLI_snli.jsonl",
    "mnli": "chaosNLI_v1.0/chaosNLI_mnli_m.jsonl",
    "alphanli": "chaosNLI_v1.0/chaosNLI_alphanli.jsonl",
}
_IN_PROCESS: dict[str, list] = {}

MIN_RATERS = 3   # below this a fraction is not a distribution (crowd.py's floor)
# `label_counter` keys, as the release writes them, mapped to their position in the
# option list. The docstring above says the order was checked by hand on one example;
# `_distribution` turns that one check into a per-row assertion, because a positional
# misread is silent -- it inverts entailment and contradiction and every target still
# looks like a well-formed distribution.
COUNTER_KEYS = {
    "nli": {"e": 0, "n": 1, "c": 2},
    "alphanli": {"1": 0, "2": 1},
}
# the published dist is written to two decimals; the counts are exact
_ROUNDING = 0.005 + 1e-9


def download(refresh: bool = False) -> bytes:
    if _ZIP.exists() and not refresh:
        return _ZIP.read_bytes()
    req = urllib.request.Request(KAGGLE, headers=UA)
    body = urllib.request.urlopen(req, timeout=600).read()
    CACHE.mkdir(parents=True, exist_ok=True)
    _ZIP.write_bytes(body)
    return body


def rows(subset: str) -> list[dict]:
    if subset in _IN_PROCESS:
        return _IN_PROCESS[subset]
    if not _ZIP.exists():
        _IN_PROCESS[subset] = []
        return []
    z = zipfile.ZipFile(io.BytesIO(_ZIP.read_bytes()))
    name = FILES[subset]
    if name not in z.namelist():
        _IN_PROCESS[subset] = []
        return []
    out = []
    for line in z.open(name):
        line = line.decode("utf8", "replace").strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    _IN_PROCESS[subset] = out
    return out


def _state(ex: dict, subset: str) -> str | None:
    if subset == "alphanli":
        obs1, obs2 = ex.get("obs1"), ex.get("obs2")
        h1, h2 = ex.get("hyp1"), ex.get("hyp2")
        if not (obs1 and obs2 and h1 and h2):
            return None
        return (f"Beginning: {obs1}\nEnding: {obs2}\n\n"
                f"Option 1: {h1}\nOption 2: {h2}")
    premise, hypothesis = ex.get("premise"), ex.get("hypothesis")
    if not premise or not hypothesis:
        return None
    return f"Premise: {premise}\nHypothesis: {hypothesis}"


def _distribution(r: dict, subset: str, n_options: int) -> list[float] | None:
    """The annotator distribution, derived from the published counts and cross-checked.

    Three things can go wrong with a soft target and none of them raise: too few
    annotators for a fraction to mean anything, a `label_dist` that does not agree with
    the `label_count` beside it, and `label_counter` -- the only field that names its
    labels -- disagreeing with the position the other two use. All three are checked here
    rather than assumed, and a row that fails any of them is dropped rather than flattened
    into something that merely looks like a distribution.

    The exact `count / total` is returned, not the published two-decimal `label_dist`:
    with 100 annotators per item they are the same number, and the derivation is the one
    that stays right if a later release annotates a different number of times.
    """
    counts = r.get("label_count")
    dist = r.get("label_dist")
    if not isinstance(counts, list) or len(counts) != n_options:
        return None
    if not isinstance(dist, list) or len(dist) != n_options:
        return None
    if any(not isinstance(c, (int, float)) or c < 0 for c in counts):
        return None
    total = sum(counts)
    if total < MIN_RATERS:
        return None
    out = [c / total for c in counts]
    if any(abs(p - float(d)) > _ROUNDING for p, d in zip(out, dist)):
        return None
    keys = COUNTER_KEYS["alphanli" if subset == "alphanli" else "nli"]
    counter = r.get("label_counter")
    if isinstance(counter, dict):
        named = [0.0] * n_options
        for key, value in counter.items():
            index = keys.get(str(key))
            if index is None:
                return None
            named[index] = value
        if any(abs(a - b) > 1e-9 for a, b in zip(named, counts)):
            return None
    return out


def _loader(subset: str):
    def load(n: int) -> Iterator[Example]:
        made = 0
        for r in rows(subset):
            if made >= n:
                return
            ex = r.get("example") or {}
            state = _state(ex, subset)
            if not state:
                continue
            if subset == "alphanli":
                # "1"/"2" is a bare numeral pair with nothing to read, which the
                # answerability gate rejects and rightly so. The options name the two
                # candidates the state actually lays out.
                options = ["option_1", "option_2"]
            else:
                options = NLI_OPTIONS
            dist = _distribution(r, subset, len(options))
            if dist is None:
                continue
            made += 1
            yield Example(
                task=f"chaosnli_{subset}",
                state=state[:4000],
                questions=[Question(
                    id="relation",
                    # alphaNLI's candidates are *middles*: the state gives the
                    # beginning and the ending, and Option 1 / Option 2 are the two
                    # hypotheses for what happened in between. "Which ending is more
                    # plausible?" asked about the one span both options share.
                    question=("Which option better explains how the beginning leads "
                              "to the ending?" if subset == "alphanli"
                              else "What is the relationship between the premise and "
                                   "the hypothesis?"),
                    options=list(options),
                    # 100 annotators, kept as the distribution rather than a majority
                    target=list(dist),
                )],
            )
    return load


def tasks() -> list[RealTask]:
    if not _ZIP.exists():
        return []
    out = []
    for subset in FILES:
        if not rows(subset):
            continue
        out.append(RealTask(
            row=25, name=f"chaosnli_{subset}", licence=LICENCE, url=URL,
            load=_loader(subset), soft=True,
            notes=f"ChaosNLI {subset}; soft target over 100 annotators; "
                  f"reserved whole-family to testreal"))
    return out
