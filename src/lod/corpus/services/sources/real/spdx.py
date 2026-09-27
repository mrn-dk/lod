"""Source-table row 15 — SPDX licence identification.

State is an excerpt of a licence's text; the question is which licence it came from,
over the 60-way option set. Labels are the SPDX licence list itself, which is a real
system's classification, so this is a real source.

Two things here are design decisions rather than transcription, and both change what the
task measures:

**The title is stripped.** Licence texts open with their own name -- "MIT License",
"Apache License\\nVersion 2.0" -- and the option list is licence ids. Left in, the task is
string matching against the state, solvable without reading anything, and it would report
as high accuracy while teaching the model nothing about deciding. Excerpts are drawn from
past the opening lines and any surviving mention of the licence's own name or id is
removed. `leakage_rate` measures what fraction of built examples still contain their own
answer, so the claim is checked rather than asserted.

**Several excerpts per licence.** There are 740 licences and one text each, so one example
per licence would be a 60-option question with 60 examples. Windows are drawn at several
offsets, which is honest -- each window is genuinely a different piece of evidence for the
same label -- but it does mean states within this task share vocabulary. That is contained
because the split is by task name, so every SPDX row lands in one split and never
straddles train and testreal.

"Top 60" is not defined by SPDX, which publishes no usage ranking. OSI-approved and
non-deprecated is the closest available proxy for "commonly used", ordered by SPDX's own
`referenceNumber` for determinism.
"""

from __future__ import annotations

import re
from typing import Iterator

from lod.schema import Example, Question
from lod.corpus.services.sources.base import fetch_json
from lod.corpus.services.sources.real.base import RealTask

BASE = "https://raw.githubusercontent.com/spdx/license-list-data/main/json"
LICENCE = "CC0-1.0 (SPDX licence list)"
URL = "https://github.com/spdx/license-list-data"

N_LICENCES = 60
WINDOW_WORDS = 130
SKIP_OPENING_WORDS = 60   # past the title block, which names the licence
EXCERPTS_PER_LICENCE = 12


def _norm(text: str) -> str:
    """Lower-case word sequence, space-padded so `in` is a whole-word containment test."""
    return " " + " ".join(re.findall(r"[a-z0-9]+", text.lower())) + " "


def licence_ids(k: int = N_LICENCES) -> list[str]:
    """The first `k` OSI-approved ids whose licence texts are pairwise distinct.

    SPDX publishes some licences twice under two ids with *the same text*: `GPL-3.0-only`
    and `GPL-3.0-or-later`, `GPL-2.0-only` and `GPL-2.0-or-later`, `MPL-2.0` and
    `MPL-2.0-no-copyleft-exception`. The difference is in how a project *applies* the
    licence, not in the licence text, so an excerpt of one is an excerpt of the other and
    "which licence is this text from" has two right answers. All three pairs were in the
    60-way set, and 198 of 1,956 built examples (10.1 %) asked for one half of a pair
    (domain-10 audit). The later id of each pair
    is skipped and the list filled from the next distinct licence.
    """
    d = fetch_json(f"{BASE}/licenses.json", timeout=90)
    osi = [l for l in d["licenses"]
           if l.get("isOsiApproved") and not l.get("isDeprecatedLicenseId")]
    osi.sort(key=lambda l: l.get("referenceNumber", 1 << 30))
    out: list[str] = []
    seen: set[str] = set()
    for l in osi:
        if len(out) >= k:
            break
        try:
            key = _norm(licence_text(l["licenseId"]))
        except Exception:
            continue
        if not key.strip() or key in seen:
            continue
        seen.add(key)
        out.append(l["licenseId"])
    return out


def licence_text(lid: str) -> str:
    d = fetch_json(f"{BASE}/details/{lid}.json", timeout=90)
    return d.get("licenseText") or ""


# Words that appear in licence *names* but carry no identity on their own: scrubbing
# them would blank half of every licence body and hide nothing. Anything outside this
# list is treated as a distinguishing part of the name ("Nethack", "Multics",
# "Frameworx", "Apache") and is redacted even when it stands alone.
_GENERIC_NAME_WORDS = frozenset("""
licence license licenses licences public general open source free software the of and or
for a an version v1 v2 v3 and/or agreement terms notice attribution common commons
community standard limited modified original revised clause clauses variant only later
plus with exception exceptions international unported universal derivative work works
new no
""".split())

# A stem followed by an optional version tail: "CERN-OHL", "CERN-OHL-S-2.0", "MulanPSL2",
# "gpl-3.0", "python2.7". The stem must stand on its own -- "GPL" inside "LGPL" is a
# *different* licence and redacting it would destroy a distractor rather than an answer.
_VERSION_TAIL = r"(?:[-_. ]?v?\d+(?:[-_.]\d+)*)?"


def _stems(lid: str, name: str) -> set[str]:
    """Every string that names this licence on its own: id prefixes and name words.

    The scrub used to cover the full id, the full name and every multi-word *run* of the
    name. Two families slipped through it, both measured on an earlier build:

    * **Id prefixes.** SPDX ids are an abbreviation plus a version, and prose and URLs use
      the abbreviation alone: "CERN-OHL" and "CERN" for `CERN-OHL-S-2.0`, "OCLC" for
      `OCLC-2.0`, "UPL" for `UPL-1.0`, "gpl-3.0" for `GPL-3.0-only`. 135 of 1,956 built
      states (**6.90 %**) contained a prefix form of their own answer.
    * **Single words of the name.** Runs started at length 2, so the one word that *is*
      the brand survived alone: "Nethack" for `NGPL`, "Multics" for `Multics`,
      "Frameworx" for `Frameworx-1.0`, "Python" for `Python-2.0.1`.

    Both are the same failure the module docstring already claims to have fixed, so they
    are fixed here rather than argued about, and `leakage_rate` is widened to see them.
    """
    stems: set[str] = set()
    parts = [p for p in lid.split("-") if p]
    for j in range(1, len(parts) + 1):
        head = parts[:j]
        # a version component is not a name; stop the prefix at the first one
        if j > 1 and re.fullmatch(r"v?\d+(?:\.\d+)*", head[-1]):
            break
        for sep in ("-", " ", "", ".", "_"):
            form = sep.join(head)
            if len(form) >= 3 and re.search(r"[A-Za-z]", form):
                stems.add(form)
    for word in name.split():
        w = word.strip("(),.")
        # >= 3, not 4: "GNU" (the GPL family) and "Sun" (`SPL-1.0`, "Sun Microsystems")
        # are the brand word of their names and survived alone -- 28 of 1,956 states
        # (1.43 %) in a domain-10 audit.
        if len(w) >= 3 and w.lower() not in _GENERIC_NAME_WORDS and re.search(r"[A-Za-z]", w):
            stems.add(w)
    return stems


def _scrub(text: str, lid: str, name: str) -> str:
    """Remove the licence's own id and name, so the answer is not sitting in the state.

    Prose sometimes names the licence by a fragment of its published name rather than the
    full string -- "the EU DataGrid" for the published name "EU DataGrid Software
    License", and the licence body's own credit URL as "eu-datagrid.org" -- which a
    single full-string token missed and measurably leaked (66/3903 SPDX examples, found
    by `leakage_rate`). Every contiguous multi-word run of the name (length >= 2) is
    scrubbed as well, joined both by a space and by a hyphen (URLs and identifiers favour
    hyphens where prose favours spaces), so a partial or URL-cased mention is caught the
    same as the full one.

    `_stems` adds the two families that survived that fix: an abbreviation prefix of the
    id ("CERN-OHL" for `CERN-OHL-S-2.0`) and a single distinguishing word of the name
    ("Nethack" for `NGPL`). A stem is matched on its own, with an optional version tail,
    so "GPL" inside "LGPL" is left alone -- that is a distractor, not the answer.
    """
    stems = _stems(lid, name)
    # A URL or domain naming the licence goes whole, before the token passes below cut
    # it into pieces that no longer match: "blueoakcouncil.org" for `BlueOak-1.0.0`,
    # "pythonlabs.com" for `Python-2.0.1`, "www.gnu.org" for the GPL family -- 41 of
    # 1,956 states (2.10 %) carried one (`spdx.answer_in_url`).
    low_stems = [st.lower() for st in stems if len(st) >= 3]
    out = _URL.sub(lambda m: "[REDACTED]" if any(st in m.group(0).lower() for st in low_stems)
                   else m.group(0), text)
    tokens = set(filter(None, {lid, name, name.replace(" License", ""),
                               lid.replace("-", " "), lid.replace("-", "."),
                               lid.replace("-", "")}))
    words = name.split()
    for i in range(len(words)):
        for j in range(i + 2, len(words) + 1):
            run = words[i:j]
            tokens.add(" ".join(run))
            tokens.add("-".join(run))
    # longest first, then alphabetical: set order varies with PYTHONHASHSEED, and two
    # overlapping runs redacted in a different order give a different state, so the
    # same build produced different SPDX states run to run (domain-10 audit)
    for token in sorted(filter(None, tokens), key=lambda t: (-len(t), t)):
        out = re.sub(re.escape(token), "[REDACTED]", out, flags=re.IGNORECASE)
    for stem in sorted(stems, key=lambda t: (-len(t), t)):
        out = re.sub(r"(?<![A-Za-z0-9])" + re.escape(stem) + _VERSION_TAIL
                     + r"(?![A-Za-z0-9])", "[REDACTED]", out, flags=re.IGNORECASE)
    # The version is part of the title: "[REDACTED] Version 1.1" is APSL-1.1 rather than
    # APSL-1.0 once the name is gone. Only a version introduced by "version"/"v" or by a
    # redaction is taken; a bare "1.1" in a section number is left alone.
    for ver in _versions(lid, name):
        out = re.sub(r"(?:\[REDACTED\]\s*,?\s*(?:version\s*|v\.?\s*)?|\bversion\s*|\bv\.?\s*)"
                     + re.escape(ver) + r"(?![0-9]|\.[0-9])", "[REDACTED]", out,
                     flags=re.IGNORECASE)
    out = re.sub(r"\[REDACTED\](?:\s*\[REDACTED\])+", "[REDACTED]", out)
    return out


_URL = re.compile(r"(?:https?://|www\.)[^\s<>()\"']+|"
                  r"\b[\w-]+(?:\.[\w-]+)*\.(?:org|com|net|edu|gov|io|info)\b[^\s<>()\"']*",
                  re.IGNORECASE)


def _versions(lid: str, name: str) -> list[str]:
    """Version strings of this licence, longest first: "1.0.0", "1.0", "1" for BlueOak."""
    found = set(re.findall(r"(?<![A-Za-z0-9])v?(\d+(?:\.\d+)*)", lid + " " + name))
    out = set()
    for v in found:
        out.add(v)
        while v.endswith(".0"):
            v = v[:-2]
            out.add(v)
    return sorted(out, key=len, reverse=True)


def excerpts(text: str, lid: str, name: str, n: int = EXCERPTS_PER_LICENCE) -> list[str]:
    words = text.split()
    body = words[SKIP_OPENING_WORDS:]
    if len(body) < WINDOW_WORDS:
        body = words
    if len(body) < 30:
        return []
    # Windows never overlap. The stride used to be (len - WINDOW) / n, so a 1,654-word
    # licence cut 33 windows 44 words apart and MIT's 170 words gave 33 windows 1 word
    # apart: 1,362 of 1,956 states (69.6 %) shared >= 50 % of their 10-word shingles with
    # another state of the same licence, and the val carve-out is drawn from this same
    # task. A short licence now yields one or two excerpts, which is all the evidence
    # it has.
    out, step = [], max(WINDOW_WORDS, (len(body) - WINDOW_WORDS) // max(1, n))
    for i in range(n):
        start = i * step
        if start + 30 > len(body):
            break
        chunk = " ".join(body[start : start + WINDOW_WORDS])
        out.append(_scrub(chunk, lid, name))
    return out


def leakage_rate(examples: list[Example], ids: list[str],
                 names: dict[str, str] | None = None) -> float:
    """Fraction of examples whose state still contains their own answer.

    The scrub is a regex over names that vary in punctuation and casing, so it can miss.
    This is the check that the task is not secretly string matching.

    It used to test the *option string* alone, which is the one form the scrub could not
    miss, so it reported 0.000 on an earlier build while 135 of 1,956 states (6.90 %) named
    their own answer by an id prefix -- "OCLC" for `OCLC-2.0`, "CERN-OHL" for
    `CERN-OHL-S-2.0`. A metric that can only see the case that never happens is not a
    check, so it now tests every form `_scrub` removes: pass `names` (licence id ->
    published name) to include the name's distinguishing words as well.
    """
    if not examples:
        return 0.0
    names = names or {}
    leaked = 0
    for e in examples:
        q = e.questions[0]
        answer = q.options[q.target] if isinstance(q.target, int) else None
        if not answer:
            continue
        if answer.lower() in e.state.lower():
            leaked += 1
            continue
        for stem in _stems(answer, names.get(answer, "")):
            if re.search(r"(?<![A-Za-z0-9])" + re.escape(stem) + _VERSION_TAIL
                         + r"(?![A-Za-z0-9])", e.state, flags=re.IGNORECASE):
                leaked += 1
                break
    return leaked / len(examples)


def found_elsewhere(chunk: str, own: str, norms: dict[str, str]) -> str | None:
    """Another licence whose text contains every unredacted span of this excerpt.

    Sibling licences share most of their text -- APSL 1.0 and 1.1, LPL 1.0 and 1.02,
    SPL and CUA-OPL, MIT and MIT-0 -- so a window cut from the shared part is a window of
    both, and naming one of them is a guess. 482 of 1,956 built states (24.6 %) were
    wholly contained in another option's licence (`spdx.excerpt_contained_in_other_option`).
    Such a window is dropped rather than labelled; a window that reaches a clause where
    the siblings differ is kept, and that clause is the point of the task.
    """
    segs = [_norm(s) for s in chunk.split("[REDACTED]")]
    segs = [s for s in segs if s.strip()]
    if not segs:
        return own
    for other, text in norms.items():
        if other != own and all(s in text for s in segs):
            return other
    return None


TEMPLATE = "Which licence is this text from?"
_BANK = None


def _question(key: str) -> str:
    """The question, through the phrasing bank (unchanged while the bank is empty)."""
    global _BANK
    if _BANK is None:
        from lod.phrasings import PhrasingBank
        _BANK = PhrasingBank.load()
    return _BANK.pick("d10.spdx_licence", TEMPLATE, key)


def published_names() -> dict[str, str]:
    """Licence id -> SPDX's own published name, the option criteria for this task."""
    d = fetch_json(f"{BASE}/licenses.json", timeout=90)
    return {l["licenseId"]: l.get("name", "") for l in d["licenses"]}


def _load(n: int) -> Iterator[Example]:
    by_id = published_names()
    ids = licence_ids()
    texts = {lid: licence_text(lid) for lid in ids}
    norms = {lid: _norm(t) for lid, t in texts.items()}
    made = 0
    per = max(1, n // max(1, len(ids)))
    for lid in ids:
        if made >= n:
            return
        for chunk in excerpts(texts[lid], lid, by_id.get(lid, ""), per):
            if made >= n:
                return
            if found_elsewhere(chunk, lid, norms):
                continue
            made += 1
            yield Example(
                task="spdx_licence",
                state=chunk,
                questions=[Question(
                    id="licence",
                    question=_question(chunk),
                    options=list(ids),
                    target=ids.index(lid),
                )],
            )


def tasks() -> list[RealTask]:
    return [RealTask(row=15, name="spdx_licence", licence=LICENCE, url=URL,
                     load=_load, ordinal=False,
                     notes=f"{N_LICENCES} OSI-approved licences; title stripped from the "
                           f"excerpt so the answer is not in the state")]
