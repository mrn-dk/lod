"""Source-table row 7 — where a passage sits in a document, from real converted PDFs.

Row 7 is "document structure". Until this module it was
`ml4pubmed/pubmed-classification-20k`: one sentence of a biomedical abstract, labelled
with the rhetorical section an indexer assigned it. Measured on an earlier build, that row
is 3,696 questions with a 0.334 majority baseline, a median state of **30 Qwen tokens**
with 69 % of states under the 40-token floor, and no document anywhere in sight — the
state is a single sentence. It teaches genre register, not layout. This module is the
layout half.

**The outline is the label.** A PDF carries an outline (bookmarks) only if whoever built
it put one there, and when it is there it is the author's own section map: title, nesting
depth, and the page each section starts on. Nothing is inferred from font sizes or
guessed from regexes, and a PDF without an outline is **dropped**, not reconstructed.
Measured over 25 CC-BY arXiv papers, 48 % carry an outline of three entries or more,
which is why the fetch over-collects.

**Two converters, because one converter is one damage signature.** `pymupdf4llm` emits
markdown and gets two-column maths wrong in its own way (`<sup>` soup, joined words);
poppler's `pdftotext -layout` emits fixed-width plain text and gets it wrong in a
different way (column bleed, running heads dropped into the body). A model trained on one
learns that fixture. `meta["converter"]` is on every example and the two engines get
separate task names, so a per-converter reading is a groupby, not a re-run.

    MD: 'and the RE of _R_ ( **_θ_**<sup>ˆ</sup> (full)PC<sup>)comparedto</sup>...'
    PT: 'and the RE of R(θ̂ PC ) compared to R(θ̂ PC ...'

**A passage is an interior page.** `section`, `next_section`, `prev_section` and `depth`
all take one, so the heading itself and its numbering are off the page: "3.1" would
answer `depth` on its own, and "3.2 follows 3.1" would answer `next_section` on its own.
`parent` is the exception -- it names the subsection in the question and shows the page
that subsection starts on, because the answer is a *different* heading that the page does
not contain.

 A page belongs to exactly one section when no outline
entry *starts* on it and some entry starts before it. That is decidable from the outline
alone, so the label is derived and never adjudicated. Pages before the first entry (title
page, abstract) and at or after the last entry (references, appendices, funding notes)
are excluded, which is what stops "Funding" being labelled with whatever section happened
to be open last.

**Section numbering is stripped from every title.** Left in, `next_section` is answered by
counting — "3.2 Foo" is followed by "3.3 Bar" and the passage is decoration. Stripped, the
ordering question is about what a paper actually does after its methods.

Six things make an example here, and code derives all six: the passage, the containing
section, the next and previous headings, the heading's depth, and its parent. No LLM is
involved at any point; there is nothing here for one to write.

Stage 1 is `scripts/fetch_data.py --rows 7`. `pymupdf` is imported lazily inside the functions that
need it, the way `gefs.py` imports xarray, so `pytest` and `--help` do not need
`uv sync --extra pdf`. poppler is a system package: `apt-get install -y poppler-utils`,
or point `LOD_PDFTOTEXT` at a binary and `LOD_PDF_LD_LIBRARY_PATH` at its libraries.
Without it the pdftotext half of the tasks is *omitted with a printed warning* rather than
silently emitted empty.
"""

from __future__ import annotations

import json
import os
import random
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterator

from lod.schema import Example, Question
from lod.corpus.services.sources.base import CACHE
from lod.corpus.services.sources.real.base import RealTask, disjoint_slice

PDF_DIR = CACHE / "pdfs"
MANIFEST = PDF_DIR / "manifest.jsonl"
DERIVED = PDF_DIR / "derived"          # outline plans and converted page text
URL = "https://arxiv.org/help/oa"

MD = "pymupdf4llm"
PT = "pdftotext"
CONVERTERS = (MD, PT)

MIN_TOC = 4              # an outline of three entries is a title page, not a section map
MIN_PAGES = 6
MAX_PAGES = 80           # a 300-page thesis converts slowly and adds one document
MAX_SPAN = 6             # a page this far past its heading is back matter, not a section
MIN_WORDS = 120          # a figure page converts to a screenful of spaces; count words
MIN_PROSE_WORDS = 80     # ... and a plot's axis labels count as words, so count prose
MAX_CHARS = 6000         # states are truncated here; the cap is on characters, not pages
MIN_OPTS, MAX_OPTS = 4, 6

# Sections that exist in every paper and carry no content signal, so they are not used as
# a label or as a distractor. `references` also marks where the document's body ends.
_BOILER = re.compile(
    r"^(references?|bibliography|acknowledg|funding|appendi|supplementary|"
    r"author contribution|conflict|declaration|data availability|abbreviation|"
    r"contents|index|notation)", re.I)
# Leading section numbering: "3 Foo", "3.1. Foo", "2.The Foo", "IV. Foo", "A.2 Foo",
# "(3) Foo". Stripped everywhere, so the ordering questions cannot be counted to.
# A letter or roman numeral needs its terminating punctuation, or "A Survey of X" loses
# its first word and "3D Convolutions" becomes "D Convolutions".
_NUMBER = re.compile(
    r"^\s*(?:[\(\[]?(?:\d+|[IVXLC]{1,6}|[A-Z])(?:[.\-]\d+)*[.\):\]]\s*"
    r"|\d+(?:[.\-]\d+)*\s+)")
_WS = re.compile(r"\s+")
# A bare capital as appendix numbering ("B Proof of ..."); applied only to a document that
# letters its entries, since "A Survey of X" is otherwise a sentence.
_LETTER = re.compile(r"^[A-Z]\s+(?=[A-Z0-9])")
_LETTER_HEAD = re.compile(r"^([A-Z])\s+[A-Z0-9]")
# Bookmarks that are not sections. One PDF in the cache bookmarks every figure and table
# caption ("Figure 4: Overview of mean user agreements ...") and nests its real sections
# *under* them; another bookmarks LaTeX labels ("thm:memorycapacity"). Neither outline is
# a section map, so neither document is labelled.
_CAPTION = re.compile(r"^\s*(?:fig(?:ure)?s?|tab(?:le)?s?|algorithm|listing)\.?\s*[A-Z]?\d", re.I)
_LABEL = re.compile(r"^[A-Za-z]+[:_][A-Za-z0-9:_.-]+$")
_ALNUM = re.compile(r"[^a-z0-9 ]+")
# A numbered reference list, which is what sits in the gap when a paper's outline names
# its appendices but not its bibliography. Such a page is inside no section.
_CITE = re.compile(r"^\s*(\[\d+\]|\d+\.\s+[A-Z][a-z]+,|[A-Z][a-z]+,\s+[A-Z]\.)")
# A word, for the purpose of telling prose from a plot's axis labels. "F=0.7", "15" and
# "-" are not words; `pdftotext -layout` puts a dozen of them on one very wide line.
_WORDISH = re.compile(r"[A-Za-z]{3,}")
# Back matter the outline does not name. A heading line that is only "References",
# "Acknowledgements", "Appendix B" ... means an un-bookmarked section starts on this page,
# so the page is not wholly inside the section that owns it.
_BACK_HEAD = re.compile(
    r"(?mi)^\s*(?:#+\s*)?[*_]*\s*(?:(?:[A-Z]|\d+)\.?\s+)?"
    r"(?:references|bibliography|acknowledge?ments?|appendix(?:\s+[A-Z0-9]+)?|appendices|"
    r"supplementary materials?)\s*[*_:.]*\s*$")
# ... and a bibliography that runs on without a heading. Venue and link markers, not
# years alone: a related-work page is dense in years too, and is not back matter.
_BIB = re.compile(r"\bProceedings\b|\bProc\.|\bConference\b|\bJournal\b|\bTransactions\b|"
                  r"\barXiv preprint\b|\barXiv:\d|\bdoi\b|https?://|\bpp\.|\bvol\.|"
                  r"\bCited on pages?\b|\bURL\b", re.I)
_YEAR = re.compile(r"\b(?:19|20)\d{2}[a-z]?\b")
_PICTURE = re.compile(r"<!--\s*Start of picture text\s*-->.*?<!--\s*End of picture text\s*-->",
                      re.S | re.I)
_BIB_ENTRY = re.compile(r"(?m)^\s*(?:-\s*)?\[\d{1,3}\]\s+[A-Z]")   # "- [12] Smith, ..."


# --------------------------------------------------------------------------- outline


def clean_title(raw: str, letters: bool = False) -> str:
    """A heading with its numbering, footnote marks and runs of space removed.

    `letters` says the document numbers its appendices with a bare capital -- "A Code",
    "B Proof of Proposition 3.3" -- which the general rule leaves alone because "A Survey
    of X" is a sentence. It is decided per document by `_letter_numbered`.
    """
    t = _WS.sub(" ", raw.replace("\u00a0", " ")).strip(" .·—-–*")
    prev = None
    while prev != t:                       # "A.2 3.1 Foo" happens; strip until stable
        prev = t
        t = _NUMBER.sub("", t).strip()
        if letters:
            t = _LETTER.sub("", t).strip()
    return t.strip(" .·—-–*")


def _letter_numbered(titles: list[str]) -> bool:
    """True when the outline letters its entries: both an "A X" and a "B Y" heading."""
    heads = {m.group(1) for t in titles if (m := _LETTER_HEAD.match(t.strip()))}
    return {"A", "B"} <= heads


def normalise(s: str) -> str:
    """Lower-case, alphanumeric-only, for the string-match solvability test."""
    return _WS.sub(" ", _ALNUM.sub(" ", s.lower())).strip()


def _manifest() -> list[dict]:
    if not MANIFEST.exists():
        return []
    rows = []
    seen = set()
    for line in MANIFEST.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["id"] in seen or not (PDF_DIR / r["path"]).exists():
            continue
        seen.add(r["id"])
        rows.append(r)
    rows.sort(key=lambda r: r["id"])       # deterministic order across rebuilds
    return rows


def outline_plan(pdf: Path) -> dict | None:
    """The document's section map, and the pages each question family can use.

    Returns None for a PDF this source cannot label: no outline, an outline that is not a
    section map, an outline too short to have an inside, or a body with no page that
    belongs to exactly one section.
    """
    import pymupdf                          # lazy: pytest must not need --extra pdf

    try:
        doc = pymupdf.open(pdf)
        toc = doc.get_toc(simple=True)      # [level, title, 1-based page]
        n_pages = doc.page_count
    except Exception:
        return None
    if not MIN_PAGES <= n_pages <= MAX_PAGES or len(toc) < MIN_TOC:
        return None
    return plan_from_toc(toc, n_pages)


def plan_from_toc(toc: list, n_pages: int) -> dict | None:
    """`outline_plan` without the PDF: the raw outline in, the plan out.

    **Ownership is decided on the raw outline, before any entry is filtered out.** An entry
    this module will not use -- a bookmark that starts lower-case, one longer than an
    option can be, a second "Introduction", a "Notation" section -- still *ends* the entry
    before it. Filtering first and assigning pages second handed that entry's pages to its
    predecessor: measured on the cache, 9 of 358 `section` labels named the wrong section
    ("Introduction" for a page of "Notation", "Rank-frequency dependence models" for a
    page of "Temperature in linguistics"), and the ordering families answered with a
    heading two steps away (17 of 277 `next_section`, 23 of 235 `prev_section`). A page
    whose raw owner is not usable is now skipped, and so is an ordering question whose
    raw neighbour is not.
    """
    titles_raw = [t for _, t, _ in toc]
    if any(_CAPTION.match(t) or _LABEL.match(t.strip()) for t in titles_raw):
        return None
    # A single top-level entry that opens the outline is the paper's title, with the real
    # sections nested one level down (and sometimes the author list beside them as a
    # "section"). Its levels are off by one, so `depth` and `parent` would be wrong.
    if toc[0][0] == 1 and sum(1 for lv, _, _ in toc if lv == 1) == 1:
        return None
    pages = [int(p) for _, _, p in toc if 1 <= p <= n_pages]
    if any(b < a for a, b in zip(pages, pages[1:])):
        return None                         # outline order is not page order: no owner
    letters = _letter_numbered(titles_raw)

    raw: list[dict] = []
    for level, t, page in toc:
        title = clean_title(t, letters)
        usable = (2 <= len(title) <= 90 and 1 <= page <= n_pages
                  # A heading that starts lower-case is a bookmark the producer split
                  # mid-phrase ("preserving homeomorphisms"): a nonsense option and a
                  # nonsense label. Real headings start with a capital or a digit.
                  and (title[0].isupper() or title[0].isdigit()))
        raw.append({"level": int(level), "title": title, "page": int(page),
                    "boiler": bool(_BOILER.match(title)), "kept": usable})
    # The body ends at the first back-matter heading; everything from there on is
    # references and appendices, whose pages must not be labelled with a real section.
    body_end = next((i for i, e in enumerate(raw) if e["boiler"] and i >= MIN_TOC // 2),
                    len(raw))
    counts: dict[str, int] = {}
    for e in raw:
        counts[normalise(e["title"])] = counts.get(normalise(e["title"]), 0) + 1
    entries: list[dict] = []
    seen: set[str] = set()
    for i, e in enumerate(raw):
        n = normalise(e["title"])
        # A duplicated title makes "which section" ill-posed; keep the first occurrence.
        if i >= body_end or e["boiler"] or not e["kept"] or n in seen:
            e["kept"] = False
            continue
        seen.add(n)
        e["dup"] = counts[n] > 1            # named in a question, it would be ambiguous
        entries.append(dict(e, ri=i))
    if len(entries) < MIN_TOC:
        return None

    kept_at = {e["ri"]: k for k, e in enumerate(entries)}
    starts = {e["page"] for e in raw if 1 <= e["page"] <= n_pages}
    first, last = entries[0]["page"], entries[-1]["page"]
    interior: list[dict] = []
    for page in range(first + 1, last):
        if page in starts:
            continue
        ri = max((i for i, e in enumerate(raw) if 1 <= e["page"] < page), default=None)
        if ri not in kept_at:
            continue
        owner = kept_at[ri]
        # A page many pages past its heading, with no heading in between, is back matter
        # the outline forgot to name -- a bibliography, a long appendix -- not a section.
        if page - entries[owner]["page"] > MAX_SPAN:
            continue
        interior.append({"page": page, "entry": owner})
    if not interior:
        return None
    return {"pages": n_pages, "entries": entries, "interior": interior,
            "raw": [{k: e[k] for k in ("level", "title", "page", "kept")} for e in raw],
            "heading_pages": sorted({e["page"] for e in entries})}


def _raw(plan: dict) -> list[dict]:
    """The unfiltered outline; a hand-built plan without one is its own raw outline."""
    return plan.get("raw") or [dict(e, kept=True) for e in plan["entries"]]


def _kept(plan: dict, ri: int) -> int | None:
    """The `entries` index of raw outline entry `ri`, or None if it was filtered out."""
    for k, e in enumerate(plan["entries"]):
        if e.get("ri", k) == ri:
            return k
    return None


def neighbour(plan: dict, idx: int, step: int) -> int | None:
    """The entry immediately before (-1) or after (+1) entry `idx` in the *raw* outline,
    or None when that entry is not one this module offers."""
    ri = plan["entries"][idx].get("ri", idx) + step
    if not 0 <= ri < len(_raw(plan)):
        return None
    return _kept(plan, ri)


def ancestors(plan: dict, idx: int) -> list[int]:
    """Raw-outline indices of the entries enclosing entry `idx`, innermost first."""
    raw = _raw(plan)
    ri = plan["entries"][idx].get("ri", idx)
    level, out = raw[ri]["level"], []
    for j in range(ri - 1, -1, -1):
        if raw[j]["level"] < level:
            out.append(j)
            level = raw[j]["level"]
    return out


# ------------------------------------------------------------------------ conversion


def _pdftotext_bin() -> str | None:
    """The poppler binary, from the environment or the PATH. None means not installed."""
    env = os.environ.get("LOD_PDFTOTEXT")
    if env and Path(env).exists():
        return env
    return shutil.which("pdftotext")


def _convert(pdf: Path, pages: list[int], converter: str) -> dict[int, str]:
    """One engine's text for the given 1-based pages. Both engines, one signature."""
    if converter == MD:
        import pymupdf
        import pymupdf4llm

        doc = pymupdf.open(pdf)
        out = {}
        for p in pages:
            try:
                out[p] = pymupdf4llm.to_markdown(doc, pages=[p - 1], show_progress=False)
            except Exception:
                continue
        return out
    binary = _pdftotext_bin()
    if binary is None:
        raise RuntimeError("pdftotext not installed: apt-get install -y poppler-utils, "
                           "or set LOD_PDFTOTEXT")
    env = dict(os.environ)
    lib = os.environ.get("LOD_PDF_LD_LIBRARY_PATH")
    if lib:
        env["LD_LIBRARY_PATH"] = lib
    out = {}
    for p in pages:
        try:
            r = subprocess.run([binary, "-layout", "-f", str(p), "-l", str(p),
                                str(pdf), "-"],
                               capture_output=True, env=env, timeout=120)
        except Exception:
            continue
        if r.returncode == 0:
            out[p] = r.stdout.decode("utf-8", "replace")
    return out


_MEM: dict[str, dict] = {}


def document(rec: dict, converter: str) -> dict | None:
    """One document's plan plus one engine's page text.

    Only the *converted text* is cached on disk. The plan is recomputed every time, in
    milliseconds, because it is the part this module keeps changing; caching it meant
    every tightening of the outline rules threw away an hour of conversion.
    """
    key = f"{rec['id']}::{converter}"
    if key in _MEM:
        return _MEM[key]
    plan = outline_plan(PDF_DIR / rec["path"])
    if plan is None:
        _MEM[key] = None
        return None
    DERIVED.mkdir(parents=True, exist_ok=True)
    cached = DERIVED / f"{rec['id'].replace('/', '_')}.{converter}.json"
    text: dict[str, str] = {}
    if cached.exists():
        try:
            text = json.loads(cached.read_text())
        except Exception:
            text = {}
    want = sorted({d["page"] for d in plan["interior"]} | set(plan["heading_pages"]))
    missing = [p for p in want if str(p) not in text]
    if missing:
        text.update({str(k): v for k, v in
                     _convert(PDF_DIR / rec["path"], missing, converter).items()})
        cached.write_text(json.dumps(text))
    plan["text"] = text
    plan["id"] = rec["id"]
    plan["licence"] = rec.get("licence", "")
    plan["url"] = rec.get("url", "")
    _MEM[key] = plan
    return plan


def is_reference_page(text: str) -> bool:
    """A page that is mostly a bibliography, whatever the outline says owns it."""
    lines = [l for l in text.splitlines() if l.strip()]
    if len(lines) < 8:
        return False
    return sum(1 for l in lines if _CITE.match(l)) >= max(4, 0.2 * len(lines))


def is_back_matter(text: str) -> bool:
    """A page that is, or starts, a bibliography or other back matter.

    `is_reference_page` counts citation-shaped *lines*, and `pymupdf4llm` writes a whole
    bibliography as a handful of very long ones, so it passed: measured on the cache, 71
    of 1,275 states (5.6 %) were reference lists or acknowledgements labelled with the
    last body section -- "Conclusion" 45 times -- because the outline did not bookmark
    the bibliography. Every page this flags was read and is back matter; no related-work
    page is among them.
    """
    if _BACK_HEAD.search(text) or len(_BIB_ENTRY.findall(text)) >= 4:
        return True
    venues, years = len(_BIB.findall(text)), len(_YEAR.findall(text))
    # the last, short page of a bibliography has few years but is nothing else
    return (venues >= 3 and years >= 10) or \
        (venues >= 6 and venues + years >= len(text.split()) / 25)


def prose_words(text: str) -> int:
    """Words that sit on a line with enough other words to be a sentence.

    A plot converts to hundreds of axis labels and legend fragments, which clears a plain
    word count -- `pdftotext -layout` spreads "25", "F=0.7", "Phase lag [rad]" across one
    very wide line, and a naive split calls that a dozen words. Counting prose *lines*
    instead looked equivalent and was not: `pymupdf4llm` puts a whole paragraph on one
    line, so a line threshold threw away 46 % of the markdown side and 2 % of the
    plain-text one.
    """
    # `pymupdf4llm` fences a figure's own text -- axis labels, legends -- as one
    # `<br>`-joined line between picture-text markers, which reads as a long sentence.
    text = _PICTURE.sub(" ", text)
    total = 0
    for line in text.splitlines():
        words = _WORDISH.findall(line)
        if len(words) >= 8:
            total += len(words)
    return total


def passage(plan: dict, page: int) -> str | None:
    """The converted page as a state, or None if the conversion produced too little."""
    raw = plan.get("text", {}).get(str(page), "")
    body = raw.strip()
    # Whitespace is not content. `pdftotext -layout` renders a full-page figure as
    # hundreds of spaces, which sailed past a character-count floor and shipped a state
    # with nothing in it to read.
    if len(body.split()) < MIN_WORDS or is_reference_page(body) or is_back_matter(body):
        return None
    # The floors are checked on what the model will actually see. `pdftotext -layout`
    # pads a page with runs of spaces, so a 6,000-character cut can hold only a table's
    # header and a screen of blanks: 18 of 1,186 states had fewer than 80 prose words
    # after the cut, 9 of them none at all, while the whole page had passed.
    state = body[:MAX_CHARS]
    if prose_words(body) < MIN_PROSE_WORDS or prose_words(state) < MIN_PROSE_WORDS or \
            len(state.split()) < MIN_WORDS:
        return None
    return state


# --------------------------------------------------------------------------- options


def options_for(titles: list[str], gold: str, rng: random.Random,
                exclude: tuple[str, ...] = (), state: str | None = None) -> list[str] | None:
    """A sorted option set of 4-6 titles from the same document, always containing gold.

    Distractors come from the *same* document, which is the only honest distractor set
    here: a section list from another paper would be separable by topic alone.

    Three things are kept out. `exclude` is the heading the question itself names -- an
    ordering question that offers its own anchor back as an option gives away a free
    elimination and reads as a mistake. No two options may contain one another:
    "Solution strategy" against "Solution strategy and design of the numerical
    framework" is not a question about document structure, it is a coin flip.

    And, given `state`, no distractor may appear verbatim in the passage. `leaks()` drops
    an example whose *gold* is readable off the state, so before this the two sides of
    the option set were treated asymmetrically, and the asymmetry ran the wrong way:
    measured on an earlier build, the gold option appeared in its own state 0.0 % of the
    time across all ten tasks (by construction) while some distractor did in 9.9 %-34.3 %.
    "An option is written on the page" is the strongest surface cue an option-scoring
    model has, and the build had made it a *guarantee of being wrong*. On domain 12's
    166 testreal questions, the 28 carrying a visible distractor scored 0.1786 -- chance
    is 0.1696 -- against 0.3043 on the rest, and the model picked one of the visible
    distractors on 32.1 % of them against a 16.7 % per-option rate. That is not a harder
    example, it is an anti-correlated one, so the filter is symmetric now: every option
    offered is absent from the passage.
    """
    banned = {normalise(x) for x in exclude} | {normalise(gold)}
    gold_n = normalise(gold)
    seen_state = normalise(state) if state is not None else None
    pool: list[str] = []
    for t in titles:
        n = normalise(t)
        if n in banned or n in gold_n or gold_n in n:
            continue
        if seen_state is not None and n and n in seen_state:
            continue
        if any(n in normalise(p) or normalise(p) in n for p in pool):
            continue
        pool.append(t)
    if len(pool) < MIN_OPTS - 1:
        return None
    k = min(MAX_OPTS, max(MIN_OPTS, 1 + len(pool))) - 1
    picked = rng.sample(pool, min(k, len(pool)))
    return sorted(set(picked) | {gold})


def _q(qid: str, text: str, options: list[str], gold: str, meta: dict,
       instructions: str | None = None) -> Question | None:
    if gold not in options or not (MIN_OPTS <= len(options) <= MAX_OPTS):
        return None
    return Question(id=qid, question=text, options=options,
                    target=options.index(gold), meta=meta, instructions=instructions)


def leaks(state: str, options: list[str], gold: str) -> bool:
    """True when the gold option is readable off the state by string match.

    Row 36 shipped at 100 % solvable this way. Distractors are kept off the page by
    `options_for(state=...)`, so together no option offered is ever written on the page.
    """
    return normalise(gold) in normalise(state)


# ----------------------------------------------------------------------- the families


DEPTHS = ["a top-level section", "a second-level subsection",
          "a third-level subsection", "a fourth-level subsection"]

_QUESTIONS = {
    "section": "Which section of this document does the passage above come from?",
    "next_section": 'Which heading comes immediately after "{t}" in this document\'s '
                    "section map?",
    "prev_section": 'Which heading comes immediately before "{t}" in this document\'s '
                    "section map?",
    "parent": 'The passage above is where the subsection "{t}" begins. Which '
              "top-level section of this document contains it?",
    "depth": 'The passage above comes from the section headed "{t}". How deep does '
             "that heading sit in this document's section map?",
}
# Phrasing-bank ids, one per template; `lod/assets/phrasing_specs/d12.json` describes each.
QUESTION_IDS = {f: f"d12.{f}" for f in _QUESTIONS}
_BANK = None


def _phrase(family: str, key: str, **slots: str) -> str:
    """The family's question, in a wording chosen by the phrasing bank for this record."""
    global _BANK
    if _BANK is None:
        from lod.phrasings import PhrasingBank
        _BANK = PhrasingBank.load()
    return _BANK.pick(QUESTION_IDS[family], _QUESTIONS[family], key).format(**slots)

_INSTRUCTIONS = (
    "The passage is the raw output of a PDF-to-text converter and carries its damage: "
    "broken columns, stray running heads, mangled maths. Answer from the document's "
    "structure, not from its formatting being clean."
)


def _examples(family: str, converter: str, records: list[dict],
              limit: int) -> Iterator[Example]:
    task = f"pdfdocs_{family}_{converter}"
    rng = random.Random(f"{family}:{converter}")
    made = 0
    for rec in records:
        if made >= limit:
            return
        plan = document(rec, converter)
        if plan is None:
            continue
        entries = plan["entries"]
        raw = _raw(plan)
        titles = [e["title"] for e in entries]
        tops = [e["title"] for e in entries if e["level"] == 1]
        base_meta = {"converter": converter, "doc": plan["id"],
                     "licence": plan.get("licence", ""), "source_url": plan.get("url", "")}

        if family in ("section", "next_section", "prev_section", "depth"):
            spots = [(d["page"], d["entry"]) for d in plan["interior"]]
        else:
            spots = [(e["page"], i) for i, e in enumerate(entries)]

        for page, idx in spots:
            if made >= limit:
                return
            entry = entries[idx]
            # A question that names a heading the outline carries twice ("Experiments"
            # in each of two studies) does not say which one it means.
            if family != "section" and entry.get("dup"):
                continue
            state = passage(plan, page)
            if state is None:
                continue
            meta = dict(base_meta, page=page, of_pages=plan["pages"],
                        heading=entry["title"], level=entry["level"])
            key = f"{plan['id']}:{page}:{family}:{converter}"

            if family == "section":
                # The page of subsection "Estimator" is also a page of its enclosing
                # "Method", so offering both gives the question two right answers. It did
                # in 76 of 358 `section` questions; the enclosing sections are held out.
                enclosing = tuple(raw[j]["title"] for j in ancestors(plan, idx))
                gold = entry["title"]
                opts = options_for(titles, gold, rng, exclude=enclosing, state=state)
                text = _phrase(family, key)
            elif family == "next_section":
                nb = neighbour(plan, idx, +1)
                if nb is None:
                    continue
                gold = entries[nb]["title"]
                opts = options_for(titles, gold, rng, exclude=(entry["title"],),
                                   state=state)
                text = _phrase(family, key, t=entry["title"])
            elif family == "prev_section":
                nb = neighbour(plan, idx, -1)
                if nb is None:
                    continue
                gold = entries[nb]["title"]
                opts = options_for(titles, gold, rng, exclude=(entry["title"],),
                                   state=state)
                text = _phrase(family, key, t=entry["title"])
            elif family == "parent":
                if entry["level"] < 2:
                    continue
                top = next((j for j in ancestors(plan, idx) if raw[j]["level"] == 1), None)
                parent = None if top is None else _kept(plan, top)
                if parent is None or len(tops) < MIN_OPTS:
                    continue
                gold = entries[parent]["title"]
                opts = options_for(tops, gold, rng, exclude=(entry["title"],), state=state)
                text = _phrase(family, key, t=entry["title"])
            else:                                        # depth
                if not 1 <= entry["level"] <= len(DEPTHS):
                    continue
                gold, opts = DEPTHS[entry["level"] - 1], list(DEPTHS)
                text = _phrase(family, key, t=entry["title"])

            if not opts:
                continue
            # The question names the heading the passage starts at, so `depth` and
            # `parent` legitimately show it; what must never be free is the *answer*.
            if leaks(state, opts, gold):
                continue
            q = _q(family, text, opts, gold, meta, _INSTRUCTIONS)
            if q is None:
                continue
            yield Example(state=state, questions=[q], task=task)
            made += 1


def _loader(family: str, converter: str, task_index: int, n_tasks: int):
    def load(n: int) -> list[Example]:
        recs = disjoint_slice(_manifest(), task_index, n_tasks)
        return list(_examples(family, converter, recs, n))
    return load


FAMILIES = ("section", "next_section", "prev_section", "parent", "depth")

# Left to the name hash, all ten tasks land in `train` bar one, and that one is `depth`
# -- the weakest family and only on one converter, so nothing could be read per converter
# at eval. The split is declared instead, and declared the way row 13's lichess
# split is: hold out a *structure*, and pair each held-out family against a
# trained one so the transfer question is sharp.
#   prev_section (held out) against next_section (trained) -- the same walk of the
#     outline in the other direction.
#   parent (held out) against depth (trained) -- the same nesting, named rather than
#     counted.
# Both converters go to the same split as their family, so every split carries both
# damage signatures and a per-converter reading survives.
FORCE_SPLIT = {"section": "train", "next_section": "train", "depth": "train",
               "parent": "devreal", "prev_section": "testreal"}

# ... and one override, because the split above left the row's own thesis unmeasured.
# An earlier build sent 166 questions of this row to `testreal` and every one of them was
# `prev_section`, the family whose answer is a property of the *outline* rather than of
# the passage: the previous heading is not on the page, and `leaks()` guarantees it is
# not. Measured over the built corpus, ranking the options by how much vocabulary they
# share with the passage puts the gold at mean normalised rank 0.54 for `prev_section`,
# `next_section` and `parent` -- *worse* than random -- against 0.44 for `section`, and
# a lookup keyed on the question text alone reproduces 0.967-1.000 of `prev_section`'s
# labels, so its state is very nearly redundant. `section` is the one family whose label
# the passage determines, and it was trained-only, so the row's headline number said
# nothing about reading a converted PDF.
#
# Holding `section` out whole would leave it untrained instead, which is worse. It is
# held out by *converter* instead: trained on the markdown damage signature, evaluated
# on the fixed-width one. That is the transfer this row was built to ask about, and the
# two converters already draw from disjoint documents, so no state crosses the split.
FORCE_SPLIT_CONVERTER = {("section", PT): "testreal"}


def split_for(family: str, converter: str) -> str:
    return FORCE_SPLIT_CONVERTER.get((family, converter), FORCE_SPLIT[family])


def family_for(family: str, converter: str) -> str:
    """The held-out unit (`RealTask.family`). A question family is held out whole, both
    converters together, so its two tasks are one family -- except where it is held out
    by converter (`FORCE_SPLIT_CONVERTER`), and then that converter's task is its own
    family, distinct from the family that trains on the other converter."""
    if (family, converter) in FORCE_SPLIT_CONVERTER:
        return f"pdfdocs_{family}_{converter}"
    return f"pdfdocs_{family}"

_NOTES = {
    "section": "state is one interior page of converted PDF; the label is the outline "
               "entry that owns that page",
    "next_section": "the outline's reading order, with section numbering stripped so "
                    "the answer cannot be counted to",
    "prev_section": "as next_section, backwards",
    "parent": "which top-level section contains this subsection, from the outline's "
              "nesting",
    "depth": "outline nesting depth of the heading the page starts at; ordinal",
}


def tasks() -> list[RealTask]:
    """Five question families per converter, over disjoint documents.

    The families ask about the same pages, so sharing a document pool between them would
    route one task's states into `train` and another's into `testreal` and the dedup
    would then correctly delete the whole eval side (`base.disjoint_slice`). Each task
    therefore gets its own slice of the manifest.
    """
    recs = _manifest()
    if not recs:
        return []
    converters = [MD]
    if _pdftotext_bin() is not None:
        converters.append(PT)
    else:
        print("  pdfdocs: pdftotext not found, shipping one converter only "
              "(apt-get install -y poppler-utils, or set LOD_PDFTOTEXT)")
    pairs = [(f, c) for c in converters for f in FAMILIES]
    out = []
    for i, (family, converter) in enumerate(pairs):
        out.append(RealTask(
            row=7, name=f"pdfdocs_{family}_{converter}",
            licence="CC-BY-4.0 / CC-BY-SA-4.0 / CC0-1.0 (per document, in meta)",
            url=URL, load=_loader(family, converter, i, len(pairs)),
            ordinal=(family == "depth"), force_split=split_for(family, converter),
            family=family_for(family, converter),
            # every family but `depth` draws its options from the document at hand, so
            # the option set changes example to example
            per_example_options=(family != "depth"),
            notes=f"{_NOTES[family]}; converter {converter}"))
    return out
