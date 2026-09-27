"""Open-licence arXiv PDFs, for the document-structure source (`real/pdfdocs.py`).

The reader needs PDFs whose publisher embedded a section map -- the PDF outline -- because
that outline is the label. PDFs without one are dropped at load time, so this over-collects
(about half of arXiv's CC-BY papers carry an outline of three entries or more).

arXiv's OAI-PMH endpoint is the only arXiv interface that publishes each paper's licence.
Only the Creative Commons subset that permits redistributing derived text is kept; the
default `nonexclusive-distrib` licence grants arXiv a licence, not us, and NoDerivs
variants forbid the chunked passages the corpus is made of. The licence travels with each
paper into the manifest and from there into every question's `meta`.

Stdlib only, so fetching needs no PDF library.

Layout: `<raw root>/pdfs/<arxiv id>.pdf` and `manifest.jsonl`.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from lod.corpus.repositories import web
from lod.corpus.services.sources.real.pdfdocs import MANIFEST, PDF_DIR

OAI = "http://export.arxiv.org/oai2"

OK_LICENCES = {
    "http://creativecommons.org/licenses/by/4.0/": "CC-BY-4.0",
    "http://creativecommons.org/licenses/by-sa/4.0/": "CC-BY-SA-4.0",
    "http://creativecommons.org/publicdomain/zero/1.0/": "CC0-1.0",
    "http://creativecommons.org/licenses/by/3.0/": "CC-BY-3.0",
}

# (set, from, until): several fields and windows, because outline conventions differ by
# community and LaTeX class, and one month of one field would be a narrow corpus.
WINDOWS = (
    ("cs", "2024-01-01", "2024-01-10"),
    ("math", "2024-03-01", "2024-03-08"),
    ("q-bio", "2023-09-01", "2023-10-01"),
    ("stat", "2024-05-01", "2024-05-12"),
    ("econ", "2023-06-01", "2023-08-01"),
    ("eess", "2024-02-01", "2024-02-08"),
)
TARGET = 600

_ID = re.compile(r"<id>([^<]+)</id>")
_LIC = re.compile(r"<license>([^<]+)</license>")
_TOK = re.compile(r"<resumptionToken[^>]*>([^<]+)</resumptionToken>")
_TITLE = re.compile(r"<title>(.*?)</title>", re.S)


def outputs() -> list[Path]:
    return [MANIFEST]


def list_records(oai_set: str, frm: str, until: str, pages: int = 2) -> list[dict]:
    """The CC-licensed arXiv ids in one window, with licence and title."""
    url = (f"{OAI}?verb=ListRecords&metadataPrefix=arXiv"
           f"&from={frm}&until={until}&set={oai_set}")
    out: list[dict] = []
    for _ in range(pages):
        body = web.get(url, timeout=180, retries=6, pause=10).decode("utf-8", "replace")
        for rec in body.split("<record>")[1:]:
            lic, aid = _LIC.search(rec), _ID.search(rec)
            if not (lic and aid) or lic.group(1) not in OK_LICENCES:
                continue
            title = _TITLE.search(rec)
            out.append({"id": aid.group(1), "licence": OK_LICENCES[lic.group(1)],
                        "set": oai_set,
                        "title": " ".join(title.group(1).split()) if title else ""})
        tok = _TOK.search(body)
        if not tok:
            break
        url = f"{OAI}?verb=ListRecords&resumptionToken={tok.group(1)}"
        time.sleep(5)
    return out


def fetch_pdf(aid: str, delay: float) -> Path | None:
    dest = PDF_DIR / f"{aid.replace('/', '_')}.pdf"
    if dest.exists() and dest.stat().st_size > 20_000:
        return dest
    try:
        body = web.get(f"https://arxiv.org/pdf/{aid}", timeout=120, retries=6, pause=10)
    except Exception as e:  # noqa: BLE001
        print(f"  skip {aid}: {type(e).__name__} {e}", flush=True)
        return None
    if not body.startswith(b"%PDF"):
        print(f"  skip {aid}: not a PDF ({body[:20]!r})", flush=True)
        return None
    web.save(body, dest)
    time.sleep(delay)
    return dest


def fetch(target: int = TARGET, delay: float = 3.0) -> int:
    """Top the cache up to `target` listed PDFs. Returns how many were added."""
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    have = ({json.loads(l)["id"] for l in MANIFEST.open() if l.strip()}
            if MANIFEST.exists() else set())
    wanted: list[dict] = []
    for oai_set, frm, until in WINDOWS:
        if len(have) + len(wanted) >= target:
            break
        recs = [r for r in list_records(oai_set, frm, until) if r["id"] not in have]
        print(f"  {oai_set} {frm}..{until}: {len(recs)} CC papers", flush=True)
        wanted += recs
        time.sleep(5)
    todo = wanted[: max(0, target - len(have))]
    kept = 0
    with MANIFEST.open("a") as mf:
        for i, rec in enumerate(todo):
            p = fetch_pdf(rec["id"], delay)
            if p is None:
                continue
            rec["path"] = p.name
            rec["url"] = f"https://arxiv.org/abs/{rec['id']}"
            mf.write(json.dumps(rec) + "\n")
            mf.flush()
            kept += 1
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(todo)} ({kept} kept)", flush=True)
    return kept
