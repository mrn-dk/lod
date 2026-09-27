"""Source-table row 26 — Survey of Professional Forecasters recession probabilities.

Row 26 names FiveThirtyEight, which no longer exists. The SPF is a better substitute than
the Kalshi markets already standing in for it, because it carries what Kalshi's snapshot
could not: a probability recorded *before* the outcome. Every quarter since 1968, several
dozen professional economists each state a probability that real GDP will decline in each
of the next five quarters. That is a genuine forecast, made by people, with a checkable
answer — the cleanest `meta.p` in the corpus.

`RECESS{k}` is one forecaster's probability (in percent) that real GDP falls in quarter
`survey + (k-1)`. One schema per horizon: a nowcast and a five-quarter-ahead forecast are
different problems with very different skill, and splitting them also lets a probe see
whether the model's confidence decays with horizon the way the forecasters' does.

**Real-time vintages, read from the Philadelphia Fed's own matrix.** Every growth rate is
taken *within one vintage* of the Real-Time Data Set for Macroeconomists (`ROUTPUT`,
monthly vintages, quarterly observations), so a rebasing can never sit between the two
levels being compared:

- the state's `recent_real_gdp_growth` is the four quarters before the survey, from the
  vintage of the survey's middle month -- the data the panel was actually sent;
- the realised outcome (`meta.outcome`) is whether GDP fell in the target quarter as
  *first published*: the vintage of the middle month of the following quarter.

An earlier version took both from an external `spf_actuals.csv`, which is not a GDP
series. Its 2020Q2 is **above** 2020Q1 (19,073.1 vs 19,072.5; the published real-time
fall was 9.1 %), 2008Q4 equals 2008Q3, and it starts in 1992. Measured by
a domain-5 audit: 99.0 % of history values and 32.2 % of outcomes disagreed with
the Philadelphia Fed vintages, and the 94 surveys before 1991Q2 -- 1968-1990, four
recessions -- were silently dropped for want of an outcome.

The state is the information a forecaster actually had: the survey date, the target
quarter, and the real-time GDP path up to the survey. It does not include anything from
after the survey.

**Two things this source got wrong in earlier builds, both measured rather than
suspected.**

*The state did not say whose forecast it was.* `RECESS{k}` is one economist's answer and
roughly thirty of them answer each survey, so a state built from (survey quarter, target
quarter, horizon) alone is shared by thirty rows carrying thirty different `p`. Measured
on an earlier build: 2,452 train questions over **80** distinct states, 172 devreal over 5,
134 testreal over **4** -- 100 % of this source's questions sit in a group whose members
contradict each other. No reader can do better than each group's median, which is MAE
0.070 (train) to 0.074 (devreal); the `probability_recovery` gate's bar is 0.10, so SPF is
not what fails it, but it cannot help either. `build()` now carries the published `ID`
column into the record and `_loader` renders it as `forecaster_id`, which makes the target
a function of the state again. A cache written before this needs `build(refresh=True)`.

*The eval split had its economics deleted.* `recent_real_gdp_growth` is populated here
for 92 % of rows -- three or four quarters of realised growth, read from the in-repo
real-time vintage file -- and an earlier build shipped it **empty on 76 % of
`spf_recession_h4`**. That was not this module: a forecaster's history is a function of
the survey quarter and so is word-for-word shared with the trained horizons, and
`dedup_against_train`'s shingle test deleted 1,156 of 1,290 h4 examples for it. The 134
survivors were exactly the 134 whose history was empty or one quarter long -- the test
selected the signal-free states. `template_state=True` below marks the task so
`services/generate.py` can exempt it, as it already does for every code-labelled row.
"""

from __future__ import annotations

import csv
import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Iterator

from lod.schema import Example, Question
from lod.corpus.services.sources.base import CACHE, UA
from lod.corpus.services.sources.real.base import RealTask
from lod.paths import ASSETS

PROB_URL = ("https://www.philadelphiafed.org/-/media/FRBP/Assets/Surveys-And-Data/"
            "survey-of-professional-forecasters/historical-data/spfmicrodata.xlsx")
ROUTPUT_URL = ("https://www.philadelphiafed.org/-/media/FRBP/Assets/Surveys-And-Data/"
               "real-time-data/data-files/xlsx/routputMvQd.xlsx")
# a derived extract of ROUTPUT (see `build_realtime`), small enough to live in the repo
REALTIME = ASSETS / "spf_realtime_gdp.csv"
LICENCE = "public domain (Federal Reserve Bank of Philadelphia)"
URL = ("https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/"
       "survey-of-professional-forecasters")

MICRO = CACHE / "spfmicrodata.xlsx"
HORIZONS = (1, 2, 3, 4, 5)
ROUTPUT = CACHE / "routputMvQd.xlsx"
_IN_PROCESS: dict[str, object] = {}


def xlsx_rows(path: Path, sheet: str) -> Iterator[list]:
    """Rows of one worksheet as lists of cell strings, stdlib only.

    `openpyxl` is not a dependency of this project, so `build(refresh=True)` could not run
    here at all -- which is how a cache without `forecaster` survived the fix that needed
    it. An xlsx is a zip of XML; two sheets of numbers need nothing more than this.
    """
    import re
    import xml.etree.ElementTree as ET
    import zipfile

    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
          "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships"}
    z = zipfile.ZipFile(path)
    wb = ET.fromstring(z.read("xl/workbook.xml"))
    rels = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
    rid = {e.get("name"): e.get(f"{{{ns['r']}}}id")
           for e in wb.find("m:sheets", ns)}[sheet]
    target = next(r.get("Target") for r in rels if r.get("Id") == rid).lstrip("/")
    target = target if target.startswith("xl/") else "xl/" + target
    shared: list[str] = []
    if "xl/sharedStrings.xml" in z.namelist():
        for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", ns):
            shared.append("".join(t.text or "" for t in si.iter(f"{{{ns['m']}}}t")))
    for row in ET.fromstring(z.read(target)).iter(f"{{{ns['m']}}}row"):
        cells: dict[int, str | None] = {}
        for c in row.findall("m:c", ns):
            letters = re.match(r"[A-Z]+", c.get("r")).group(0)
            idx = 0
            for ch in letters:
                idx = idx * 26 + ord(ch) - 64
            v = c.find("m:v", ns)
            cells[idx - 1] = (None if v is None else
                              shared[int(v.text)] if c.get("t") == "s" else v.text)
        if cells:
            yield [cells.get(i) for i in range(max(cells) + 1)]


def _float(x) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def build_realtime(refresh: bool = False) -> Path:
    """Extract, from ROUTPUT, the levels each SPF survey's vintage held for the six
    quarters before it. That is everything `history` and `declined` read."""
    if REALTIME.exists() and not refresh:
        return REALTIME
    if not ROUTPUT.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        ROUTPUT.write_bytes(urllib.request.urlopen(
            urllib.request.Request(ROUTPUT_URL, headers=UA), timeout=300).read())
    import re

    rows = list(xlsx_rows(ROUTPUT, "routput"))
    out = ["vintage,quarter,real_gdp"]
    for ci, name in enumerate(rows[0][1:], start=1):
        m = re.match(r"ROUTPUT(\d\d)M(\d+)$", str(name))
        if not m or int(m.group(2)) not in (2, 5, 8, 11):
            continue          # the survey is taken in the middle month of the quarter
        yy = int(m.group(1))
        vy, vq = (1900 + yy if yy >= 60 else 2000 + yy), (int(m.group(2)) + 1) // 3
        level = {}
        for r in rows[1:]:
            v = _float(r[ci]) if ci < len(r) else None
            if v is not None:
                y, q = str(r[0]).split(":Q")
                level[(int(y), int(q))] = v
        for back in range(6, 0, -1):
            oy, oq = _add_quarters(vy, vq, -back)
            if (oy, oq) in level:
                out.append(f"{vy}Q{vq},{oy}Q{oq},{level[(oy, oq)]}")
    REALTIME.parent.mkdir(parents=True, exist_ok=True)
    REALTIME.write_text("\n".join(out) + "\n")
    return REALTIME


def realtime() -> dict[tuple[int, int], dict[tuple[int, int], float]]:
    """survey quarter -> {observation quarter: real GDP level in that survey's vintage}."""
    if "realtime" in _IN_PROCESS:
        return _IN_PROCESS["realtime"]          # type: ignore[return-value]
    out: dict = {}
    with REALTIME.open() as f:
        for r in csv.DictReader(f):
            vy, vq = (int(x) for x in r["vintage"].split("Q"))
            oy, oq = (int(x) for x in r["quarter"].split("Q"))
            out.setdefault((vy, vq), {})[(oy, oq)] = float(r["real_gdp"])
    _IN_PROCESS["realtime"] = out
    return out


def history(year: int, quarter: int) -> list[dict]:
    """Growth in the four quarters before the survey, as the survey's vintage had them."""
    v = realtime().get((year, quarter), {})
    out = []
    for back in range(4, 0, -1):
        q = _add_quarters(year, quarter, -back)
        prev = _add_quarters(*q, -1)
        if q in v and prev in v:
            out.append({"quarter": f"{q[0]}Q{q[1]}",
                        "real_gdp_growth_pct": round((v[q] / v[prev] - 1.0) * 100, 2)})
    return out


def declined(year: int, quarter: int) -> int | None:
    """Did real GDP fall in this quarter, as first published? None until it is."""
    v = realtime().get(_add_quarters(year, quarter, 1), {})
    prev = _add_quarters(year, quarter, -1)
    if (year, quarter) not in v or prev not in v:
        return None
    return int(v[(year, quarter)] < v[prev])


def _add_quarters(year: int, quarter: int, k: int) -> tuple[int, int]:
    idx = (year * 4 + (quarter - 1)) + k
    return idx // 4, idx % 4 + 1


def download(refresh: bool = False) -> Path:
    if MICRO.exists() and not refresh:
        return MICRO
    CACHE.mkdir(parents=True, exist_ok=True)
    MICRO.write_bytes(urllib.request.urlopen(
        urllib.request.Request(PROB_URL, headers=UA), timeout=900).read())
    return MICRO


def build(refresh: bool = False) -> list[dict]:
    """One row per (forecaster, survey quarter, horizon) with a scorable outcome."""
    cache = CACHE / "spf_recess.jsonl"
    if cache.exists() and not refresh:
        return [json.loads(l) for l in cache.read_text().splitlines() if l.strip()]

    build_realtime()
    it = xlsx_rows(download(), "RECESS")
    header = [str(c) for c in next(it)]
    col = {h: i for i, h in enumerate(header)}

    out: list[dict] = []
    for row in it:
        try:
            year, quarter = int(float(row[col["YEAR"]])), int(float(row[col["QUARTER"]]))
        except (TypeError, ValueError):
            continue
        hist = history(year, quarter)
        for k in HORIZONS:
            key = f"RECESS{k}"
            if key not in col or col[key] >= len(row):
                continue
            p = _float(row[col[key]])
            if p is None:
                continue                       # '#N/A' -- the forecaster skipped it
            p /= 100.0
            if not 0.0 <= p <= 1.0:
                continue
            ty, tq = _add_quarters(year, quarter, k - 1)
            fell = declined(ty, tq)
            if fell is None:
                continue                       # not yet published: no outcome to keep
            history_rows = hist
            # The forecaster's own id. Without it the state is a pure function of
            # (survey quarter, horizon) while `p` is not: ~30 economists answer the same
            # survey and each states a different probability, so an earlier build shipped
            # with 2,452 train questions over **80** distinct states and 134 testreal
            # questions over **4**, every group carrying contradictory targets. That is
            # not a soft label, it is the same question asked twice with two answers, and
            # it caps any model at MAE 0.070-0.074 on this source however well it reads.
            # `ID` is published in the RECESS sheet beside `RECESS1..5`, so naming which
            # forecaster is being asked about costs nothing and makes the target a
            # function of the state again. Requires `build(refresh=True)` to reach a
            # cache written before this existed; `_loader` tolerates its absence.
            row_out = {"survey": f"{year}Q{quarter}", "target": f"{ty}Q{tq}",
                       "horizon": k, "p": p, "declined": fell,
                       "history": history_rows}
            fid = _float(row[col["ID"]]) if "ID" in col else None
            if fid is not None:
                row_out["forecaster"] = int(fid)
            out.append(row_out)
    cache.write_text("".join(json.dumps(r) + "\n" for r in out))
    return out


def rows() -> list[dict]:
    if "rows" in _IN_PROCESS:
        return _IN_PROCESS["rows"]              # type: ignore[return-value]
    cache = CACHE / "spf_recess.jsonl"
    if not cache.exists():
        _IN_PROCESS["rows"] = []
        return []
    _IN_PROCESS["rows"] = [json.loads(l) for l in cache.read_text().splitlines() if l.strip()]
    return _IN_PROCESS["rows"]


QUESTION_ID = "d05.spf_decline"
QUESTION = "Will real GDP decline in {target_quarter}?"
INSTRUCTIONS = ("Answer with the probability the forecaster named in the record gave, in "
                "that survey, to real GDP falling in the target quarter (quarter over "
                "quarter, as first published). It is this economist's stated forecast, "
                "not the realised outcome.")
_BANK: list = []


def _question(target: str, key: str) -> str:
    if not _BANK:
        from lod.phrasings import PhrasingBank
        _BANK.append(PhrasingBank.load())
    return _BANK[0].pick(QUESTION_ID, QUESTION, key).format(target_quarter=target)


def _half(r: dict) -> int:
    """Which side of the horizon holdout this (survey, forecaster) pair may feed."""
    h = hashlib.sha1(f"{r['survey']}:{r.get('forecaster')}".encode()).digest()
    return h[0] % 2


# Share of the held-out pairs that feed the devreal horizon; the rest feed testreal.
# Measured on the cached RECESS sheet: ~4,080 held-out pairs per horizon, so testreal's
# h3/h4 keep ~2,700 each (above the 2,000 per-task quota) and devreal's h5 gets ~1,300.
DEV_SHARE_OF_HELD = 3            # one pair in three


def _part(r: dict) -> str:
    """Which split's horizons this (survey, forecaster) pair may feed.

    `_half` keeps the trained horizons off the held-out pairs. The held-out half is then
    cut again, dev from test: one economist's h5 answer moves with their h3 and h4, so a
    devreal h5 row on a pair that testreal also asks about is a near-duplicate of a test
    row, and the split that selects checkpoints and fits T would be reading test.
    """
    if _half(r) == 0:
        return "train"
    h = hashlib.sha1(f"held:{r['survey']}:{r.get('forecaster')}".encode()).digest()
    return "devreal" if h[0] % DEV_SHARE_OF_HELD == 0 else "testreal"


def split_of_horizon(horizon: int) -> str:
    if horizon in RESERVED_TESTREAL_HORIZONS:
        return "testreal"
    if horizon in RESERVED_DEVREAL_HORIZONS:
        return "devreal"
    return "train"


def spread(horizon: int) -> list[dict]:
    """This horizon's rows in a fixed pseudo-random order, restricted to one half of the
    (survey, forecaster) pairs.

    Two defects, both measured by a domain-5 audit. (1) The cache is chronological and
    the loader took the first `n`, so a 2,000-row quota was 1968-1985 only. (2) One
    economist answers all five horizons in one survey, and the five answers move
    together, so a held-out h3 row whose h1 and h2 siblings were trained on is a
    near-duplicate of training data under a different horizon label. Trained horizons
    draw from one half of the pairs and held-out horizons from the other, so no
    (survey, forecaster) pair is on both sides -- and within the held-out half the
    devreal horizon and the testreal horizons draw from disjoint pairs (`_part`).
    """
    split = split_of_horizon(horizon)
    rs = [r for r in rows() if r["horizon"] == horizon and _part(r) == split]
    return sorted(rs, key=lambda r: hashlib.sha1(
        f"{r['survey']}:{r.get('forecaster')}:{horizon}".encode()).digest())


def _loader(horizon: int):
    def load(n: int) -> Iterator[Example]:
        made = 0
        for r in spread(horizon):
            if made >= n:
                return
            made += 1
            record = {"survey_quarter": r["survey"],
                      "target_quarter": r["target"],
                      "quarters_ahead": horizon - 1}
            if r.get("forecaster") is not None:
                # which of the panel's economists is being asked about; see build()
                record["forecaster_id"] = r["forecaster"]
            record["recent_real_gdp_growth"] = r["history"]
            state = json.dumps(record, indent=2)
            yield Example(
                task=f"spf_recession_h{horizon}",
                state=state,
                questions=[Question(
                    id="declines",
                    question=_question(r["target"], f"{r['survey']}:{r.get('forecaster')}:{horizon}"),
                    options=["no", "yes"],
                    target=r["declined"],
                    meta={"p": r["p"]},   # the economists' own probability
                    instructions=INSTRUCTIONS,
                )],
            )
    return load


# h3 already reached testreal, by hash rather than by design. SPF is the best-behaved of
# the three meta.p sources for the middle of the range (22.2 % of its p mass is extreme,
# vs. 62-68 % for GEFS/Manifold -- domain-5 verification), which is exactly
# where `probability_recovery` reads, so it is worth deliberately widening rather than
# leaving to luck. h4 is forced alongside h3, so testreal holds two of the five horizons
# (the 2- and 3-quarters-ahead forecasts); h1 and h2 (nowcast and 1-quarter-ahead, where
# the ensemble spread is tightest and training signal strongest) stay trained; h5 stays
# the devreal check. This reads as depth, the same shape as row 13's lichess holdout:
# near-term trained, far-term held out. An earlier note here read SPF's h1 p<0.1 bin as
# "realising a decline 32.4 % of the time" and called it forecaster miscalibration. It was
# the fabricated actuals file: against the first-published real-time series that bin
# realises 3.6 % (n=4,030) and the 0.9+ bin 81 % (n=462) -- the nowcast is well
# calibrated; longer horizons are compressed toward the base rate (h3 0.9+ bin realises
# 42 %, n=33).
RESERVED_TESTREAL_HORIZONS = (3, 4)
RESERVED_DEVREAL_HORIZONS = (5,)


def tasks() -> list[RealTask]:
    if not (CACHE / "spf_recess.jsonl").exists():
        return []
    present = sorted({r["horizon"] for r in rows()})
    out = []
    for h in present:
        force = split_of_horizon(h)
        out.append(RealTask(row=26, name=f"spf_recession_h{h}", licence=LICENCE, url=URL,
                            load=_loader(h), meta_p=True, force_split=force,
                            template_state=True,
                            notes=f"SPF RECESS{h}: individual economists' probability that "
                                  f"real GDP falls {h-1} quarters ahead; outcome and GDP "
                                  f"history from the Philadelphia Fed real-time vintages "
                                  f"(ROUTPUT); forced to {force}"))
    return out
