"""Source-table row 9 — NVD CVE records.

Nine schemas over one state (the vulnerability description): the CWE it was assigned,
the seven CVSS v3 base metrics, and the severity band. Labels come from NVD analysts,
so this is a real source in the sense `base.py` defines, and the CVSS metrics are
*declared* ordinal scales rather than ordinal by our interpretation — the standard defines
the order, which is exactly the bar the `ordinal` tag sets.

That matters disproportionately: rows 11 and 13 are reserved to `testreal`, so this row
and row 4 are the only ones the source table annotates as ordinal at all, and ordinal is
the scarcest of the mixing-rule buckets.

Only CVSS v3.x is used. Old CVEs carry `cvssMetricV2`, whose fields are named differently
(`accessVector`, `authentication`) and whose severity has three bands rather than five;
mixing the two versions under one question would put two different label sets behind the
same schema key.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Iterator

from lod.schema import Example, Question
from lod.corpus.services.sources.base import CACHE, fetch_json
from lod.corpus.repositories import raw_store as store
from lod.corpus.services.sources.real.base import RealTask, disjoint_slice

API = "https://services.nvd.nist.gov/rest/json/cves/2.0"
LICENCE = "public domain (NIST NVD)"
URL = "https://nvd.nist.gov/developers/vulnerabilities"

# (question id, question text, options, ordinal). The option lists are the CVSS v3
# enumerations verbatim, in their defined order, so an ordinal question's options are
# already sorted low to high -- the model never sees the order, but a human auditing
# the provenance manifest can check the claim.
#
# The source table settles the ordinality question for row 9 directly: "all but CWE
# ordinal". That includes attackVector and userInteraction, which a first pass here marked
# non-ordinal on the grounds that CVSS does not number them. The table is followed, and the
# reading is defensible -- both are ordered by exposure: PHYSICAL < LOCAL < ADJACENT < NETWORK,
# and NONE < REQUIRED.
METRICS: tuple[tuple[str, str, str, list[str], bool], ...] = (
    ("av", "attackVector",
     "What is the CVSS v3 Attack Vector of this vulnerability, i.e. the context from "
     "which an attacker can exploit it?",
     ["PHYSICAL", "LOCAL", "ADJACENT_NETWORK", "NETWORK"], True),
    ("ac", "attackComplexity",
     "What is the CVSS v3 Attack Complexity of this vulnerability, i.e. whether exploiting "
     "it depends on conditions beyond the attacker's control?",
     ["LOW", "HIGH"], True),
    ("pr", "privilegesRequired",
     "What is the CVSS v3 Privileges Required value of this vulnerability, i.e. the level "
     "of privileges an attacker must hold before exploiting it?",
     ["NONE", "LOW", "HIGH"], True),
    ("ui", "userInteraction",
     "What is the CVSS v3 User Interaction value of this vulnerability, i.e. whether a "
     "user other than the attacker must take part for an exploit to succeed?",
     ["NONE", "REQUIRED"], True),
    ("c", "confidentialityImpact",
     "What is the CVSS v3 Confidentiality Impact of this vulnerability, i.e. how much "
     "information an attacker can read when it is exploited?",
     ["NONE", "LOW", "HIGH"], True),
    ("i", "integrityImpact",
     "What is the CVSS v3 Integrity Impact of this vulnerability, i.e. how much an "
     "attacker can modify when it is exploited?",
     ["NONE", "LOW", "HIGH"], True),
    ("a", "availabilityImpact",
     "What is the CVSS v3 Availability Impact of this vulnerability, i.e. how much it "
     "degrades the availability of the affected component when exploited?",
     ["NONE", "LOW", "HIGH"], True),
    ("sev", "baseSeverity",
     "What is the CVSS v3 base severity rating of this vulnerability, i.e. the qualitative "
     "band its CVSS v3 base score falls in?",
     ["LOW", "MEDIUM", "HIGH", "CRITICAL"], True),
)
CWE_QUESTION = "Which CWE weakness class was this vulnerability assigned?"

# Domain-7 audit: an earlier build's wordings came from enrichment, not from here, and
# one of them -- "Based on the supplied CVSS assessment, what level of integrity impact
# ..." -- points at an assessment `state_of` deliberately removes. The questions above
# name the CVSS metric they ask for, the task prefix is protected from enrichment
# (`nvd_` in enrich.schemas.PROTECTED_PREFIXES), and variety comes from the frozen
# phrasing bank, whose specs are `lod/assets/phrasing_specs/d07.json`.
_BANK = None


def _phrase(template_id: str, template: str, key: str) -> str:
    global _BANK
    if _BANK is None:
        from lod.phrasings import PhrasingBank
        _BANK = PhrasingBank.load()
    return _BANK.pick(f"d07.{template_id}", template, key)


# Class imbalance. The rule, measured by a domain-7 audit on each task's own slice:
# rebalance a task when its majority label is over 70 % AND a keyword reader gains under
# 5 points on always-majority -- i.e. when the task, as published, scores the prior rather
# than the prose. Two tasks qualify:
#
#   attackComplexity   88.5 % LOW over the pool; keyword 0.887 vs majority 0.908 on the
#                      built task (reading *loses*). Capped: majority 0.696, keyword 0.732.
#   privilegesRequired 75.3 % NONE on its slice; keyword 0.776 vs 0.753 (+2.3).
#                      Capped: majority 0.626, keyword 0.750 (+12.4).
#
# The cap is k x the runner-up label's count (k = 2), so every minority example is kept.
# attackVector (+13.7), availabilityImpact (+11.5) and the rest already reward reading and
# keep NVD's published distribution.
REBALANCE_K: dict[str, int] = {"attackComplexity": 2, "privilegesRequired": 2}

PAGE = 2000  # the NVD maximum
_CACHE_FILE = CACHE / "nvd_cves.jsonl"

# Oracle's Critical Patch Update advisories paste the published CVSS assessment into the
# CVE description itself, in two machine-readable forms:
#
#   "CVSS 3.0 Base Score 8.2 (Confidentiality, Integrity and Availability impacts)."
#   "CVSS Vector: (CVSS:3.0/AV:N/AC:L/PR:N/UI:R/S:C/C:H/I:H/A:H)."
#
# The first is the answer to `nvd_baseSeverity` with one comparison -- whose thresholds
# the option descriptions state outright ("CVSS base score 7.0 to 8.9") -- and the second
# is the answer to all seven metric questions, spelled out. Measured on an earlier build:
# 191 of 2,000 `nvd_baseSeverity` states (9.6 %) carried the vector string, 190 carried
# the score, and the band that score falls in was the gold option 187 times out of 190
# (98.4 %). Which task is exposed is an accident of `disjoint_slice`: the same rows under
# a different slicing would hand the label to any of the nine.
#
# So the published assessment comes out of the state before the state is used, the same
# way row 10 strips a licence's title and row 13 blanks the level field it parsed. What
# is left is the advisory's prose, which is what the question is supposed to be read
# from.
#
# `[^.]*` would stop inside the score's own decimal point and leave "2 (Confidentiality,
# Integrity and Availability impacts)." behind -- which still names the C/I/A half of the
# assessment. Match the number, then the parenthetical that lists the impacts.
_CVSS_SCORE = re.compile(
    r"\s*CVSS\s*(?:v)?3\.[01]\s*Base\s*Score\s*[0-9]+(?:\.[0-9]+)?"
    r"\s*(?:\([^)]*\))?\s*\.?", re.I)
_CVSS_VECTOR = re.compile(r"\s*CVSS\s*Vector\s*:?\s*\(?\s*CVSS:3\.[01]/[A-Z:/]+\)?\.?",
                          re.I)
_BARE_VECTOR = re.compile(r"\s*\(?\s*CVSS:3\.[01]/[A-Z:/]+\)?\.?")


# What the three patterns above miss, measured on the cached pool by a domain-7 audit:
# a vector with no "CVSS:3.x/" prefix ("combined CVSSv3 score of
# 5.8 (AV:N/AC:L/PR:N/UI:N/S:C/C:L/I:N/A:N)", CVE-2017-10617), a score in running prose
# ("giving a CVSS Base Score of 8.8", CVE-2017-10202), and Android's vendor rating ("This
# issue is rated as High severity due to ...", 19 CVEs -- NVD's band disagrees with it on
# all 19, so it is a wrong answer planted in the state as much as a leaked one). Each is
# removed as a whole sentence: stripping only the number leaves "giving a ." behind.
_LEAK_SENTENCE = re.compile(
    r"\b(?:AV|AC|PR|UI|Au):[A-Z]\b"
    r"|CVSS\w*\s*(?:v?\d(?:\.\d)?\s*)?(?:base\s*)?score\s*(?:of|is|:)?\s*\d"
    r"|\bbase\s*score\s*(?:of|is|:)?\s*\d"
    r"|\b(?:critical|high|medium|moderate|low|important)\s+severity\b"
    r"|\bseverity\s*(?:rating|level)?\s*(?:of|:|is|=)\s*(?:critical|high|medium|moderate|low)\b",
    re.I)
# a sentence boundary is a period followed by whitespace, so "5.8" and "v3.0" stay whole
_SENTENCES = re.compile(r"(?<=[.!?])\s+")


def state_of(desc: str) -> str:
    """The advisory prose, with the published CVSS assessment taken back out."""
    for pattern in (_CVSS_VECTOR, _CVSS_SCORE, _BARE_VECTOR):
        desc = pattern.sub(" ", desc)
    desc = re.sub(r"\s+", " ", desc).strip()
    desc = " ".join(s for s in _SENTENCES.split(desc) if not _LEAK_SENTENCE.search(s))
    return desc.strip()[:4000]


def _cvss3(cve: dict) -> dict | None:
    """NVD's own CVSS v3 base metrics (the `Primary` entry, source nvd@nist.gov), 3.1
    before 3.0; a CNA's `Secondary` entry only when NVD published none. None if no v3.

    Taking `arr[0]` of the first non-empty version picked a CNA's assessment over NVD's
    whenever the CNA scored in 3.1 and NVD in 3.0: CVE-2017-12994 carries exactly that
    (cvssMetricV31 = one Secondary entry, cvssMetricV30 = NVD's Primary), 1 of the 60
    CVEs the domain-7 audit fetched from the API.
    """
    metrics = cve.get("metrics") or {}
    entries = [e for key in ("cvssMetricV31", "cvssMetricV30") for e in metrics.get(key) or []]
    for e in entries:
        if e.get("type") == "Primary" and e.get("cvssData"):
            return e["cvssData"]
    return next((e["cvssData"] for e in entries if e.get("cvssData")), None)


def _cwe(cve: dict) -> str | None:
    for w in cve.get("weaknesses") or []:
        for d in w.get("description") or []:
            v = d.get("value", "")
            # NVD-CWE-Other / NVD-CWE-noinfo are placeholders, not classifications
            if v.startswith("CWE-"):
                return v
    return None


def download(limit: int | None = None, sleep: float = 6.5, years: int = 6,
             refresh: bool = False) -> list[dict]:
    """Page the NVD API into a local cache, once.

    Paged by publication date, most recent first, and *not* from startIndex 0. The API
    returns CVEs oldest-first, and everything before roughly 2016 carries only
    `cvssMetricV2`; starting at the beginning spends a 6.5 s rate-limit sleep per page to
    yield nothing usable. Measured: several minutes without a single row.

    Unauthenticated NVD allows about 5 requests per 30 s and answers 403 beyond that, so
    the sleep is deliberate, and the result is cached: rebuilding the corpus must not
    re-download the set or re-earn the rate limit.
    """
    # An existing cache is authoritative unless `refresh` says otherwise. Comparing its
    # length against the caller's `limit` would be wrong: the nine task loaders all call
    # download() with no argument, so any default larger than what the feed actually
    # yielded would make each of them re-download the whole thing.
    local_feeds = sorted((store.RAW_ROOT / "nvd").glob("nvdcve-2.0-*.json.gz"))
    if local_feeds and not refresh:
        import gzip
        target = limit if limit is not None else 20000
        rows = []
        for feed in local_feeds:
            with gzip.open(feed, "rt", encoding="utf-8") as f:
                payload = json.load(f)
            for item in payload.get("vulnerabilities", []):
                cve = item.get("cve") or {}
                metrics = _cvss3(cve)
                if not metrics:
                    continue
                desc = next((x.get("value") for x in cve.get("descriptions") or []
                             if x.get("lang") == "en"), "")
                if not desc or desc.startswith("** REJECT") or cve.get("vulnStatus") == "Rejected":
                    continue
                # Keyed by the CVSS *field* name, the same as the API branch below and
                # the same as the `nvd_cves.jsonl` cache on disk -- `_metric_loader`
                # reads `r.get("attackVector")`. Keying these by the short id instead
                # (`av`, `ac`, ...) left every metric lookup returning None, so all
                # eight `nvd_<metric>` tasks silently yielded **zero** examples whenever
                # a local NVD JSON feed was present; only `nvd_cwe` survived.
                rows.append({"id": cve.get("id"), "desc": desc.strip(), "cwe": _cwe(cve),
                             **{field: metrics.get(field)
                                for _, field, _, _, _ in METRICS}})
                if len(rows) >= target:
                    break
            if len(rows) >= target:
                break
        return rows if limit is None else rows[:limit]
    if _CACHE_FILE.exists() and not refresh:
        rows = [json.loads(l) for l in _CACHE_FILE.read_text().splitlines() if l.strip()]
        if rows:
            return rows if limit is None else rows[:limit]

    target = limit if limit is not None else 20000
    CACHE.mkdir(parents=True, exist_ok=True)
    out: list[dict] = []
    # the API caps a date range at 120 days, so walk back in 120-day windows
    end = datetime.now(timezone.utc)
    windows = max(1, int(years * 365 / 120))
    for _ in range(windows):
        if len(out) >= target:
            break
        start_dt = end - timedelta(days=119)
        params = (f"?resultsPerPage={PAGE}"
                  f"&pubStartDate={start_dt.strftime('%Y-%m-%dT%H:%M:%S.000')}"
                  f"&pubEndDate={end.strftime('%Y-%m-%dT%H:%M:%S.000')}")
        end = start_dt
        d = fetch_json(f"{API}{params}", timeout=180, retries=3)
        vulns = d.get("vulnerabilities") or []
        if not vulns:
            time.sleep(sleep)
            continue
        for v in vulns:
            cve = v.get("cve") or {}
            desc = next((x["value"] for x in cve.get("descriptions") or []
                         if x.get("lang") == "en"), None)
            cvss = _cvss3(cve)
            if not desc or not cvss:
                continue
            # a rejected CVE has a placeholder description and no real labels
            if desc.startswith("** REJECT") or cve.get("vulnStatus") == "Rejected":
                continue
            out.append({"id": cve.get("id"), "desc": desc, "cwe": _cwe(cve),
                        **{name: cvss.get(name) for _, name, _, _, _ in METRICS}})
        time.sleep(sleep)

    _CACHE_FILE.write_text("".join(json.dumps(r) + "\n" for r in out))
    return out if limit is None else out[:limit]


def top_cwes(rows: list[dict], k: int = 50) -> list[str]:
    counts: dict[str, int] = {}
    for r in rows:
        if r.get("cwe"):
            counts[r["cwe"]] = counts.get(r["cwe"], 0) + 1
    return [c for c, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:k]]


def _usable(rows: list[dict], label_of) -> list[tuple[dict, str, str]]:
    """(row, state, label) for the rows of one task's slice that can be asked about.

    One state, one answer. NVD reuses a description verbatim across sibling CVEs -- 237
    groups, 840 rows of the cached pool; the largest are 36 Adobe Flash advisories reading
    "Unspecified vulnerability in Adobe Flash Player 21.0.0.242 and earlier ..." -- and a
    task that kept all of them repeated one example up to 36 times (433 of 14,257 records
    of a row-9 build, 3.0 %). Where the siblings were *scored differently* the same state
    carried two gold answers (5 such groups in that build), which no reading of the text
    can resolve, so those groups are dropped whole rather than settled by whichever CVE
    came first.
    """
    groups: dict[str, list[tuple[dict, str]]] = {}
    for r in rows:
        label = label_of(r)
        if label is None:
            continue
        groups.setdefault(state_of(r["desc"]), []).append((r, label))
    out = []
    for state, members in groups.items():
        if state and len({label for _, label in members}) == 1:
            out.append((members[0][0], state, members[0][1]))
    return out


def _rebalanced(items: list[tuple[dict, str, str]], k: int | None) -> list[tuple[dict, str, str]]:
    """Cap the majority label at k x the runner-up's count; pool order is kept.

    Which majority rows stay is decided by a hash of the CVE id, not by position: the
    pool is ordered by id, and keeping the first ones would also select an era.
    """
    if not k:
        return items
    counts: dict[str, int] = {}
    for _, _, label in items:
        counts[label] = counts.get(label, 0) + 1
    if len(counts) < 2:
        return items
    ranked = sorted(counts.items(), key=lambda kv: -kv[1])
    majority, cap = ranked[0][0], k * ranked[1][1]
    if counts[majority] <= cap:
        return items

    def h(r: dict) -> bytes:
        return hashlib.sha1(f"nvd-rebalance:{r['id']}".encode()).digest()

    keep = {id(r) for r, _, _ in sorted((x for x in items if x[2] == majority),
                                         key=lambda x: h(x[0]))[:cap]}
    return [x for x in items if x[2] != majority or id(x[0]) in keep]


def _metric_loader(field: str, question: str, options: list[str],
                   task_index: int, n_tasks: int):
    def load(n: int) -> Iterator[Example]:
        # a disjoint slice of the CVE pool: the nine row-9 tasks describe the same
        # records, and sharing states across tasks makes the split and the dedup fight
        rows = disjoint_slice(download(), task_index, n_tasks)
        items = _usable(rows, lambda r: r.get(field) if r.get(field) in options else None)
        items = _rebalanced(items, REBALANCE_K.get(field))
        for r, state, val in items[:n]:
            yield Example(
                task=f"nvd_{field}",
                state=state,
                questions=[Question(id=field, question=_phrase(field, question, r["id"]),
                                    options=list(options), target=options.index(val),
                                    meta={"cve": r["id"]})],
            )
    return load


def _cwe_loader(k: int = 50, task_index: int = 0, n_tasks: int = 1):
    def load(n: int) -> Iterator[Example]:
        all_rows = download()
        # the option set is fixed from the whole pool so it does not depend on the slice
        opts = top_cwes(all_rows, k)
        rows = disjoint_slice(all_rows, task_index, n_tasks)
        allowed = set(opts)
        items = _usable(rows, lambda r: r.get("cwe") if r.get("cwe") in allowed else None)
        for r, state, cwe in items[:n]:
            yield Example(
                task="nvd_cwe",
                state=state,
                questions=[Question(id="cwe", question=_phrase("cwe", CWE_QUESTION, r["id"]),
                                    options=list(opts), target=opts.index(cwe),
                                    meta={"cve": r["id"]})],
            )
    return load


def tasks() -> list[RealTask]:
    """Nine tasks, one per schema. Separate tasks, not one example with nine questions:
    the split is hashed by task name, so bundling them would force all nine into the
    same split and lose eight schemas' worth of independent train/test signal."""
    n_tasks = 1 + len(METRICS)
    out = [
        RealTask(row=9, name="nvd_cwe", licence=LICENCE, url=URL,
                 load=_cwe_loader(task_index=0, n_tasks=n_tasks), ordinal=False,
                 family="nvd",
                 notes="top 50 CWE classes by frequency; disjoint CVE slice 1/9")
    ]
    for i, (_, field, question, options, ordinal) in enumerate(METRICS, start=1):
        out.append(RealTask(
            row=9, name=f"nvd_{field}", licence=LICENCE, url=URL,
            load=_metric_loader(field, question, options, i, n_tasks), ordinal=ordinal,
            # nine schemas over slices of one CVE pool: the dataset is the held-out unit,
            # as for a Hub source's columns (make_devood.py)
            family="nvd",
            notes=f"CVSS v3 {field}" + (" (declared ordinal scale)" if ordinal else "")
                  + f"; disjoint CVE slice {i+1}/9"
                  + (f"; majority label capped at {REBALANCE_K[field]}x the runner-up"
                     if field in REBALANCE_K else ""),
        ))
    return out
