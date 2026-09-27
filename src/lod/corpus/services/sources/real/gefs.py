"""Source-table row 27 — probability of precipitation from the GEFS v12 reforecast.

Row 27 asks for an NWS gridpoint forecast, next-day observation, and `meta.p` = PoP. The
NWS API serves only current forecasts, so the historical pairing it needs is not
retrievable from it. NOAA's GEFS v12 reforecast archive is the same quantity from the same
agency, and it is the corpus's main supply of `meta.p`: Manifold's resolved markets top
out around 750 usable train questions, well under the 3 % `meta.p` floor.

**PoP is the ensemble fraction.** Five members are run per cycle (c00, p01-p04), so the
probability a gridpoint sees rain is the share of members forecasting it -- a real
forecast probability produced by a real system, not a number we chose.

**The verification is a short-lead forecast, not a gauge observation.** GEFS reforecast
carries no observations. A forecast issued at cycle C valid at time T is scored against
the 24-hour forecast from the cycle one day before T -- which is far closer to truth than
the long-lead forecast being scored, and is standard practice for reforecast verification.
It is still model output, and calling it an observation would overstate it, so it is named
for what it is here and in the provenance manifest.

Each lead needs its *own* verifying cycle, because the truth must be valid at the same
moment as the forecast. A first version used one verifying cycle per forecast cycle and
scored every lead against it, so a 1-day and a 10-day forecast were both checked against
weather valid on the same afternoon. The resulting PoP was still monotonic -- weather is
autocorrelated, so the ranking survived -- but hopelessly miscalibrated: a PoP of 1.0
verified at 0.384. A probability that only ranks is exactly what `meta.p` must not be.

**State is a record, not prose.** Row 27 says "forecast text"; gridded GRIB has none, so
the state is the gridpoint's own forecast values rendered as a record, which is the shape
row 30 uses. The label and the forecast probability are both real, which is what `meta.p`
needs.

One schema per lead time. A single schema is capped at the per-schema row limit, which is
exactly why Manifold alone could not reach 3 %, and lead time genuinely changes the
question: "will it rain in 3 days" and "in 10 days" are different forecasting problems
with different skill.

**Five questions per gridpoint, and they are not paraphrases.** See the note above
`THRESHOLDS_MM`. The short version: an earlier build shipped one wording per lead, written
by the enrichment runner rather than by this module, and the day-10 wording asked whether
*at least one member* forecast 1 mm while the target still encoded the member *fraction*.
Those two agree only on a unanimous ensemble, which is 22.2 % of that task's rows, so
945 of the `probability_recovery` gate's 1,969 questions were labelled with the answer to
a question they did not ask.
The answer is not a better paraphrase -- it is to ask the different functions on purpose,
each with its own option keys, its own code-derived target, and the criterion written
into `instructions` and `descriptions`, which the rewrite path cannot replace.
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from typing import Iterator

from lod.schema import Example, Question
from lod.corpus.services.sources.base import CACHE, UA
from lod.corpus.services.sources.real.base import RealTask

BASE = "https://noaa-gefs-retrospective.s3.amazonaws.com/GEFSv12/reforecast"
LICENCE = "public domain (NOAA GEFS v12 reforecast)"
URL = "https://registry.opendata.aws/noaa-gefs-reforecast/"

MEMBERS = ("c00", "p01", "p02", "p03", "p04")
RAIN_MM = 1.0            # a wet gridpoint: >= 1 mm accumulated over the window
LEAD_DAYS = (1, 3, 5, 7, 10)
GEFS_DIR = CACHE / "gefs"
_ROWS = CACHE / "gefs_pop.jsonl"
_IN_PROCESS: dict[str, list] = {}


def _fetch_member(cycle: str, member: str) -> "object":
    """Open one member's precipitation file, downloading it once."""
    import xarray as xr

    GEFS_DIR.mkdir(parents=True, exist_ok=True)
    dest = GEFS_DIR / f"apcp_{cycle}_{member}.grib2"
    if not dest.exists():
        url = f"{BASE}/{cycle[:4]}/{cycle}/{member}/Days:1-10/apcp_sfc_{cycle}_{member}.grib2"
        req = urllib.request.Request(url, headers=UA)
        dest.write_bytes(urllib.request.urlopen(req, timeout=900).read())
    # The perturbed-member files carry both control (cf) and perturbed (pf) messages, so
    # cfgrib refuses them without a dataType filter; only c00 is homogeneous, which is why
    # a probe on c00 alone passed and the build then failed on every cycle.
    kind = "cf" if member == "c00" else "pf"
    return xr.open_dataset(dest, engine="cfgrib",
                           backend_kwargs={"indexpath": "",
                                           "filter_by_keys": {"dataType": kind}})


def verifying_cycle(cycle: str, lead_days: int) -> str:
    """The cycle whose 24-hour forecast is valid at the same time as this forecast."""
    from datetime import datetime, timedelta

    t = datetime.strptime(cycle, "%Y%m%d%H") + timedelta(days=lead_days - 1)
    return t.strftime("%Y%m%d") + "00"


def build(cycles: tuple[str, ...], points: int = 900,
          refresh: bool = False) -> list[dict]:
    """Sample gridpoints into (forecast PoP, verified outcome) rows."""
    if _ROWS.exists() and not refresh:
        return [json.loads(l) for l in _ROWS.read_text().splitlines() if l.strip()]

    import numpy as np

    out: list[dict] = []
    rng = np.random.default_rng(0)
    for cycle in cycles:
        try:
            members = [_fetch_member(cycle, m) for m in MEMBERS]
        except Exception as e:
            print(f"  gefs: {cycle} unavailable ({type(e).__name__})", flush=True)
            continue

        nlat, nlon = members[0].tp.shape[1], members[0].tp.shape[2]
        lat_i = rng.integers(80, nlat - 80, points)      # skip the poles
        lon_i = rng.integers(0, nlon, points)

        def at_lead(ds, hours: int):
            """Select by step *value*, never by index: the control member carries 80
            steps and the perturbed members 79, so positional indexing would compare
            different lead times across the ensemble and corrupt every probability."""
            target = np.timedelta64(hours, "h")
            idx = int(np.argmin(np.abs(ds.step.values - target)))
            return ds.tp.values[idx]

        for lead in LEAD_DAYS:
            hours = lead * 24
            if np.timedelta64(hours, "h") > members[0].step.values.max():
                continue
            fc = np.stack([at_lead(m, hours) for m in members])       # [5, lat, lon]
            # a 24-hour forecast valid at the SAME moment as this lead
            try:
                truth = _fetch_member(verifying_cycle(cycle, lead), "c00")
            except Exception:
                continue
            tv = at_lead(truth, 24)
            truth.close()
            for la, lo in zip(lat_i, lon_i):
                vals = fc[:, la, lo]
                if not np.isfinite(vals).all():
                    continue
                wet = int((vals >= RAIN_MM).sum())
                obs = tv[la, lo]
                if not np.isfinite(obs):
                    continue
                lat = float(members[0].latitude.values[la])
                lon = float(members[0].longitude.values[lo])
                out.append({
                    "lead": lead, "cycle": cycle,
                    "lat": round(lat, 2), "lon": round(lon, 2),
                    "members_mm": [round(float(v), 2) for v in vals],
                    "p": wet / len(MEMBERS),
                    "rained": int(float(obs) >= RAIN_MM),
                })
        for m in members:
            m.close()

    CACHE.mkdir(parents=True, exist_ok=True)
    _ROWS.write_text("".join(json.dumps(r) + "\n" for r in out))
    return out


def rows() -> list[dict]:
    if "rows" in _IN_PROCESS:
        return _IN_PROCESS["rows"]
    if not _ROWS.exists():
        _IN_PROCESS["rows"] = []
        return []
    _IN_PROCESS["rows"] = [json.loads(l) for l in _ROWS.read_text().splitlines() if l.strip()]
    return _IN_PROCESS["rows"]


# What the member values ARE. The reforecast's APCP messages alternate 3- and 6-hour
# buckets, and `at_lead` picks the one whose window ENDS at lead*24 h, which is always the
# 6-hour bucket (18-24 h into that day). Decoded from the GRIB2 section 4 time ranges by
# a domain-5 audit: 50 of 50 messages read at lead*24 h are 6-hour windows. In an earlier
# version neither the state nor the question said so, and "at least 1 mm of
# precipitation 3 days after the forecast" reads as a daily total -- a quantity four times
# the window this target was computed over.
WINDOW_HOURS = 6


def _valid(r: dict) -> str:
    from datetime import datetime, timedelta

    t = datetime.strptime(r["cycle"], "%Y%m%d%H") + timedelta(days=r["lead"])
    return t.strftime("%Y-%m-%d %H:00Z")


def _state(r: dict) -> str:
    return json.dumps({
        "forecast_issued": f"{r['cycle'][:4]}-{r['cycle'][4:6]}-{r['cycle'][6:8]} "
                           f"{r['cycle'][8:]}:00Z",
        "lead_days": r["lead"],
        "latitude": r["lat"], "longitude": r["lon"],
        "precip_window": f"{WINDOW_HOURS} h ending {_valid(r)}",
        "ensemble_precip_mm": r["members_mm"],
    }, indent=2)


# ---- semantic variants -------------------------------------------------------------
#
# Four wordings of the *same* question is a near-paraphrase set, and a near-paraphrase is
# memorisable: whatever the words, the answer is still "the fraction of members at or
# above 1 mm", so nothing in the prompt has to be read except the state. Worse, one of
# the four wordings an earlier build's enrichment produced for this source ("Does at least
# one ensemble member forecast at least 1 mm ...?", 945 of the `probability_recovery`
# gate's 1,969 rows) asks a *different* function of the same members than `meta.p` encodes:
# its literal answer is `any(v >= 1 mm)`, which equals `wet / 5` only when the ensemble is
# unanimous, measured at 22.2 % of those rows. A wording that is answerable from the state
# by a different computation than the target is not a paraphrase; it is a wrong label.
#
# So the variants here are genuinely different *functions* of the same five numbers, each
# with a target this module derives exactly:
#
#   fraction  the share of members at or above a threshold -- a real forecast probability,
#             and the only shape that may carry `meta.p`. Three thresholds, so the same
#             gridpoint yields three different probabilities.
#   any       does *at least one* member reach the threshold. Deterministic, 0 or 1.
#   all       do *all* members reach it. Deterministic, 0 or 1.
#
# Same state, different question, different answer: a model that ignores the question
# cannot do better than the marginal on four of the five. That is what makes the criteria
# load-bearing here rather than decorative.
#
# Thresholds are 0.2, 1 and 5 mm. Measured over the 36,000 cached gridpoints: at 1 mm the
# fraction is saturated at 0 or 1 on 65.1 % of rows, at 5 mm on 88.3 %, and at 10 mm on
# 95.5 % -- a 10 mm question is a constant "no" dressed as a probability, so it is not
# asked. 0.2 mm (45.7 % saturated, mean 0.385) is the best-spread threshold the source
# has and is included precisely to put mass in the middle of the range, which is where
# the `probability_recovery` gate reads and where calibration is hard.
THRESHOLDS_MM = (0.2, 1.0, 5.0)

# The three question functions, as templates. Registered in
# lod/assets/phrasing_specs/d05.json, which is what the phrasing-bank build reads; with
# an empty bank `pick` returns the template, so this is exactly the wording an earlier
# build shipped.
TEMPLATE_IDS = {"fraction": "d05.gefs_share", "any": "d05.gefs_any", "all": "d05.gefs_all"}
TEMPLATES = {
    "fraction": ("Will this location see at least {mm} of precipitation in the {hours} "
                 "hours ending {day} after the forecast was issued?"),
    "any": ("Does at least one ensemble member forecast at least {mm} of precipitation at "
            "this location in the {hours} hours ending {day} after the forecast was issued?"),
    "all": ("Do all five ensemble members forecast at least {mm} of precipitation at this "
            "location in the {hours} hours ending {day} after the forecast was issued?"),
}
_BANK: list = []


def _bank():
    if not _BANK:
        from lod.phrasings import PhrasingBank
        _BANK.append(PhrasingBank.load())
    return _BANK[0]


@dataclass(frozen=True)
class Variant:
    """One question asked of a gridpoint's five member values.

    `kind` is the function of the members, not a wording: the wording may be rewritten
    downstream (`services/enrich` reads `question` back from a model), but the
    criterion lives in `descriptions` and `instructions`, which that path never replaces
    -- published descriptions win in its merge and `instructions` is not read back at
    all. So a reworded question cannot change what is being asked.
    """

    qid: str
    kind: str                       # "fraction" | "any" | "all"
    threshold: float
    options: tuple[str, str]

    def question(self, lead: int, key: str = "") -> str:
        """The question, worded from the phrasing bank when it has this template. `key`
        is the record's identity, so a rebuild picks the same wording for it."""
        day = f"{lead} day" + ("" if lead == 1 else "s")
        mm = f"{self.threshold:g} mm"
        tid = TEMPLATE_IDS[self.kind]
        return _bank().pick(tid, TEMPLATES[self.kind], f"{key}:{self.qid}").format(
            mm=mm, day=day, hours=WINDOW_HOURS)

    def instructions(self) -> str:
        """The decision rule, in the one field the enrichment runner never rewrites."""
        mm = f"{self.threshold:g} mm"
        if self.kind == "fraction":
            return (f"Count how many of the five ensemble member values are at least "
                    f"{mm}. Answer with that count as a share of five -- this is a "
                    f"probability, not a yes/no reading of any single member.")
        if self.kind == "any":
            return (f"Answer yes if any one of the five ensemble member values is at "
                    f"least {mm}, no if every one of them is below it. This is not the "
                    f"forecast probability; one wet member out of five is still one "
                    f"member.")
        return (f"Answer yes only if every one of the five ensemble member values is at "
                f"least {mm}. A single member below it makes the answer no.")

    def descriptions(self) -> list[str]:
        mm = f"{self.threshold:g} mm"
        if self.kind == "fraction":
            return [f"The location does not reach {mm} of precipitation over the "
                    f"{WINDOW_HOURS}-hour window.",
                    f"The location reaches at least {mm} of precipitation over the "
                    f"{WINDOW_HOURS}-hour window."]
        if self.kind == "any":
            return [f"Every one of the five ensemble members forecasts less than {mm}.",
                    f"At least one of the five ensemble members forecasts {mm} or more."]
        return [f"At least one of the five ensemble members forecasts less than {mm}.",
                f"All five ensemble members forecast {mm} or more."]

    def target(self, members: list[float]) -> "int | list[float]":
        """The answer, derived from the members. Never guessed, never model-written."""
        wet = sum(1 for v in members if v >= self.threshold)
        if self.kind == "fraction":
            p = wet / len(members)
            return [1.0 - p, p]
        if self.kind == "any":
            return int(wet >= 1)
        return int(wet == len(members))

    def probability(self, members: list[float]) -> "float | None":
        """`meta.p` for this variant, or None where the target is not a probability."""
        if self.kind != "fraction":
            return None
        return sum(1 for v in members if v >= self.threshold) / len(members)


def variants() -> list[Variant]:
    """The question set asked of every gridpoint, in a fixed order.

    Every variant gets its *own* option keys. `enrich.schemas.collect_schemas` keys a
    schema on `(task, option tuple)` and rewrites one question per schema, so two
    variants sharing `["no", "yes"]` inside one task would be handed the same rewritten
    wording -- the "all members" question would reach the model asking about the
    probability, with the deterministic target still attached. Distinct keys make that
    collision impossible rather than unlikely.
    """
    out: list[Variant] = []
    for t in THRESHOLDS_MM:
        tag = f"{t:g}".replace(".", "p")
        opts = (("no", "yes") if t == RAIN_MM
                else (f"below_{tag}mm", f"at_least_{tag}mm"))
        out.append(Variant(f"pop_{tag}mm", "fraction", t, opts))
    out.append(Variant("any_member_5mm", "any", 5.0,
                       ("no_member_reaches_it", "at_least_one_member_reaches_it")))
    # No "all members >= 1 mm" variant: it answered "no" on 3,247 of 3,330 built
    # questions (97.5 %), a constant target. The "all" kind stays available.
    return out


VARIANTS = variants()
# the variant whose target is the 1 mm probability and whose realised outcome this module
# actually verified. Only this one may carry `meta.outcome`: the reforecast verification
# was run at 1 mm (`RAIN_MM`) and at no other threshold, so an outcome on the 0.2 mm or
# 5 mm question would be a number nobody measured.
VERIFIED = "pop_1mm"


def questions_for(r: dict, lead: int) -> list[Question]:
    """Every variant's question for one gridpoint row, with code-derived targets."""
    members = [float(v) for v in r["members_mm"]]
    out: list[Question] = []
    for v in VARIANTS:
        p = v.probability(members)
        meta: dict | None = None
        target: "int | list[float]" = v.target(members)
        if p is not None:
            meta = {"p": p}
            if v.qid == VERIFIED:
                # `generate.soften_meta_p` turns a hard target into [1-p, p] and
                # moves the draw to `meta.outcome`; handing it the outcome here keeps
                # that one variant byte-identical to what row 27 has always shipped.
                target = r["rained"]
        out.append(Question(
            id=v.qid,
            question=v.question(lead, key=f"{r['cycle']}:{r['lat']}:{r['lon']}:{lead}"),
            options=list(v.options),
            target=target,
            meta=meta,
            instructions=v.instructions(),
            descriptions=v.descriptions(),
        ))
    return out


def spread(lead: int) -> list[dict]:
    """This lead's rows, one gridpoint per (cycle, lat, lon), interleaved across cycles.

    The cache is written cycle by cycle, 900 points each, and the loader used to take the
    first `n` rows: at the default quota that is 666 rows, all from the first cycle, so every
    GEFS task in an earlier build was one forecast date (2019-01-01) -- one weather regime,
    one season, and the `probability_recovery` gate's day-7/day-10 probe was that same date
    again (domain-5 audit: 5 of 5 tasks). Round-robin over cycles
    makes any prefix cover every date. The sampler also drew gridpoints with replacement,
    so a few points repeat within a cycle; a repeat is the same record twice and is dropped.
    """
    by_cycle: dict[str, list[dict]] = {}
    seen: set = set()
    for r in rows():
        if r["lead"] != lead:
            continue
        key = (r["cycle"], r["lat"], r["lon"])
        if key in seen:
            continue
        seen.add(key)
        by_cycle.setdefault(r["cycle"], []).append(r)
    out: list[dict] = []
    queues = [by_cycle[c] for c in sorted(by_cycle)]
    for i in range(max((len(q) for q in queues), default=0)):
        out.extend(q[i] for q in queues if i < len(q))
    return out


def _loader(lead: int):
    def load(n: int) -> Iterator[Example]:
        # `n` is an example quota but the corpus budget is counted in *questions*, and a
        # gridpoint now carries five. Yielding n // len(THRESHOLDS_MM) examples keeps the
        # number of `meta.p` questions this task contributes exactly what it was before
        # the variants existed, so row 27 cannot quietly grow past its share.
        limit = max(1, n // len(THRESHOLDS_MM))
        made = 0
        for r in spread(lead):
            if made >= limit:
                return
            made += 1
            yield Example(
                task=f"gefs_pop_day{lead}",
                state=_state(r),
                questions=questions_for(r, lead),
            )
    return load


# The `probability_recovery` gate (probability recovery on unseen forecast families)
# previously read zero GEFS rows: without a forced split, every lead but day5 hashed to
# train and day5 hashed to devreal, so none reached testreal. A held-out *lead time* is a
# real transfer question -- "will a probability recovered at short lead (1-3 days) still be
# recovered at long lead (7-10)", a genuinely different forecasting problem per this
# module's own docstring -- so day7 and day10 are now forced to testreal (two long leads,
# so the probe is not one family alone), day5 stays the devreal check, and day1/day3
# anchor training at the short end where the ensemble spread is tightest and the signal
# is strongest.
RESERVED_TESTREAL_LEADS = (7, 10)
RESERVED_DEVREAL_LEADS = (5,)


def tasks() -> list[RealTask]:
    if not _ROWS.exists():
        return []
    present = {r["lead"] for r in rows()}
    out = []
    for d in sorted(present):
        if d in RESERVED_TESTREAL_LEADS:
            force = "testreal"
        elif d in RESERVED_DEVREAL_LEADS:
            force = "devreal"
        else:
            force = "train"
        out.append(RealTask(row=27, name=f"gefs_pop_day{d}", licence=LICENCE, url=URL,
                            load=_loader(d), meta_p=True, soft=True, force_split=force,
                            template_state=True,
                            notes=f"GEFS v12 reforecast, {d}-day lead; five questions per "
                                  f"gridpoint -- the member share at "
                                  f"{', '.join(f'{t:g}' for t in THRESHOLDS_MM)} mm "
                                  f"(meta.p), plus a deterministic any-member and "
                                  f"all-members question; meta.p = share of 5 "
                                  f"ensemble members forecasting >= the threshold; verified "
                                  f"against a short-lead forecast, not a gauge observation; "
                                  f"forced to {force} -- day{{7,10}} are the probability_recovery gate's held-out long "
                                  f"leads, day5 is the devreal check, day{{1,3}} train"))
    return out
