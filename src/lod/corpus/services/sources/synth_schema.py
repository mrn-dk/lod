"""synth_schema — procedurally invented schemas with executable ground truth.

The out-of-distribution bet is that *many* schemas teach the model to read an
arbitrary question and an arbitrary option list rather than memorise 23 label
sets. Real corpora give us schemas slowly; this generator gives them by the
thousand, and every label is computed, so there is no annotation noise.

A *world* is a sampled record: an entity from a domain pack, a subset of its
fields, each field's value drawn from a subsampled vocabulary. Questions are
then asked about that record in six state formats. A schema is
`(question text, option set)`; because field names, vocabulary subsets and
action names are all sampled, the schema space is combinatorially large.

Disjointness for held-out: `generate(..., split="heldout")` draws domains from a
reserved slice of the domain list and action/category names from a reserved
slice of their pools, so a held-out schema shares no domain, no field name and
no option vocabulary with a training schema. That is the level-2 (source-level)
"whole generator family never seen" test, applied inside one generator.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field as dc_field
from types import SimpleNamespace

from lod.schema import Example, Question
from lod.corpus.services.sources.base import SourceInfo, SourceResult, schema_key

YN = ["no", "yes"]

# --------------------------------------------------------------------------
# vocabulary pools. A topic is a *pool*; a world subsamples it, so the option
# set a question ranges over is itself sampled, not fixed.
# --------------------------------------------------------------------------

TOPICS: dict[str, list[str]] = {
    "status": ["open", "pending", "resolved", "escalated", "archived", "draft", "blocked", "cancelled"],
    "severity": ["trivial", "minor", "moderate", "major", "critical", "catastrophic"],
    "region": ["north", "south", "east", "west", "central", "coastal", "highland", "offshore"],
    "material": ["steel", "aluminium", "copper", "titanium", "polymer", "ceramic", "glass", "composite"],
    "colour": ["amber", "crimson", "indigo", "olive", "slate", "teal", "violet", "ochre"],
    "channel": ["email", "phone", "chat", "postal", "portal", "sms", "fax", "kiosk"],
    "department": ["billing", "logistics", "research", "compliance", "facilities", "procurement",
                   "payroll", "security"],
    "condition": ["nominal", "degraded", "failing", "offline", "recovering", "unknown"],
    "tier": ["bronze", "silver", "gold", "platinum", "basic", "premium", "enterprise"],
    "frequency": ["hourly", "daily", "weekly", "monthly", "quarterly", "annually"],
    "medium": ["paper", "digital", "microfilm", "tape", "disc", "cloud"],
    "climate": ["arid", "humid", "temperate", "alpine", "tropical", "polar"],
    "species": ["oak", "birch", "cedar", "willow", "maple", "spruce", "alder"],
    "cuisine": ["nordic", "levantine", "andean", "creole", "alpine", "coastal"],
    "phase": ["intake", "triage", "review", "approval", "dispatch", "closure", "audit"],
    "carrier": ["rail", "barge", "truck", "airfreight", "courier", "pipeline"],
    "licence": ["permissive", "restricted", "embargoed", "public", "internal", "classified"],
    "unitclass": ["metric", "imperial", "nautical", "astronomical"],
    "origin": ["domestic", "imported", "reclaimed", "synthetic", "donated", "seized"],
    "finish": ["matte", "gloss", "brushed", "anodised", "powdered", "raw"],
    "risk": ["negligible", "low", "guarded", "elevated", "severe"],
    "ownership": ["leased", "owned", "shared", "franchised", "trust-held"],
    "verdict": ["approved", "rejected", "deferred", "withdrawn", "amended"],
    "habitat": ["wetland", "grassland", "woodland", "dune", "reef", "tundra"],
    "grade": ["prototype", "pilot", "production", "legacy", "retired"],
    "protocol": ["ambient", "chilled", "frozen", "pressurised", "inert", "shielded"],
    "source": ["survey", "sensor", "interview", "ledger", "satellite", "manifest"],
    "cadence": ["continuous", "batched", "on-demand", "scheduled", "triggered"],
    "posture": ["permissive", "balanced", "strict", "lockdown"],
    "locale": ["harbour", "depot", "annex", "vault", "quarry", "hangar", "silo"],
}

# field names are sampled per world, so a schema's question text varies too
FIELD_NAMES: dict[str, list[str]] = {
    "status": ["status", "state", "disposition", "stage", "standing"],
    "severity": ["severity", "impact", "seriousness", "gravity"],
    "region": ["region", "zone", "district", "sector", "territory"],
    "material": ["material", "substrate", "composition", "stock"],
    "colour": ["colour", "finish colour", "tint", "livery"],
    "channel": ["channel", "intake channel", "contact route", "submission route"],
    "department": ["department", "owning team", "business unit", "desk"],
    "condition": ["condition", "health", "operating condition", "serviceability"],
    "tier": ["tier", "service tier", "plan", "grade of service"],
    "frequency": ["frequency", "cadence", "review interval", "reporting period"],
    "medium": ["medium", "storage medium", "format", "carrier medium"],
    "climate": ["climate", "climate band", "environment"],
    "species": ["species", "timber type", "stock species"],
    "cuisine": ["cuisine", "kitchen style", "menu origin"],
    "phase": ["phase", "workflow phase", "current step", "lifecycle stage"],
    "carrier": ["carrier", "transport mode", "haulage", "shipping mode"],
    "licence": ["licence class", "access class", "handling class"],
    "unitclass": ["unit system", "measurement system"],
    "origin": ["origin", "provenance", "sourcing"],
    "finish": ["finish", "surface treatment", "coating"],
    "risk": ["risk band", "exposure", "threat level"],
    "ownership": ["ownership", "tenure", "holding type"],
    "verdict": ["verdict", "outcome", "decision"],
    "habitat": ["habitat", "biotope", "terrain"],
    "grade": ["grade", "maturity", "build grade"],
    "protocol": ["handling protocol", "storage protocol", "transport protocol"],
    "source": ["source", "evidence source", "data origin"],
    "cadence": ["cadence", "delivery mode", "processing mode"],
    "posture": ["posture", "policy posture", "enforcement level"],
    "locale": ["locale", "site", "facility", "holding point"],
}

NUM_FIELDS: list[tuple[str, int, int, str]] = [
    ("weight", 1, 900, "kg"), ("volume", 1, 400, "litres"), ("age", 0, 60, "months"),
    ("depth", 1, 250, "metres"), ("retries", 0, 12, ""), ("headcount", 1, 400, ""),
    ("backlog", 0, 500, "items"), ("latency", 5, 900, "ms"), ("temperature", -30, 95, "C"),
    ("pressure", 1, 300, "bar"), ("duration", 1, 240, "hours"), ("distance", 1, 1500, "km"),
    ("quantity", 1, 999, "units"), ("balance", 0, 9000, "EUR"), ("altitude", 0, 4000, "metres"),
    ("humidity", 5, 99, "%"), ("voltage", 3, 480, "V"), ("yield", 1, 100, "%"),
    ("cycles", 1, 5000, ""), ("defects", 0, 40, ""), ("occupancy", 0, 100, "%"),
    ("turnover", 0, 5000, "EUR"), ("span", 1, 90, "days"), ("throughput", 1, 800, "per hour"),
]

ENTITIES: list[str] = [
    "consignment", "work order", "specimen", "permit", "inspection", "tenancy", "vessel",
    "batch", "claim", "asset", "reservation", "survey", "shipment", "installation",
    "licence application", "maintenance job", "grant", "lot", "enrolment", "dispatch",
    "sample", "filing", "contract", "requisition", "incident", "appraisal", "berth",
    "consultation", "haul", "pallet", "subscription", "field trial", "audit record",
    "quarantine hold", "salvage item", "lease", "expedition", "kiln run", "moorage",
    "rota entry",
]

ACTIONS: list[tuple[str, str]] = [
    ("escalated to a supervisor", "Should this be escalated to a supervisor?"),
    ("held for manual review", "Should this be held for manual review?"),
    ("released automatically", "Can this be released automatically?"),
    ("flagged for audit", "Should this be flagged for audit?"),
    ("routed to the priority queue", "Does this belong in the priority queue?"),
    ("re-inspected", "Does this require a re-inspection?"),
    ("insured separately", "Must this be insured separately?"),
    ("quarantined", "Should this be quarantined?"),
    ("expedited", "Should this be expedited?"),
    ("refused at intake", "Should this be refused at intake?"),
    ("archived immediately", "Can this be archived immediately?"),
    ("assigned a second approver", "Does this need a second approver?"),
    ("logged as an exception", "Should this be logged as an exception?"),
    ("charged the surcharge", "Does the surcharge apply?"),
    ("scheduled for disposal", "Should this be scheduled for disposal?"),
    ("sent to the overflow site", "Should this go to the overflow site?"),
]

# derived-category questions: a named scale whose bands are invented per world
BANDS: list[tuple[str, list[str]]] = [
    ("handling band", ["routine", "supervised", "restricted"]),
    ("clearance level", ["open", "limited", "sealed"]),
    ("routing class", ["standard", "expedited", "bespoke"]),
    ("storage class", ["ambient", "controlled", "vaulted"]),
    ("fee band", ["waived", "reduced", "full", "premium"]),
    ("review depth", ["spot check", "partial", "full audit"]),
    ("crew size", ["solo", "pair", "team"]),
    ("packaging class", ["loose", "boxed", "crated", "palletised"]),
    ("escort requirement", ["none", "single escort", "convoy"]),
    ("notice period", ["same day", "two days", "one week"]),
]

SCORE_NAMES: list[str] = [
    "condition score", "priority score", "readiness rating", "confidence grade",
    "completeness score", "urgency rating", "suitability score", "compliance rating",
]

FORMATS = ("json", "kv", "prose", "yaml", "table", "logline")


# --------------------------------------------------------------------------
# world sampling
# --------------------------------------------------------------------------


@dataclass
class World:
    entity: str
    cats: list[tuple[str, str, list[str], str]]  # (field name, topic, option pool, value)
    nums: list[tuple[str, int, str]]             # (field name, value, unit)
    fmt: str
    ident: str
    schemas: set = dc_field(default_factory=set)


def _pool(rng: random.Random, names: list[str], lo: int, hi: int) -> list[str]:
    """A schema's option set is a *subsample* of the topic pool, not the pool."""
    k = min(len(names), rng.randint(lo, hi))
    return rng.sample(names, k)


def sample_world(rng: random.Random, topics: list[str], entities: list[str],
                 num_fields: list[tuple[str, int, int, str]] = NUM_FIELDS) -> World:
    entity = rng.choice(entities)
    chosen = rng.sample(topics, min(len(topics), rng.randint(3, 5)))
    cats = []
    for t in chosen:
        name = rng.choice(FIELD_NAMES[t])
        opts = _pool(rng, TOPICS[t], 3, min(7, len(TOPICS[t])))
        cats.append((name, t, opts, rng.choice(opts)))
    nums = []
    band_lo, band_hi = rng.choice([(1, 60), (10, 500), (100, 5000)])
    for nm, lo, hi, unit in rng.sample(num_fields, rng.randint(2, min(3, len(num_fields)))):
        a, b = max(lo, band_lo), min(hi, band_hi)
        nums.append((nm, rng.randint(a, b) if a <= b else rng.randint(lo, hi), unit))
    ident = f"{rng.choice('ABCDEFGHJKLMNPRSTVWXYZ')}{rng.randint(100, 999)}-{rng.randint(10, 99)}"
    return World(entity, cats, nums, rng.choice(FORMATS), ident)


def _num(v: int, unit: str) -> str:
    return f"{v} {unit}".strip()


def _a(word: str) -> str:
    return ("an " if word[:1].lower() in "aeiou" else "a ") + word


def render(w: World, hide: set[str]) -> str:
    """The record in one of six state formats, with `hide` fields removed."""
    cats = [(n, v) for n, _t, _o, v in w.cats if n not in hide]
    nums = [(n, _num(v, u)) for n, v, u in w.nums if n not in hide]
    pairs = [(w.entity + " id", w.ident)] + cats + nums
    if w.fmt == "json":
        body = {k.replace(" ", "_"): v for k, v in pairs}
        return json.dumps(body, indent=2)
    if w.fmt == "yaml":
        return "\n".join(f"{k.replace(' ', '_')}: {v}" for k, v in pairs)
    if w.fmt == "kv":
        return "\n".join(f"{k}: {v}" for k, v in pairs)
    if w.fmt == "table":
        keys = [k for k, _ in pairs]
        vals = [v for _, v in pairs]
        return ("| " + " | ".join(keys) + " |\n|"
                + "|".join("---" for _ in keys) + "|\n| " + " | ".join(vals) + " |")
    if w.fmt == "logline":
        return (f"record {w.ident} "
                + " ".join(f"{k.replace(' ', '_')}={v}" for k, v in pairs[1:]))
    sent = [f"This {w.entity} is recorded as {w.ident}."]
    for k, v in cats:
        sent.append(f"Its {k} is {v}.")
    for k, v in nums:
        sent.append(f"The {k} is {v}.")
    return " ".join(sent)


# --------------------------------------------------------------------------
# question builders. Each returns (Question, schema_key) or None.
# --------------------------------------------------------------------------

CLOZE_TEMPLATES = [
    "What is the {f} of this {e}?",
    "Which {f} does this {e} have?",
    "According to the record, what is the {e}'s {f}?",
    "Select the {f} recorded for this {e}.",
]


def q_cloze(w: World, rng: random.Random) -> tuple[Question, str] | None:
    """Structural: the answer is a field removed from the state."""
    if not w.cats:
        return None
    name, _t, opts, val = rng.choice(w.cats)
    text = rng.choice(CLOZE_TEMPLATES).format(f=name, e=w.entity)
    return Question(f"cloze_{name}", text, list(opts), opts.index(val)), name


def q_threshold(w: World, rng: random.Random) -> tuple[Question, str] | None:
    """Executable: compare a numeric field against a stated bound."""
    if not w.nums:
        return None
    name, val, unit = rng.choice(w.nums)
    lo = max(1, int(val * 0.4)) if val > 2 else 1
    hi = max(lo + 1, int(abs(val) * 1.6) + 2)
    bound = rng.randint(lo, hi)
    above = rng.random() < 0.5
    if above:
        text = rng.choice([f"Is the {name} above {_num(bound, unit)}?",
                           f"Does the {name} exceed {_num(bound, unit)}?"])
        ans = int(val > bound)
    else:
        text = rng.choice([f"Is the {name} at most {_num(bound, unit)}?",
                           f"Is the {name} {_num(bound, unit)} or less?"])
        ans = int(val <= bound)
    return Question(f"thr_{name}", text, list(YN), ans), None


def q_compare(w: World, rng: random.Random) -> tuple[Question, str] | None:
    """Executable: which of the record's numeric fields is largest."""
    if len(w.nums) < 2:
        return None
    opts = [n for n, _v, _u in w.nums]
    vals = [v for _n, v, _u in w.nums]
    if len(set(vals)) != len(vals):
        return None
    pick_max = rng.random() < 0.5
    tgt = vals.index(max(vals) if pick_max else min(vals))
    text = (f"Which field of this {w.entity} holds the "
            f"{'largest' if pick_max else 'smallest'} number?")
    return Question("cmp", text, opts, tgt), None


def _cond(w: World, rng: random.Random, used: set[str], avoid: set[str],
          want_true: bool) -> tuple[str, bool, str] | None:
    """One atomic policy condition in prose, its truth value, and the field it reads.

    The condition is *built to* `want_true` rather than sampled and measured.
    Sampling gives 66 % "no": "the {field} is {value}" holds only 1/k of the
    time for a k-option field, and an "and" of two such clauses is rarer still,
    so the model could score 0.66 on every policy question without reading one.

    `used` keeps a two-clause policy from testing the same field twice (which
    makes one clause dead); `avoid` keeps it off fields hidden from the state.
    """
    cats = [c for c in w.cats if c[0] not in used and c[0] not in avoid]
    nums = [n for n in w.nums if n[0] not in used]
    if cats and (not nums or rng.random() < 0.5):
        name, _t, opts, val = rng.choice(cats)
        if want_true:
            want = val
        else:
            others = [o for o in opts if o != val]
            if not others:
                return None
            want = rng.choice(others)
        return f"the {name} is {want}", want == val, name
    if not nums:
        return None
    name, val, unit = rng.choice(nums)
    # "exceeds bound" should be true iff want_true, so put the bound the right
    # side of the value; keep it a plausible-looking round-ish number
    if want_true:
        bound = rng.randint(max(0, val - max(2, abs(val) // 2)), max(0, val - 1)) if val > 0 else 0
        if bound >= val:
            return None
    else:
        bound = rng.randint(val, val + max(2, abs(val) // 2) + 3)
        if bound < val:
            return None
    return f"the {name} exceeds {_num(bound, unit)}", val > bound, name


def q_policy(w: World, rng: random.Random, actions: list[tuple[str, str]],
             avoid: set[str]) -> tuple[Question, str, set[str]] | None:
    """Executable: an explicit policy in the state decides a yes/no action."""
    action, text = rng.choice(actions)
    used: set[str] = set()
    # choose the answer first, then build clauses that produce it: 50/50 by
    # construction, for both the one-clause and the two-clause form
    truth = rng.random() < 0.5
    two = rng.random() < 0.55
    op = "and" if rng.random() < 0.5 else "or"
    if not two:
        want1, want2 = truth, None
    elif op == "and":
        want1, want2 = (True, True) if truth else rng.choice([(True, False), (False, True),
                                                              (False, False)])
    else:
        want1, want2 = (False, False) if not truth else rng.choice([(True, True), (True, False),
                                                                    (False, True)])
    first = _cond(w, rng, used, avoid, want1)
    if first is None:
        return None
    c1, v1, f1 = first
    used.add(f1)
    clause = c1
    if want2 is not None:
        second = _cond(w, rng, used, avoid, want2)
        if second is None:
            return None
        c2, v2, f2 = second
        used.add(f2)
        clause = f"{c1} {op} {c2}"
        truth = (v1 and v2) if op == "and" else (v1 or v2)
    else:
        truth = v1
    policy = f"Policy: {_a(w.entity)} is {action} when {clause}."
    return Question("policy", text, list(YN), int(truth)), policy, used


def q_band(w: World, rng: random.Random, bands: list[tuple[str, list[str]]]) -> tuple[Question, str] | None:
    """Executable: a stated banding rule over a numeric field names one of k classes."""
    if not w.nums:
        return None
    name, val, unit = rng.choice(w.nums)
    band_name, labels = rng.choice(bands)
    labels = list(labels)
    rng.shuffle(labels)
    lo = max(1, int(abs(val) * 0.3) + 1)
    cuts = sorted(rng.sample(range(lo, lo + max(4, int(abs(val) * 1.5) + 5)), len(labels) - 1))
    idx = len(cuts)
    for i, c in enumerate(cuts):
        if val <= c:
            idx = i
            break
    parts = [f"{labels[0]} when the {name} is at most {_num(cuts[0], unit)}"]
    for i in range(1, len(cuts)):
        parts.append(f"{labels[i]} up to {_num(cuts[i], unit)}")
    parts.append(f"otherwise {labels[-1]}")
    rule = f"Rule: the {band_name} is " + ", ".join(parts) + "."
    text = f"Which {band_name} applies to this {w.entity}?"
    return Question("band", text, labels, idx), rule


def q_score(w: World, rng: random.Random, names: list[str]) -> tuple[Question, str] | None:
    """Ordinal: a 1..N score defined by a stated formula (the ordinal label axis)."""
    if not w.nums:
        return None
    name, val, unit = rng.choice(w.nums)
    top = rng.choice([4, 5, 5, 6])
    if val < 0:
        return None
    # score(step) = min(top, val//step + 1). Invert it: collect the step range
    # that lands on each score, then choose the score uniformly. Sampling `step`
    # instead put 42 % of the mass on the top band.
    choices: list[tuple[int, int, int]] = []          # (score, step_lo, step_hi)
    for sc in range(1, top + 1):
        if sc == 1:
            lo, hi = val + 1, val + 1 + max(5, val // 2)
        elif sc == top:
            lo, hi = 1, val // (top - 1)
        else:
            lo, hi = val // sc + 1, val // (sc - 1)
        if lo <= hi and lo >= 1:
            choices.append((sc, lo, hi))
    if not choices:
        return None
    score, lo, hi = rng.choice(choices)
    step = rng.randint(lo, hi)
    score_name = rng.choice(names)
    rule = (f"Scale: the {score_name} is 1 for a {name} under {_num(step, unit)}, "
            f"and rises by one for every further {_num(step, unit)}, capped at {top}.")
    text = f"What {score_name} does this {w.entity} get?"
    return Question("score", text, [str(i) for i in range(1, top + 1)], score - 1), rule


def q_absence(w: World, rng: random.Random) -> tuple[Question, str] | None:
    """Type-safety probe: the honest answer is that the record does not say."""
    if not w.cats:
        return None
    name, _t, opts, val = rng.choice(w.cats)
    present = rng.random() < 0.5
    opts2 = list(opts) + ["not stated"]
    text = f"What is the {name} of this {w.entity}, if the record states it?"
    return Question(f"abs_{name}", text, opts2,
                    opts2.index(val) if present else len(opts2) - 1), (None if present else name)


# --------------------------------------------------------------------------
# generation
# --------------------------------------------------------------------------

BUILDERS = ("cloze", "threshold", "compare", "policy", "band", "score", "absence")


def _one(rng: random.Random, topics, entities, actions, bands, scores, nums,
         family: str = "synth_schema") -> Example | None:
    """One record plus 2-5 questions about it.

    Decidability is structural, not checked afterwards: the field-removing
    questions (cloze, absence) run first and fix `hide`, and every later
    question is built with `hide` excluded, so no question can ever depend on a
    field the state does not show. `claimed` stops two questions owning the
    same field, which would make one of them unanswerable.
    """
    w = sample_world(rng, topics, entities, nums)
    kinds = rng.sample(BUILDERS, rng.randint(2, 5))
    questions: list[Question] = []
    hide: set[str] = set()
    claimed: set[str] = set()
    preamble: list[str] = []

    for kind in [k for k in kinds if k in ("cloze", "absence")]:
        got = q_cloze(w, rng) if kind == "cloze" else q_absence(w, rng)
        if got is None:
            continue
        q, fname = got
        # q_absence returns None when it answers from a *visible* field; it
        # still owns that field, which its question id carries
        owner = fname or q.id.removeprefix("abs_")
        if owner in claimed:
            continue
        claimed.add(owner)
        if fname:
            hide.add(fname)
        questions.append(q)

    for kind in [k for k in kinds if k not in ("cloze", "absence")]:
        if kind == "threshold":
            got = q_threshold(w, rng)
        elif kind == "compare":
            got = q_compare(w, rng)
        elif kind == "band":
            got = q_band(w, rng, bands)
        elif kind == "score":
            got = q_score(w, rng, scores)
        else:
            pol = q_policy(w, rng, actions, hide)
            if pol is None:
                continue
            q, rule, _used = pol
            preamble.append(rule)
            questions.append(q)
            continue
        if got is None:
            continue
        q, rule = got
        if rule:
            preamble.append(rule)
        questions.append(q)

    if not questions:
        return None
    body = render(w, hide)
    state = ("\n".join(preamble) + "\n\n" + body) if preamble else body
    return Example(task=f"{family}:{w.fmt}", state=state, questions=questions)


def _generate(n: int, seed: int, split: str, info: SourceInfo,
              fmts: tuple[str, ...] | None = None) -> SourceResult:
    rng = random.Random(seed)
    # disjoint slices so a held-out schema shares no domain, field or option set
    # Three disjoint slices of every pool. A held-out schema then shares no
    # entity, no field name, no option vocabulary and no action with a training
    # schema, and dev-OOD (used for checkpoint selection and temperature) is
    # unseen too without being the held-out set we report.
    topics = sorted(TOPICS)
    if split == "heldout":
        topics, ents, nums = topics[-8:], ENTITIES[-8:], NUM_FIELDS[-6:]
        acts, bands, scores = ACTIONS[-4:], BANDS[-3:], SCORE_NAMES[-2:]
    elif split == "devood":
        topics, ents, nums = topics[-14:-8], ENTITIES[-14:-8], NUM_FIELDS[-11:-6]
        acts, bands, scores = ACTIONS[-7:-4], BANDS[-5:-3], SCORE_NAMES[-3:-2]
    elif split == "train":
        topics, ents, nums = topics[:-14], ENTITIES[:-14], NUM_FIELDS[:-11]
        acts, bands, scores = ACTIONS[:-7], BANDS[:-5], SCORE_NAMES[:-3]
    else:
        ents, acts, bands, scores, nums = ENTITIES, ACTIONS, BANDS, SCORE_NAMES, NUM_FIELDS
    res = SourceResult(info=info)
    tries = 0
    while len(res.examples) < n and tries < n * 12:
        tries += 1
        ex = _one(rng, topics, ents, acts, bands, scores, nums, info.name)
        if ex is None or (fmts and not ex.task.endswith(tuple(fmts))):
            continue
        res.examples.append(ex)
        for q in ex.questions:
            res.schemas.add(schema_key(q.question, q.options))
    return res


def _info(name: str, state_format: str, notes: str) -> SourceInfo:
    return SourceInfo(name=name, url="generated", licence="CC0-1.0 (generated here)", tier=1,
                      ground_truth_type="executable", state_format=state_format,
                      option_type="fixed", notes=notes)


INFO = _info("synth_schema", "structured", "procedurally invented schemas, computed labels")


def generate(n: int, seed: int = 0, split: str = "train") -> SourceResult:
    return _generate(n, seed, split, INFO, ("json", "kv", "yaml"))


def _variant(name: str, fmts: tuple[str, ...], state_format: str, notes: str,
             force_split: str | None = None):
    info = _info(name, state_format, notes)

    def gen(n: int, seed: int = 0, split: str = "train",
            _i=info, _f=fmts, _fs=force_split) -> SourceResult:
        return _generate(n, seed, _fs or split, _i, _f)

    return SimpleNamespace(INFO=info, generate=gen)


# Registered separately so the provenance manifest carries a
# state_format row per family and the axis-coverage report can see prose and tabular
# covered.
#
# The three held-out levels (task, source, format-shift) are all measurable inside this one
# generator, which is the point of building it:
#   level 2 (source-level)  synth_schema_ood      unseen vocabulary slice
#   level 3 (format-shift)  synth_schema_fmtshift *trained* vocabulary slice
#                                                 rendered as a table or log
#                                                 line, formats never trained on
synth_schema_prose = _variant(
    "synth_schema_prose", ("prose",), "prose",
    "synth_schema rendered as prose sentences")
synth_schema_ood = _variant(
    "synth_schema_ood", FORMATS, "structured",
    "level-2 OOD: entities, fields, option vocabularies and actions never in train")
synth_schema_fmtshift = _variant(
    "synth_schema_fmtshift", ("table", "logline"), "tabular",
    "level-3 OOD: trained schemas re-rendered as markdown tables and log lines",
    force_split="train")
