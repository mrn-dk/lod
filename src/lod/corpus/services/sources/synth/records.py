"""Random records with distractor fields, rendered as JSON or as prose.

The state has to be long enough that the answer is not simply the whole state: under
about 40 tokens a question stops being "find the relevant field and apply the rule" and
becomes "there is only one number here". Every domain therefore carries eight to twelve
fields and a rule names one or two of them; the rest are distractors by construction.

Each record renders two ways from the same values. JSON is the shape the API contract is
written for; prose is the shape most callers actually have, and rendering the identical
record both ways is how the text path gets covered without a teacher model and without
a second label.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from datetime import date, timedelta


@dataclass(frozen=True)
class Num:
    name: str
    lo: float
    hi: float
    unit: str = ""
    integral: bool = True
    # (name of the same quantity in another unit, how many of those per one of these).
    # `unit_mismatch` rules are written against this one and the state carries the other.
    alt: tuple[str, float] | None = None
    label: str = ""
    gloss: str = ""


@dataclass(frozen=True)
class Enum:
    name: str
    values: tuple[str, ...]
    label: str = ""
    gloss: str = ""


@dataclass(frozen=True)
class Stamp:
    name: str
    label: str = ""
    gloss: str = ""


@dataclass(frozen=True)
class Items:
    name: str
    pool: tuple[str, ...]
    lo: int = 0
    hi: int = 6
    label: str = ""
    gloss: str = ""


@dataclass(frozen=True)
class Domain:
    name: str
    noun: str                     # "shipment", "sensor reading", ...
    nums: tuple[Num, ...]
    enums: tuple[Enum, ...]
    stamps: tuple[Stamp, ...] = ()
    items: tuple[Items, ...] = ()
    ids: tuple[str, ...] = ()
    # (measured field, its limit or target, how a rule refers to the comparison without
    # naming either). This is what lets a rule say "the reading is above its target" --
    # the shape the reefer probe is written in, and the shape an earlier model was never
    # shown: every rule it trained on named its field outright.
    pairs: tuple[tuple[str, str, str], ...] = ()

    @property
    def fields(self) -> list:
        return list(self.nums) + list(self.enums) + list(self.stamps) + list(self.items)


DOMAINS: tuple[Domain, ...] = (
    Domain(
        "shipment", "shipment",
        nums=(Num("gross_weight_kg", 0.4, 40.0, " kg", False, ("g", 1000.0),
                  gloss="how heavy the shipment is"),
              Num("weight_limit_kg", 0.4, 40.0, " kg", False,
                  label="weight limit", gloss="the most the shipment is allowed to weigh"),
              Num("declared_value_usd", 15, 4200, " USD",
                  gloss="what the shipment is declared to be worth"),
              Num("transit_days", 1, 21, " days", label="transit time",
                  gloss="how long the shipment spends in transit"),
              Num("parcel_count", 1, 9, gloss="how many parcels the shipment is made of"),
              Num("volume_litres", 2, 260, " L", False,
                  gloss="how much space the shipment takes up")),
        enums=(Enum("carrier", ("northline", "tessera", "harbourway", "kestrel", "ridgeway"),
                    gloss="which company is carrying it"),
               Enum("service_level", ("economy", "standard", "express", "same_day"),
                    gloss="how fast a service was paid for"),
               Enum("destination_zone", ("zone_a", "zone_b", "zone_c", "zone_d"),
                    gloss="which delivery zone it is going to"),
               Enum("customs_status", ("cleared", "pending", "held", "not_required"),
                    gloss="where it stands with customs")),
        stamps=(Stamp("dispatched_on", gloss="when it was sent"),),
        items=(Items("handling_flags",
                     ("fragile", "hazardous", "refrigerated", "oversize", "stackable",
                      "signature_required", "live_animal"),
                     gloss="what special handling it needs"),),
        ids=("consignment_ref",),
        pairs=(("gross_weight_kg", "weight_limit_kg", "its weight against its limit"),),
    ),
    Domain(
        "sensor", "sensor reading",
        nums=(Num("temperature_c", -30.0, 58.0, " °C", False,
                  gloss="how warm it is where the sensor sits"),
              Num("setpoint_c", -30.0, 58.0, " °C", False, label="setpoint",
                  gloss="the temperature it is supposed to hold"),
              Num("humidity_pct", 5, 99, " %", gloss="how damp the air is"),
              Num("battery_mv", 2400, 4200, " mV", True, ("V", 0.001),
                  label="battery level", gloss="how much charge is left"),
              Num("sample_interval_s", 5, 900, " s", True, ("minutes", 1 / 60.0),
                  gloss="how often it takes a reading"),
              Num("signal_dbm", -119, -41, " dBm", gloss="how strong its radio link is")),
        enums=(Enum("firmware_channel", ("stable", "beta", "canary", "legacy"),
                    gloss="which firmware track it is on"),
               Enum("mount", ("wall", "duct", "outdoor_pole", "handheld"),
                    gloss="how it is mounted"),
               Enum("link", ("lorawan", "wifi", "cellular", "wired"),
                    gloss="how it gets its data out")),
        stamps=(Stamp("last_calibrated_on", gloss="when it was last calibrated"),),
        items=(Items("active_alarms",
                     ("low_battery", "out_of_range", "tamper", "drift", "no_uplink"),
                     gloss="which alarms are currently raised"),),
        ids=("device_id",),
        pairs=(("temperature_c", "setpoint_c",
                "the temperature it reports against the temperature it should hold"),),
    ),
    Domain(
        "incident", "incident ticket",
        nums=(Num("affected_users", 1, 42000, gloss="how many people it is affecting"),
              Num("open_minutes", 3, 5400, " min", True, ("hours", 1 / 60.0),
                  label="open time", gloss="how long it has been open"),
              Num("sla_minutes", 3, 5400, " min", label="SLA",
                  gloss="how long it is allowed to stay open"),
              Num("restarts", 0, 14, gloss="how many times it has been restarted"),
              Num("error_rate_pct", 0, 97, " %", False,
                  gloss="what share of requests are failing")),
        enums=(Enum("component", ("ingest", "scheduler", "billing_api", "search",
                                  "notifier", "object_store"),
                    gloss="which part of the system it is in"),
               Enum("environment", ("production", "staging", "canary"),
                    gloss="which environment it is in"),
               Enum("reported_by", ("customer", "synthetic_probe", "on_call", "partner"),
                    gloss="who raised it"),
               Enum("paging_policy", ("follow_the_sun", "business_hours", "always"),
                    gloss="when it is allowed to page someone")),
        stamps=(Stamp("opened_on", gloss="when it was raised"),),
        items=(Items("linked_changes", ("deploy_9931", "config_4417", "schema_221",
                                        "rollback_88", "flag_flip_3"),
                     gloss="which changes are linked to it"),),
        ids=("ticket_key",),
        pairs=(("open_minutes", "sla_minutes",
                "how long it has been open against how long it is allowed"),),
    ),
    Domain(
        "build", "build job",
        nums=(Num("duration_s", 20, 5400, " s", True, ("minutes", 1 / 60.0),
                  gloss="how long it took to run"),
              Num("budget_s", 20, 5400, " s", label="time budget",
                  gloss="how long it was allowed to take"),
              # 1 GB = 1000 MB (SI). It was 1/1024, which is GiB: a rule saying "GB" is
              # read by most people -- and by the domain-2 audit's independent evaluator --
              # as the SI unit, and the two disagree by 2.4 % of the threshold.
              Num("artefact_size_mb", 1, 1800, " MB", False, ("GB", 1 / 1000.0),
                  gloss="how big the thing it produced is"),
              Num("test_failures", 0, 37, gloss="how many tests did not pass"),
              Num("changed_files", 1, 410, gloss="how much of the tree it touched"),
              Num("queue_wait_s", 0, 1700, " s", gloss="how long it waited to start")),
        enums=(Enum("runner_class", ("small", "medium", "large", "gpu"),
                    gloss="what size machine it ran on"),
               Enum("trigger", ("push", "pull_request", "schedule", "manual"),
                    gloss="what set it off"),
               Enum("branch_kind", ("main", "release", "feature", "hotfix"),
                    gloss="what kind of branch it ran against")),
        stamps=(Stamp("started_on", gloss="when it started"),),
        items=(Items("cache_hits", ("deps", "compiler", "docker_layers", "assets"),
                     gloss="which caches it managed to reuse"),),
        ids=("run_id",),
        pairs=(("duration_s", "budget_s",
                "how long it took against how long it was allowed"),),
    ),
    Domain(
        "order", "purchase order",
        nums=(Num("line_total_eur", 9, 9800, " EUR", False,
                  gloss="what the order comes to"),
              Num("line_count", 1, 24, gloss="how many lines it has"),
              Num("discount_pct", 0, 45, " %", gloss="how much was taken off"),
              Num("lead_time_days", 1, 120, " days", gloss="how long it will take to arrive"),
              Num("promised_lead_days", 1, 120, " days", label="promised lead time",
                  gloss="how long the supplier said it would take"),
              Num("unit_mass_g", 20, 9000, " g", True, ("kg", 0.001),
                  gloss="what one unit weighs")),
        enums=(Enum("supplier_tier", ("tier_1", "tier_2", "tier_3", "unrated"),
                    gloss="how the supplier is rated"),
               Enum("payment_terms", ("prepaid", "net_14", "net_30", "net_60"),
                    gloss="when it has to be paid"),
               Enum("warehouse", ("dock_north", "dock_south", "overflow", "bonded"),
                    gloss="where it is going to be received")),
        stamps=(Stamp("raised_on", gloss="when it was raised"),),
        items=(Items("approvals", ("buyer", "finance", "legal", "quality", "plant"),
                     gloss="who has signed it off"),),
        ids=("po_number",),
        pairs=(("lead_time_days", "promised_lead_days",
                "how long it will actually take against what was promised"),),
    ),
    Domain(
        "vehicle", "vehicle telemetry record",
        nums=(Num("odometer_km", 120, 480000, " km", True, ("miles", 0.621371),
                  gloss="how far it has travelled"),
              Num("fuel_level_pct", 0, 100, " %", gloss="how much fuel is left"),
              Num("coolant_temp_c", -12, 128, " °C", False, label="coolant temperature",
                  gloss="how hot the coolant is running"),
              Num("coolant_limit_c", -12, 128, " °C", False, label="coolant limit",
                  gloss="how hot the coolant is allowed to run"),
              Num("tyre_pressure_kpa", 140, 340, " kPa", True, ("bar", 0.01),
                  gloss="how hard the tyres are inflated"),
              Num("idle_minutes", 0, 240, " min", label="idle time",
                  gloss="how long it has been sitting idle")),
        enums=(Enum("drivetrain", ("diesel", "petrol", "hybrid", "battery"),
                    gloss="what drives it"),
               Enum("depot", ("depot_e1", "depot_w4", "depot_n2", "depot_s7"),
                    gloss="which depot it belongs to"),
               Enum("duty_cycle", ("urban", "regional", "long_haul"),
                    gloss="what kind of work it does")),
        stamps=(Stamp("serviced_on", gloss="when it was last serviced"),),
        items=(Items("fault_codes", ("p0301", "p0420", "u0100", "c1234", "b1318"),
                     gloss="which fault codes are stored"),),
        ids=("fleet_tag",),
        pairs=(("coolant_temp_c", "coolant_limit_c",
                "how hot the coolant is running against how hot it may run"),),
    ),
)

DOMAINS_BY_NAME = {d.name: d for d in DOMAINS}
TODAY = date(2026, 6, 1)
_WORDS = ("axle", "birch", "cobalt", "delta", "ember", "flint", "garnet", "hollow",
          "indigo", "juniper", "kestrel", "larch", "marlow", "nimbus", "onyx", "pivot")


def _ref(rng: random.Random) -> str:
    return f"{rng.choice(_WORDS)}-{rng.randrange(1000, 9999)}"


def make_record(domain: Domain, rng: random.Random) -> dict:
    """One record, every field populated. Missing fields are removed later, by name."""
    rec: dict = {}
    for name in domain.ids:
        rec[name] = _ref(rng)
    for n in domain.nums:
        v = rng.uniform(n.lo, n.hi)
        rec[n.name] = int(round(v)) if n.integral else round(v, 1)
    for e in domain.enums:
        rec[e.name] = rng.choice(e.values)
    for s in domain.stamps:
        rec[s.name] = (TODAY - timedelta(days=rng.randrange(0, 900))).isoformat()
    for it in domain.items:
        k = rng.randint(it.lo, min(it.hi, len(it.pool)))
        rec[it.name] = sorted(rng.sample(it.pool, k))
    return rec


UNIT_TOKENS = frozenset({"kg", "g", "usd", "eur", "pct", "c", "mv", "v", "s", "dbm",
                         "km", "kpa", "mb", "gb", "mm", "min", "minutes", "days",
                         "litres", "hours"})


def _label(name: str, explicit: str = "") -> str:
    """Field name as English: `gross_weight_kg` -> "gross weight".

    The unit token is dropped because the prose already carries the unit after the
    number, and "a gross weight kg of 2.0 kg" is not a sentence anybody writes.
    """
    if explicit:
        return explicit
    parts = name.split("_")
    if len(parts) > 1 and parts[-1] in UNIT_TOKENS:
        parts = parts[:-1]
    return " ".join(parts)


def _article(word: str) -> str:
    # "u" is left out on purpose: every "u" label here reads "a unit mass", not "an"
    return "an" if word[:1].lower() in "aeio" else "a"


def field_ref(f, shape: str, mode: str = "direct") -> str:
    """How a rule names this field, given how the state presents it.

    `direct` quotes what the state shows: the key for JSON, the phrase and unit for prose.
    Naming the key at a state that never contains it would make the prose half a
    different, harder task -- one about guessing which phrase a key meant.

    `indirect` names the field by **what it records** and never by its identifier: "the
    field that records how heavy the shipment is, in kg". Locating the evidence becomes
    part of the task instead of being given. Every rule an earlier model trained on was
    `direct`, which is why a rule phrased as "outside its required temperature range" was
    out of distribution for it -- and why it answered that one confidently and wrongly.
    A field with no gloss falls back to `direct`, so adding a field cannot silently
    produce a rule that points at nothing.
    """
    if mode == "indirect":
        gloss = getattr(f, "gloss", "")
        if gloss:
            unit = getattr(f, "unit", "").strip()
            return (f"the field that records {gloss}, in {unit}," if unit
                    else f"the field that records {gloss}")
    if shape == "json":
        return f"`{f.name}`"
    lab = _label(f.name, getattr(f, "label", ""))
    unit = getattr(f, "unit", "").strip()
    return f"{lab} ({unit})" if unit else lab


def as_prose(domain: Domain, rec: dict) -> str:
    """The same record as a paragraph. No teacher, no second label -- the values are
    the values, only the shape around them changes."""
    ident = next((rec[i] for i in domain.ids if i in rec), None)
    head = f"The {domain.noun}" + (f" {ident}" if ident else "")
    clauses: list[str] = []
    for f in domain.fields:
        if f.name not in rec:
            continue
        v = rec[f.name]
        lab = _label(f.name, getattr(f, "label", ""))
        if isinstance(f, Num) and f.unit:
            clauses.append(f"{_article(lab)} {lab} of {v}{f.unit}")
        elif isinstance(f, Num):
            clauses.append(f"{lab} {v}")           # a plain count is not "a count of 6"
        elif isinstance(f, Enum):
            clauses.append(f"{lab} {v}")
        elif isinstance(f, Stamp):
            clauses.append(f"{lab} {v}")
        else:
            clauses.append(f"{lab} " + (", ".join(v) if v else "none recorded"))
    return f"{head} has {'; '.join(clauses)}."


def as_json(rec: dict) -> str:
    return json.dumps(rec, indent=2)


def render(domain: Domain, rec: dict, shape: str) -> str:
    return as_json(rec) if shape == "json" else as_prose(domain, rec)


def render_pair(domain: Domain, a: dict, b: dict, shape: str,
                keys: tuple[str, str] = ("record_a", "record_b")) -> str:
    """Two records in one state, so a contrastive pair shares a sequence.

    Question blocks attend to the state and to themselves, never to each other, so two
    questions over one state is the only way to put both halves of a state flip in front
    of the model under one prefill -- and the only way to be sure they land in the same
    batch rather than merely the same epoch.
    """
    if shape == "json":
        return json.dumps({keys[0]: a, keys[1]: b}, indent=2)
    return (f"Record A. {as_prose(domain, a)}\n\n"
            f"Record B. {as_prose(domain, b)}")
