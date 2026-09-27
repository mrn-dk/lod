"""api_schema — OpenAPI payload validation from APIs.guru.

Ground truth is executable: the state carries the schema summary *and* the
payload, so every question is decidable by reading the state and applying the
rules. Each (API, definition, enum field) is its own schema.

Source: https://apis.guru/  — the collected specs are CC0 (tier 1).
"""

from __future__ import annotations

import json
import random

from lod.schema import Example, Question
from lod.corpus.services.sources.base import SourceInfo, SourceResult, fetch_json, schema_key

INFO = SourceInfo(
    name="api_schema",
    url="https://api.apis.guru/v2/list.json",
    licence="CC0-1.0",
    tier=1,
    ground_truth_type="executable",
    state_format="structured",
    option_type="fixed",
    notes="OpenAPI specs; payload validated against the schema carried in the state",
)
YN = ["no", "yes"]
_SAMPLE = {"string": "example", "integer": 42, "number": 3.5, "boolean": True,
           "array": [], "object": {}}


def _defs(spec: dict) -> dict:
    return spec.get("components", {}).get("schemas") or spec.get("definitions") or {}


def _enum_fields(d: dict) -> list[tuple[str, list[str]]]:
    out = []
    for name, prop in (d.get("properties") or {}).items():
        if isinstance(prop, dict) and isinstance(prop.get("enum"), list):
            vals = [str(v) for v in prop["enum"] if v is not None]
            if 2 <= len(vals) <= 20 and all(len(v) < 40 for v in vals):
                out.append((name, vals))
    return out


def _payload(d: dict, rng: random.Random, enum_choice: dict[str, str]) -> dict:
    body = {}
    for name, prop in list((d.get("properties") or {}).items())[:8]:
        if not isinstance(prop, dict):
            continue
        if name in enum_choice:
            body[name] = enum_choice[name]
        elif isinstance(prop.get("enum"), list) and prop["enum"]:
            body[name] = str(rng.choice(prop["enum"]))
        else:
            body[name] = _SAMPLE.get(prop.get("type", "string"), "example")
    return body


def _summary(defname: str, d: dict, required: list[str]) -> str:
    lines = [f"Schema: {defname}"]
    for name, prop in list((d.get("properties") or {}).items())[:8]:
        if not isinstance(prop, dict):
            continue
        t = prop.get("type", "string")
        bit = f"  {name}: {t}"
        if isinstance(prop.get("enum"), list):
            bit += " (allowed: " + ", ".join(str(v) for v in prop["enum"][:20]) + ")"
        if name in required:
            bit += " [required]"
        lines.append(bit)
    return "\n".join(lines)


def generate(n: int, seed: int = 0, n_apis: int = 220, per_schema: int = 8) -> SourceResult:
    rng = random.Random(seed)
    res = SourceResult(info=INFO)
    index = fetch_json(INFO.url)
    names = sorted(index)
    rng.shuffle(names)

    for api in names:
        if len(res.examples) >= n:
            break
        if n_apis <= 0:
            break
        versions = index[api].get("versions") or {}
        if not versions:
            continue
        entry = versions[sorted(versions)[-1]]
        try:
            spec = fetch_json(entry["swaggerUrl"], timeout=45)
        except Exception:
            continue
        n_apis -= 1
        for defname, d in list(_defs(spec).items())[:12]:
            if not isinstance(d, dict) or len(res.examples) >= n:
                break
            fields = _enum_fields(d)
            if not fields:
                continue
            required = [r for r in (d.get("required") or []) if isinstance(r, str)]
            field, allowed = fields[0]
            task = f"api_{api.replace('.', '_')[:28]}_{defname[:24]}"
            summary = _summary(defname, d, required)
            for _ in range(min(per_schema, max(1, n - len(res.examples)))):
                good = rng.random() < 0.5
                value = rng.choice(allowed) if good else f"{rng.choice(allowed)}_x{rng.randrange(9)}"
                body = _payload(d, rng, {field: value})
                dropped = None
                if good and required and rng.random() < 0.35:
                    dropped = next((r for r in required if r in body), None)
                    if dropped:
                        body.pop(dropped)
                        good = False
                state = (f"{summary}\n\nPayload:\n" + json.dumps(body, indent=2))
                qs = [Question("valid", "Is this payload valid against the schema?",
                               list(YN), int(good))]
                if field in body:
                    qs.append(Question("enum_value", f"What is the value of '{field}'?",
                                       list(allowed) if value in allowed else list(allowed) + [value],
                                       (list(allowed) + [value]).index(value)
                                       if value not in allowed else allowed.index(value)))
                if not good:
                    culprit = dropped or field
                    names_opt = sorted(set(list(body) + [culprit]))[:12]
                    if culprit in names_opt and len(names_opt) >= 2:
                        qs.append(Question("bad_field", "Which field breaks the schema?",
                                           names_opt, names_opt.index(culprit)))
                rng.shuffle(qs)
                res.examples.append(Example(state=state, questions=qs, task=task))
                for q in qs:
                    res.schemas.add(schema_key(q.question, q.options))
    return res
