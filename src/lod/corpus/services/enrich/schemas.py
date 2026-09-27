"""Which schemas an LLM may rewrite, and how its answer is checked and applied.

The deterministic build publishes option criteria only where the upstream source does
(NVD's CVSS tables, MITRE's CWE catalogue, the GoEmotions taxonomy). Elsewhere a schema
arrives with an option key and a question generated from a column name:

    "What is the sexual explicit of this text?"   options: no / yes

Enrichment rewrites both, once per *schema* -- a (task, option-key tuple) -- never per
record: every record sharing a schema gets the same question and the same criteria, so
a 28,000-question source costs a few dozen calls.

Rules the model cannot break, enforced here after it answers rather than asked for:

- option keys, order and count, question ids and targets are the input record's. The
  answer is read for two fields only: `question` and `descriptions`.
- a description that merely repeats its option key is dropped to None: a key echoed as
  its own definition is worse than none.
- a task whose options vary per example gets its question rewritten and its criteria
  left alone; there is no stable option set to describe.
- published criteria win over a generated gloss, option by option.
- `PROTECTED_PREFIXES` tasks are never touched: their question or criteria *are* the
  label, so a paraphrase would be a relabelling.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict

# beyond this many distinct option sets a task has no stable schema to describe
MAX_OPTION_SETS = 6
# Every option gets a criterion: options are alphabetical, so a lower cut would leave an
# arbitrary tail undescribed -- and "criteria as input" means an option without one is
# scored blind.
MAX_DESCRIBED_OPTIONS = 256

# Tasks whose question or criteria are the label, not a gloss on it. A rule source's
# description *is* the rule ("The `gross_weight_kg` is greater than 12") over a fixed
# ["no", "yes"], so one rewrite per schema would apply one record's rule to every record
# of the task: not a worse description, a wrong label. The same holds wherever the
# question names which function of the state to compute (forecast rows), where the
# question is templated through the phrasing bank from the source's own codebook, or
# where the criteria are the publisher's definitions.
PROTECTED_PREFIXES = (
    # code-labelled generators: the criteria are the program
    "rule_", "entity_fact", "fever_", "vitaminc_", "browser_", "sensor_", "toolgate_",
    "entityres_", "diff_", "sched_", "pdfdocs_", "relational_", "chessfact_",
    "compose_", "post_", "catalog_", "longctx_", "rl_", "longdoc_",
    # the question names what to compute from the state (a member fraction, a
    # threshold); a paraphrase toward "will it rain" changes the target's meaning
    "gefs_", "spf_",
    # the target is the market's close price, as the instructions define it
    "manifold_",
    # the codebook's prompt, turned to the end the target counts
    "hatespeech__",
    # templated questions with source-written criteria
    "loghub_", "lichess_", "tabfact_", "nvd_",
    # Super-NaturalInstructions: the question is the task's published definition
    "ni_",
    # the label the maintainers applied, over a vocabulary mixing types and areas
    "gh_label_", "gh_is_bug",
    # SPDX's published names; UNFAIR-ToS's category definitions
    "spdx_", "lexglue_unfair_tos",
    # sentiment/stance, multilingual and topic rows: code-template questions naming
    # what each source's label means, criteria from the publishers
    "sent44_", "ml_", "topic_",
)


def protected(task: str, prefixes: tuple[str, ...] = PROTECTED_PREFIXES) -> bool:
    return task.startswith(prefixes)


def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def schema_key(task: str, options: list[str]) -> str:
    raw = json.dumps([task, list(options)], ensure_ascii=False).encode()
    return hashlib.sha256(raw).hexdigest()[:20]


def collect_schemas(corpus: dict[str, list[dict]],
                    prefixes: tuple[str, ...] = PROTECTED_PREFIXES) -> tuple[dict, set[str]]:
    """schema key -> a representative; plus the tasks whose options are not stable."""
    sets_per_task: dict[str, set] = defaultdict(set)
    for records in corpus.values():
        for record in records:
            if protected(record["task"], prefixes):
                continue
            for q in record["questions"]:
                sets_per_task[record["task"]].add(tuple(q["options"]))
    unstable = {t for t, s in sets_per_task.items() if len(s) > MAX_OPTION_SETS}

    reps: dict[str, dict] = {}
    for records in corpus.values():
        for record in records:
            if protected(record["task"], prefixes):
                continue
            for q in record["questions"]:
                task = record["task"]
                options = [] if task in unstable else list(q["options"])
                k = schema_key(task, options)
                if k not in reps:
                    state = record["state"]
                    reps[k] = {"task": task, "options": list(q["options"]),
                               "question": q["question"], "describe": task not in unstable,
                               "state": state if isinstance(state, str) else json.dumps(state)}
    return reps, unstable


def parse_json(text: str) -> dict | None:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if 0 <= start < end:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                return None
    return None


def clean(result: dict, options: list[str], describe: bool) -> dict | None:
    """Keep only what the model is allowed to decide, and only if it is usable."""
    question = str(result.get("question") or "").strip()
    if not (10 <= len(question) <= 400):
        return None
    out: dict = {"question": question}
    if not describe:
        return out
    raw = result.get("descriptions")
    if not isinstance(raw, list) or len(raw) != len(options[:MAX_DESCRIBED_OPTIONS]):
        return out
    keys = {norm(o) for o in options}
    texts: list[str | None] = []
    for option, desc in zip(options, raw):
        d = str(desc or "").strip()
        # a key echoed back as its own definition is worse than no definition
        if not d or norm(d) == norm(option) or (norm(d) in keys and len(d) < 25):
            texts.append(None)
        else:
            texts.append(d[:300])
    texts += [None] * (len(options) - len(texts))
    if any(texts):
        out["descriptions"] = texts
    return out


def apply(record: dict, cache: dict, unstable: set[str],
          prefixes: tuple[str, ...] = PROTECTED_PREFIXES) -> dict:
    """The record with its schema's cached question and criteria, if any."""
    task = record["task"]
    if protected(task, prefixes):
        return record
    questions = []
    for q in record["questions"]:
        options = [] if task in unstable else list(q["options"])
        got = cache.get(schema_key(task, options))
        q = dict(q)
        if got:
            q["question"] = got["question"]
            # published criteria (attached at build time) win over a generated gloss --
            # per option, not per question: a source that defines some of its labels and
            # not others is normal, and keeping its list wholesale would trade a gloss
            # for a hole
            already = q.get("descriptions") or []
            fresh = got.get("descriptions") or []
            if len(fresh) == len(q["options"]):
                q["descriptions"] = [
                    (already[i] if i < len(already) and already[i] else fresh[i])
                    for i in range(len(q["options"]))
                ]
        questions.append(q)
    out = dict(record)
    out["questions"] = questions
    return out
