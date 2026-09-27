"""Seeds for tool-call and action gating (`synth/toolgate.py`, row 38).

Not part of a normal fetch. The generator reads two frozen files that ship inside the
package, `toolgate_http.json` and `toolgate_scenarios.json`; this module regenerates them
on request, and checks the generator's hand-written CLI table against the local man pages.
Neither seed puts third-party prose into the corpus.

CLI. The man pages are the authority for two facts only: that a command exists, and which
gating-relevant flags it accepts. The effect class of each call (read / append / overwrite
/ delete / restart, and what restores it) is a table in `toolgate.CLI_CALLS`, because a
keyword rule over the NAME line cannot be trusted to produce it -- `survey_man` prints the
measurement (on the reference machine it tagged 55 % of pages, and put `cut`, `colrm` and
`clear` in the destructive bucket). `verify_table` checks every command and flag in the
table against the page and reports any claim that fails.

OpenAPI. About a dozen REST specs under permissive licences, via the apis.guru directory.
Only `(method, path template)` is kept -- the interface surface, which carries the whole
gating signal:

    GET                                      -> read
    POST /xs, spec also has DELETE /xs/{id}  -> undoable by one call
    POST /xs, it does not                    -> restorable only from a backup
    DELETE | PUT | PATCH                     -> restorable only from a backup
    path ends in {param}                     -> one item
    deeper literal segment                   -> one collection
    top-level literal segment                -> everything

Scenarios. An LLM writes the setting a call sits in -- an invented system, the operator's
goal, one line of unrelated colour -- and never sees a call, an environment, a backup or
an option, so it cannot produce a target even by accident. Anything that mentions the
decision vocabulary is dropped rather than edited.
"""

from __future__ import annotations

import collections
import concurrent.futures
import glob
import gzip
import json
import os
import random
import re
import time
from pathlib import Path

from lod.corpus.repositories import raw_store as store
from lod.corpus.repositories import web
from lod.corpus.services.sources.synth import toolgate

MAN_DIRS = ("/usr/share/man/man1", "/usr/share/man/man8", "/usr/share/man/man5")
GURU = "https://api.apis.guru/v2/list.json"
PERMISSIVE = ("apache", "mit", "cc-by", "bsd")
SKIP_HOSTS = ("amazonaws.com", "azure.com", "googleapis.com", "windows.net")
OPENROUTER = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "deepseek/deepseek-v4.1-flash"


def scratch_dir() -> Path:
    """Where the apis.guru listing is cached, so a rerun sees the same directory."""
    return store.RAW_ROOT / "toolspecs"


# ---- man pages ------------------------------------------------------------------------

def deroff(text: str) -> str:
    text = text.replace("\\f(", "\\f")
    for esc in ("\\fB", "\\fI", "\\fR", "\\fP", "\\/", "\\&"):
        text = text.replace(esc, "")
    text = text.replace("\\-", "-").replace("\\ ", " ")
    text = re.sub(r"\\s[+-]?\d", "", text)
    return text.replace("\\e", "\\")


def read_page(path: str) -> str:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", errors="replace") as handle:  # type: ignore[operator]
        return deroff(handle.read())


def find_page(command: str) -> str | None:
    for directory in MAN_DIRS:
        for candidate in sorted(glob.glob(f"{directory}/{command}.[0-9]*")):
            return candidate
    return None


def name_line(text: str) -> str | None:
    match = re.search(r"^\.S[Hh]\s+\"?NAME\"?\s*$(.*?)^\.S[Hh]", text, re.M | re.S)
    if match:
        body = " ".join(l for l in match.group(1).splitlines() if not l.startswith("."))
        body = re.sub(r"\s+", " ", body).strip()
        if body:
            return body
    match = re.search(r"^\.Nd\s+(.*)$", text, re.M)
    return match.group(1).strip() if match else None


DESTRUCTIVE = ("remove", "delete", "erase", "destroy", "wipe", "shred", "purge",
               "truncate", "format", "overwrite", "discard", "drop", "revoke",
               "uninstall", "unlink", "kill", "terminate", "halt", "reboot",
               "shutdown", "reset", "clear", "flush", "prune", "expire")
MUTATING = ("create", "make", "add", "install", "write", "set", "modify", "change",
            "edit", "update", "rename", "move", "copy", "extract", "archive",
            "compress", "generate", "build", "compile", "import", "upload", "send",
            "publish", "apply", "register", "start", "stop", "restart", "enable",
            "disable", "mount", "unmount", "link", "sign", "encrypt", "decrypt",
            "convert", "patch", "sort")
READONLY = ("print", "display", "show", "list", "report", "query", "view", "describe",
            "dump", "check", "test", "verify", "compare", "search", "find", "count",
            "calculate", "inspect", "read", "examine", "lookup", "locate", "browse",
            "monitor", "watch", "trace", "measure", "identify", "determine", "output")
_STEMS = {k: {w.rstrip("s") for w in v}
          for k, v in (("destructive", DESTRUCTIVE), ("mutating", MUTATING),
                       ("readonly", READONLY))}


def keyword_tag(description: str) -> str | None:
    """The labelling rule the generator *rejects*, kept so the rejection stays measured."""
    body = description.lower()
    body = body.split(" - ", 1)[1] if " - " in body else body
    head = re.findall(r"[a-z][a-z-]+", body)[:6]
    for klass in ("destructive", "mutating", "readonly"):
        for word in head:
            if word.rstrip("s") in _STEMS[klass]:
                return klass
    return None


def survey_man() -> dict:
    counts: collections.Counter = collections.Counter()
    pages = sorted(glob.glob("/usr/share/man/man1/*"))
    for path in pages:
        try:
            text = read_page(path)
        except OSError:
            counts["unreadable"] += 1
            continue
        description = name_line(text)
        if not description:
            counts["no_name_section"] += 1
            continue
        counts[keyword_tag(description) or "untagged"] += 1
    counts["pages"] = len(pages)
    return dict(counts)


FLAG_RE = re.compile(r"(?<![\w-])(--[a-z][a-z0-9-]{1,24}|-[a-zA-Z])(?![\w])")


def verify_table(calls) -> tuple[list[str], list[str]]:
    """-> (commands with no page here, "command flag" claims the page does not contain)."""
    missing, unverified = [], []
    cache: dict[str, set[str]] = {}
    for call in calls:
        command = call.command
        if command not in cache:
            page = find_page(command)
            if page is None:
                missing.append(command)
                cache[command] = set()
                continue
            cache[command] = set(FLAG_RE.findall(read_page(page)))
        flags = cache[command]
        if not flags:
            continue
        for flag in call.flag_spellings():
            if flag not in flags:
                unverified.append(f"{command} {flag}")
    return sorted(set(missing)), sorted(set(unverified))


# ---- OpenAPI --------------------------------------------------------------------------

METHODS = ("get", "post", "put", "delete", "patch")


def _try_get(url: str, timeout: int = 15) -> bytes | None:
    try:
        return web.get(url, timeout=timeout, retries=1)
    except Exception:  # noqa: BLE001 -- an unreachable spec is simply skipped
        return None


def operations(spec: dict) -> list[tuple[str, str]]:
    out = set()
    for path, item in (spec.get("paths") or {}).items():
        if not isinstance(item, dict) or not path.startswith("/"):
            continue
        for method in item:
            if method.lower() in METHODS:
                out.add((method.upper(), path))
    return sorted(out)


def segments(path: str) -> list[str]:
    return [s for s in path.strip("/").split("/") if s]


def is_param(segment: str) -> bool:
    return segment.startswith("{") and segment.endswith("}")


def blast_of(path: str) -> str:
    segs = segments(path)
    if not segs:
        return "everything"
    if is_param(segs[-1]):
        return "one_item"
    literal = [s for s in segs if not is_param(s)]
    return "everything" if len(literal) <= 1 else "one_collection"


def derive(method: str, path: str, all_ops: set[tuple[str, str]]) -> dict | None:
    """Effect and what restores it, from the method and the path shape alone."""
    segs = segments(path)
    if not segs or len(segs) > 6 or is_param(segs[0]):
        return None   # a store root has to be a literal segment
    if method == "GET":
        effect, restorable = "read", "none"
    elif method == "POST":
        # a create is undoable by one call exactly when the spec also exposes a delete
        # for an item of the collection being posted to
        has_delete = any(m == "DELETE" and segments(p)[:len(segs)] == segs
                         and len(segments(p)) == len(segs) + 1 and is_param(segments(p)[-1])
                         for m, p in all_ops)
        effect, restorable = "append", ("undo_command" if has_delete else "backup")
    elif method in ("PUT", "PATCH"):
        effect, restorable = "overwrite", "backup"
    else:
        effect, restorable = "delete", "backup"
    return {"method": method, "path": path, "effect": effect,
            "restorable": restorable, "blast": blast_of(path)}


def fetch_openapi(limit: int, seed: int) -> list[dict]:
    scratch = scratch_dir()
    listing = scratch / "apis_guru_list.json"
    if not listing.exists():
        body = _try_get(GURU, timeout=60)
        if body is None:
            raise RuntimeError("could not fetch the apis.guru directory")
        web.save(body, listing)
    directory = json.loads(listing.read_text())
    candidates = []
    for key, entry in directory.items():
        if key.startswith(SKIP_HOSTS):
            continue
        preferred = entry.get("versions", {}).get(entry.get("preferred"), {})
        licence = (((preferred.get("info") or {}).get("license") or {}).get("name") or "")
        if not any(p in licence.lower() for p in PERMISSIVE):
            continue
        candidates.append((key, licence, preferred.get("swaggerUrl")))
    random.Random(seed).shuffle(candidates)
    kept: list[dict] = []
    for key, licence, url in candidates:
        if len(kept) >= limit:
            break
        body = _try_get(url) if url else None
        if body is None or len(body) > 3_000_000:
            continue
        try:
            spec = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        ops = operations(spec)
        with_param = sum(1 for _, p in ops if "{" in p)
        writes = sum(1 for m, _ in ops if m in ("DELETE", "PUT", "PATCH"))
        if len(ops) < 10 or with_param < 4 or writes < 4:
            continue
        kept.append({"api": key, "licence": licence, "url": url, "ops": ops})
        print(f"  kept {key} ({licence}) {len(ops)} operations", flush=True)
    return kept


def build_http(kept: list[dict], per_api: int, seed: int) -> dict:
    apis, calls, dropped = [], [], 0
    for entry in kept:
        all_ops = {(m, p) for m, p in entry["ops"]}
        derived = [d for d in (derive(m, p, all_ops) for m, p in entry["ops"]) if d]
        dropped += len(entry["ops"]) - len(derived)
        rng = random.Random(f"{seed}:{entry['api']}")
        # round-robin over effects, so one large API cannot supply only its favourite verb
        by_effect: dict[str, list[dict]] = collections.defaultdict(list)
        for d in derived:
            by_effect[d["effect"]].append(d)
        picked: list[dict] = []
        while len(picked) < per_api and any(by_effect.values()):
            for effect in sorted(by_effect):
                bucket = by_effect[effect]
                if bucket and len(picked) < per_api:
                    picked.append(bucket.pop(rng.randrange(len(bucket))))
        for d in picked:
            d["api"] = entry["api"]
        calls.extend(picked)
        apis.append({"api": entry["api"], "licence": entry["licence"], "url": entry["url"],
                     "operations": len(entry["ops"]), "kept": len(picked)})
    return {"note": "method and path template only; no descriptions, summaries or schemas",
            "apis": apis, "calls": calls, "dropped_unshaped": dropped}


# ---- scenarios ------------------------------------------------------------------------

FORBIDDEN = (
    "production", "prod ", "staging", "scratch", "sandbox", "backup", "restore",
    "reversib", "irreversib", "destructive", "undo", "roll back", "rollback",
    "safe", "unsafe", "danger", "risk", "approve", "approval", "confirm", "permission",
    "block", "allow", "escalate", "dry run", "dry-run", "recover", "blast radius",
    "should", "must not", "caution", "careful", "verify first", "double-check",
)

SCEN_PROMPT = """You are writing SETTINGS for an operations simulator. A setting is the \
background a command-line or API agent is working in. It is scene-setting only.

Category: {category} -- {gloss}

Return a JSON array of exactly {n} objects, each with exactly these keys:
  "system": a short invented name for the service or repository involved, 2-4 words, \
no real company or product names
  "task":   one sentence, at most 18 words, saying what the operator asked the agent to \
get done today. High level. Do not name any specific command, flag, path or endpoint.
  "note":   one sentence, at most 18 words, of unrelated operational colour -- who is on \
call, what shipped last week, which dashboard is noisy.

Hard rules. Breaking any one of them makes the object useless:
- Never mention whether anything is risky, safe, reversible, recoverable, approved, \
confirmed, blocked or allowed.
- Never mention backups, restores, rollbacks, dry runs, environments, tiers, or \
permissions.
- Never give advice and never say what the agent should or should not do.
- Vary the domain, the scale and the tone across the {n} objects.

Output the JSON array and nothing else. Seed {seed}."""

SCEN_CATEGORIES = {
    "filesystem": "files, directories and copies on a host",
    "vcs": "a git working tree and its remotes",
    "package": "operating-system packages on a host",
    "archive": "tarballs and compressed bundles",
    "process": "long-running processes, containers and services",
    "text": "log and configuration files edited in place",
    "sysadmin": "mounts, scheduled jobs and machine power state",
    "http": "a JSON HTTP API behind a service account",
    "accounts": "local user accounts, groups and password ageing on a host",
}


def _parse_json(text: str):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("["), text.rfind("]")
        if 0 <= start < end:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                return None
    return None


def _usable(item) -> bool:
    if not isinstance(item, dict) or set(item) != {"system", "task", "note"}:
        return False
    for key in ("system", "task", "note"):
        value = item[key]
        if not isinstance(value, str):
            return False
        value = value.strip()
        if not 3 <= len(value) <= 160 or "\n" in value:
            return False
        if any(bad in value.lower() for bad in FORBIDDEN):
            return False
    return len(item["task"].split()) <= 22 and len(item["note"].split()) <= 22


def _chat(token: str, model: str, prompt: str, timeout: float) -> str:
    import urllib.request
    req = urllib.request.Request(
        OPENROUTER, method="POST",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        data=json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                         "temperature": 0.7, "max_tokens": 6000}).encode())
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.loads(r.read())
    return body["choices"][0]["message"]["content"] or ""


def generate_scenarios(model: str = DEFAULT_MODEL, per_call: int = 15, rounds: int = 4,
                       concurrency: int = 32, budget: float = 300.0) -> dict:
    """One request per (category, round), each with its own timeout and a shared
    wall-clock budget: a provider that stalls on a few long generations must cost those
    requests, not the whole pass."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    token = os.environ.get("OPENROUTER_API_KEY")
    if not token:
        raise RuntimeError("OPENROUTER_API_KEY is required to generate scenarios")

    deadline = time.monotonic() + budget
    stats = {"asked": 0, "returned": 0, "dropped": 0, "failed": 0}
    out: dict[str, list] = {c: [] for c in SCEN_CATEGORIES}

    def one(job):
        category, round_ = job
        prompt = SCEN_PROMPT.format(category=category, gloss=SCEN_CATEGORIES[category],
                                    n=per_call, seed=f"{category}-{round_}")
        for _ in range(2):
            if time.monotonic() > deadline:
                break
            try:
                parsed = _parse_json(_chat(token, model, prompt, timeout=75.0))
                if parsed is not None:
                    return category, parsed
            except Exception:  # noqa: BLE001
                pass
        return category, None

    jobs = [(c, r) for c in SCEN_CATEGORIES for r in range(rounds)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        for category, parsed in pool.map(one, jobs):
            stats["asked"] += per_call
            if parsed is None:
                stats["failed"] += 1
                print(f"  {category}: no json", flush=True)
                continue
            items = parsed if isinstance(parsed, list) else parsed.get("items", [])
            kept = 0
            for item in items or []:
                stats["returned"] += 1
                if _usable(item):
                    out[category].append({k: item[k].strip()
                                          for k in ("system", "task", "note")})
                    kept += 1
                else:
                    stats["dropped"] += 1
            print(f"  {category}: {kept} usable", flush=True)

    for category, items in out.items():
        seen, unique = set(), []
        for item in items:
            marker = item["system"].lower()
            if marker not in seen:
                seen.add(marker)
                unique.append(item)
        out[category] = unique
    stats["kept"] = sum(len(v) for v in out.values())
    return {"stats": stats, "scenarios": out}


# ---- entry point ----------------------------------------------------------------------

def regenerate(openapi: bool = True, scenarios: bool = False, apis: int = 12,
               per_api: int = 40, seed: int = 7, model: str = DEFAULT_MODEL) -> str:
    """Check the CLI table, and rewrite the package's frozen seed files.

    `openapi` rewrites `toolgate_http.json` from apis.guru; `scenarios` rewrites
    `toolgate_scenarios.json` with the LLM (needs OPENROUTER_API_KEY).
    """
    print("  man-page keyword survey (the rule the generator does not use):", survey_man())
    missing, unverified = verify_table(toolgate.CLI_CALLS)
    print(f"  CLI table: {len(toolgate.CLI_CALLS)} call templates over "
          f"{len({c.command for c in toolgate.CLI_CALLS})} commands; "
          f"no man page: {missing or 'none'}; flags not in the page: {unverified or 'none'}")
    wrote = []
    if scenarios:
        payload = generate_scenarios(model=model)
        toolgate.SCENARIOS.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n")
        wrote.append(f"{toolgate.SCENARIOS.name} ({payload['stats']['kept']} scenarios)")
    if openapi:
        payload = build_http(fetch_openapi(apis, seed), per_api, seed)
        toolgate.HTTP_SPECS.write_text(json.dumps(payload, indent=1) + "\n")
        wrote.append(f"{toolgate.HTTP_SPECS.name} ({len(payload['calls'])} calls "
                     f"from {len(payload['apis'])} APIs)")
    return "wrote " + ", ".join(wrote) if wrote else "checked only"
