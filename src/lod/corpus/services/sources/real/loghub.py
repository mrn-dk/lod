"""Source-table row 14 — log level, component and burst escalation, parsed out of real logs.

Row 14 has no label column. A log mirror ships one `text` column per line and the labels
the spec asks for — level and component — are *inside* it:

    [Thu Jun 09 06:07:04 2005] [notice] LDAP: Built with OpenLDAP ...
    081109 204655 556 INFO dfs.DataNode$PacketResponder: Received block ...

So the generic column adapter correctly finds nothing, and the row needs a parser. The
labels are still real: a developer chose that severity when they wrote the logging call,
and the component is the emitting subsystem. Nothing here is invented — a line whose level
cannot be read is dropped rather than guessed.

**Domain-13 audit.** `loghub_level` asked for the severity of a single log
line whose state *was* that line -- `"[notice] LDAP: ..."` -> `NOTICE`. Measured on an
earlier build's `testreal.jsonl` (2,000 rows, the only split this row reached): the gold option
string appeared verbatim in its own state, case-insensitively, **100 %** of the time,
every state was exactly one line, and the cached raw store (`bolu61_loghub_2`, one 2005
Apache `error_log`) carried two levels (`NOTICE` 354 / `ERROR` 1,646) and one component
clearing the frequency floor, so `loghub_component` shipped zero rows and the "ordered
severity scale" was an 82.3 % majority-class coin flip with the answer printed in the
question. Line *format* was a constant too: learn `[Sun Jul 03 04:07:55 2005] [notice]`
once and the row is over.

**Domain-13 rework.** Four changes, each aimed at one of those:

*Sixteen formats, not one.* `scripts/fetch_data.py --rows 14` caches LogHub's published 2,000-line
sample of each of its 16 systems under `~/.cache/lod-sources/loghub/`. Nine of them
carry a machine-written severity field, in nine different positions and vocabularies
(`INFO` in field 4 of HDFS, field 3 of Spark, field 9 of BGL; `[notice]` in a bracket in
Apache; a bare `D`/`W`/`E` in Android; `Info` after a comma in Windows). Format is now a
variable and the system name travels in `meta` so a per-system reading stays possible.

*Positional parsing, never a keyword search.* The old parser searched the whole line for
any word that looked like a level, which reads "authentication failure ... error" as a
severity the logger never wrote. Each system has its own anchored pattern and the level is
whatever sits in that system's level *field*; a line that does not match, or whose field
holds an unknown token, is dropped. That is the rule the module already had and it is now
enforced structurally rather than by hoping the regex lands in the right place. Syslog-shaped
systems (Linux, Mac, OpenSSH, Thunderbird) and the field-delimited ones (HPC, HealthApp,
Proxifier) publish no severity at all, so they contribute components and bursts only.

*The answer is redacted out of the state.* `_scrub_level` blanks the exact span the level
was read from, the way `spdx._scrub` and `hf._scrub_label` do, so the question is what
severity this message *deserves*, not which bracket to copy. `loghub_component` gets the
same treatment for its own answer.

*A structural holdout.* The supercomputer family (BGL, HPC, Thunderbird) is reserved to
`testreal` with `force_split`, and the rest is forced to `train`. A drop from train to
testreal on row 14 therefore measures generalising over log *dialect* -- an unseen
timestamp shape, an unseen component vocabulary -- not memorising a format. The desktop
OS family (Windows, Linux, Mac) is held out the same way to `devreal` (`DEV_FAMILY`), so
the split that selects checkpoints and fits T measures the same kind of transfer.

`loghub_burst_escalate` keeps its shape from the first version: the state is a run of
consecutive real lines from one system, and the question -- did severity increase partway
through the run -- is not answered by any single bracketed token, only by comparing the
first parsed level against the rest. Bursts whose lines are all byte-identical are dropped:
they were a free "steady", and they were most of what a single Apache log could produce.

**Domain-13 audit, second round.** Three defects found by re-deriving every target in
an earlier build and probing what a reader who never looks at the message can score:

*The format was the answer.* `balanced_cap` balanced each task's histogram **pooled over
systems** and nothing balanced it *within* one, so the per-system label prior survived
intact: Android 100/114 DEBUG, HDFS 55/56 INFO, OpenStack 103/108 INFO, Windows 35/35
INFO, Zookeeper 125/142 WARN. Since the line format names the system, a lookup on the
state's skeleton up to `[LEVEL]` -- no message words at all -- scored **0.8158** on
`loghub_level` against a 0.400 majority, and **0.7664** on `loghub_burst_escalate`
against 0.5498. `_balance_within_systems` now caps each system's own histogram first,
drops a system that writes only one level (Spark and Windows are 2,000/2,000 INFO: a free
answer once the format is recognised) or that cannot be brought under
`MAX_SYSTEM_MAJORITY`, and only then caps the pooled histogram. Measured after: level
0.5163 over four formats, burst 0.5316 over six.

*The question and the target disagreed.* The burst question asked whether severity
"increase[s] at any point after the first line" while `escalation_label` compares every
later line against the **first** one. A run `WARN INFO WARN WARN INFO WARN` increases at a
point and is labelled `steady`; that was 62/642 = **9.7 %** of the trained task and 4/120
of the held-out one. The wording now states the comparison the code performs, and the two
options carry the criteria that define it.

*What the held-out side cannot be asked.* Two facts are recorded rather than fixed,
because fixing either means abandoning the structural holdout. BGL is the corpus's only
source of `FATAL`, so **40.3 %** of `loghub_level_supercomputer`'s eval labels are a class
that is correct in *no* training example, while `DEBUG` is 14.7 % of training and 0 % of
eval. And BGL publishes two components, one of which (`APP`) names itself in its own
message paths (`/bgl/apps/...`), as Thunderbird's `xinetd` does in `/etc/xinetd.d/`: 38 of
120 held-out component questions carry the gold name inside a longer word. That is
message evidence and is left in; what was *not* evidence was the substring redaction that
turned it into `/bgl/[COMPONENT]s/`, whose placeholder count scored 0.798 against a 0.473
marginal. Redaction is whole-token now (0.542 against 0.508). The task is weak on that family and the
number is published here.

**Domain-13 audit, third round.** Three more ways the answer survived
the redaction, each found by an audit against the raw files with an independent parser:

*Padding.* Log4j pads the level to five columns, so Zookeeper's `INFO  [` kept two spaces
after `[LEVEL]` and `ERROR [` one -- all 13 of its ERROR lines readable off the gap. The
redaction now swallows the padding (`_scrub(eat_padding=True)`).

*The alert annotation.* BGL's first field is LogHub's alert label, and every flagged line
is FATAL; shown in 85/85 held-out level states it lifted a lookup from 0.400 to 0.541.
`display_text` strips it from every BGL/Thunderbird state.

*A level with no place on the scale.* BGL's `SEVERE` was aliased to FATAL, but BG/L ranks
it below ERROR; 2/85 held-out level labels were two steps too high. It is now unmapped:
dropped from the level task, and a burst that prints it is dropped.

Licence. LogHub is research data, not OSI-licensed software; its LICENSE grants use "for
research or academic work" on condition that the repository is credited and the ISSRE'23
paper cited. The module used to claim "MIT (Loghub)", which the repository does not say.
`LICENCE` below is the corrected string and `scripts/fetch_data.py --rows 14` stores the LICENSE
text verbatim beside the logs.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from typing import Callable, Iterator

from lod.phrasings import PhrasingBank
from lod.schema import Example, Question
from lod.corpus.repositories import raw_store as store
from lod.corpus.services.sources.real.base import RealTask

LICENCE = ("Loghub research-use (research/academic only; attribution to "
           "github.com/logpai/loghub + ISSRE'23 citation required; not OSI)")
URL = "https://github.com/logpai/loghub"
KEY = "bolu61_loghub_2"          # the single-Apache-log HF mirror, kept as a fallback

# Ordered low to high. Every system's vocabulary is canonicalised onto this scale, so the
# option list means the same thing whichever dialect the line came from, and a reader
# auditing the "ordered severity scale" claim can check it here.
LEVEL_ORDER = ("DEBUG", "INFO", "NOTICE", "WARN", "ERROR", "FATAL")
# The option key is an identifier and the description carries the meaning. These
# ship from the source rather than from enrichment because the scale is the task -- the
# distance between WARN and ERROR is what an ordinal target is scored against, and an LLM
# gloss of "WARN" would be a restatement of the key, which enrichment drops to None.
LEVEL_CRITERIA = {
    "DEBUG": "Tracing a developer left in: internal state, entry and exit, no consequence",
    "INFO": "The normal course of events, worth recording but needing nobody",
    "NOTICE": "Normal but significant: a configuration or lifecycle event to be aware of",
    "WARN": "Something unexpected that the system handled and carried on from",
    "ERROR": "An operation failed; this request or this task did not complete",
    "FATAL": "The component cannot continue -- a crash, an abort, or hardware giving up",
}
LEVEL_ALIASES = {
    "V": "DEBUG", "VERBOSE": "DEBUG", "D": "DEBUG", "DEBUG": "DEBUG", "TRACE": "DEBUG",
    "I": "INFO", "INFO": "INFO", "INFORMATION": "INFO",
    "N": "NOTICE", "NOTICE": "NOTICE",
    "W": "WARN", "WARN": "WARN", "WARNING": "WARN",
    "E": "ERROR", "ERROR": "ERROR", "ERR": "ERROR",
    "F": "FATAL", "FATAL": "FATAL", "CRIT": "FATAL",
    "CRITICAL": "FATAL", "ALERT": "FATAL", "EMERG": "FATAL", "PANIC": "FATAL",
}
# Deliberately absent: BGL's `SEVERE` (and `FAILURE`). BG/L's own RAS scale runs INFO <
# WARNING < SEVERE < ERROR < FATAL < FAILURE, so SEVERE sits *below* ERROR and has no slot
# on the shared scale. It used to be aliased to FATAL, which labelled seven BGL lines two
# steps too severe and would have called an `ERROR -> SEVERE` run an escalation. A line
# whose level field holds it now has no level (dropped from `loghub_level`), and a burst
# that prints it is dropped rather than judged (`escalation_label`).

# The holdout is by *family*, not by system: the supercomputer logs share a positional,
# node-addressed dialect that nothing on the train side looks like, so reserving them
# tests transfer over format rather than over which file a line came from.
FAMILIES: dict[str, tuple[str, ...]] = {
    "distributed": ("HDFS", "Hadoop", "Spark", "Zookeeper", "OpenStack"),
    "supercomputer": ("BGL", "HPC", "Thunderbird"),
    "os": ("Windows", "Linux", "Mac"),
    "mobile": ("Android", "HealthApp"),
    "server_app": ("Apache", "OpenSSH", "Proxifier"),
}
FAMILY_OF: dict[str, str] = {s: fam for fam, ss in FAMILIES.items() for s in ss}
HELD_OUT_FAMILY = "supercomputer"
HELD_OUT = FAMILIES[HELD_OUT_FAMILY]
# Dev's own held-out family, never trained and never tested. The OS family is the only one
# whose holdout keeps every severity trained: DEBUG is Android's alone and NOTICE Apache's
# alone, so holding out `mobile` or `server_app` would hold out a level, which is a
# primitive, not a structure; `distributed` is most of the trained rows. Windows logs INFO
# only and Linux and Mac no level at all, so this family ships a component task and
# nothing else -- a dialect holdout over component vocabulary, ~190 questions, and the cost
# to training is those same components.
DEV_FAMILY = "os"
DEV_HELD_OUT = FAMILIES[DEV_FAMILY]

# One anchored pattern per system. `lvl` and `comp` are optional named groups; a system
# that publishes neither is not listed. Anchoring at ^ is the point: the level is the
# token in *that system's level field*, so a message that happens to contain the word
# "error" is not mistaken for an ERROR line, and a line that does not match the shape at
# all is dropped rather than half-read.
# `sandboxd[129] ([31211]):` -- macOS sandboxd names the sandboxed pid as well, so the
# optional second parenthesised pid is part of the shape, not a different one.
_SYSLOG = (r"^\w{3}\s+\d+ \d\d:\d\d:\d\d \S+ (?P<comp>[^\s:\[]+)"
           r"(?:\[\d+\])?(?: \(\[\d+\]\))?:\s")
_FORMATS: dict[str, str] = {
    # date time pid LEVEL component: message
    "HDFS": r"^\d{6} \d{6} \d+ (?P<lvl>[A-Z]+) (?P<comp>[\w.$]+):\s",
    # date time,ms LEVEL [thread] component: message
    "Hadoop": r"^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d+ (?P<lvl>[A-Z]+) \[.*?\] (?P<comp>[\w.$]+):\s",
    # yy/mm/dd time LEVEL component: message
    "Spark": r"^\d\d/\d\d/\d\d \d\d:\d\d:\d\d (?P<lvl>[A-Z]+) (?P<comp>[\w.$]+):\s",
    # date time,ms - LEVEL  [thread:Class@line] - message
    "Zookeeper": r"^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d+ - (?P<lvl>[A-Z]+)\s+\[(?P<comp>.*?)\] - ",
    # logfile date time pid LEVEL component [req-...] message
    "OpenStack": r"^\S+ \d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+ \d+ (?P<lvl>[A-Z]+) (?P<comp>[\w.$-]+)\s",
    # flag epoch date node timestamp node type COMPONENT LEVEL message
    "BGL": r"^\S+ \d+ \S+ \S+ \S+ \S+ \S+ (?P<comp>[A-Z_]+) (?P<lvl>[A-Z]+)\s",
    # date time, Level  Component  message
    "Windows": r"^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d, (?P<lvl>[A-Za-z]+)\s+(?P<comp>\S+)\s",
    # mm-dd time pid tid L Component: message
    "Android": r"^\d\d-\d\d \d\d:\d\d:\d\d\.\d+\s+\d+\s+\d+ (?P<lvl>[VDIWEF]) (?P<comp>[^:]+):\s",
    # [Day Mon dd hh:mm:ss yyyy] [level] message      (no component field)
    "Apache": r"^\[[^\]]+\] \[(?P<lvl>[a-z]+)\]\s",
    # id node component subtype epoch flag message     (no level field)
    "HPC": r"^\d+ \S+ (?P<comp>\S+) \S+ \d+ \d+\s",
    # flag epoch date node Mon dd hh:mm:ss host program[pid]: message
    "Thunderbird": r"^\S+ \d+ \S+ \S+ \w{3}\s+\d+ \d\d:\d\d:\d\d \S+ (?P<comp>[^\s:\[]+)(?:\[\d+\])?:\s",
    "Linux": _SYSLOG,
    "Mac": _SYSLOG,
    "OpenSSH": _SYSLOG,
    # yyyymmdd-hh:mm:ss:ms|Component|pid|message
    "HealthApp": r"^\d{8}-\d+:\d+:\d+:\d+\|(?P<comp>[^|]+)\|",
    # [mm.dd hh:mm:ss] program.exe - message
    "Proxifier": r"^\[\d\d\.\d\d \d\d:\d\d:\d\d\] (?P<comp>\S+)(?: \*\d+)? - ",
}
_COMPILED = {name: re.compile(p) for name, p in _FORMATS.items()}

# BGL and Thunderbird lines open with an *annotation*, not something the machine wrote:
# LogHub's first field is `-` for a non-alert line and an alert category (`KERNDTLB`,
# `APPSEV`, ...) where the dataset's authors flagged the line as a failure. In BGL every
# flagged line is FATAL (143 of 354), so leaving the field in the state lifted a lookup on
# it from the 0.400 system prior to 0.541 on `loghub_level_supercomputer`, and the
# `APP*` flags name the `APP` component outright. The field is stripped from every state
# these systems contribute; parsing still sees the raw line.
ALERT_FLAG_SYSTEMS = ("BGL", "Thunderbird")


def display_text(text: str, system: str) -> str:
    """What a reader is shown of a raw line: the line minus any annotation LogHub added."""
    if system in ALERT_FLAG_SYSTEMS:
        parts = text.split(" ", 1)
        return parts[1] if len(parts) == 2 else ""
    return text

# Zookeeper's component hides inside the thread field: `QuorumPeer[myid=1]/0:0:...:2181:
# FastLeaderElection@774` is one thread, one class and one source line. The class is the
# emitting component; the `@774` is where in it.
_COMP_FIXUP: dict[str, Callable[[str], str]] = {
    "Zookeeper": lambda s: s.rsplit(":", 1)[-1].split("@")[0].strip(),
}

# A component has to be an identifier, not a fragment of prose: qualified (`dfs.FSNamesystem`,
# `com.apple.CDScheduler`), an all-caps subsystem tag (`CBS`, `KERNEL`), or a plain program
# name (`sshd`, `crond(pam_unix)`, `chrome.exe`). The first version's regex matched any word
# before a colon and produced an option set of noise, which is worse than no task at all.
_COMPONENT_OK = re.compile(r"^[A-Za-z][\w.$()/-]{1,60}$")

# The shortcut floor for a state is 40 tokens, and the first version shipped `loghub_level`
# states at p5 = 31 -- short single lines that are a lookup, not a task. Measured over every
# level-bearing line in the cache with `Qwen/Qwen3-0.6B-Base`, 128 characters is the
# smallest floor at which no line falls under 40 tokens; 140 is that with a margin. A
# character floor rather than a token count keeps the loader free of a `transformers`
# import, which the corpus build does not otherwise need.
MIN_STATE_CHARS = 140

_IN_PROCESS: dict[str, object] = {}

# Question templates. Registered in `lod/assets/phrasing_specs/d13.json`; wordings come
# from the frozen phrasing bank, keyed by the record's own line, and the template itself
# is what ships while the bank is empty. `enrich.schemas.PROTECTED_PREFIXES` carries
# `loghub_` so enrichment cannot reword these: the burst question once said "at any
# point" while the code compared against the first line, and a paraphrase would reopen it.
LEVEL_QUESTION_ID = "d13.level"
LEVEL_QUESTION = ("A log line from a production system, with the severity the logger "
                  "printed blanked out as [LEVEL]. What severity did it have?")
COMPONENT_QUESTION_ID = "d13.component"
COMPONENT_QUESTION = ("A log line with the name of the emitting component blanked out as "
                      "[COMPONENT]. Which component of this system wrote it?")
BURST_QUESTION_ID = "d13.burst"
_BANK: PhrasingBank | None = None


def _phrasing(template_id: str, template: str, key: str) -> str:
    global _BANK
    if _BANK is None:
        _BANK = PhrasingBank.load()
    return _BANK.pick(template_id, template, key)


def _record_key(r: dict) -> str:
    return f"{r.get('system', '')}:{r.get('lineno', r.get('text', ''))}"


def cache_dir():
    """Where `scripts/fetch_data.py --rows 14` put the logs, under whatever raw root is configured."""
    return store.RAW_ROOT / "loghub"


def canon_level(raw: str) -> str | None:
    """A system's own severity token on the shared scale, or None when it is not one."""
    return LEVEL_ALIASES.get(raw.strip().upper())


def read_line(text: str, system: str) -> dict | None:
    """-> {level, level_span, component, component_span} for one line of `system`.

    None when the line does not match that system's shape at all. `level` and `component`
    are each None when the system has no such field or the field held something
    unrecognised -- the caller drops rather than guesses (THE INVARIANT).
    """
    pattern = _COMPILED.get(system)
    if pattern is None:
        return None
    m = pattern.match(text)
    if m is None:
        return None
    groups = m.groupdict()

    level = level_span = None
    level_raw = groups.get("lvl")
    if groups.get("lvl") is not None:
        level = canon_level(groups["lvl"])
        if level is not None:
            level_span = m.span("lvl")

    component = component_span = None
    raw_comp = groups.get("comp")
    if raw_comp is not None:
        fix = _COMP_FIXUP.get(system)
        cand = fix(raw_comp) if fix else raw_comp.strip()
        if cand and _COMPONENT_OK.match(cand) and canon_level(cand) is None:
            component = cand
            # after a fixup the span is the sub-span of the captured group
            start, end = m.span("comp")
            off = raw_comp.find(cand)
            component_span = ((start + off, start + off + len(cand)) if off >= 0
                              else (start, end))
    return {"level": level, "level_span": level_span, "level_raw": level_raw,
            "component": component, "component_span": component_span}


def _scrub(text: str, span: tuple[int, int] | None, placeholder: str,
           everywhere: str | None = None, eat_padding: bool = False) -> str:
    """Blank the span the answer was read from, the way `spdx._scrub` blanks a licence id.

    `everywhere` additionally removes every other verbatim occurrence of the answer. That
    is on for the component -- a syslog line names its daemon in the message as often as
    in the program field, and leaving the second copy measured a 23.6 % string match --
    and off for the level, deliberately: the level field is the machine's printed answer,
    but the word "error" inside the message is *evidence*, and reading evidence is the
    task. The residual match rate is reported rather than scrubbed away.

    `everywhere` matches whole tokens only (not preceded or followed by an identifier
    character). A substring match turned BGL's `/bgl/apps/` into `/bgl/[COMPONENT]s/`, so
    the *count* of placeholders named the component: a lookup on option set plus marker
    count scored 0.798 on `loghub_component_supercomputer` against a 0.473 marginal. The
    redaction itself was the leak. A name inside a longer word is message evidence, the
    same as "error" in a level line, and its rate was measured by the domain-13 audit.

    `eat_padding` also swallows the whitespace after the span and writes exactly one space.
    Log4j pads its level to five columns (`%-5p`), so in Zookeeper `INFO  [` and `WARN  [`
    keep two spaces after the redaction and `ERROR [` one: the width of the gap was the
    answer. Windows pads `Info` the same way to a fixed column.
    """
    out = text
    if span is not None:
        start, end = span
        if eat_padding:
            ws = len(out[end:]) - len(out[end:].lstrip(" "))
            if ws:
                end += ws
                placeholder = placeholder + " "
        out = out[:start] + placeholder + out[end:]
    if everywhere:
        out = re.sub(rf"(?<![\w.$]){re.escape(everywhere)}(?![\w.$])", placeholder, out,
                     flags=re.IGNORECASE)
    return out


def systems_available() -> list[str]:
    """The cached systems, in the fetch script's family order."""
    root = cache_dir()
    if not root.is_dir():
        return []
    have = {p.stem[: -len("_2k")] for p in root.glob("*_2k.log")}
    return [s for s in FAMILY_OF if s in have]


def parsed_rows() -> list[dict]:
    """Every cached log line with whatever could be read off it. Parsed once.

    Preserves file order within a system and never interleaves two systems, because
    `iter_bursts` reads consecutive entries as a run a human would have watched scroll
    past; a run that straddled two machines would not be one.
    """
    cached = _IN_PROCESS.get("rows")
    if cached is not None:
        return cached                                    # type: ignore[return-value]
    out: list[dict] = []
    root = cache_dir()
    for system in systems_available():
        family = FAMILY_OF[system]
        path = root / f"{system}_2k.log"
        for lineno, line in enumerate(path.open(errors="replace")):
            text = line.rstrip("\r\n").rstrip()
            if not text:
                continue
            got = read_line(text, system)
            out.append({"text": text, "system": system, "family": family,
                        "lineno": lineno, "parsed": got is not None,
                        "level": (got or {}).get("level"),
                        "level_raw": (got or {}).get("level_raw"),
                        "level_span": (got or {}).get("level_span"),
                        "component": (got or {}).get("component"),
                        "component_span": (got or {}).get("component_span")})
    if not out:
        out = _legacy_rows()
    _IN_PROCESS["rows"] = out
    return out


def _legacy_rows() -> list[dict]:
    """The original single-Apache-log store, used only when the 16-system cache is absent.

    Kept so a machine that has the old HF mirror but not the new fetch still builds
    something, rather than the source silently contributing nothing (a loader
    that can return empty is the trap that let the row-1 sweep claim a dataset).
    """
    try:
        raw = store.load(KEY)
    except FileNotFoundError:
        return []
    out = []
    for lineno, r in enumerate(raw):
        text = str(r.get("text") or "").strip()
        if not text:
            continue
        got = read_line(text, "Apache")
        out.append({"text": text, "system": "Apache", "family": FAMILY_OF["Apache"],
                    "lineno": lineno, "parsed": got is not None,
                    "level": (got or {}).get("level"),
                    "level_raw": (got or {}).get("level_raw"),
                    "level_span": (got or {}).get("level_span"),
                    "component": (got or {}).get("component"),
                    "component_span": (got or {}).get("component_span")})
    return out


def parse_report() -> dict[str, dict]:
    """Per system: lines seen, lines parsed, lines dropped, and the level histogram.

    Requirement 1 of the domain-13 brief is that the parser *says* how much it threw
    away; a parser whose drop rate nobody prints is a parser nobody can audit.
    """
    report: dict[str, dict] = {}
    for r in parsed_rows():
        s = report.setdefault(r["system"], {
            "family": r["family"], "lines": 0, "parsed": 0, "dropped": 0,
            "with_level": 0, "with_component": 0, "levels": Counter()})
        s["lines"] += 1
        if r["parsed"]:
            s["parsed"] += 1
        else:
            s["dropped"] += 1
        if r["level"]:
            s["with_level"] += 1
            s["levels"][r["level"]] += 1
        if r["component"]:
            s["with_component"] += 1
    for s in report.values():
        s["parse_rate"] = s["parsed"] / s["lines"] if s["lines"] else 0.0
        s["levels"] = dict(s["levels"].most_common())
    return report


# ---------------------------------------------------------------------------
# slicing: no log line ever backs two tasks
# ---------------------------------------------------------------------------

BLOCK = 30
# Bursts get two blocks in four. They are the scarcest thing in the cache -- a 6-line run
# needs two recognised levels *and* a level change to be worth anything, and the held-out
# family has exactly one level-bearing system -- while the single-line tasks have an order
# of magnitude more candidates than they can use. The cycle is where that is said once.
TASK_BLOCK_CYCLE = ("level", "component", "burst", "burst")


def _task_block(rows: list[dict], which: str) -> list[dict]:
    """Each system's lines cut into contiguous blocks of `BLOCK` and dealt round-robin to
    the three tasks, so no log line ever backs two of them.

    Blocks rather than the first version's three contiguous thirds. A log is not stationary --
    Hadoop's 150 ERROR lines are all in the back half of its sample, because that is when the job
    failed -- so cutting the file in three gave the level task a first third with no
    ERROR in it at all and the burst task a tail that was nothing but failure. Dealing
    30-line blocks gives every task a sample of the whole run while keeping each block
    contiguous, which `iter_bursts` needs: a burst has to be lines a human would have
    watched scroll past together.
    """
    by_system: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_system[r["system"]].append(r)
    out: list[dict] = []
    for lines in by_system.values():
        for b in range(0, len(lines), BLOCK):
            if TASK_BLOCK_CYCLE[(b // BLOCK) % len(TASK_BLOCK_CYCLE)] == which:
                out += lines[b:b + BLOCK]
    return out


# `held_out` throughout this module is False (the trained families), True (the test
# holdout) or "dev" (the dev holdout); the booleans keep their old meaning.
def _family_side(family: str) -> bool | str:
    return True if family == HELD_OUT_FAMILY else "dev" if family == DEV_FAMILY else False


def _side(rows: list[dict], held_out: bool | str) -> list[dict]:
    return [r for r in rows if _family_side(r["family"]) == held_out]


MIN_TASK_EXAMPLES = 60
# What a task has to reach to ship, which is not the same number as the floor
# `balanced_cap` relaxes towards. Balancing within systems costs examples -- the held-out
# burst pool goes from 60 at a 0.650 majority to 46 at 0.543 -- and 46 questions whose
# majority class is a coin flip measure more than 60 that are two thirds one answer.
MIN_SHIPPABLE_EXAMPLES = 40


def balanced_cap(sizes: list[int], max_share: float,
                 min_pool: int = MIN_TASK_EXAMPLES) -> int:
    """A per-class cap that holds the majority class down without emptying the task.

    The first version's `loghub_level` was 82.3 % ERROR and a first pass at the rework came
    out 87.1 % INFO, which is worse: a logger writes overwhelmingly INFO, so a loader that
    simply takes what it finds hands the corpus a majority-class task however many systems
    it reads. Capping is selection from real data, not fabrication -- every example kept is
    a line somebody's machine actually wrote.

    The largest cap meeting `max_share` is preferred. When that leaves fewer than
    `min_pool` examples the target is relaxed in steps and the best cap at the first
    workable step is used instead, because a perfectly balanced task of ten questions
    measures nothing either. The held-out BGL sample is the case that needs it: it writes
    INFO, FATAL and two ERROR lines, so no cap can get the majority under 0.40 and the
    honest answer is 0.50 over 144 questions, reported as such.
    """
    sizes = [n for n in sizes if n > 0]
    if not sizes:
        return 0
    caps = []
    for q in range(1, max(sizes) + 1):
        kept = [min(n, q) for n in sizes]
        caps.append((q, sum(kept), max(kept) / sum(kept)))
    target = max_share
    while target <= 0.90:
        feasible = [c for c in caps if c[2] <= target + 1e-9]
        if feasible:
            q, pool, _ = max(feasible, key=lambda c: c[1])
            if pool >= min_pool or target > 0.85:
                return q
        target += 0.05
    return max(sizes)


def _thin(rows: list, q: int) -> list:
    """`q` items spread evenly across `rows`, not the first `q`.

    The first `q` of a class would all come from the start of whichever system logs that
    level most, which puts a whole task inside one hour of one machine.
    """
    if q <= 0 or not rows:
        return []
    if len(rows) <= q:
        return list(rows)
    step = len(rows) / q
    return [rows[int(i * step)] for i in range(q)]


def _round_robin(groups: list[list]) -> Iterator:
    """Interleave the groups so any prefix is close to balanced.

    `generate.sample_examples` takes `load(limit)` and only sub-samples if it gets
    more than it asked for, so a loader that yields its majority class first hands the
    corpus a majority-class task whatever the pool looks like. Interleaving makes a quota
    smaller than the pool a balanced sample too, not just the pool as a whole.
    """
    groups = [g for g in groups if g]
    i = 0
    while groups:
        alive = []
        for g in groups:
            if i < len(g):
                yield g[i]
                alive.append(g)
        groups = alive
        i += 1


# A system whose own histogram cannot be brought under this is not asked about: its
# format alone answers the question. 0.60 is the loosest value at which the format-only
# lookup on `loghub_level` stays under the pooled 0.40 majority plus a fifth.
MAX_SYSTEM_MAJORITY = 0.60
# Per-system pools are small by construction, so the within-system cap is allowed to
# relax at a much lower floor than the whole-task one.
MIN_SYSTEM_EXAMPLES = 8


def _balance_within_systems(items, system_of, label_of, labels, max_share,
                            min_pool: int = MIN_SYSTEM_EXAMPLES,
                            max_system_majority: float = MAX_SYSTEM_MAJORITY) -> list:
    """Cap each system's own label histogram, then cap the pooled one.

    Balancing only the pooled histogram leaves the *conditional* prior untouched, and the
    line format names the system, so the conditional prior is readable without looking at
    the message. Measured on an earlier build: a lookup keyed on the state's skeleton up to the
    redaction scored 0.8158 on `loghub_level` (0.400 majority) and 0.7664 on
    `loghub_burst_escalate` (0.5498). Two systems are dropped outright rather than capped
    -- one that writes a single level (Spark, Windows: 2,000/2,000 INFO) and one no cap
    can bring under `max_system_majority` (HDFS's level block is 98 % INFO) -- because
    there is no sample of them that is not a format lookup.

    The pooled cap still runs afterwards: within-system balance does not imply pooled
    balance when the systems differ in which levels they write at all.
    """
    by_system: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for it in items:
        by_system[system_of(it)][label_of(it)].append(it)
    kept: list[list] = []
    for system in sorted(by_system):
        groups = [by_system[system][l] for l in labels]
        if sum(1 for g in groups if g) < 2:
            continue
        q = balanced_cap([len(g) for g in groups], max_share, min_pool=min_pool)
        thinned = [_thin(g, q) for g in groups]
        sizes = [len(g) for g in thinned if g]
        if not sizes or max(sizes) / sum(sizes) > max_system_majority:
            continue
        kept.append(list(_round_robin(thinned)))
    if not kept:
        return []
    interleaved = list(_round_robin(kept))
    by_label: dict[str, list] = defaultdict(list)
    for it in interleaved:
        by_label[label_of(it)].append(it)
    groups = [by_label[l] for l in labels]
    q = balanced_cap([len(g) for g in groups], max_share)
    return list(_round_robin([_thin(g, q) for g in groups]))


# ---------------------------------------------------------------------------
# loghub_level
# ---------------------------------------------------------------------------

LEVEL_MAX_SHARE = 0.40


def level_state(r: dict) -> str:
    scrubbed = _scrub(r["text"], r["level_span"], "[LEVEL]", eat_padding=True)
    return display_text(scrubbed, r.get("system", ""))[:4000]


def _unique_states(rows: list[dict]) -> list[dict]:
    """First occurrence of each state. Thunderbird's sample writes some lines twice to the
    second (`PCI interrupt ... -> IRQ 177`, 3 pairs in the held-out component task), and
    two copies of one question are one question counted twice."""
    seen: set[str] = set()
    out = []
    for r in rows:
        if r["state"] not in seen:
            seen.add(r["state"])
            out.append(r)
    return out


def level_candidates(held_out: bool | str) -> list[dict]:
    """Level-bearing lines whose *scrubbed* state clears the token floor.

    The floor is measured after redaction, not before: `[LEVEL]` is shorter than the
    token it replaces, and measuring the raw line let two examples through at 39 tokens.
    """
    key = f"level_cand_{held_out}"
    got = _IN_PROCESS.get(key)
    if got is None:
        got = [dict(r, state=level_state(r))
               for r in _side(_task_block(parsed_rows(), "level"), held_out)
               if r["level"]]
        got = _unique_states([r for r in got if len(r["state"]) >= MIN_STATE_CHARS])
        _IN_PROCESS[key] = got
    return got                                           # type: ignore[return-value]


def level_options() -> list[str]:
    """The shared option set: every level with real support, low to high.

    Derived from the *candidate* pool rather than from every parsed line, so no option is
    offered that no example can ever be. `NOTICE` is the case in point: Apache is the
    only system that writes it and no Apache line in the sample reaches 110 characters,
    so every one of them falls under the token floor and `NOTICE` has no support.

    Shared between the train side and the held-out side on purpose -- the holdout is
    meant to change the *dialect*, not the schema, so the two sides are the same question
    asked of different machines.
    """
    present = {r["level"] for r in level_candidates(False) + level_candidates(True)}
    return [l for l in LEVEL_ORDER if l in present]


def _level_pool(held_out: bool | str) -> list[dict]:
    return _balance_within_systems(level_candidates(held_out),
                                   lambda r: r["system"], lambda r: r["level"],
                                   level_options(), LEVEL_MAX_SHARE)


def _level_loader(held_out: bool | str):
    def load(n: int) -> Iterator[Example]:
        opts = level_options()
        if len(opts) < 2:
            return
        for r in _level_pool(held_out)[:n]:
            yield Example(
                task=_name("loghub_level", held_out),
                state=r["state"],
                questions=[Question(
                    id="level",
                    question=_phrasing(LEVEL_QUESTION_ID, LEVEL_QUESTION, _record_key(r)),
                    options=list(opts),
                    descriptions=[LEVEL_CRITERIA[l] for l in opts],
                    target=opts.index(r["level"]),
                    meta={"system": r["system"], "family": r["family"]})],
            )
    return load


# ---------------------------------------------------------------------------
# loghub_component
# ---------------------------------------------------------------------------

MIN_COMPONENT_COUNT = 8
MAX_COMPONENTS = 30
COMPONENT_MAX_SHARE = 0.40


def components_by_system(rows: list[dict], min_count: int = MIN_COMPONENT_COUNT,
                         k: int = MAX_COMPONENTS) -> dict[str, list[str]]:
    """Per system, the components frequent enough to be a label, for systems with >= 2.

    Per system rather than pooled: a pooled option list lets the line's own *format*
    eliminate nine tenths of it before the message is read, which turns a subsystem
    question back into a format question. A line from Spark is asked which Spark
    component wrote it.
    """
    counts: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        if r["component"]:
            counts[r["system"]][r["component"]] += 1
    out = {}
    for system, cnt in counts.items():
        keep = [c for c, n in cnt.most_common(k) if n >= min_count]
        if len(keep) >= 2:
            out[system] = keep
    return out


def component_state(r: dict) -> str:
    scrubbed = _scrub(r["text"], r["component_span"], "[COMPONENT]",
                      everywhere=r["component"])
    return display_text(scrubbed, r.get("system", ""))[:4000]


def component_candidates(held_out: bool | str) -> list[dict]:
    key = f"comp_cand_{held_out}"
    got = _IN_PROCESS.get(key)
    if got is None:
        got = [dict(r, state=component_state(r))
               for r in _side(_task_block(parsed_rows(), "component"), held_out)
               if r["component"]]
        got = _unique_states([r for r in got if len(r["state"]) >= MIN_STATE_CHARS])
        _IN_PROCESS[key] = got
    return got                                           # type: ignore[return-value]


def _component_pool(held_out: bool | str) -> list[tuple[dict, list[str]]]:
    block = component_candidates(held_out)
    opts_by_system = components_by_system(block)
    out: list[tuple[dict, list[str]]] = []
    for system in sorted(opts_by_system):
        opts = opts_by_system[system]
        by_comp: dict[str, list[dict]] = defaultdict(list)
        for r in block:
            if r["system"] == system and r["component"] in opts:
                by_comp[r["component"]].append(r)
        groups = [by_comp[c] for c in opts]
        q = balanced_cap([len(g) for g in groups], COMPONENT_MAX_SHARE)
        out.append([(r, opts) for r in _round_robin([_thin(g, q) for g in groups])])
    return list(_round_robin(out))


def _component_loader(held_out: bool | str):
    def load(n: int) -> Iterator[Example]:
        for r, opts in _component_pool(held_out)[:n]:
            yield Example(
                task=_name("loghub_component", held_out),
                state=r["state"],
                questions=[Question(
                    id="component",
                    question=_phrasing(COMPONENT_QUESTION_ID, COMPONENT_QUESTION,
                                       _record_key(r)),
                    options=list(opts),
                    target=opts.index(r["component"]),
                    meta={"system": r["system"], "family": r["family"]})],
            )
    return load


# ---------------------------------------------------------------------------
# loghub_burst_escalate
# ---------------------------------------------------------------------------

BURST_SIZE = 6
BURST_OPTIONS = ["steady", "escalates"]
# The criteria, not decoration: they name the baseline the comparison is against. An earlier
# wording asked whether severity "increase[s] at any point after the first line", which a
# literal reader answers `escalates` for `WARN INFO WARN ...` while `escalation_label`
# answers `steady` -- 62/642 = 9.7 % of the trained task disagreed with its own question.
BURST_CRITERIA = ["No line after the first is more severe than the first line: the run "
                  "holds that severity or falls below it",
                  "At least one line after the first reports a higher severity than the "
                  "first line"]
BURST_QUESTION = ("A run of consecutive log lines from one system. Taking the first line "
                  "that carries a severity as the baseline, is the run steady, with no "
                  "later line above that baseline, or does it escalate, with some later "
                  "line reporting a higher severity?")
BURST_MAX_SHARE = 0.55


def iter_bursts(rows: list[dict], size: int = BURST_SIZE) -> Iterator[list[dict]]:
    """Non-overlapping runs of `size` consecutive rows *of one system*, in log order.

    Real bursts, not synthesised ones: `parsed_rows()` preserves file order and
    `_task_block` hands out contiguous blocks, so a chunk is exactly what a reader
    watching that log would have seen scroll past together. Runs never straddle two
    systems, because a run of lines from two different machines is not a burst.
    """
    by_system: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_system[r.get("system", "")].append(r)
    for lines in by_system.values():
        for i in range(0, len(lines) - size + 1, size):
            yield lines[i:i + size]


def escalation_label(group: list[dict]) -> str | None:
    """-> "steady" / "escalates", or None when the burst cannot be judged.

    None in three cases, each a drop rather than a guess: a line prints a level field that
    has no place on the shared scale (BGL's `SEVERE`), so the reader cannot be expected to
    rank it; fewer than two lines carry a
    recognised level, so there is nothing to compare; or every line of the burst is
    byte-identical, which is a free "steady" that teaches nothing. The second case is
    what made the first version's burst task degenerate -- one Apache error_log repeats the
    same line for pages -- so it is measured (`burst_identical_share`) as well as dropped.
    """
    if any(r.get("level_raw") is not None and r["level"] is None for r in group):
        return None                  # a printed level with no place on the scale (SEVERE)
    leveled = [r["level"] for r in group if r["level"] in LEVEL_ORDER]
    if len(leveled) < 2:
        return None
    texts = [r.get("text") for r in group]
    if all(texts) and len(set(texts)) == 1:
        return None
    first = LEVEL_ORDER.index(leveled[0])
    label = ("escalates" if any(LEVEL_ORDER.index(l) > first for l in leveled[1:])
             else "steady")
    # Android's own scale has VERBOSE *below* DEBUG, and the shared scale folds the two.
    # A run whose answer depends on that fold (`V ... D` and nothing else rising) has two
    # defensible answers, so it is dropped. Measured at zero on the current cache; this
    # is the guard, not a repair.
    fine = [(LEVEL_ORDER.index(r["level"]),
             -1 if str(r.get("level_raw") or "").upper() in _BELOW_DEBUG else 0)
            for r in group if r["level"] in LEVEL_ORDER]
    fine_label = "escalates" if any(f > fine[0] for f in fine[1:]) else "steady"
    return label if fine_label == label else None


_BELOW_DEBUG = {"V", "VERBOSE", "TRACE"}


def burst_identical_share(held_out: bool | str) -> tuple[int, int]:
    """(bursts whose lines are all identical, bursts with >= 2 levels) -- the first version's
    defect, measured on the current cache rather than asserted away."""
    block = _side(_task_block(parsed_rows(), "burst"), held_out)
    same = total = 0
    for group in iter_bursts(block):
        if len([r for r in group if r["level"] in LEVEL_ORDER]) < 2:
            continue
        total += 1
        same += len({r["text"] for r in group}) == 1
    return same, total


def burst_candidates(held_out: bool | str) -> list[tuple[list[dict], str]]:
    key = f"burst_cand_{held_out}"
    got = _IN_PROCESS.get(key)
    if got is None:
        block = _side(_task_block(parsed_rows(), "burst"), held_out)
        got = [(g, lab) for g in iter_bursts(block)
               if (lab := escalation_label(g)) is not None]
        _IN_PROCESS[key] = got
    return got                                           # type: ignore[return-value]


def _burst_pool(held_out: bool | str) -> list[tuple[list[dict], str]]:
    return _balance_within_systems(burst_candidates(held_out),
                                   lambda gl: gl[0][0]["system"], lambda gl: gl[1],
                                   BURST_OPTIONS, BURST_MAX_SHARE)


def _burst_loader(held_out: bool | str):
    def load(n: int) -> Iterator[Example]:
        for group, label in _burst_pool(held_out)[:n]:
            yield Example(
                task=_name("loghub_burst_escalate", held_out),
                state="\n".join(display_text(r["text"], r["system"])[:500]
                                for r in group)[:4000],
                questions=[Question(
                    id="escalate",
                    question=_phrasing(BURST_QUESTION_ID, BURST_QUESTION,
                                       _record_key(group[0])),
                    options=list(BURST_OPTIONS),
                    descriptions=list(BURST_CRITERIA),
                    target=BURST_OPTIONS.index(label),
                    meta={"system": group[0]["system"],
                          "family": group[0]["family"]})],
            )
    return load


# ---------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------

HELD_OUT_SUFFIX = "_supercomputer"
DEV_SUFFIX = "_" + DEV_FAMILY


def _name(base: str, held_out: bool | str) -> str:
    if held_out == "dev":
        return base + DEV_SUFFIX
    return base + HELD_OUT_SUFFIX if held_out else base


def _split_for(held_out: bool | str) -> str:
    """The holdout is a structure, not a sample.

    The train side is *forced* to train rather than left to the name hash: the hash is
    what put every row-14 question in `testreal` in an earlier build, so the model had never
    seen a log line at all and there was nothing for the holdout to be a holdout of.
    `carve_val` then takes the in-dialect dev slice out of train.
    """
    if held_out == "dev":
        return "devreal"
    return "testreal" if held_out else "train"


_NOTE_SUFFIX = (" Licence: LogHub research/academic use, credit github.com/logpai/loghub "
                "and cite Zhu et al., ISSRE 2023.")


def tasks() -> list[RealTask]:
    if not systems_available() and not store.has(KEY):
        return []
    rows = parsed_rows()
    if not rows:
        return []
    out: list[RealTask] = []
    for held_out in (False, "dev", True):
        side = (DEV_FAMILY if held_out == "dev" else HELD_OUT_FAMILY if held_out else
                "distributed/mobile/server-app")
        split = _split_for(held_out)
        systems = sorted({r["system"] for r in _side(rows, held_out)})
        if not systems:
            continue
        if (len(_level_pool(held_out)) >= MIN_SHIPPABLE_EXAMPLES
                and len(level_options()) >= 2):
            out.append(RealTask(
                row=14, name=_name("loghub_level", held_out), licence=LICENCE, url=URL,
                load=_level_loader(held_out), ordinal=True, force_split=split,
                notes=f"severity read from each system's own level field and then "
                      f"blanked out of the state; {side} family ({', '.join(systems)}); "
                      f"DEBUG < INFO < NOTICE < WARN < ERROR < FATAL." + _NOTE_SUFFIX))
        if len(_component_pool(held_out)) >= MIN_SHIPPABLE_EXAMPLES:
            out.append(RealTask(
                row=14, name=_name("loghub_component", held_out), licence=LICENCE,
                url=URL, load=_component_loader(held_out), force_split=split,
                per_example_options=True,
                notes=f"emitting component, options are that system's own top "
                      f"{MAX_COMPONENTS} by frequency and the name is blanked out of the "
                      f"state; {side} family." + _NOTE_SUFFIX))
        if len(_burst_pool(held_out)) >= MIN_SHIPPABLE_EXAMPLES:
            out.append(RealTask(
                row=14, name=_name("loghub_burst_escalate", held_out), licence=LICENCE,
                url=URL, load=_burst_loader(held_out), force_split=split,
                notes=f"state is {BURST_SIZE} consecutive real lines from one system; "
                      f"target is whether severity rises after the first line, not any "
                      f"single line's own level; {side} family." + _NOTE_SUFFIX))
    return out


def _cli() -> None:                                      # pragma: no cover - diagnostics
    """`uv run python -m lod.corpus.services.sources.real.loghub` -- the per-system parse report."""
    print(json.dumps(parse_report(), indent=1, default=str))


if __name__ == "__main__":                               # pragma: no cover
    _cli()
