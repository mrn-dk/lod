"""Source-table row 5 — per-repository issue label taxonomies from GH Archive.

One schema per repository. A repo's label vocabulary is its own invented taxonomy, so
every qualifying repo contributes a distinct (question, option set), which is what makes
this the largest single source of *schemas* in the corpus — measured, one day of archive
yields 1,045 repos with at least five distinct labels.

Two things here are measured decisions:

**Issues per repo are whatever the window yields.** Row 5 asks for 100 single-label
issues per repo. Measured over a full day, the count of repos reaching 100 is zero, the
800th-ranked repo accumulates roughly none per day, and forcing it implies ~100 days of
archive and ~260 GB. The corpus's schema target is a count of schemas and row 5 defines
one schema per repo, so the 800 repos are the binding requirement; issues per repo only
feed the question total. The achieved distribution is reported by `issue_count_summary` and
belongs in the provenance manifest.

**The raw archive is streamed, never cached.** A 14-day window is ~37 GB of gzip that is
useless once the issues are extracted; only the extracted issues are written to disk,
which is roughly three orders of magnitude smaller.

The label question and the is-a-bug question draw from *disjoint repos*: a label task per
repo big enough to have one, and `gh_is_bug` over the repos that are not. They are
separate tasks, so the split hash can route them to different splits, and sharing states
-- or a repo's issues -- across splits is exactly what makes the devreal/testreal dedup
delete an eval set. (Until a domain-8 audit they took two halves of every repo's issues,
which kept states apart but put every eval repo's other half in `gh_is_bug`'s train.)

Only the extracted issues are cached, not the raw events, so an issue's author and its
later label events are not available here. What the loaders do about that -- workflow
labels, title tags, bots and template copies -- is under "What the model is shown" below.
"""

from __future__ import annotations

import gzip
import io
import json
import re
import unicodedata
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from datetime import date, timedelta
from itertools import zip_longest
from typing import Iterator

from lod.schema import Example, Question
from lod.corpus.services.sources.base import CACHE, UA
from lod.corpus.services.sources.real.base import RealTask

LICENCE = "CC-BY-4.0 (GH Archive aggregation of public GitHub events)"
URL = "https://www.gharchive.org/"

MIN_LABELS = 5           # row 5
MAX_LABELS = 30          # beyond this a "taxonomy" is usually bot-generated noise
MIN_ISSUES = 8           # see the module docstring: 100 is unreachable at sane cost
WINDOW_DAYS = 14
START = date(2024, 5, 1)

BUGGY = ("bug", "defect", "regression", "crash", "broken")
# `bug` as a bare substring says yes to "BugFix", "debug" and -- worst -- "not a bug".
# Measured over the cached window: 59,076 issues, 15,716 of them matched the substring
# rule and 66 of those matches were wrong in exactly that way (38 "BugFix", 23 "not a
# bug" and variants, 3 "debug"/"debugging"). Small, and still a label that says the
# opposite of its target. `_is_bug` matches whole words and refuses a negated one.
NOT_BUGGY = re.compile(r"\b(not|non|no)[\s\-_]*a?[\s\-_]*(bug|defect)\b|\bbugfix", re.I)
_WORD = re.compile(r"[^0-9a-z]+")
# auto-generated vocabularies (comment bots, commit hashes) are not taxonomies
JUNK = re.compile(r"^[0-9a-f]{16,}$|^gitalk$|^\s*$")
_CACHE_FILE = CACHE / "gharchive_issues.jsonl"


def _is_bug(label: str) -> bool:
    """Whether a maintainer's own label calls this issue a bug.

    This is the one target in row 5 that is *derived* rather than published: the label is
    GitHub's, the boolean is ours. So the derivation is kept narrow and legible -- whole
    words from `BUGGY`, and never a label that negates one.
    """
    if NOT_BUGGY.search(label):
        return False
    words = set()
    for word in _WORD.split(label.lower()):
        words.add(word)
        # "ios-bugs", "new-regressions", "crashes" are the same label pluralised
        if word.endswith("es"):
            words.add(word[:-2])
        if word.endswith("s"):
            words.add(word[:-1])
    return any(b in words for b in BUGGY)


def _clean(label: str) -> str | None:
    lab = label.strip()
    if not lab or len(lab) > 34 or JUNK.match(lab.lower()):
        return None
    return lab


def _hours(days: int, start: date) -> Iterator[str]:
    for d in range(days):
        day = start + timedelta(days=d)
        for h in range(24):
            # GH Archive does not zero-pad the hour: 2024-03-07-9.json.gz
            yield f"{day.isoformat()}-{h}"


# Read once per process. These are called once per task, and row 5 alone has ~2,000
# tasks; without this the build re-parses the whole cache for each of them. Same
# failure the raw store had, and the fetch/build split before that.
_IN_PROCESS: dict[str, list] = {}


def collect(days: int = WINDOW_DAYS, start: date = START,
            refresh: bool = False, verbose: bool = True) -> list[dict]:
    """Stream the window and keep only single-label issues. Cached as extracted issues."""
    if _CACHE_FILE.exists() and not refresh:
        if "issues" not in _IN_PROCESS:
            _IN_PROCESS["issues"] = [json.loads(l) for l in
                                     _CACHE_FILE.read_text().splitlines() if l.strip()]
        return _IN_PROCESS["issues"]

    CACHE.mkdir(parents=True, exist_ok=True)
    vocab: dict[str, set] = defaultdict(set)
    issues: list[dict] = []
    seen_ids: set[int] = set()
    gb = 0.0

    for i, hour in enumerate(_hours(days, start)):
        try:
            req = urllib.request.Request(f"https://data.gharchive.org/{hour}.json.gz",
                                         headers=UA)
            body = urllib.request.urlopen(req, timeout=300).read()
        except (urllib.error.URLError, TimeoutError, OSError):
            continue
        gb += len(body) / 1e9
        with gzip.open(io.BytesIO(body)) as f:
            for line in f:
                try:
                    ev = json.loads(line)
                except Exception:
                    continue
                if ev.get("type") != "IssuesEvent":
                    continue
                iss = (ev.get("payload") or {}).get("issue") or {}
                repo = (ev.get("repo") or {}).get("name")
                iid = iss.get("id")
                if not repo or iid is None or iid in seen_ids:
                    continue
                labels = [x for x in ((l.get("name") for l in iss.get("labels") or []))
                          if x]
                labels = [c for c in (_clean(l) for l in labels) if c]
                if not labels:
                    continue
                vocab[repo].update(labels)
                if len(labels) != 1:
                    continue
                title = (iss.get("title") or "").strip()
                bodytext = (iss.get("body") or "")[:3000]
                if not title:
                    continue
                seen_ids.add(iid)
                issues.append({"repo": repo, "label": labels[0], "title": title,
                               "body": bodytext})
        if verbose and (i + 1) % 24 == 0:
            q = sum(1 for r, v in vocab.items() if MIN_LABELS <= len(v) <= MAX_LABELS)
            print(f"  gharchive: day {(i+1)//24:2}/{days}  {gb:5.1f} GB  "
                  f"{len(issues):7,} issues  {q:5,} repos qualify on labels", flush=True)

    keep = {r for r, v in vocab.items() if MIN_LABELS <= len(v) <= MAX_LABELS}
    out = [x for x in issues if x["repo"] in keep]
    for x in out:
        x["vocab"] = sorted(vocab[x["repo"]])
    _CACHE_FILE.write_text("".join(json.dumps(x) + "\n" for x in out))
    return out


def qualifying_repos(issues: list[dict], min_issues: int = MIN_ISSUES) -> list[str]:
    counts: dict[str, int] = defaultdict(int)
    for x in issues:
        counts[x["repo"]] += 1
    return sorted(r for r, c in counts.items() if c >= min_issues)


def issue_count_summary(issues: list[dict]) -> dict[str, int]:
    """How many repos reach each issue count. Belongs in the provenance manifest: row 5
    asked for 100 per repo and this is the honest record of what the window actually
    produced."""
    counts: dict[str, int] = defaultdict(int)
    for x in issues:
        counts[x["repo"]] += 1
    out = {}
    for thr in (5, 8, 10, 25, 50, 100):
        out[f">={thr}"] = sum(1 for c in counts.values() if c >= thr)
    return out


def _task_name(repo: str) -> str:
    return "gh_label_" + re.sub(r"[^A-Za-z0-9]+", "_", repo).strip("_").lower()[:60]


# ---------------------------------------------------------------------------------------
# What the model is shown, and which labels it chooses among (domain-8 audit).
#
# Measured by that audit on the loaders as they stood, over 23,919 label
# questions from 2,079 repos:
#   * 11.7 % had a *workflow* label as the answer -- `stale`, `triage`, `status`,
#     `good first issue`, `Status: Untriaged`, `1k PEQ` -- and 61.4 % offered one as a
#     wrong option. Those say where an issue is in a process, not what it is about: the
#     stale bot applies `stale` after 60 quiet days, and an issue template applies
#     `needs triage` to every issue it opens. Neither is readable from the issue text,
#     and an issue labelled `bug` is very often *also* `confirmed` or `P1`.
#   * 7.5 % carried the answer as a title tag, `[Bug] ...`, `[FRONT] ...`, `[HackerNews]`
#     -- the issue template writes the tag and applies the label in one act.
#   * 4.7 % of states occurred more than once: uptime monitors ("🛑 X is down", 1.6 %),
#     CI failure bots, and classroom repos whose issues are one template copied into
#     thirty team repos, which then land on both sides of the split.
#   * `bug` and `Bug`, or `bug` and `type: bug`, in one option set (0.3 %).
# Nothing below relabels an issue: a workflow-labelled issue is dropped, a workflow label
# is removed from the options (it was not applied to any issue that remains), and two
# spellings of one label are merged to the spelling the repo uses most.

# Normalised (see `_norm`) label text that names a workflow state, not a topic.
PROCESS = re.compile(r"""^(?:
    stale|inactive|no[ ]issue[ ]activity|abandoned|lifecycle\b.*|frozen|locked|archived
  | wont[ ]?fix|won[ ]t[ ]fix|will[ ]not[ ]fix|not[ ]planned|invalid|duplicated?|dupe|spam
  | (?:can[ ]?t|can[ ]not|cannot|unable[ ]to)[ ]reproduce|works?[ ]?for[ ]?me|not[ ]a[ ]bug
  | (?:needs?|pending|awaiting|waiting|requires?)(?:[ ].*)?
  | triaged?|untriaged|triage[ ].*|.*[ ]triage|investigating|under[ ]investigation
  | (?:status|state|resolution|priority|prio|severity|sev|stage|phase|size|effort
     |estimate|points?|story[ ]points?|milestone|sprint)\b.*
  | 状态.*|优先级.*
  | p[ ]?[0-4]|s[0-4]
  | (?:high|low|medium|mid|normal|top|critical|urgent)(?:[ ]priority)?|blocker|blocking
  | important
  | in[ ]progress|wip(?:[ ].*)?|work[ ]in[ ]progress|doing|done|todo|to[ ]do|backlog
  | icebox|ready(?:[ ].*)?|(?:in[ ])?review|reviewed|approved|accepted|rejected|declined
  | closed|open|new|fixed(?:[ ].*)?|resolved|released|confirmed|unconfirmed|verified
  | unvalidated|reproduced|blocked|on[ ]hold|planned|answered|solved|unread
  | good[ ]first[ ]issues?|help[ ]wanted|first[ ]timers?[ ]only|up[ ]for[ ]grabs
  | hacktoberfest.*|beginner[ ]friendly|easy|contributions?[ ]welcome|prs?[ ]welcome
  | bounty|.*\bpeq|.*allocpeq|sponsored
)$""", re.X)
# a namespace word in front of the label proper: `type: bug`, `kind/bug`, `C-bug`, `T-bug`
_NAMESPACE = {"type", "kind", "t", "c", "category", "cat", "issue", "tag", "topic",
              "label", "class"}
# `:bug:` as a word of its own; `Bug:Reproduceable:No` is three words, not a shortcode
_SHORTCODE = re.compile(r"(?:^|(?<=\s)):[a-z0-9_+\-]+:(?=\s|$)")
_NONWORD = re.compile(r"[\W_]+")


def _norm(label: str) -> str:
    """Lower case, emoji, symbols and `:shortcodes:` gone, punctuation to single spaces."""
    text = "".join(ch for ch in _SHORTCODE.sub(" ", label.lower())
                   # U+2139 is the one emoji Unicode files as a letter
                   if not unicodedata.category(ch).startswith("S") and ch != "\u2139")
    return _NONWORD.sub(" ", text).strip()


# workflow words that make a label a workflow label wherever they sit in it:
# `Bug Priority:Low`, `Bug:Level of Effort:Low`, `staled-issue`, `pr-ready`
PROCESS_ANYWHERE = re.compile(r"\b(?:priority|prio|severity|effort|estimate|triage|triaged"
                              r"|stale|staled|reproduc\w*|repro|wontfix|duplicate|bounty"
                              r"|sprint|backlog|pr ready|ready for \w+)\b")


def _is_process(label: str) -> bool:
    """Whole label, or any `/`- or `|`-separated half of it: `wontfix / 不做`,
    `status/triage`. (`kind/bug` is safe: neither `kind` nor `bug` is a workflow word.)"""
    parts = [label] + [p for p in re.split(r"[/|]", label) if p.strip()]
    return any(PROCESS.match(_norm(p)) or PROCESS_ANYWHERE.search(_norm(p)) for p in parts)


def _alias_key(label: str) -> str:
    """Two labels with the same key are the same label spelled twice."""
    toks = _norm(label).split()
    if len(toks) > 1 and toks[0] in _NAMESPACE:
        toks = toks[1:]
    if toks and len(toks[-1]) > 3 and toks[-1].endswith("s") and not toks[-1].endswith("ss"):
        toks[-1] = toks[-1][:-1]
    # `Front-end` / `frontend`, `feature-request` / `Feature Request`
    return "".join(toks)


FEATURE = ("feature", "feat", "enhancement", "improvement", "proposal", "idea",
           "suggestion", "documentation", "docs", "doc")


def _scoped(label: str) -> bool:
    """`module: proposals`, `area/docs`, `component: crash-reporter`: a label naming a part
    of the project, whose words say nothing about what kind of issue it is. decidim's
    `module: proposals` read as a feature label before this."""
    parts = re.split(r"[:/]", label, maxsplit=1)
    if len(parts) < 2 or not parts[1].strip():
        return False
    head = _norm(parts[0])
    return head not in _NAMESPACE and not _is_bug(parts[0]) and not _is_feature_word(parts[0])


def _is_feature_word(label: str) -> bool:
    words = set()
    for word in _WORD.split(label.lower()):
        words.add(word)
        if word.endswith("s"):
            words.add(word[:-1])
    return any(f in words for f in FEATURE)


def _is_bug_label(label: str) -> bool:
    """The "yes" side of `gh_is_bug`: `_is_bug`, and not scoped to a part of the project."""
    return _is_bug(label) and not _scoped(label)


def _is_feature(label: str) -> bool:
    """A maintainer's label saying the issue asks for something new or for docs --
    the "no" side of `gh_is_bug`. Same whole-word reading as `_is_bug`."""
    if _is_bug(label) or NOT_BUGGY.search(label) or _scoped(label):
        return False
    return _is_feature_word(label)


# A title's leading tag -- `[Bug]`, `(feat)`, `【需求】`, `<FRONT>` -- and then a short
# `Bug:` / `feat:` / `Edit:` prefix. Both are how an issue template (or a bot) writes the
# label into the title; the text after them is the author's.
_LEAD_SYMBOLS = re.compile("^[\\s\u2600-\u27bf\U0001F000-\U0001FFFF\ufe0f\u200d]+")
_TITLE_TAG = re.compile(r"^\s*(?:\[[^\]\n]{1,40}\]|【[^】\n]{1,40}】|\([^)\n]{1,30}\)"
                        r"|<[^>\n]{1,30}>)\s*[:：\-–|]?\s*")
_TITLE_PREFIX = re.compile(r"^\s*[^\s:：\[\]/]+(?:\s+[^\s:：\[\]/]+){0,2}\s*[:：](?:\s+|$)")
# body scaffolding an issue template writes: headings, bold-only lines, HTML comments
_COMMENT = re.compile(r"<!--.*?-->", re.S)
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s.*$|^\s*(?:\*\*|__)[^*_\n]{1,80}(?:\*\*|__)\s*:?\s*$"
                      r"|^\s*_No response_\s*$"
                      # the checklist a template makes the author tick: "- [X] I understand
                      # this is a bug report", "- [x] I have searched for existing issues"
                      r"|^\s*[-*]\s*\[[ xX]\]\s.*$", re.M)


def _scrub_title(title: str) -> str:
    t = _LEAD_SYMBOLS.sub("", title)
    for _ in range(3):
        stripped = _TITLE_TAG.sub("", t, count=1)
        if stripped == t:
            break
        t = stripped
    return _TITLE_PREFIX.sub("", t, count=1).strip()


def _scrub_body(body: str) -> str:
    b = _HEADING.sub("", _COMMENT.sub("", body))
    return re.sub(r"\n\s*\n(\s*\n)+", "\n\n", b).strip()


# Issues nobody filed by hand. `collect()` does not keep the author (the raw stream is not
# cached), so these are read off the text: measured against 9 raw hours of the window,
# every rule below fires on automation (uptime monitors, CI bots, dependency-upgrade
# bots, classroom templates), and 5.5 % of the traced examples were opened by a bot
# account before them.
_UPTIME = re.compile("^\\s*(?:\U0001F6D1|\u26a0\ufe0f?|\U0001F7E5|\U0001F7E7|\U0001F7E8|\U0001F7E9|\u2705|\u274c)"
                     r"\s*.+\b(?:is down|has degraded performance|is up)\s*$", re.I)
_HEXRUN = re.compile(r"\b[0-9a-f]{7,}\b")
_DIGITS = re.compile(r"\d+")


def _skeleton(text: str, n: int) -> str:
    t = _DIGITS.sub("#", _HEXRUN.sub("#", text.lower()))
    return re.sub(r"\s+", " ", t).strip()[:n]


_LEAD_TAG = re.compile(r"^\W*[\[\(【<]([^\]\)】>\n]{1,40})[\]\)】>]")


def _automated(issues: list[dict]) -> set[int]:
    """Indices of issues that are copies or machine output rather than an issue report.

    * an uptime-monitor title ("🛑 api.example.com is down");
    * the same title and opening (digits and hashes masked, title tag and template
      headings removed, i.e. as the model would see them) twice in one repo -- a CI or
      monitor bot re-filing, a news feed posting one story under two tags, or an
      unedited template filed twice;
    * the same title five times in one repo whatever the body ("Samples deployment
      failed for eshop-aws");
    * the same title and opening in two repos -- a classroom or org template copied into
      every team repo, which the split then puts on both sides;
    * a feed: a repo in which >= 80 % of titles open with a tag that *is* the issue's own
      label and no label is a bug or feature label. The template repos that tag titles
      (`[Bug]`, `[Feature]`) have both kinds and keep their issues, scrubbed; the one
      repo this removes in the cached window is `SecOpsNews/news`, 500 security-news
      headlines filed by github-actions[bot] and labelled with the outlet's name.
    """
    full: dict[tuple, list[int]] = defaultdict(list)
    title: dict[tuple, list[int]] = defaultdict(list)
    across: dict[tuple, set] = defaultdict(set)
    keys = []
    for i, x in enumerate(issues):
        t = _skeleton(_scrub_title(x["title"]), 200)
        b = _skeleton(_scrub_body(x["body"]), 300)
        keys.append((t, b))
        full[(x["repo"], t, b)].append(i)
        title[(x["repo"], t)].append(i)
        across[(t, b)].add(x["repo"])
    out = {i for i, x in enumerate(issues) if _UPTIME.match(x["title"])}
    for idx in full.values():
        if len(idx) >= 2:
            out.update(idx)
    for (_, t), idx in title.items():
        if len(idx) >= 5 and len(t.split()) >= 2:
            out.update(idx)
    for i in range(len(issues)):
        if len(across[keys[i]]) >= 2:
            out.add(i)
    by_repo: dict[str, list[int]] = defaultdict(list)
    for i, x in enumerate(issues):
        by_repo[x["repo"]].append(i)
    for idx in by_repo.values():
        vocab = issues[idx[0]]["vocab"]
        if any(_is_bug(v) or _is_feature(v) for v in vocab):
            continue
        mirrored = 0
        for i in idx:
            m = _LEAD_TAG.match(issues[i]["title"])
            mirrored += bool(m) and _alias_key(m.group(1)) == _alias_key(issues[i]["label"])
        if mirrored >= 0.8 * len(idx):
            out.update(idx)
    return out


MIN_OPTIONS = 3      # topical labels left after the workflow ones are removed
MIN_WORDS = 3        # a state shorter than this is a label, not an issue


def _prepared() -> dict[str, dict]:
    """repo -> {"options": the repo's topical labels, one spelling each,
                "rows": [{"label", "title", "body", "raw_label"}]}.

    Every issue that survives is one a maintainer labelled with a topical label, shown
    without the title tag or template headings that restate that label.
    """
    if "prepared" in _IN_PROCESS:
        return _IN_PROCESS["prepared"]
    issues = collect()
    auto = _automated(issues)
    groups: dict[str, list[dict]] = defaultdict(list)
    for i, x in enumerate(issues):
        if i not in auto:
            groups[x["repo"]].append(x)
    out: dict[str, dict] = {}
    for repo, rows in groups.items():
        topical = [v for v in rows[0]["vocab"] if not _is_process(v)]
        used = Counter(x["label"] for x in rows)
        by_key: dict[str, list[str]] = defaultdict(list)
        for v in topical:
            by_key[_alias_key(v)].append(v)
        canon = {}
        for spellings in by_key.values():
            keep = sorted(spellings, key=lambda s: (-used[s], s))[0]
            for s in spellings:
                canon[s] = keep
        options = sorted(set(canon.values()))
        kept = []
        for x in rows:
            if x["label"] not in canon:        # a workflow label, or not in the vocab
                continue
            t, b = _scrub_title(x["title"]), _scrub_body(x["body"])
            if len((t + " " + b).split()) < MIN_WORDS:
                continue
            kept.append({"label": canon[x["label"]], "raw_label": x["label"],
                         "title": t, "body": b})
        out[repo] = {"options": options, "rows": kept}
    _IN_PROCESS["prepared"] = out
    return out


def _label_repos() -> list[str]:
    """Repos that get their own label task: enough issues, enough topical labels, and
    more than one answer among them -- a repo whose every issue is `bug` is a constant,
    and the answerability gate would only drop it later."""
    out = []
    for repo, p in _prepared().items():
        golds = {x["label"] for x in p["rows"]}
        if (len(p["rows"]) >= MIN_ISSUES and len(p["options"]) >= MIN_OPTIONS
                and len(golds) >= 2):
            out.append(repo)
    return sorted(out)


def _state(x: dict) -> str:
    return (x["title"] + ("\n\n" + x["body"] if x["body"] else ""))[:4000]


LABEL_QUESTION = "Which one of this repository's labels did its maintainers apply to this issue?"
BUG_QUESTION = ("Did the maintainers label this issue a bug report, rather than a feature "
                "request, enhancement or documentation issue?")
_BANK = None


def _phrase(template_id: str, template: str, key: str) -> str:
    global _BANK
    if _BANK is None:
        from lod.phrasings import PhrasingBank
        _BANK = PhrasingBank.load()
    return _BANK.pick(template_id, template, key)


def _label_loader(repo: str):
    def load(n: int) -> Iterator[Example]:
        p = _prepared().get(repo) or {"options": [], "rows": []}
        opts = p["options"]
        for i, x in enumerate(p["rows"]):
            if i >= n:
                return
            state = _state(x)
            yield Example(
                task=_task_name(repo),
                state=state,
                questions=[Question(id="label",
                                    question=_phrase("d08.gh_label", LABEL_QUESTION,
                                                     f"{repo}\x00{state[:200]}"),
                                    options=list(opts), target=opts.index(x["label"]),
                                    meta={"repo": repo})],
            )
    return load


def _bug_candidates() -> list[dict]:
    """The is-a-bug pool: issues labelled as a bug or as a feature/docs request, from
    repos that have *both* kinds of label and are too small for a label task of their own.

    Three things were wrong with the half-of-every-repo pool this replaces (domain-8
    audit): 34.6 % of its `no` answers came from a label that says nothing about bug-ness
    (`frontend`, `needs triage`, `question`), 23.6 % from repos with no feature label
    at all, where "not labelled bug" cannot mean "not a bug"; and every repo in it also
    had a `gh_label_*` task that the split hash could send to eval. Drawing from the
    repos that have no label task keeps the two tasks apart whatever the seed.

    Round-robin over repos with the two answers interleaved, so a prefix of any length
    is broad and close to balanced.
    """
    labelled = set(_label_repos())
    held = []
    for repo, p in sorted(_prepared().items()):
        if repo in labelled:
            continue
        if not (any(_is_bug_label(o) for o in p["options"])
                and any(_is_feature(o) for o in p["options"])):
            continue
        rows = [dict(x, repo=repo) for x in p["rows"]
                if _is_bug_label(x["raw_label"]) or _is_feature(x["raw_label"])]
        if rows:
            held.append(rows)
    yes: list[dict] = []
    no: list[dict] = []
    for i in range(max((len(h) for h in held), default=0)):
        for rows in held:
            if i < len(rows):
                (yes if _is_bug_label(rows[i]["raw_label"]) else no).append(rows[i])
    out = []
    for a, b in zip_longest(yes, no):
        if a is not None:
            out.append(a)
        if b is not None:
            out.append(b)
    return out


def _bug_loader():
    def load(n: int) -> Iterator[Example]:
        made = 0
        for x in _bug_candidates():
            if made >= n:
                return
            made += 1
            state = _state(x)
            yield Example(
                task="gh_is_bug",
                state=state,
                questions=[Question(id="is_bug",
                                    question=_phrase("d08.gh_is_bug", BUG_QUESTION,
                                                     f"{x['repo']}\x00{state[:200]}"),
                                    options=["no", "yes"],
                                    target=int(_is_bug_label(x["raw_label"])),
                                    meta={"repo": x["repo"]})],
            )
    return load


# `task_quotas` in services/generate.py divides row 5's budget evenly over its ~1,900
# tasks, which hands this pooled task the ~20 questions of one small repo. It asks for
# the per-task cap instead; see the `min_quota` note on `RealTask`.
BUG_MIN_QUOTA = 2000


def tasks(max_repos: int | None = None) -> list[RealTask]:
    """One task per qualifying repo, plus the shared is-a-bug task.

    Returns an empty list before `collect()` has run, so importing this module and
    building the registry never touches the network.
    """
    if not _CACHE_FILE.exists():
        return []
    repos = _label_repos()
    if max_repos:
        repos = repos[:max_repos]
    out = [RealTask(row=5, name=_task_name(r), licence=LICENCE, url=URL,
                    load=_label_loader(r),
                    notes=f"{r}: per-repo label taxonomy, workflow labels removed")
           for r in repos]
    out.append(RealTask(row=5, name="gh_is_bug", licence=LICENCE, url=URL,
                        load=_bug_loader(), min_quota=BUG_MIN_QUOTA,
                        notes="bug vs feature/docs label, from repos without a label task"))
    return out
