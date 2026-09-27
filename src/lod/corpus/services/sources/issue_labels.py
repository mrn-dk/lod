"""issue_labels — repo-specific label taxonomies from GH Archive.

Ground truth is taxonomic: maintainers applied the labels. Each repository's own
label vocabulary is a distinct schema, which is what makes this scale — there are
thousands of repos in a single hour of events.

Source: https://www.gharchive.org/ — public GitHub event stream (tier 1).
"""

from __future__ import annotations

import random
import re
from collections import defaultdict

from lod.schema import Example, Question
from lod.corpus.services.sources.base import (
    SourceInfo,
    SourceResult,
    fetch_jsonl_gz,
    schema_key,
)

INFO = SourceInfo(
    name="issue_labels",
    url="https://data.gharchive.org/",
    licence="CC-BY-4.0 (GH Archive aggregation of public GitHub events)",
    tier=1,
    ground_truth_type="taxonomic",
    state_format="prose",
    option_type="fixed",
    notes="single-label issues only; per-repo label vocabulary; issue text is authors' own",
)
YN = ["no", "yes"]
# GH Archive does not zero-pad the hour: 2024-03-07-9.json.gz, not -09
HOURS = ["2024-01-15-12", "2024-03-07-9", "2024-06-19-16", "2024-09-11-21",
         "2024-11-05-3", "2025-02-18-14", "2025-05-22-8", "2025-08-14-19"]
BUGGY = ("bug", "defect", "regression", "crash")
FEATURE = ("feature", "enhancement", "proposal", "idea")
# auto-generated label vocabularies (comment bots, hashes) are not taxonomies
JUNK = re.compile(r"^[0-9a-f]{16,}$|^gitalk$|^\s*$")


def _clean(label: str) -> str | None:
    lab = label.strip()
    if not lab or len(lab) > 34 or JUNK.match(lab.lower()):
        return None
    return lab


def generate(n: int, seed: int = 0, hours: int = 4, per_repo: int = 6,
             max_labels: int = 30, min_labels: int = 3) -> SourceResult:
    rng = random.Random(seed)
    res = SourceResult(info=INFO)
    vocab: dict[str, set] = defaultdict(set)
    issues: dict[str, list] = defaultdict(list)

    for hour in HOURS[:hours]:
        try:
            # fetch_jsonl_gz is lazy, so the download happens inside this loop
            stream = list(fetch_jsonl_gz(f"https://data.gharchive.org/{hour}.json.gz", timeout=180))
        except Exception as e:
            print(f"  issue_labels: skipping {hour}: {type(e).__name__}")
            continue
        for ev in stream:
            if ev.get("type") != "IssuesEvent":
                continue
            payload = ev.get("payload") or {}
            issue = payload.get("issue") or {}
            labels = [_clean(x.get("name", "")) for x in (issue.get("labels") or [])]
            labels = [x for x in labels if x]
            title = (issue.get("title") or "").strip()
            if not labels or not title or len(title) < 12:
                continue
            repo = (ev.get("repo") or {}).get("name", "")
            if not repo:
                continue
            vocab[repo].update(labels)
            if len(labels) == 1:  # single-label only, so the target is unambiguous
                body = re.sub(r"\s+", " ", (issue.get("body") or ""))[:700]
                issues[repo].append((title, body, labels[0]))

    repos = [r for r, v in vocab.items()
             if min_labels <= len(v) <= max_labels and len(issues.get(r, [])) >= 2]
    rng.shuffle(repos)
    for repo in repos:
        if len(res.examples) >= n:
            break
        options = sorted(vocab[repo])
        task = "gh_" + re.sub(r"[^A-Za-z0-9]+", "_", repo)[:48]
        has_bug = [o for o in options if any(b in o.lower() for b in BUGGY)]
        has_feat = [o for o in options if any(f in o.lower() for f in FEATURE)]
        for title, body, gold in issues[repo][:per_repo]:
            if gold not in options or len(res.examples) >= n:
                continue
            state = f"Repository: {repo}\nIssue: {title}"
            if body:
                state += f"\n\n{body}"
            qs = [Question("label", "Which label applies to this issue?", options,
                           options.index(gold))]
            if has_bug and has_feat:  # only decidable when the repo distinguishes them
                qs.append(Question("is_bug", "Is this a bug report?", list(YN),
                                   int(gold in has_bug)))
            res.examples.append(Example(state=state, questions=qs, task=task))
            for q in qs:
                res.schemas.add(schema_key(q.question, q.options))
    return res
