"""Source-table row 31 — TabFact: is the statement entailed by the table?

The working mirror ships the dataset in an RL-training shape rather than as columns: the
state is inside `prompt`, a chat message list, and the label is nested at
`reward_model.ground_truth`. The generic column adapter finds nothing, so this reads both
out directly.

The system message and the "Instruction" preamble are stripped. They are scaffolding for a
reasoning model -- "enclosed within <think> </think>" -- and have nothing to do with the
decision; leaving them in would put the same 400 tokens in front of every state, wasting
the token budget on a constant and teaching the model a prefix rather than a table.

Domain-11 verification found the trailing "Answer Format" block -- the
```json {"answer": ...}``` instructions for a generative model -- survived that cut and
was the same constant at the *end* of 97.7% of an earlier build's states (3,909/4,000; the
rest lost it to the `[:4000]` truncation on long tables, which made its presence
inconsistent rather than merely wasteful). That is the exact anti-pattern the paragraph
above warns about, just at the other end of the state, so it is stripped here too.

A second domain-11 verification found three more things, all measured against a later
build:

**The claim was being truncated off the end.** `state[:4000]` cuts from the end and the
`Statement` block is at the end: 55 of 2,000 states (2.75 %) arrived as a bare table with
nothing asked about it, collapsing onto 8 distinct strings, 2 of which carry *both*
labels. `fit()` drops an oversize row instead -- 89 of the mirror's 4,000 (2.23 %),
against a 2,000 quota -- because trimming the table instead would quietly make the
published label stop matching the state.

**A "yes" to everything scored 0.7525.** The mirror is 69 % entailed and its leading rows
are 75 %, and the loader read it in file order. Published TabFact is balanced by
construction, so the draw is evened here and the baseline is 0.5000.

**The row was never evaluated.** One task name is one `split_of` hash draw, and it drew
`train`, so all 2,000 questions of domain 11 sat in training and none in devreal or
testreal. The row now emits two tasks over disjoint *tables*, forced to `train` and
`testreal`. That measures unseen tables of a seen schema rather than an unseen schema,
which is what a single-source row can honestly support and is the same bargain the
lichess and chess rows take.

There is no seeded generator behind this task, despite what an earlier description of the
row said. TabFact is a published dataset with published claims and published
entailed/refuted labels; nothing here generates a claim, computes truth over cells, or
paraphrases anything, and the corpus carries exactly one render shape (a markdown pipe
table), never JSON or "damaged CSV".

A domain-11 audit (built from this code) found four
more, all in this loader:

**The balance was bought by table.** Evening the classes by taking the first k of each in
file order kept every refuted claim and the *earliest* entailed ones, and the mirror is
ordered by table: 60.5 % of train and 76.1 % of testreal examples sat in a table whose
every kept claim had the same label, and the yes-rate was 0.72 in the first half of the
file and 0.28 in the second. The label was a property of the table, not of the claim.
Every refuted claim sits in a table with at least as many entailed ones, so balancing
*within* each table costs nothing (2,430 either way) and makes the table carry no label.

**The held-out side was selected by the shingle dedup.** 345 of 1,072 testreal examples
(32 %) were dropped for sharing a ten-word window with train, and none of those windows
was a leak: the most-overlapping dropped table shared 12.7 % of its shingles, and the
windows were Wikipedia column headers ("home team score away team away team score venue
crowd date"). The test deleted every AFL, NBA-season and TV-episode table from eval. The
row is now `template_state`; exact dedup still applies.

**Same-page tables straddled the split.** The partition hashed table *text*, so two
tables of one Wikipedia page (1926 VFL season, round 3 and round 4) could land on both
sides. It now hashes the page id from the TabFact table id.

**Four states carry both labels.** TabFact's lemmatised claims collide within a table
("circinus be the constellation with the fewest object ...": entailed at index 2,
refuted at index 6), so one state had two targets. Such a state is dropped, and an exact
duplicate state is kept once.

And the row now has devreal coverage too: three tasks over disjoint pages, 2:1:1.
"""

from __future__ import annotations

import hashlib
import re
from typing import Iterator

from lod.phrasings import PhrasingBank
from lod.schema import Example, Question
from lod.corpus.repositories import raw_store as store
from lod.corpus.services.sources.real.base import RealTask

KEY = "raywithyou_tabfact"
LICENCE = "CC-BY-4.0 (TabFact)"
URL = "https://huggingface.co/datasets/Raywithyou/TabFact"
LABELS = {"entailed": 1, "refuted": 0}


_ANSWER_FORMAT_MARKER = "\n\n\nAnswer Format"
_STATEMENT = "\n\n\nStatement\n"
MAX_CHARS = 4000


def _state(row: dict) -> str | None:
    """The table and statement, without the reasoning-model scaffolding."""
    for m in row.get("prompt") or []:
        if m.get("role") != "user":
            continue
        text = str(m.get("content") or "")
        # keep from the table onwards; the preamble is identical on every row
        idx = text.find("Table")
        text = text[idx:] if idx > 0 else text
        # drop the trailing "Answer Format" json-schema block; it is the same
        # constant on every row (a generative-model instruction, not table content)
        end = text.find(_ANSWER_FORMAT_MARKER)
        if end != -1:
            text = text[:end]
        return text.strip()
    return None


def _label(row: dict) -> int | None:
    gt = (row.get("reward_model") or {}).get("ground_truth")
    if isinstance(gt, list) and gt:
        gt = gt[0]
    return LABELS.get(str(gt).strip().lower())


def fit(state: str, limit: int = MAX_CHARS) -> str | None:
    """The state, or None when table and claim together do not fit the budget.

    `state[:MAX_CHARS]` truncates from the end, and the claim is at the end: on
    an earlier build, 55 of 2,000 built states (2.75 %) had lost their `Statement` block
    entirely and were a bare table with nothing asked about it. They collapsed onto 8
    distinct strings, and because the dropped claims differed while the surviving prefix
    did not, 2 of those strings carry *both* labels -- a question with no answer and two
    targets. That is the case the corpus rule covers: a scenario whose answer cannot be
    derived from its state is dropped, not guessed.

    Trimming the table instead of the claim is not a fix either, because a claim about a
    row that was trimmed away stops being decidable and the published label silently
    stops matching the state. So an oversize row is dropped whole. It is cheap: 89 of the
    mirror's 4,000 rows (2.23 %) exceed the budget, against a 2,000-question quota.
    """
    if _STATEMENT not in state:
        return None
    return state if len(state) <= limit else None


def _table(state: str) -> str:
    """The table half of a state."""
    return state.split(_STATEMENT, 1)[0]


def _group(row: dict, state: str) -> str:
    """The Wikipedia page a row's table came from -- the key the tasks are partitioned on.

    TabFact ids are `<source>-<page>-<table>.html.csv`, and one page often yields several
    tables (a season's rounds, a roster split alphabetically). Hashing the table text put
    such siblings on both sides of the split; the page id keeps them together. A row
    without the id falls back to its table title, which is the page title.
    """
    rid = str((row.get("extra_info") or {}).get("id") or "")
    m = re.match(r"TabFact_\d+-(\d+)-\d+\.html\.csv_\d+$", rid)
    if m:
        return "page:" + m.group(1)
    t = re.search(r"Table Title: (.*)", state)
    return "title:" + (t.group(1) if t else _table(state))


def _rows() -> list[tuple[str, int, str]]:
    """Every usable (state, label, page), in file order, one per distinct state.

    A state the mirror carries twice with *different* labels is dropped: TabFact's
    lemmatised claims collide inside a table, and a state with two targets has no answer.
    """
    seen: dict[str, tuple[int, str]] = {}
    conflicted: set[str] = set()
    for r in store.load(KEY):
        state, target = _state(r), _label(r)
        if not state or target is None:
            continue
        state = fit(state)
        if state is None:
            continue
        if state in seen:
            if seen[state][0] != target:
                conflicted.add(state)
            continue
        seen[state] = (target, _group(r, state))
    return [(s, t, g) for s, (t, g) in seen.items() if s not in conflicted]


def _h(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


# (task name, split, share of pages). Row 31 emitting one task name gave it one
# `split_of` hash draw, which drew `train`, so the domain had no eval coverage at all;
# a train/testreal pair then left it without devreal, which is what early stopping
# selects on. Three tasks over *disjoint pages*, each forced to its side, measure unseen
# tables of a seen schema -- what a single-source row can honestly support, the same
# bargain the lichess and chess rows take. The row's question budget is per-row and
# split across its tasks (`generate.task_quotas`), so this costs the corpus nothing.
TASKS = (("tabfact_entailed", "train", 2),
         ("tabfact_entailed_dev_tables", "devreal", 1),
         ("tabfact_entailed_unseen_tables", "testreal", 1))


def _part(group: str) -> int:
    """Which task a page belongs to, by the TASKS weights."""
    v = _h(group) % sum(w for _, _, w in TASKS)
    for i, (_, _, w) in enumerate(TASKS):
        if v < w:
            return i
        v -= w
    raise AssertionError("unreachable")


QUESTION_ID = "d11.tabfact_entailed"
QUESTION = "Is the statement entailed by the table?"
_BANK: PhrasingBank | None = None


def _phrasing(key: str) -> str:
    global _BANK
    if _BANK is None:
        _BANK = PhrasingBank.load()
    return _BANK.pick(QUESTION_ID, QUESTION, key)


def balanced(rows: list[tuple[str, int, str]]) -> list[tuple[str, int]]:
    """(neg, pos) pairs drawn from the SAME table, in a hash order.

    Every refuted claim is kept, matched with one entailed claim of its own table chosen
    by hash rather than by file position, so each table contributes as many yeses as
    noes and a table carries no label. Every prefix of whole pairs is balanced too, which
    matters because `sample_examples` reads `load(n)` from the front.
    """
    by_table: dict[str, tuple[list[str], list[str]]] = {}
    for state, target, _ in rows:
        by_table.setdefault(_table(state), ([], []))[target].append(state)
    pairs = []
    for neg, pos in by_table.values():
        neg, pos = sorted(neg, key=_h), sorted(pos, key=_h)
        pairs.extend(zip(neg, pos))
    pairs.sort(key=lambda p: _h(p[0] + p[1]))
    return [x for n, p in pairs for x in ((n, 0), (p, 1))]


def _loader(index: int, task: str):
    def load(n: int) -> Iterator[Example]:
        mine = [r for r in _rows() if _part(r[2]) == index]
        for state, target in balanced(mine)[:n]:
            yield Example(
                task=task,
                state=state,
                questions=[Question(id="entailed",
                                    question=_phrasing(state),
                                    options=["no", "yes"], target=target)],
            )
    return load


def tasks() -> list[RealTask]:
    if not store.has(KEY):
        return []
    return [RealTask(row=31, name=name, licence=LICENCE, url=URL,
                     load=_loader(i, name), force_split=split,
                     # Wikipedia column headers repeat across pages, and a ten-word window
                     # of them made the shingle dedup delete 32 % of testreal -- every
                     # sports-results and TV-episode table -- for a template, not a leak.
                     # Exact dedup still applies; pages are disjoint across the tasks.
                     template_state=True,
                     notes="state and label read out of the mirror's RL format; "
                           "reasoning-model preamble and trailing answer-format "
                           "instructions both stripped; a row whose table and claim do "
                           "not both fit the state budget is dropped, not truncated; a "
                           "state published with both labels is dropped; balanced within "
                           "each table; Wikipedia pages disjoint across the row-31 tasks; "
                           f"{split}")
            for i, (name, split, _) in enumerate(TASKS)]
