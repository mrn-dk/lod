"""Shared plumbing for the real-task sources.

A **real schema** is a (question, option set) whose labels come from people or a real
system on real text or records. Nothing under `lod/corpus/services/sources/` other than this package
counts as real, and the distinction is enforced here rather than trusted: every task
carries `real`, and the acceptance report counts real and synthetic questions separately.

Real sources need no change to the data model: `Question` already takes a soft target
as a probability vector and a free `meta` dict, so `meta.p` needs no new field.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator

from lod.schema import Example, Question

SPLITS = ("train", "devreal", "testreal")
# hashed by task name, seed 0, 70/15/15
SPLIT_WEIGHTS = ((("train"), 70), (("devreal"), 15), (("testreal"), 15))
# whole families reserved for testreal, never in train or devreal
RESERVED_TESTREAL_ROWS = (11, 25)

SHAPES = ("bool", "enum3-5", "enum6-20", "enum>20", "ordinal", "soft")


@dataclass
class RealTask:
    """One task: a schema, where its rows come from, and what shape its labels are.

    `ordinal` is tagged per task, and only where the option set has a
    declared total order (ratings, severities, priorities, none/low/high,
    negative/neutral/positive). It may overlap with `soft`.
    """

    row: int                      # source-table row, 1-51 (see `__post_init__`)
    name: str                     # task name; the split hash is taken over this
    licence: str
    url: str
    load: Callable[[int], Iterable[Example]]  # load(n) -> examples, already questioned
    ordinal: bool = False
    soft: bool = False
    meta_p: bool = False
    per_example_options: bool = False
    real: bool = True
    # A family that must land in one split whatever the name hashes to --
    # a rule family held out of training, or one that must be trained on.
    force_split: str | None = None
    # This task's state is a *record* rendered from a fixed template, not collected prose,
    # so the shingle half of `dedup_against_train` reads its template as a leak and
    # deletes whichever states are commonest. `shingle_exempt` in `services/generate.py`
    # already does this for the code-labelled rows; a real source can be template-shaped
    # too (row 27's gridpoint record, row 26's SPF survey record), and the exact-match
    # test still applies. Measured on an earlier build: the shingle test dropped 1,052 of
    # 2,000 `gefs_pop_day10` examples and 1,156 of 1,290 `spf_recession_h4` examples, and it
    # dropped them non-randomly -- every dry gridpoint and every forecaster who had a
    # realised-GDP history. See the note on `dedup_against_train`.
    template_state: bool = False
    # A floor on this task's share of its row's budget. `task_quotas` splits a row's
    # budget evenly over its tasks, which starves a task that pools a whole row: row 5
    # is ~1,900 one-repo label tasks plus `gh_is_bug`, which drew from all of them and
    # was handed the ~19 questions of one small repo (domain-8 audit). Capped by
    # `--sample-per-task` like every other quota.
    min_quota: int = 0
    # The unit a held-out split is taken over. Two tasks in one family share a
    # structure (a rule combination, a page layout, a dataset's record pool), so a dev
    # task whose family also trains measures recall of that structure, not transfer.
    # None means the task name with a trailing _eval/_dev/_test/_train stripped, which
    # is what every generator's twin convention already implies. See `family_of`.
    family: str | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        # 1-32 are the original source table; 33-36 are the first additions
        # (33 rule application, 34 grounding, 35 randomised-attribute entities,
        # 36 accessibility trees and the decisions a navigator makes on them).
        # 37-41 are the domain pass (domains 15-19): 37 control and
        # sensor, 38 tool-call and action gating, 39 entity resolution, 40 code and
        # diffs, 41 scheduling and allocation. 42 is multi-hop and ranking (domain 20);
        # 43 is chess positions as facts, which joins domain 4.
        # 44-46 are the classification pass: 44 English sentiment (domain 3),
        # 45 English topic/intent (domain 1), 46 multilingual classification (domain 21).
        # 47-49 are the generalisation pass: 47 compositional rules (domain 2), 48 exact
        # posteriors (domain 22), 49 many-option matching and linking (domain 23).
        # 50-51 are the release round: 50 broad real-language decisions (domain 24),
        # 51 long-context reading (domain 25).
        if not 1 <= self.row <= 51:
            raise ValueError(f"{self.name}: row {self.row} outside the source table")
        if self.force_split not in (None, "train", "devreal", "testreal"):
            raise ValueError(f"{self.name}: force_split {self.force_split!r}")
        if self.real and not self.licence:
            raise ValueError(f"{self.name}: a real task without a licence cannot enter train")


_TWIN_SUFFIX = re.compile(r"_(eval|dev|test|train|probe)$")


def family_of(task: "RealTask") -> str:
    """The held-out unit of a task: its declared `family`, else its name without the
    twin suffix. `services.splits.devood` keeps only dev tasks whose family never
    trains, which is the property the dev split needs to steer checkpoints and T."""
    if task.family:
        return task.family
    return _TWIN_SUFFIX.sub("", task.name)


def split_of(task_name: str, row: int, seed: int = 0) -> str:
    """Deterministic 70/15/15 by task name, with the reserved families forced.

    Hashing the *task name* rather than the row is what makes the split a whole-task
    holdout: every question of a task lands in exactly one split, so `testreal` measures
    transfer to unseen tasks rather than unseen rows of seen tasks.
    """
    if row in RESERVED_TESTREAL_ROWS:
        return "testreal"
    h = hashlib.sha256(f"{seed}:{task_name}".encode()).digest()
    v = int.from_bytes(h[:8], "big") % 100
    if v < 70:
        return "train"
    return "devreal" if v < 85 else "testreal"


def shape_of(q: Question, ordinal: bool = False) -> set[str]:
    """A question's shape buckets. Ordinal and soft may overlap the others."""
    out: set[str] = set()
    n = len(q.options)
    if isinstance(q.target, list):
        out.add("soft")
    if ordinal:
        out.add("ordinal")
    if n == 2:
        out.add("bool")
    elif n <= 5:
        out.add("enum3-5")
    elif n <= 20:
        out.add("enum6-20")
    else:
        out.add("enum>20")
    return out


def disjoint_slice(rows: list, task_index: int, n_tasks: int) -> list:
    """The slice of a shared record pool belonging to one task of a multi-task source.

    Sources like NVD, Jira and Civil Comments ask several questions about the *same*
    records. Splitting by task name then routes those tasks into different splits while
    they still share states, and the devreal/testreal dedup -- doing exactly its job --
    deletes the entire eval side. Measured on a three-row build: 300 devreal and 600
    testreal examples dropped, every one of them.

    Partitioning the pool so each task draws from disjoint records removes the conflict at
    the source: no state is ever shared between two tasks, so the split is free to send
    them wherever the hash says. The cost is that each task sees pool_size / n_tasks
    records, which is a reason to download a larger pool, not a reason to share states.
    """
    if n_tasks <= 1:
        return rows
    n = len(rows)
    lo = (n * task_index) // n_tasks
    hi = (n * (task_index + 1)) // n_tasks
    return rows[lo:hi]


_WORD = re.compile(r"\w+")


def shingles(text: str, k: int = 10) -> set[str]:
    """10-word shingles, for the devreal/testreal-against-train dedup."""
    words = _WORD.findall(text.lower())
    if len(words) < k:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}


def dedup_against_train(
    train: list[Example], others: dict[str, list[Example]], k: int = 10,
    shingle_exempt: set[str] | None = None,
) -> dict[str, int]:
    """Drop any devreal/testreal example whose state matches a train state.

    Exact match first (cheap, catches re-uploads of the same corpus), then 10-word
    shingle overlap (catches the same text reached by a different route). Returns the
    counts, which the build records in its metadata.

    `shingle_exempt` names tasks that get the exact test but not the shingle test. The
    shingle test asks "did this text appear in training"; for a template-generated source
    it answers "yes" for every row, because two independently drawn shipment records both
    contain "carrier kestrel; service level express; destination zone zone_b" and that is
    ten words. Measured on an earlier build: it dropped two thirds of the generated
    eval rows, 1,538 per rule family down to about 470, for sharing a template phrase
    rather than a fact. Exact match still applies, and it is the test that means anything
    when the state was drawn from a distribution rather than collected.

    A *real* source can be template-shaped too, and when it is, this test does not merely
    thin the eval split -- it selects it. Row 27 renders every gridpoint as the same JSON
    record, so a ten-word window falling inside `ensemble_precip_mm` is identical for
    every dry gridpoint; measured on an earlier build, 1,052 of 2,000 `gefs_pop_day10`
    examples were dropped and the dry share of what survived fell from 54.1 % to 20.6 %,
    taking the split's mean `meta.p` from 0.1595 to 0.2958 against a training set left at 0.16. Row
    26's SPF record is worse: 1,156 of 1,290 `spf_recession_h4` examples were dropped, and
    the 134 survivors were *exactly* the 134 whose `recent_real_gdp_growth` was empty or
    one quarter long, because a forecaster's GDP history is a function of the survey
    quarter and so is shared with the trained horizons. The `probability_recovery` gate was
    then asked to recover a probability from a state the test had stripped of its signal.
    `RealTask.template_state` marks such a task so `services/generate.py` can exempt it.
    """
    exempt = shingle_exempt or set()
    exact = {e.state for e in train}
    train_shingles: set[str] = set()
    for e in train:
        train_shingles |= shingles(e.state, k)

    dropped: dict[str, int] = {}
    for split, examples in others.items():
        kept, n_exact, n_shingle = [], 0, 0
        for e in examples:
            if e.state in exact:
                n_exact += 1
                continue
            if e.task not in exempt and shingles(e.state, k) & train_shingles:
                n_shingle += 1
                continue
            kept.append(e)
        examples[:] = kept
        dropped[f"{split}_exact"] = n_exact
        dropped[f"{split}_shingle"] = n_shingle
    return dropped


@dataclass
class Accounting:
    """The measured value of every mixing rule and acceptance criterion.

    Measured, not asserted: `print_rules` prints each rule beside its measured value,
    and a rule that is only checked in a conditional is a rule nobody can audit.
    """

    questions_by_row: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    schemas_by_row: dict[int, set] = field(default_factory=lambda: defaultdict(set))
    shape_schemas: dict[str, set] = field(default_factory=lambda: defaultdict(set))
    n_questions: int = 0
    n_real_questions: int = 0
    n_soft: int = 0
    n_ordinal: int = 0
    n_meta_p: int = 0
    n_per_example: int = 0

    def add(self, task: RealTask, example: Example) -> None:
        for q in example.questions:
            self.n_questions += 1
            self.questions_by_row[task.row] += 1
            key = (q.question, tuple(q.options) if len(q.options) <= 40 else ("free", len(q.options)))
            self.schemas_by_row[task.row].add(key)
            if task.real:
                self.n_real_questions += 1
            for sh in shape_of(q, task.ordinal):
                self.shape_schemas[sh].add(key)
            if isinstance(q.target, list):
                self.n_soft += 1
            if task.ordinal:
                self.n_ordinal += 1
            if (q.meta or {}).get("p") is not None:
                self.n_meta_p += 1
            if task.per_example_options:
                self.n_per_example += 1

    def fraction(self, n: int) -> float:
        return n / self.n_questions if self.n_questions else 0.0

    def row_cap(self, row: int) -> float:
        """Rows 1-3 may take 15 % of train questions, every other row 10 %.

        Rows 1-3 are the Hub sweep, tasksource and Super-NaturalInstructions -- the three
        that supply schema *count*, which is what the real-schema target (>= 1,500 in
        train) is short of, so they are allowed to weigh more heavily than a single
        hand-specified source.
        """
        return 0.15 if row in (1, 2, 3) else 0.10

    def worst_row(self) -> tuple[int, float, float]:
        """The row closest to breaching its own cap: (row, measured share, its cap)."""
        worst = (0, 0.0, 0.10)
        for row, n in self.questions_by_row.items():
            share, cap = self.fraction(n), self.row_cap(row)
            if share - cap > worst[1] - worst[2]:
                worst = (row, share, cap)
        return worst

    def rules(self) -> list[tuple[str, float, float, bool]]:
        """(rule, measured, required, ok). Shares are of train questions."""
        row, share, cap = self.worst_row()
        return [
            (f"max share from one row (worst: row {row})", share, cap,
             share <= cap + 1e-9),
            ("soft-target questions", self.fraction(self.n_soft), 0.10,
             self.fraction(self.n_soft) >= 0.10),
            ("ordinal questions", self.fraction(self.n_ordinal), 0.10,
             self.fraction(self.n_ordinal) >= 0.10),
            ("meta.p questions", self.fraction(self.n_meta_p), 0.03,
             self.fraction(self.n_meta_p) >= 0.03),
            ("per-example-option questions", self.fraction(self.n_per_example), 0.05,
             self.fraction(self.n_per_example) >= 0.05),
            ("real questions", self.fraction(self.n_real_questions), 0.80,
             self.fraction(self.n_real_questions) >= 0.80),
        ]

    def print_rules(self) -> bool:
        print(f"\n{'mixing rule':38} {'measured':>10} {'required':>10}   ok")
        ok_all = True
        for name, measured, required, ok in self.rules():
            comp = "<=" if "max share" in name else ">="
            print(f"{name:38} {measured*100:9.2f}% {comp}{required*100:8.1f}%   "
                  f"{'yes' if ok else 'NO'}")
            ok_all &= ok
        return ok_all
