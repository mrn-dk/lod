"""Answerability filter: reject schemas whose option set carries no meaning, and
schemas whose *examples* cannot be answered from the state they ship with.

The corpus treats the option key as an arbitrary identifier and the *description* as
what carries the meaning ("criteria as input"), and a source whose criteria its publisher
does not publish does not enter the corpus. Put together, an
option set of bare integers with no descriptions and no declared order is an identifier
with no meaning anywhere in the prompt: the question cannot be answered from the state,
only memorised from the marginal.

An earlier build shipped 38,683 such questions from the row-1 hub sweep alone -- asking for
a response-body length given an IP address, or for `PCBA-2242` given a SMILES string. They
teach the model that the right move is to predict the label prior, which is the opposite
of what a calibrated decision model should learn.

This module is the gate. It runs on built examples, after the loaders, so it sees exactly
what the model would have seen.

**The option set was only half of it.** Measured on an earlier shipped build of the
corpus, over domain 1 (rows 1, 3, 6, 8, 10, 16, 18; 70,906 questions):

- 215 tasks / 10,126 questions, **14.3 %** of the domain, give **the same answer on every
  example**. `muennighoff_natural_instructions__task_name` is 2,000 questions of
  "task001" over two options. A question-only lookup scores 1.000 on all of it, and it is
  22.1 % of the domain's `testreal`, so it was inflating the number the gates report.
- 714 questions carry a Fernet/base64 blob as their state and 252 carry nothing but an
  image URL. `tensorshield_reddit_dataset_226__datetime` asks for a calendar date given
  `Z0FBQUFBQm42...`.
- 13 tasks / 545 questions have an option set of bare **hex** hashes -- Criteo's
  anonymised categorical features, `2997ef88` against `2ba8d787`. The bare-numeral rule
  is exactly the same argument and did not cover the other base.
- 7 tasks / 294 questions have a state drawn from a closed vocabulary of a dozen short
  tokens: `nishan-chatterjee/llm_bias_detection` asks which of fourteen languages the
  text is in, given the string `numeric`.

None of these is about the option set, so `rejection` could not see them. `filter_examples`
now looks at the examples as well, which is the same "it sees exactly what the model would
have seen" argument one step further on.
"""

from __future__ import annotations

import re
from statistics import median

NUMERIC = re.compile(r"^-?\d+(?:\.\d+)?$")
# The same argument as NUMERIC, in base 16. Criteo publishes its categorical features as
# anonymised 8-hex-digit hashes, and both the option set and the state are one of those,
# so neither end of the question says anything. Six digits is the floor because `abc`,
# `dead` and `beef` are words and `fff` is a colour.
HEXISH = re.compile(r"^[0-9a-f]{6,}$", re.I)
# An address is an identifier too. `rdpahalavan_cic_ids2017__source_ip` asks which of 144
# IP addresses a record came from, and the LLM enrichment gave all 144 the same gloss --
# "A possible source IP address for the network traffic record."
IPV4 = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$")
# A state that is one base64/Fernet blob, or one bare URL, is not a state: the reader is
# holding a pointer to the content rather than the content.
OPAQUE_BLOB = re.compile(r"^[A-Za-z0-9+/=_-]{60,}$")
BARE_URL = re.compile(r"^https?://\S+$")

# option sets that are numerals but genuinely name a category, not a quantity
#
# ...or where the numeral IS the answer the state asks for. The bigbench tasks below have
# per-example options, and the question in the state is "What is 4 plus 2?", "How many
# intersection points are there?", "the length of the longest common subsequence", "the
# number of the house where the pianist lives", or an ASCII-art digit: the option "6" means
# six, it is not an identifier standing for a label. `rejection` also reads only the
# first example's options, which for a per-example task is one question's. (domain-4 audit)
NUMERIC_OK_TASKS: frozenset[str] = frozenset({
    "bb_arithmetic", "bb_cs_algorithms", "bb_elementary_math_qa", "bb_intersect_geometry",
    "bb_logic_grid_puzzle", "bb_mnist_ascii", "bb_parsinlu_qa",
    # Domain-1 audit: the published definition is the question and says what the numeral
    # means -- "choose the middle statement ... by writing "1" or "2"" -- and the state
    # prints "Middle 1:" and "Middle 2:". Positional references, not identifiers.
    "ni_task069_abductivenli_classification", "ni_task070_abductivenli_incorrect_classification",
})

# A task needs this many examples before "every answer is the same" is evidence of a
# degenerate schema rather than of a small sample.
MIN_FOR_CONSTANT = 8
# States repeating this often mean the sampled rows differ in a column nobody is being
# asked about.
REPEAT_STATE_SHARE = 0.5
SHORT_STATE_CHARS = 20
OPAQUE_STATE_SHARE = 0.8
# How often the gold option may be the *only* option spelled out in the state. Above this
# the task is a string match; `hf._scrub_label` exists for sources where that is fixable,
# and this catches the ones where it was not applied.
ANSWER_IN_STATE_SHARE = 0.8
# yes/no and true/false are excluded from the leak test: "no" occurs in ordinary prose
# and matching it says nothing about the answer being stated.
TRIVIAL_OPTION_SETS = (
    frozenset({"no", "yes"}), frozenset({"false", "true"}), frozenset({"0", "1"}),
    frozenset({"n", "y"}), frozenset({"negative", "positive"}),
)


def numeric_share(options: list[str]) -> float:
    if not options:
        return 1.0
    return sum(1 for o in options if NUMERIC.match(str(o).strip())) / len(options)


def opaque_share(options: list[str]) -> float:
    """Share of options that are an identifier in *any* base, not a word."""
    if not options:
        return 1.0
    return sum(1 for o in options
               if NUMERIC.match(str(o).strip()) or HEXISH.match(str(o).strip())
               or IPV4.match(str(o).strip())) / len(options)


def rejection(task: str, options: list[str], described: bool) -> str | None:
    """Why this schema must not enter the corpus, or None to keep it.

    `described` says whether the task's options carry descriptions by the time the
    corpus is written -- published definitions or LLM-written ones. A numeric option set
    *with* descriptions is a legitimate `score` ladder and is kept.
    """
    opts = [str(o) for o in options]
    if len(opts) < 2:
        return "fewer than two options"
    if task in NUMERIC_OK_TASKS or described:
        return None
    share = opaque_share(opts)
    if share >= 0.5:
        kind = "numerals" if numeric_share(opts) >= 0.5 else "hex identifiers"
        return f"{share:.0%} of options are bare {kind} with no description"
    return None


def _gold(question) -> str:
    """The option this question is labelled with, or "" where there is no hard target."""
    target = getattr(question, "target", None)
    if target is None or isinstance(target, (list, tuple)):
        return ""
    index = int(target)
    options = list(question.options)
    return str(options[index]) if 0 <= index < len(options) else ""


def _spelled_out(option: str, state: str) -> bool:
    option = str(option).strip().lower()
    if not option:
        return False
    return re.search(r"(?<![A-Za-z0-9])" + re.escape(option) + r"(?![A-Za-z0-9])",
                     state) is not None


def example_rejection(examples) -> str | None:
    """Why this task's *examples* cannot be answered, or None to keep them.

    Every test here is a property of the task as a whole, not of one row, because that is
    the unit the split hash moves: a task whose answer never varies is degenerate in
    whichever split it lands in.
    """
    items = [e for e in examples if getattr(e, "questions", None)]
    if not items:
        return None
    states = [str(e.state or "") for e in items]
    n = len(states)

    golds = {_gold(q) for e in items for q in e.questions}
    golds.discard("")
    if golds and len(golds) == 1 and n >= MIN_FOR_CONSTANT:
        only = next(iter(golds))
        return f"every one of {n} examples answers {only!r}"

    stripped = [s.strip() for s in states]
    opaque = sum(1 for s in stripped if OPAQUE_BLOB.fullmatch(s) or BARE_URL.fullmatch(s))
    if opaque >= OPAQUE_STATE_SHARE * n:
        return f"{opaque / n:.0%} of states are an opaque blob or a bare URL"

    distinct = len(set(stripped))
    if n >= MIN_FOR_CONSTANT and distinct <= REPEAT_STATE_SHARE * n:
        if median(len(s) for s in stripped) < SHORT_STATE_CHARS:
            return (f"states are {distinct} short tokens repeated over {n} examples")

    options = [str(o).strip().lower() for o in items[0].questions[0].options]
    if frozenset(options) not in TRIVIAL_OPTION_SETS:
        leaked = 0
        asked = 0
        for e in items:
            low = str(e.state or "").lower()
            for q in e.questions:
                gold = _gold(q)
                if not gold:
                    continue
                asked += 1
                present = {str(o).strip().lower() for o in q.options
                           if _spelled_out(o, low)}
                if present == {gold.strip().lower()}:
                    leaked += 1
        if asked and leaked >= ANSWER_IN_STATE_SHARE * asked:
            return (f"the state spells out its own answer, and no other option, "
                    f"on {leaked / asked:.0%} of examples")
    return None


def filter_examples(examples, task: str, described: bool):
    """Drop every example of a task whose schema is rejected. -> (kept, reason|None)."""
    items = list(examples)
    if not items:
        return items, None
    options = items[0].questions[0].options if items[0].questions else []
    reason = rejection(task, list(options), described)
    if reason is None:
        reason = example_rejection(items)
    return ([], reason) if reason else (items, None)
