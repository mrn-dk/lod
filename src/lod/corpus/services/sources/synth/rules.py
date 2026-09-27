"""Row 33 — apply a rule written in the criteria to evidence in the state.

Rebuilt from thirteen hand-written families into a **grammar** (`grammar.py`) after an
earlier model made the failure mode legible: trained families 0.978, held-out families
0.598, and the two worst held-out results were the two that were not simply a trained
family rearranged.

    range_outside   0.982   a trained family negated. Transferred.
    membership      0.616   a trained family generalised. Barely.
    disjunction     0.527   two trained families joined by `or`. Chance.
    ranking         0.267   a different *shape*: compare across records. Chance.

Three changes follow from that, and they are the whole of this module's design.

**Every primitive is trained; whole combinations are held out.** `threshold`, `range`,
`equals`, `member`, `count`, `age`, `converted`, `relative`, `across` and `across_mean`
each appear alone and in some combinations. The `rule_transfer` gate reads combinations the
model has never seen, built only from primitives it has. That asks the transfer question
worth asking -- not "have you met `or`" but "can you apply `or` to two things you know".

**Comparing across records is a trained primitive, not a held-out curiosity.** It was the
worst result in that earlier model precisely because its shape appeared nowhere in training.

**Half the rules never name their field.** `indirect` mode names a field only by what it
records; `relative` compares two fields of one record without naming either. Every rule that
model trained on quoted its key, so locating the evidence was never part of the task -- and
on the reefer probe, whose rule says "outside its required temperature range", it answered
confidently and wrongly. That probe flipping to a correct answer, or to a low-confidence
one, is the sign that the model is reading rules rather than matching them.

Unchanged from the earlier generator, because they worked: soft targets within 3 % of a
threshold; a rule naming a field the record does not carry is **not-in-context**, with no
score target at all; both JSON and prose renderings of the same record; contrastive pairs
with both halves in one sequence.

**What a later audit found, and what `rule_transfer` is actually measuring.** The gate
reported trained 0.9601 against held-out 0.8473 -- an 0.1128 drop against an 0.10 bar --
and the comparison is not matched. Ten of the twenty-four trained families are a *single
primitive* and no held-out family is, and `select` is a different shape in a different
format rather than a combination of anything. Compare combinations with combinations and
the drop is 0.0587; drop `select` from the held-out mean and it is 0.0616. Both clear the
bar. Scoring each combination against the product of its own primitives' accuracies, the
mean residual is -0.0025 over the twelve trained combinations and -0.0064 over five of the
six held-out ones: composition itself is very nearly free, on families never trained.

What is left after that arithmetic is one real result and one confound, and averaging
them into a single number hid both.

    and(across,relative)  0.6223  residual -0.2833   real: cross-record conjunction
    select                0.5400  not a combination  confound: a new shape AND a new
                                                     format in the same family

`and(across,relative)` is genuine. `relative` reads two fields of one record and
`across` reads one field of every record, and nothing trained combines those two things
-- the trained cross-record combinations, `or(across,threshold)` and
`and(across_mean,member)`, are also the two worst trained families (0.839, 0.891) and
carry the only other negative residuals. That is the earlier model's `ranking` failure
again at the level of a *difficulty class* rather than a shape, so
`and(across_mean,relative)` is now trained, for the same reason `across` itself is.

`select` is not. Its rule sits in the question text and its option descriptions name only
a record key, so the token each option is scored at carries none of the rule -- unlike
every other family in this module, where the option's own span *is* the rule. Its 0.54
therefore mixes "a new shape" with "a format never trained on" and cannot be read as a
transfer result at all. It stays held out, because a shape control is worth having, and
it is reported beside the transfer number rather than inside it:
`HELDOUT_COMBO_FAMILIES` is the transfer measurement, `HELDOUT_SHAPE_FAMILIES` is the
control.

Two labelling defects the same audit turned up, both now fixed: `Age.negated` flipped
"more than N days" to "less than N days", which is not its complement at exactly N days
(10 rows in the audited build whose `no` criteria described something untrue of the
record), and `_select_example` bounded only the winning record, so 3.7 % of `select`
questions had a second record that literally satisfies the stated rule.
"""

from __future__ import annotations

import hashlib
import itertools
import json as _json
import random
from typing import Iterator

from lod.phrasings import PhrasingBank
from lod.schema import Example, Question
from lod.corpus.services.sources.real.base import RealTask
from lod.corpus.services.sources.synth.grammar import (
    Age,
    AboveMean,
    AcrossExtreme,
    And,
    Converted,
    CountAtLeast,
    Ctx,
    Equals,
    Member,
    Missing,
    Node,
    Not,
    Or,
    Range,
    Relative,
    Threshold,
    _days_before,
    _fmt,
    _round,
    soft,
)
from lod.corpus.services.sources.synth.records import (
    DOMAINS,
    Domain,
    as_prose,
    field_ref,
    make_record,
    render,
    render_pair,
)

LICENCE = "generated by lod/corpus/services/sources/synth/rules.py; no third-party data"
URL = "https://github.com/(this repo)/blob/main/src/lod/corpus/services/sources/synth/rules.py"
ROW = 33

P_MISSING = 0.12          # share of rows whose rule names a field the record lacks
P_INDIRECT = 0.50         # share of rules that name no field, only what it records

# ---- what is trained, and what rule_transfer reads ----------------------------------
# A spec is a primitive name, or a tuple (op, *operands). Every primitive below appears
# in TRAINED, alone. HELDOUT contains no primitive that TRAINED does not.

PRIMITIVES = ("threshold", "range", "equals", "member", "count", "age", "converted",
              "relative", "across", "across_mean")

TRAINED_COMBOS = (
    ("and", "threshold", "equals"),
    ("or", "threshold", "member"),
    ("and", "range", "count"),
    ("not", "member"),
    ("and", "age", "equals"),
    ("or", "converted", "threshold"),
    ("and", "relative", "equals"),
    ("or", "across", "threshold"),
    ("and", "across_mean", "member"),
    ("not", "relative"),
    ("and", "threshold", "count"),
    ("or", "equals", "age"),
    # Cross-record composition needs more than one trained representative. On
    # an earlier model the two that existed were the two worst trained combinations
    # (`or(across,threshold)` 0.839, `and(across_mean,member)` 0.891) and the one
    # held-out combination in the same class, `and(across,relative)`, scored 0.622 --
    # a composition residual of -0.28 where every other held-out combination sits
    # within +/- 0.06 of the product of its own primitives. That is the earlier `ranking`
    # mistake in miniature: a *difficulty class* with no trained representative reads
    # as a composition failure. `relative` reads two fields of one record, which is the
    # part `or(across,threshold)` and `and(across_mean,member)` never exercise.
    ("and", "across_mean", "relative"),
)

# Never trained on. Each is a combination of primitives that are, so a model that has
# learned to apply a rule should manage them and a model that has memorised twelve
# combinations should not.
HELDOUT_COMBOS = (
    ("or", "range", "age"),
    ("and", "member", "count"),
    ("and", "across", "relative"),
    # not(or(...)), not not(and(threshold,equals)): the latter's criteria are trained
    # `and_threshold_equals` with the answers swapped, so it held out nothing -- an earlier
    # model scored it 0.99 against 0.989 for the trained family.
    ("not", ("or", "threshold", "equals")),
    ("or", "converted", "member"),
    ("and", "range", "relative"),
)

# Held out of train *and* test: the dev split's own unseen combinations, which is what
# selects checkpoints and fits the temperature. `_eval` twins are new draws of trained
# combinations, so a dev split made of them measured recall of a structure while test
# measured transfer to an unseen one, and the two disagreed about which checkpoint was
# best. These follow the `rule_transfer` rule exactly as `HELDOUT_COMBOS` does -- every
# leaf is a trained primitive, no combination is a trained or a test one under another
# name, and none is a trained combination negated (the `not(and(threshold,equals))` trap
# above) -- and one of them is cross-record, the class test's `and(across,relative)`
# represents.
DEV_COMBOS = (
    ("or", "range", "count"),
    ("and", "age", "member"),
    ("and", "converted", "equals"),
    ("and", "across", "equals"),
    ("not", ("and", "range", "member")),
    ("or", "relative", "count"),
)

# Choice forms. `select` is held out and is the only held-out form with a different
# *shape* rather than a new combination -- it is the control on the control.
CHOICE_TRAINED = ("band", "routing")
CHOICE_HELDOUT = ("select",)


def slug(spec) -> str:
    """A task-name-safe rendering of a spec. `("not", ("and","threshold","equals"))`
    becomes `not_and_threshold_equals`, which is also its signature."""
    if isinstance(spec, str):
        return spec
    return "_".join(slug(x) for x in spec)


def sig_of(spec) -> str:
    """The canonical signature a built node will report, for cross-checking the split."""
    if isinstance(spec, str):
        return spec
    op, *rest = spec
    return f"{op}({','.join(sig_of(x) for x in rest)})"


TRAINED_SPECS = tuple(PRIMITIVES) + TRAINED_COMBOS
TRAINED_FAMILIES = tuple(slug(s) for s in TRAINED_SPECS) + CHOICE_TRAINED
HELDOUT_FAMILIES = tuple(slug(s) for s in HELDOUT_COMBOS) + CHOICE_HELDOUT
FAMILIES = TRAINED_FAMILIES + HELDOUT_FAMILIES
# Not in `FAMILIES`: that tuple is what `rule_transfer` and the probe stream read, and a
# dev family must stay out of every test grouping.
DEV_FAMILIES = tuple(slug(s) for s in DEV_COMBOS)
DEV_COMBO_FAMILIES = DEV_FAMILIES
CROSS_RECORD = ("across", "across_mean")

# ---- the groups rule_transfer should actually compare -------------------------------
# `acc_trained_families` is a mean over `TRAINED_FAMILIES` and `acc_heldout_families` a
# mean over `HELDOUT_FAMILIES`, and those two sets do not have the same shape, so their
# difference is not a transfer measurement. On an earlier model:
#
#     trained, all 24 families      0.9601   (10 of them are a single primitive)
#     trained, combinations only    0.9573
#     held out, all 7 families      0.8473   -> drop 0.1128, gate FAILS
#     held out, combinations only   0.8986   -> drop 0.0587 against trained combos
#     `select` alone                0.5400
#
# Two things are being added together there. Ten of the twenty-four trained families are
# a *single primitive*; no held-out family is, so 42 % of the trained mean has no
# counterpart on the other side. And `select` is not a combination of primitives at all
# -- it is a different shape (the answer is a record) carried in a different format (the
# rule sits in the question text, and the option descriptions name only a record key, so
# the position each option is scored at contains none of the rule). It is the control on
# the control and it belongs beside the transfer number, not inside it.
#
# The matched comparison is `TRAINED_COMBO_FAMILIES` against `HELDOUT_COMBO_FAMILIES`:
# same depth, same leaf count, both drawn from the same ten trained primitives.
TRAINED_COMBO_FAMILIES = tuple(slug(s) for s in TRAINED_COMBOS)
HELDOUT_COMBO_FAMILIES = tuple(slug(s) for s in HELDOUT_COMBOS)
# A different shape rather than a new combination. Report it, never average it in.
HELDOUT_SHAPE_FAMILIES = CHOICE_HELDOUT
PRIMITIVE_FAMILIES = tuple(PRIMITIVES)


def needs_records(spec) -> bool:
    if isinstance(spec, str):
        return spec in CROSS_RECORD
    return any(needs_records(x) for x in spec[1:])


# ---- building a node from a spec ----------------------------------------------------

def _nums(d: Domain, with_alt: bool = False):
    return [f for f in d.nums if (f.alt if with_alt else True)]


def _primitive(name: str, d: Domain, rng: random.Random) -> Node:
    if name == "threshold":
        f = rng.choice(_nums(d))
        span = f.hi - f.lo
        thr = _round(f, rng.uniform(f.lo + 0.15 * span, f.hi - 0.15 * span))
        return Threshold(f, thr, rng.random() < 0.5)
    if name == "range":
        f = rng.choice(_nums(d))
        span = f.hi - f.lo
        lo = rng.uniform(f.lo, f.hi - 0.3 * span)
        hi = _round(f, lo + rng.uniform(0.15 * span, 0.4 * span))
        return Range(f, _round(f, lo), hi, rng.random() < 0.5)
    if name == "equals":
        e = rng.choice(list(d.enums))
        return Equals(e, rng.choice(e.values))
    if name == "member":
        e = rng.choice(list(d.enums))
        k = rng.randint(2, max(2, len(e.values) - 1))
        return Member(e, tuple(sorted(rng.sample(list(e.values), k))))
    if name == "count":
        it = rng.choice(list(d.items))
        return CountAtLeast(it, rng.randint(1, 4))
    if name == "age":
        s = rng.choice(list(d.stamps))
        return Age(s, rng.choice((7, 14, 30, 60, 90, 180, 365)), rng.random() < 0.5)
    if name == "converted":
        f = rng.choice(_nums(d, with_alt=True))
        span = f.hi - f.lo
        thr = rng.uniform(f.lo + 0.15 * span, f.hi - 0.15 * span)
        return Converted(f, round(thr * f.alt[1], 3), rng.random() < 0.5)
    if name == "relative":
        a_name, b_name, gloss = rng.choice(list(d.pairs))
        by = {f.name: f for f in d.nums}
        return Relative(by[a_name], by[b_name], gloss, rng.random() < 0.5)
    if name == "across":
        return AcrossExtreme(rng.choice(_nums(d)), rng.random() < 0.5)
    if name == "across_mean":
        return AboveMean(rng.choice(_nums(d)), rng.random() < 0.5)
    raise ValueError(f"unknown primitive {name!r}")


def build(spec, d: Domain, rng: random.Random) -> Node:
    if isinstance(spec, str):
        return _primitive(spec, d, rng)
    op, *rest = spec
    kids = [build(x, d, rng) for x in rest]
    if op == "and":
        return And(*kids)
    if op == "or":
        return Or(*kids)
    if op == "not":
        return Not(*kids)
    raise ValueError(f"unknown combinator {op!r}")


def build_distinct(spec, d: Domain, rng: random.Random, tries: int = 24) -> Node:
    """A node whose operands read different fields, where it can manage it.

    `and(threshold, count)` over the same field is not wrong, but it is a degenerate
    question, and drawing operands independently produces one often enough to matter.
    """
    best = build(spec, d, rng)
    for _ in range(tries):
        if len(set(best.reads)) == len(best.reads):
            return best
        best = build(spec, d, rng)
    return best


BALANCE_TRIES = 16


# ---- verdict-first sampling ---------------------------------------------------------
# `build_balanced` used to redraw a whole rule until its verdict came out as the one owed.
# That balanced the *label* and left the answer readable from part of the rule
# (measured by a shortcut audit, n = 1,200 per family):
#
#   * one clause decided most combinations. `not(or(threshold,equals))` is `no` whenever
#     either clause holds, and `equals` holds 1 time in |values|, so nearly every `no`
#     was the threshold's doing: "answer = the threshold clause" scored 0.91 and the
#     equality clause flipped the answer on only 59 % of questions. `and(threshold,
#     equals)` 0.93 on its second clause, `or(converted,member)` 0.90, `and(age,equals)`
#     0.94. A model that reads one clause of a two-clause rule and ignores the other was
#     right on nine held-out questions in ten, which is why every model scored 0.96-1.00
#     on held-out combinations that are meant to test composition.
#   * a primitive was readable without its constant. Redrawn thresholds sat anywhere in
#     the middle 70 % of the range, so "the value is on the rule's side of the field's
#     midpoint" answered `threshold` 0.81, `converted` 0.84, `across_mean` 0.86; the
#     direction word alone answered `age` 0.89 ("more than N days" with N <= 365 against
#     stamps up to 900 days old) and `range` 0.72 (a narrow band rarely contains a value,
#     so "outside" meant `yes`); list length answered `count` 0.90.
#
# So the verdict comes first, clause by clause. A combination draws a *pattern* of clause
# truths giving the answer it owes, weighted so that each clause is the deciding one
# equally often (a pattern where no single flip changes the answer -- both clauses of an
# `and` false -- is kept, rarely, so it is not unseen). Each clause then places its
# constant just either side of the record's own value (`_leaf`): the rule's side of the
# value is drawn independently of where the value sits, so neither the value nor the
# constant on its own says which side it is.
P_NONCRITICAL = 0.1       # weight of a clause pattern no single clause flip would change
NEAR = 0.15               # a constant sits within this fraction of the range of the value
P_SOFT = 0.08             # share of constants placed inside the soft band around the value
AGE_DAYS = (7, 14) + tuple(range(30, 1081, 30))


def _spec_leaves(spec) -> list[str]:
    if isinstance(spec, str):
        return [spec]
    return [x for s in spec[1:] for x in _spec_leaves(s)]


def _spec_eval(spec, it) -> bool:
    if isinstance(spec, str):
        return next(it)
    op, *rest = spec
    vals = [_spec_eval(s, it) for s in rest]
    return all(vals) if op == "and" else any(vals) if op == "or" else not vals[0]


def _patterns(spec, want: bool) -> list[tuple[tuple[bool, ...], float]]:
    """Every assignment of truths to the clauses of `spec` that answers `want`, weighted
    1 when flipping some single clause would change the answer and `P_NONCRITICAL`
    otherwise."""
    n = len(_spec_leaves(spec))
    out = []
    for pat in itertools.product((False, True), repeat=n):
        if _spec_eval(spec, iter(pat)) is not want:
            continue
        critical = any(_spec_eval(spec, iter(pat[:i] + (not pat[i],) + pat[i + 1:]))
                       is not want for i in range(n))
        out.append((pat, 1.0 if critical else P_NONCRITICAL))
    return out


def _decisive(spec, pat: tuple[bool, ...]) -> list[int]:
    """The clauses whose truth alone, flipped, would flip the answer."""
    v = _spec_eval(spec, iter(pat))
    return [i for i in range(len(pat))
            if _spec_eval(spec, iter(pat[:i] + (not pat[i],) + pat[i + 1:])) is not v]


def _node_leaves(node: Node) -> list[Node]:
    if isinstance(node, (And, Or)):
        return _node_leaves(node.a) + _node_leaves(node.b)
    if isinstance(node, Not):
        return _node_leaves(node.a)
    return [node]


def _count_decisive(spec, node: Node, ctx: Ctx, bal: list[int]) -> None:
    """Tally, in `bal[2:]`, which clause decided the answer `node` gives."""
    try:
        pat = tuple(lf.p(ctx) >= 0.5 for lf in _node_leaves(node))
    except Missing:
        return
    for i in _decisive(spec, pat):
        bal[2 + i] += 1


def _assemble(spec, it) -> Node:
    if isinstance(spec, str):
        return next(it)
    op, *rest = spec
    kids = [_assemble(s, it) for s in rest]
    return {"and": And, "or": Or, "not": Not}[op](*kids)


def _pick(fields, avoid):
    free = [f for f in fields if f.name not in avoid]
    return free or list(fields)


def _gap(span: float, rng: random.Random) -> float:
    """How far a constant sits from the value it is compared with. Mostly outside the
    soft band, so the target is a confident one; `P_SOFT` of the time inside it, which
    is about the share of soft targets the independently drawn rules used to give."""
    if rng.random() < P_SOFT:
        return rng.uniform(0.005, 0.035) * span
    return rng.uniform(0.035, NEAR) * span


def _near(f, v: float, below: bool, rng: random.Random) -> float | None:
    """A constant below (or above) `v` by `_gap`, inside the field's range."""
    span = f.hi - f.lo
    room = (v - f.lo) if below else (f.hi - v)
    d = _gap(span, rng)
    if d >= room:
        if room <= 0.01 * span:
            return None
        d = rng.uniform((0.035 if room > 0.045 * span else 0.005) * span, room)
    return v - d if below else v + d


def _leaf(name: str, d: Domain, rng: random.Random, ctx: Ctx, t: bool,
          avoid: set) -> Node | None:
    """A primitive of kind `name` that is `t` of `ctx`, or None when this draw cannot be
    (the caller draws again). Every choice that does not decide `t` -- field, direction,
    width, set size -- is drawn as `_primitive` draws it."""
    rec = ctx.focus
    if name in ("threshold", "converted"):
        f = rng.choice(_pick(_nums(d, with_alt=name == "converted"), avoid))
        greater = rng.random() < 0.5
        thr = _near(f, rec[f.name], t == greater, rng)
        if thr is None:
            return None
        if name == "converted":
            return Converted(f, round(thr * f.alt[1], 3), greater)
        return Threshold(f, _round(f, thr), greater)
    if name == "range":
        f = rng.choice(_pick(_nums(d), avoid))
        v, span = rec[f.name], f.hi - f.lo
        inside = rng.random() < 0.5
        w = rng.uniform(0.15 * span, 0.4 * span)
        # The band is placed anywhere in the range and kept only if it holds (or misses)
        # the value as owed. Placing it *around* the value instead -- centred on it, or a
        # small gap to one side -- made where the band sits say whether it holds the
        # value: "a band near the middle of the range, and the rule says within" answered
        # 0.64. Placed uniformly, the band's position is independent of the answer.
        for _ in range(40):
            lo = rng.uniform(f.lo, f.hi - w)
            a, b = _round(f, lo), _round(f, lo + w)
            if min(abs(v - a), abs(v - b)) < 0.035 * span and rng.random() >= P_SOFT:
                continue                  # an edge in the soft band: only P_SOFT of them
            if (a <= v <= b) is (t == inside):
                return Range(f, a, b, inside)
        return None
    if name == "equals":
        e = rng.choice(_pick(d.enums, avoid))
        others = [x for x in e.values if x != rec[e.name]]
        return Equals(e, rec[e.name] if t else rng.choice(others))
    if name == "member":
        e = rng.choice(_pick(d.enums, avoid))
        k = rng.randint(2, max(2, len(e.values) - 1))
        others = [x for x in e.values if x != rec[e.name]]
        vals = ([rec[e.name]] + rng.sample(others, k - 1)) if t else rng.sample(others, k)
        return Member(e, tuple(sorted(vals)))
    if name == "count":
        it = rng.choice(_pick(d.items, avoid))
        n_have, most = len(rec[it.name]), min(it.hi, len(it.pool))
        # exactly at the list's length (`yes`) or one past it (`no`). Anything wider
        # leaks through the ends: with `n` one or two either side, "the rule asks for
        # at most 2" answered 0.69, because a small `n` that is `no` needs a list shorter
        # than it and lists are rarely empty. One past the longest list the field can
        # hold is allowed, or a full list would always be `yes`; an empty list is always
        # `no`, which is the rule, not a shortcut.
        n = n_have if t else n_have + 1
        return CountAtLeast(it, n) if 1 <= n <= most + 1 else None
    if name == "age":
        s = rng.choice(_pick(d.stamps, avoid))
        days = _days_before(rec[s.name])
        older = rng.random() < 0.5
        ok = [n for n in AGE_DAYS if ((days > n) if older else (days < n)) is t]
        if not ok:
            return None
        # the two cut-offs nearest the stamp on the side owed. The cut-offs are evenly
        # spaced, as the stamps are: on the old doubling grid (7, 14, 30, ... 365) a
        # cut-off was the nearest one *below* twice as many stamps as it was the nearest
        # one above, so the number alone said `yes` two times in three.
        ok.sort(key=lambda n: abs(n - days))
        return Age(s, rng.choice(ok[:2]), older)
    if name == "relative":
        a_name, b_name, gloss = rng.choice(list(d.pairs))
        by = {f.name: f for f in d.nums}
        a, b = rec[a_name], rec[b_name]
        if a == b:
            return None
        return Relative(by[a_name], by[b_name], gloss, (a > b) is t)
    if name in ("across", "across_mean"):
        if not ctx.records:
            return None
        opts = []
        for f in _pick(_nums(d), avoid):
            for up in (True, False):
                nd = AcrossExtreme(f, up) if name == "across" else AboveMean(f, up)
                if _verdict(nd, ctx) is t:
                    opts.append(nd)
        # records drawn around a shared centre sit close to their mean, so most
        # `across_mean` rules would be a soft target; keep the confident ones but for
        # `P_SOFT` of the time, as `_gap` does
        sure = [nd for nd in opts if abs(nd.p(ctx) - 0.5) > 0.497]
        if sure and rng.random() >= P_SOFT:
            opts = sure
        return rng.choice(opts) if opts else None
    raise ValueError(f"unknown primitive {name!r}")


def build_targeted(spec, d: Domain, rng: random.Random, ctx: Ctx,
                   want: bool, used: list[int] | None = None) -> Node | None:
    """A node of shape `spec` answering `want`, drawn clause-first (see above), or None.

    `used[i]` is how often clause `i` has decided the answer so far in this task. Some
    records force a clause (no `count` rule is `yes` of an empty list), which on its own
    left `count` the deciding clause of `and(member,count)` 0.70 of the time against 0.75
    for `member`; the patterns in which the least-used clause decides are drawn three
    times as often, so the task evens that out over its other records."""
    names = _spec_leaves(spec)
    pats = _patterns(spec, want)
    if not pats:
        return None
    weights = [w for _, w in pats]
    if used and len(names) > 1:
        low = min(used)
        j = rng.choice([i for i, u in enumerate(used) if u == low])
        weights = [w * (3.0 if j in _decisive(spec, p) else 1.0) for p, w in pats]
    for _ in range(BALANCE_TRIES):
        pat = rng.choices([p for p, _ in pats], weights=weights)[0]
        avoid: set = set()
        kids = []
        for name, t in zip(names, pat):
            nd = _leaf(name, d, rng, ctx, t, avoid)
            if nd is None:
                break
            avoid |= set(nd.reads)
            kids.append(nd)
        else:
            node = _assemble(spec, iter(kids))
            try:
                p = node.p(ctx)
            except Missing:
                continue
            if _verdict(node, ctx) is want and abs(p - 0.5) > 1e-9:
                return node
    return None


# ---- the records a rule is applied to ----------------------------------------------
# A `relative` rule compares two fields of one record, and `make_record` draws them
# independently over the same range -- so "the first field is high" said "the first is
# higher" 74 % of the time without looking at the second. The measured field is drawn
# near its limit instead (`_gap`), either side equally, as a limit and a reading
# usually are.
# The cross-record rules ask where the focus record ranks among three or four. Drawn
# independently over the whole range, the greatest of four values is near the top of
# the range, so "the value is high" answered "holds the greatest" 0.78 and "is above the
# average" 0.86. In a cross-record state each field is drawn around a shared centre, so
# a value's rank among the records shown is independent of where it sits in the range.
CLUSTER = 0.10


def _tie_pairs(d: Domain, rec: dict, rng: random.Random) -> None:
    by = {f.name: f for f in d.nums}
    for a_name, b_name, _ in d.pairs:
        b = by[b_name]
        below = rng.random() < 0.5
        # which side is drawn first and kept wherever it fits: reflecting an offset that
        # ran off the range put every limit near the top of it below its reading, and
        # "the reading is high" answered `relative` 0.61
        v = _near(b, rec[a_name], below, rng)
        if v is None:
            v = _near(b, rec[a_name], not below, rng)
        if v is not None:
            rec[b_name] = _round(b, v)


# `make_record` leaves a list empty one time in six or seven, and no `count` rule
# ("lists at least N entries", N >= 1) is `yes` of an empty list -- so "the list is
# empty" answered every such question and pushed `count` to be the deciding clause of
# its combinations (the count clause alone answered 0.84 of `and(member,count)` after
# clause-first sampling, before this). An
# empty list stays, at half the rate: it is still the case "none recorded" reads.
P_KEEP_EMPTY = 0.5


def _fill_lists(d: Domain, rec: dict, rng: random.Random) -> None:
    for it in d.items:
        if not rec[it.name] and rng.random() >= P_KEEP_EMPTY:
            k = rng.randint(1, min(it.hi, len(it.pool)))
            rec[it.name] = sorted(rng.sample(it.pool, k))


def _record(d: Domain, rng: random.Random) -> dict:
    rec = make_record(d, rng)
    _tie_pairs(d, rec, rng)
    _fill_lists(d, rec, rng)
    return rec


def _records(d: Domain, rng: random.Random, n: int) -> list[dict]:
    recs = [make_record(d, rng) for _ in range(n)]
    for f in d.nums:
        span = f.hi - f.lo
        c = rng.uniform(f.lo + CLUSTER * span, f.hi - CLUSTER * span)
        for r in recs:
            r[f.name] = _round(f, c + rng.uniform(-CLUSTER, CLUSTER) * span)
    for r in recs:
        _tie_pairs(d, r, rng)
        _fill_lists(d, r, rng)
    return recs


def _verdict(node: Node, ctx: Ctx) -> bool | None:
    """`yes`, `no`, or None when the record cannot answer the rule at all -- or answers
    it both ways, which a tie on a cross-record extreme does."""
    try:
        if node.ambiguous(ctx):
            return None
        return node.p(ctx) >= 0.5
    except Missing:
        return None


def build_balanced(spec, d: Domain, rng: random.Random, ctx: Ctx,
                   bal: list[int]) -> Node:
    """A node of the right shape whose answer is the one this task owes.

    Drawing a rule's constants independently of the record makes the *label prior* the
    task. `Equals` fires with probability 1/|values|, so `and(x, equals)` answers `no`
    84 % of the time and `or(x, member)` answers `yes` 75 % -- and `rule_transfer` scores
    raw accuracy against a 0.70 floor, which four families clear by predicting a constant.
    A rule nobody has to read is not a rule application.

    So the answer this task has emitted less often is chosen first and the rule is built
    to give it, clause by clause (`build_targeted`, see "verdict-first sampling"). The
    rule's *shape* is untouched; only its constants move, which is the same freedom
    `build_distinct` already takes. `bal[2:]` counts which clause decided each answer so
    far. Only when no clause-first rule fits does it fall back to redrawing whole rules
    up to `BALANCE_TRIES` times, as it used to; an abstention there is returned
    immediately -- it has no verdict to balance.
    """
    want = bal[1] < bal[0]
    n_leaves = len(_spec_leaves(spec))
    bal.extend([0] * (2 + n_leaves - len(bal)))
    # When the record admits no rule of this shape with the answer owed (no `count`
    # rule is `yes` of an empty list), give it the other answer, drawn the same way: a
    # free draw there put every "at least 2/3/4 entries" of an empty list on `no` and
    # made a small number in the rule read as `yes`.
    for w in (want, not want):
        node = build_targeted(spec, d, rng, ctx, w, bal[2:2 + n_leaves])
        if node is not None:
            bal[int(w)] += 1
            _count_decisive(spec, node, ctx, bal)
            return node
    # nothing clause-first fits (a tie on every cross-record field): draw freely
    best = build_distinct(spec, d, rng)
    got = _verdict(best, ctx)
    if got is None:
        if not best.ambiguous(ctx):
            return best
        # a tie on a cross-record extreme: neither label is right, so keep drawing
        for _ in range(BALANCE_TRIES - 1):
            best = build_distinct(spec, d, rng)
            got = _verdict(best, ctx)
            if got is not None or not best.ambiguous(ctx):
                break
        if got is None:
            return best                   # the caller drops a node that is still tied
    for _ in range(BALANCE_TRIES - 1):
        if got is want:
            break
        cand = build_distinct(spec, d, rng)
        alt = _verdict(cand, ctx)
        if alt is None:
            continue
        best, got = cand, alt
    bal[int(got)] += 1
    _count_decisive(spec, best, ctx, bal)
    return best


# ---- questions ----------------------------------------------------------------------

def _sentence(text: str) -> str:
    return (text[:1].upper() + text[1:]).rstrip(".") + "."


QUESTION_TEXT = {
    "escalate": "Does this {noun} meet the escalation rule?",
    "review": "Does this {noun} need review?",
    "exempt": "Does this {noun} qualify for the exemption?",
    "flag": "Should this {noun} be flagged?",
}
# The choice forms' question text. Kept apart so `_noul`'s draw over QUESTION_TEXT is
# unchanged. Every template here is registered in lod/assets/phrasing_specs/d02.json.
CHOICE_TEXT = {
    "band": "Which band does this {noun} fall into?",
    "route": "Which action does the rule select for this {noun}?",
    "select": "Which record satisfies the rule: {rule}?",
}
_BANK: PhrasingBank | None = None


def _q(name: str, key: str, **fmt) -> str:
    """The question text for template `name`, through the phrasing bank. With an empty
    bank this is the template itself; `key` is stable per record so a rebuild picks the
    same wording and two records of one task need not."""
    global _BANK
    if _BANK is None:
        _BANK = PhrasingBank.load()
    template = QUESTION_TEXT.get(name) or CHOICE_TEXT[name]
    return _BANK.pick(f"d02.{name}", template, key).format(**fmt)


def _noul(node: Node, d: Domain, shape: str, mode: str, ctx: Ctx, qid: str,
          rng: random.Random, prefix: str = "") -> Question:
    name = rng.choice(list(QUESTION_TEXT))
    text = _q(name, f"{d.noun}:{node.render(shape, mode)}:{qid}", noun=d.noun)
    # not str.capitalize(): it lowercases everything after the first character, which
    # turns "signal (dBm)" into "signal (dbm)", "line total (USD)" into "(usd)", and
    # every enum value into lower case. Only the first letter should move.
    criteria = [_sentence(node.negated(shape, mode)),
                _sentence(node.render(shape, mode))]
    try:
        p = min(max(node.p(ctx), 0.0), 1.0)
    except Missing as exc:
        return Question(qid, prefix + text, ["no", "yes"], None,
                        {"abstain": True, "reason": "field_not_in_state",
                         "missing": str(exc), "sig": node.sig, "mode": mode},
                        None, criteria)
    return Question(qid, prefix + text, ["no", "yes"], [1.0 - p, p],
                    {"sig": node.sig, "mode": mode}, None, criteria)


def _drop_field(recs: list[dict], nodes: list[Node], rng: random.Random,
                ctx: Ctx, bal: list[int] | None = None) -> None:
    """Remove one of the fields the first rule reads -- the *same* field from every record
    -- and only one whose absence leaves every rule that reads it undecided.

    An `and` with one operand already false is `no` whatever the missing field holds, and
    an `or` with one already true is `yes`. Dropping a field there and labelling the row
    `not_in_context` gave it two defensible answers: on the current code 295 of 1,605
    abstentions (18.4 %) were decided by the operand still present (domain-2 audit,
    `abstain_but_answer_determined_by_other_operand`). If no field of the first rule can
    go without deciding some rule, nothing is dropped and the row stays labelled.

    The field used to be drawn once per record, which only shows in the cross-record
    families, where a node reads two or three fields and each of three or four records
    dropped its own. Measured on an earlier build, 9.6 % of `and(across,relative)` states,
    10.2 % of `and(across_mean,member)` and 8.9 % of `or(across,threshold)` carried
    records with *different key sets* -- a shape no rule ever names, nothing else in the
    corpus produces, and nobody chose. The records are meant to differ in their values.
    """
    reads = [r for r in dict.fromkeys(nodes[0].reads) if any(r in rec for rec in recs)]
    rng.shuffle(reads)
    before = [_verdict(nd, ctx) for nd in nodes]
    for name in reads:
        saved = [(rec, rec[name]) for rec in recs if name in rec]
        for rec, _ in saved:
            del rec[name]
        if all(not (lo >= 0.5 or hi < 0.5)
               for nd in nodes if name in nd.reads
               for lo, hi in [nd.bounds(ctx)]):
            # an abstention has no verdict: the balance counted one, so take it back,
            # or the labelled rows drift toward whichever answer abstains less often
            if bal is not None:
                for nd, v in zip(nodes, before):
                    if v is not None and name in nd.reads:
                        bal[int(v)] -= 1
            return
        for rec, v in saved:
            rec[name] = v


def _keys(n: int) -> list[str]:
    return [f"record_{chr(ord('a') + i)}" for i in range(n)]


def _render_many(d: Domain, recs: list[dict], keys: list[str], shape: str) -> str:
    if shape == "json":
        return _json.dumps(dict(zip(keys, recs)), indent=2)
    return "\n\n".join(f"Record {k.rsplit('_', 1)[1].upper()}. {as_prose(d, r)}"
                       for k, r in zip(keys, recs))


def _distinct_nodes(spec, d: Domain, rng: random.Random, ctx: Ctx, bal: list[int],
                    n: int) -> list[Node]:
    """Up to `n` balanced rules for one state, no two with the same criteria and none
    that the state answers both ways.

    Drawn independently, two questions of one example were the same question 4.4 % of
    the time (`count` 46 of 300 examples, `relative` 40: a domain has one comparable pair,
    so `relative` has only two rules per domain to draw from), and a tie on a
    cross-record extreme made 0.28 % of cross-record rules true in both readings.
    """
    out: list[Node] = []
    seen: set[str] = set()
    for _ in range(n):
        for _ in range(6):
            node = build_balanced(spec, d, rng, ctx, bal)
            key = node.render("json", "direct")
            if key not in seen and not node.ambiguous(ctx):
                seen.add(key)
                out.append(node)
                break
            got = _verdict(node, ctx)          # rejected: take back its balance count
            if got is not None:
                bal[int(got)] -= 1
    return out


def _example(spec, d: Domain, rng: random.Random, shape: str, task: str,
             n_questions: int = 1, bal: list[int] | None = None) -> Example:
    mode = "indirect" if rng.random() < P_INDIRECT else "direct"
    bal = [0, 0] if bal is None else bal
    if needs_records(spec):
        n = rng.randint(3, 4)
        keys = _keys(n)
        recs = _records(d, rng, n)
        focus = rng.randrange(n)
        ctx = Ctx(focus=recs[focus], records=dict(zip(keys, recs)), focus_key=keys[focus])
        nodes = _distinct_nodes(spec, d, rng, ctx, bal, n_questions)
        if not nodes:
            return _example(spec, d, rng, shape, task, n_questions, bal)
        if rng.random() < P_MISSING:
            _drop_field(recs, nodes, rng, ctx, bal)
        letter = keys[focus].rsplit("_", 1)[1].upper()
        qs = [_noul(nd, d, shape, mode, ctx, f"q{i}", rng, prefix=f"For record {letter}: ")
              for i, nd in enumerate(nodes)]
        return Example(state=_render_many(d, recs, keys, shape), questions=qs, task=task)

    rec = _record(d, rng)
    ctx = Ctx(focus=rec)
    nodes = _distinct_nodes(spec, d, rng, ctx, bal, n_questions)
    if rng.random() < P_MISSING:
        _drop_field([rec], nodes, rng, ctx, bal)
    qs = [_noul(nd, d, shape, mode, ctx, f"q{i}", rng) for i, nd in enumerate(nodes)]
    return Example(state=render(d, rec, shape), questions=qs, task=task)


# ---- choice forms --------------------------------------------------------------------

def _band_example(d, rng, shape, task) -> Example:
    mode = "indirect" if rng.random() < P_INDIRECT else "direct"
    rec = _record(d, rng)
    f = rng.choice(_nums(d))
    k = rng.randint(3, 5)
    span = f.hi - f.lo
    edges = [_round(f, f.lo + span * i / k) for i in range(k + 1)]
    keys = [f"tier_{chr(ord('a') + i)}" for i in range(k)]
    ref = field_ref(f, shape, mode)
    crit = [_sentence(f"{ref} is from {_fmt(f, edges[i])} up to {_fmt(f, edges[i + 1])}")
            for i in range(k)]
    crit[-1] = crit[-1].rstrip(".") + " (the top band)."
    if f.name not in rec:
        target, meta = None, {"abstain": True, "reason": "field_not_in_state"}
    else:
        v = rec[f.name]
        w = [soft(v, edges[i], span, True) * soft(v, edges[i + 1], span, False)
             for i in range(k)]
        if v <= edges[0]:
            w[0] = max(w[0], soft(v, edges[1], span, False))
        if v >= edges[-1]:
            w[-1] = max(w[-1], soft(v, edges[-2], span, True))
        tot = sum(w) or 1.0
        target, meta = [x / tot for x in w], {"sig": "band", "mode": mode}
    q = Question("band", _q("band", f"{d.noun}:{crit[0]}:{rec.get(f.name)}",
                            noun=d.noun), keys, target, meta, None, crit)
    return Example(state=render(d, rec, shape), questions=[q], task=task)


# A routing state holds one record, so a routing rule may not be a cross-record one.
# It used to draw from `TRAINED_COMBOS` whole, two of which read every record shown.
# `AcrossExtreme.p` and `AboveMean.p` raise `Missing` when there is nothing to compare
# against, so 204 of 2,031 routing questions in an earlier build -- 10.0 %, every abstention
# the family emitted -- were labelled `not_in_context` with a `missing` field that was
# sitting in the state, under a criterion reading "this record holds the smallest
# `changed_files` of all the records shown" beside a state showing exactly one record.
# `_routing_example` drops no field, so a routing abstention is always this bug.
ROUTING_SPECS = tuple(s for s in TRAINED_COMBOS if not needs_records(s)) + PRIMITIVES[:6]


def _routing_example(d, rng, shape, task, bal: list[int] | None = None) -> Example:
    """Several rules as option descriptions, evaluated in order, with a stated default.

    The shape the API contract exists for: rules in the criteria, one record, pick the
    option whose rule fires.

    `bal` counts the actions emitted so far, and the rules are redrawn (up to
    `BALANCE_TRIES`) toward the least-emitted one, as `build_balanced` does for the
    binary families. Drawn freely, `escalate` was the answer 46.7 % of the time against
    25.7 % `hold` (domain-2 audit): a constant beat chance by 14 points.
    """
    mode = "indirect" if rng.random() < P_INDIRECT else "direct"
    rec = _record(d, rng)
    bal = [0, 0, 0] if bal is None else bal
    want = min(range(3), key=bal.__getitem__)
    for _ in range(BALANCE_TRIES):
        first = build_distinct(rng.choice(ROUTING_SPECS), d, rng)
        second = _primitive(rng.choice(("equals", "member")), d, rng)
        try:
            p1, p2 = first.p(Ctx(focus=rec)), second.p(Ctx(focus=rec))
        except Missing:
            break
        got = 0 if p1 >= 0.5 else (1 if p2 >= 0.5 else 2)
        if got == want:
            break
    else:
        got = None
    if got is not None:
        bal[got] += 1
    keys = ["escalate", "hold", "route_standard"]
    # Every option states its whole condition. The options are shuffled here and again
    # at training time, and `lod.training.data.augment_criteria` makes keys opaque and withholds
    # or decoys options, so a criterion that points at another one -- "the first rule
    # does not apply", "neither of the rules above" -- points at the wrong option or at
    # nothing: the default was not listed last in 62 % of routing questions and the
    # "first rule" was not listed first in 60 % (domain-2 audit, before this change).
    r1, r2 = first.render(shape, mode), second.render(shape, mode)
    crit = [f"Choose this when {r1}.",
            f"Choose this when {r2}, unless this also holds: {r1}.",
            f"Choose this when neither of these holds: {r1}; {r2}."]
    order = list(range(3))
    rng.shuffle(order)
    keys = [keys[i] for i in order]
    crit = [crit[i] for i in order]
    pos = {old: new for new, old in enumerate(order)}
    ctx = Ctx(focus=rec)
    text = _q("route", f"{d.noun}:{r1}:{r2}", noun=d.noun)
    try:
        p1 = first.p(ctx)
        p2 = second.p(ctx)
    except Missing as exc:
        q = Question("route", text,
                     keys, None,
                     {"abstain": True, "reason": "field_not_in_state", "missing": str(exc)},
                     None, crit)
        return Example(state=render(d, rec, shape), questions=[q], task=task)
    out = [0.0, 0.0, 0.0]
    out[pos[0]] = p1
    out[pos[1]] = (1.0 - p1) * p2
    out[pos[2]] = (1.0 - p1) * (1.0 - p2)
    q = Question("route", text, keys, out, {"sig": "routing", "mode": mode}, None, crit)
    return Example(state=render(d, rec, shape), questions=[q], task=task)


def _select_tell(node: Node, recs: list[dict]) -> int | None:
    """The record a reader would pick without applying the rule, where that is not the
    rule itself: for `within` a range, the record farthest from the others (0.44 against
    chance 0.29). One-sided rules -- threshold, count, age, `outside` a range -- have no
    such pick: exactly one record passing them *is* the extreme or the odd one out.
    (`relative` had one, the record whose first field is most extreme the rule's way,
    0.41; drawing each limit near its reading, `_tie_pairs`, took it to chance.)"""
    if isinstance(node, Range) and node.inside:
        vals = [r[node.f.name] for r in recs]
        mu = sum(vals) / len(vals)
        return max(range(len(vals)), key=lambda i: abs(vals[i] - mu))
    return None


def _select_example(d, rng, shape, task) -> Example:
    """Which of several records satisfies the rule -- held out.

    A cross-record *choice*: the evidence is spread over records and the answer is a
    record, not a verdict. Every rule inside it is trained; the shape is not.
    """
    mode = "indirect" if rng.random() < P_INDIRECT else "direct"
    n = rng.randint(3, 4)
    keys = _keys(n)
    recs = [_record(d, rng) for _ in range(n)]
    for _ in range(60):
        node = build_distinct(rng.choice(tuple(PRIMITIVES[:8])), d, rng)
        try:
            ps = [node.p(Ctx(focus=r)) for r in recs]
        except Missing:
            continue
        # Exactly one record satisfies it, and no other comes close. Bounding only the
        # winner let a runner-up sit anywhere below 0.9: on an earlier build, 50 of 1,354
        # `select` questions (3.7 %, and 5 of the 150 `rule_transfer` scores) had a second
        # record that literally satisfies the stated rule while the target names one of
        # them.
        hits = [i for i, p in enumerate(ps) if p > 0.9]
        if (len(hits) == 1 and all(p < 0.1 for i, p in enumerate(ps) if i != hits[0])
                and not (_select_tell(node, recs) == hits[0] and rng.random() < 0.5)):
            target = [1.0 if i == hits[0] else 0.0 for i in range(n)]
            runner_up = max(p for i, p in enumerate(ps) if i != hits[0])
            rule = node.render(shape, mode)
            q = Question("select",
                         _q("select", f"{d.noun}:{rule}:{hits[0]}", rule=rule),
                         keys, target,
                         # `runner_up` is the closest any *other* record comes to
                         # satisfying the rule, so the uniqueness the question claims is
                         # checkable after the fact rather than only at draw time.
                         {"sig": "select", "mode": mode,
                          "runner_up": round(float(runner_up), 4)}, None,
                         [f"The record filed under {k}." for k in keys])
            return Example(state=_render_many(d, recs, keys, shape), questions=[q],
                           task=task)
        # nudge one record until exactly one satisfies it, rather than redrawing forever
        recs = [_record(d, rng) for _ in range(n)]
    return _example("threshold", d, rng, shape, task)


# ---- contrastive pairs ---------------------------------------------------------------

def _contrast_criteria(d, rng, shape, task) -> Example:
    """One state, two rules differing only in a threshold, opposite answers, one sequence."""
    mode = "indirect" if rng.random() < P_INDIRECT else "direct"
    rec = _record(d, rng)
    for _ in range(40):
        f = rng.choice(_nums(d))
        v = rec[f.name]
        span = f.hi - f.lo
        below = _round(f, v - rng.uniform(0.12, 0.3) * span)
        above = _round(f, v + rng.uniform(0.12, 0.3) * span)
        if below < f.lo or above > f.hi:
            continue
        # Both the direction and the order are drawn. With `greater` fixed and the lower
        # threshold always first, question `c0` was `yes` and `c1` was `no` in 300 of
        # 300 examples: the answer was the question's position (domain-2 audit).
        greater = rng.random() < 0.5
        thrs = [below, above]
        rng.shuffle(thrs)
        qs = []
        for i, thr in enumerate(thrs):
            node = Threshold(f, thr, greater)
            qs.append(_noul(node, d, shape, mode, Ctx(focus=rec), f"c{i}", rng))
            qs[-1].question = _q("escalate", f"{d.noun}:{qs[-1].descriptions[1]}",
                                 noun=d.noun)
        return Example(state=render(d, rec, shape), questions=qs, task=task)
    return _example("threshold", d, rng, shape, task)


def _contrast_state(d, rng, shape, task) -> Example:
    """One rule, two records in one state, opposite answers. Everything but the field the
    rule reads is held fixed, so nothing else can explain the two answers."""
    mode = "indirect" if rng.random() < P_INDIRECT else "direct"
    a = _record(d, rng)
    b = dict(a)
    f = rng.choice(_nums(d))
    span = f.hi - f.lo
    thr = _round(f, rng.uniform(f.lo + 0.3 * span, f.hi - 0.3 * span))
    low = _round(f, rng.uniform(f.lo, thr - 0.12 * span))
    high = _round(f, rng.uniform(thr + 0.12 * span, f.hi))
    # Which record sits above the threshold, and which way the rule points, are both
    # drawn. Record A used to be below and the rule was always "greater than", so
    # "For record A" was `no` and "For record B" `yes` in 300 of 300 examples.
    if rng.random() < 0.5:
        low, high = high, low
    a[f.name], b[f.name] = low, high
    for i in d.ids:
        if i in b:
            b[i] = b[i].rsplit("-", 1)[0] + f"-{rng.randrange(1000, 9999)}"
    node = Threshold(f, thr, rng.random() < 0.5)
    keys = ("record_a", "record_b")
    qs = []
    for i, (k, r) in enumerate(zip(keys, (a, b))):
        letter = k.rsplit("_", 1)[1].upper()
        q = _noul(node, d, shape, mode, Ctx(focus=r), f"s{i}", rng,
                  prefix=f"For record {letter}: ")
        q.question = f"For record {letter}: " + _q(
            "escalate", f"{d.noun}:{q.descriptions[1]}:{letter}", noun=d.noun)
        qs.append(q)
    return Example(state=render_pair(d, a, b, shape, keys), questions=qs, task=task)


# ---- tasks ---------------------------------------------------------------------------

SPEC_BY_SLUG = {slug(s): s for s in TRAINED_SPECS + HELDOUT_COMBOS + DEV_COMBOS}


def _loader(name: str, task: str, seed: int):
    def load(n: int) -> Iterator[Example]:
        h = hashlib.sha256(f"{seed}:{name}:{task}".encode()).digest()
        rng = random.Random(int.from_bytes(h[:8], "big"))
        bal = [0, 0]          # `no`/`yes` emitted so far, so the task stays near even
        bal3 = [0, 0, 0]      # routing: escalate / hold / route_standard emitted so far
        for i in range(n):
            d = DOMAINS[i % len(DOMAINS)]
            shape = "json" if rng.random() < 0.55 else "prose"
            if name == "band":
                yield _band_example(d, rng, shape, task)
            elif name == "routing":
                yield _routing_example(d, rng, shape, task, bal3)
            elif name == "select":
                yield _select_example(d, rng, shape, task)
            elif name == "contrast_criteria":
                yield _contrast_criteria(d, rng, shape, task)
            elif name == "contrast_state":
                yield _contrast_state(d, rng, shape, task)
            else:
                yield _example(SPEC_BY_SLUG[name], d, rng, shape, task,
                               n_questions=rng.randint(1, 3), bal=bal)
    return load


def tasks() -> list[RealTask]:
    """Trained specs in `train`, twice over; new combinations in `testreal`, and a disjoint
    set of new combinations in `devreal`.

    The `_eval` twin is a second independent draw of a trained family, on a different
    seed. It carries `force_split=None`, so `split_of` hashes its name: nineteen of the
    twenty-four landed in `train` in an earlier build and five did not. That is extra
    training data with a misleading name, not a held-out set -- `probe_tasks()` below is
    the held-out stream, and it is the one `rule_transfer` should score.

    So the twin is now forced to `train` as well. Hashed, three of them
    (`rule_count_eval`, `rule_equals_eval`, `rule_routing_eval`) landed in `testreal` and
    three (`rule_not_member_eval`, `rule_not_relative_eval`, `rule_threshold_eval`) in
    `devreal` -- splits that exist to measure *unseen* schemas, holding six families the
    model trains on under their own names (domain-2 audit, seed-0 build of row 33).

    `DEV_COMBOS` go to `devreal`: unseen combinations of trained primitives, disjoint from
    test's, so the split that selects checkpoints measures transfer as test does.
    """
    out: list[RealTask] = []
    trained = [slug(s) for s in TRAINED_SPECS] + list(CHOICE_TRAINED) + \
              ["contrast_criteria", "contrast_state"]
    for name in trained:
        for suffix, split in (("", "train"), ("_eval", "train")):
            task = f"rule_{name}{suffix}"
            out.append(RealTask(
                row=ROW, name=task, licence=LICENCE, url=URL,
                load=_loader(name, task, 0 if not suffix else 991),
                soft=True, real=False, force_split=split,
                notes=f"rule spec {name}; code-labelled"))
    for name in [slug(s) for s in HELDOUT_COMBOS] + list(CHOICE_HELDOUT):
        task = f"rule_{name}"
        out.append(RealTask(
            row=ROW, name=task, licence=LICENCE, url=URL,
            load=_loader(name, task, 7717), soft=True, real=False,
            force_split="testreal",
            notes=f"rule spec {name}; NEVER trained on -- the rule_transfer gate reads this"))
    for name in DEV_FAMILIES:
        task = f"rule_{name}"
        out.append(RealTask(
            row=ROW, name=task, licence=LICENCE, url=URL,
            load=_loader(name, task, 5323), soft=True, real=False,
            force_split="devreal",
            notes=f"rule spec {name}; held out of train and test; a dev structure -- "
                  f"selects checkpoints and fits T"))
    return out


# ---- the stream rule_transfer scores --------------------------------------------------

PROBE_SEED = 20260921


def probe_tasks() -> list[RealTask]:
    """Every family again, from a seed no corpus task uses. Nothing registers these.

    An earlier rule_transfer probe compared trained families against held-out ones by
    loading `rule_{fam}` from `tasks()` -- the same task object, and so the same
    deterministic stream, that the generate stage had already written into the corpus.
    Measured on an earlier build: 139 to 147 of every 150 states it scored were verbatim in
    `train.jsonl`, a mean of 95.5 %. `acc_trained_families = 0.978` was therefore recall of
    training examples, and the 0.380 drop it reported was an upper bound on the transfer
    gap rather than a measurement of it. The `_eval` twins meant to prevent this were left
    on `force_split=None`, so `split_of` hashed them and nineteen of twenty-four landed
    back in train.

    These tasks are not returned by `tasks()`, so no split can ever contain them. With
    `rule_transfer` reading `probe_tasks()`, the trained-family number becomes
    held-out-example accuracy for the first time.
    """
    out: list[RealTask] = []
    for name in list(TRAINED_FAMILIES) + list(HELDOUT_FAMILIES):
        out.append(RealTask(
            row=ROW, name=f"rule_{name}", licence=LICENCE, url=URL,
            load=_loader(name, f"rule_{name}_probe", PROBE_SEED),
            soft=True, real=False, force_split=None,
            notes=f"rule spec {name}; probe only -- never registered, never in a split"))
    return out
