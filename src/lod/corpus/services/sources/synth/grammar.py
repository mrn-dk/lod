"""A grammar of rule primitives, and the combinations built from them.

An earlier generator had thirteen hand-written rule families. It worked — trained-family
accuracy went from 0.543 to 0.978 — and it did not transfer: held-out families scored
0.598, and the two that failed worst said why.

    range_outside   0.982   a trained family negated. Transfers.
    membership      0.616   a trained family generalised. Barely.
    disjunction     0.527   two trained families joined by `or`. Chance.
    ranking         0.267   a *different shape*: compare a field across records. Chance.

So the model learned nine rules, not rule application. Two things follow, and this module
is both of them.

**Rules compose.** A rule is an expression over primitives, not a family. `Threshold`,
`Range`, `Equals`, `Member`, `CountAtLeast`, `Age`, `Converted`, `Relative` and the
`Across*` family are the leaves; `And`, `Or` and `Not` combine them. Every primitive is
trained. What `rule_transfer` holds out is whole **combinations** of primitives the model
has already met individually, which is the transfer question worth asking: not "have you
seen `or`" but "can you apply `or` to two things you know".

**Rules may not name their field.** Every earlier rule quoted the key or the label, so the
model never had to find the evidence — and on a rule phrased the way a person writes one,
"outside its required temperature range", it was confidently wrong. Half of the rules here
are drawn in `indirect` mode, which names a field only by what it records, and `Relative`
compares two fields of one record without naming either.

A node knows four things: its `sig` (the canonical shape, which is what the split is
taken over), the fields it `reads`, how to `render` itself as criteria text, and how to
evaluate itself to a probability.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from datetime import date
from typing import Sequence

from lod.corpus.services.sources.synth.records import (
    TODAY,
    Domain,
    Enum,
    Items,
    Num,
    Stamp,
    field_ref,
)

SOFT_BAND = 0.03          # a threshold is soft within this fraction of the field's range


def soft(value: float, thr: float, span: float, greater: bool = True,
         exact: bool = False) -> float:
    """p(`value > thr`), smoothed over a band `SOFT_BAND * span` wide.

    tanh rather than a step, because the loss is a proper scoring rule and the optimum it
    points at is the true probability. Far from the threshold this is 1.0 or 0.0 to four
    decimals; near it, it is the honest answer to a rule applied to a measurement.

    `exact` turns the band off. A band models measurement uncertainty, and a count of
    parcels has none: when the field and the threshold are both whole numbers, `value ==
    thr` is the only case the band ever reaches, and there it returns 0.5 for a question
    whose answer is flatly `no`. That was 0.7 % of this domain's binary targets telling
    the model to toss a coin over "is 4 greater than 4".
    """
    if exact and float(value) == float(thr):
        return float(value > thr) if greater else float(value < thr)
    scale = max(span * SOFT_BAND, 1e-9)
    p = 0.5 * (1.0 + math.tanh(2.5 * (value - thr) / scale))
    return p if greater else 1.0 - p


def _round(f: Num, v: float) -> float:
    return int(round(v)) if f.integral else round(v, 1)


def _fmt(f: Num, v: float) -> str:
    return f"{_round(f, v)}{f.unit}"


def _bare(ref: str) -> str:
    """A field reference with the trailing comma an indirect-with-unit reference carries.

    `field_ref(..., "indirect")` ends "..., in kg," so that it reads as a parenthetical
    before " is". Followed by anything else -- ", converted,", ", and", the end of the
    criterion -- the comma doubled: 6.5 % of this domain's questions rendered
    "..., in kg,, converted, is at least 9089.015 g" (measured by the domain-2 audit).
    """
    return ref[:-1] if ref.endswith(",") else ref


def _days_before(stamp: str, ref: date = TODAY) -> int:
    y, m, d = (int(x) for x in stamp.split("-"))
    return (ref - date(y, m, d)).days


class Missing(Exception):
    """A field the rule names is not in the record. Not an error: an abstention."""


# ---- context ----------------------------------------------------------------------

@dataclass
class Ctx:
    """What a rule is evaluated against.

    `focus` is the record the question is about. `records` is every record in the state,
    which is empty for a single-record state and populated for the cross-record ones --
    so an `Across` node and a `Threshold` node can sit on either side of the same `and`
    and both find what they need.
    """

    focus: dict
    records: dict[str, dict] | None = None
    focus_key: str = ""

    def get(self, name: str):
        if name not in self.focus:
            raise Missing(name)
        return self.focus[name]


# ---- nodes -------------------------------------------------------------------------

class Node:
    sig = "node"

    @property
    def reads(self) -> tuple[str, ...]:      # pragma: no cover - overridden
        raise NotImplementedError

    def render(self, shape: str, mode: str) -> str:   # pragma: no cover - overridden
        raise NotImplementedError

    def negated(self, shape: str, mode: str) -> str:
        """The criteria text for the `no` branch. Overridden where English has a
        shorter way to say it than "it is not the case that"."""
        return f"It is not the case that {self.render(shape, mode)[0].lower()}" \
               f"{self.render(shape, mode)[1:]}"

    def p(self, ctx: Ctx) -> float:          # pragma: no cover - overridden
        raise NotImplementedError

    def ambiguous(self, ctx: Ctx) -> bool:
        """True when the record makes both the rule and its negation read true."""
        return False

    def bounds(self, ctx: Ctx) -> tuple[float, float]:
        """The range `p` can take over every value a missing field might have.

        A leaf whose field is absent can be anything; `And`/`Or`/`Not` propagate that.
        When the range sits wholly on one side of 0.5 the record *does* answer the rule
        even though a field is gone -- "X > 5, and Y is cleared" with Y pending is `no`
        whatever X is -- and labelling that `not_in_context` gives the question two
        defensible answers. `rules._drop_field` uses this to only remove a field whose
        absence actually leaves the rule undecided.
        """
        try:
            p = self.p(ctx)
        except Missing:
            return 0.0, 1.0
        return p, p


@dataclass
class Threshold(Node):
    f: Num
    thr: float
    greater: bool
    sig = "threshold"

    @property
    def reads(self):
        return (self.f.name,)

    def render(self, shape, mode):
        word = "greater than" if self.greater else "less than"
        return f"{field_ref(self.f, shape, mode)} is {word} {_fmt(self.f, self.thr)}"

    def negated(self, shape, mode):
        word = "at most" if self.greater else "at least"
        return f"{field_ref(self.f, shape, mode)} is {word} {_fmt(self.f, self.thr)}"

    def p(self, ctx):
        return soft(ctx.get(self.f.name), self.thr, self.f.hi - self.f.lo, self.greater,
                    exact=self.f.integral)


@dataclass
class Range(Node):
    f: Num
    lo: float
    hi: float
    inside: bool
    sig = "range"

    @property
    def reads(self):
        return (self.f.name,)

    def _band(self):
        return f"{_fmt(self.f, self.lo)} to {_fmt(self.f, self.hi)}"

    def render(self, shape, mode):
        where = "within" if self.inside else "outside"
        return f"{field_ref(self.f, shape, mode)} is {where} the range {self._band()}"

    def negated(self, shape, mode):
        where = "outside" if self.inside else "within"
        return f"{field_ref(self.f, shape, mode)} is {where} the range {self._band()}"

    def p(self, ctx):
        v = ctx.get(self.f.name)
        span = self.f.hi - self.f.lo
        if self.f.integral:
            # "within the range 4 to 9" reads inclusively, and whole numbers have no
            # band for the edges to sit inside -- so the edges are in, exactly.
            inside = float(self.lo <= v <= self.hi)
        else:
            inside = min(soft(v, self.lo, span, True), soft(v, self.hi, span, False))
        return inside if self.inside else 1.0 - inside


@dataclass
class Equals(Node):
    f: Enum
    value: str
    sig = "equals"

    @property
    def reads(self):
        return (self.f.name,)

    def render(self, shape, mode):
        return f"{field_ref(self.f, shape, mode)} is {self.value}"

    def negated(self, shape, mode):
        return f"{field_ref(self.f, shape, mode)} is anything other than {self.value}"

    def p(self, ctx):
        return 1.0 if ctx.get(self.f.name) == self.value else 0.0


@dataclass
class Member(Node):
    f: Enum
    values: tuple[str, ...]
    sig = "member"

    @property
    def reads(self):
        return (self.f.name,)

    def _listed(self):
        return ", ".join(self.values[:-1]) + f" or {self.values[-1]}"

    def render(self, shape, mode):
        return f"{field_ref(self.f, shape, mode)} is one of {self._listed()}"

    def negated(self, shape, mode):
        return f"{field_ref(self.f, shape, mode)} is none of {self._listed()}"

    def p(self, ctx):
        return 1.0 if ctx.get(self.f.name) in self.values else 0.0


@dataclass
class CountAtLeast(Node):
    f: Items
    n: int
    sig = "count"

    @property
    def reads(self):
        return (self.f.name,)

    def render(self, shape, mode):
        return f"{field_ref(self.f, shape, mode)} lists at least {self.n} entries"

    def negated(self, shape, mode):
        return f"{field_ref(self.f, shape, mode)} lists fewer than {self.n} entries"

    def p(self, ctx):
        return 1.0 if len(ctx.get(self.f.name)) >= self.n else 0.0


@dataclass
class Age(Node):
    f: Stamp
    days: int
    older: bool
    sig = "age"

    @property
    def reads(self):
        return (self.f.name,)

    def render(self, shape, mode):
        word = "more" if self.older else "less"
        return (f"the date in {field_ref(self.f, shape, mode)} is {word} than "
                f"{self.days} days before {TODAY.isoformat()}")

    def negated(self, shape, mode):
        # NOT "less than N" for "more than N". `p` counts whole days and is exact at the
        # boundary, so at exactly N days "more than N" is false -- and "less than N" is
        # false too. Both branches then read false while one of them is labelled 1.0,
        # which is the same defect `soft(exact=True)` fixed on the probability side and
        # was left standing on the text side. Measured on an earlier build: 10 questions
        # across `age`, `or(equals,age)`, `or(range,age)` and `and(age,equals)` whose `no`
        # criteria described something that is not true of the record.
        word = "or fewer" if self.older else "or more"
        return (f"the date in {field_ref(self.f, shape, mode)} is {self.days} days "
                f"{word} before {TODAY.isoformat()}")

    def p(self, ctx):
        # Whole days on both sides, and both are exact: the reference date is printed in
        # the criterion and the stamp in the state, so "more than 365 days before" has
        # one answer. A band here is not measurement noise -- there is none -- and it
        # used to reach ~22 days either side of a 365-day rule: a stamp 366 days back
        # was labelled 0.52 `yes` (domain-2 audit, `confident_far_from_boundary`).
        days = _days_before(ctx.get(self.f.name))
        return float(days > self.days) if self.older else float(days < self.days)


@dataclass
class Converted(Node):
    f: Num
    thr_alt: float
    greater: bool
    sig = "converted"

    @property
    def reads(self):
        return (self.f.name,)

    def _thr_base(self):
        return self.thr_alt / self.f.alt[1]

    def render(self, shape, mode):
        word = "greater than" if self.greater else "less than"
        return (f"{_bare(field_ref(self.f, shape, mode))}, converted, is {word} "
                f"{self.thr_alt} {self.f.alt[0]}")

    def negated(self, shape, mode):
        word = "at most" if self.greater else "at least"
        return (f"{_bare(field_ref(self.f, shape, mode))}, converted, is {word} "
                f"{self.thr_alt} {self.f.alt[0]}")

    def p(self, ctx):
        return soft(ctx.get(self.f.name), self._thr_base(), self.f.hi - self.f.lo,
                    self.greater)


@dataclass
class Relative(Node):
    """One field of a record compared against another field of the same record.

    Carries no constant, so it cannot be answered by matching a number, and in `indirect`
    mode it names neither field. This is the shape of "the cargo has been outside its
    required temperature range" -- and the shape nothing in the earlier corpus had.
    """

    a: Num
    b: Num
    gloss: str
    greater: bool
    sig = "relative"

    @property
    def reads(self):
        return (self.a.name, self.b.name)

    def render(self, shape, mode):
        word = "higher" if self.greater else "lower"
        if mode == "indirect":
            return f"comparing {self.gloss}, the first is {word}"
        return (f"{field_ref(self.a, shape, mode)} is "
                f"{'above' if self.greater else 'below'} "
                f"{field_ref(self.b, shape, mode)}")

    def negated(self, shape, mode):
        word = "not higher" if self.greater else "not lower"
        if mode == "indirect":
            return f"comparing {self.gloss}, the first is {word}"
        return (f"{field_ref(self.a, shape, mode)} is "
                f"{'at or below' if self.greater else 'at or above'} "
                f"{field_ref(self.b, shape, mode)}")

    def p(self, ctx):
        return soft(ctx.get(self.a.name), ctx.get(self.b.name),
                    self.a.hi - self.a.lo, self.greater,
                    exact=self.a.integral and self.b.integral)


@dataclass
class AcrossExtreme(Node):
    """The focus record holds the largest (or smallest) value of a field, of all records.

    A different *shape* from every other primitive: the evidence is spread over several
    records and the rule is a comparison, not a predicate. `ranking` was this shape in the
    earlier generator, it was held out, and it scored 0.267 -- chance. It is trained now.
    """

    f: Num
    greatest: bool
    sig = "across"

    @property
    def reads(self):
        return (self.f.name,)

    def render(self, shape, mode):
        word = "greatest" if self.greatest else "smallest"
        if mode == "indirect":
            # "holds the greatest the field that records how many parcels the shipment
            # is made of of all the records shown" was the old indirect rendering
            return (f"of all the records shown, this record holds the {word} value in "
                    f"{_bare(field_ref(self.f, shape, mode))}")
        return (f"this record holds the {word} {field_ref(self.f, shape, mode)} "
                f"of all the records shown")

    def negated(self, shape, mode):
        word = "greatest" if self.greatest else "smallest"
        if mode == "indirect":
            return (f"some other record holds the {word} value in "
                    f"{_bare(field_ref(self.f, shape, mode))}")
        return (f"some other record holds the {word} "
                f"{field_ref(self.f, shape, mode)}")

    def _vals(self, ctx):
        if not ctx.records:
            raise Missing(self.f.name)
        vals = []
        for rec in ctx.records.values():
            if self.f.name not in rec:
                raise Missing(self.f.name)
            vals.append(rec[self.f.name])
        return vals

    def ambiguous(self, ctx) -> bool:
        """The focus record shares the extreme with another record. "This record holds
        the greatest X" and "some other record holds the greatest X" are then both true,
        so neither label is right: 0.28 % of cross-record questions (domain-2 audit)."""
        try:
            vals = self._vals(ctx)
            mine = ctx.get(self.f.name)
        except Missing:
            return False
        best = max(vals) if self.greatest else min(vals)
        return mine == best and vals.count(best) > 1

    def p(self, ctx):
        vals = self._vals(ctx)
        mine = ctx.get(self.f.name)
        best = max(vals) if self.greatest else min(vals)
        return 1.0 if mine == best else 0.0


@dataclass
class AboveMean(Node):
    """The focus record's value is above the mean of every record shown."""

    f: Num
    above: bool
    sig = "across_mean"

    @property
    def reads(self):
        return (self.f.name,)

    def _whose(self, shape, mode):
        # "this record's the field that records ..." is not English; the indirect form
        # names the record first instead
        if mode == "indirect":
            return f"in this record, {field_ref(self.f, shape, mode)}"
        return f"this record's {field_ref(self.f, shape, mode)}"

    def render(self, shape, mode):
        word = "above" if self.above else "below"
        return (f"{self._whose(shape, mode)} is {word} the average "
                f"across all the records shown")

    def negated(self, shape, mode):
        word = "below or equal to" if self.above else "above or equal to"
        return (f"{self._whose(shape, mode)} is {word} the average "
                f"across all the records shown")

    def p(self, ctx):
        if not ctx.records:
            raise Missing(self.f.name)
        vals = []
        for rec in ctx.records.values():
            if self.f.name not in rec:
                raise Missing(self.f.name)
            vals.append(rec[self.f.name])
        mean = sum(vals) / len(vals)
        # exact at equality for whole-number fields, as `Threshold` is: a count sitting
        # exactly on the mean is flatly not above it, not a coin flip
        return soft(ctx.get(self.f.name), mean, self.f.hi - self.f.lo, self.above,
                    exact=self.f.integral)


# ---- combinators -------------------------------------------------------------------

@dataclass
class And(Node):
    a: Node
    b: Node

    @property
    def sig(self):
        return f"and({self.a.sig},{self.b.sig})"

    @property
    def reads(self):
        return self.a.reads + self.b.reads

    def render(self, shape, mode):
        return f"{self.a.render(shape, mode)}, and {self.b.render(shape, mode)}"

    def negated(self, shape, mode):
        return (f"either {self.a.negated(shape, mode)}, or "
                f"{self.b.negated(shape, mode)}, or both")

    def p(self, ctx):
        return self.a.p(ctx) * self.b.p(ctx)

    def ambiguous(self, ctx):
        return self.a.ambiguous(ctx) or self.b.ambiguous(ctx)

    def bounds(self, ctx):
        (al, ah), (bl, bh) = self.a.bounds(ctx), self.b.bounds(ctx)
        return al * bl, ah * bh


@dataclass
class Or(Node):
    a: Node
    b: Node

    @property
    def sig(self):
        return f"or({self.a.sig},{self.b.sig})"

    @property
    def reads(self):
        return self.a.reads + self.b.reads

    def render(self, shape, mode):
        return (f"{self.a.render(shape, mode)}, or {self.b.render(shape, mode)}, "
                f"or both")

    def negated(self, shape, mode):
        return f"{self.a.negated(shape, mode)}, and {self.b.negated(shape, mode)}"

    def p(self, ctx):
        pa, pb = self.a.p(ctx), self.b.p(ctx)
        return pa + pb - pa * pb

    def ambiguous(self, ctx):
        return self.a.ambiguous(ctx) or self.b.ambiguous(ctx)

    def bounds(self, ctx):
        (al, ah), (bl, bh) = self.a.bounds(ctx), self.b.bounds(ctx)
        return al + bl - al * bl, ah + bh - ah * bh

@dataclass
class Not(Node):
    a: Node

    @property
    def sig(self):
        return f"not({self.a.sig})"

    @property
    def reads(self):
        return self.a.reads

    def render(self, shape, mode):
        return self.a.negated(shape, mode)

    def negated(self, shape, mode):
        return self.a.render(shape, mode)

    def p(self, ctx):
        return 1.0 - self.a.p(ctx)

    def ambiguous(self, ctx):
        return self.a.ambiguous(ctx)

    def bounds(self, ctx):
        lo, hi = self.a.bounds(ctx)
        return 1.0 - hi, 1.0 - lo
