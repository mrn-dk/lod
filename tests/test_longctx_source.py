"""Row 51 (generated half): long states whose deciding evidence sits at a recorded depth.

The properties that make this row worth its tokens: the label is re-derivable from the
state alone, the depth recorded in `meta.gen` is where the evidence actually is (measured
with the backbone's tokenizer), almost no answer can be read from the old 2,304-token
prefix, and dev/test hold out whole domains and formats.
"""

from __future__ import annotations

import csv
import io
import json
import random

import pytest

from lod.corpus.services.sources.synth import longctx as L
from lod.paths import DI_BLOCKLIST

DI = DI_BLOCKLIST


@pytest.fixture(scope="module")
def specs():
    return L.specs()


@pytest.fixture(scope="module")
def sample(specs):
    """Two examples of every task, drawn exactly as the build draws them."""
    out = []
    for s in specs:
        out += [(s, ex) for ex in L.generate(s, 2)]
    return out


def _cell(v: str):
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return v


def _parse_records(state: str, fmt: str, coll: str) -> list[dict]:
    if fmt == "json":
        return json.loads(state)[coll]
    assert fmt == "csv"
    body = state.split("\n", 1)[1]
    return [{k: _cell(v) for k, v in row.items()} for row in csv.DictReader(io.StringIO(body))]


def test_every_task_is_registered_and_yields(specs):
    tasks = L.tasks()
    assert len(tasks) == len(specs) == 33
    for t in tasks:
        assert t.row == 51 and t.name.startswith("longctx_") and not t.real
        assert t.force_split in ("train", "devreal", "testreal") and t.family == t.name
    assert next(iter(tasks[0].load(1)), None) is not None


def test_deterministic(specs):
    for s in (specs[0], specs[-1]):
        a = [(e.state, [q.target for q in e.questions]) for e in L.generate(s, 2)]
        b = [(e.state, [q.target for q in e.questions]) for e in L.generate(s, 2)]
        assert a == b


@pytest.mark.parametrize("fam", L.RECORD_FAMILIES)
@pytest.mark.parametrize("fmt", ["json", "csv"])
def test_record_labels_rederive_from_the_parsed_state(fam, fmt):
    """An independent reading: parse the rendered state, recompute every target."""
    for dom in ("shipment", "order"):
        for i in range(3):
            rng = random.Random(L._seed("t", fam, fmt, dom, i))
            ex = L.record_example(fam, dom, rng, "train", "t", lengths=[3000 + 2000 * i],
                                  fmt=fmt)
            cd = L.CDOMS[dom]
            recs = _parse_records(ex.state, fmt, cd.coll)
            for q in ex.questions:
                assert L.rederive_records(q, recs, cd.idf) == q.target


def test_document_labels_rederive(sample):
    n = 0
    for s, ex in sample:
        if s.kind != "doc":
            continue
        for q in ex.questions:
            m = q.meta
            if s.fam == "log":
                assert L.FINAL[q.target] == L.rederive_log(ex.state, m["target_id"], s.variant)
            elif s.fam == "config":
                got = L.rederive_config(ex.state, m["target_id"], m["key"], s.variant)
                assert L.CONF_KEYS[m["key"]][q.target] == got
            else:
                got = L.rederive_policy(ex.state, m["target_id"], m["amount"], s.variant)
                assert L.ROLES[q.target] == got
            n += 1
    assert n >= 30


def test_depth_is_recorded_and_is_where_the_evidence_is(sample):
    for s, ex in sample:
        starts = L.token_starts(ex.state)
        for q in ex.questions:
            g = q.meta["gen"]
            assert g["state_tokens"] == len(starts)
            assert 0 <= g["depth_tokens"] <= g["state_tokens"]
            assert abs(g["depth_frac"] - g["depth_tokens"] / g["state_tokens"]) < 1e-3
            assert g["family"] == s.fam and g["n_records"] > 0
            if s.fam in ("lookup", "log", "config"):
                # the deciding item (its record, log line or section) starts at the
                # depth: its id is within a line's width of it
                tid = q.meta["target_id"]
                hits = [L.token_at(starts, m.start()) for m in L._id_pattern(tid).finditer(ex.state)]
                assert min(abs(g["depth_tokens"] - h) for h in hits) <= 64


def test_lengths_in_range(sample):
    for s, ex in sample:
        n = ex.questions[0].meta["gen"]["state_tokens"]
        assert 1500 <= n <= L.HARD_CAP, (s.name, n)


def test_answers_mostly_beyond_the_old_prefix(sample):
    """A question answerable from the first 2,304 tokens teaches a 2k model nothing new."""
    qs = [q for _, ex in sample for q in ex.questions]
    early = sum(q.meta["gen"]["depth_tokens"] < L.PREFIX_TOKENS for q in qs)
    assert early / len(qs) < 0.2


def test_splits_hold_out_whole_domains_and_formats(specs):
    by = {}
    for s in specs:
        by.setdefault(s.split, set()).add((s.kind, s.variant))
    assert not by["train"] & by["devreal"]
    assert not by["train"] & by["testreal"]
    assert not by["devreal"] & by["testreal"]
    fams = {}
    for t in L.tasks():
        fams.setdefault(t.family, set()).add(t.force_split)
    assert all(len(v) == 1 for v in fams.values())


def test_no_state_shared_between_tasks(sample):
    seen = {}
    for s, ex in sample:
        assert seen.setdefault(ex.state, s.name) == s.name


def test_di_overlap_zero(sample):
    from lod.corpus.services.decontaminate import Blocklist
    bl = Blocklist.load(DI)
    if bl is None:
        pytest.skip("no DI blocklist")
    kept, dropped = bl.filter([ex for _, ex in sample])
    assert dropped == 0
