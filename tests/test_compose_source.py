"""Row 47: compositional rules, held out by composition and by depth.

The claims this file holds the generator to:

* the split is a function of the signature: no signature is in two splits, depth 4 never
  trains, and every primitive, quantifier inner form, aggregate, cross-record mode and
  connective appears in training;
* every label re-derives from the *rendered* state (parsed back from JSON, YAML, CSV,
  a markdown table, `key: value` blocks or path lines) by an evaluator written here, which
  shares no code with the generator's;
* an abstention is real: with the removed field filled in, some values make the rule
  true and others false;
* states fit the training budget under the Qwen3 tokenizer.
"""

from __future__ import annotations

import csv
import io
import json
import random
from collections import Counter, defaultdict

import pytest
import yaml

from lod import sentinel
from lod.serialize import flatten_paths, unflatten_paths
from lod.corpus.services.sources.real.base import family_of
from lod.corpus.services.sources.synth import compose as C

N_PER_TASK = 120


@pytest.fixture(scope="module")
def tasks():
    return C.tasks()


@pytest.fixture(scope="module")
def traced(tasks):
    """(task, example, question, state object, format) for a slice of every task."""
    out = []
    for t in tasks:
        C.TRACE = []
        try:
            exs = list(t.load(N_PER_TASK))
            trace = {id(q): (st, fmt) for q, st, _trees, fmt, _mode in C.TRACE}
        finally:
            C.TRACE = None
        for ex in exs:
            for q in ex.questions:
                st, fmt = trace[id(q)]
                out.append((t, ex, q, st.as_obj(), fmt))
    return out


# ---- an evaluator that shares nothing with the generator ------------------------------

def _num(s):
    if isinstance(s, (int, float)) and not isinstance(s, bool):
        return s
    try:
        return int(s)
    except (TypeError, ValueError):
        pass
    try:
        return float(s)
    except (TypeError, ValueError):
        return s


def _rel(a, op, b):
    a, b = float(a), float(b)
    return {">": a > b, "<": a < b, ">=": a >= b, "<=": a <= b,
            "==": a == b, "!=": a != b}[op]


def _pred(s, rec, policy):
    """A record-level rule on one record. Raises KeyError when a field is absent."""
    k = s["k"]
    if k == "cmp":
        return _rel(rec[s["f"]], s["op"], s["c"])
    if k == "eq":
        return (str(rec[s["f"]]) == str(s["v"])) is (s["op"] == "==")
    if k == "range":
        v = float(rec[s["f"]])
        return (float(s["lo"]) <= v <= float(s["hi"])) is s["inside"]
    if k == "member":
        return (str(rec[s["f"]]) in [str(x) for x in s["vals"]]) is s["inside"]
    if k == "fref":
        return _rel(rec[s["a"]], s["op"], rec[s["b"]])
    if k == "pref":
        return _rel(rec[s["f"]], s["op"], policy[s["key"]])
    raise AssertionError(k)


def evaluate(node, obj, meta, idf):
    if "op" in node and "kids" in node:
        v = [evaluate(k, obj, meta, idf) for k in node["kids"]]
        op = node["op"]
        if op == "and":
            return all(v)
        if op == "or":
            return any(v)
        if op == "not":
            return not v[0]
        if op == "implies":
            return (not v[0]) or v[1]
        if op == "xor":
            return sum(v) == 1
        raise AssertionError(op)
    recs = obj[meta["coll"]]
    policy = obj.get(meta["policy"], {})
    me = [r for r in recs if str(r[idf]) == str(meta["focus"])]
    assert len(me) == 1, "the focus record is named exactly once"
    me = me[0]
    k = node["k"]
    if k in ("cmp", "eq", "range", "member", "fref", "pref"):
        return _pred(node, me, policy)
    if k in ("all", "any", "none", "atleast"):
        n_true = sum(_pred(node["phi"], r, policy) for r in recs)
        return {"all": n_true == len(recs), "any": n_true >= 1, "none": n_true == 0,
                "atleast": n_true >= node.get("n", 0)}[k]
    if k == "agg":
        if node["fn"] == "count":
            a = sum(str(r[node["f"]]) == str(node["v"]) for r in recs)
        else:
            xs = [float(r[node["f"]]) for r in recs]
            a = {"sum": sum(xs), "mean": sum(xs) / len(xs), "min": min(xs),
                 "max": max(xs)}[node["fn"]]
            a = round(a, 6)
        return _rel(a, node["op"], node["c"])
    if k == "cross":
        f = node["f"]
        mine = float(me[f])
        if node["mode"] == "named":
            other = [r for r in recs if str(r[idf]) == str(node["other"])]
            assert len(other) == 1 and other[0] is not me
            return _rel(mine, node["op"], other[0][f])
        rest = [float(r[f]) for r in recs if r is not me]
        higher = [x for x in rest if (x > mine if node["hi"] else x < mine)]
        if node["mode"] == "extreme":
            return not higher and mine not in rest
        return len(higher) + 1 <= node["n"]
    raise AssertionError(k)


# ---- parsers for the rendered formats ---------------------------------------------------

def _blocks(text):
    return [b for b in text.split("\n\n") if b.strip()]


def parse_state(text, fmt, meta):
    coll, pname = meta["coll"], meta["policy"]
    if fmt == "json":
        return json.loads(text)
    if fmt == "yaml":
        return yaml.safe_load(text)
    if fmt == "paths":
        return unflatten_paths(text)
    if fmt == "csv":
        out = {}
        for block in _blocks(text):
            head, _, body = block.partition("\n")
            name = head.lstrip("# ").strip()
            rows = list(csv.DictReader(io.StringIO(body)))
            if name == coll:
                out[coll] = [{k: _num(v) for k, v in r.items()} for r in rows]
            else:
                out[name] = {r["setting"]: _num(r["value"]) for r in rows}
        return out
    if fmt == "table":
        out, name = {}, None
        for block in _blocks(text):
            if not block.startswith("|"):
                name = block.rstrip(":").strip()
                continue
            lines = block.split("\n")
            head = [c.strip() for c in lines[0].strip("|").split("|")]
            rows = [[c.strip() for c in ln.strip("|").split("|")] for ln in lines[2:]]
            if name == coll:
                out[coll] = [{h: _num(c) for h, c in zip(head, r)} for r in rows]
            else:
                out[name] = {r[0]: _num(r[1]) for r in rows}
        return out
    if fmt == "kv":
        out = {coll: []}
        for block in _blocks(text):
            head, *lines = block.split("\n")
            kv = dict(ln.split(": ", 1) for ln in lines)
            kv = {k: _num(v) for k, v in kv.items()}
            if head.strip("[]") == pname:
                out[pname] = kv
            else:
                out[coll].append(kv)
        return out
    raise AssertionError(fmt)


def _idf(meta):
    return next(cd for cd in C.CDOMAINS if cd.dom.name == meta["domain"]).idf


# ---- shape -------------------------------------------------------------------------------

def test_tasks_are_registered_on_row_47_with_explicit_splits(tasks):
    assert len(tasks) == 25
    for t in tasks:
        assert t.row == C.ROW == 47 and t.real is False and t.soft is True
        assert t.name.startswith("compose_"), "PROTECTED_PREFIXES key"
        assert t.force_split in ("train", "devreal", "testreal"), t.name
    assert len({t.name for t in tasks}) == len(tasks)


def test_families_are_disjoint_across_splits(tasks):
    by_split = defaultdict(set)
    for t in tasks:
        by_split[t.force_split].add(family_of(t))
    splits = list(by_split)
    for a in splits:
        for b in splits:
            if a < b:
                assert not by_split[a] & by_split[b], (a, b)


def test_every_task_yields_what_it_is_asked_for(tasks):
    for t in tasks:
        assert len(list(t.load(7))) == 7


def test_generation_is_seeded(tasks):
    for t in (tasks[0], tasks[-1]):
        a = [e.to_dict() for e in t.load(12)]
        b = [e.to_dict() for e in t.load(12)]
        assert a == b


def test_targets_are_well_formed(traced):
    for t, ex, q, obj, fmt in traced:
        p = q.target_probs()
        assert p is not None and abs(sum(p) - 1) < 1e-9 and len(p) == len(q.options)
        assert len(set(q.options)) == len(q.options)
        assert len(q.descriptions) == len(q.options) and all(q.descriptions)
        assert q.meta["split"] == t.force_split
        assert q.meta["format"] == fmt and fmt in C.FORMATS


# ---- the split -----------------------------------------------------------------------------

def test_signature_split_is_a_function_of_the_signature():
    rng = random.Random(0)
    seen = {}
    for _ in range(6000):
        d = rng.randint(1, 4)
        sk = C.sample_skeleton(rng, d)
        sig = C.sig_of(sk)
        sp = C.split_of_sig(sig, C.depth_of(sk))
        assert seen.setdefault(sig, sp) == sp
        if d == 1:
            assert sp == "train"
        if d == 4:
            assert sp in ("devreal", "testreal")
    by = Counter(seen.values())
    assert by["devreal"] and by["testreal"] and by["train"]


def test_commutative_children_share_a_signature():
    assert C.sig_of(("and", "cmp", ("or", "agg", "all"))) == \
        C.sig_of(("and", ("or", "all", "agg"), "cmp"))
    assert C.sig_of(("implies", "cmp", "agg")) != C.sig_of(("implies", "agg", "cmp"))


def test_no_signature_is_in_two_splits_and_depth_4_never_trains(traced):
    split_of_sig = {}
    for t, ex, q, obj, fmt in traced:
        sigs = q.meta.get("sigs") or [q.meta["sig"]]
        for s in sigs:
            prev = split_of_sig.setdefault(s, t.force_split)
            assert prev == t.force_split, f"{s} in {prev} and {t.force_split}"
        depths = q.meta.get("depths") or [q.meta["depth"]]
        if t.force_split == "train":
            assert max(depths) <= 3
        else:
            assert min(depths) >= 2
    assert any(q.meta["depth"] == 4 for t, ex, q, o, f in traced if t.force_split == "devreal")
    assert any(q.meta["depth"] == 4 for t, ex, q, o, f in traced if t.force_split == "testreal")


def test_every_primitive_and_connective_is_trained(traced):
    kinds, inner, ops, fns, modes = Counter(), Counter(), Counter(), Counter(), Counter()

    def walk(n):
        if "kids" in n:
            ops[n["op"]] += 1
            for k in n["kids"]:
                walk(k)
            return
        kinds[n["k"]] += 1
        if "phi" in n:
            inner[n["phi"]["k"]] += 1
        if n["k"] == "agg":
            fns[n["fn"]] += 1
        if n["k"] == "cross":
            modes[n["mode"]] += 1

    for t, ex, q, obj, fmt in traced:
        if t.force_split != "train":
            continue
        for tree in (q.meta.get("trees") or [q.meta.get("tree")]):
            if tree:
                walk(tree)
    assert set(kinds) == set(C.KINDS), set(C.KINDS) - set(kinds)
    assert set(inner) == set(C.RECORD_KINDS)
    assert set(ops) == set(C.OPS)
    assert set(fns) == set(C.AGG_FNS)
    assert set(modes) == set(C.CROSS_MODES)
    # and every one of them alone, at depth 1
    solo = {q.meta["sig"] for t, ex, q, o, f in traced
            if t.force_split == "train" and q.meta["depth"] == 1}
    assert solo >= set(C.KINDS)


def test_held_out_templates_never_reach_training(traced):
    held = {name: [t.lstrip("#") for t in ts] for name, ts in C.HELDOUT_TEMPLATES.items()}
    assert sum(len(v) for v in held.values()) >= 40
    for t, ex, q, obj, fmt in traced:
        if t.force_split == "train":
            assert q.meta["tpl"] == "train"
    tpl = Counter(q.meta["tpl"] for t, ex, q, o, f in traced if t.force_split != "train")
    assert tpl["heldout"] > 0.3 * sum(tpl.values()) and tpl["seen"] > 0.2 * sum(tpl.values())


def test_train_text_never_uses_a_held_out_question_template(traced):
    """Checked on the text, for the templates whose rendered form is recognisable."""
    held_q = [t.split("{")[0] for t in C.HELDOUT_TEMPLATES["q_noul"] |
              C.HELDOUT_TEMPLATES["q_route"] if not t.startswith("{")]
    for t, ex, q, obj, fmt in traced:
        if t.force_split == "train":
            for h in held_q:
                if len(h) > 8:
                    assert not q.question.startswith(h), (h, q.question)


# ---- formats ---------------------------------------------------------------------------------

def test_formats_parse_back_to_the_state(traced):
    n = Counter()
    for t, ex, q, obj, fmt in traced:
        if fmt == "prose":
            continue
        got = parse_state(ex.state, fmt, q.meta)
        if fmt in ("json", "yaml", "paths"):
            assert got == obj, fmt
        else:
            # text formats carry strings; numbers read back as numbers, lists as text
            coll = q.meta["coll"]
            assert len(got[coll]) == len(obj[coll])
            for a, b in zip(got[coll], obj[coll]):
                assert set(a) == set(b)
                for k, v in b.items():
                    if not isinstance(v, list):
                        assert _num(a[k]) == v or str(a[k]) == str(v), (fmt, k, a[k], v)
        n[fmt] += 1
    assert set(n) == set(C.FORMATS) - {"prose"}


def test_all_seven_formats_are_used(traced):
    assert {fmt for *_, fmt in traced} == set(C.FORMATS)


def test_flatten_paths_round_trips():
    cases = [{"orders": [{"weight_kg": 14, "flags": ["fragile"], "none": None,
                          "s": "12", "t": "true", "e": [], "x": "a b"}],
              "policy": {"max kg": 3.5, "nested": {}}},
             [1, [2, 3], {"a": "x: y"}], "a: b", 5, [], {}, {"a.b": {"c[0]": '"q'}}]
    for c in cases:
        assert unflatten_paths(flatten_paths(c)) == c
    text = flatten_paths({"orders": [{"weight_kg": 1}, {"weight_kg": 14}]})
    assert "orders[1].weight_kg: 14" in text.split("\n")
    assert all(ln.startswith("orders[") for ln in text.split("\n"))


# ---- labels ------------------------------------------------------------------------------------

def test_labels_agree_with_an_independent_reading_of_the_rendered_state(traced):
    checked = 0
    for t, ex, q, obj, fmt in traced:
        if q.meta.get("abstain") or fmt == "prose":
            continue
        state = parse_state(ex.state, fmt, q.meta)
        idf = _idf(q.meta)
        gold = max(range(len(q.options)), key=q.target.__getitem__)
        if q.meta["qshape"] == "noul":
            v = evaluate(q.meta["tree"], state, q.meta, idf)
            assert q.options[gold] == ("yes" if v else "no"), (q.meta["sig"], q.descriptions)
        else:
            vals = [None if tr is None else evaluate(tr, state, q.meta, idf)
                    for tr in q.meta["trees"]]
            assert sum(v is True for v in vals) == (0 if q.meta.get("none_apply") else 1)
            want = vals.index(True) if True in vals else q.meta["sentinel"]
            assert gold == want
        checked += 1
    assert checked > 1000


def test_prose_labels_agree_with_the_traced_state(traced):
    """Prose is not parsed back; its label is checked against the state it was rendered
    from, and that state's values are checked to appear in the prose."""
    for t, ex, q, obj, fmt in traced:
        if fmt != "prose" or q.meta.get("abstain") or q.meta["qshape"] != "noul":
            continue
        v = evaluate(q.meta["tree"], obj, q.meta, _idf(q.meta))
        assert q.target[q.options.index("yes")] == float(v)
        for r in obj[q.meta["coll"]]:
            assert str(r[_idf(q.meta)]) in ex.state


def test_yes_and_no_are_balanced_per_task(traced):
    by = defaultdict(list)
    for t, ex, q, obj, fmt in traced:
        if q.meta["qshape"] == "noul" and not q.meta.get("abstain"):
            by[t.name].append(q.target[q.options.index("yes")])
    for name, ys in by.items():
        share = sum(ys) / len(ys)
        assert 0.40 <= share <= 0.60, f"{name}: yes {share:.2f} of {len(ys)}"


def test_routing_gold_position_is_not_fixed(traced):
    by_k = defaultdict(Counter)
    for t, ex, q, obj, fmt in traced:
        if q.meta["qshape"] == "route":
            by_k[len(q.options)][max(range(len(q.options)), key=q.target.__getitem__)] += 1
    assert set(by_k) == {2, 3}          # 3 = three rules, or two and a sentinel
    for k, pos in by_k.items():
        n = sum(pos.values())
        if n >= 100:
            assert len(pos) == k and max(pos.values()) / n < 1 / k + 0.12, (k, pos)


# ---- abstention --------------------------------------------------------------------------------

def _completions(obj, meta, missing, rng):
    """Fill the removed field back in: extremes, then random values."""
    cd = next(cd for cd in C.CDOMAINS if cd.dom.name == meta["domain"])
    recs = obj[meta["coll"]]
    if missing.startswith("policy:"):
        key = missing.split(":", 1)[1]
        f = cd.num(key.split("_", 1)[1])
        for v in [f.lo - 1, f.hi + 1] + [rng.uniform(f.lo, f.hi) for _ in range(60)]:
            o = json.loads(json.dumps(obj))
            o.setdefault(meta["policy"], {})[key] = v
            yield o
        return
    f = next(x for x in cd.dom.fields if x.name == missing)
    if isinstance(f, C.Enum):
        pools = [[v] for v in f.values] + [list(f.values)] * 80
    else:
        # every record at one value near a constant of the rule (inside a band, just
        # either side of a threshold), then independent random values
        anchors = _consts(meta) + [float(v) for v in obj.get(meta["policy"], {}).values()] + \
            [float(v) for r in recs for v in r.values()
             if isinstance(v, (int, float)) and not isinstance(v, bool)]
        near = sorted({x + d for x in anchors for d in (-1.5, -1, -0.5, 0, 0.5, 1, 1.5)})
        pools = [[f.lo], [f.hi]] + [[x] for x in near] + [None] * 120
    for pool in pools:
        o = json.loads(json.dumps(obj))
        for r in o[meta["coll"]]:
            if pool is None:
                r[missing] = rng.uniform(f.lo, f.hi)
            else:
                r[missing] = rng.choice(pool)
        yield o


def test_an_abstention_is_undecided_by_what_the_state_holds(traced):
    rng = random.Random(0)
    n = 0
    for t, ex, q, obj, fmt in traced:
        if not q.meta.get("abstain"):
            continue
        assert sentinel.is_sentinel_key(q.options[q.meta["sentinel"]])
        assert q.target[q.meta["sentinel"]] == 1.0
        missing = q.meta["missing"]
        field = missing.split(":", 1)[1] if missing.startswith("policy:") else missing
        if missing.startswith("policy:"):
            assert field not in obj.get(q.meta["policy"], {})
        else:
            assert all(field not in r for r in obj[q.meta["coll"]])
        idf = _idf(q.meta)
        tree = q.meta["tree"] if q.meta["qshape"] == "noul" else \
            q.meta["trees"][q.meta["undecided"]]
        outcomes = {evaluate(tree, o, q.meta, idf) for o in _completions(obj, q.meta, missing, rng)}
        assert outcomes == {True, False}, (q.meta["sig"], missing)
        n += 1
    assert n > 25


def _consts(meta):
    out = []

    def walk(n):
        if isinstance(n, dict):
            for k, v in n.items():
                if k in ("c", "lo", "hi") and isinstance(v, (int, float)):
                    out.append(float(v))
                walk(v)
        elif isinstance(n, list):
            for x in n:
                walk(x)
    walk(meta.get("tree") or meta.get("trees"))
    los = [x for x in out]
    return out + [(a + b) / 2 for a in los for b in los]


def test_sentinel_is_mostly_a_decoy(traced):
    """Its presence must not say "abstain": listed natively, it is the answer about one
    time in five, as the augmentation's own sentinels are."""
    with_s = [q for t, e, q, o, f in traced if "sentinel" in q.meta]
    right = sum(1 for q in with_s if q.target[q.meta["sentinel"]] == 1.0)
    assert 0.10 < right / len(with_s) < 0.30
    train = [q for t, e, q, o, f in traced if "sentinel" in q.meta and t.force_split == "train"]
    for q in train:
        assert sentinel.split_of(q.descriptions[q.meta["sentinel"]]) == "train"


# ---- lengths -----------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def qwen():
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base", local_files_only=True)
    except Exception as exc:          # pragma: no cover - needs a local HF cache
        pytest.skip(f"Qwen3 tokenizer not cached: {exc}")


def _packed_text(ex):
    parts = [ex.state]
    for q in ex.questions:
        parts.append(f"\n\nQuestion: {q.question}\nOptions:")
        for o, d in zip(q.options, q.descriptions):
            parts.append(f"\n- {o}: {d}")
    return "".join(parts)


def test_states_fit_the_budget(qwen, tasks):
    exs = [ex for t in tasks for ex in t.load(40)]
    state = sorted(len(qwen(ex.state).input_ids) for ex in exs)
    total = sorted(len(qwen(_packed_text(ex)).input_ids) for ex in exs)
    p95 = lambda xs: xs[int(0.95 * len(xs))]  # noqa: E731
    assert state[0] >= 40
    assert p95(state) <= 1500, f"p95 state {p95(state)}"
    assert p95(total) <= 3072, f"p95 total {p95(total)}"


def test_an_indirect_rule_never_names_a_field_key(traced):
    """`indirect` names a field only by what it records; locating it is the task."""
    n = 0
    for t, ex, q, obj, fmt in traced:
        if q.meta["mode"] != "indirect":
            continue
        cd = next(cd for cd in C.CDOMAINS if cd.dom.name == q.meta["domain"])
        text = " ".join(d for d in q.descriptions if d)
        for f in cd.dom.fields:
            if "_" in f.name and f.name != cd.idf:
                assert f.name not in text, (f.name, text)
        assert "`" not in text
        n += 1
    assert n > 500


def test_generation_does_not_depend_on_the_hash_seed():
    """Two processes with different PYTHONHASHSEED build the same examples: a set of
    field names iterated anywhere in construction would make every rebuild differ."""
    import os
    import subprocess
    import sys
    code = ("import hashlib, json; from lod.corpus.services.sources.synth import compose as C; "
            "t = [t for t in C.tasks() if t.name == 'compose_dev_d23'][0]; "
            "print(hashlib.sha256(json.dumps([e.to_dict() for e in t.load(150)]).encode()).hexdigest())")
    outs = set()
    for seed in ("1", "2"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        outs.add(subprocess.run([sys.executable, "-c", code], env=env, capture_output=True,
                                text=True, check=True).stdout.strip())
    assert len(outs) == 1


def test_the_row_alone_passes_the_sentinel_audit(tasks):
    """The sentinel audit's six acceptance bars, on compose's training tasks alone,
    after the load-time augmentation it replays. An earlier build of this row failed
    four (sentinel on 80 % of choice questions, right 42 % of the time)."""
    from lod.corpus.services import audit as SA
    from lod.training.data import P_DECOY, P_WITHHELD
    exs = []
    for t in tasks:
        if t.force_split == "train":
            for e in t.load(250):
                e.task = t.name
                exs.append(e)
    r = SA.sentinel_audit(exs, P_WITHHELD, P_DECOY)
    for key, bar in SA.ACCEPTANCE.items():
        assert r[key] is not None and r[key] <= bar, (key, r[key], bar)
