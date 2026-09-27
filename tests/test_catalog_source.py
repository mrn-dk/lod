"""Row 49: many-option matching and linking (catalog, entity linking, link choice, routing)."""

from __future__ import annotations

import random
import re
from collections import Counter, defaultdict

import pytest

from lod.schema import Example
from lod.corpus.services.sources.real.base import family_of
from lod.corpus.services.sources.synth import catalog as C


@pytest.fixture(scope="module")
def tasks():
    return C.tasks()


def _replay(task, n):
    return list(C.instances(*task.load.spec[:4], n, task.load.spec[4]))


@pytest.fixture(scope="module")
def sample(tasks):
    """(task, example, instance) for every task: 25 each, 6 for the large-N tasks."""
    out = []
    for t in tasks:
        for ex, inst in _replay(t, 6 if t.name.endswith("_largeN") else 25):
            out.append((t, ex, inst))
    return out


@pytest.fixture(scope="module")
def train_sample(tasks):
    out = []
    for t in tasks:
        if t.force_split == "train":
            out += [(t, ex, inst) for ex, inst in _replay(t, 80)]
    return out


@pytest.fixture(scope="module")
def tok():
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
    except Exception as exc:                     # no cache on this machine
        pytest.skip(f"tokenizer unavailable: {exc}")


# ---- shape of the output --------------------------------------------------------------

def test_every_task_yields_what_it_is_asked_for(tasks):
    for t in tasks:
        got = list(t.load(4))
        assert len(got) == 4, f"{t.name} produced {len(got)}"
        assert all(isinstance(e, Example) and e.task == t.name for e in got)


def test_the_loaders_stop_at_their_caps(tasks):
    caps = {"train": C.TRAIN_CAP, "twin": C.TWIN_CAP, "held": C.HELDOUT_CAP,
            "large": C.LARGE_CAP}
    assert 14 * caps["train"] >= 25_000 and 14 * caps["twin"] >= 1_000
    assert 4 * caps["held"] >= 2_500 and 4 * caps["large"] >= 300
    twin = next(t for t in tasks if t.name.endswith("_eval"))
    assert len(list(twin.load(10_000))) == C.TWIN_CAP


def test_targets_and_options_are_well_formed(sample):
    for t, ex, _ in sample:
        q = ex.questions[0]
        assert isinstance(q.target, int) and 0 <= q.target < len(q.options)
        assert len({o.lower() for o in q.options}) == len(q.options), t.name
        if q.descriptions is not None:
            assert len(q.descriptions) == len(q.options)
        assert q.question and q.instructions
        assert len(q.options) >= 2


def test_generation_is_seeded(tasks):
    for t in tasks[:3] + [t for t in tasks if t.name.endswith("_largeN")][:1]:
        a = [e.to_dict() for e in t.load(4)]
        b = [e.to_dict() for e in t.load(4)]
        assert a == b, t.name


def test_tasks_are_generated_row_49_with_per_example_options(tasks):
    for t in tasks:
        assert t.row == 49 and not t.real and t.per_example_options
        assert t.name.startswith("catalog_")
        assert t.force_split in ("train", "devreal", "testreal"), t.name


# ---- the label -----------------------------------------------------------------------

def test_exactly_one_entry_satisfies_every_constraint(sample):
    for t, ex, inst in sample:
        q = ex.questions[0]
        sat = [i for i, c in enumerate(inst.cells)
               if all(con.ok(c[con.attr.name]) for con in inst.cons)]
        assert sat == [q.target], f"{t.name}: {sat} vs {q.target}"


def test_the_options_print_the_entries_they_stand_for(sample):
    """The description (or, for URL keys, the key) is rendered from the same cell the
    label was computed on, in the same order."""
    for t, ex, inst in sample:
        q = ex.questions[0]
        dom = inst.dom
        for i, cell in enumerate(inst.cells):
            if dom.key == "url":
                assert q.options[i] == C._url(dom, cell)
            if dom.desc:
                assert q.descriptions[i] == C._desc(dom, cell)


def test_a_threshold_check_by_hand():
    a = C.BACKPACKS.attr("capacity")
    con = C.Con(a, ">", t=30)
    assert not con.ok((30, 0)) and con.ok((32, 0))
    w = C.BACKPACKS.attr("weight")
    con = C.Con(w, "<=", t=1.0)
    assert con.ok((2.2, 1)) and not con.ok((2.4, 1))          # 0.998 kg, 1.089 kg
    assert C.Con(w, "range", t=1.0, hi=1.2).ok((1.2, 0))


def test_the_state_carries_every_constraint(sample):
    """Each constraint is stated: its threshold, its value, its alias, or its qualifier.
    An answer the state does not determine would be a guess."""
    decade = re.compile(r"(early|mid|late|the) (\d{4})s")
    for t, ex, inst in sample:
        st = ex.state
        for con in inst.cons:
            a = con.attr
            if a.kind == "flag":
                assert (a.need if con.op == "need" else a.avoid) in st or \
                    "required" in st or "must" in st or "wanted" in st, (t.name, a.name)
            elif a.kind == "num":
                ts = [C.tshow(a, x) for x in con.numbers()]
                sym = any(x in st for x in ts)
                dec = decade.search(st) is not None
                assert sym or dec, (t.name, a.name, ts, st)
            else:
                words = [str(v) for v in ([con.t] if con.op != "in" else con.vals)]
                if not con.literal:
                    words = [C.alias_of(a, v) for v in ([con.t] if con.op != "in"
                                                        else con.vals)]
                assert all(w in st for w in words), (t.name, a.name, words, st)


# ---- hard negatives ------------------------------------------------------------------

def test_hard_negatives_fail_exactly_one_constraint(sample):
    n_hn = 0
    for t, ex, inst in sample:
        q = ex.questions[0]
        assert q.meta["hard_neg"] == inst.hn
        assert len(q.meta["hn_kind"]) == len(inst.hn)
        gold = inst.cells[q.target]
        for i in inst.hn:
            assert len(inst.fails(i)) == 1, (t.name, i, inst.fails(i))
            diff = [a for a in inst.dom.attrs if inst.cells[i][a.name] != gold[a.name]]
            assert len(diff) == 1 and diff[0].name == inst.fails(i)[0], t.name
            n_hn += 1
    assert n_hn > 1000


def test_every_kind_of_minimal_edit_occurs(train_sample):
    kinds = Counter(k for _, ex, _ in train_sample for k in ex.questions[0].meta["hn_kind"])
    for k in ("value", "synonym", "drop", "add", "unit"):
        assert kinds[k] > 0, kinds


def test_hard_negative_share(train_sample):
    """A grid, not a star (see the module docstring): the share of one-edit neighbours in
    the gold's block falls with N -- ~0.3 at N <= 10, ~0.1 past 100 -- and near misses
    (any entry failing exactly one constraint) are a larger share still."""
    shares = defaultdict(list)
    for _, ex, _ in train_sample:
        g = ex.questions[0].meta["gen"]
        shares["small" if g["n"] <= 10 else "large" if g["n"] > 100 else "mid"].append(
            g["hn_share"])
    mean = {k: sum(v) / len(v) for k, v in shares.items()}
    allv = [x for v in shares.values() for x in v]
    assert mean["small"] > 0.25 and mean["large"] < mean["small"]
    assert 0.12 < sum(allv) / len(allv) < 0.45, mean
    near = [ex.questions[0].meta["gen"]["near_share"] for _, ex, _ in train_sample]
    assert sum(near) / len(near) > sum(allv) / len(allv)


# ---- held-out structure --------------------------------------------------------------

def test_splits_hold_out_whole_domains(tasks):
    fam_split = defaultdict(set)
    dom_split = defaultdict(set)
    for t in tasks:
        fam_split[family_of(t)].add(t.force_split)
        for d in t.load.spec[1]:
            if not t.name.endswith("_eval"):
                dom_split[d.name].add(t.force_split)
    for fam, splits in fam_split.items():
        # a trained domain's `_eval` twin is in-structure evaluation, the one exception
        assert splits in ({"train", "devreal"}, {"devreal"}, {"testreal"}, {"train"}), \
            (fam, splits)
    for d, splits in dom_split.items():
        assert len(splits) == 1, (d, splits)
    for fam in C.FAMILIES:
        tr, dv, te = (set(C.TRAINED[fam]), set(C.DEV_HELDOUT[fam]), set(C.TEST_HELDOUT[fam]))
        assert tr and dv and te
        assert not (tr & dv) and not (tr & te) and not (dv & te)


def test_twins_carry_their_trained_siblings_family(tasks):
    names = {t.name for t in tasks}
    for t in tasks:
        if t.name.endswith("_eval"):
            assert t.force_split == "devreal"
            assert family_of(t) == t.name[: -len("_eval")] in names


def test_dev_holds_out_families_that_never_train(tasks):
    trained = {family_of(t) for t in tasks if t.force_split == "train"}
    dev_ood = [t for t in tasks if t.force_split == "devreal" and family_of(t) not in trained]
    test = {family_of(t) for t in tasks if t.force_split == "testreal"}
    assert {t.name.split("_")[1] for t in dev_ood} == set(C.FAMILIES)
    assert not ({family_of(t) for t in dev_ood} & test)
    assert not (test & trained)


def test_every_family_trains(tasks):
    fams = {t.name.split("_")[1] for t in tasks if t.force_split == "train"}
    assert fams == set(C.FAMILIES)


def test_twins_do_not_repeat_training_states(tasks):
    for t in tasks:
        if not t.name.endswith("_eval"):
            continue
        sib = next(x for x in tasks if x.name == t.name[: -len("_eval")])
        a = {e.state for e in sib.load(200)}
        assert not a & {e.state for e in t.load(72)}, t.name


# ---- option count --------------------------------------------------------------------

def test_option_count_is_heavy_tailed(train_sample):
    ns = [len(ex.questions[0].options) for _, ex, _ in train_sample]
    share = lambda lo, hi: sum(lo <= n <= hi for n in ns) / len(ns)  # noqa: E731
    assert share(2, 10) > 0.20, share(2, 10)
    assert share(11, 100) > 0.30, share(11, 100)
    assert share(101, 512) > 0.10, share(101, 512)
    assert max(ns) >= 120
    # log-uniform below the budget's cut: 2-4 about as common as 5-10 and as 11-25
    assert 0.5 < share(2, 4) / share(11, 25) < 2.0


def test_large_n_tasks_are_large_test_only_and_tagged(tasks, sample):
    large = [t for t in tasks if t.name.endswith("_largeN")]
    assert {t.name.split("_")[1] for t in large} == set(C.FAMILIES)
    assert all(t.force_split == "testreal" for t in large)
    for t, ex, _ in sample:
        m = ex.questions[0].meta
        if t.name.endswith("_largeN"):
            assert m.get("large_n") is True
            assert C.LARGE_MIN <= len(ex.questions[0].options) <= C.LARGE_MAX
        else:
            assert "large_n" not in m


def test_meta_gen_records_the_draw(sample):
    for t, ex, inst in sample:
        g = ex.questions[0].meta["gen"]
        assert g["family"] == t.name.split("_")[1]
        assert g["domain"] == inst.dom.name
        assert g["n"] == len(ex.questions[0].options)
        assert g["k"] == len(inst.cons) >= 2
        assert len(g["attrs"]) == len(g["ops"]) == len(g["cases"]) == g["k"]
        assert 0.0 <= g["hn_share"] <= g["near_share"] <= 1.0


# ---- the budget ----------------------------------------------------------------------

def test_every_example_fits_the_training_budget(tok, tasks):
    """Packer at the training settings keeps every non-large-N example whole: nothing
    dropped, no state truncated."""
    from lod.model.packing import Packer
    pk = Packer(tok, max_state_tokens=C.MAX_STATE_TOKENS,
                max_total_tokens=C.MAX_TOTAL_TOKENS)
    worst = 0
    for t in tasks:
        if t.name.endswith("_largeN"):
            continue
        for ex in t.load(12):
            p = pk.pack(ex)
            assert p is not None, f"{t.name}: dropped by the packer"
            st = len(tok("State:\n" + ex.state, add_special_tokens=False)["input_ids"])
            state_len = sum(1 for b in p.block_ids if b == 0)
            assert state_len >= st, f"{t.name}: state truncated"
            worst = max(worst, len(p.input_ids))
    assert worst <= C.MAX_TOTAL_TOKENS


# ---- shortcuts ---------------------------------------------------------------------------

def _words(s):
    return set(re.findall(r"\d+(?:\.\d+)?|\w+", s.lower()))


def _chance_and(heur, items):
    rng = random.Random(0)
    hit = chance = 0.0
    for _, ex, inst in items:
        q = ex.questions[0]
        opts = [f"{o} {d or ''}" for o, d in zip(q.options, q.descriptions or [None] * 999)]
        scores = heur(ex, opts)
        top = max(scores)
        pick = rng.choice([i for i, s in enumerate(scores) if s == top])
        hit += pick == q.target
        chance += 1 / len(q.options)
    return hit / len(items), chance / len(items)


def test_the_gold_is_not_the_option_sharing_most_words_with_the_state(train_sample):
    acc, ch = _chance_and(lambda ex, o: [len(_words(x) & _words(ex.state)) for x in o],
                          train_sample)
    assert acc < ch + 0.05, (acc, ch)


def test_the_gold_is_not_the_option_most_or_least_like_the_rest(train_sample):
    def df_score(sign):
        def f(ex, opts):
            ws = [_words(x) for x in opts]
            df = Counter(w for s in ws for w in s)
            return [sign * sum(df[w] - 1 for w in s) / max(1, len(s)) for s in ws]
        return f
    for sign in (1, -1):
        acc, ch = _chance_and(df_score(sign), train_sample)
        assert acc < ch + 0.05, (sign, acc, ch)


def test_the_gold_is_not_the_longest_or_shortest_option(train_sample):
    for sign in (1, -1):
        acc, ch = _chance_and(lambda ex, o: [sign * len(x) for x in o], train_sample)
        assert acc < ch + 0.05, (sign, acc, ch)


def test_the_gold_position_is_uniform(train_sample):
    ds = [min(9, int(10 * ex.questions[0].target / len(ex.questions[0].options)))
          for _, ex, _ in train_sample if len(ex.questions[0].options) >= 10]
    c = Counter(ds)
    for d in range(10):
        assert 0.06 < c[d] / len(ds) < 0.14, (d, c[d] / len(ds))


def test_no_single_constraint_decides(train_sample):
    """Applying one constraint and guessing among its survivors: no attribute of a domain
    does much better than the mean over the question's constraints, and one constraint
    alone leaves only the answer rarely once N is past a handful."""
    by = defaultdict(lambda: defaultdict(list))
    unique = []
    for _, ex, inst in train_sample:
        mean = []
        for con in inst.cons:
            surv = sum(con.ok(c[con.attr.name]) for c in inst.cells)
            by[inst.dom.name][con.attr.name].append(1 / surv)
            mean.append(1 / surv)
            if inst.n > 10:
                unique.append(surv == 1)
    assert sum(unique) / len(unique) < 0.05
    for dom, per in by.items():
        allv = [x for v in per.values() for x in v]
        avg = sum(allv) / len(allv)
        for a, v in per.items():
            if len(v) >= 40:
                assert sum(v) / len(v) < avg + 0.10, (dom, a, sum(v) / len(v), avg)
