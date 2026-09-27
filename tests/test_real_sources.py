"""The corpus plumbing, and the invariants the source loaders must hold.

Everything here runs offline. The loaders themselves need network, so what is tested is
the logic that decides what a corpus *is* -- splits, dedup, shapes, mixing-rule
accounting -- plus the per-source rules that are pure functions of already-fetched rows.
A network test that skips when the network is down tells you nothing on the day it
matters.
"""

import json
from pathlib import Path

import pytest
from lod.paths import RAW_ROOT

from lod.schema import Example, Question
from lod.corpus.services.sources.real import (
    crowd,
    gefs,
    gharchive,
    kalshi,
    manifold,
    nvd,
    spdx,
    spf,
    tabfact,
)
from lod.corpus.services.sources.real import MODULES as REAL_MODULES
from lod.corpus.services.sources.real.table import SPECS
from lod.corpus.services.sources.real.base import (
    RESERVED_TESTREAL_ROWS,
    Accounting,
    dedup_against_train,
    shape_of,
    shingles,
    split_of,
)


# ---- splits ------------------------------------------------------------------

def test_split_is_deterministic_and_roughly_70_15_15():
    names = [f"task_{i}" for i in range(4000)]
    counts = {"train": 0, "devreal": 0, "testreal": 0}
    for n in names:
        counts[split_of(n, row=1)] += 1
    assert counts["train"] / 4000 == pytest.approx(0.70, abs=0.03)
    assert counts["devreal"] / 4000 == pytest.approx(0.15, abs=0.03)
    assert counts["testreal"] / 4000 == pytest.approx(0.15, abs=0.03)
    # same name, same split, every time -- the split must not move between builds
    assert all(split_of(n, 1) == split_of(n, 1) for n in names[:50])


@pytest.mark.parametrize("row", RESERVED_TESTREAL_ROWS)
def test_reserved_families_never_reach_train(row):
    """Rows 11, 13 and 25 are held out whole; no task name may route them elsewhere."""
    assert all(split_of(f"task_{i}", row) == "testreal" for i in range(500))


def test_split_is_by_task_not_by_row():
    """Every question of a task lands in one split, or testreal measures unseen *rows*
    of seen tasks rather than unseen tasks, which is the whole point of the split."""
    assert len({split_of("nvd_baseSeverity", 9) for _ in range(20)}) == 1


# ---- dedup -------------------------------------------------------------------

def test_dedup_drops_exact_and_near_matches():
    shared = "the quick brown fox jumps over the lazy dog and then keeps running onward"
    train = [Example(state=shared, task="t")]
    others = {
        "devreal": [Example(state=shared, task="d")],                      # exact
        "testreal": [Example(state=shared + " forever", task="e"),         # shingle
                     Example(state="a completely different sentence entirely here now",
                             task="f")],
    }
    counts = dedup_against_train(train, others)
    assert counts["devreal_exact"] == 1
    assert counts["testreal_shingle"] == 1
    assert others["devreal"] == []
    assert len(others["testreal"]) == 1


def test_short_states_still_shingle():
    assert shingles("only five words here now")


# ---- shapes ------------------------------------------------------------------

def test_shape_buckets_and_overlap():
    hard_bool = Question("q", "?", ["no", "yes"], target=1)
    assert shape_of(hard_bool) == {"bool"}
    # by design: ordinal may overlap soft and the enum buckets
    soft_ord = Question("q", "?", ["neg", "neutral", "pos"], target=[0.2, 0.5, 0.3])
    assert shape_of(soft_ord, ordinal=True) == {"soft", "ordinal", "enum3-5"}
    assert shape_of(Question("q", "?", [str(i) for i in range(30)], target=0)) == {"enum>20"}


# ---- accounting --------------------------------------------------------------

def _task(row, ordinal=False, meta_p=False, per_example=False, real=True):
    from lod.corpus.services.sources.real.base import RealTask
    return RealTask(row=row, name=f"t{row}", licence="x", url="u", load=lambda n: [],
                    ordinal=ordinal, meta_p=meta_p, per_example_options=per_example,
                    real=real)


def test_accounting_measures_each_rule():
    acc = Accounting()
    ordinal_task = _task(9, ordinal=True)
    plain_task = _task(6)
    for _ in range(10):
        acc.add(ordinal_task, Example(state="s", task="t9", questions=[
            Question("a", "sev?", ["LOW", "HIGH"], target=0)]))
    for _ in range(90):
        acc.add(plain_task, Example(state="s", task="t6", questions=[
            Question("b", "cat?", ["x", "y"], target=1)]))
    assert acc.n_questions == 100
    assert acc.fraction(acc.n_ordinal) == pytest.approx(0.10)
    rules = {name: (measured, ok) for name, measured, _, ok in acc.rules()}
    assert rules["ordinal questions"][1] is True
    # 90 of 100 questions from row 6 must trip its 10 % cap
    cap_rule = next(k for k in rules if k.startswith("max share from one row"))
    assert rules[cap_rule][1] is False


def test_row_caps_follow_spec_v2():
    """Rows 1-3 carry a 15 % cap; every other row 10 %."""
    acc = Accounting()
    assert acc.row_cap(1) == acc.row_cap(2) == acc.row_cap(3) == 0.15
    assert acc.row_cap(4) == acc.row_cap(9) == acc.row_cap(32) == 0.10


def test_worst_row_reports_the_actual_breach():
    """The reported row is the one furthest past *its own* cap, not the largest.

    Row 1 at 14 % is inside its 15 % allowance while row 4 at 11 % is outside its 10 %,
    so a check that merely picked the biggest row would clear the breach and flag the
    compliant one.
    """
    acc = Accounting()
    q = lambda: [Question("a", "?", ["no", "yes"], target=0)]
    for _ in range(14):
        acc.add(_task(1), Example(state="s", task="t1", questions=q()))
    for _ in range(11):
        acc.add(_task(4), Example(state="s", task="t4", questions=q()))
    for row in range(6, 21):                      # 15 rows x 5 = 75, each well inside 10 %
        for _ in range(5):
            acc.add(_task(row), Example(state="s", task=f"t{row}", questions=q()))

    assert acc.n_questions == 100
    assert acc.fraction(acc.questions_by_row[1]) == pytest.approx(0.14)   # <= 0.15, fine
    assert acc.fraction(acc.questions_by_row[4]) == pytest.approx(0.11)   # >  0.10, breach

    row, share, cap = acc.worst_row()
    assert (row, cap) == (4, 0.10)
    assert share == pytest.approx(0.11)
    cap_rule = next(n for n, _, _, _ in acc.rules() if n.startswith("max share from one row"))
    assert dict((n, ok) for n, _, _, ok in acc.rules())[cap_rule] is False


def test_accounting_counts_real_separately():
    acc = Accounting()
    acc.add(_task(9), Example(state="s", task="r", questions=[
        Question("a", "?", ["no", "yes"], target=0)]))
    acc.add(_task(9, real=False), Example(state="s", task="s", questions=[
        Question("a", "?", ["no", "yes"], target=0)]))
    assert acc.n_real_questions == 1 and acc.n_questions == 2


# ---- per-source rules --------------------------------------------------------

def test_nvd_ordinal_flags_follow_spec_v2():
    """Row 9: "all but CWE ordinal". CWE is the one categorical schema."""
    by_name = {t.name: t for t in nvd.tasks()}
    for field in ("attackVector", "attackComplexity", "privilegesRequired",
                  "userInteraction", "confidentialityImpact", "integrityImpact",
                  "availabilityImpact", "baseSeverity"):
        assert by_name[f"nvd_{field}"].ordinal, field
    assert not by_name["nvd_cwe"].ordinal
    assert sum(1 for t in nvd.tasks() if t.ordinal) == 8


def test_nvd_tasks_are_separate_schemas():
    names = [t.name for t in nvd.tasks()]
    assert len(names) == len(set(names)) == 9
    assert all(t.row == 9 and t.licence for t in nvd.tasks())


def test_nvd_top_cwes_orders_by_frequency():
    rows = [{"cwe": "CWE-79"}] * 3 + [{"cwe": "CWE-89"}] * 2 + [{"cwe": None}]
    assert nvd.top_cwes(rows, 2) == ["CWE-79", "CWE-89"]


# ---- domain 7: security advisory triage -----------------------------------------
#
# No literal string-match shortcut and no arity leak here -- each nvd_* task's option
# count is a fixed METRICS constant, never sampled. The real free-label risk is class
# imbalance: majority-class baselines run 44.9%-90.3% per ordinal task, so accuracy on
# these tasks is only meaningful read against that baseline.

def test_nvd_gold_option_rarely_appears_verbatim_in_the_description():
    """Row 36's lesson: a shortcut hiding in the state is not caught by reading the
    code, only by measuring it against real records. Word-boundary search for the gold
    CVSS value inside the description, over the actual cached CVE pool, so a future
    rewrite that starts naming the label verbatim (the way "NETWORK"/"remote" already
    brushes close for attackVector) is caught here rather than found later in a probe.

    Measured on the full cached pool (domain-7 pass): 0.0-15.5% depending on
    the field (`nvd_attackComplexity`/`nvd_confidentialityImpact` 0%, `nvd_attackVector`
    highest at 15.5%). The 0.25 bound below leaves headroom for n=300 sampling noise
    while still catching a row-36-style regression.
    """
    import re
    for t in nvd.tasks():
        if t.name == "nvd_cwe":
            continue  # target is "CWE-NNN"; never expected to appear in prose
        hits = total = 0
        for ex in t.load(300):
            for q in ex.questions:
                gold = q.options[q.target].replace("_", " ").lower()
                total += 1
                if re.search(r"\b" + re.escape(gold) + r"\b", ex.state.lower()):
                    hits += 1
        assert total, t.name
        rate = hits / total
        assert rate < 0.25, f"{t.name}: gold value appears verbatim {rate:.2%} of the time"


def test_nvd_option_count_is_fixed_per_task():
    """The arity leak: a task whose option count varies from example
    to example lets the label be predicted from len(options) alone -- measured at
    P=1.00 on a 4-option `nvd_attackVector` example and 0.00 on a 5-option one (that was
    the sentinel/decoy augmentation applied at load time, not this corpus, but the fix --
    "both paths now emit N options from N" -- should mean this source's own option lists
    never vary either). nvd.py's option lists are literal per-metric constants
    (`METRICS`), never sampled, so every example of one task must carry exactly the same
    option count; this locks that in."""
    for t in nvd.tasks():
        sizes = {len(q.options) for ex in t.load(200) for q in ex.questions}
        assert len(sizes) == 1, f"{t.name}: option count varies across examples {sizes}"


def test_manifold_spread_bins_probabilities():
    rows = [{"p": 0.05}, {"p": 0.55}, {"p": 0.95}, {"p": 1.0}]
    spread = manifold.probability_spread(rows, bins=10)
    assert spread["0.0-0.1"] == 1 and spread["0.5-0.6"] == 1
    assert spread["0.9-1.0"] == 2  # p == 1.0 must land in the last bin, not overflow


def test_manifold_task_declares_meta_p():
    tasks = manifold.tasks()
    if not tasks:
        pytest.skip("no cached manifold rows; run scripts/fetch_data.py --rows 28")
    t = tasks[0]
    assert t.meta_p and t.row == 28 and t.licence


# ---- domain 5: forecasting and probability recovery ---------------------------
#
# Kalshi's snapshot is taken after settlement (kalshi.py's module docstring: measured
# 5.3% of last_price between 10c and 90c, vs 0.0% for bid/ask), so it must never declare
# meta_p -- doing so would hand the `probability_recovery` probe a target it can read off
# the label. GEFS, Manifold and SPF are the corpus's only three sources of a genuine
# pre-outcome probability, and `probability_recovery` is scored only on rows that carry
# one, so this split must hold. Domain-5 verification found the `probability_recovery`
# gate's actual testreal probe is just two task families, 277 rows, because Kalshi's
# held-out families carry no meta.p at all -- so "more Kalshi" cannot move that gate
# until this changes.

def test_kalshi_tasks_never_declare_meta_p():
    tasks = kalshi.tasks()
    if not tasks:
        pytest.skip("no cached kalshi rows")
    assert all(not t.meta_p and t.row == 26 for t in tasks)


def test_gefs_tasks_declare_meta_p():
    tasks = gefs.tasks()
    if not tasks:
        pytest.skip("no cached gefs rows")
    assert all(t.meta_p and t.row == 27 for t in tasks)


def test_spf_tasks_declare_meta_p():
    tasks = spf.tasks()
    if not tasks:
        pytest.skip("no cached spf rows")
    assert all(t.meta_p and t.row == 26 for t in tasks)


# ---- domain 5: the question must decide the answer ------------------------------
#
# Domain-5 verification. An earlier build shipped one wording per GEFS lead, none of them
# written by `gefs.py`: `enrich.apply` overwrites `question` from a model's
# answer keyed on (task, option tuple), and gefs_ is not in its PROTECTED_PREFIXES. One
# of the five wordings it produced -- "Does at least one ensemble member forecast at
# least 1 mm ...?" on gefs_pop_day10, 945 of the `probability_recovery` gate's 1,969
# questions -- asks `any(v >= 1mm)` while `meta.p` encodes `count(v >= 1mm) / 5`. Those
# agree only on a unanimous ensemble: 22.2 % of that task's rows. These tests pin down
# the shape that makes such a rewrite unable to change the answer.

_MEMBERS_MIXED = [0.0, 0.3, 1.2, 6.0, 0.1]


def _row(members, lead=7, rained=1):
    return {"lead": lead, "cycle": "2019010100", "lat": 1.0, "lon": 2.0,
            "members_mm": list(members), "p": sum(1 for v in members if v >= 1.0) / len(members),
            "rained": rained}


def test_gefs_variant_targets_are_the_member_function_they_name():
    """Each variant's target is exactly the function of the five member values its own
    question names -- the fraction at or above a threshold, `any`, or `all`. This is the
    invariant the day-10 wording broke."""
    members = _MEMBERS_MIXED
    got = {q.id: q.target for q in gefs.questions_for(_row(members), 7)}
    assert got["pop_0p2mm"] == [1 - 3 / 5, 3 / 5]     # 0.3, 1.2, 6.0
    assert got["pop_5mm"] == [1 - 1 / 5, 1 / 5]       # 6.0 alone
    assert got["any_member_5mm"] == 1                 # 6.0 >= 5
    assert "all_members_1mm" not in got               # 97.5 % "no": a constant target
    # pop_1mm keeps the verified outcome; transform_data.soften_meta_p turns it into
    # [1-p, p] and moves the draw to meta.outcome, which is what row 27 has always shipped
    assert got["pop_1mm"] == 1
    assert [q for q in gefs.questions_for(_row(members), 7)
            if q.id == "pop_1mm"][0].meta == {"p": 2 / 5}


def test_gefs_meta_p_only_where_the_target_is_a_probability():
    """`meta.p` is what `probability_recovery` scores against, so it may only sit on a
    question whose answer genuinely is a probability. The deterministic any/all variants
    must carry none, and no variant but the verified 1 mm one may claim an outcome."""
    qs = gefs.questions_for(_row(_MEMBERS_MIXED), 7)
    for q in qs:
        p = (q.meta or {}).get("p")
        if q.id.startswith("pop_"):
            assert p is not None and isinstance(q.target, list) or q.id == gefs.VERIFIED
        else:
            assert q.meta is None, f"{q.id}: deterministic variant must not carry meta.p"
        # no fabricated outcome: only the threshold this module actually verified
        assert "outcome" not in (q.meta or {}), q.id


def test_gefs_variants_have_disjoint_option_keys():
    """`enrich.collect_schemas` keys a schema on (task, option tuple) and
    rewrites one question per schema. Two variants of one task sharing an option tuple
    would be handed the same wording, so the all-members question would reach the model
    asking about the probability with the deterministic target still attached."""
    keys = [tuple(v.options) for v in gefs.VARIANTS]
    assert len(set(keys)) == len(keys), keys
    assert len({v.qid for v in gefs.VARIANTS}) == len(gefs.VARIANTS)


def test_gefs_criterion_lives_where_the_rewrite_path_cannot_reach():
    """`apply()` reads back only `question` and `descriptions`, and published
    descriptions win its merge; `instructions` it never touches. So every variant must
    state its decision rule in both, naming its own threshold -- then a reworded question
    cannot change what is being asked."""
    for q in gefs.questions_for(_row(_MEMBERS_MIXED), 7):
        assert q.instructions and len(q.instructions) > 40, q.id
        assert q.descriptions and all(d and d.strip() for d in q.descriptions), q.id
        threshold = [v.threshold for v in gefs.VARIANTS if v.qid == q.id][0]
        mm = f"{threshold:g} mm"
        assert mm in q.question and mm in q.instructions, q.id
        assert any(mm in d for d in q.descriptions), q.id


def test_gefs_questions_are_not_answerable_by_one_rule():
    """The point of the variants: the same state must not give the same answer to every
    question, or the model can score well without reading the question at all."""
    rows = gefs.rows()
    if not rows:
        pytest.skip("no cached gefs rows")
    varied = 0
    sample = rows[:2000]
    for r in sample:
        answers = {q.target[1] if isinstance(q.target, list) else q.target
                   for q in gefs.questions_for(r, r["lead"])}
        varied += len(answers) > 1
    assert varied / len(sample) > 0.4, f"only {varied / len(sample):.1%} of states vary"


def test_gefs_loader_keeps_the_meta_p_question_budget():
    """A quota is counted in examples but the corpus budget is questions, and a gridpoint
    now carries five. The loader divides by the number of probability variants so row 27
    contributes the same number of meta.p questions it always did."""
    rows = gefs.rows()
    if not rows:
        pytest.skip("no cached gefs rows")
    examples = list(gefs._loader(7)(300))
    n_p = sum(1 for e in examples for q in e.questions if (q.meta or {}).get("p") is not None)
    assert n_p == 300, n_p
    assert len(examples) == 100


def test_template_state_is_declared_by_the_record_shaped_sources():
    """`dedup_against_train`'s shingle test reads a rendered record's template as a leak.
    Measured on an earlier build it deleted 1,052 of 2,000 gefs_pop_day10 examples -- dry
    gridpoints first, taking the split's mean meta.p from 0.1595 to 0.2958 against a
    training set left at 0.16 -- and 1,156 of 1,290 spf_recession_h4 examples, leaving
    exactly the 134 whose realised-GDP history was empty. Both sources must say so."""
    for module in (gefs, spf):
        tasks = module.tasks()
        if not tasks:
            continue
        assert all(t.template_state for t in tasks), module.__name__


def test_shingle_exemption_restores_the_forecast_distribution():
    """The exemption is only worth having if it actually realigns the splits: with it,
    the held-out lead's meta.p distribution must match the trained leads' instead of
    being pulled two-fold towards wet."""
    rows = gefs.rows()
    if not rows:
        pytest.skip("no cached gefs rows")
    train = list(gefs._loader(1)(600)) + list(gefs._loader(3)(600))
    test = list(gefs._loader(10)(600))
    def mean_p(examples):
        ps = [q.meta["p"] for e in examples for q in e.questions
              if (q.meta or {}).get("p") is not None]
        return sum(ps) / len(ps)
    before = mean_p(test)
    dedup_against_train(train, {"testreal": test},
                        shingle_exempt={t.name for t in gefs.tasks() if t.template_state})
    assert test, "the exemption must not leave the split empty"
    assert mean_p(test) == pytest.approx(before, abs=1e-9)
    assert abs(mean_p(test) - mean_p(train)) < 0.05


def test_spf_state_names_the_forecaster_when_the_cache_carries_one(monkeypatch):
    """Thirty economists answer one survey with thirty different probabilities. Without
    an id in the state they are thirty contradictory targets on one question; with it the
    target is a function of the state again. An older cache has no id and must still
    load, unchanged."""
    base = {"survey": "1999Q2", "target": "1999Q4", "horizon": 3, "p": 0.15,
            "declined": 0, "history": []}
    named = dict(base, forecaster=84)
    # domain-5 audit: `spread` keeps each (survey, forecaster) pair on one side of the
    # horizon holdout; pin both rows to h3's side so this test is about the state alone
    monkeypatch.setattr(spf, "_part", lambda r: "testreal")
    spf._IN_PROCESS["rows"] = [named, dict(base, p=0.4, forecaster=91)]
    try:
        states = [json.loads(e.state) for e in spf._loader(3)(10)]
        assert sorted(s["forecaster_id"] for s in states) == [84, 91]
        assert len({json.dumps(s) for s in states}) == 2
        spf._IN_PROCESS["rows"] = [base]
        old = json.loads(next(iter(spf._loader(3)(10))).state)
        assert "forecaster_id" not in old
        assert list(old) == ["survey_quarter", "target_quarter", "quarters_ahead",
                             "recent_real_gdp_growth"]
    finally:
        spf._IN_PROCESS.pop("rows", None)


def test_spf_history_is_populated_from_the_in_repo_vintage_file():
    """`recent_real_gdp_growth` is not structurally empty -- an earlier build's eval
    split was emptied by the shingle dedup, not by this loader. Everything it needs is
    `spf_actuals.csv`, which is in the repo."""
    rows = spf.rows()
    if not rows:
        pytest.skip("no cached spf rows")
    h4 = [r for r in rows if r["horizon"] == 4]
    with_history = sum(1 for r in h4 if len(r["history"]) >= 3)
    assert with_history / len(h4) > 0.8, f"{with_history}/{len(h4)}"


# ---- domain 3: affect and safety -----------------------------------------------
#
# crowd.py's whole reason to exist is that a naive column read throws away the
# disagreement: the corpus documentation claims "target is the rater fraction, not the
# majority". Domain-3 verification confirmed this by hand for 20
# goemotions/hatespeech records against the raw agg caches (exact match, no
# rounding), but only ~29% of this domain's questions are genuinely soft --
# most goemotions columns are unanimous "no" across >=3 raters, which is a
# legitimate 0.0/1.0 fraction, not a defect. The one real defect this domain
# found lives in `hf.py`'s generic soft_cols path (not owned by this file):
# momererkoc/social_bias_frames ships one row per *annotator*, and the loader
# emits each annotator's own 0/0.5/1 judgement as if it were the item's rater
# fraction, with no aggregation over the post -- so the same state text reaches
# the corpus more than once under contradictory targets.

def test_loader_emits_exact_rater_fraction_not_rounded():
    """crowd._loader must reproduce frac[col] as [1-p, p] exactly -- no rounding,
    no thresholding to the nearest class. This is the soft-target backbone's core
    invariant; the documented soft target is false the day this stops holding."""
    rows = [
        {"text": "a", "frac": {"x": 0.0}},
        {"text": "b", "frac": {"x": 1.0}},
        {"text": "c", "frac": {"x": 1 / 3}},
        {"text": "d", "frac": {"x": 0.6}},
    ]
    load = crowd._loader(lambda: rows, "x", "t", "q?", index=0, n_tasks=1)
    examples = list(load(10))
    assert len(examples) == 4
    got = [e.questions[0].target for e in examples]
    expected = [[1.0, 0.0], [0.0, 1.0], [1 - 1 / 3, 1 / 3], [0.4, 0.6]]
    for g, e in zip(got, expected):
        assert g == pytest.approx(e)


def test_loader_skips_columns_missing_from_frac():
    """A column absent from an item's frac dict must be dropped, not defaulted to 0 --
    defaulting would fabricate a fraction over raters who never answered."""
    rows = [{"text": "a", "frac": {"other": 0.5}}]
    load = crowd._loader(lambda: rows, "x", "t", "q?", index=0, n_tasks=1)
    assert list(load(10)) == []


# ---- domain 9: entailment and NLI ----------------------------------------------
#
# Domain-9 verification confirmed `text_cols` + `state_for()`
# holds in the built corpus: an earlier build's states for
# stanfordnlp_snli__label and nyu_mll_multi_nli__label are `{"premise": ..., "hypothesis":
# ...}` JSON, no parse-tree column. These tests guard chaosnli.py's own two sharp edges:
# the label_dist column order (the module's docstring warns this silently inverts
# entailment/contradiction if read positionally without checking `label_counter`) and the
# alphaNLI two-option framing, which is a different shape (two candidate endings, not a
# premise/hypothesis pair).

from lod.corpus.services.sources.real import chaosnli


def test_chaosnli_nli_options_order():
    # Verified against the published label_counter in the module docstring:
    # {'c': 20, 'e': 12, 'n': 68} against label_dist [0.12, 0.68, 0.2] for one example --
    # so position 0 is entailment, 1 is neutral, 2 is contradiction.
    assert chaosnli.NLI_OPTIONS == ["entailment", "neutral", "contradiction"]


def test_chaosnli_state_carries_both_spans_for_snli_and_mnli():
    ex = {"premise": "A dog runs.", "hypothesis": "An animal moves."}
    state = chaosnli._state(ex, "snli")
    assert "Premise: A dog runs." in state
    assert "Hypothesis: An animal moves." in state


def test_chaosnli_state_returns_none_when_a_span_is_missing():
    assert chaosnli._state({"premise": "A dog runs.", "hypothesis": ""}, "snli") is None
    assert chaosnli._state({"premise": "", "hypothesis": "An animal moves."}, "mnli") is None


def test_chaosnli_alphanli_state_is_two_candidate_endings_not_a_pair():
    ex = {"obs1": "It rained.", "obs2": "The ground was wet.",
          "hyp1": "A storm passed.", "hyp2": "A pipe burst."}
    state = chaosnli._state(ex, "alphanli")
    assert "Beginning: It rained." in state
    assert "Ending: The ground was wet." in state
    assert "Option 1: A storm passed." in state
    assert "Option 2: A pipe burst." in state



# A second domain-9 pass re-measured the soft targets in a later build and they are
# intact: every one of the 4,593 chaosnli_* targets sums to 1 within 1.1e-16, every
# component is a multiple of 1/100 (100 annotators, exactly), only 1.1 % / 0.0 % / 7.4 %
# of snli / mnli / alphanli are one-hot, and mean entropies are 0.798 / 1.072 / 0.412
# bits -- the documented 0.4-1.1 bits. None of that was *checked* by the loader,
# though: it read `label_dist` positionally and trusted it. `_distribution` now derives
# the target from `label_count` and cross-checks it against both `label_dist` and the
# only field that names its labels, `label_counter`. Re-running the loader over the
# cached release after the change emits the same 1,514 / 1,599 / 1,532 rows with zero
# target changes, so the guard costs nothing and catches a silent inversion.

def _one_chaosnli_example(subset: str, row: dict):
    saved = chaosnli._IN_PROCESS.get(subset)
    chaosnli._IN_PROCESS[subset] = [row]
    try:
        return list(chaosnli._loader(subset)(1))[0]
    finally:
        if saved is None:
            chaosnli._IN_PROCESS.pop(subset, None)
        else:
            chaosnli._IN_PROCESS[subset] = saved


def test_chaosnli_distribution_is_derived_from_the_published_counts():
    row = {"label_count": [12, 68, 20], "label_dist": [0.12, 0.68, 0.2],
           "label_counter": {"c": 20, "e": 12, "n": 68}}
    assert chaosnli._distribution(row, "mnli", 3) == pytest.approx([0.12, 0.68, 0.2])


def test_chaosnli_rejects_a_label_counter_that_disagrees_with_the_position():
    """The docstring's one hand-check, as a per-row assertion. `label_counter` names its
    labels and `label_dist` does not, so a release that swapped entailment and
    contradiction would still yield three plausible-looking floats."""
    inverted = {"label_count": [12, 68, 20], "label_dist": [0.12, 0.68, 0.2],
                "label_counter": {"e": 20, "n": 68, "c": 12}}
    assert chaosnli._distribution(inverted, "snli", 3) is None


def test_chaosnli_rejects_a_dist_that_does_not_match_its_counts():
    row = {"label_count": [12, 68, 20], "label_dist": [0.5, 0.3, 0.2],
           "label_counter": {"c": 20, "e": 12, "n": 68}}
    assert chaosnli._distribution(row, "mnli", 3) is None


def test_chaosnli_enforces_a_rater_floor():
    """crowd.py's floor, applied here too: two opinions are not a distribution. ChaosNLI
    ships 100 for every row, so this drops nothing today -- it is what stops a future
    release with a thinner tail arriving as a soft target that is really a coin."""
    thin = {"label_count": [1, 1, 0], "label_dist": [0.5, 0.5, 0.0],
            "label_counter": {"e": 1, "n": 1}}
    assert chaosnli._distribution(thin, "snli", 3) is None
    assert chaosnli.MIN_RATERS == 3


def test_chaosnli_alphanli_counter_keys_are_the_two_candidates():
    row = {"label_count": [97, 3], "label_dist": [0.97, 0.03],
           "label_counter": {"1": 97, "2": 3}}
    assert chaosnli._distribution(row, "alphanli", 2) == pytest.approx([0.97, 0.03])
    # the nli keys must not be accepted for alphanli, or the mapping is not a mapping
    assert chaosnli._distribution({**row, "label_counter": {"e": 97, "n": 3}},
                                  "alphanli", 2) is None


def test_chaosnli_target_is_a_distribution_not_a_majority():
    """The whole point of row 25. A soft target flattened to one-hot is indistinguishable
    from a hard label, and the built corpus has almost none: 126 of 4,593 questions."""
    row = {"example": {"premise": "A dog runs.", "hypothesis": "An animal moves."},
           "label_count": [30, 70, 0], "label_dist": [0.3, 0.7, 0.0],
           "label_counter": {"e": 30, "n": 70}}
    target = _one_chaosnli_example("snli", row).questions[0].target
    assert isinstance(target, list) and len(target) == 3
    assert sum(target) == pytest.approx(1.0)
    assert max(target) < 1.0                      # not flattened to the majority label
    assert target == pytest.approx([0.3, 0.7, 0.0])


# That build holds 4,593 chaosnli_* questions in `testreal`, 1,514 of them carved into
# `calib.jsonl` (temperature fitting, which is not training) and the remaining 3,079 in
# `testreal_eval.jsonl`. `train.jsonl` and `val.jsonl` hold zero. Checked at the text
# level too: 800 sampled ChaosNLI premises, grepped over the whole of train and val,
# matched nothing, and none of the 3,112 snli/mnli premise-hypothesis pairs appears in
# the row-24 tasks that *are* trained (`stanfordnlp_snli__label`, `multi_nli__genre`).

def test_chaosnli_row_is_reserved_whole_to_testreal():
    assert 25 in RESERVED_TESTREAL_ROWS
    for subset in ("snli", "mnli", "alphanli"):
        assert split_of(f"chaosnli_{subset}", 25) == "testreal"

# ---- domain 11: tabular entailment ---------------------------------------------
#
# An earlier description of row 11 described a seeded generator -- code
# generating claims over cells, an LLM paraphrase step, three render shapes (JSON,
# markdown, damaged CSV). Domain-11 verification found none of that is
# real: `tabfact.py` reads the *published* TabFact claims and *published*
# entailed/refuted labels straight out of a HF mirror, and every one of the 4,000
# rows of an earlier build renders as the same single markdown-pipe-table shape. These tests
# hold the loader to what it actually does, not to the generator once described.
#
# The same pass found the trailing "Answer Format" json-schema block -- a constant
# generative-model instruction, unrelated to the table -- surviving on 97.7% of
# states (3,909/4,000), which is the exact "wasting the token budget on a constant"
# failure the module's own docstring warns about for the *preamble*. Fixed by
# stripping it in `_state()` too; this test is what keeps it stripped.

_TABFACT_ROW = {
    "prompt": [
        {"role": "system", "content": "You are a reasoning model. Think step by step, "
                                       "enclosed within <think> </think>."},
        {"role": "user", "content": (
            "Instruction\nRead the table and decide.\n\n"
            "Table\nTable Title: demo\nTable Content:\n| a | b |\n| --- | --- |\n"
            "| 1 | 2 |\n\n\nStatement\nrow 1 has a value of 1 in column a\n\n\n"
            "Answer Format\nThe final answer should be either \"entailed\" or "
            "\"refuted\" and use the following format:\n```json\n"
            "{\n    \"answer\": \"entailed\" or \"refuted\"\n}\n```"
        )},
    ],
    "reward_model": {"ground_truth": ["entailed"]},
}


def test_tabfact_state_strips_preamble_and_trailing_answer_format():
    state = tabfact._state(_TABFACT_ROW)
    assert state.startswith("Table\n")
    assert "Instruction" not in state
    assert "Answer Format" not in state
    assert "```json" not in state
    assert state.endswith("row 1 has a value of 1 in column a")


def test_tabfact_label_reads_published_ground_truth():
    assert tabfact._label(_TABFACT_ROW) == 1
    assert tabfact._label({"reward_model": {"ground_truth": ["refuted"]}}) == 0
    assert tabfact._label({"reward_model": {"ground_truth": ["unknown"]}}) is None
    assert tabfact._label({}) is None


# A second domain-11 verification measured three more failures on a later build,
# all of them in this file's own loader rather than in the mirror.


@pytest.fixture
def tabfact_tasks():
    """The row-31 tasks, over the real mirror rows. Skipped without a raw store."""
    from lod.corpus.repositories import raw_store as _store
    if not _store.has(tabfact.KEY):
        for root in (RAW_ROOT, Path("data/raw")):
            if (root / "v4-raw" / f"{tabfact.KEY}.jsonl").exists() or \
                    (root / "store" / f"{tabfact.KEY}.jsonl").exists():
                _store.configure(root)
                break
    if not _store.has(tabfact.KEY):
        pytest.skip("no TabFact raw rows; run scripts/fetch_data.py --rows 31")
    ts = tabfact.tasks()
    assert len(ts) == 3
    return ts

def test_tabfact_drops_a_row_whose_claim_would_be_truncated_away():
    """`state[:4000]` cut from the end and the claim is at the end: 55 of 2,000 built
    states (2.75 %) were a bare table with nothing asked about it, on 8 distinct strings,
    2 of which carried both labels. An unanswerable state is dropped, not guessed."""
    table = "Table\nTable Content:\n" + "| a | b |\n" * 900
    oversize = table + tabfact._STATEMENT + "row 1 has a value of 1"
    assert len(oversize) > tabfact.MAX_CHARS
    assert tabfact.fit(oversize) is None
    small = "Table\nTable Content:\n| a | b |" + tabfact._STATEMENT + "row 1 is 1"
    assert tabfact.fit(small) == small


def test_tabfact_never_trims_the_table_under_the_claim():
    """Trimming the table to make room is the other way to lose the answer: the claim
    would survive while the row it is about disappeared, and the published label would
    quietly stop matching the state. Nothing is trimmed at all."""
    table = "Table\nTable Content:\n" + "| a | b |\n" * 900
    out = tabfact.fit(table + tabfact._STATEMENT + "claim")
    assert out is None or out.startswith(table)


def test_tabfact_state_without_a_statement_block_is_dropped():
    assert tabfact.fit("Table\nTable Content:\n| a | b |") is None


def test_tabfact_emits_a_trained_and_an_evaluated_task(tabfact_tasks):
    """One task name is one `split_of` hash draw, and it drew `train`: every one of
    domain 11's 2,000 questions in an earlier build sat in training and none in devreal or
    testreal, so the domain had no eval coverage at all."""
    by_split = {t.force_split for t in tabfact_tasks}
    # domain-11 audit: a devreal task too, so early stopping sees the domain
    assert by_split == {"train", "devreal", "testreal"}
    assert len({t.name for t in tabfact_tasks}) == len(tabfact_tasks)
    assert all(t.row == 31 for t in tabfact_tasks)


def test_tabfact_tasks_share_no_table_and_no_state(tabfact_tasks):
    """550 tables carry the mirror's 4,000 claims, so a row-index slice would put the
    same table on both sides and the shingle dedup would delete the held-out one."""
    seen_tables, seen_states = [], []
    for t in tabfact_tasks:
        ex = list(t.load(300))
        assert ex, t.name
        seen_tables.append({tabfact._table(e.state) for e in ex})
        seen_states.append({e.state for e in ex})
    for i in range(len(seen_tables)):
        for j in range(i + 1, len(seen_tables)):
            assert not seen_tables[i] & seen_tables[j]
            assert not seen_states[i] & seen_states[j]


def test_tabfact_majority_class_is_not_the_answer(tabfact_tasks):
    """The mirror is 69 % entailed and its leading rows 75 %, so reading it in file
    order handed the task a 0.7525 majority baseline -- answer "yes" and stop reading.
    Published TabFact is balanced; the skew is the mirror's."""
    for t in tabfact_tasks:
        ex = list(t.load(200))
        yes = sum(1 for e in ex if e.questions[0].target == 1)
        assert abs(yes - len(ex) / 2) <= 1, (t.name, yes, len(ex))


def test_tabfact_every_built_state_carries_its_claim(tabfact_tasks):
    for t in tabfact_tasks:
        for e in t.load(200):
            assert tabfact._STATEMENT in e.state
            assert len(e.state) <= tabfact.MAX_CHARS
            assert e.state.split(tabfact._STATEMENT, 1)[1].strip()


def test_tabfact_is_a_published_label_reader_not_a_generator():
    # No claim-generation, truth-computation or paraphrase step exists in the module's
    # *code* (as opposed to its docstrings, which discuss the earlier description);
    # `_load` only reads `prompt`/`reward_model` fields already present on the mirror row.
    import inspect
    src = inspect.getsource(tabfact._loader) + inspect.getsource(tabfact._rows) + \
        inspect.getsource(tabfact._state) + inspect.getsource(tabfact._label)
    for absent in ("paraphrase", "generate_claim", "compute_truth", "llm", "openrouter"):
        assert absent not in src.lower()


# ---- domain 8: software project signals (GH Archive issue labels, row 5) -----
#
# An earlier description called this row "the main source of per-example options" -- measured against
# an earlier build, that is false. Every gh_label_<repo> task carries one fixed option list
# (the repo's label vocabulary) shared by every example of that task; the option set only
# differs *between* tasks (repos), which is ordinary per-task schema diversity, not the
# per-example-option shape `Accounting.n_per_example` counts (which only bigbench, row 2,
# sets `per_example_options=True` for -- see `instructions.py`). The corpus-wide count
# confirms it: of ~15,824 questions whose (task, question-id) pair actually shows more
# than one option set across its own examples, 78.6 % are `bb_*` and the rest are
# `synth/rules.py` tasks; zero are `gh_label_*` or `gh_is_bug`. These tests lock in the
# mechanism so the claim cannot silently drift back.

def test_gharchive_label_tasks_are_not_per_example_options():
    """A repo's label vocabulary is fixed for the whole task -- it is per-task diversity,
    not the per-example shape the mixing rule's 5 % floor is measuring."""
    fake = gharchive.RealTask(row=5, name="gh_label_x", licence="CC-BY-4.0", url="u",
                              load=lambda n: [])
    assert fake.per_example_options is False


def test_gharchive_clean_drops_bot_hashes_and_gitalk_but_not_emoji():
    """`_clean` targets auto-generated vocab (long hex ids, the gitalk bot label, blank
    strings) -- it must not also strip emoji-prefixed labels, which are still a
    maintainer's own taxonomy entry (e.g. "🐞 Bug"), just decorated."""
    assert gharchive._clean("bug") == "bug"
    assert gharchive._clean("🐞 Bug") == "🐞 Bug"
    assert gharchive._clean("a" * 16) is None          # bot hash id
    assert gharchive._clean("gitalk") is None
    assert gharchive._clean("   ") is None
    assert gharchive._clean("x" * 40) is None           # over the 34-char cap


def _gh_rows(repo: str, labels: list[str], vocab: list[str]) -> list[dict]:
    """Cache rows with distinct text. The loaders drop an issue whose title and opening
    repeat in its repo once digits are masked (bot and template copies, domain-8 audit), so
    `issue 1` / `issue 2` fixtures would all be dropped as one issue filed nine times."""
    words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel",
             "india", "juliet", "kilo", "lima"]
    tag = repo.replace("/", "")
    return [{"repo": repo, "label": lab, "vocab": vocab,
             "title": f"{words[i]} widget misbehaves in {tag} {words[-1 - i]}",
             "body": f"steps: open the {words[i]} panel of {tag}, press {words[-1 - i]}"}
            for i, lab in enumerate(labels)]


def test_gharchive_label_and_bug_loaders_never_share_state(monkeypatch, tmp_path):
    """The label task and `gh_is_bug` must never share an issue -- and, since the domain-8
    audit, never a repo: `gh_is_bug` draws only from repos too small for a label task, so
    the same repo cannot be trained under one task and evaluated under the other whatever
    the split seed.

    `disjoint_slice()` from `.base` is not used here -- this source predates it and
    implements the same guarantee by hand. This test stands in for that utility's own.
    """
    cache = tmp_path / "gharchive_issues.jsonl"
    vocab = ["bug", "enhancement", "documentation", "question", "wontfix"]
    rows = _gh_rows("o/r", [vocab[i % 4] for i in range(9)], vocab)
    rows += _gh_rows("o/s", ["bug", "enhancement", "bug", "documentation"], vocab)
    cache.write_text("".join(json.dumps(r) + "\n" for r in rows))

    monkeypatch.setattr(gharchive, "_CACHE_FILE", cache)
    monkeypatch.setattr(gharchive, "_IN_PROCESS", {})

    label = list(gharchive._label_loader("o/r")(100))
    bug = list(gharchive._bug_loader()(100))
    assert label and bug
    assert {ex.state for ex in label}.isdisjoint({ex.state for ex in bug})
    assert {ex.questions[0].meta["repo"] for ex in bug} == {"o/s"}
    assert [t.name for t in gharchive.tasks()] == ["gh_label_o_r", "gh_is_bug"]


def test_gharchive_tasks_empty_without_cache(monkeypatch, tmp_path):
    """Importing the module and calling `tasks()` must never touch the network -- a
    missing cache file means an empty registry, not a fetch."""
    monkeypatch.setattr(gharchive, "_CACHE_FILE", tmp_path / "does_not_exist.jsonl")
    assert gharchive.tasks() == []


# ---- domain 8: the is-a-bug target, and what row 29 turned out to be ------------
#
# The domain-8 pass measured row 5 in an earlier build and the schema-breadth
# claim holds: 2,052 `gh_label_*` tasks carry 2,015 distinct option-set tuples (98.2 %),
# 9,723 distinct labels of which 8,443 appear in exactly one repo, and the median
# Jaccard overlap between a repo's vocabulary and GitHub's nine default labels is 0.103.
# These are not 2,000 copies of one task.
#
# Two things in the same row did not hold.

def test_gharchive_is_bug_reads_whole_words_and_refuses_a_negation():
    """`gh_is_bug` is the one row-5 target that is *derived* rather than published, so
    the derivation has to be defensible. The substring rule it replaced said yes to
    "BugFix" (30 issues in the cached window), "not a bug" (23 across its spellings) and
    "debug" (3): 66 issues labelled as bugs by a label that says they are not."""
    assert gharchive._is_bug("bug")
    assert gharchive._is_bug("type: bug")
    assert gharchive._is_bug("kind/bug")
    assert gharchive._is_bug("\N{LADY BEETLE} bug")
    assert gharchive._is_bug("C-bug")
    assert gharchive._is_bug("ios-bugs")              # plural is the same label
    assert gharchive._is_bug("new-regressions")
    assert gharchive._is_bug("crashes")
    assert not gharchive._is_bug("not a bug")
    assert not gharchive._is_bug("Type: Not a bug")
    assert not gharchive._is_bug("not-a-bug")
    assert not gharchive._is_bug("BugFix")
    assert not gharchive._is_bug("debug")
    assert not gharchive._is_bug("debugging")
    assert not gharchive._is_bug("bugsnag")           # a product, not a taxonomy entry


def test_gharchive_is_bug_spans_repos_and_balances_at_a_small_quota(monkeypatch,
                                                                    tmp_path):
    """Taken one repo at a time, a small quota came from 2 repos, 18 `no` to 1 `yes` --
    a binary task with a 94.7 % majority class, measured on an earlier build. Round-robin
    over repos with the two answers interleaved makes a prefix of any length both broad
    and close to balanced."""
    vocab = ["bug", "enhancement", "documentation", "question", "wontfix"]
    rows = []
    for repo in range(8):
        # letters, not digits: the copy detector masks digits, so o/r0..o/r7 would read
        # as one template copied into eight repos
        rows += _gh_rows(f"o/r{'abcdefgh'[repo]}", ["enhancement", "enhancement",
                                                   "documentation", "bug", "enhancement",
                                                   "bug"], vocab)
    cache = tmp_path / "gharchive_issues.jsonl"
    cache.write_text("".join(json.dumps(r) + "\n" for r in rows))
    monkeypatch.setattr(gharchive, "_CACHE_FILE", cache)
    monkeypatch.setattr(gharchive, "_IN_PROCESS", {})

    examples = list(gharchive._bug_loader()(8))
    assert len(examples) == 8
    repos = {ex.questions[0].meta["repo"] for ex in examples}
    assert len(repos) >= 4
    targets = [ex.questions[0].target for ex in examples]
    assert abs(sum(targets) - len(targets) / 2) <= 1


def test_gharchive_is_bug_still_never_shares_a_state_with_a_label_task(monkeypatch,
                                                                       tmp_path):
    """Round-robin must not have cost the disjointness guarantee."""
    vocab = ["bug", "enhancement", "documentation", "question", "wontfix"]
    rows = _gh_rows("o/r", [vocab[i % 4] for i in range(9)], vocab)
    rows += _gh_rows("o/s", ["bug", "enhancement"], vocab)
    cache = tmp_path / "gharchive_issues.jsonl"
    cache.write_text("".join(json.dumps(r) + "\n" for r in rows))
    monkeypatch.setattr(gharchive, "_CACHE_FILE", cache)
    monkeypatch.setattr(gharchive, "_IN_PROCESS", {})

    label_states = {ex.state for ex in gharchive._label_loader("o/r")(100)}
    bug_states = {ex.state for ex in gharchive._bug_loader()(100)}
    assert label_states and bug_states
    assert label_states.isdisjoint(bug_states)


def test_row_29_asks_nothing_the_state_can_answer():
    """The other half of domain 8. `manoelalmeida-io/github-pullrequests` has
    `is_pull_request` False on 534 of 534 cached rows and `repository_url`
    huggingface/datasets on 534 of 534: it is one repo's issues, not PR events, and
    an earlier build asked "What is the current status of this pull request?" over 131 of
    them. The three columns that got through detection -- `state`, `state_reason`,
    `author_association` -- are outcome and identity metadata that an issue body does
    not carry, so they are skipped, and `title` is pinned into the state because the
    longest-string-column rule was dropping it."""
    # Domain-8 audit: with those columns skipped the spec yielded 0 tasks while the
    # catalogue still listed row 29 as covered, so the spec is removed; domain 8 is
    # row 5.
    assert not [s for s in SPECS if s.row == 29]
    assert not [s for s in SPECS if "pullrequests" in s.path]


def test_row_24_snli_declares_the_no_consensus_marker_as_not_a_label():
    """SNLI's -1 is its no-consensus marker. `_keep_value` drops the *rows*, but the
    option set for an explicitly declared `label_cols` is built without consulting it,
    so all 2,000 `stanfordnlp_snli__label` questions in an earlier build carry a fourth
    option "-1" that is the answer in none of them -- and enrichment, handed a
    live-looking option, wrote it a description ("No valid relationship label is
    available for this pair."). The spec says the right thing; `hf.detect_label_cols`
    does not honour it on that branch. This pins the declaration so the fix has something
    to satisfy."""
    snli = next(s for s in SPECS if s.path == "stanfordnlp/snli")
    assert snli.drop_label_re == r"^-1$"
    assert snli.force_split_cols == {"label": "train"}


def test_issue_labels_is_not_wired_into_the_real_registry():
    # `sources/issue_labels.py` is older schema machinery registered only in
    # `sources/__init__.py`'s `REGISTRY`, which the build does not read: the build reads
    # `sources.real.REGISTRY()`, whose `MODULES` (asserted below) holds the actual
    # gh_label_*/gh_is_bug source (`sources/real/gharchive.py`).
    assert "issue_labels" not in [m.__name__.rsplit(".", 1)[-1] for m in REAL_MODULES]
    assert "gharchive" in [m.__name__.rsplit(".", 1)[-1] for m in REAL_MODULES]


def test_tabular_cloze_is_not_wired_into_the_real_registry():
    # `sources/tabular_cloze.py` (masked-cell cloze, not entailment) is likewise
    # registered only in `sources/__init__.py`'s `REGISTRY`; the build's real registry
    # must not include it, or it would become a second source for this domain.
    assert "tabular_cloze" not in [m.__name__.rsplit(".", 1)[-1] for m in REAL_MODULES]


# ---- domain 13: logs and events (LogHub, row 14) -----------------------------
#
# Domain-13 verification measured an earlier build's 2,000 `testreal.jsonl` rows
def test_spdx_scrub_catches_a_name_fragment_not_the_full_string():
    """The EUDatagrid licence's published name is "EU DataGrid Software License", and
    its own body credits "software developed by the EU DataGrid" -- a 2-word fragment
    that the pre-fix scrub (whole-name / whole-id tokens only) let straight through.
    Measured against the built corpus: 66/3903 spdx_licence examples leaked this way."""
    text = "This product includes software developed by the EU DataGrid project."
    out = spdx._scrub(text, "EUDatagrid", "EU DataGrid Software License")
    assert "EU DataGrid" not in out
    assert "[REDACTED]" in out


def test_spdx_scrub_catches_hyphenated_url_form_of_the_name():
    text = 'See http://www.eu-datagrid.org/ for the original licence steward.'
    out = spdx._scrub(text, "EUDatagrid", "EU DataGrid Software License")
    assert "eu-datagrid" not in out.lower()


def test_spdx_scrub_still_redacts_the_full_id_and_name():
    text = "The MIT License grants rights under MIT."
    out = spdx._scrub(text, "MIT", "MIT License")
    assert "MIT" not in out


def test_spdx_leakage_rate_flags_an_unscrubbed_mention():
    ids = ["MIT", "Apache-2.0"]
    leaked = Example(task="spdx_licence", state="Permission under the MIT terms.",
                     questions=[Question("licence", "which?", ids, target=0)])
    clean = Example(task="spdx_licence", state="Permission is hereby granted, [REDACTED].",
                     questions=[Question("licence", "which?", ids, target=0)])
    assert spdx.leakage_rate([leaked], ids) == 1.0
    assert spdx.leakage_rate([clean], ids) == 0.0


def test_spdx_excerpts_skip_the_opening_title_block():
    """SKIP_OPENING_WORDS exists so the excerpt never starts at the licence's own
    self-naming first line ("MIT License", "Apache License\\nVersion 2.0", ...)."""
    body = "word " * 300
    text = "The Foo License " + body
    chunks = spdx.excerpts(text, "Foo", "Foo License", n=3)
    assert chunks
    for c in chunks:
        assert "Foo License" not in c


# Domain-10 verification re-ran the leak hunt the docstring claims to have
# closed, against the states of an earlier build rather than against a hand-written string.
# Two families of "the state names its own answer" survived the name-fragment fix, and
# `leakage_rate` could see neither, because it only ever tested the *option string* --
# the one form the scrub cannot miss. Measured over all 1,956 built `spdx_licence`
# states: 135 (6.90 %) contained an abbreviation prefix of their own SPDX id, and the
# brand word of the name ("Nethack" for NGPL) survived alone because the name-run scrub
# started at length 2. Both are 0 after the fix.

def test_spdx_scrub_catches_an_id_prefix_without_its_version():
    """`CERN-OHL-S-2.0`'s own body says "CERN-OHL" and "CERN"; 9 of its 33 built states
    named the answer that way. Prose and URLs drop the version, the full-id token did
    not, so every `-`-delimited prefix of the id is a stem now."""
    text = "Licensed under CERN-OHL. See the CERN Open Hardware repository."
    out = spdx._scrub(text, "CERN-OHL-S-2.0", "CERN Open Hardware Licence Version 2 - Strongly Reciprocal")
    assert "CERN-OHL" not in out
    assert "CERN" not in out


def test_spdx_scrub_catches_a_lowercase_url_id_form():
    """`GPL-3.0-only` is written "gpl-3.0" in gnu.org URLs: an id prefix plus a version
    tail, which the full-id token missed in both directions."""
    out = spdx._scrub("see https://www.gnu.org/licenses/gpl-3.0.html for the text",
                      "GPL-3.0-only", "GNU General Public License v3.0 only")
    assert "gpl-3.0" not in out.lower()


def test_spdx_scrub_catches_a_single_brand_word_of_the_name():
    """NGPL's published name is "Nethack General Public License" and its body says
    "NetHack" thirty times. Multi-word runs started at length 2, so the one word that
    *is* the answer went through untouched."""
    out = spdx._scrub("our license is intended to give everyone the right to share NetHack.",
                      "NGPL", "Nethack General Public License")
    assert "nethack" not in out.lower()
    assert "[REDACTED]" in out


def test_spdx_scrub_leaves_a_sibling_licence_alone():
    """The scrub must remove the *answer*, not the distractors: "LGPL" and "AGPL" name
    other options in the same 60-way set, and blanking them would delete evidence that
    makes an example harder rather than free."""
    text = "This is not the LGPL, nor the AGPL, nor the Lesser General Public License."
    out = spdx._scrub(text, "GPL-3.0-only", "GNU General Public License v3.0 only")
    assert "LGPL" in out
    assert "AGPL" in out


def test_spdx_scrub_keeps_ordinary_licence_vocabulary():
    """Generic name words carry no identity -- redacting "Public"/"Software"/"License"
    out of every licence body would blank the task instead of the answer."""
    text = ("Permission is hereby granted to any person obtaining a copy of this "
            "software to use it in public and to distribute the source.")
    out = spdx._scrub(text, "OSL-3.0", "Open Software License 3.0")
    assert "software" in out.lower()
    assert "public" in out.lower()
    assert "source" in out.lower()


def test_spdx_stems_stop_at_the_version_component():
    """A version is not a name: `Python-2.0.1` must yield "Python", never "Python-2"."""
    stems = {s.lower() for s in spdx._stems("Python-2.0.1", "Python License 2.0.1")}
    assert "python" in stems
    assert not any(s.startswith("python") and any(c.isdigit() for c in s) for s in stems)


def test_spdx_leakage_rate_sees_an_id_prefix_not_just_the_option_string():
    """The old metric tested `option.lower() in state.lower()` and nothing else, so it
    reported 0.000 on a corpus where 6.90 % of states named their own answer. A check
    that can only see the case that never happens is not a check."""
    ids = ["OCLC-2.0", "MIT"]
    prefix_leak = Example(
        task="spdx_licence",
        state="This Research Public License is offered by OCLC Online Computer Library.",
        questions=[Question("licence", "which?", ids, target=0)])
    assert spdx.leakage_rate([prefix_leak], ids) == 1.0
    name_leak = Example(
        task="spdx_licence", state="give everyone the right to share NetHack.",
        questions=[Question("licence", "which?", ["NGPL", "MIT"], target=0)])
    assert spdx.leakage_rate([name_leak], ["NGPL", "MIT"],
                             {"NGPL": "Nethack General Public License"}) == 1.0
    clean = Example(task="spdx_licence", state="Permission is hereby granted, [REDACTED].",
                    questions=[Question("licence", "which?", ids, target=0)])
    assert spdx.leakage_rate([clean], ids) == 0.0


def test_spdx_built_excerpts_do_not_name_their_own_answer():
    """The end-to-end claim, over every excerpt the module actually builds: no state may
    contain any form of its own licence id. Runs offline from the fetch cache."""
    lid, name = "CERN-OHL-S-2.0", "CERN Open Hardware Licence Version 2 - Strongly Reciprocal"
    text = ("Preamble word " * 80
            + "This is the CERN-OHL-S v2 and the CERN Open Hardware Licence. "
            + "See ohwr.org/cern_ohl_s_v2.txt for CERN-OHL. " + "body word " * 200)
    for chunk in spdx.excerpts(text, lid, name, n=6):
        low = chunk.lower()
        for form in ("cern-ohl-s-2.0", "cern-ohl-s", "cern-ohl", "cern"):
            assert form not in low, (form, chunk)


# ---- domain 7 audit -----------------------------------------------------------------

def test_nvd_local_feed_rows_are_keyed_the_way_the_loaders_read_them(tmp_path,
                                                                    monkeypatch):
    """`download()` has two row builders and they disagreed on the key.

    The API branch keys each metric by its CVSS *field* name (`attackVector`), which is
    also what the `nvd_cves.jsonl` cache on disk holds and what `_metric_loader` reads
    back with `r.get(field)`. The local-JSON-feed branch keyed the same values by the
    short id from `METRICS[0]` (`av`, `ac`, `pr`, ...), so on any machine with a feed in
    `<raw-root>/nvd/` every `r.get("attackVector")` returned None, every row failed the
    `val not in options` guard, and all eight `nvd_<metric>` tasks yielded zero
    examples -- silently, since an empty loader is indistinguishable from a small one.
    Only `nvd_cwe` (which reads `r["cwe"]`) survived.
    """
    import gzip
    import json as _json

    from lod.corpus.repositories import raw_store as store

    feed = tmp_path / "nvd"
    feed.mkdir()
    # one CVE per task slice and then some: the nine row-9 loaders take disjoint slices
    # of the pool, so a one-row feed would starve eight of them for reasons that have
    # nothing to do with the key names this test is about
    payload = {"vulnerabilities": [{"cve": {
        "id": f"CVE-2024-{i:05d}",
        "vulnStatus": "Analyzed",
        "descriptions": [{"lang": "en",
                          "value": f"A remote attacker can read file number {i}."}],
        "weaknesses": [{"description": [{"value": "CWE-22"}]}],
        "metrics": {"cvssMetricV31": [{"cvssData": {
            "attackVector": "NETWORK", "attackComplexity": "LOW",
            "privilegesRequired": "NONE", "userInteraction": "NONE",
            "confidentialityImpact": "HIGH", "integrityImpact": "NONE",
            "availabilityImpact": "NONE", "baseSeverity": "HIGH"}}]},
    }} for i in range(36)]}
    with gzip.open(feed / "nvdcve-2.0-2024.json.gz", "wt", encoding="utf-8") as f:
        _json.dump(payload, f)
    monkeypatch.setattr(store, "RAW_ROOT", tmp_path)

    rows = nvd.download()
    assert len(rows) == 36
    for _, field, _, options, _ in nvd.METRICS:
        assert field in rows[0], f"{field} missing; rows keyed by the short id again?"
        assert rows[0][field] in options, (field, rows[0][field])
    assert rows[0]["cwe"] == "CWE-22"
    # and every loader actually gets an example out of it
    monkeypatch.setattr(nvd, "download", lambda limit=None, **kw: list(rows))
    for t in nvd.tasks():
        assert list(t.load(1)), f"{t.name} yielded nothing from a local feed"


def test_nvd_strips_the_published_cvss_assessment_but_keeps_the_prose():
    """Oracle CPU advisories paste the CVSS assessment into the CVE description.

    Both forms are the answer written out: the base score decides `nvd_baseSeverity`
    against thresholds the option descriptions themselves state, and the vector string
    decides all seven metric questions. `state_of` takes them out and leaves the prose
    the question is meant to be read from."""
    oracle = ("Vulnerability in the Oracle SOA Suite component of Oracle Fusion "
              "Middleware (subcomponent: Fabric Layer). Easily exploitable "
              "vulnerability allows unauthenticated attacker with network access via "
              "HTTP to compromise Oracle SOA Suite. "
              "CVSS 3.0 Base Score 8.2 (Confidentiality, Integrity and Availability "
              "impacts). "
              "CVSS Vector: (CVSS:3.0/AV:N/AC:L/PR:N/UI:R/S:C/C:H/I:H/A:H).")
    got = nvd.state_of(oracle)
    assert "CVSS" not in got, got
    assert "8.2" not in got and "AV:N" not in got, got
    assert got.endswith("compromise Oracle SOA Suite.")
    assert "Easily exploitable vulnerability" in got
    # the bare parenthesised vector, with no "CVSS Vector:" label in front of it
    assert nvd.state_of("A remote attacker can read files. "
                        "(CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:N/A:N)") == \
        "A remote attacker can read files."
    # a description that carries no assessment is returned unharmed
    plain = "Buffer overflow in Sync Breeze Enterprise 10.0.28 allows remote attackers."
    assert nvd.state_of(plain) == plain
    assert nvd.state_of("x" * 5000) == "x" * 4000


@pytest.mark.skipif(not (nvd.CACHE / "nvd_cves.jsonl").exists(),
                    reason="NVD row cache not on this machine")
def test_nvd_no_state_in_the_pool_carries_the_published_cvss_string():
    """The strip above, measured against the whole cached pool rather than a fixture.

    Domain-7 audit: 635 of 21,255 cached CVEs (3.0 %) carry the published assessment in
    their description, and `disjoint_slice` had put 191 of them in `nvd_baseSeverity`'s
    slice -- 9.6 % of that task, where the printed base score agreed with the gold band
    187 times out of 190 (98.4 %). Which task is hit is an accident of the slicing, so
    the bound is over the pool, not over one task."""
    import re
    published = re.compile(r"CVSS:3\.[01]/|CVSS\s*3\.[01]\s*Base\s*Score", re.I)
    rows = nvd.download()
    assert len(rows) > 1000
    raw = sum(1 for r in rows if published.search(r["desc"]))
    assert raw, "no row carries the assessment -- has the pool changed shape?"
    left = [r["id"] for r in rows if published.search(nvd.state_of(r["desc"]))]
    assert not left, f"{len(left)} states still print the published CVSS: {left[:5]}"
    # and stripping never guts a description
    assert not [r["id"] for r in rows
                if len(r["desc"]) >= 40 and len(nvd.state_of(r["desc"])) < 40]


@pytest.mark.skipif(not (nvd.CACHE / "nvd_cves.jsonl").exists(),
                    reason="NVD row cache not on this machine")
def test_nvd_records_how_much_of_privileges_required_is_oracle_boilerplate():
    """Not a leak this file can strip -- a difficulty ceiling this file must not let
    grow. Oracle's CPU template restates the CVSS vector in English ("unauthenticated
    attacker", "low privileged attacker", "highly privileged attacker"), and on
    an earlier build those three keywords fired on 217 of `nvd_privilegesRequired`'s 2,000
    states and were right **217 of 217**. The prose is the advisory's own, so it stays;
    what is pinned is the share, because the day these templated rows dominate the task
    the task has stopped being about reading a vulnerability description."""
    rules = [("unauthenticated attacker", "NONE"),
             ("low privileged attacker", "LOW"),
             ("highly privileged attacker", "HIGH")]
    task = {t.name: t for t in nvd.tasks()}["nvd_privilegesRequired"]
    fired = correct = total = 0
    for ex in task.load(600):
        for q in ex.questions:
            total += 1
            pred = next((v for k, v in rules if k in ex.state), None)
            if pred is None:
                continue
            fired += 1
            correct += q.options[q.target] == pred
    assert total >= 500
    assert fired / total < 0.25, (
        f"{fired / total:.1%} of nvd_privilegesRequired is answered by a three-keyword "
        f"Oracle-template lookup ({correct}/{fired} correct)")


@pytest.mark.skipif(not (nvd.CACHE / "nvd_cves.jsonl").exists(),
                    reason="NVD row cache not on this machine")
def test_nvd_no_common_phrase_is_a_deterministic_label():
    """The verbatim-gold check is necessary and not sufficient, and "decisive" cannot be
    the bar: NVD prose is correlated with the metric by construction, so on the real pool
    `allows remote` covers 55 % of `nvd_attackVector` states at P(NETWORK) = 0.950 and
    `a crafted` covers 33 % of `nvd_userInteraction` at P(REQUIRED) = 0.917. Those are
    the source being readable. What must not exist is a common phrase that is a *label*:
    never wrong. That is the shape a pasted assessment or a rendering artefact takes."""
    import collections as _c
    import re

    def grams(s):
        w = re.findall(r"[a-z]+", s.lower())
        return set(w) | {" ".join(w[i:i + 2]) for i in range(len(w) - 1)}

    for t in nvd.tasks():
        rows = [(grams(ex.state), q.target)
                for ex in t.load(400) for q in ex.questions]
        n = len(rows)
        assert n >= 100, t.name
        doc, joint = _c.Counter(), _c.defaultdict(_c.Counter)
        for g, y in rows:
            for x in g:
                doc[x] += 1
                joint[x][y] += 1
        for x, c in doc.items():
            if c < 0.25 * n:
                continue
            p = max(joint[x].values()) / c
            assert p < 0.995, (
                f"{t.name}: '{x}' covers {c / n:.1%} of states and is the gold label "
                f"{p:.1%} of the time -- that is a label, not a correlation")
