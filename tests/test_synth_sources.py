"""The rule, entity and grounding sources (rows 33-35).

What these pin is not "the code runs" but the three properties the layer exists for: the
state is long enough that the answer is not the whole state, a rule the record cannot
answer is an abstention rather than a `no`, and the families `rule_transfer` measures
were never trained on.
"""

from __future__ import annotations

import collections
import itertools
import random

import pytest

from lod.corpus.repositories import raw_store as store
from lod.paths import RAW_ROOT
from lod.corpus.services.sources.synth import entities, records, rules


def take(task, n):
    return list(itertools.islice(task.load(n), n))


BY_NAME = {t.name: t for t in rules.tasks()}
ENT = {t.name: t for t in entities.tasks()}

# The design rule: "If any gate fails, check state length first. Under 40 tokens is a shortcut."
# Counted with the real tokenizer, because a JSON state is two to three times as many
# tokens as it is whitespace-separated words and the whitespace count says nothing.
MIN_STATE_TOKENS = 40


@pytest.fixture(scope="module")
def tok():
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
    except Exception as exc:                     # no cache on this machine
        pytest.skip(f"tokenizer unavailable: {exc}")


def test_every_held_out_combination_is_made_of_trained_primitives():
    """The whole point of the `rule_transfer` split. If a held-out combination contains a
    primitive the model never saw, its failure says nothing about composition -- it says
    the model was asked something new, which is a different and much weaker question."""
    def prims(spec):
        if isinstance(spec, str):
            return {spec}
        out = set()
        for x in spec[1:]:
            out |= prims(x)
        return out

    trained = set(rules.PRIMITIVES)
    for combo in rules.HELDOUT_COMBOS:
        missing = prims(combo) - trained
        assert not missing, f"{rules.slug(combo)} needs untrained {missing}"
    # and every primitive really is trained, alone
    for prim in rules.PRIMITIVES:
        assert prim in rules.TRAINED_FAMILIES
    assert not set(rules.TRAINED_FAMILIES) & set(rules.HELDOUT_FAMILIES)


def _prims(spec):
    if isinstance(spec, str):
        return {spec}
    return set().union(*(_prims(x) for x in spec[1:]))


def _canon(spec):
    """A signature up to operand order, so `and(a,b)` and `and(b,a)` count as one."""
    if isinstance(spec, str):
        return spec
    op, *rest = spec
    kids = [_canon(x) for x in rest]
    return (op, *(sorted(kids, key=repr) if op in ("and", "or") else kids))


def test_dev_combinations_are_held_out_of_train_and_test_and_built_from_trained():
    """The dev split selects checkpoints and fits T, so it must measure what test
    measures -- an unseen combination of trained primitives -- on combinations test
    never reads."""
    trained = {_canon(s) for s in rules.TRAINED_SPECS}
    test = {_canon(s) for s in rules.HELDOUT_COMBOS}
    # a negated trained combination holds out nothing (an earlier build:
    # not(and(threshold,equals)))
    negated = {_canon(("not", s)) for s in rules.TRAINED_SPECS}
    for combo in rules.DEV_COMBOS:
        c = _canon(combo)
        assert c not in trained | test | negated, rules.slug(combo)
        if combo[0] == "not":
            assert _canon(combo[1]) not in trained | test, rules.slug(combo)
        assert _prims(combo) <= set(rules.PRIMITIVES), rules.slug(combo)
    assert len({_canon(c) for c in rules.DEV_COMBOS}) == len(rules.DEV_COMBOS)
    assert any(rules.needs_records(c) for c in rules.DEV_COMBOS), "a cross-record one"
    assert not set(rules.DEV_FAMILIES) & set(rules.FAMILIES)
    for fam in rules.DEV_FAMILIES:
        assert BY_NAME[f"rule_{fam}"].force_split == "devreal"
        node = rules.build(rules.SPEC_BY_SLUG[fam], records.DOMAINS[0], random.Random(1))
        assert node.sig == rules.sig_of(rules.SPEC_BY_SLUG[fam])


def test_a_signature_is_the_split_key():
    """Task names are slugs of signatures, and a built node reports the signature its
    spec promised. If those drift, `rule_transfer` is measuring a different split than it
    reports."""
    rng = random.Random(0)
    for spec in rules.TRAINED_SPECS + rules.HELDOUT_COMBOS:
        node = rules.build(spec, records.DOMAINS[0], rng)
        assert node.sig == rules.sig_of(spec), (rules.slug(spec), node.sig)


def test_half_the_rules_name_no_field():
    """`indirect` mode is the fix for the reefer probe: every rule of an earlier build
    quoted its key, so locating the evidence was never part of the task."""
    seen = collections.Counter()
    for fam in ("threshold", "and_threshold_equals", "relative"):
        for ex in take(BY_NAME[f"rule_{fam}"], 60):
            for q in ex.questions:
                seen[(q.meta or {}).get("mode")] += 1
    assert seen["indirect"] > 0 and seen["direct"] > 0
    share = seen["indirect"] / (seen["indirect"] + seen["direct"])
    assert 0.35 <= share <= 0.65, share


def test_an_indirect_rule_never_names_the_field():
    rng = random.Random(3)
    d = records.DOMAINS[0]
    for spec in ("threshold", "equals", "relative", "count"):
        node = rules.build(spec, d, rng)
        text = node.render("json", "indirect") + node.negated("json", "indirect")
        for name in node.reads:
            assert f"`{name}`" not in text, (spec, name, text)


def test_comparing_across_records_is_trained():
    """It was an earlier model's worst held-out result at 0.267 -- chance -- because its
    shape appeared nowhere in training."""
    assert "across" in rules.TRAINED_FAMILIES
    assert "across_mean" in rules.TRAINED_FAMILIES
    for ex in take(BY_NAME["rule_across"], 20):
        assert ("record_a" in ex.state) or ("Record A." in ex.state)
        assert ("record_b" in ex.state) or ("Record B." in ex.state)


def test_unit_names_survive_into_the_criteria():
    """str.capitalize() lowercases everything after the first character, so it turned
    "signal (dBm)" into "signal (dbm)" and "line total (EUR)" into "(eur)"."""
    bad = []
    for fam in ("threshold", "range", "converted", "relative"):
        for ex in take(BY_NAME[f"rule_{fam}"], 60):
            if ex.state.startswith("{"):
                continue                      # prose states are the ones that show units
            for q in ex.questions:
                for d in q.descriptions or []:
                    for wrong, right in (("(dbm)", "dBm"), ("(usd)", "USD"),
                                         ("(eur)", "EUR"), ("(mb)", "MB"),
                                         ("(kpa)", "kPa"), ("(mv)", "mV")):
                        if wrong in (d or ""):
                            bad.append((fam, right, d))
    assert not bad, bad[:3]


@pytest.mark.parametrize("family", rules.FAMILIES + rules.DEV_FAMILIES)
def test_every_family_makes_answerable_questions(family):
    examples = take(BY_NAME[f"rule_{family}"], 40)
    assert len(examples) == 40
    labelled = 0
    for ex in examples:
        for q in ex.questions:
            assert q.has_descriptions, "the rule lives in the criteria"
            if q.target is None:
                assert q.meta and q.meta.get("abstain"), \
                    "no target and no abstain flag is a question nobody can score"
            else:
                probs = q.target_probs()
                assert abs(sum(probs) - 1.0) < 1e-6
                labelled += 1
    assert labelled >= 40, "most rows should carry a label"


@pytest.mark.parametrize("name", sorted(BY_NAME) + sorted(ENT))
def test_no_state_is_short_enough_to_be_a_shortcut(name, tok):
    task = {**BY_NAME, **ENT}[name]
    shortest = min(len(tok(ex.state).input_ids) for ex in take(task, 60))
    assert shortest >= MIN_STATE_TOKENS, f"{name}: shortest state {shortest} tokens"


def test_a_missing_field_is_not_in_context_rather_than_no():
    """The failure this prevents: "the record does not say" learned as "the answer is
    no", which is confidently wrong in exactly the case abstention exists for."""
    seen = 0
    for ex in take(BY_NAME["rule_count"], 400):
        for q in ex.questions:
            if q.meta and q.meta.get("abstain"):
                assert q.target is None
                assert q.meta["reason"] == "field_not_in_state"
                seen += 1
    assert seen > 5, "P_MISSING should put some not-in-context rows in every task"


def test_contrastive_pairs_share_a_sequence_and_disagree():
    for name in ("rule_contrast_criteria", "rule_contrast_state"):
        flips = 0
        for ex in take(BY_NAME[name], 60):
            assert len(ex.questions) == 2, "both halves must share one state"
            a, b = (q.target_probs()[1] for q in ex.questions)
            flips += (a < 0.5) != (b < 0.5)
        assert flips >= 55, f"{name}: the pair is only a contrast if the answers differ"


def test_held_out_families_never_reach_train():
    for family in rules.HELDOUT_FAMILIES:
        task = BY_NAME[f"rule_{family}"]
        assert task.force_split == "testreal"
    for family in rules.TRAINED_FAMILIES:
        assert BY_NAME[f"rule_{family}"].force_split == "train"


def test_the_probe_stream_is_never_in_the_corpus():
    """The `rule_transfer` probe loaded `rule_{fam}` from `tasks()` -- the same
    deterministic stream that had already been written into `train.jsonl`. On an earlier
    build, 139 to 147 of every 150 states it scored were verbatim training examples, so
    `acc_trained_families = 0.978` was recall. `probe_tasks()` draws from a seed no
    corpus task uses."""
    probe = {t.name: t for t in rules.probe_tasks()}
    assert set(probe) == {f"rule_{f}" for f in rules.FAMILIES}
    for fam in rules.FAMILIES:
        seen = set()
        for name in (f"rule_{fam}", f"rule_{fam}_eval"):
            if name in BY_NAME:
                seen |= {ex.state for ex in take(BY_NAME[name], 120)}
        fresh = {ex.state for ex in take(probe[f"rule_{fam}"], 120)}
        assert not (fresh & seen), f"{fam}: probe reuses a corpus state"


@pytest.mark.parametrize("family", rules.FAMILIES + rules.DEV_FAMILIES)
def test_no_family_is_answerable_from_its_label_prior(family):
    """`Equals` fires with probability 1/|values|, so `and(x, equals)` answered `no`
    84 % of the time in an earlier build and `or(x, member)` answered `yes` 75 %. The
    gate scores raw accuracy against a 0.70 floor, which those families clear by
    predicting a constant -- a rule nobody has to read is not a rule application."""
    seen = collections.Counter()
    for ex in take(BY_NAME[f"rule_{family}"], 300):
        for q in ex.questions:
            if q.target is None:
                seen["abstain"] += 1
            else:
                seen[q.options[max(range(len(q.target)), key=q.target.__getitem__)]] += 1
    n = sum(seen.values())
    share = seen.most_common(1)[0][1] / n
    assert share <= 0.60, f"{family}: majority class is {share:.3f} of {n}"


def test_a_whole_number_threshold_is_not_a_coin_flip():
    """`soft()` smooths a band around the threshold, which models measurement noise. A
    count of parcels has none: `value == thr` was the only case the band reached for an
    integral field, and it returned 0.5 for "is 4 greater than 4"."""
    ties = total = 0
    for family in rules.FAMILIES:
        for ex in take(BY_NAME[f"rule_{family}"], 120):
            for q in ex.questions:
                if q.target is None or len(q.target) != 2:
                    continue
                total += 1
                ties += abs(q.target[1] - 0.5) < 1e-9
    assert total > 2000
    assert ties / total < 0.005, f"{ties}/{total} targets are an exact coin flip"


def test_the_two_shapes_are_the_same_task():
    """JSON and prose both appear, and the criteria name the field the way the state
    presents it -- otherwise the prose half is a different, harder task."""
    states = [ex.state for ex in take(BY_NAME["rule_threshold"], 80)]
    assert any(s.startswith("{") for s in states)
    assert any(not s.startswith("{") for s in states)
    for ex in take(BY_NAME["rule_threshold"], 60):
        if any((q.meta or {}).get("mode") == "indirect" for q in ex.questions):
            continue            # an indirect rule names no field in either shape
        json_shape = ex.state.startswith("{")
        for q in ex.questions:
            has_key = any("`" in (d or "") for d in q.descriptions)
            assert has_key == json_shape


def test_entity_facts_come_in_three_cases_and_the_gap_is_never_no():
    cases = [ex.questions[0].meta["case"] for ex in take(ENT["entity_fact_train"], 60)]
    assert set(cases) == {"supported", "refuted", "not_in_context"}
    for ex in take(ENT["entity_fact_train"], 60):
        q = ex.questions[0]
        if q.meta["case"] == "not_in_context":
            assert q.target is None and q.meta["abstain"]
            assert len(ex.state) > 120, "the record is never empty"


def test_not_in_context_attribute_count_never_drops_below_six():
    """Row 35 as documented: "the record still carries six or more other attributes". Verify
    the floor directly (not just the >120-char proxy above)."""
    import json as _json
    counts = {"supported": [], "refuted": [], "not_in_context": []}
    for ex in take(ENT["entity_fact_train"], 300):
        state = ex.state
        case = ex.questions[0].meta["case"]
        if state.startswith("{"):
            counts[case].append(len(_json.loads(state)) - 1)  # minus 'entity' key
    assert min(counts["not_in_context"]) >= 6


def test_not_in_context_attribute_count_matches_supported_and_refuted():
    """The not_in_context record drops the asked attribute but backfills a different,
    unused one, so it must carry exactly as many attributes as its supported/refuted
    twin -- not one fewer. A regression here reopens the length shortcut domain 6 found:
    a length-only classifier reached 81% accuracy against a 67% majority baseline when
    the record was silently one attribute short. Pin the
    per-case *distributions*, not just the mean, so a partial regression is caught."""
    import json as _json
    counts = {"supported": [], "refuted": [], "not_in_context": []}
    lens = {"supported": [], "refuted": [], "not_in_context": []}
    for name in ("entity_fact_train", "entity_fact_unseen", "entity_fact_unseen_dev"):
        for ex in take(ENT[name], 900):
            state = ex.state
            case = ex.questions[0].meta["case"]
            lens[case].append(len(state))
            if state.startswith("{"):
                counts[case].append(len(_json.loads(state)) - 1)
    assert counts["supported"], "no json-shape examples sampled"
    assert collections.Counter(counts["not_in_context"]) == collections.Counter(counts["supported"])
    assert collections.Counter(counts["not_in_context"]) == collections.Counter(counts["refuted"])
    # length should now be uninformative: a length-only split should not beat a coin
    # flip's worth of headroom over the majority class by more than a couple of points
    present = lens["supported"] + lens["refuted"]
    absent = lens["not_in_context"]
    majority = max(len(present), len(absent)) / (len(present) + len(absent))
    best_acc = max(
        (sum(1 for x in absent if x < t) + sum(1 for x in present if x >= t))
        / (len(present) + len(absent))
        for t in sorted(set(present) | set(absent))
    )
    assert best_acc <= majority + 0.03, (
        f"length-only threshold reaches {best_acc:.4f} vs majority baseline "
        f"{majority:.4f} -- the length shortcut is back")


def test_g9_entities_were_never_trained_on():
    trained = {ex.state.split(" has ")[0].split(" operates")[0].split(" was ")[0]
               for ex in take(ENT["entity_fact_train"], 120)}
    unseen = {ex.state.split(" has ")[0].split(" operates")[0].split(" was ")[0]
              for ex in take(ENT["entity_fact_unseen"], 120)}
    assert not (trained & unseen)
    assert ENT["entity_fact_unseen"].force_split == "testreal"


def test_the_generator_is_reproducible():
    a = [ex.state for ex in take(BY_NAME["rule_routing"], 20)]
    b = [ex.state for ex in take(BY_NAME["rule_routing"], 20)]
    assert a == b


@pytest.mark.skipif(not RAW_ROOT.is_dir(),
                    reason="raw store not on this machine")
def test_grounding_maps_not_enough_info_to_abstention():
    from lod.corpus.services.sources.real import grounding
    store.configure(RAW_ROOT)
    tasks = {t.name: t for t in grounding.tasks()}
    if not tasks:
        pytest.skip("FEVER/VitaminC not fetched here")
    seen = {"supported": 0, "refuted": 0, "not_in_context": 0}
    for ex in take(tasks["fever_claim"], 300):
        for q in ex.questions:
            seen[q.meta["case"]] += 1
            if q.meta["case"] == "not_in_context":
                assert q.target is None and q.meta["abstain"]
            else:
                assert q.target is not None
    assert all(v > 0 for v in seen.values()), seen


# ---- domain-2 audit -----------------------------------------------------------------
# Five defects the domain-2 audit of row 33 found by re-deriving every target from the
# rendered criteria and the state, independently of the grammar. Each test below pins
# one of them; the counts in the docstrings are measured on an earlier build.

def _leaves(spec):
    return (spec,) if isinstance(spec, str) else tuple(
        l for x in spec[1:] for l in _leaves(x))


def test_a_negated_age_criterion_is_the_complement_of_its_rule():
    """`Age.p` counts whole days and is exact at the boundary, so "more than 30 days
    before" is false at exactly 30 -- and the old negation, "less than 30 days before",
    is false there too. Both branches then read false while one of them is labelled 1.0.
    10 questions in that build (`age`, `or(equals,age)`, `or(range,age)`,
    `and(age,equals)`) shipped with a `no` criterion describing something untrue of the
    record."""
    from datetime import timedelta

    from lod.corpus.services.sources.synth.grammar import Age, Ctx

    stamp = records.DOMAINS[0].stamps[0]
    for older in (True, False):
        node = Age(stamp, 30, older)
        text = node.negated("json", "direct")
        assert "more than" not in text and "less than" not in text, text
        # at exactly the boundary the rule is false and its negation must be true
        rec = {stamp.name: (records.TODAY - timedelta(days=30)).isoformat()}
        assert node.p(Ctx(focus=rec)) == 0.0
        assert ("30 days or fewer" in text) if older else ("30 days or more" in text)


def test_a_routing_rule_only_reads_the_one_record_it_is_shown():
    """A routing state holds one record. The first rule used to be drawn from
    `TRAINED_COMBOS` whole, two of which read every record shown, so 204 of the 2,031
    routing questions in that build -- 10.0 %, and every abstention the family emitted --
    carried a criterion reading "of all the records shown" beside a single-record state
    and were then labelled `not_in_context` naming a field that was in the state."""
    for spec in rules.ROUTING_SPECS:
        assert not rules.needs_records(spec), spec
    abstained = 0
    for ex in take(BY_NAME["rule_routing"], 400):
        for q in ex.questions:
            for text in q.descriptions:
                assert "records shown" not in text, text
            abstained += q.target is None
    assert abstained == 0, f"{abstained} routing questions abstain, and none can"


def test_every_record_in_a_cross_record_state_carries_the_same_fields():
    """`_drop_field` drew the field to remove once per record, which only shows where a
    node reads more than one: 9.6 % of `and(across,relative)` states in that build,
    10.2 % of `and(across_mean,member)` and 8.9 % of `or(across,threshold)` held records
    with different key sets. The records are meant to differ in their values."""
    import json as _json
    ragged = seen = 0
    for family in ("and_across_relative", "and_across_mean_member",
                   "or_across_threshold", "and_across_mean_relative"):
        for ex in take(BY_NAME[f"rule_{family}"], 200):
            if not ex.state.startswith("{"):
                continue
            recs = _json.loads(ex.state)
            if len(recs) < 2 or not all(isinstance(v, dict) for v in recs.values()):
                continue
            seen += 1
            ragged += len({frozenset(v) for v in recs.values()}) > 1
    assert seen > 200
    assert ragged == 0, f"{ragged}/{seen} cross-record states have ragged records"


def test_select_names_the_only_record_that_satisfies_the_rule():
    """Bounding only the winner at p > 0.9 left a runner-up anywhere below it: 50 of
    1,354 `select` questions in that build (3.7 %, and 5 of the 150 `rule_transfer`
    scores) had a second record that literally satisfies the stated rule."""
    seen = 0
    for ex in take(BY_NAME["rule_select"], 200):
        for q in ex.questions:
            if q.target is None:
                continue
            seen += 1
            assert q.meta["runner_up"] < 0.1, (q.meta, q.question)
    assert seen >= 200


def test_the_transfer_comparison_is_matched():
    """`rule_transfer` subtracts a mean over `HELDOUT_FAMILIES` from a mean over
    `TRAINED_FAMILIES`, and on an earlier model that was 0.9601 - 0.8473 = 0.1128 against
    an 0.10 bar. Ten of the twenty-four trained families are a single primitive and no
    held-out family is; `select` is a different shape in a different format rather than a
    combination. Combinations against combinations the drop is 0.0587. These groups exist
    so the gate can compare like with like, and `select` is reported beside the number,
    not in it."""
    assert set(rules.TRAINED_COMBO_FAMILIES) <= set(rules.TRAINED_FAMILIES)
    assert set(rules.HELDOUT_COMBO_FAMILIES) <= set(rules.HELDOUT_FAMILIES)
    assert set(rules.HELDOUT_SHAPE_FAMILIES) <= set(rules.HELDOUT_FAMILIES)
    assert not set(rules.HELDOUT_COMBO_FAMILIES) & set(rules.HELDOUT_SHAPE_FAMILIES)
    assert set(rules.HELDOUT_COMBO_FAMILIES) | set(rules.HELDOUT_SHAPE_FAMILIES) \
        == set(rules.HELDOUT_FAMILIES)
    # no single primitive on either side of the matched comparison
    assert not set(rules.PRIMITIVE_FAMILIES) & set(rules.TRAINED_COMBO_FAMILIES)
    assert not set(rules.PRIMITIVE_FAMILIES) & set(rules.HELDOUT_COMBO_FAMILIES)
    # and both sides are built from the same ten primitives
    for spec in rules.TRAINED_COMBOS + rules.HELDOUT_COMBOS:
        assert set(_leaves(spec)) <= set(rules.PRIMITIVES), spec


def test_every_held_out_difficulty_class_has_a_trained_representative():
    """An earlier build held out `ranking`, whose *shape* -- compare a field across
    records -- appeared nowhere in training, and it scored 0.267, chance.
    `and(across,relative)` repeated that at the level of a difficulty class rather than a
    shape: it pairs a cross-record leaf with a leaf reading two fields of one record, and
    no trained combination did. It scored 0.6223 against a composition residual of
    -0.2833, where the other five held-out combinations sit within 0.06 of the product of
    their own primitives."""
    cross = {"across", "across_mean"}
    two_field = {"relative"}

    def klass(spec):
        lv = _leaves(spec)
        return (any(l in cross for l in lv), any(l in two_field for l in lv))

    trained = {klass(s) for s in rules.TRAINED_COMBOS}
    for spec in rules.HELDOUT_COMBOS:
        assert klass(spec) in trained, (
            f"{rules.slug(spec)} is the only combination of its kind and nothing "
            f"trained shares it -- its score measures the class, not the transfer")


def test_no_question_or_criterion_names_its_own_gold_option():
    """Row 36 once shipped 100 % solvable by exact string match. Row 33's option keys are
    `no`/`yes`, tier and action names, so check the gold key as a *word*: on an earlier
    build it never appears in a question, and reaches 0.71 % of routing criteria only
    because the sensor domain glosses `setpoint_c` as "the temperature it should hold"
    and one action is called `hold`."""
    import re as _re
    leaked = total = 0
    for family in rules.FAMILIES:
        for ex in take(BY_NAME[f"rule_{family}"], 60):
            for q in ex.questions:
                if q.target is None:
                    continue
                total += 1
                g = max(range(len(q.target)), key=q.target.__getitem__)
                blob = q.question + " " + " ".join(d or "" for d in q.descriptions)
                words = set(_re.findall(r"[a-z_][a-z0-9_]*", blob.lower()))
                others = [o.lower() for i, o in enumerate(q.options) if i != g]
                if q.options[g].lower() in words and not any(o in words for o in others):
                    leaked += 1
    assert total > 2000
    assert leaked / total < 0.02, f"{leaked}/{total} questions state their own answer"


def test_the_rule_is_the_criteria_text_everywhere_but_the_shape_control():
    """Row 33 as documented: "the rule is the criteria text, the evidence is the state". Every
    family keeps the rule inside the option's own span, which is the token the option is
    scored at -- except `select`, which puts it in the question and gives each option a
    description naming only a record key. That is why `select`'s 0.54 mixes a new shape
    with a format nothing trained uses, and why it is not in the transfer average."""
    for family in rules.FAMILIES:
        long_question = 0
        for ex in take(BY_NAME[f"rule_{family}"], 40):
            for q in ex.questions:
                assert q.has_descriptions
                long_question += len(q.question) > 90
        if family in rules.HELDOUT_SHAPE_FAMILIES:
            assert long_question > 0, f"{family} is the shape control; the rule is in it"
        else:
            assert long_question == 0, (
                f"{family} carries its rule in the question, not the criteria")


def _vitc_rows(n=400):
    """Synthetic VitaminC rows in VitaminC's own grid: per case_id two claims, each
    human-labelled against each of two revisions, each revision supporting one claim
    and refuting the other (the only shape `vitaminc_contrast` accepts)."""
    out = []
    for i in range(n):
        page = f"Page {i}"
        rev = {1: f"Revision one of page {i} states the value is {i + 5}, "
                  f"which is comfortably above the threshold.",
               2: f"Revision two of page {i} states the value is {i - 5}, "
                  f"which is below the threshold entirely."}
        claims = {1: f"Subject {i} recorded a value above {i}.",
                  2: f"Subject {i} recorded a value below {i}."}
        for c in (1, 2):
            for r in (1, 2):
                out.append({"case_id": f"c{i}", "page": page, "claim": claims[c],
                            "unique_id": f"c{i}_{c}{r}",
                            "label": "SUPPORTS" if c == r else "REFUTES",
                            "evidence": rev[r]})
    return out


def test_entity_fact_eval_pools_are_disjoint_from_each_other_and_from_train():
    """`entity_fact_unseen` (testreal) and `entity_fact_unseen_dev` (devreal) drew from
    one shared `unseen` pool with different seeds, and an earlier build shipped **242 of
    ~480 entity names in both**. Selecting a checkpoint on devreal was then selecting on
    half the names `unseen_entity_abstention` scores. `entity_split` now partitions the
    held-out quarter in two."""
    import json as _json

    def names(task, n):
        out = set()
        for ex in take(ENT[task], n):
            st = ex.state
            out.add(_json.loads(st)["entity"] if st.lstrip().startswith("{")
                    else " ".join(st.split()[:2]))
        return out

    tr = names("entity_fact_train", 600)
    un = names("entity_fact_unseen", 600)
    dv = names("entity_fact_unseen_dev", 600)
    assert not (tr & un), sorted(tr & un)[:5]
    assert not (tr & dv), sorted(tr & dv)[:5]
    assert not (un & dv), f"{len(un & dv)} names in both eval pools: {sorted(un & dv)[:5]}"
    # and the partition is a function of the name alone, for any name
    assert {entities.entity_split(n) for n in (tr | un | dv)} <= {
        "train", "unseen", "unseen_dev"}
    for n in tr:
        assert entities.entity_split(n) == "train"
    for n in un:
        assert entities.entity_split(n) == "unseen"
    for n in dv:
        assert entities.entity_split(n) == "unseen_dev"


def test_entity_fact_names_the_abstention_it_expects():
    """Both option descriptions assert the record *has* the attribute, so on a
    `not_in_context` question neither is true and nothing else in the prompt says a
    third answer exists. `grounding.py` carries that sentence; row 35 shipped 2,758
    abstention questions in an earlier build without it."""
    for name in ("entity_fact_train", "entity_fact_unseen", "entity_fact_unseen_dev"):
        for ex in take(ENT[name], 60):
            q = ex.questions[0]
            assert q.instructions, f"{name}: no instructions on a {q.meta['case']} question"
            assert "not in context" in q.instructions.lower()


def test_entity_fact_is_exactly_an_attribute_lookup_and_nothing_else():
    """Not a leak -- a difficulty ceiling, recorded so it cannot drift silently.

    Measured over all 8,240 row-35 questions in an earlier build: "does the record give this
    attribute exactly this value" separates supported from refuted 5,482 / 5,482, and
    "does the claimed value appear anywhere in the state at all" -- which needs no
    attribute alignment whatsoever -- reached 0.9901 there (1.000 on the generator
    before decoys; ~0.60 now that refuted and not-in-context claims usually name a value
    the record gives another attribute of the same type). What this test pins is the
    *label*: the exact-match oracle must
    stay exactly right, because the day it does not, the generator has started emitting
    a question its own criteria do not answer.
    """
    import json as _json
    checked = 0
    for name in ("entity_fact_train", "entity_fact_unseen", "entity_fact_unseen_dev"):
        for ex in take(ENT[name], 240):
            q = ex.questions[0]
            attr = q.meta["attr"]
            claim = q.question.split("is this claim correct: ", 1)[1].rstrip("?")
            phrase = entities._phrase(attr, "\x00")
            head, _, tail = phrase.partition("\x00")
            value = claim.split(head, 1)[1]
            value = value[:len(value) - len(tail)] if tail else value
            if ex.state.lstrip().startswith("{"):
                rec = _json.loads(ex.state)
                rec.pop("entity", None)
                match = attr in rec and str(rec[attr]) == value
            else:
                match = head + value + tail in ex.state
            case = q.meta["case"]
            assert match == (case == "supported"), (name, case, attr, value, ex.state)
            if case == "not_in_context":
                assert q.target is None
            checked += 1
    assert checked >= 700, checked


# ---- domain 6: the other half, row 34 ------------------------------------------------

def test_vitaminc_contrast_does_not_put_the_answer_in_the_position(monkeypatch):
    """The leak this closes, measured on an earlier build: `pick` walked
    SUPPORTS -> REFUTES -> NOT ENOUGH INFO in a fixed order and every qualifying
    `case_id` has a SUPPORTS row, so revision A was `supported` in **2,000 of 2,000**
    contrast examples and revision B in 0. Answering yes to `claim_a` and no to
    `claim_b` scored 1.0000 over all 2,989 scored questions with the state removed."""
    from lod.corpus.services.sources.real import grounding

    monkeypatch.setattr(grounding.store, "load", lambda key: _vitc_rows())
    examples = list(grounding.vitaminc_contrast(300))
    assert len(examples) == 300

    per_position = [collections.Counter() for _ in range(4)]
    for ex in examples:
        assert [q.id for q in ex.questions] == ["claim1_a", "claim1_b",
                                                "claim2_a", "claim2_b"]
        cases = [q.meta["case"] for q in ex.questions]
        assert sorted(cases) == ["refuted", "refuted", "supported", "supported"], cases
        for i, c in enumerate(cases):
            per_position[i][c] += 1
    for i, counts in enumerate(per_position):
        share = counts["supported"] / sum(counts.values())
        assert 0.35 <= share <= 0.65, (
            f"position {i} is `supported` {share:.1%} of the time -- the selection "
            f"order is readable from the position again")


def test_vitaminc_contrast_swap_keeps_the_state_and_the_label_together(monkeypatch):
    """Swapping the pair must move the evidence with the label, or the contrast is a
    mislabel rather than a shuffle: revision A's text must be the text of the row whose
    label revision A's question carries."""
    from lod.corpus.services.sources.real import grounding

    rows = _vitc_rows(120)
    monkeypatch.setattr(grounding.store, "load", lambda key: rows)
    label = {(grounding.clean(r["claim"]), grounding.clean(r["evidence"])): r["label"]
             for r in rows}

    swapped = 0
    for ex in grounding.vitaminc_contrast(100):
        head, _, tail = ex.state.partition("\n\n")
        texts = {"a": head.split(": ", 1)[1], "b": tail.split(": ", 1)[1]}
        for q in ex.questions:
            text = texts[q.id.rsplit("_", 1)[1]]
            want = "SUPPORTS" if q.meta["case"] == "supported" else "REFUTES"
            assert label[(q.meta["claim"], text)] == want, (q.id, q.meta["case"])
        if ex.questions[0].meta["case"] != "supported":
            swapped += 1
    assert swapped, "no pair was ever swapped"


def test_vitaminc_contrast_order_is_reproducible(monkeypatch):
    """The swap is a hash of the `case_id`, not an rng draw, so two builds of one slice
    agree -- the generator stays as reproducible as it was before the fix."""
    from lod.corpus.services.sources.real import grounding

    monkeypatch.setattr(grounding.store, "load", lambda key: _vitc_rows(60))
    a = [(ex.state, [q.meta["case"] for q in ex.questions])
         for ex in grounding.vitaminc_contrast(50)]
    b = [(ex.state, [q.meta["case"] for q in ex.questions])
         for ex in grounding.vitaminc_contrast(50)]
    assert a and a == b
    assert grounding._ab_order("c7") == grounding._ab_order("c7")
    orders = [grounding._ab_order(f"c{i}") for i in range(200)]
    assert 60 <= sum(orders) <= 140, sum(orders)
