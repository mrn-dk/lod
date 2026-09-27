"""Row 39: entity resolution under controlled corruption.

The two tests that matter most here are `test_character_similarity_does_not_solve_pairs`
and `test_most_similar_roster_entry_does_not_solve_rosters`. Every other property of this
source can be true while the task is still a string-similarity benchmark, which is the
one thing this domain must not be.
"""

from __future__ import annotations

import collections
import difflib
import json
import re

import pytest

from lod import sentinel
from lod.corpus.services.sources.synth import entityres as E


@pytest.fixture(scope="module")
def tasks():
    return E.tasks()


@pytest.fixture(scope="module")
def sample(tasks):
    return [(t, ex) for t in tasks for ex in t.load(90)]


def _gold(q):
    probs = q.target_probs()
    return q.options[max(range(len(probs)), key=probs.__getitem__)]


def test_every_task_yields_what_it_is_asked_for(tasks):
    assert tasks, "row 39 registered no tasks"
    for t in tasks:
        got = list(t.load(20))
        assert len(got) == 20, f"{t.name} produced {len(got)}"
        assert t.row == E.ROW and t.real is False
        assert t.name.startswith("entityres_"), \
            f"{t.name} is outside PROTECTED_PREFIXES and the enricher would rewrite it"


def test_targets_are_well_formed(sample):
    for t, ex in sample:
        for q in ex.questions:
            probs = q.target_probs()
            assert probs is not None and abs(sum(probs) - 1.0) < 1e-9
            assert len(probs) == len(q.options)
            assert len(set(q.options)) == len(q.options), f"{t.name}: duplicate options"
            assert q.has_descriptions and len(q.descriptions) == len(q.options)
            assert q.instructions and "Matching policy" in q.instructions


def test_the_policy_is_the_criteria_not_the_question(sample):
    """Non-negotiable 4: the matching policy is an input, the way `rules.py` puts the
    rule in the criteria. A question that only asks "are these the same" with the policy
    nowhere in the prompt is a different, unanswerable task."""
    for t, ex in sample:
        for q in ex.questions:
            for clause in ("settles the question on its own", "fall back to",
                           "never decide anything", "not evidence either way"):
                assert clause in q.instructions, f"{t.name}: policy missing {clause!r}"


# ---- the held-out structure -----------------------------------------------------------

def test_held_out_families_never_reach_training(tasks):
    for t in tasks:
        if "heldout" in t.name:
            assert t.force_split == "testreal", f"{t.name} in {t.force_split}"
        elif t.name.startswith("entityres_dev_"):
            assert t.force_split == "devreal", f"{t.name} in {t.force_split}"
        else:
            assert t.force_split in ("train", "devreal")


def test_held_out_families_are_new_combinations_of_trained_operations(tasks):
    """The `rule_transfer` pattern. A drop on a held-out family has to mean "could not
    compose operations it has", not "met an operation it has never seen"."""
    trained_ops = set().union(*(set(f) for f in E.TRAINED_FAMILIES))
    for family in E.HELDOUT_FAMILIES:
        new = set(family) - trained_ops
        assert not new, f"{E.slug(family)} introduces untrained operations {new}"
        assert family not in E.TRAINED_FAMILIES, f"{E.slug(family)} is also trained"
    assert set(E.OPS) == trained_ops, \
        f"operations never trained alone: {set(E.OPS) - trained_ops}"


def test_a_held_out_family_is_never_a_trained_family_in_disguise(tasks):
    """An operation that does not apply to a record type is a no-op on it. Without this
    filter `("nickname", "initial", "date_style")` over a company record is exactly
    `("date_style",)`, which is trained, and the `rule_transfer` gate would be reading a
    trained family."""
    for t in tasks:
        if "heldout" not in t.name:
            continue
        for ex in t.load(150):
            for q in ex.questions:
                rt = q.meta["record_type"]
                family = next(f for f in E.HELDOUT_FAMILIES
                              if E.slug(f) == q.meta["family"])
                eff = E.effective(family, rt)
                assert eff, f"{q.meta['family']} does nothing to a {rt} record"
                assert eff not in E.TRAINED_FAMILIES, \
                    f"{q.meta['family']} on {rt} reduces to trained {E.slug(eff)}"


def test_held_out_families_actually_appear_in_the_held_out_tasks(tasks):
    seen = collections.Counter(
        q.meta["family"] for t in tasks if "heldout" in t.name
        for ex in t.load(120) for q in ex.questions)
    assert set(seen) == {E.slug(f) for f in E.HELDOUT_FAMILIES}
    trained = {q.meta["family"] for t in tasks
               if "heldout" not in t.name and not t.name.startswith("entityres_dev_")
               for ex in t.load(30) for q in ex.questions}
    assert not (set(seen) & trained), "a held-out family leaked into a trained task"


def test_dev_families_are_disjoint_from_trained_and_heldout(tasks):
    """Dev holds out compositions of its own. Every operation is trained alone, no dev
    family does to any record type what a trained or a held-out family does, and the dev
    tasks emit only dev families -- so selecting a checkpoint on them reads transfer
    without reading a structure the test split measures."""
    trained_ops = set().union(*(set(f) for f in E.TRAINED_FAMILIES))
    for family in E.DEV_FAMILIES:
        assert set(family) <= trained_ops, family
        assert family not in E.TRAINED_FAMILIES + E.HELDOUT_FAMILIES, family
    for rt in E.RECORD_TYPES:
        trained = {E.substantive(f, rt) for f in E.TRAINED_FAMILIES}
        held = {E.substantive(f, rt) for f in E.HELDOUT_FAMILIES}
        for _, family in E.combos((rt,), E.DEV_FAMILIES, True):
            sub = E.substantive(family, rt)
            assert len(sub) >= 2 and sub not in trained and sub not in held, (rt, family)
    dev = [t for t in tasks if t.name.startswith("entityres_dev_")]
    assert {t.name for t in dev} == {"entityres_dev_pair", "entityres_dev_roster"}
    seen = collections.Counter(q.meta["family"] for t in dev
                               for ex in t.load(120) for q in ex.questions)
    assert set(seen) == {E.slug(f) for f in E.DEV_FAMILIES}, seen
    other = {q.meta["family"] for t in tasks if not t.name.startswith("entityres_dev_")
             for ex in t.load(30) for q in ex.questions}
    assert not set(seen) & other


def test_twins_share_the_trained_family(tasks):
    from lod.corpus.services.sources.real.base import family_of
    fam = {t.name: family_of(t) for t in tasks}
    assert fam["entityres_pair_eval"] == fam["entityres_roster_eval"] == \
        fam["entityres_pair_person"] == fam["entityres_decisive"]
    assert fam["entityres_heldout_pair"] == fam["entityres_heldout_roster"]
    assert fam["entityres_dev_pair"] == fam["entityres_dev_roster"]
    assert len({fam["entityres_pair_eval"], fam["entityres_heldout_pair"],
                fam["entityres_dev_pair"]}) == 3


# ---- the label is code's, not the construction's ----------------------------------------

def test_every_label_is_the_policy_applied_to_the_rendered_records(tasks):
    """`verdict` is the only thing that labels a pair, and it reads the surface the model
    reads. Re-deriving it here from the module's own policy is the cheap half; the
    expensive half is that `build_pair` returns None rather than relabelling."""
    for want in (E.OPT_ONE, E.OPT_TWO):
        for rt in E.RECORD_TYPES:
            import random
            rng = random.Random(f"{rt}:{want}")
            for family in E.TRAINED_FAMILIES[:6] + E.HELDOUT_FAMILIES[:2]:
                for mode in ("id", "date"):
                    a, b = E.draw_pair(rt, family, want, mode, rng)
                    assert E.verdict(a, b)[0] == want


def test_unresolved_pairs_carry_a_soft_target_and_no_hidden_certainty(sample):
    n_soft = 0
    for t, ex in sample:
        for q in ex.questions:
            if q.meta.get("unresolved"):
                n_soft += 1
                assert q.target == [0.5, 0.5]
                assert q.meta["clause"] == "names_alone"
    assert n_soft, "no unresolved pairs were generated at all"
    total = sum(len(ex.questions) for _, ex in sample)
    assert n_soft / total >= 0.05, f"soft share {n_soft / total:.3f}"


# ---- the string-match line ---------------------------------------------------------------

def test_answer_is_not_recoverable_by_string_match(sample):
    """The row-36 failure: the gold option named verbatim in the prompt. A roster's
    options are entry keys and every key is in the state by construction, so the test
    for those is the similarity test below, not this one."""
    leaks = []
    for t, ex in sample:
        for q in ex.questions:
            g = _gold(q)
            if g in (q.instructions or ""):
                leaks.append((t.name, q.id, "instructions"))
            if q.id in ("pair", "pairwise", "decisive") and g in ex.state:
                leaks.append((t.name, q.id, "state"))
    assert not leaks, f"{len(leaks)} gold options appear verbatim: {leaks[:5]}"


def _pair_blocks(state: str) -> tuple[str, str]:
    if state.lstrip().startswith("{"):
        d = json.loads(state)
        return (json.dumps(d["record_from_system_a"], sort_keys=True),
                json.dumps(d["record_from_system_b"], sort_keys=True))
    a, b = state.split("\n\nRecord from system B. ")
    return a.replace("Record from system A. ", ""), b


def _sim(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def test_character_similarity_does_not_solve_pairs(sample):
    """The whole design point. Matched pairs are driven apart on the surface and
    unmatched pairs pulled together, so the best threshold anyone can pick on character
    similarity -- tuned on this very data, in whichever direction helps -- should land
    near the majority class, not near 1."""
    rows = []
    for t, ex in sample:
        for q in ex.questions:
            if q.id == "pair" and not q.meta.get("unresolved"):
                a, b = _pair_blocks(ex.state)
                rows.append((_sim(a, b), _gold(q)))
    assert len(rows) > 200
    ys = [y for _, y in rows]
    majority = max(ys.count(E.OPT_ONE), ys.count(E.OPT_TWO)) / len(ys)
    best = 0.0
    for i in range(101):
        thr = i / 100
        for flip in (False, True):
            hits = 0
            for s, y in rows:
                guess = E.OPT_ONE if (s >= thr) != flip else E.OPT_TWO
                hits += guess == y
            best = max(best, hits / len(rows))
    assert best <= majority + 0.12, (
        f"a tuned character-similarity threshold reaches {best:.3f} against a "
        f"majority-class {majority:.3f}: this is a string-similarity benchmark")


def _roster_blocks(state: str):
    if state.lstrip().startswith("{"):
        d = json.loads(state)
        return (json.dumps(d["query"], sort_keys=True),
                {k: json.dumps(v, sort_keys=True) for k, v in d["roster"].items()})
    head, rest = state.split("\nRoster.\n", 1)
    cands = {}
    for line in rest.splitlines()[2:]:
        if " | " in line:
            k, _, body = line.partition(" | ")
            cands[k.strip()] = body
    return head.replace("Query record: ", "").strip(), cands


def test_most_similar_roster_entry_does_not_solve_rosters(sample):
    hits = total = 0
    for t, ex in sample:
        for q in ex.questions:
            if q.id != "roster":
                continue
            query, cands = _roster_blocks(ex.state)
            assert cands, f"{t.name}: roster state parsed to no entries"
            total += 1
            hits += max(cands, key=lambda k: _sim(query, cands[k])) == _gold(q)
    assert total > 200
    assert hits / total <= 0.45, (
        f"picking the most similar roster entry reaches {hits / total:.3f}")


# ---- shape ---------------------------------------------------------------------------------

def test_states_are_long_enough_to_be_a_task(sample):
    """Under about 40 tokens a question is a shortcut rather than a task. Word count is
    the cheap floor; the tokenizer reading is in the findings (min 125, p95 1,533)."""
    for t, ex in sample:
        words = len(re.findall(r"\S+", ex.state))
        assert words >= 40, f"{t.name}: {words}-word state"


def test_option_counts_have_a_real_spread(sample):
    ks = [len(q.options) for _, ex in sample for q in ex.questions]
    assert max(ks) >= 32, f"max option count {max(ks)}"
    assert len(set(ks)) >= 12, f"only {len(set(ks))} distinct option counts"
    binary = sum(1 for k in ks if k == 2) / len(ks)
    assert binary <= 0.60, f"{binary:.0%} of questions are binary"
    assert sum(1 for k in ks if k > 8) / len(ks) >= 0.20


def test_labels_are_not_a_constant_per_question_kind(sample):
    for kind in ("pair", "pairwise", "roster"):
        ys = collections.Counter(_gold(q) for _, ex in sample for q in ex.questions
                                 if q.id == kind)
        assert len(ys) > 1, f"{kind} has a single label"
        if kind in ("pair", "pairwise"):
            share = max(ys.values()) / sum(ys.values())
            assert share <= 0.70, f"{kind} is {share:.0%} one label"


def test_roster_has_exactly_one_defensible_answer(sample):
    for t, ex in sample:
        for q in ex.questions:
            if q.id == "roster":
                assert sum(1 for p in q.target_probs() if p > 0) == 1


def test_generation_is_seeded(tasks):
    for t in tasks[:3]:
        assert [e.state for e in t.load(5)] == [e.state for e in t.load(5)]


def test_dates_round_trip_through_every_style():
    import datetime
    d = datetime.date(1983, 4, 7)
    for style in E.DATE_STYLES:
        assert E.parse_date(E.fmt_date(d, style)) == (1983, 4, 7), style


def test_identifier_normalisation_ignores_only_dress():
    assert E.norm_id("QQ 12 34 56 C") == E.norm_id("qq-12-34-56-c") == "qq123456c"
    assert E.norm_id("https://doi.org/10.1234/x") == E.norm_id("doi:10.1234/x")
    assert E.norm_id("SC 045 591") != E.norm_id("SC 045 592")


def test_hidden_keys_never_reach_the_state(sample):
    for t, ex in sample:
        for hidden in ("_entity", "_surname", "_style", "_nino", "_dob"):
            assert hidden not in ex.state, f"{t.name} leaked {hidden}"


# ---- the shortcuts measured against an earlier corpus build ---------------------------
# Three of these are regressions: the corpus shipped with all three open and every number
# below is the one measured on that build before the fix.

@pytest.fixture(scope="module")
def wide(tasks):
    """A bigger draw than `sample`: the three properties below are rates, not per-example
    invariants, and 90 examples per task is not enough to measure a rate."""
    return [(t, ex) for t in tasks for ex in t.load(400)]


def test_the_follow_up_entry_key_does_not_predict_its_own_answer(wide):
    """An earlier build: the entry named in the stem settled the follow-up 94.2 % of the
    time.

    The roster follow-up names an entry -- "And entry e14 on its own" -- and the entry was
    picked as the *first* candidate with the wanted verdict. `?01` was `two_entities` 95 %
    of the time and every entry from `?03` up was `one_entity` 100 % of the time, so the
    state did not need reading. The entry is now drawn uniformly from the eligible ones.
    """
    by_index = collections.defaultdict(collections.Counter)
    for _, ex in wide:
        for q in ex.questions:
            if q.id == "pairwise":
                by_index[int(q.meta["entry"][1:])][_gold(q)] += 1
    n = sum(sum(c.values()) for c in by_index.values())
    assert n > 500, f"only {n} follow-ups drawn"
    best = sum(c.most_common(1)[0][1] for c in by_index.values()) / n
    overall = collections.Counter(y for c in by_index.values() for y in c.elements())
    majority = max(overall.values()) / n
    assert best <= majority + 0.10, (
        f"the entry index alone reaches {best:.3f} against a majority class of "
        f"{majority:.3f}: the follow-up is answerable from the question stem")


def test_the_sentinel_is_not_worth_taking_on_sight(wide):
    """An earlier build: P(sentinel is the answer | a sentinel is offered) = 0.530.

    Seven times the 1/n a roster entry gets and nearly three times the 0.188 the corpus
    audit reports, so "if a none-of-these option is offered, take it" was worth 53 % of
    the rosters carrying one. That is an earlier model's sentinel failure with a wider
    surface.
    """
    offered = correct = total = 0
    for _, ex in wide:
        for q in ex.questions:
            if q.id != "roster":
                continue
            total += 1
            idx = [i for i, d in enumerate(q.descriptions or ())
                   if sentinel.is_sentinel_description(d)]
            assert len(idx) <= 1, "more than one sentinel on one roster"
            if idx:
                offered += 1
                correct += q.target_probs()[idx[0]] == 1.0
    assert total > 500 and offered > 200
    rate = correct / offered
    assert rate <= 0.25, (
        f"the sentinel is the answer {rate:.3f} of the times it is offered; "
        f"answering it on sight is a rule worth that much")


def test_every_sentinel_wording_comes_from_the_grammar(wide):
    """The bug this replaced shipped ONE fixed wording 1,402 times."""
    seen = collections.Counter()
    for _, ex in wide:
        for q in ex.questions:
            if q.id != "roster":
                continue
            for k, d in zip(q.options, q.descriptions or ()):
                if sentinel.is_sentinel_description(d):
                    assert sentinel.is_sentinel_key(k.removesuffix("_x")), \
                        f"sentinel description on a non-sentinel key {k!r}"
                    seen[d] += 1
    assert len(seen) > 500, f"only {len(seen)} distinct sentinel wordings"
    top, n = seen.most_common(1)[0]
    assert n <= 0.02 * sum(seen.values()), \
        f"{top!r} is {n} of {sum(seen.values())} sentinel wordings"


# ---- a reader has to be able to reach the answer ---------------------------------------

def _roster_records(state: str) -> tuple[dict, dict]:
    """The query and the roster, back out of the rendered state."""
    def unprose(text):
        text = text.strip().removesuffix(".")
        out = {}
        for part in text.split("; "):
            if ": " in part:
                k, v = part.split(": ", 1)
                out[k.strip().replace(" ", "_")] = v
        return out
    if state.lstrip().startswith("{"):
        d = json.loads(state)
        return d["query"], d["roster"]
    lines = state.split("\n")
    query = unprose(lines[0][len("Query record: "):])
    i = lines.index("Roster.")
    head = [c.strip().replace(" ", "_") for c in lines[i + 1].split(" | ")]
    roster = {}
    for line in lines[i + 3:]:
        if " | " not in line:
            continue
        cells = line.split(" | ")
        roster[cells[0].strip()] = {k: v for k, v in zip(head[1:], cells[1:])
                                    if v != "(not recorded)"}
    return query, roster


def test_no_two_roster_entries_look_the_same_and_answer_differently(wide):
    """An earlier build: four rosters had two defensible answers.

    All four were `initial`-family person rosters where the gold entry and a distractor
    both rendered `T. Ozdemir` with the same date of birth and the query carried no
    national insurance number. `verdict` separates them on a hidden given-name index; a
    reader holding only the state cannot. This reads the *state* rather than the records,
    so it is a check on what was written down and not on what the generator knew.
    """
    bad = []
    for t, ex in wide:
        for q in ex.questions:
            if q.id != "roster":
                continue
            rt = q.meta["record_type"]
            idf, datef = E.ID_FIELD[rt], E.DATE_FIELD[rt]
            namef = E.NAME_FIELD[rt]
            query, roster = _roster_records(ex.state)
            gold = _gold(q)
            seen = {}
            for key, rec in roster.items():
                if idf in query and idf in rec:
                    ev = ("id", E.norm_id(query[idf]), E.norm_id(rec[idf]))
                elif datef in query and datef in rec:
                    name = E.depunct(str(rec.get(namef, ""))).lower()
                    extra = str(rec.get("authors", ""))[:40].lower() if rt == "publication" else ""
                    ev = ("date", E.parse_date(query[datef]), E.parse_date(rec[datef]),
                          name, E.depunct(extra))
                else:
                    ev = ("unresolved", key)
                if ev in seen and (seen[ev] == gold) != (key == gold):
                    bad.append((t.name, seen[ev], key, gold))
                seen[ev] = key
    assert not bad, f"{len(bad)} rosters where two entries read alike: {bad[:3]}"


# ---- domain-17 audit: regressions for what a literal reading of the policy caught ------
# An independent audit re-derived every label from the policy text with its own reader;
# each test below pins one thing that reader found.

def _fold(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[.'’\-]", "", str(s).lower())).strip()


def test_the_doi_dress_is_named_in_clause_1():
    """Clause 1 said only "spacing, punctuation and letter case" vary, and a DOI is also
    written after `doi:` or as a link. Read literally, 6.4 % of resolved pairs flipped."""
    text = E.policy("publication")
    assert "`doi:`" in text and "https://doi.org/" in text
    for rt in ("person", "company"):
        assert "doi" not in E.policy(rt).lower()


def test_a_slip_lands_on_one_publication_string():
    """One `typo` op used to slip the title AND every author with the same seed: two slips
    in what clause 3 reads, where it allows one."""
    import random
    rng = random.Random(5)
    for _ in range(300):
        rec = E.new_publication(rng)
        E.randomise_style(rec, rng)
        clean = E.surface(rec, random.Random(0))
        E.apply_op("typo", rec, rng)
        slipped = E.surface(rec, random.Random(0))
        moved = [k for k in ("title", "venue") if clean[k] != slipped[k]]
        moved += ["author%d" % i for i, (x, y) in
                  enumerate(zip(clean["authors"], slipped["authors"])) if x != y]
        assert len(moved) <= 1, moved
        assert all(m in ("title", "venue", "author0") for m in moved), moved


def test_a_slip_never_spells_another_known_name():
    """`OZDEMIR, Rick` with R -> M reads as Michael, not as Richard mistyped."""
    import random
    rng = random.Random(9)
    for _ in range(3000):
        rec = E.new_person(rng)
        E.randomise_style(rec, rng)
        for op in ("reorder", "nickname"):
            if rng.random() < 0.5:
                E.apply_op(op, rec, rng)
        rec["_style"]["typo_field"] = "name"
        rec["_style"]["typo_seed"] = rng.randrange(1 << 30)
        slipped = E.person_name(rec, rng)
        rec["_style"].pop("typo_field")
        clean = E.person_name(rec, rng)
        new = E._tokens(slipped) - E._tokens(clean)
        assert not (new & E.KNOWN_NAME_TOKENS), (clean, slipped)


def _two_entity_date_pairs(rt, family, n, seed):
    import random
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        got = E.build_pair(rt, family, E.OPT_TWO, "date", rng)
        if got is not None:
            out.append(got)
    return out


def test_no_two_entity_pair_is_reconciled_by_an_initial():
    """A twin written `T. Ozdemir` beside `Theodore Ozdemir` on the same date of birth is
    one person under clause 3 (initials), whatever the generator knows about her. The
    audit's reader found 6-11 per 1,500 such pairs in every `initial` family."""
    bad = []
    for family in (("initial",), ("initial", "id_style")):
        for a, b in _two_entity_date_pairs("person", family, 1500, 3):
            sa, sb = a["_surf"], b["_surf"]
            if E.DATE_FIELD["person"] not in sa or E.DATE_FIELD["person"] not in sb:
                continue
            if E.parse_date(sa["date_of_birth"]) != E.parse_date(sb["date_of_birth"]):
                continue
            na, nb = _fold(sa["full_name"]), _fold(sb["full_name"])
            ta, tb = na.replace(",", "").split(), nb.replace(",", "").split()
            # `t ozdemir` against `theodore ozdemir` (either order, either side)
            for x, y in ((ta, tb), (tb, ta)):
                initials = [w for w in x if len(w) == 1]
                if initials and set(x) - set(initials) <= set(y) and \
                        any(w.startswith(initials[0]) for w in set(y) - set(x)):
                    bad.append((sa["full_name"], sb["full_name"]))
    assert not bad, f"{len(bad)} pairs reconciled by an initial: {bad[:3]}"


def test_a_written_down_record_never_gains_a_part_number():
    """`near_miss(..., "series")` set `Part I` on the roster query after its surface was
    written, so the query's duplicate was called `two_entities` and the roster redrawn."""
    import random
    rng = random.Random(2)
    for _ in range(50):
        q = E.new_publication(rng)
        E.randomise_style(q, rng)
        E.materialise(q, rng)
        other = E.near_miss(q, "series", rng)
        assert q["_part"] is None and other["_part"] == "II"
    fresh = E.new_publication(rng)
    E.near_miss(fresh, "series", rng)
    assert fresh["_part"] == "I", "an unwritten record still becomes Part I"


def test_held_out_families_are_substantive_combinations():
    """`("nickname", "initial", "date_style")` on a person was `("initial",)` -- initial
    overwrites nickname, and date_style changes nothing `vary_against` had not -- and
    `("date_style", "phone_style", "drop")` was `("drop",)`. Both were read by the
    `rule_transfer` gate as held-out compositions."""
    for rt in E.RECORD_TYPES:
        trained = {E.substantive(f, rt) for f in E.TRAINED_FAMILIES}
        for _, family in E.combos((rt,), E.HELDOUT_FAMILIES, True):
            sub = E.substantive(family, rt)
            assert len(sub) >= 2 and sub not in trained, (rt, family, sorted(sub))
    for family in E.HELDOUT_FAMILIES:
        assert not {"nickname", "initial"} <= set(family), family
        assert any(len(E.substantive(family, rt)) >= 2 for rt in E.RECORD_TYPES), family


def test_company_translit_is_not_a_corruption():
    assert E.effective(("translit",), "company") == ()


def test_a_company_renamed_under_one_registration_number_is_one_company(tasks):
    """Clause 3: a different legal suffix is a different company *unless the registration
    number says so*. Nothing exercised the exception, so "names differ -> two" was never
    wrong on a company."""
    import random
    rng = random.Random(4)
    renamed = 0
    for _ in range(300):
        a, b = E.draw_pair("company", ("surname_change",), E.OPT_ONE, "id", rng)
        assert E.verdict(a, b) == (E.OPT_ONE, "strong_identifier")
        renamed += _fold(a["_surf"]["legal_name"]) != _fold(b["_surf"]["legal_name"])
    assert renamed >= 250, renamed
    for _ in range(50):
        assert E.build_pair("company", ("surname_change",), E.OPT_ONE, "date", rng) is None


def test_the_name_alone_does_not_decide_most_pairs():
    """Pre-audit generator: "same name -> one entity", reading neither the identifier nor
    the date, scored 0.81 on resolved pairs against a 0.50 majority (independent audit
    with its own name reader; 0.85 on companies), because two in three near misses
    changed the name and no matched company ever did. Measured here with the policy's own
    notion of "the same name" (`name_key`), which is the strongest form of the rule."""
    import random
    rng = random.Random(11)
    hits = total = 0
    for rt in E.RECORD_TYPES:
        pool = E.combos((rt,), E.TRAINED_FAMILIES, False)
        for i in range(900):
            _, family = pool[i % len(pool)]
            mode = E._draw_mode(rng)
            if mode == "none":
                continue
            want = E.OPT_ONE if rng.random() < 0.5 else E.OPT_TWO
            a, b = E.draw_pair(rt, family, want, mode, rng)
            got, _ = E.verdict(a, b)
            guess = E.OPT_ONE if E.name_key(a) == E.name_key(b) else E.OPT_TWO
            total += 1
            hits += guess == got
    rate = hits / total
    assert rate <= 0.78, f"the name alone decides {rate:.3f} of resolved pairs"


def test_every_stem_is_a_registered_template():
    from lod.paths import ASSETS
    from lod.phrasings import PhrasingBank
    spec = json.loads((ASSETS / "phrasing_specs" / "d17.json").read_text())
    n = 0
    for kind, pool in E.STEM_POOLS.items():
        for i, t in enumerate(pool):
            tid = E.template_id(kind, i)
            assert spec[tid]["template"] == t
            assert spec[tid]["answer_meaning"] and spec[tid]["state_kind"]
            assert PhrasingBank({}).pick(tid, t, "k") == t
            n += 1
    assert n == len(spec)
