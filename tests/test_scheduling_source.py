"""Row 41: scheduling and allocation, the one domain whose options interact.

Most of this file exists to hold one line honest. The corpus documentation claims that here "picking one
forecloses another", and that is only true if the answer cannot be had by scoring each
option against the state on its own. `test_every_option_survives_the_state_alone` is the
measurement: it re-checks every option against everything the state says -- availability,
certification, which shifts are already booked -- and requires every option to pass. When
that number falls to one, the interaction has become decoration.
"""

from __future__ import annotations

import itertools
import random
import re
from collections import Counter

import pytest

from lod.corpus.services.sources.synth import scheduling as S


@pytest.fixture(scope="module")
def tasks():
    return S.tasks()


@pytest.fixture(scope="module")
def sample(tasks):
    """A slice of every task, questions with their example. Kept small: the label comes
    from enumerating an assignment space, so generation is not free."""
    out = []
    for t in tasks:
        for ex in t.load(24):
            for q in ex.questions:
                out.append((t, ex, q))
    return out


@pytest.fixture(scope="module")
def large(tasks):
    """A bigger slice of every task, for the families `sample` holds too few of."""
    return [ex for t in tasks for ex in t.load(150)]


@pytest.fixture(scope="module")
def moves(large):
    """`move` questions, roster shape only. Since `move` keeps only draws that need both
    rule types (scheduling_shortcuts.py) it is ~3-8 % of a task, not the 15 % it once
    was, and `sample` holds a handful."""
    return [ex.questions[0] for ex in large
            if ex.questions[0].meta["family"] == "move"
            and ex.questions[0].meta["shape"] == "roster"]


def _gold(q):
    probs = q.target_probs()
    return q.options[max(range(len(probs)), key=probs.__getitem__)]


# ---- shape and well-formedness -------------------------------------------------------

def test_every_task_yields_what_it_is_asked_for(tasks):
    for t in tasks[:6]:
        got = list(t.load(24))
        assert len(got) == 24, f"{t.name} produced {len(got)}"


def test_the_loader_is_not_silently_empty(tasks):
    """A generator that returns [] is the failure row 36 and the row-1 sweep both hit."""
    # 11 trained pairs and their 11 twins, 4 dev pairs, 6 test pairs
    assert len(tasks) == 32
    for t in tasks:
        assert next(iter(t.load(3)), None) is not None, f"{t.name} produced nothing"


def test_targets_are_well_formed(sample):
    for t, ex, q in sample:
        probs = q.target_probs()
        assert probs is not None and abs(sum(probs) - 1.0) < 1e-9
        assert len(probs) == len(q.options)
        assert len(set(q.options)) == len(q.options), f"{t.name}: duplicate options"
        assert q.instructions, f"{t.name}: no criteria"


def test_generation_is_seeded(tasks):
    t = tasks[0]
    assert [e.state for e in t.load(6)] == [e.state for e in t.load(6)]


def test_tasks_are_marked_generated_and_on_row_41(tasks):
    for t in tasks:
        assert t.row == S.ROW == 41 and t.real is False
        assert t.name.startswith("sched_"), f"{t.name} misses the PROTECTED_PREFIXES key"


# ---- the held-out structure ----------------------------------------------------------

def test_held_out_combinations_never_reach_training(tasks):
    held = {f"sched_{S.combo_name(c)}" for c in S.HELDOUT_COMBOS}
    dev = {f"sched_{S.combo_name(c)}" for c in S.DEV_COMBOS}
    for t in tasks:
        if t.name in held:
            assert t.force_split == "testreal", f"{t.name} held out but in {t.force_split}"
        elif t.name in dev:
            assert t.force_split == "devreal", f"{t.name} dev-held but in {t.force_split}"
        elif t.name.endswith("_eval"):
            # in-distribution twin: devreal, so testreal stays held-out structure only
            assert t.force_split == "devreal", t.name
        else:
            assert t.force_split == "train", t.name


def test_held_out_combinations_introduce_no_unseen_constraint_type():
    """The holdout has to be a structure, not a sample. Every rule type inside a held-out
    pair appears in some trained pair, so a drop there is a failure to compose two known
    constraints rather than a first encounter with one."""
    trained = {t for combo in S.TRAINED_COMBOS for t in combo}
    for combo in S.HELDOUT_COMBOS + S.DEV_COMBOS:
        unseen = set(combo) - trained
        assert not unseen, f"{combo} introduces unseen constraint types {unseen}"
        assert combo not in S.TRAINED_COMBOS
    assert len(S.TRAINED_COMBOS) + len(S.DEV_COMBOS) + len(S.HELDOUT_COMBOS) == 21
    assert trained == set(S.RULE_TYPES)


def test_dev_pairs_are_disjoint_from_test_and_keep_every_type_trained_twice():
    """Dev's held-out pairs are a second holdout, not a share of test's, and moving them
    out of training must leave each rule type composed with at least two others."""
    assert not set(S.DEV_COMBOS) & set(S.HELDOUT_COMBOS)
    per_type = {t: sum(t in c for c in S.TRAINED_COMBOS) for t in S.RULE_TYPES}
    assert min(per_type.values()) >= 2, per_type


def test_dev_pairs_generate_every_family():
    """A dev pair has to carry the same question families test's pairs do, or selecting
    on it weighs the row differently from the number it is selecting for."""
    for combo in S.DEV_COMBOS:
        fams = set()
        for ex in S._loader(combo, 6061)(60):
            q = ex.questions[0]
            assert q.meta["combo"] == "+".join(combo)
            assert 0 <= q.target < len(q.options)
            fams.add(q.meta["family"])
        assert {"pick", "grant"} <= fams, (combo, fams)


def test_held_out_rule_sentences_are_all_seen_in_training():
    """Vocabulary as well as type: a held-out week must not read differently, only be
    arranged differently. Every rule sentence template a held-out task can emit is one a
    trained task can emit too."""
    def templates(combos, seed):
        seen = set()
        for combo in combos:
            for i in range(40):
                inst = S.make_instance(S._rng(seed, f"{combo}:{i}"), combo, tight=i % 3)
                if inst is None:
                    continue
                for r in inst.rules:
                    text = S.rule_text(inst, r)
                    for n in inst.names:
                        text = text.replace(n, "<name>")
                    seen.add(text)
        return seen
    new = (templates(S.HELDOUT_COMBOS, 7717) | templates(S.DEV_COMBOS, 6061)) \
        - templates(S.TRAINED_COMBOS, 0)
    assert not new, f"held-out combinations use unseen wording: {sorted(new)[:4]}"


def test_the_model_wrote_wording_and_nothing_else():
    """THE INVARIANT, checked at the one place a model's output reaches this file. The
    thread styles are the only text here the model proposed; none of them may carry a
    fact -- no number, no name, no site, no shift, no certification."""
    banned = set(S.FIRSTS) | set(S.LASTS) | set(S.SITES) | set(S.SKILLS) | set(S.DAYS)
    assert len(S.THREAD_STYLES) >= 10
    for style in S.THREAD_STYLES:
        assert set(style) == {"subject", "shifts", "staff", "leave", "booked", "open"}
        for key, text in style.items():
            assert isinstance(text, str) and 3 <= len(text) <= 200
            assert not any(c.isdigit() for c in text), f"{key!r} carries a number: {text}"
            for word in banned:
                assert word not in text, f"{key!r} carries {word!r}"
    assert len({st["subject"] for st in S.THREAD_STYLES}) == len(S.THREAD_STYLES)


# ---- three shapes, one task ----------------------------------------------------------

def test_the_three_shapes_agree_on_the_answer(tasks):
    """Same instance, three renderings. If the target moved with the shape the domain
    would be measuring parsing, not scheduling."""
    n = 0
    for t in tasks[:10]:
        batch = list(t.load(24))
        for i in range(0, len(batch) - 2, 3):
            trio = batch[i:i + 3]
            shapes = [e.questions[0].meta["shape"] for e in trio]
            assert shapes == list(S.SHAPES), shapes
            qs = [e.questions[0] for e in trio]
            assert len({q.question for q in qs}) == 1
            assert len({tuple(q.options) for q in qs}) == 1
            assert len({q.target for q in qs}) == 1
            assert len({q.instructions for q in qs}) == 1
            assert len({e.state for e in trio}) == 3, "two shapes rendered identically"
            n += 1
    assert n > 40


def test_every_shape_carries_the_facts_the_answer_needs():
    """A shape that dropped a fact would make its version of the question unanswerable;
    the check is that each rendering names every shift and every person."""
    for combo in (S.TRAINED_COMBOS[0], S.HELDOUT_COMBOS[0]):
        for i in range(12):
            inst = S.make_instance(S._rng(0, f"facts:{combo}:{i}"), combo, tight=i % 3)
            if inst is None:
                continue
            for shape in S.SHAPES:
                state = S.render(inst, shape)
                for s in range(inst.n_slots):
                    assert f"shift {s + 1}" in state or f"| {s + 1} |" in state
                for name in inst.names:
                    assert name in state, f"{shape} drops {name}"
                for p in range(inst.n_staff):
                    for s in inst.away[p]:
                        assert f"shift {s + 1}" in state


# ---- the claim this domain exists for ------------------------------------------------

def _state_survivors(t, ex, q) -> int:
    """How many options survive a check of everything the state says about them alone.

    That is the whole of the "score each option independently" strategy the other
    eighteen domains allow. For this domain it must leave the field untouched.
    """
    fam = q.meta["family"]
    state = ex.state
    if fam in ("feasible", "culprit", "move"):
        # yes/no, a rule, or a shift number: the state says nothing about any one of them
        # that it does not say about all of them.
        return len(q.options)
    survivors = 0
    for opt in q.options:
        if fam == "grant":
            who, _, shift = opt.partition(" on shift ")
            pairs = [(who, int(shift))]
        else:                                     # pick: a whole roster
            pairs = []
            for part in opt.split("; "):
                num, _, who = part.partition(" ")
                pairs.append((who, int(num)))
        ok = True
        for who, shift in pairs:
            # unavailable for that shift?
            for line in state.splitlines():
                if line.strip().startswith(who) or f"| {who} |" in line:
                    if "not available" in line and f"shift {shift} " in line:
                        ok = False
            if f"shift {shift} (" in state and "Confirmed so far" in state:
                pass
        survivors += ok
    return survivors


def test_every_option_survives_the_state_alone(sample):
    """The measurement the documented claim stands on. If most questions came down to one
    survivor, the constraints would be decoration and this would be domain 1 in a hat."""
    per_family: dict[str, list[float]] = {}
    for t, ex, q in sample:
        per_family.setdefault(q.meta["family"], []).append(
            _state_survivors(t, ex, q) / len(q.options))
    for fam, shares in per_family.items():
        worst = min(shares)
        assert worst == 1.0, f"{fam}: a question left only {worst:.0%} of its options"


def test_the_answer_needs_more_than_one_shift_read_at_a_time():
    """Every rule type is coupling: none of them can be decided from one shift's row.
    A unary rule type would let the model eliminate an option by a single lookup."""
    for t in S.RULE_TYPES:
        combo = (t, "rest") if t != "rest" else ("rest", "one_per_day")
        inst = next((x for x in (S.make_instance(S._rng(0, f"cpl:{t}:{i}"), combo,
                                                 tight=1) for i in range(40))
                     if x is not None), None)
        assert inst is not None, f"{t}: no instance in 40 draws"
        # Changing who covers ONE shift can flip a rule that says nothing about that
        # shift. That is what "coupling" means here, and it is why an option cannot be
        # scored on its own: its shifts constrain each other.
        rng = random.Random(f"flip:{t}")
        flips = 0
        for _ in range(200):
            a = tuple(rng.choice(x) for x in inst.allowed)
            s = rng.randrange(inst.n_slots)
            b = a[:s] + (rng.choice(inst.allowed[s]),) + a[s + 1:]
            va, vb = set(S.violated(inst, a, inst.rules)), set(S.violated(inst, b, inst.rules))
            flips += va != vb
        assert flips > 0, f"{t}: no single change alters any rule -- rules are inert"


def test_no_single_rule_cuts_the_field_down_in_the_pick_family(tasks):
    """`pick` spreads its distractors over as many broken rules as it can. Where the
    instance carries more than one rule and more than two distractors, checking one rule
    must not be enough to find the answer."""
    ok = tot = 0
    for t in tasks[:12]:
        for ex in t.load(24):
            q = ex.questions[0]
            if q.meta["family"] != "pick" or len(q.options) < 4:
                continue
            tot += 1
            ok += q.meta.get("rules_broken", 1) > 1
    assert tot, "no multi-option pick questions"
    assert ok / tot > 0.75, f"only {ok / tot:.0%} of pick questions break >1 rule"


# ---- the solver is right -------------------------------------------------------------

def test_the_solver_agrees_with_an_independent_recheck():
    """A wrong solver is a silently poisoned domain, so the label is re-derived here by a
    second route: `pick`'s answer is re-checked rule by rule against the roster it names,
    and `feasible`'s against a fresh brute-force enumeration."""
    checked = Counter()
    for combo in (S.TRAINED_COMBOS[0], S.TRAINED_COMBOS[7], S.HELDOUT_COMBOS[2]):
        for i in range(40):
            rng = S._rng(0, f"chk:{combo}:{i}")
            inst = S.make_instance(rng, combo, tight=i % 3)
            if inst is None:
                continue
            brute = [a for a in itertools.product(*[list(x) for x in inst.allowed])
                     if not S.violated(inst, a, inst.rules)]
            an = S.analyse(inst)
            assert an is not None
            domain, viols = an
            assert [a for a, v in zip(domain, viols) if not v] == brute
            got = S._f_pick(inst, rng, (domain, viols, brute))
            if got is None:
                continue
            _, _, opts, _, gold, _, _ = got
            for j, opt in enumerate(opts):
                a = tuple(inst.names.index(part.split(" ", 1)[1])
                          for part in opt.split("; "))
                broken = S.violated(inst, a, inst.rules)
                assert (not broken) == (j == gold), \
                    f"option {j} broken={broken} gold={gold}"
            checked["pick"] += 1
    assert checked["pick"] > 20


def test_the_culprit_is_the_only_rule_whose_removal_helps():
    """The family's whole claim, re-derived: lift the answer and the week works, lift
    anything else and it still does not. The rules are rebuilt from the option *text*
    against the week the question is asked of (which `culprit` may have narrowed), and
    the week is brute-forced here, not through the generator's search."""
    n = Counter()
    for combo in S.COMBOS:
        for i in range(30):
            rng = S._rng(0, f"cul:{combo}:{i}")
            inst = S.make_instance(rng, combo, tight=i % 3)
            if inst is None:
                continue
            an = S.analyse(inst)
            domain, viols = an
            feas = [a for a, v in zip(domain, viols) if not v]
            got = S._f_culprit(inst, rng, (domain, viols, feas))
            if got is None:
                continue
            _, _, opts, _, gold, _, meta, week = got
            by_text = {S.rule_text(week, c): c for t in S.RULE_TYPES
                       for c in S._candidate_rules(t, week.n_staff)}
            rules = [by_text[o] for o in opts]
            assert {r["t"] for r in rules} == set(combo), "a type outside the combination"
            assert len(rules) >= 3
            space = list(itertools.product(*[list(x) for x in week.allowed]))
            assert not [a for a in space if not S.violated(week, a, rules)], \
                "the week is coverable, so nothing makes it impossible"
            restores = [j for j in range(len(rules))
                        if any(not S.violated(week, a, rules[:j] + rules[j + 1:])
                               for a in space)]
            assert restores == [gold], f"rules {restores} restore the week, gold {gold}"
            assert meta["culprit_type"] == rules[gold]["t"]
            # a narrowed week only ever adds leave
            assert all(set(a) <= set(b) for a, b in zip(week.allowed, inst.allowed))
            n[combo] += 1
            if n[combo] >= 4:
                break
    # an earlier build: 11 of 21 combinations carried no culprit at all (domain-19 audit)
    assert len(n) >= 17, f"only {len(n)} combinations yield a culprit: {sorted(n)}"
    assert sum(n.values()) > 60


def test_culprit_is_not_starved():
    """An earlier build shipped `culprit` at 2.5 % of the row (510 of 20,196) against the
    15 % its FAMILY_SHARE asks for: only a coverable week plus one *added* rule was
    tried, so the single global rules could never be the answer. Measured per instance
    offered to it."""
    got = tried = 0
    for combo in S.COMBOS:
        for i in range(1, 9):
            rng = S._rng(0, f"{'_'.join(combo)}:{i}")
            inst = S.make_instance(rng, combo, tight=i % 3)
            if inst is None:
                continue
            domain, viols = S.analyse(inst)
            feas = [a for a, v in zip(domain, viols) if not v]
            tried += 1
            got += S._f_culprit(inst, rng, (domain, viols, feas)) is not None
    assert got / tried > 0.18, f"culprit builds on only {got / tried:.0%} of weeks"


def test_violation_matrix_agrees_with_violated():
    """The vectorised search and the plain checker must be the same rules."""
    rng = random.Random(3)
    for combo in S.COMBOS:
        inst = next(x for x in (S.make_instance(S._rng(1, f"vm:{combo}:{i}"), combo, tight=i % 3)
                                for i in range(60)) if x is not None)
        rules = [c for t in S.RULE_TYPES for c in S._candidate_rules(t, inst.n_staff)]
        domain = [tuple(rng.choice(x) for x in inst.allowed) for _ in range(300)]
        M = S.violation_matrix(inst, domain, rules)
        for a, row in zip(domain, M):
            assert set(S.violated(inst, a, rules)) == {j for j, v in enumerate(row) if v}


def test_grant_answers_are_grantable_and_the_rest_are_not():
    for combo in (S.TRAINED_COMBOS[3], S.HELDOUT_COMBOS[0]):
        n = 0
        for i in range(60):
            rng = S._rng(0, f"gr:{combo}:{i}")
            inst = S.make_instance(rng, combo, tight=i % 3)
            if inst is None:
                continue
            an = S.analyse(inst)
            if an is None:
                continue
            domain, viols = an
            feas = [a for a, v in zip(domain, viols) if not v]
            got = S._f_grant(inst, rng, (domain, viols, feas))
            if got is None:
                continue
            _, _, opts, _, gold, booked, _ = got
            for j, opt in enumerate(opts):
                who, _, shift = opt.partition(" on shift ")
                p, s = inst.names.index(who), int(shift) - 1
                fixed = dict(booked)
                fixed[s] = p
                left = [a for a in feas if all(a[x] == y for x, y in fixed.items())]
                assert bool(left) == (j == gold), f"{opt}: {len(left)} completions"
            n += 1
        assert n > 5, f"{combo}: only {n} grant questions"


# ---- the shortcuts a generated source ships with unless it is measured ---------------

def test_the_answer_is_not_recoverable_by_string_match(sample):
    """Row 36 shipped at 100 % solvable this way. The heuristic here is the strongest
    cheap one: if exactly one option appears verbatim in the state or the criteria, take
    it. It must do no better than picking at random.

    Only `feasible` ever has exactly one option in the text ("no" is a substring of
    "not available"), so the heuristic is "always say no" and scores exactly the share of
    `no`s. Since `feasible` goes out in yes/no pairs that share is exactly one half:
    random, and allowed."""
    hits = tot = 0
    for t, ex, q in sample:
        text = ex.state + "\n" + (q.instructions or "")
        present = [o for o in q.options if o in text]
        if len(present) != 1:
            continue
        tot += 1
        hits += present[0] == _gold(q)
    if tot:
        assert hits / tot <= 0.5, f"verbatim match picks the answer {hits / tot:.0%} of the time"


def test_no_option_is_given_away_by_its_own_criteria(sample):
    """The gold option must not be named by the instructions that ask for it."""
    for t, ex, q in sample:
        if q.meta["family"] in ("culprit", "feasible"):
            continue         # the options *are* the criteria / are yes and no
        assert _gold(q) not in (q.instructions or ""), f"{t.name}: gold in instructions"


def test_the_gold_option_is_not_at_a_fixed_position(sample, moves):
    """A solver that returns the first feasible assignment in enumeration order puts the
    answer at index 0 every time, and the model learns the index instead of the task.

    Counted once per instance, not once per rendering: the three shapes carry the same
    shuffled option list, so counting all three would understate the spread threefold.
    """
    for fam in ("pick", "grant", "move"):
        qs = moves if fam == "move" else [
            q for _, _, q in sample if q.meta["family"] == fam and q.meta["shape"] == "roster"]
        assert len(qs) > 30, fam
        share_first = sum(q.target == 0 for q in qs) / len(qs)
        expect = sum(1.0 / len(q.options) for q in qs) / len(qs)
        # 3 sigma on the binomial, so a real bias fails and sampling noise does not
        sigma = (expect * (1 - expect) / len(qs)) ** 0.5
        assert abs(share_first - expect) < max(0.05, 3 * sigma), \
            f"{fam}: answer at index 0 {share_first:.1%}, chance {expect:.1%}"


def test_the_majority_class_baseline_is_weak(sample):
    """Per family, the best constant answer, measured against the chance rate rather than
    a flat number: `move` offers four to six shifts, so 1 in 5 is chance, not a prior."""
    for fam in S.FAMILIES:
        qs = [q for _, _, q in sample if q.meta["family"] == fam]
        if len(qs) < 30:
            continue
        top = Counter(_gold(q) for q in qs).most_common(1)[0][1] / len(qs)
        chance = sum(1.0 / len(q.options) for q in qs) / len(qs)
        assert top <= chance + 0.10, \
            f"{fam}: majority class {top:.0%} against chance {chance:.0%}"


def test_option_counts_are_not_all_binary(sample):
    ks = [len(q.options) for _, _, q in sample]
    assert max(ks) >= 20, f"max option count {max(ks)}"
    assert sum(1 for k in ks if k == 2) / len(ks) < 0.35
    assert sum(1 for k in ks if k > 8) / len(ks) > 0.05


def test_states_are_long_enough_to_be_a_task(tok, sample):
    """Under 40 tokens a state is a shortcut, not a task. Measured with the tokenizer,
    not with whitespace."""
    lens = sorted(len(tok(ex.state).input_ids) for _, ex, _ in sample)
    assert lens[0] >= 40, f"shortest state is {lens[0]} tokens"
    p95 = lens[int(0.95 * len(lens))]
    assert p95 < 2304, f"p95 state is {p95} tokens, over ship_v4.sh's --max-state-tokens"


# ---- the same claim, read off the rendered text rather than the generator -------------

def _parse_roster(state):
    shifts, staff, booked, sec = [], {}, {}, None
    for ln in state.split("\n"):
        if ln.startswith("Shifts to cover"):
            sec = "s"; continue
        if ln.startswith("Staff"):
            sec = "p"; continue
        if ln.startswith("Already booked"):
            sec = "b"; continue
        if not ln.strip():
            continue
        cells = [c.strip() for c in ln.strip("|").split("|")] if ln.startswith("|") else None
        if sec == "s" and cells and cells[0] not in ("#", "---"):
            shifts.append((int(cells[0]), None if cells[2] == "-" else cells[2]))
        elif sec == "p" and cells and cells[0] not in ("name", "---"):
            held = set() if cells[1] == "none recorded" else {x.strip() for x in cells[1].split(",")}
            away = {int(m) for m in re.findall(r"shift (\d+) \(", cells[2])}
            staff[cells[0]] = (held, away)
        elif sec == "b" and ln.startswith("- ") and "nothing booked yet" not in ln:
            m = re.match(r"- shift (\d+) \([^)]*\): (.+)$", ln)
            booked[int(m.group(1))] = m.group(2).strip()
    return shifts, staff, booked


def test_no_option_is_ruled_out_by_certifications_leave_or_bookings(sample):
    """The stronger reading of the same claim, and the one the docstring above promises.

    `_state_survivors` reads only the "not available" lines, so it would pass a field in
    which half the plans put an uncertificated person on a certificated shift, or a
    `grant` request on a shift that is already taken. This re-parses the rendered roster
    and checks all three: availability, certification and booking. Every option of every
    question has to survive all of them, because the state is unary by design and only
    the coupling policy in the criteria may cut the field down.
    """
    checked = pairs = 0
    for _t, ex, q in sample:
        fam = q.meta["family"]
        if fam not in ("pick", "grant") or q.meta["shape"] != "roster":
            continue
        shifts, staff, booked = _parse_roster(ex.state)
        need = dict(shifts)
        assert need, "the roster shape rendered no shifts"
        for opt in q.options:
            if fam == "grant":
                who, _, num = opt.partition(" on shift ")
                wants = [(who, int(num))]
                assert int(num) not in booked, f"{opt} asks for a shift already taken"
            else:
                wants = []
                for part in opt.split("; "):
                    num, _, who = part.partition(" ")
                    wants.append((who, int(num)))
            for who, num in wants:
                held, away = staff[who]
                assert num not in away, f"{opt}: {who} is away for shift {num}"
                if need[num] is not None:
                    assert need[num] in held, f"{opt}: {who} lacks {need[num]}"
                pairs += 1
        checked += 1
    assert checked > 100, f"only {checked} questions carried a person-shift option"
    assert pairs > 500


def test_the_culprit_is_not_pickable_from_its_rule_type(tasks):
    """`culprit`'s options are rule sentences, so the one state-blind strategy left is a
    prior over rule *types*. An earlier build shipped a culprit that was `min_shifts` in
    68 % of cases, `buddy` in 30 % and `max_shifts` in 2 %, and four of the seven types
    never appeared at all -- a reader who simply prefers `min_shifts` then `buddy` scored
    0.342 against 0.246 chance. It is bounded here so it cannot get worse unnoticed.
    """
    kinds = ("min_shifts", "buddy", "max_shifts", "pair_conflict", "handover",
             "rest", "one_per_day")
    hit = expect = n = 0.0
    seen = Counter()
    # An earlier build's `culprit` was 2.5 % of the row and eleven combinations carried
    # none; since the domain-19 audit it is searched rather than waited for and 19 of 21
    # carry it, but the sample is still drawn from the combinations that carry it most.
    carriers = [t for t in tasks
                if any(k in t.name for k in ("mn", "mx", "bu"))]
    for t in carriers:
        for ex in t.load(150):
            q = ex.questions[0]
            if q.meta["family"] != "culprit" or q.meta["shape"] != "roster":
                continue
            seen[q.meta["culprit_type"]] += 1
            types = [_rule_type(o) for o in q.options]
            assert None not in types, f"unparsed rule option in {q.options}"
            for k in kinds:
                idx = [i for i, x in enumerate(types) if x == k]
                if idx:
                    hit += (q.target in idx) / len(idx)
                    break
            expect += 1.0 / len(q.options)
            n += 1
    assert n > 40, f"only {n} culprit questions drawn"
    assert hit / n <= expect / n + 0.15, \
        f"a rule-type prior scores {hit / n:.3f} against {expect / n:.3f} chance"


def _rule_type(line):
    line = line.strip()
    if re.match(r"^.+ may cover at most \d+ shifts? this week\.$", line):
        return "max_shifts"
    if re.match(r"^.+ must cover at least \d+ shifts? this week\.$", line):
        return "min_shifts"
    if line.startswith("Nobody may cover two shifts"):
        return "rest"
    if line.startswith("Nobody may cover more than one shift"):
        return "one_per_day"
    if " must not both be on duty on the same day." in line:
        return "pair_conflict"
    if " is handing over to " in line:
        return "handover"
    if " may only be on duty on a day when " in line:
        return "buddy"
    return None


# ---- domain-19 audit regressions -----------------------------------------------------

def test_every_week_carries_both_rule_types_of_its_combination():
    """A drawn pair already taken used to be skipped, which dropped the only `handover`
    of some `pair_conflict+handover` weeks: 0.7 % of questions labelled with a
    combination carried one of its types only."""
    for combo in S.COMBOS:
        for i in range(40):
            inst = S.make_instance(S._rng(0, f"types:{combo}:{i}"), combo, tight=i % 3)
            if inst is not None:
                assert {r["t"] for r in inst.rules} == set(combo), (combo, inst.rules)


def test_no_floor_is_above_what_the_person_can_take(sample):
    """"Hana must cover at least 2" with Hana free for one shift is a row lookup; it was
    the whole answer to 3 % of `feasible` and 45 % of `culprit` questions."""
    for t, ex, q in sample:
        if q.meta["shape"] != "roster":
            continue
        _, staff, _ = _parse_roster(ex.state)
        shifts, _, _ = _parse_roster(ex.state)
        need = dict(shifts)
        lines = q.options if q.meta["family"] == "culprit" else \
            [ln[2:] for ln in (q.instructions or "").split("\n") if ln.startswith("- ")]
        for ln in lines:
            m = re.match(r"^(.+) must cover at least (\d+) shifts? this week\.$", ln)
            if not m:
                continue
            held, away = staff[m.group(1)]
            can = [n for n in need if n not in away and (need[n] is None or need[n] in held)]
            assert len(can) >= int(m.group(2)), f"{t.name}: {ln} but free for {can}"


def test_the_calendar_never_says_more_than_the_leave():
    """The calendar's per-day line read "off today: Ana" for anyone away from ANY shift
    that day -- a whole-day claim that was false in 84 % of listings, where the label
    still used her for the rest of the day."""
    for combo in (S.TRAINED_COMBOS[0], S.HELDOUT_COMBOS[1]):
        for i in range(20):
            inst = S.make_instance(S._rng(0, f"cal:{combo}:{i}"), combo, tight=i % 3)
            if inst is None:
                continue
            state = S.render(inst, "calendar")
            assert "off today" not in state
            for d, day in enumerate(inst.days):
                block = state.split(f"\n{day}\n", 1)[1] if d else state.split(f"{day}\n", 1)[1]
                line = next(ln for ln in block.split("\n") if ln.startswith("    away: "))
                for p, name in enumerate(inst.names):
                    mine = [s + 1 for s in inst.away[p] if inst.day_of(s) == d]
                    if mine:
                        word = "shift" if len(mine) == 1 else "shifts"
                        assert f"{name} ({word} {', '.join(map(str, mine))})" in line, line
                    else:
                        assert name not in line, line


def test_the_email_thread_lists_what_its_last_message_points_at():
    """Ten of fourteen model-written `open` lines say "every shift listed below"; the
    message is the last in the thread, so the shifts must follow it."""
    for i in range(30):
        inst = S.make_instance(S._rng(0, f"mail:{i}"), S.TRAINED_COMBOS[i % len(S.TRAINED_COMBOS)], tight=i % 3)
        if inst is None:
            continue
        mail = S.render(inst, "email")
        tail = mail.split("\n")
        last = max(j for j, ln in enumerate(tail) if ln.startswith("From: "))
        listed = [ln for ln in tail[last + 2:] if ln.startswith("  shift ")]
        assert len(listed) == inst.n_slots
        assert all(ln.endswith(" — open") for ln in listed)


def test_move_is_not_answered_by_the_movers_own_shift(moves):
    """An earlier build: the answer was the other shift the mover already held 59 % of
    the time (chance 20 %) and the nearest shift 50 %. Both guesses skip the policy."""
    held = near = chance = n = 0.0
    for q in moves:
        s0 = int(re.search(r"now has to take shift (\d+)", q.instructions).group(1))
        x = re.search(r"\. (.+?) now has to take", q.instructions).group(1)
        idx = [i for i, d in enumerate(q.descriptions) if d.endswith(x)]
        held += (q.target in idx) / len(idx) if idx else 1 / len(q.options)
        nums = [int(re.match(r"shift (\d+)", o).group(1)) for o in q.options]
        dd = [abs(k - s0) for k in nums]
        idx = [i for i, v in enumerate(dd) if v == min(dd)]
        near += (q.target in idx) / len(idx)
        chance += 1 / len(q.options)
        n += 1
    # 36 at 150 per task since four trained pairs moved to dev (32 tasks, not 36); the
    # 0.15 margin is still over two standard errors at that size
    assert n > 30
    assert held / n < chance / n + 0.15, f"mover's-own-shift guess {held / n:.2f}"
    assert near / n < chance / n + 0.15, f"nearest-shift guess {near / n:.2f}"


def test_feasible_is_not_read_off_how_crowded_the_week_is(large):
    """Balanced overall, "few names per shift -> no" scored 0.80 against a 0.58
    majority. Balanced within width bands and rule counts, it must not beat the
    majority by much.

    The threshold is fitted on the sample it is scored on, so the sample must be large
    enough for that optimism to stay under the bar: on 60 per task (~110 questions) a
    fitted threshold reached 0.605 against an exact 0.50 majority while scoring 0.40 on
    held-out halves -- noise, not a cue. 150 per task holds ~250."""
    rows = []
    for ex in large:
        q = ex.questions[0]
        if q.meta["family"] != "feasible" or q.meta["shape"] != "roster":
            continue
        shifts, staff, _ = _parse_roster(ex.state)
        need = dict(shifts)
        width = sum(1 for n in need for held, away in staff.values()
                    if n not in away and (need[n] is None or need[n] in held)) / len(need)
        rows.append((width, q.options[q.target]))
    assert len(rows) > 60
    maj = max(Counter(lab for _, lab in rows).values()) / len(rows)
    best = maj
    for thr in sorted({w for w, _ in rows}):
        acc = sum((w >= thr) == (lab == "yes") for w, lab in rows) / len(rows)
        best = max(best, acc, 1 - acc)
    assert best < maj + 0.10, f"width threshold {best:.2f} against majority {maj:.2f}"


def test_the_question_goes_through_the_phrasing_bank(monkeypatch):
    """Wired, and one wording per instance across its three shapes."""
    from lod.phrasings import PhrasingBank
    bank = PhrasingBank({"d19.feasible": ["Is there any way to staff the whole week "
                                          "without breaking a rule?"]})
    monkeypatch.setattr(S, "_BANK", bank)
    out = None
    for i in range(60):
        out = S.build(("max_shifts", "rest"), 0, i, "feasible")
        if out:
            break
    assert out and {ex.questions[0].question for ex in out} == {
        "Is there any way to staff the whole week without breaking a rule?"}
