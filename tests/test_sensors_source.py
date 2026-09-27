"""Row 37: control and sensor -- the action depends on the series, not on the reading.

The tests that matter here are the last three. Everything above them is hygiene; those
three are the domain's reason to exist, and if any of them goes red the row is a
label-recovery task wearing a time series and should not ship.
"""

from __future__ import annotations

import collections
import json
import random
import re

import pytest

from lod.corpus.services.sources.synth import sensors


@pytest.fixture(scope="module")
def tasks():
    return sensors.tasks()


@pytest.fixture(scope="module")
def sample(tasks):
    """(task name, example) for a few hundred of each. Built once: six tasks at 300 is
    1,800 simulations and the history tests need a sample, not a spot check."""
    return [(t.name, ex) for t in tasks for ex in t.load(300)]


def _gold(q):
    probs = q.target_probs()
    return q.options[max(range(len(probs)), key=probs.__getitem__)]


# ---- hygiene ---------------------------------------------------------------------

def test_every_task_yields_what_it_is_asked_for(tasks):
    assert len(tasks) == 8
    for t in tasks:
        got = list(t.load(20))
        assert len(got) == 20, f"{t.name} produced {len(got)}"
        assert all(e.task == t.name for e in got)


def test_rows_and_realness(tasks):
    for t in tasks:
        assert t.row == 37 and t.real is False and t.soft is True
        assert t.name.startswith("sensor_"), \
            f"{t.name} is outside PROTECTED_PREFIXES and the enricher would rewrite it"


def test_targets_are_well_formed(sample):
    for name, ex in sample:
        for q in ex.questions:
            probs = q.target_probs()
            assert probs is not None and abs(sum(probs) - 1.0) < 1e-9
            assert len(probs) == len(q.options) == 5
            assert len(set(q.options)) == 5, f"{name}: duplicate options"
            assert set(q.options) == set(sensors.ACTIONS)
            assert q.has_descriptions and len(q.descriptions) == 5
            assert all(d for d in q.descriptions)
            assert q.instructions


def test_generation_is_seeded(tasks):
    t = tasks[0]
    assert [e.state for e in t.load(5)] == [e.state for e in t.load(5)]


def test_no_option_sits_at_a_fixed_index(sample):
    """The options are rotated, so position must not carry the answer."""
    idx = collections.Counter(ex.questions[0].options.index(_gold(ex.questions[0]))
                              for _, ex in sample)
    share = [idx[i] / len(sample) for i in range(5)]
    assert max(share) < 0.30, f"gold position is not flat: {share}"


# ---- three shapes of one simulation ----------------------------------------------

def test_all_three_render_shapes_appear_in_every_task(sample):
    by = collections.defaultdict(collections.Counter)
    for name, ex in sample:
        by[name][ex.questions[0].meta["shape"]] += 1
    for name, counts in by.items():
        assert set(counts) == set(sensors.SHAPES), f"{name}: {dict(counts)}"
        assert min(counts.values()) / sum(counts.values()) > 0.20


def test_json_shape_parses_and_carries_the_series(sample):
    for name, ex in sample:
        if ex.questions[0].meta["shape"] != "json":
            continue
        body = json.loads(ex.state)
        assert body["readings"] and "value" in body["readings"][0]
        assert body["unit"] and body["measure"]


def test_the_three_shapes_are_the_same_task(sample):
    """One simulation, three renderings: every reading has to survive each of them.

    If a shape dropped or rounded a value the question would stop being the same
    question and the shape comparison would measure parsing instead of control.
    """
    plant = sensors.PLANTS["jacket_temp"]
    rng = random.Random(11)
    values = sensors._series(plant, "ramp_up", 74.0, 1.5, rng, 12)
    events = [(3, "alarm_raised")]
    framing = sensors.FRAMINGS["jacket_temp"][0]
    for shape in sensors.SHAPES:
        text = sensors._render(plant, framing, values, events, 480, shape)
        for v in values:
            assert sensors._num(plant, v) in text, f"{shape} lost {v}"
        assert "alarm" in text.lower()


def test_states_are_long_enough_to_be_a_task(sample):
    """The floor is 40 Qwen tokens.

    Word count is a strict lower bound on the token count for this text -- every reading
    is a number plus a unit, which tokenise into several pieces each -- so a word floor
    holds the line without loading a tokenizer into a unit test. Measured once with
    `Qwen/Qwen3-0.6B-Base` over 1,200 states: min 149, p5 177, p50 253, p95 438, max
    501, comfortably above the floor and comfortably inside `--max-state-tokens 2304`.
    The shortest state is a nine-row table at 50 words.
    """
    words = sorted(len(ex.state.split()) for _, ex in sample)
    assert words[0] >= 50, f"shortest state is {words[0]} words"


# ---- the holdout is a structure ----------------------------------------------------

def test_held_out_families_never_reach_training(tasks):
    by = {t.name: t.force_split for t in tasks}
    assert by == {
        "sensor_action_train_thermal": "train",
        "sensor_action_train_store": "train",
        "sensor_action_trained_dev": "devreal",
        "sensor_action_trained_eval": "testreal",
        "sensor_action_heldout_plant": "testreal",
        "sensor_action_heldout_trend": "testreal",
        "sensor_action_dev_plant": "devreal",
        "sensor_action_dev_trend": "devreal",
    }
    # dev selects on its own holdouts, never on test's
    for t in tasks:
        if t.name.startswith("sensor_action_heldout"):
            assert t.force_split == "testreal"


def test_dev_holdouts_are_disjoint_from_test_and_training(tasks):
    """The dev split's held-out plant and trend are neither trained nor tested on."""
    dev_plants = {p.key for p in sensors.DEV_PLANTS}
    assert not dev_plants & {p.key for p in sensors.TRAINED_PLANTS + sensors.HELDOUT_PLANTS}
    assert not set(sensors.DEV_TRENDS) & set(sensors.TRAINED_TRENDS + sensors.HELDOUT_TRENDS)
    seen = collections.defaultdict(set)
    for t in tasks:
        for ex in t.load(150):
            m = ex.questions[0].meta
            seen[t.force_split] |= {m["plant"], m["trend"]}
    dev_only = dev_plants | set(sensors.DEV_TRENDS)
    assert dev_only <= seen["devreal"]
    assert not dev_only & (seen["train"] | seen["testreal"])


def test_twins_share_the_trained_family(tasks):
    from lod.corpus.services.sources.real.base import family_of
    fam = {t.name: family_of(t) for t in tasks}
    trained = {fam["sensor_action_train_thermal"], fam["sensor_action_train_store"]}
    assert trained == {fam["sensor_action_trained_dev"]} == {fam["sensor_action_trained_eval"]}
    held = [fam[n] for n in fam if "heldout" in n or "_dev_" in n]
    assert len(set(held)) == 4 and not set(held) & trained


def test_held_out_plants_are_never_simulated_in_training(tasks):
    trained = {ex.questions[0].meta["plant"]
               for t in tasks if t.force_split == "train" for ex in t.load(200)}
    held = {p.key for p in sensors.HELDOUT_PLANTS}
    assert not (trained & held), f"held-out plants leaked into training: {trained & held}"
    assert trained == {p.key for p in sensors.TRAINED_PLANTS}


def test_held_out_trend_is_never_simulated_in_training(tasks):
    trained = {ex.questions[0].meta["trend"]
               for t in tasks if t.force_split == "train" for ex in t.load(200)}
    assert not (trained & set(sensors.HELDOUT_TRENDS))
    assert trained == set(sensors.TRAINED_TRENDS)


def test_held_out_families_use_no_primitive_training_never_saw(tasks):
    """The point of the holdout. A held-out family must be a new *combination* of parts
    the model has met, not a new part -- otherwise a drop on it measures unseen
    vocabulary rather than a failure to generalise over structure.
    """
    # trends: segments
    trained_segments = set().union(*(sensors.SEGMENTS[s] for s in sensors.TRAINED_TRENDS))
    for s in sensors.HELDOUT_TRENDS + sensors.DEV_TRENDS:
        new = set(sensors.SEGMENTS[s]) - trained_segments
        assert not new, f"{s} introduces unseen segments {new}"

    # plants: the structural primitives a plant is made of
    def shape_of(p):
        return (p.direction, p.integral, p.lo > p.rmin, p.hi < p.rmax,
                (p.crit > p.hi) if p.direction == "high" else (p.crit < p.lo))
    trained_shapes = {shape_of(p) for p in sensors.TRAINED_PLANTS}
    for p in sensors.HELDOUT_PLANTS + sensors.DEV_PLANTS:
        assert shape_of(p) in trained_shapes, f"{p.key} is a plant shape never trained"

    # surface: nothing a held-out task emits may be a kind of thing training never saw
    trained_tasks = [t for t in tasks if t.force_split == "train"]
    held_tasks = [t for t in tasks if t.name.startswith(("sensor_action_heldout",
                                                          "sensor_action_dev_"))]
    seen = {(ex.questions[0].meta["shape"], _gold(ex.questions[0]))
            for t in trained_tasks for ex in t.load(300)}
    for t in held_tasks:
        for ex in t.load(200):
            key = (ex.questions[0].meta["shape"], _gold(ex.questions[0]))
            assert key in seen, f"{t.name} emits {key}, unseen in training"


def test_the_control_and_the_holdouts_are_the_same_difficulty_on_paper(tasks):
    """A held-out number means nothing without a control. The trained-family testreal
    task must look like the held-out ones on everything except the held-out structure,
    or a gap could be read as the eval split simply being harder."""
    stats = {}
    for t in tasks:
        gs = [_gold(ex.questions[0]) for ex in t.load(400)]
        c = collections.Counter(gs)
        stats[t.name] = c.most_common(1)[0][1] / len(gs)
    control = stats["sensor_action_trained_eval"]
    for name in ("sensor_action_heldout_plant", "sensor_action_heldout_trend",
                 "sensor_action_dev_plant", "sensor_action_dev_trend"):
        assert abs(stats[name] - control) < 0.10, f"{name} {stats[name]} vs {control}"


# ---- the three that are the point --------------------------------------------------

def test_the_answer_is_not_recoverable_by_string_match(sample):
    """Row 36 shipped at 100 % solvable this way and nobody noticed until it was
    measured. The gold action's key may appear in its own description and nowhere
    else -- not in the state, not in the question, not in the instructions."""
    hits = 0
    for name, ex in sample:
        for q in ex.questions:
            gold = _gold(q)
            blob = ex.state + "\n" + q.question + "\n" + (q.instructions or "")
            hits += gold in blob or gold.replace("_", " ") in blob.lower()
    assert hits == 0, f"{hits}/{len(sample)} answers appear verbatim outside their option"


def test_no_single_class_is_the_answer(sample):
    c = collections.Counter(_gold(ex.questions[0]) for _, ex in sample)
    n = len(sample)
    assert c.most_common(1)[0][1] / n < 0.45, dict(c)
    assert min(c.values()) / n > 0.05, dict(c)
    assert len(c) == 5


def test_some_targets_are_genuinely_soft(sample):
    """The corpus has a 10 % soft-target floor and the physics supplies honest softness: a
    reading sitting on a band edge is a probability, not a class."""
    maxp = [max(ex.questions[0].target_probs()) for _, ex in sample]
    share = sum(1 for m in maxp if m < 0.97) / len(maxp)
    assert share > 0.10, f"only {share:.1%} of targets are non-degenerate"
    assert sum(1 for m in maxp if m > 0.99) / len(maxp) > 0.50, \
        "almost nothing is decided; the policy has gone mushy"


def test_the_reading_alone_does_not_give_the_action(sample):
    """The domain's whole contribution, measured rather than asserted.

    Three oracles over the same examples, each allowed to memorise the best answer for
    every value of its feature -- an upper bound on what any classifier could do with
    that feature:

        the last reading, normalised to its band      barely above the majority class
        + the sign of the rate of change              better, still far short
        + run length, latched alarm, projection       essentially exact

    If the first were high, the same reading would imply the same action and this row
    would be teaching nothing the rest of the corpus does not already teach.
    """
    def oracle(feature):
        buckets = collections.defaultdict(collections.Counter)
        for _, ex in sample:
            q = ex.questions[0]
            buckets[feature(q.meta)][_gold(q)] += 1
        return sum(b.most_common(1)[0][1] for b in buckets.values()) / len(sample)

    # the band and limit are this example's, not the plant's: they vary per example
    def z(m):
        return min(max(int(((m["last"] - m["lo"]) / (m["hi"] - m["lo"]) + 0.5) * 8), -4), 12)

    def projected_past(m):
        p = sensors.PLANTS[m["plant"]]
        return (m["projected"] > m["crit"]) if p.direction == "high" \
            else (m["projected"] < m["crit"])

    # rule 4's in-band limb: projected out of the band within the horizon. Omitted from
    # the full oracle until a shortcut audit found "inside and moving
    # toward the edge" stood in for it; since the generator draws the near miss (inside,
    # drifting toward the edge, projected to stay in), the clause has to be in the key.
    def projected_out(m):
        return (m["rate"] > 0 and m["projected"] > m["hi"]) or \
            (m["rate"] < 0 and m["projected"] < m["lo"])

    majority = collections.Counter(
        _gold(ex.questions[0]) for _, ex in sample).most_common(1)[0][1] / len(sample)
    value_only = oracle(z)
    plus_trend = oracle(lambda m: (z(m), m["rate"] > 0))
    full = oracle(lambda m: (z(m), m["rate"] > 0, m["runs_outside"] >= m["k"],
                             m["alarm_latched"], projected_past(m), projected_out(m)))

    assert value_only - majority < 0.25, (
        f"the last reading alone reaches {value_only:.3f} against a majority baseline "
        f"of {majority:.3f}: the history is not load-bearing")
    assert full > 0.90, f"the full series only reaches {full:.3f}; the policy is not "\
        "recoverable from the state and the targets would be noise"
    assert full - plus_trend > 0.15, (
        f"trend adds {full - plus_trend:.3f} beyond value+sign(rate); recent history "
        f"(run length, latched alarm, projection) must carry real weight")


# ---- the rendered state is enough ------------------------------------------------
# Added by the row-37 audit of an earlier corpus build. Nothing here read the state back
# before: the shape tests check that every number survives the rendering, which is not
# the same as checking that the policy can be re-derived from what survived.

_NUM = r"(-?\d+(?:\.\d+)?)"


def _read_back(state):
    """(values, alarm events) recovered from a rendered state, whatever the shape."""
    if state.lstrip().startswith("{"):
        body = json.loads(state)
        times = [r["t"] for r in body["readings"]]
        return ([r["value"] for r in body["readings"]],
                [(times.index(e["t"]), e["event"]) for e in body["events"]])
    lines = state.split("\n")
    values, events = [], []
    if lines[0].startswith("Operator log"):
        for line in lines[1:]:
            m = re.match(r"^\d\d:\d\d\s\s(.*)$", line)
            if not m:
                continue
            rest = m.group(1)
            if rest.startswith("ALARM RAISED"):
                events.append((len(values) - 1, "alarm_raised"))
            elif rest.startswith("Alarm on this loop cleared"):
                events.append((len(values) - 1, "alarm_cleared"))
            else:
                values.append(float(re.search(_NUM + r"\s+.+\.$", rest).group(1)))
        return values, events
    for line in lines:
        if not re.match(r"^\d\d:\d\d\s", line):
            continue
        cells = [c.strip() for c in line.split("|")]
        values.append(float(cells[1]))
        if cells[2] == "alarm raised":
            events.append((len(values) - 1, "alarm_raised"))
        elif cells[2] == "alarm cleared":
            events.append((len(values) - 1, "alarm_cleared"))
    return values, events


def _read_thresholds(q):
    """band, critical limit and horizon from the instructions; `k` from rule 2."""
    m = re.search(r"band for this loop is " + _NUM + r" to " + _NUM +
                  r" (.+?) and the critical limit is " + _NUM, q.instructions)
    lo, hi, crit = float(m.group(1)), float(m.group(2)), float(m.group(4))
    k = h = direction = None
    for d in q.descriptions:
        m = re.search(r"outside the band for (\d+) or more consecutive", d)
        if m:
            k = int(m.group(1))
        m = re.search(r"projects it (above|below) that limit within (\d+) readings", d)
        if m:
            direction, h = ("high" if m.group(1) == "above" else "low"), int(m.group(2))
    assert None not in (k, h, direction), "the criteria stopped carrying k, h or the side"
    return lo, hi, crit, direction, k, h


def _decide_from_text(values, events, lo, hi, crit, direction, k, h):
    """The five rules, ordered, with hard edges. An independent transcription of the
    policy from the words the example gives the model -- not a call into `sensors`."""
    last, r = values[-1], sensors.rate(values)
    projected = last + r * h
    if (last > crit or projected > crit) if direction == "high" \
            else (last < crit or projected < crit):
        return "shut_down_the_loop"
    outside = last < lo or last > hi
    improving = (r < 0) if last > (hi + lo) / 2 else (r > 0)
    if outside:
        if improving:
            return "hold_and_watch"
        runs = 0
        for v in reversed(values):
            if v < lo or v > hi:
                runs += 1
            else:
                break
        latched = False
        for _, kind in events:
            latched = kind == "alarm_raised"
        return "escalate_to_duty_engineer" if (runs >= k or latched) \
            else "correct_the_setpoint"
    if (r > 0 and projected > hi) or (r < 0 and projected < lo):
        return "hold_and_watch"
    return "leave_the_loop_alone"


def test_the_target_is_derivable_from_the_rendered_state_alone(sample):
    """Derivability, end to end and per shape.

    The state, the instructions and the option criteria have to carry every number the
    policy reads -- the series, the band, the critical limit, the horizon and the
    run-length threshold -- or the target is not something a reader could reach and the
    row is teaching noise. Hard edges disagree with the soft target only where the soft
    target is genuinely undecided, so the floor is high but not 1.0.
    """
    by_shape = collections.Counter()
    ok = collections.Counter()
    for name, ex in sample:
        q = ex.questions[0]
        values, events = _read_back(ex.state)
        assert values, f"{name}: no readings recovered from a {q.meta['shape']} state"
        got = _decide_from_text(values, events, *_read_thresholds(q))
        by_shape[q.meta["shape"]] += 1
        ok[q.meta["shape"]] += got == _gold(q)
    assert set(by_shape) == set(sensors.SHAPES)
    for shape, n in by_shape.items():
        assert ok[shape] / n > 0.98, \
            f"{shape}: only {ok[shape]}/{n} targets re-derive from the rendered state"


def test_the_three_shapes_carry_the_same_series(sample):
    """Read back, not just present. `_num` prints `120` for an integral plant and `72.6`
    otherwise, and a shape that lost a digit would still pass a substring check."""
    plant = sensors.PLANTS["queue_backlog"]
    rng = random.Random(5)
    values = sensors._series(plant, "ramp_up", 118.0, 6.0, rng, 11)
    events = [(2, "alarm_raised"), (7, "alarm_cleared")]
    framing = sensors.FRAMINGS["queue_backlog"][0]
    for shape in sensors.SHAPES:
        got, ev = _read_back(sensors._render(plant, framing, values, events, 300, shape))
        assert got == values, f"{shape} read back as {got}"
        assert ev == events, f"{shape} read back events as {ev}"


# ---- how much history, and which part of it --------------------------------------

def test_three_readings_are_not_the_whole_series(sample):
    """`rate` reads three readings. If that were the whole of the history the run-length
    test and the latched alarm would be decoration, so measure the gap: the policy
    applied to a three-reading window against the policy applied to the series."""
    short = full = 0
    for name, ex in sample:
        q = ex.questions[0]
        values, events = _read_back(ex.state)
        thresholds = _read_thresholds(q)
        gold = _gold(q)
        window = values[-3:]
        cut = len(values) - 3
        short += _decide_from_text(window, [(i - cut, kind) for i, kind in events
                                            if i >= cut], *thresholds) == gold
        full += _decide_from_text(values, events, *thresholds) == gold
    n = len(sample)
    assert full / n > 0.98, f"the series itself only reaches {full / n:.3f}"
    assert (full - short) / n > 0.05, (
        f"three readings already reach {short / n:.3f} against {full / n:.3f}; the run "
        f"length and the alarm have stopped mattering")


def test_the_latched_alarm_decides_on_its_own_often_enough_to_be_learnable(tasks):
    """The clause that is not in the numbers.

    An alarm raised earlier and never cleared lifts rule 3 to rule 2 on a reading whose
    run length is too short to do it. Measured on an earlier build that was true of 0.6 % of
    examples -- the alarm was almost always a second reason for an answer the run-length
    test had already given, which is a clause a model can ignore at no cost. The
    `escalate_alarm` intent draws the case deliberately; this is its floor.
    """
    decisive = seen = 0
    for t in tasks:
        for ex in t.load(400):
            q = ex.questions[0]
            seen += 1
            if _gold(q) != "escalate_to_duty_engineer":
                continue
            decisive += q.meta["alarm_latched"] and q.meta["runs_outside"] < q.meta["k"]
    assert seen > 1000
    assert decisive / seen > 0.02, (
        f"the latched alarm is the only reason for the answer in {decisive / seen:.3%} "
        f"of examples; rule 2's second limb is not being exercised")
