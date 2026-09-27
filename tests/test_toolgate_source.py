"""Row 38: tool-call and action gating.

The tests that matter here are the three the domain could fail silently on:

  * the gate is not a lookup table of command names (`test_no_command_predicts_the_gate`),
  * the answer is not sitting in the state as a string (`test_answer_is_not_a_string_match`),
  * the held-out families are a *structure* and introduce no primitive training never saw
    (`test_held_out_families_are_built_from_trained_primitives`).

Row 36 shipped 100 % string-match solvable and nobody noticed until it was measured, so
these are measurements with thresholds, not smoke tests.
"""

from __future__ import annotations

import collections
import json
import random
import re

import pytest

from lod.corpus.services.sources.synth import toolgate


@pytest.fixture(scope="module")
def tasks():
    return toolgate.tasks()


@pytest.fixture(scope="module")
def sample(tasks):
    """(task, example) pairs across every task, loaded once."""
    return [(t, ex) for t in tasks for ex in t.load(30)]


def _gold(q):
    probs = q.target_probs()
    return q.options[max(range(len(probs)), key=probs.__getitem__)]


# ---- the generator produces what it says --------------------------------------------

def test_every_task_yields_what_it_is_asked_for(tasks):
    for t in tasks:
        got = list(t.load(20))
        assert len(got) == 20, f"{t.name} produced {len(got)}"


def test_generation_is_seeded(tasks):
    for t in tasks[:4]:
        assert [e.state for e in t.load(5)] == [e.state for e in t.load(5)]


def test_targets_are_well_formed(sample):
    for t, ex in sample:
        assert ex.questions, f"{t.name}: example with no questions"
        for q in ex.questions:
            probs = q.target_probs()
            assert probs is not None and abs(sum(probs) - 1.0) < 1e-9
            assert len(probs) == len(q.options) == len(q.descriptions)
            assert len(set(q.options)) == len(q.options), f"{t.name}: duplicate options"
            assert all(d for d in q.descriptions), f"{t.name}: an option without criteria"


def test_json_states_parse_and_carry_the_facts_the_policy_reads(sample):
    seen = 0
    for _, ex in sample:
        if not ex.state.lstrip().startswith("{"):
            continue
        seen += 1
        body = json.loads(ex.state)
        assert "run" in body
        assert ("environment" in body) or ("environments" in body)
        assert ("proposed_call" in body) or ("proposed_calls" in body)
    assert seen > 100, f"only {seen} JSON states in the sample"


def test_both_renderings_occur(sample):
    shapes = collections.Counter("json" if ex.state.lstrip().startswith("{") else "prose"
                                 for _, ex in sample)
    assert min(shapes.values()) / sum(shapes.values()) > 0.15, shapes


def test_a_missing_seed_file_raises_rather_than_returning_nothing(monkeypatch, tmp_path):
    """An earlier build's failure: a loader that reads a path that is not there returns `[]`
    and the build reports the row as contributing nothing that anyone reads."""
    monkeypatch.setattr(toolgate, "HTTP_SPECS", tmp_path / "gone.json")
    with pytest.raises(FileNotFoundError):
        toolgate._load_http()
    monkeypatch.setattr(toolgate, "SCENARIOS", tmp_path / "gone.json")
    monkeypatch.setattr(toolgate, "_SCEN", None)
    with pytest.raises(FileNotFoundError):
        toolgate.scenario("filesystem", random.Random(0))


def test_the_http_seed_is_attributed_and_permissive():
    payload = json.loads(toolgate.HTTP_SPECS.read_text())
    assert payload["calls"] and payload["apis"]
    for api in payload["apis"]:
        assert api["licence"] and api["url"]
        assert any(p in api["licence"].lower() for p in ("apache", "mit")), api
    # method and path template only -- no descriptions, summaries or schemas
    for call in payload["calls"][:50]:
        assert set(call) == {"method", "path", "effect", "restorable", "blast", "api"}


# ---- the policy is exactly the criteria ----------------------------------------------

def test_gate_is_the_table_the_criteria_state():
    """A transcription check. If `gate()` and the option criteria drift apart the corpus
    teaches a policy nobody wrote down."""
    expected = {
        ("scratch", "one_item"): "run_now", ("scratch", "one_collection"): "run_now",
        ("scratch", "everything"): "ask_first",
        ("staging", "one_item"): "ask_first", ("staging", "one_collection"): "ask_first",
        ("staging", "everything"): "refuse",
        ("production", "one_item"): "ask_first",
        ("production", "one_collection"): "refuse",
        ("production", "everything"): "refuse",
    }
    for tier in toolgate.TIERS:
        for blast in toolgate.BLASTS:
            assert toolgate.gate(True, tier, blast) == "run_now"
            assert toolgate.gate(False, tier, blast) == expected[(tier, blast)]


def test_a_backup_counts_only_when_it_is_recent_and_covers_the_target():
    target = "/srv/exports/archive/report-1.csv"
    assert toolgate.backup_covers({"taken_hours_ago": 3, "covers": "/srv/exports"}, target)
    assert not toolgate.backup_covers({"taken_hours_ago": 40, "covers": "/srv/exports"},
                                      target)
    assert not toolgate.backup_covers({"taken_hours_ago": 3, "covers": "/srv/other"},
                                      target)
    assert not toolgate.backup_covers(None, target)


def test_rendered_targets_round_trip_through_the_blast_rule():
    """The radius is chosen first and the target rendered from it. If the rule could not
    read the radius back, the `blast` family would be labelling noise."""
    rng = random.Random(5)
    pool = list(toolgate.CLI_CALLS) + toolgate._load_http()
    for call in pool:
        for blast in call.blasts:
            for _ in range(4):
                root = toolgate.store_root(call, rng)
                target = toolgate.render_target(call, blast, root, rng)
                argv = toolgate.render_argv(call, target, False)
                got = toolgate.blast_of(target, argv, root)
                assert got == blast, f"{call.key} {blast} -> {target!r} read as {got}"


def test_the_same_command_gets_different_gates(tasks):
    """The design point. If `rm` always answered `refuse` this module would be a lookup
    table of command names.

    Counted over the CLI call templates, where 800 questions over 57 templates is enough
    observations per template to mean something; the 439 HTTP endpoints are each drawn
    well under once, so counting them here would measure the sample size.
    """
    by_call = collections.defaultdict(set)
    for t in tasks:
        if not t.name.startswith("toolgate_gate_"):
            continue
        for ex in t.load(400):
            for q in ex.questions:
                if q.meta["category"] != "http":
                    by_call[q.meta["call"]].add(_gold(q))
    multi = [k for k, v in by_call.items() if len(v) > 1]
    assert len(multi) / len(by_call) > 0.75, \
        f"only {len(multi)}/{len(by_call)} calls ever get more than one gate"
    for key in ("rm_tree", "rm_file"):
        assert len(by_call.get(key, ())) > 1, f"{key} always answers {by_call.get(key)}"


# ---- the holdout is a structure ------------------------------------------------------

def test_held_out_families_never_reach_training(tasks):
    for t in tasks:
        held = any(t.name.endswith(f"_{c}") for c in toolgate.HELDOUT_CATEGORIES)
        dev = any(t.name.endswith(f"_{c}") for c in toolgate.DEV_CATEGORIES)
        if held:
            assert t.force_split == "testreal", f"{t.name} held out but in {t.force_split}"
        elif dev:
            assert t.force_split == "devreal", f"{t.name} dev-held but in {t.force_split}"
        else:
            # eval twins are pinned to devreal (F16-7); a hashed twin landed in testreal
            want = "devreal" if t.name.endswith("_eval") else "train"
            assert t.force_split == want, f"{t.name}: {t.force_split} (want {want})"
    assert sum(1 for t in tasks if t.force_split == "testreal") >= 6


def _primitives(pool):
    """Everything the gating policy can read about a call, as a set of tokens."""
    out = set()
    for call in pool:
        out.add(f"effect:{call.effect}")
        out.add(f"reverse:{call.reverse is not None}")
        out.add(f"dry_run:{call.dry_run is not None}")
        for blast in call.blasts:
            out.add(f"blast:{blast}")
    return out


def test_held_out_families_are_built_from_trained_primitives():
    """A drop on the held-out half has to be a failure to apply the policy to an
    unfamiliar tool, not a primitive the model was never shown."""
    trained = _primitives([c for c in toolgate.CLI_CALLS
                           if c.category in toolgate.TRAINED_CATEGORIES])
    for category in toolgate.HELDOUT_CATEGORIES + toolgate.DEV_CATEGORIES:
        pool = (toolgate._load_http() if category == "http"
                else [c for c in toolgate.CLI_CALLS if c.category == category])
        assert pool, category
        missing = _primitives(pool) - trained
        assert not missing, f"{category} introduces untrained primitives {sorted(missing)}"


def test_held_out_families_exercise_every_tier_and_backup_state(tasks):
    tiers, gates = collections.Counter(), collections.Counter()
    for t in tasks:
        if t.force_split != "testreal" or not t.name.startswith("toolgate_gate_"):
            continue
        for ex in t.load(150):
            for q in ex.questions:
                tiers[q.meta["tier"]] += 1
                gates[_gold(q)] += 1
    assert set(tiers) == set(toolgate.TIERS), tiers
    assert set(gates) == set(toolgate.GATES), gates


# ---- the three ways this data could be worthless -------------------------------------

def test_answer_is_not_a_string_match(sample):
    """The gold option must not appear as a word in the state or the instructions. Row 36
    shipped at 100 % solvable this way.

    Word boundaries, not `in`: `"no" in state` is true of every state that contains
    "none" or "not", which measures English rather than leakage.
    """
    hits = total = 0
    for t, ex in sample:
        for q in ex.questions:
            if q.id == "first":
                continue  # option ids are call ids, in the state by construction
            total += 1
            haystack = ex.state + " " + (q.instructions or "")
            hits += bool(re.search(rf"\b{re.escape(_gold(q))}\b", haystack))
    assert total
    assert hits == 0, f"{hits}/{total} golds appear as a word in their own state"


def test_no_option_key_appears_in_the_state_at_all(sample):
    for t, ex in sample:
        for q in ex.questions:
            if q.id == "first":
                continue
            for option in q.options:
                assert not re.search(rf"\b{re.escape(option)}\b", ex.state), \
                    f"{t.name}: {option!r} is in the state"


def test_no_command_predicts_the_gate(tasks):
    """The best per-*command* majority guess -- the score of the best possible lookup
    table keyed on the name of the tool being invoked, which is what `meta.tool` is:
    `rm`, `git`, `apt-get` for the CLI half, the HTTP method for the API half.

    An oracle with that table and nothing else should be well short of the policy. It is
    not zero-skill: nine of the 57 CLI templates have effect `read`, and the policy says
    those are always `run_now`, which is the policy being right rather than the data
    being a lookup table.
    """
    by_tool = collections.defaultdict(collections.Counter)
    for t in tasks:
        if not t.name.startswith("toolgate_gate_"):
            continue
        for ex in t.load(400):
            for q in ex.questions:
                by_tool[q.meta["tool"]][_gold(q)] += 1
    total = sum(sum(c.values()) for c in by_tool.values())
    best = sum(c.most_common(1)[0][1] for c in by_tool.values())
    assert best / total < 0.70, f"per-command majority scores {best / total:.2%}"


def test_majority_class_baseline_is_not_the_task(tasks):
    """A gate that is 85 % one answer teaches the marginal, not the policy."""
    for family in ("gate", "undo", "blast"):
        counts = collections.Counter()
        for t in tasks:
            if not t.name.startswith(f"toolgate_{family}_"):
                continue
            for ex in t.load(150):
                for q in ex.questions:
                    counts[_gold(q)] += 1
        share = counts.most_common(1)[0][1] / sum(counts.values())
        assert share < 0.62, f"{family}: majority class {counts.most_common(1)[0][0]} " \
                             f"at {share:.2%} ({dict(counts)})"


def test_states_are_long_enough_to_be_a_task(sample):
    """The floor is 40 Qwen tokens. A JSON state tokenises to well over one token per
    whitespace word -- braces, quotes, paths and flags all split -- so a word floor is a
    strict lower bound. Measured once with `Qwen/Qwen3-0.6B-Base` over the whole sample:
    see `tests/test_toolgate_tokens` note in the domain findings."""
    words = sorted(len(ex.state.split()) for _, ex in sample)
    assert words[0] >= 45, f"shortest state is {words[0]} words"
    assert words[len(words) * 95 // 100] < 900, "p95 state is close to the 2304-token cap"


def test_task_names_carry_the_protected_prefix(tasks):
    from lod.corpus.services.enrich import protected
    for t in tasks:
        assert t.name.startswith("toolgate_"), t.name
        assert protected(t.name), f"{t.name} would be rewritten by the enricher"
        assert t.row == toolgate.ROW and t.real is False


# ---- the state says what the allow-list is for ---------------------------------------
# Added by the row-38 audit of an earlier corpus build. That build predates the rename
# below and still ships the old label, so nothing but a test stops the rename being
# lost again.

OLD_LABELS = ("Tools this harness may run here", '"tools_allowed"',
              "one of the environment's allowed tools")


def test_the_allow_list_says_it_is_for_undoing(sample):
    """The allow-list rename, pinned.

    The list is the palette a *reversing* call must be drawn from -- the only thing
    `undoable()` uses it for. Labelled "Tools this harness may run here" it reads as a
    membership test on the forward command, which the policy never makes, and a reviewer
    who "fixed" the policy to make it one collapsed `run_now`. The label is the fix, so
    it is the thing to hold.
    """
    json_seen = prose_seen = 0
    for t, ex in sample:
        for label in OLD_LABELS:
            assert label not in ex.state, f"{t.name}: state still carries {label!r}"
        for q in ex.questions:
            for d in q.descriptions:
                assert "allowed tools" not in d, f"{t.name}: criteria still say 'allowed tools'"
    for t, ex in sample:
        if ex.state.lstrip().startswith("{"):
            body = json.loads(ex.state)
            envs = ([body["environment"]] if "environment" in body
                    else list(body["environments"].values()))
            for env in envs:
                assert "undo_tools_allowed" in env, f"{t.name}: JSON env lost the key"
                assert "tools_allowed" not in env
            json_seen += 1
        else:
            assert "Commands this harness may run to undo work here:" in ex.state
            prose_seen += 1
    assert json_seen > 50 and prose_seen > 50, (json_seen, prose_seen)


def test_the_forward_command_is_not_on_the_allow_list_and_that_is_the_point(tasks):
    """The measurement behind the rename, as a test.

    If the allow-list were the set of commands the harness may invoke, the proposed
    call's own command would have to be on it -- and it is absent from most of them
    whatever the gate says, including most of the calls the policy answers `run_now`.
    Requiring it would turn three quarters of `run_now` into `ask_first` or `refuse`.
    """
    absent = collections.Counter()
    total = collections.Counter()
    for t in tasks:
        if not t.name.startswith("toolgate_gate_"):
            continue
        for ex in t.load(300):
            body = json.loads(ex.state) if ex.state.lstrip().startswith("{") else None
            if body is None:
                continue
            allowed = body["environment"]["undo_tools_allowed"]
            forward = body["proposed_call"]["argv"].split()[0]
            for q in ex.questions:
                total[_gold(q)] += 1
                absent[_gold(q)] += forward not in allowed
    assert sum(total.values()) > 300
    for gate in ("ask_first", "refuse"):
        share = absent[gate] / total[gate]
        assert share > 0.80, (
            f"the forward command is on the allow-list in {1 - share:.0%} of {gate} "
            f"cases; the list has started to behave like a membership test")
    # and it is absent from most `run_now` too, which is why requiring it breaks the row
    assert absent["run_now"] / total["run_now"] > 0.5, (
        "the forward command is now usually allow-listed on run_now, so 'the forward "
        "command must be allowed' would no longer be a visibly wrong reading")


# ---- every environment field the criteria name changes an answer ---------------------

def test_each_clause_of_the_undo_criteria_is_load_bearing(tasks):
    """An ablation, not an assertion. Drop one clause of `undoable` and the gate must
    move: a clause that changes nothing is a clause the model can learn to ignore, and
    the environment stops being what the answer depends on."""
    rows = []
    for t in tasks:
        if not t.name.startswith("toolgate_gate_"):
            continue
        for ex in t.load(300):
            if not ex.state.lstrip().startswith("{"):
                continue
            body = json.loads(ex.state)
            env, call = body["environment"], body["proposed_call"]
            rows.append((env, call, _gold(ex.questions[0])))
    assert len(rows) > 300

    def scored(undo_fn):
        ok = 0
        for env, call, gold in rows:
            blast = toolgate.blast_of(call["target"], call["argv"], env["store_root"])
            ok += toolgate.gate(undo_fn(env, call), env["tier"], blast) == gold
        return ok / len(rows)

    def full(env, call):
        return toolgate.undoable(
            _Call(call), call["argv"], call["reverses_with"], call["effect"],
            env["last_backup"], call["target"], env["undo_tools_allowed"])

    assert scored(full) == 1.0, "the shipped label is not the policy the code applies"
    # the allow-list: assume any reversing call is usable. Measured 0.887 on this
    # sample and 0.833 over the whole of an earlier build; the threshold asks only that the
    # clause be worth more than five points.
    assert scored(lambda e, c: full(e, c) or bool(c["reverses_with"])) < 0.95
    # the backup: assume any backup counts, however old and whatever it covers
    assert scored(lambda e, c: (c["effect"] in ("overwrite", "delete")
                                and bool(e["last_backup"])) or full(e, c)) < 0.90
    # the tier and the radius
    assert scored(lambda e, c: False) < 0.90


class _Call:
    """The two fields `undoable` reads off a `Call`, taken from the rendered state."""

    def __init__(self, block):
        self.effect = block["effect"]
        self.dry_run = block["tool_dry_run_flag"]


def test_the_blast_criteria_do_not_claim_the_target_is_under_the_store_root(tasks):
    """A git ref and a process pattern are `one_collection` and sit under no store root.
    The clause used to say "a group under the store root", which is false for a fifth of
    them -- the kind of criteria/label disagreement that teaches the wrong feature."""
    assert "under the store root" not in toolgate.CRITERIA["blast"]["one_collection"]
    off_root = on_root = 0
    for t in tasks:
        if not t.name.startswith("toolgate_blast_"):
            continue
        for ex in t.load(200):
            body = json.loads(ex.state) if ex.state.lstrip().startswith("{") else None
            if body is None or _gold(ex.questions[0]) != "one_collection":
                continue
            target = body["proposed_call"]["target"] or ""
            root = body["environment"]["store_root"].rstrip("/")
            off_root += not target.startswith(root)
            on_root += target.startswith(root)
    assert off_root > 0 and on_root > 0, (off_root, on_root)


def test_no_option_sits_at_a_fixed_index(tasks):
    """An earlier build shipped every gate question with `run_now` at slot 0, `ask_first`
    at 1 and `refuse` at 2, and the criteria the same bytes in all 16,964 of them. With
    the order fixed a slot *is* a class and the option text need never be read; row 37
    rotates for exactly this reason. `first` is excluded: its options are the call ids
    the state itself uses.
    """
    by_family = collections.defaultdict(collections.Counter)
    for t in tasks:
        for ex in t.load(200):
            for q in ex.questions:
                if q.id == "first":
                    continue
                by_family[q.id][q.options[_gold(q)] if isinstance(_gold(q), int)
                                 else q.options.index(_gold(q))] += 1
    assert set(by_family) >= {"gate", "undo", "blast", "gate_a", "gate_b"}
    for family, counts in by_family.items():
        n = sum(counts.values())
        k = len(counts)
        assert k > 1, f"{family}: gold is always at slot {list(counts)[0]}"
        worst = max(counts.values()) / n
        assert worst < 1.0 / k + 0.12, f"{family}: gold position is not flat: {counts}"


def test_the_criteria_follow_the_options_they_are_attached_to(tasks):
    """A rotation that moved the options and left the criteria behind would attach every
    clause to the wrong option -- silently, because both lists are still the right
    length."""
    for t in tasks:
        for ex in t.load(60):
            for q in ex.questions:
                if q.id == "first":
                    continue
                key = "gate" if q.id.startswith("gate") else q.id
                for option, description in zip(q.options, q.descriptions):
                    assert description == toolgate.CRITERIA[key][option], \
                        f"{t.name}: {option!r} carries another option's criteria"


def test_dev_category_is_disjoint_from_train_and_test(tasks):
    """Dev holds out a category of its own: never trained, never tested, and its calls
    use commands no trained or test-held-out category uses."""
    assert not set(toolgate.DEV_CATEGORIES) & set(toolgate.TRAINED_CATEGORIES
                                                    + toolgate.HELDOUT_CATEGORIES)
    dev_cmds = {c.command for c in toolgate.CLI_CALLS if c.category in toolgate.DEV_CATEGORIES}
    other = {c.command for c in toolgate.CLI_CALLS if c.category not in toolgate.DEV_CATEGORIES}
    assert dev_cmds and not dev_cmds & other
    dev = [t for t in tasks if t.force_split == "devreal" and not t.name.endswith("_eval")]
    assert {t.name for t in dev} == {f"toolgate_{f}_accounts" for f in ("gate", "undo", "first")}
    gates = collections.Counter()
    for t in dev:
        got = list(t.load(60))
        assert len(got) == 60, t.name
        for ex in got:
            for q in ex.questions:
                assert 0 <= q.target < len(q.options)
                if q.id == "gate":
                    gates[q.options[q.target]] += 1
    assert set(gates) == set(toolgate.GATES), gates
