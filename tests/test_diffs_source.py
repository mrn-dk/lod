"""Row 40: unified diffs and what they do to existing callers.

The two tests that matter most are the last two. Row 36 shipped 100 % solvable by exact
string match because the goal named the control verbatim, and this row's version of that
failure is the commit message: if the message says "breaking change" or the conventional
type predicts the label, the model learns a conventional-commits parser and nothing else.
Both are measured here rather than asserted in a docstring.
"""

from __future__ import annotations

import re
from collections import Counter

import pytest

from lod.corpus.services.sources.synth import diffs


@pytest.fixture(scope="module")
def tasks():
    return diffs.tasks()


def _gold(q):
    probs = q.target_probs()
    return q.options[max(range(len(probs)), key=probs.__getitem__)]


def _qs(tasks, prefix, n=40):
    for t in tasks:
        if t.name.startswith(prefix):
            for ex in t.load(n):
                for q in ex.questions:
                    yield t, ex, q


# ---- shape ---------------------------------------------------------------------------

def test_every_task_yields_what_it_is_asked_for(tasks):
    assert len(tasks) >= 12
    for t in tasks:
        got = list(t.load(25))
        assert len(got) == 25, f"{t.name} produced {len(got)}"


def test_task_names_carry_the_protected_prefix(tasks):
    # enrichment keys PROTECTED_PREFIXES on `diff_`; a task named
    # anything else would have its code-derived options rewritten onto another schema.
    for t in tasks:
        assert t.name.startswith("diff_"), t.name
        assert t.row == diffs.ROW and t.real is False


def test_targets_are_well_formed(tasks):
    for t in tasks:
        for ex in t.load(10):
            assert ex.questions
            for q in ex.questions:
                probs = q.target_probs()
                assert probs is not None and abs(sum(probs) - 1.0) < 1e-9
                assert len(probs) == len(q.options)
                assert len(set(q.options)) == len(q.options), f"{t.name}: duplicate options"


def test_every_question_carries_criteria(tasks):
    """The criteria-as-input bet: the semver definitions turned an earlier model from
    `patch` to `major` at 0.669 on the card probe. Every option carries its definition."""
    for t in tasks:
        for ex in t.load(10):
            for q in ex.questions:
                assert q.has_descriptions, t.name
                assert len(q.descriptions) == len(q.options)
                assert all(d and len(d) > 8 for d in q.descriptions), t.name


def test_generation_is_seeded(tasks):
    for t in tasks[:4]:
        assert [e.state for e in t.load(5)] == [e.state for e in t.load(5)]


def test_states_are_diffs(tasks):
    for t in tasks:
        for ex in t.load(5):
            assert "--- a/" in ex.state and "+++ b/" in ex.state
            assert "@@" in ex.state
            # a pure-deletion commit (remove_field, remove_enum_member) has no `+` lines
            assert any(l[:1] in "+-" and not l.startswith(("+++", "---"))
                       for l in ex.state.splitlines())


def test_states_are_long_enough_to_be_a_task(tasks):
    """Under 40 tokens the question is a shortcut. Whitespace words under-count
    against the Qwen tokenizer on code, so this bound is conservative."""
    for t in tasks:
        shortest = min(len(ex.state.split()) for ex in t.load(20))
        # Measured with Qwen/Qwen3-0.6B-Base over every task: the shortest state the
        # generator produces is 36 whitespace words and 93 tokens, so a 30-word floor
        # here keeps the 40-token floor with room to spare. Code tokenises far
        # denser than prose, which is why the word count is the conservative proxy.
        assert shortest >= 30, f"{t.name} has a {shortest}-word state"


# ---- the holdouts --------------------------------------------------------------------

def test_held_out_families_never_reach_training(tasks):
    for t in tasks:
        if "heldout" in t.name:
            assert t.force_split == "testreal", f"{t.name} is held out but in {t.force_split}"
        elif t.name.endswith("_dev_combo"):
            assert t.force_split == "devreal", f"{t.name} is dev-held but in {t.force_split}"
        elif t.name.endswith("_eval"):
            # the controls for the holdouts: semver and breaking in testreal beside the
            # held-out families they are compared against, compat in devreal
            assert t.force_split in ("testreal", "devreal")
        else:
            assert t.force_split == "train"
    assert any("heldout_combo" in t.name for t in tasks)
    assert any("heldout_lang" in t.name for t in tasks)
    # a held-out family is only readable against a trained one in the same split
    by_split = {t.force_split for t in tasks if t.name.startswith("diff_semver")}
    assert {"train", "testreal"} <= by_split
    assert any(t.force_split == "devreal" for t in tasks)


def test_held_out_combinations_introduce_no_unseen_edit_kind(tasks):
    """The row-33 property: a held-out family must be a new *combination* of trained
    edits, never a new edit. Otherwise a drop on it measures an unseen widget."""
    trained = set()
    for spec in diffs.SINGLE_SPECS + diffs.TRAINED_COMBOS:
        trained.update(spec)
    for combo in diffs.HELDOUT_COMBOS + diffs.DEV_COMBOS:
        assert not set(combo) - trained, f"{combo} introduces {set(combo) - trained}"
        assert combo not in diffs.TRAINED_COMBOS
        assert tuple(reversed(combo)) not in diffs.TRAINED_COMBOS


def test_dev_combinations_are_disjoint_from_test_and_training(tasks):
    """Dev holds out pairings of its own: none is a held-out pairing in either order, and
    the dev tasks emit only dev pairings while no other task emits one."""
    held = {frozenset(c) for c in diffs.HELDOUT_COMBOS}
    trained = {frozenset(c) for c in diffs.TRAINED_COMBOS}
    for combo in diffs.DEV_COMBOS:
        assert frozenset(combo) not in held | trained, combo
    dev = {frozenset(c) for c in diffs.DEV_COMBOS}
    for t in tasks:
        for ex in t.load(40):
            kinds = ex.questions[0].meta.get("kinds")
            if not kinds or not isinstance(kinds[0], str):
                continue
            if t.name.endswith("_dev_combo"):
                assert frozenset(kinds) in dev, (t.name, kinds)
            else:
                assert frozenset(kinds) not in dev, (t.name, kinds)
    from lod.corpus.services.sources.real.base import family_of
    fam = {t.name: family_of(t) for t in tasks}
    assert fam["diff_semver_eval"] == fam["diff_semver_single"] == fam["diff_semver_combo"]
    assert fam["diff_breaking_eval"] == fam["diff_breaking_single"]
    assert fam["diff_compat_eval"] == fam["diff_compat_single"]
    assert fam["diff_semver_dev_combo"] == fam["diff_breaking_dev_combo"] != \
        fam["diff_semver_heldout_combo"]


def test_held_out_language_introduces_no_unseen_edit_kind():
    """Every edit kind reachable in go is also reachable in python or typescript."""
    go_kinds = {n for n, k in diffs.KINDS.items() if diffs.HELDOUT_LANG in k.langs}
    trained_kinds = {n for n, k in diffs.KINDS.items()
                     if set(k.langs) & set(diffs.TRAINED_LANGS)}
    assert not go_kinds - trained_kinds
    assert diffs.HELDOUT_LANG not in diffs.TRAINED_LANGS


def test_every_edit_kind_is_trained_alone(tasks):
    """The combination holdout only means something if every component is trained."""
    assert set(diffs.TRAINED_SINGLES) == set(diffs.KINDS)
    for name, kind in diffs.KINDS.items():
        for lang in kind.langs:
            import random
            built = sum(1 for s in range(25)
                        if diffs.build_commit((name,), random.Random(f"{name}{lang}{s}"), lang))
            assert built >= 12, f"{name} in {lang} built {built}/25 times"


def test_held_out_language_states_are_go(tasks):
    for t in tasks:
        if t.name.endswith("heldout_lang"):
            for ex in t.load(10):
                assert ".go\n" in ex.state or ".go\t" in ex.state or ".go " in ex.state


# ---- the label -----------------------------------------------------------------------

def test_the_label_is_the_worst_edit_not_the_last(tasks):
    import random
    for combo in diffs.HELDOUT_COMBOS + diffs.DEV_COMBOS + diffs.TRAINED_COMBOS:
        c = None
        for s in range(60):
            c = diffs.build_commit(combo, random.Random(f"w{combo}{s}"), "python")
            if c:
                break
        if c is None:
            continue
        want = max((diffs.KINDS[n].level for n in combo), key=diffs.LEVELS.index)
        assert c.level == want, combo


def test_no_family_is_a_single_class(tasks):
    for prefix in ("diff_semver", "diff_breaking", "diff_compat", "diff_bump"):
        ys = Counter(_gold(q) for _, _, q in _qs(tasks, prefix, 30))
        assert len(ys) >= 2, f"{prefix} is degenerate: {ys}"
        top = max(ys.values()) / sum(ys.values())
        assert top <= 0.60, f"{prefix} majority class is {top:.0%}: {ys}"


# ---- the two that hold the line ------------------------------------------------------

def _subject(state: str) -> str:
    """Everything before the first diff header: the commit line, the optional release tag
    and the subject. Splitting on a four-space indent does NOT work -- a python or
    typescript context line in the hunk body is also four-space indented, and measuring
    the "commit message" that way silently measures the diff."""
    return state.split("--- a/")[0]


def test_the_commit_message_never_states_the_impact(tasks):
    """This row's version of the row-36 leak. `fix:` and `refactor:` are allowed and are
    drawn independently of the label; the words that would give the answer away are not.

    Checked over the whole state, not just the subject: a docstring or an error message
    inside a hunk would leak just as loudly.
    """
    for t in tasks:
        for ex in t.load(30):
            low = ex.state.lower()
            for banned in diffs.MESSAGE_BANNED:
                assert banned not in low, f"{t.name}: {banned!r} in state"


def test_the_subject_line_alone_does_not_predict_the_label(tasks):
    """The subject must be worth nothing on its own.

    The first version put a mechanical description of the edit in the subject -- "pull the
    display pieces out into a private helper" -- and a multinomial naive Bayes over that
    subject alone read the semver level at **99.78 %** against a 34 % majority. The diff
    was decoration. Subjects are now drawn before any edit runs, from templates that name
    only the area.
    """
    import math
    import random as _r
    for prefix, floor in (("diff_semver", 0.08), ("diff_breaking", 0.08),
                          ("diff_compat", 0.08)):
        data = [(_subject(ex.state), _gold(q)) for _, ex, q in _qs(tasks, prefix, 200)]
        _r.Random(0).shuffle(data)
        cut = int(0.7 * len(data))
        tr, te = data[:cut], data[cut:]
        prior = Counter(y for _, y in tr)
        cnt = {c: Counter() for c in prior}
        tot = {c: 0 for c in prior}
        vocab: set[str] = set()
        for x, y in tr:
            ws = re.findall(r"[a-z]+", x.lower())
            cnt[y].update(ws)
            tot[y] += len(ws)
            vocab.update(ws)
        V = len(vocab)

        def predict(x):
            ws = re.findall(r"[a-z]+", x.lower())
            return max(prior, key=lambda c: math.log(prior[c] / len(tr))
                       + sum(math.log((cnt[c][w] + 1) / (tot[c] + V)) for w in ws))

        acc = sum(predict(x) == y for x, y in te) / len(te)
        maj = max(Counter(y for _, y in te).values()) / len(te)
        assert acc <= maj + floor, \
            f"{prefix}: naive Bayes on the subject alone scores {acc:.2%} vs {maj:.2%}"


def test_the_answer_is_not_recoverable_by_string_match(tasks):
    """The gold option must not appear as a word in the state while the others do not.

    Measured rather than eyeballed: the shipped row 36 was 100 % solvable this way.
    """
    for prefix in ("diff_semver", "diff_breaking", "diff_compat", "diff_bump", "diff_riskiest"):
        hits = total = 0
        for _, ex, q in _qs(tasks, prefix, 25):
            low = ex.state.lower()
            present = [o for o in q.options
                       if re.search(rf"\b{re.escape(o.lower().replace('_', ' '))}\b", low)
                       or re.search(rf"\b{re.escape(o.lower())}\b", low)]
            total += 1
            hits += present == [_gold(q)]
        assert total
        assert hits / total <= 0.05, f"{prefix}: {hits}/{total} solvable by string match"


def test_the_conventional_commit_type_does_not_predict_the_label(tasks):
    """The conventional type is drawn independently of the edits, so a parser that maps
    feat->minor, fix->patch and everything else->patch must not beat the majority class."""
    ys, preds = [], []
    for _, ex, q in _qs(tasks, "diff_semver", 40):
        subject = next((l.strip() for l in ex.state.splitlines() if l.startswith("    ")), "")
        m = re.match(r"^(\w+)\(", subject)
        kind = m.group(1) if m else None
        preds.append({"feat": "minor", "fix": "patch"}.get(kind, "patch"))
        ys.append(_gold(q))
    n = len(ys)
    parser = sum(p == y for p, y in zip(preds, ys)) / n
    majority = max(Counter(ys).values()) / n
    assert parser <= majority + 0.05, \
        f"a conventional-commits parser scores {parser:.2%} against a {majority:.2%} baseline"


# ---- the question has to ask what the target answers -----------------------------------
# An earlier build shipped 2,659 of its 6,000 `breaking` questions (44.3 %) asking the
# opposite of what they scored: seven of the sixteen wordings ask whether compatibility
# is *preserved*, the descriptions are all written in break-polarity, and the target was
# `int(c.breaking)` for every wording. This is the row-40 form of the failure where a
# question asks for a threshold and the target carries a fraction.

_BREAK_WORDS = ("break", "breaking", "disrupt", "disrupted", "undermine", "fail",
                "incompatibility", "notice any break")
_PRESERVE_WORDS = ("preserve", "still work", "remain valid", "continue unchanged",
                   "rely on the module unchanged", "leave every existing correct call",
                   "remain fully", "keep working")


def _asked_polarity(question: str) -> str:
    """`break` when the question asks whether something is broken, `preserve` when it
    asks whether everything still works. Read off the wording, never off the target."""
    q = question.lower()
    breaks = any(w in q for w in _BREAK_WORDS)
    keeps = any(w in q for w in _PRESERVE_WORDS)
    if breaks == keeps:
        return "ambiguous"
    return "break" if breaks else "preserve"


def test_every_breaking_wording_asks_one_thing_and_declares_it():
    for w in diffs.WORDINGS["breaking"]:
        asked = _asked_polarity(w["question"])
        assert asked != "ambiguous", f"unreadable polarity: {w['question']!r}"
        declared = w.get("polarity", "break")
        assert asked == declared, (
            f"{w['question']!r} asks about {asked} but is declared {declared}")


def test_the_breaking_target_answers_the_question_as_asked(tasks):
    """The wording, the description attached to the target and the target itself all have
    to agree. Read the polarity out of the question text, recompute the answer from the
    edit list, and compare."""
    n = bad = 0
    for t, ex, q in _qs(tasks, "diff_breaking", n=200):
        breaking = any(diffs.KINDS[k].breaking for k in q.meta["kinds"])
        asked = _asked_polarity(q.question)
        assert asked != "ambiguous", q.question
        want = "yes" if (breaking if asked == "break" else not breaking) else "no"
        n += 1
        bad += _gold(q) != want
    assert n > 500, f"only {n} breaking questions drawn"
    assert bad == 0, f"{bad}/{n} breaking questions score the opposite of what they ask"


def test_the_break_description_travels_with_the_breaking_commit(tasks):
    """A model that reads the criteria instead of the stem must land in the same place.

    Exact, not lexical: whatever the question's polarity, the description shown against
    the answer the target picks has to be the one the wording's author wrote for the
    breaking case exactly when the commit breaks a caller. An earlier build satisfied
    this and failed `test_the_breaking_target_answers_the_question_as_asked`, which is
    what made the question the misleading half rather than the descriptions.
    """
    raw = {w["question"]: w["descriptions"] for w in diffs.WORDINGS["breaking"]}
    n = 0
    for t, ex, q in _qs(tasks, "diff_breaking", n=120):
        breaking = any(diffs.KINDS[k].breaking for k in q.meta["kinds"])
        gold_desc = q.descriptions[q.options.index(_gold(q))]
        assert gold_desc == raw[q.question]["yes" if breaking else "no"], (
            f"gold description does not match the commit: breaking={breaking} "
            f"{q.question!r} -> {gold_desc!r}")
        assert set(q.descriptions) == set(raw[q.question].values()), \
            "a description was invented rather than drawn from the wording"
        n += 1
    assert n > 300


# ---- every edit in the list has to leave a trace in the diff ---------------------------

def _hunk_lines(state):
    return [l for l in state.splitlines()
            if l[:1] in "+-" and not l.startswith(("---", "+++"))]


def _commits(spec, lang, n=40):
    import random
    for s in range(n * 4):
        c = diffs.build_commit(spec, random.Random(f"cancel{spec}{lang}{s}"), lang)
        if c is not None:
            yield c
            n -= 1
            if n == 0:
                return


def test_no_edit_is_cancelled_by_the_edit_beside_it(tasks):
    """An earlier build: held-out-combination commits whose diff shows ONE of their two
    edits.

    `("add_field", "rename_field")` renamed the field `add_field` had just appended, so
    the rendered diff showed one added field and nothing else while the label said
    `major`/breaking -- 247 of 247 such commits. `("relax_validation",
    "tighten_validation")` rewrote one bound twice, so the diff only ever showed the net
    move -- a single-line change identical to a trained single; that pair is no longer a
    holdout. The shipped model answered the first group at 0.000 accuracy with mean
    confidence 0.95. Three more pairs erased an edit without changing the label:
    `widen_param_type` then `remove_param` of the widened parameter (126 of 258),
    `add_optional_param` onto the function `deprecate_with_shim` had just created, and
    `update_docstring` on the doc `fix_private_offbyone` had just rewritten.
    """
    seen = Counter()
    for t, ex, q in _qs(tasks, "diff_", n=250):
        kinds = q.meta.get("kinds")
        if not kinds or not isinstance(kinds[0], str):
            continue
        spec = tuple(sorted(kinds))
        seen[spec] += 1
        lines = _hunk_lines(ex.state)
        if "rename_field" in spec or "remove_field" in spec:
            assert any(l.startswith("-") for l in lines), \
                f"{spec}: nothing removed, so no field was renamed or dropped"
        if "tighten_validation" in spec or "relax_validation" in spec:
            before = [float(m[1]) for l in lines if l.startswith("-")
                      for m in [re.search(r"<= ?([0-9.]+)", l)] if m]
            after = [float(m[1]) for l in lines if l.startswith("+")
                     for m in [re.search(r"<= ?([0-9.]+)", l)] if m]
            assert before and after, f"{spec}: the bound is not in the diff"
            if "tighten_validation" in spec:
                assert after[0] < before[0], \
                    f"{spec} is labelled from a tightening but the diff relaxes " \
                    f"{before[0]} -> {after[0]}"
            else:
                assert after[0] > before[0], \
                    f"{spec} relaxes nothing: {before[0]} -> {after[0]}"
    assert seen[("add_field", "rename_field")] > 0, "never drawn; the test proves nothing"
    for combo in diffs.HELDOUT_COMBOS:
        assert set(combo) != {"relax_validation", "tighten_validation"}, \
            "two edits of one bound only ever show their net move"

    # the pairs that erased an edit without moving the label, built directly
    for lang in diffs.TRAINED_LANGS + (diffs.HELDOUT_LANG,):
        for c in _commits(("widen_param_type", "remove_param"), lang):
            plus = [l for l in _hunk_lines(c.diff) if l.startswith("+")]
            assert any(re.search(r"float \| int \| str|number \| string|\bany\b", l)
                       for l in plus), f"{lang}: the widening vanished\n{c.diff}"
    for lang in diffs.TRAINED_LANGS:
        for c in _commits(("deprecate_with_shim", "add_optional_param"), lang):
            lines = _hunk_lines(c.diff)
            # the parameter lands on a signature that existed before the commit
            sig = next(l for l in lines if l.startswith("+") and "trace" in l)
            name = re.search(r"(?:def|function) (\w+)\(", sig)[1]
            assert any(l.startswith("-") and re.search(rf"\b{name}\(", l) for l in lines), \
                f"{lang}: the optional parameter went onto a new function\n{c.diff}"
        for c in _commits(("fix_private_offbyone", "update_docstring"), lang):
            assert "upstream service" in c.diff and ("+ 0" in c.diff or "+ 0;" in c.diff)
            ext = [l for l in _hunk_lines(c.diff) if l.startswith("+") and "upstream service" in l]
            assert not any("zero-based" in l for l in ext), \
                f"{lang}: the docstring edit rode on the off-by-one doc\n{c.diff}"


# ---- the label table has to say what the criteria say ----------------------------------

def test_every_kind_is_labelled_the_way_the_criteria_define_it():
    """The criteria shown with every question fix the table, column by column.

    `breaking` (all sixteen wordings): yes when a correct call stops compiling/resolving
    OR still runs with altered behaviour -- so breaking iff compat is not `compatible`.
    `semver`: major is exactly a change existing callers cannot absorb unchanged. The
    compat prompt names "a check tightened or loosened" as same_call_different_behaviour.
    `relax_validation` was `minor`/not breaking with compat `behaviour`: a label every
    criterion contradicts, on 1 of 24 kinds and in every trained combo that carried it.
    """
    for name, k in diffs.KINDS.items():
        assert k.breaking == (k.compat != "compatible"), name
        assert (k.level == "major") == k.breaking, name
    assert diffs.KINDS["relax_validation"].compat == "behaviour"
    assert diffs.KINDS["relax_validation"].breaking
    # the four tiers the risk instructions state, in the order they state them
    assert [diffs.KINDS[n].tier for n in ("remove_public_fn", "tighten_validation",
                                          "add_public_fn", "rename_local")] == [3, 2, 1, 0]


def test_an_attribute_added_to_a_record_is_optional():
    """The record is a parameter as well as a result (`format_*(item)`), so an attribute
    added as REQUIRED breaks every caller that builds one -- a python dataclass call
    missing it raises, a typescript object literal missing it does not compile -- while
    the label says additive. Before the fix every python and typescript `add_field`
    rendered `carrier: str` / `carrier: string;`. Go is exempt: adding a struct field is
    compatible under the Go 1 promise (keyed literals)."""
    for lang, pat in (("python", r"^\+    \w+: \w+ \| None = None$"),
                      ("typescript", r"^\+  \w+\?: \w+;$")):
        for c in _commits(("add_field",), lang, n=20):
            added = [l for l in _hunk_lines(c.diff)
                     if re.match(r"^\+(    |  )\w+\??: ", l) and "raw" not in l]
            assert added and all(re.match(pat, l) for l in added), f"{lang}\n{c.diff}"


def test_the_go_miss_path_is_nil_nil_before_it_becomes_an_error():
    """`raise_instead_of_none` in go used to ADD the only miss branch: before it, a miss
    fell through to decoding a nil map, which panics on the first type assertion, so the
    held-out language showed "a crash became an error" under a breaking label. Go spells
    "not found" as `nil, nil`; that line must be what the edit replaces."""
    for c in _commits(("raise_instead_of_none",), "go", n=15):
        lines = _hunk_lines(c.diff)
        assert "-\t\treturn nil, nil" in lines, c.diff
        assert any('errors.New("not found")' in l for l in lines if l.startswith("+"))


def test_a_shim_names_a_symbol_that_exists():
    """`Superseded by check_reading` in a typescript file whose function is
    `checkReading` names nothing; the doc now uses the language's own spelling."""
    for lang in ("python", "typescript", "go"):
        for c in _commits(("deprecate_with_shim",), lang, n=15):
            m = re.search(r"[Ss]uperseded by (\w+)", c.diff)
            assert m, c.diff
            assert re.search(rf"(def|function|func) {m[1]}\(", c.diff), f"{lang}\n{c.diff}"


def test_the_riskiest_change_is_alone_in_the_top_tier(tasks):
    """The risk instructions state four tiers and nothing finer. The target used to be
    the highest `Kind.risk` (95 vs 90 inside the source tier), so 168 of 400 questions
    had a runner-up the criteria rank equal to the gold. Tier is re-derived here from
    the compat/level columns, not read from `Kind.tier`."""
    def tier(names):
        def one(k):
            return {"source": 3, "behaviour": 2}.get(k.compat, 1 if k.level == "minor" else 0)
        return max(one(diffs.KINDS[n]) for n in names)
    n = 0
    for t, ex, q in _qs(tasks, "diff_riskiest", n=300):
        tiers = [tier(names) for names in q.meta["kinds"]]
        top = tiers[q.target]
        assert top == max(tiers) and tiers.count(top) == 1, (tiers, q.target)
        n += 1
    assert n >= 300


# ---- how much of the diff space the generator actually reaches --------------------------

def test_the_diff_space_does_not_collapse(tasks):
    """A floor, not a clean bill of health.

    Measured on an earlier build: 24,000 states reduce to 6,208 distinct hunk-sets (74 % of
    states repeat one), the most common appears 332 times, and 92-94 % of the states in
    the three `_eval` control tasks have a hunk-set that also appears in `train`. The
    control against which the held-out drop is read is therefore mostly a memorisation
    measurement. Widening it is a generator change with a rebuild attached; this test
    only stops it getting worse.
    """
    for t in tasks:
        if t.name == "diff_riskiest":
            continue
        states = ["\n".join(_hunk_lines(ex.state)) for ex in t.load(400)]
        distinct = len(set(states))
        top = Counter(states).most_common(1)[0][1]
        assert distinct >= 0.20 * len(states), \
            f"{t.name}: only {distinct} distinct hunk-sets in {len(states)} states"
        assert top <= 0.10 * len(states), \
            f"{t.name}: one hunk-set accounts for {top} of {len(states)} states"


def test_the_phrasing_bank_keeps_the_breaking_polarity(tasks, monkeypatch):
    """With a bank loaded, a "does it preserve" wording must draw from the preserve
    template's phrasings and a "does it break" one from the break template's -- one id
    per polarity, so a rewording can never flip what the target answers."""
    from lod.phrasings import PhrasingBank
    monkeypatch.setattr(diffs, "_BANK", PhrasingBank({
        "d18.breaking": ["Does this change break a correct caller?"],
        "d18.breaking_preserve": ["Do all correct callers keep working unchanged?"],
        "d18.semver": ["Which release level does this need?"]}))
    n = 0
    for t in diffs.tasks():
        if not t.name.startswith(("diff_breaking", "diff_semver_single")):
            continue
        for ex in t.load(40):
            q = ex.questions[0]
            if q.id.startswith("breaking"):
                want = {"break": "Does this change break a correct caller?",
                        "preserve": "Do all correct callers keep working unchanged?"}
                assert q.question == want[q.meta["polarity"]]
                n += 1
            else:
                assert q.question == "Which release level does this need?"
    assert n >= 100
