"""Row 42: multi-hop over linked records, and ranking across records.

The tests that matter here are the three the brief names as the ways this domain can be
decorative rather than real:

- the **one-hop shortcut**. If a reader who follows one reference and stops does much
  better than chance, the chain is not load-bearing. It is measured, per family and per
  hop count, and required to be zero.
- **string-match solvability**. Row 36 shipped at 100 % solvable this way and nobody
  noticed until it was measured. Two tests hold that line, copied from
  `tests/test_browser_source.py`.
- the **state-only check for ranking**. If an option can be eliminated without comparing
  across records, `ranking` is back to being a predicate on one record, which is the
  shape that already scores 0.267.

And one more that no measurement can replace: `test_labels_are_rederivable_from_the_state`
re-follows every chain from the *rendered JSON*, with no access to the generator's
internals, and checks the target. A wrong labeller silently poisons the whole row.
"""

from __future__ import annotations

import collections
import json
import re

import pytest

from lod import sentinel
from lod.corpus.services.sources.synth import relational as R


@pytest.fixture(scope="module")
def tasks():
    return R.tasks()


@pytest.fixture(scope="module")
def rows(tasks):
    """(task, example, question) for a fixed sample of every task."""
    out = []
    for t in tasks:
        for ex in t.load(40):
            out.append((t, ex, ex.questions[0]))
    return out


def _gold(q):
    return q.options[q.target]


def _appears(needle: str, hay: str) -> bool:
    """Word-boundary containment. Plain `in` reports "no" inside "nothing", which is how
    a two-option family measured 0.26 % string-match solvable until it was fixed."""
    return re.search(r"(?<![\w-])" + re.escape(needle) + r"(?![\w-])", hay) is not None


def _body(ex):
    return ex.state.split("\n\n", 1)[1]


# ---- the basics ----------------------------------------------------------------------

def test_every_task_yields_what_it_is_asked_for(tasks):
    for t in tasks:
        got = list(t.load(30))
        assert len(got) == 30, f"{t.name} produced {len(got)}"


def test_tasks_are_registered_the_way_the_row_needs(tasks):
    # 15 trained cells and their twins, 4 dev-held cells, 8 test tasks
    assert len(tasks) == 42
    for t in tasks:
        assert t.row == R.ROW and not t.real
        assert t.name.startswith("relational_"), \
            f"{t.name} misses the PROTECTED_PREFIXES key and the enricher would rewrite it"
        assert t.licence and t.url


def test_generation_is_seeded(tasks):
    t = tasks[0]
    assert [e.state for e in t.load(5)] == [e.state for e in t.load(5)]


def test_targets_are_well_formed(rows):
    for _t, _ex, q in rows:
        probs = q.target_probs()
        assert probs is not None and abs(sum(probs) - 1.0) < 1e-9
        assert len(set(q.options)) == len(q.options), f"duplicate options: {q.options}"
        assert q.instructions, "the comparison rule / what a reference means is the criteria"
        if q.descriptions:
            assert len(q.descriptions) == len(q.options)


def test_every_instance_is_rendered_more_than_one_way(tasks):
    """The same values, the same question, the same target, in three shapes."""
    for t in tasks[:6] + tasks[-6:]:
        seen = collections.defaultdict(set)
        for ex in t.load(30):
            q = ex.questions[0]
            seen[(q.question, tuple(q.options), q.target)].add(q.meta["shape"])
        assert any(v == set(R.SHAPES) for v in seen.values()), \
            f"{t.name} never renders one instance all three ways"


# ---- the split ------------------------------------------------------------------------

_DEV_HELD = ({f"relational_hop{h}_{c}" for c, h in R.DEV_HOP_CELLS}
             | {f"relational_{f}_{c}" for f, c in R.DEV_RANK_CELLS})


def test_dev_cells_are_new_combinations_of_trained_values(tasks):
    """Dev's held-out structures hold out a combination, never a value: each cell's chain
    and depth, and each cell's ranking shape and comparison, train elsewhere -- and no
    cell is a test structure, has a twin, or trains."""
    trained = {t.name for t in tasks if t.force_split == "train"}
    names = {t.name for t in tasks}
    for chain, hops in R.DEV_HOP_CELLS:
        assert hops in R.TRAINED_HOPS
        assert any(f"relational_hop{hops}_" in n for n in trained), hops
        assert any(n.startswith("relational_hop") and n.endswith(f"_{chain}")
                   for n in trained), chain
    for family, comparator in R.DEV_RANK_CELLS:
        assert family in R.TRAINED_RANK_FAMILIES
        assert f"relational_{family}_" in " ".join(trained) and any(
            n.endswith(f"_{comparator}") for n in trained), (family, comparator)
    for name in _DEV_HELD:
        assert name in names and name not in trained
        assert f"{name}_eval" not in names


def test_dev_cells_generate_valid_questions(tasks):
    by = {t.name: t for t in tasks}
    for name in sorted(_DEV_HELD):
        got = list(by[name].load(50))
        assert len(got) >= 45, (name, len(got))
        for ex in got:
            q = ex.questions[0]
            assert 0 <= q.target < len(q.options), name


def test_held_out_structures_never_reach_training(tasks):
    for t in tasks:
        heldout = (any(t.name.startswith(f"relational_hop{h}_") for h in R.HELDOUT_HOPS)
                   or any(f"relational_{f}_" in t.name for f in R.HELDOUT_RANK_FAMILIES))
        dev_held = t.name in _DEV_HELD
        if heldout:
            assert t.force_split == "testreal", f"{t.name} is held out but sits in {t.force_split}"
        elif dev_held:
            assert t.force_split == "devreal", f"{t.name} is dev-held but sits in {t.force_split}"
        elif t.name.endswith("_eval"):
            assert t.force_split == "devreal", \
                f"{t.name} is a trained structure's fresh draw and belongs in devreal"
        else:
            assert t.force_split == "train"


def test_the_held_out_ordering_family_uses_no_unseen_comparison(rows):
    """The `rule_transfer` pattern, and the whole reason this domain exists.

    `ranking` scored 0.267 because its *shape* -- compare a field across records -- had
    no trained parts at all. So `rank_ordering` may only be built from comparison
    primitives that `rank_pairwise` and `rank_extremum` both already train. A drop on it
    is then a failure to compose a known comparison into a full order, not an unseen
    comparison.
    """
    used = collections.defaultdict(set)
    for _t, _ex, q in rows:
        f = q.meta["family"]
        if f.startswith("rank_"):
            used[f].add(q.meta["comparator"])
    assert used["rank_ordering"], "no held-out ordering questions were generated"
    unseen = used["rank_ordering"] - (used["rank_pairwise"] & used["rank_extremum"])
    assert not unseen, f"rank_ordering uses comparisons nothing trains: {unseen}"
    assert used["rank_ordering"] == set(R.COMPARATORS)


def test_hop_count_is_in_meta_for_a_per_hop_reading(rows):
    hops = collections.Counter(q.meta["hops"] for _t, _ex, q in rows
                               if q.meta["family"] in R.HOP_FAMILIES)
    assert set(hops) == set(R.TRAINED_HOPS) | set(R.HELDOUT_HOPS)
    assert min(hops.values()) > 50


# ---- the labeller ----------------------------------------------------------------------

def test_labels_are_rederivable_from_the_state(rows):
    """Follow every chain again, from the rendered JSON, with no help from the generator.

    This is the test a wrong labeller cannot survive: it parses the state the model will
    actually see, walks the reference fields named in the instructions, and recomputes
    the answer.
    """
    chains = {c.name: c for c in R.CHAINS}
    checked = 0
    for _t, ex, q in rows:
        if q.meta["family"] not in ("lookup", "bridge", "sameend") or q.meta["shape"] != "json":
            continue
        chain = chains[q.meta["chain"]]
        groups = json.loads(_body(ex))
        by_id = {r["record_id"]: (lv, r)
                 for lv, g in enumerate(chain.groups) for r in groups[g]}

        def walk(start, hops):
            lv, rec = by_id[start]
            for k in range(hops):
                lv, rec = by_id[rec[chain.steps[lv].field]]
            return rec

        end = walk(q.meta["root"], q.meta["hops"])
        if q.meta["family"] == "lookup":
            want = end[q.meta["mark"]]
            if q.meta.get("sentinel") == "gold":
                assert want not in q.options, "the sentinel is gold but the answer is listed"
                assert sentinel.is_sentinel_description(
                    q.descriptions[q.target]), "the gold option is not a sentinel"
            else:
                assert _gold(q) == want, f"lookup mislabelled: {_gold(q)} != {want}"
        elif q.meta["family"] == "bridge":
            assert _gold(q) == end["record_id"]
        else:
            other = walk(q.meta["root_b"], q.meta["hops"])
            same = end[q.meta["mark"]] == other[q.meta["mark"]]
            assert _gold(q) == ("yes" if same else "no")
        checked += 1
    assert checked > 200, f"only {checked} chains re-derived"


def test_ranking_labels_are_rederivable(rows):
    """The ordering the rule gives is recomputed for `rank_ordering` from the option set
    itself: the gold ordering must be consistent with every pairwise comparison the
    trained families make, which is what "composed of trained rungs" has to mean."""
    for _t, _ex, q in rows:
        if q.meta["family"] != "rank_ordering":
            continue
        perms = [o.split(", ") for o in q.options]
        assert len({frozenset(p) for p in perms}) == 1
        assert all(len(set(p)) == len(p) for p in perms)


# ---- the one-hop shortcut ---------------------------------------------------------------

@pytest.fixture(scope="module")
def hop_questions():
    """json-rendered hop questions, 200 instances per chain and hop count, one row each."""
    out = []
    for chain in R.CHAINS:
        for hops in R.TRAINED_HOPS + R.HELDOUT_HOPS:
            for ex in R._hop_loader(chain.name, hops, 1, cap=10 ** 6)(600):
                if ex.questions[0].meta["shape"] == "json":
                    out.append((ex, ex.questions[0]))
    return out


def test_the_chain_is_load_bearing(hop_questions):
    """The single most important number in this domain.

    A reader who answers from the first record alone, and a reader who follows exactly
    one reference and stops, score **chance** on `lookup` and `sameend` at two hops or
    more: their answer is offered, and it is right exactly as often as any other option.
    At one hop the one-step reader is of course right; that task is the control.

    This used to demand **zero**, and zero was the bug: a binary reader that scores zero
    is a reader that scores one once inverted, and `sameend`'s one-hop reader inverted
    scored 1.000 at 2, 3 and the held-out 4 hops; on `lookup`, "strike out the root's and
    the first hop's marks" scored 0.87-0.89 against 0.26 chance
    (a shortcut audit). `bridge`'s one-hop record still scores zero:
    it sits on the wrong level, which counting the steps already rules out (see
    `test_counting_steps_without_following_them_is_the_only_residual`).
    """
    per = collections.defaultdict(lambda: [0.0, 0.0, 0.0, 0])
    for _ex, q in hop_questions:
        f, h, k = q.meta["family"], q.meta["hops"], len(q.options)
        if f == "lookup" and q.meta["sentinel"] == "gold":
            continue       # the answer is unlisted: every listed mark is wrong alike
        if f == "lookup":
            root_ans = q.meta["stop_early"][0]
            one_ans = q.meta["stop_early"][1] if h >= 2 else None
        elif f == "bridge":
            root_ans, one_ans = q.meta["root"], q.meta.get("one_hop_record")
        cell = per[(f, h)]
        cell[2] += 1.0 / k
        if f == "sameend":
            cell[0] += 1.0 / k
            cell[1] += 1.0 if (_gold(q) == "yes") == q.meta["one_hop_same"] else 0.0
        else:
            cell[0] += (1.0 if root_ans == _gold(q) else 0.0) if root_ans in q.options else 1.0 / k
            if one_ans is None:
                cell[1] += 1.0
            else:
                cell[1] += (1.0 if one_ans == _gold(q) else 0.0) if one_ans in q.options else 1.0 / k
        cell[3] += 1
    assert per
    for (f, h), (root, one, chance, n) in sorted(per.items()):
        chance /= n
        assert n >= 50, (f, h, n)
        if f != "sameend":
            assert root / n <= chance + 0.08, \
                f"{f} at {h} hops: answering from the first record alone scores {root / n:.3f}"
        if h >= 2 and f == "bridge":
            assert one / n == 0.0, f"bridge at {h} hops: the one-hop record scores {one / n:.3f}"
        elif h >= 2:
            assert abs(one / n - chance) <= 0.10, \
                f"{f} at {h} hops: following one reference and stopping scores " \
                f"{one / n:.3f} against {chance:.3f} chance"
        if f == "lookup":
            assert abs(root / n - chance) <= 0.08, \
                f"lookup at {h} hops: the root's mark scores {root / n:.3f} against {chance:.3f}"


def test_the_distractors_make_the_wrong_hop_plausible(hop_questions):
    """Several suppliers, several certification bodies. If the option a stop-early reader
    would choose is not on the list, the chain is decorative and a one-hop guess wins.
    Every `lookup` at two hops or more offers a stop-early mark (or has it as the
    unlisted answer), and most offer both the root's and the first hop's; only a field
    with room for a single distractor beside the answer and the sentinel offers one."""
    n = miss = both = 0
    for _ex, q in hop_questions:
        if q.meta["family"] != "lookup" or q.meta["hops"] < 2:
            continue
        n += 1
        se = q.meta["stop_early"][:2]
        here = [v in q.options for v in se]
        miss += not any(here) and q.meta["sentinel"] != "gold"
        both += all(here)
    assert n > 200 and miss == 0, f"{miss}/{n} lookups offer no stop-early distractor"
    assert both / n >= 0.5, f"only {both}/{n} lookups offer both stop-early marks"


def test_every_lookup_option_lives_at_the_level_the_chain_ends_on(rows):
    """Otherwise "which group does the answer live in" cuts the field down on its own,
    and a four-hop question hands away two of its three options to anybody who can count
    the steps without following one."""
    checked = 0
    chains = {c.name: c for c in R.CHAINS}
    for _t, ex, q in rows:
        if q.meta["family"] != "lookup" or q.meta["shape"] != "json":
            continue
        chain = chains[q.meta["chain"]]
        groups = json.loads(_body(ex))
        final = {r[q.meta["mark"]] for r in groups[chain.groups[q.meta["hops"]]]}
        real = [o for o in q.options if not sentinel.is_sentinel_key(o)]
        assert set(real) <= final, f"{set(real) - final} cannot be reached at the end"
        checked += 1
    assert checked > 100


# ---- string match ------------------------------------------------------------------------

def test_answer_is_not_recoverable_by_string_match(rows):
    """The gold option may not be pickable out of the question or the criteria. Row 36
    named the control verbatim in the goal and both its navigation families were 100 %
    solvable this way."""
    hits = 0
    for _t, _ex, q in rows:
        blob = f"{q.question} {q.instructions}"
        if all(_appears(o, blob) for o in q.options):
            continue                      # `rank_pairwise` names both candidates
        hits += _appears(_gold(q), blob)
    assert hits == 0, f"{hits}/{len(rows)} answers are readable off the question"


def test_gold_is_never_the_only_option_present_in_the_state(rows):
    hits = 0
    for _t, ex, q in rows:
        present = [o for o in q.options if _appears(o, ex.state)]
        hits += present == [_gold(q)]
    assert hits == 0, f"{hits}/{len(rows)} answers are the only option in their own state"


# ---- baselines ---------------------------------------------------------------------------

def test_no_family_is_won_by_a_constant_answer(rows):
    by = collections.defaultdict(list)
    for _t, _ex, q in rows:
        by[q.meta["family"]].append(q)
    for f, qs in by.items():
        counts = collections.Counter(_gold(q) for q in qs)
        share = counts.most_common(1)[0][1] / len(qs)
        # For a binary family the bound has to come from the sample, not from a round
        # number: at 63 questions a fair coin lands 0.60 one time in twenty, and a flat
        # 0.60 made this test flake on wording changes that touch nothing but the rng
        # stream. Three sigma here, and `test_the_binary_families_are_balanced` puts the
        # tight bound on the whole stream where the sample can carry it.
        binary = bool(qs) and max(len(q.options) for q in qs) == 2
        floor = 0.5 + 3 * (0.25 / len(qs)) ** 0.5 if binary else 0.20
        assert share <= floor, f"{f}: a constant answer scores {share:.3f}"


def test_the_binary_families_are_balanced(tasks):
    """The 3-sigma bound above is loose on a 63-question sample, so the real balance is
    read off the whole stream, where `sameend`'s yes/no balancer can be held to 0.55."""
    counts = collections.Counter()
    for t in tasks:
        if "_hop" not in t.name:
            continue
        # 480, not 240: the devreal twins now stop at MAX_PER_TASK_DEV
        for ex in t.load(480):
            q = ex.questions[0]
            if len(q.options) == 2:
                counts[_gold(q)] += 1
    n = sum(counts.values())
    assert n > 900, f"only {n} binary questions drawn"
    share = counts.most_common(1)[0][1] / n
    assert share <= 0.55, f"the binary families answer {share:.3f} one way"


def test_the_gold_option_is_not_parked_at_one_index(rows):
    """A generator that emits the answer at index 0 teaches index 0. Shuffle and prove it."""
    by = collections.defaultdict(lambda: [0.0, 0.0, 0])
    for _t, _ex, q in rows:
        cell = by[q.meta["family"]]
        cell[0] += q.target == 0
        cell[1] += 1.0 / len(q.options)          # what an unbiased shuffle would give
        cell[2] += 1
    for f, (at0, expect, n) in by.items():
        se = (expect / n * (1 - expect / n) / n) ** 0.5
        assert abs(at0 - expect) / n < 4 * se + 0.01, \
            f"{f}: gold at index 0 in {at0 / n:.3f} against {expect / n:.3f} expected"


# ---- ranking: the state-only check ----------------------------------------------------------

def test_no_ranking_option_can_be_eliminated_without_comparing_records(rows):
    """`rank_ordering`'s options are permutations of one identical set, so no single
    record rules any of them out; `rank_pairwise` and `rank_extremum` name records that
    are all in the state and all carry the compared field."""
    n = 0
    for _t, ex, q in rows:
        f = q.meta["family"]
        if not f.startswith("rank_"):
            continue
        n += 1
        if f == "rank_ordering":
            assert len({frozenset(o.split(", ")) for o in q.options}) == 1
        else:
            for o in q.options:
                assert _appears(o, ex.state), f"{o} is an option that is not a record"
    assert n > 500


def test_reading_the_state_in_order_is_no_better_than_guessing(rows):
    """A reader who copies the order the state happens to list them in must do no better
    than chance. It cannot do *worse* than chance by construction either: that order is
    put on the list as a distractor whenever it differs from the true one, so the option
    exists and is wrong.

    Four of the five comparisons also have to differ from reading the biggest number off
    the page -- `field` is the base case and is meant to agree, which is what makes it
    the rung the other four are read against. `make_rank_instance` drops any instance
    whose naive order is already the true order.
    """
    hit = expect = n = 0.0
    for _t, ex, q in rows:
        if q.meta["family"] != "rank_ordering":
            continue
        body = _body(ex)
        names = set()
        for o in q.options:
            names |= set(o.split(", "))
        listed = ", ".join(sorted(names, key=body.index))
        hit += listed == _gold(q)
        expect += 1.0 / len(q.options)
        n += 1
    assert n > 200
    assert hit / n <= expect / n, \
        f"copying the state's own order scores {hit / n:.3f} against {expect / n:.3f}"


# ---- shapes, sentinel, budget -----------------------------------------------------------------

def test_option_counts_spread_from_binary_to_high_cardinality(rows):
    ks = [len(q.options) for _t, _ex, q in rows]
    assert max(ks) >= 20, f"max option count {max(ks)} is too low for a corpus that is 63 % binary"
    assert sum(1 for k in ks if k > 8) / len(ks) > 0.10
    assert sum(1 for k in ks if k == 2) / len(ks) < 0.45


def test_states_clear_the_forty_token_floor(rows):
    """Counted in characters here, at 6 characters per token, which is pessimistic for
    these states; `tests` must not pull a tokenizer. The real count is in the findings."""
    worst = min(len(ex.state) for _t, ex, _q in rows)
    assert worst // 6 >= 40, f"shortest state is {worst} characters"


def test_states_fit_the_training_budget(rows):
    """`ship_v4.sh` trains at `--max-state-tokens 2304` and the packer truncates from the
    tail, so a state over budget loses the last level of the chain -- which is exactly
    where the answer is."""
    assert R.MAX_STATE_CHARS / 2.4 <= 2304          # the worst ratio measured
    over = sum(1 for _t, ex, _q in rows if len(ex.state) > R.MAX_STATE_CHARS)
    assert over == 0, f"{over} states are over the character budget"


def test_the_sentinel_is_drawn_from_the_grammar_not_written_out(rows):
    """A fixed wording repeated 1,402 times in row 39 failed `sentinel_audit.py`'s
    max-exact-repeat bar of 50."""
    seen = []
    for _t, _ex, q in rows:
        for d in q.descriptions or []:
            if d and sentinel.is_sentinel_description(d):
                seen.append(d)
    assert seen, "no sentinel options were generated"
    counts = collections.Counter(seen)
    assert counts.most_common(1)[0][1] <= 50, \
        f"wording repeated {counts.most_common(1)[0][1]} times"
    assert len(counts) > 30


def test_held_out_tasks_draw_unseen_sentinel_wordings(tasks):
    """`sentinel.split_of` keeps a hashed 20 % of the wording space out of training, so
    the `sentinel_wordings` gate can read a wording the model has never seen on the
    held-out families too."""
    for t in tasks:
        if t.force_split != "testreal" or "_hop" not in t.name:
            continue
        for ex in t.load(60):
            for d in ex.questions[0].descriptions or []:
                if d and sentinel.is_sentinel_description(d):
                    assert sentinel.split_of(d) == "eval", f"{t.name} drew a trained wording"


def test_the_row_is_sized_for_the_corpus(tasks):
    cap = {"train": R.MAX_PER_TASK_TRAIN, "devreal": R.MAX_PER_TASK_DEV,
           "testreal": R.MAX_PER_TASK}

    def cap_of(t):
        return R.MAX_PER_TASK if t.name in _DEV_HELD else cap[t.force_split]
    total = sum(cap_of(t) for t in tasks)
    assert 20_000 <= total <= 25_000
    dev = sum(cap_of(t) for t in tasks if t.force_split == "devreal")
    assert 0.08 <= dev / total <= 0.20, f"devreal is {dev / total:.3f} of the row"


def test_the_probe_stream_is_never_in_a_split(tasks):
    """`probe_tasks()` must not overlap the corpus, or a trained-family number read off
    it is recall of training examples -- the exact failure an earlier `rule_transfer`
    probe shipped with."""
    corpus = {ex.state for t in tasks for ex in t.load(25)}
    probe = R.probe_tasks()
    assert len(probe) == 12 + 15
    overlap = sum(1 for t in probe for ex in t.load(25) if ex.state in corpus)
    assert overlap == 0, f"{overlap} probe states are verbatim in the corpus stream"


# ---- wording: the model wrote stems and the stems have to read like English -----------

_A_BEFORE_VOWEL = re.compile(r"(?<![\w-])a (?=[aeiouAEIOU])")


def test_every_indefinite_article_agrees_with_its_noun(rows):
    """An earlier build shipped "share a escalation tier" 126 times, out of the
    *hand-written* `sameend` stem that replaced the model's rejected batch -- the article
    was written into the stem and four of the six mark names begin with a consonant.
    Every surface this row emits is checked, not just the stems.
    """
    bad: list[str] = []
    for _t, _ex, q in rows:
        for text in [q.question, q.instructions or ""] + [d for d in (q.descriptions or []) if d]:
            if _A_BEFORE_VOWEL.search(text):
                bad.append(text)
    assert not bad, f"{len(bad)} wordings put 'a' before a vowel, e.g. {bad[0]!r}"


def test_the_question_and_the_criteria_agree_on_the_hop_count(rows):
    """A wording variant may not change what is being asked. The stems carry the hop
    count in their own text and the criteria carry it as a numbered step list; if the two
    ever disagree the question is unanswerable and the target is still derived from the
    steps.
    """
    n = 0
    for _t, _ex, q in rows:
        if q.meta["family"] not in R.HOP_FAMILIES:
            continue
        steps = len(re.findall(r"^  step (\d+): ", q.instructions, re.M))
        assert steps == q.meta["hops"], f"{steps} steps written for {q.meta['hops']} hops"
        m = re.search(r"\ball (\d+) steps\b|\bchain of (\d+) steps\b"
                      r"|\bfull set of (\d+) steps\b|\b(\d+) steps\b", q.question)
        if m:
            said = int(next(g for g in m.groups() if g))
        else:
            assert re.search(r"\bthe single step\b|\bone step\b", q.question), q.question
            said = 1
        assert said == steps, f"question says {said} steps, criteria say {steps}"
        n += 1
    assert n > 400


# ---- the two shortcuts that do not need the chain or the rule ------------------------

def test_the_true_ordering_is_not_the_consensus_of_the_option_set(rows):
    """`rank_ordering` built only from adjacent swaps of the true order puts the answer
    at the centre of its own field: every distractor is one transposition from it and two
    or more from each other, so picking the option closest to all the others answers the
    question without reading a record or the rule. On an earlier corpus build that scored
    0.416-0.490 against 0.279-0.302 chance. `rank_instance_examples` now seeds the field
    from a second pivot and drops an instance the repair cannot fix.
    """
    hit = expect = n = 0.0
    for _t, _ex, q in rows:
        if q.meta["family"] != "rank_ordering":
            continue
        perms = [tuple(o.split(", ")) for o in q.options]
        total = [sum(R._kendall(p, o) for o in perms) for p in perms]
        best = [i for i, x in enumerate(total) if x == min(total)]
        hit += (1.0 / len(best)) if q.target in best else 0.0
        expect += 1.0 / len(q.options)
        n += 1
    assert n > 100
    assert hit / n <= expect / n + 0.02, \
        f"the consensus of the options scores {hit / n:.3f} against {expect / n:.3f}"


def test_the_sentinel_option_is_no_likelier_than_any_other_option(tasks):
    """The sentinel is the one option a reader can find without following anything: it is
    the only option absent from the state and the only one carrying a description. An
    earlier build made it gold 45 % of the time it was offered, so that reader scored
    0.442-0.484 against 0.250-0.280 chance. It must now be gold about as often as any
    other option is, and the row must still supply as many target-absent questions as it
    did.
    """
    offered = gold = total = 0
    share = 0.0
    for t in tasks:
        if "_hop" not in t.name:
            continue
        for ex in t.load(240):
            q = ex.questions[0]
            if q.meta["family"] != "lookup":
                continue
            total += 1
            if q.meta["sentinel"] == "absent":
                continue
            offered += 1
            share += 1.0 / len(q.options)
            gold += q.meta["sentinel"] == "gold"
    assert offered > 400, f"only {offered} sentinel questions drawn"
    rate = gold / offered
    assert rate <= share / offered + 0.06, \
        f"the sentinel is gold {rate:.3f} of the time against {share / offered:.3f} chance"
    # and the abstention supply this row carries has not been traded away for it
    assert gold / total >= 0.10, f"only {gold / total:.3f} of lookup questions abstain"


def test_counting_steps_without_following_them_is_the_only_residual(rows):
    """The honest residual, measured rather than asserted away.

    `bridge`'s options are record ids, and the answer always sits at the level the step
    list ends on, so a reader who counts the steps and then guesses among the options
    from that level alone beats chance without following one reference. It is bounded
    here, and `lookup` -- whose options are all carried by records at the end level --
    must show no such gap at all.
    """
    got: dict[tuple[str, int], list[float]] = collections.defaultdict(lambda: [0.0, 0.0, 0])
    for _t, ex, q in rows:
        fam = q.meta["family"]
        if fam not in ("bridge", "lookup") or q.meta["shape"] != "json":
            continue
        levels = list(json.loads(_body(ex)).values())
        hops = q.meta["hops"]
        if fam == "bridge":
            end = {r["record_id"] for r in levels[hops]}
        else:
            end = {r[q.meta["mark"]] for r in levels[hops]}
        kept = [o for o in q.options if o in end]
        cell = got[(fam, hops)]
        cell[0] += (1.0 / len(kept)) if (kept and _gold(q) in kept) else 0.0
        cell[1] += 1.0 / len(q.options)
        cell[2] += 1
    assert got, "no json-shaped hop questions were drawn"
    for (fam, hops), (hit, expect, n) in sorted(got.items()):
        if n < 20:
            continue
        if fam == "lookup":
            assert hit / n <= expect / n + 0.05, \
                f"lookup h{hops}: counting steps scores {hit / n:.3f} vs {expect / n:.3f}"
        else:
            assert hit / n <= 0.50, \
                f"bridge h{hops}: counting steps scores {hit / n:.3f} vs {expect / n:.3f}"


# ---- regressions for what an independent audit found ------------------------------------

def test_every_trained_structure_is_measured_in_devreal_and_none_in_testreal(tasks):
    """Before: the `_eval` twins were hashed, 15 of 19 landed in train and 4 in testreal,
    and devreal held nothing from this row -- `rank_pairwise` was never evaluated at all,
    and four trained structures sat beside the held-out ones in testreal."""
    dev = collections.defaultdict(set)
    for t in tasks:
        if t.force_split == "devreal" and t.name not in _DEV_HELD:
            m = re.match(r"relational_(hop\d|rank_pairwise|rank_extremum)_(\w+)_eval$", t.name)
            assert m, t.name
            dev[m.group(1)].add(m.group(2))
        if t.force_split == "testreal":
            assert (re.match(r"relational_hop4_", t.name)
                    or t.name.startswith("relational_rank_ordering_")), t.name
    assert set(dev) == {f"hop{h}" for h in R.TRAINED_HOPS} | set(R.TRAINED_RANK_FAMILIES)
    # every trained cell has its twin; the dev-held cells train nowhere and have none
    for fam in R.TRAINED_RANK_FAMILIES:
        assert dev[fam] == set(R.COMPARATORS) - {c for f, c in R.DEV_RANK_CELLS if f == fam}
    for h in R.TRAINED_HOPS:
        assert dev[f"hop{h}"] == ({c.name for c in R.CHAINS}
                                  - {c for c, hh in R.DEV_HOP_CELLS if hh == h})


def _option_counts(ex, q):
    return {o: len(re.findall(r"(?<![\w-])" + re.escape(o) + r"(?![\w-])", ex.state))
            for o in q.options}


def test_how_often_an_option_is_printed_does_not_give_the_lookup_answer(tasks):
    """Before: "pick the option printed least often" scored 0.481-0.500 on lookup at 2-4
    hops against 0.261 chance, because the stop-early values were planted twice. And once
    counts were balanced, "drop options printed before the end level" scored 0.34-0.45."""
    rare = early = chance = n = 0.0
    chains = {c.name: c for c in R.CHAINS}
    for t in tasks:
        if "_hop" not in t.name:
            continue
        for ex in t.load(150):
            q = ex.questions[0]
            if q.meta["family"] != "lookup" or q.meta["hops"] < 2 or q.meta["shape"] != "json":
                continue
            gold = _gold(q)
            cnt = _option_counts(ex, q)
            present = [o for o in q.options if cnt[o]]
            lo = min(cnt[o] for o in present)
            best = [o for o in present if cnt[o] == lo]
            rare += (1.0 / len(best)) if gold in best else 0.0
            groups = json.loads(_body(ex))
            chain = chains[q.meta["chain"]]
            seen = {r[q.meta["mark"]] for g in chain.groups[:q.meta["hops"]] for r in groups[g]}
            rest = [o for o in q.options if o not in seen] or list(q.options)
            early += (1.0 / len(rest)) if gold in rest else 0.0
            chance += 1.0 / len(q.options)
            n += 1
    assert n > 200
    assert rare / n <= chance / n + 0.05, f"rarest option scores {rare / n:.3f} vs {chance / n:.3f}"
    assert early / n <= chance / n + 0.05, \
        f"dropping early-level options scores {early / n:.3f} vs {chance / n:.3f}"


def _surface_pick_pairwise(ex, q):
    """Sort the two named records by the number the rule names first, ignoring the rule."""
    recs = {r["name"]: r for r in json.loads(_body(ex))}
    first = q.instructions.split("\n")[0]
    field = re.findall(r"`(\w+)`", first)[0]
    high = "lowest" not in first.split("--")[-1] and "lowest first" not in first
    a, b = q.options
    va, vb = float(recs[a][field]), float(recs[b][field])
    if va == vb:
        return None
    return (a if va > vb else b) if high else (a if va < vb else b)


def test_the_trained_ranking_rungs_cannot_be_read_off_the_printed_number(tasks):
    """Before: on the four comparisons whose key is not the printed number, the pair the
    rule orders was ordered the same way by the printed number 0.66-0.88 of the time, so
    the pairwise rung taught "compare the number you see" -- the very distractor
    `rank_ordering` plants. Now half the pairs are ones the printed number gets wrong."""
    per = collections.defaultdict(lambda: [0.0, 0])
    for t in tasks:
        if not re.match(r"relational_rank_pairwise_(ratio|mixed_units)$", t.name):
            continue
        for ex in t.load(600):
            q = ex.questions[0]
            if q.meta["shape"] != "json":
                continue
            pick = _surface_pick_pairwise(ex, q)
            cell = per[q.meta["comparator"]]
            cell[0] += 0.5 if pick is None else float(pick == _gold(q))
            cell[1] += 1
    for comp in ("ratio", "mixed_units"):
        hit, n = per[comp]
        assert n >= 150
        assert hit / n <= 0.60, f"pairwise {comp}: printed number alone scores {hit / n:.3f}"


def test_ratio_rankings_are_not_decided_by_a_near_tie():
    """Before: 6.9 % of `rank_ordering:ratio` questions hinged on two quotients within 1 %."""
    for d in R.RANK_DOMAINS:
        import random
        rng = random.Random(7)
        made = 0
        for _ in range(300):
            inst = R.make_rank_instance(d, "ratio", rng.choice((3, 5, 8, 12, 16)), rng)
            if inst is None:
                continue
            ks = sorted(abs(k) for k in inst.keys)
            assert all((b - a) / b >= R.RATIO_MIN_GAP for a, b in zip(ks, ks[1:]))
            made += 1
        assert made > 50, f"{d.name}: the gap rule starves the ratio comparator ({made})"


def _rank_keys(first_line, recs):
    """An independent rule applier for every comparator, from the criteria text alone."""
    from fractions import Fraction as F
    m = re.match(r"Rank the \w+ by `(\w+)`, (highest|lowest) first\. Where two of them have "
                 r"the same `\w+`, the one with the (higher|lower) `(\w+)` ranks ahead\.$",
                 first_line)
    if m:
        p, pd, sd, s = m.groups()
        return [((1 if pd == "highest" else -1) * F(str(r[p])),
                 (1 if sd == "higher" else -1) * F(str(r[s]))) for r in recs]
    m = re.match(r"Rank the \w+ by `(\w+)`, (highest|lowest) first\.$", first_line)
    if m:
        f, d = m.groups()
        return [((1 if d == "highest" else -1) * F(str(r[f])),) for r in recs]
    m = re.search(r"`(\w+)` divided by `(\w+)` -- (highest|lowest) first", first_line)
    if m:
        a, b, d = m.groups()
        return [((1 if d == "highest" else -1) * F(str(r[a])) / F(str(r[b])),) for r in recs]
    m = re.search(r"reports `(\w+)` .* given in `(\w+)` .* to (\w+) before you compare "
                  r"anything \((.*)\), then rank (highest|lowest) first", first_line)
    if m:
        f, uf, base, table, d = m.groups()
        per = {base: F(1)}
        for part in table.split("; "):
            _one, _b, _eq, x, u = part.split(" ")
            per[u] = F(x)
        return [((1 if d == "highest" else -1) * F(str(r[f])) / per[r[uf]],) for r in recs]
    m = re.search(r"how close `(\w+)` sits to the target of (\d+)", first_line)
    assert m, first_line
    f, t = m.groups()
    return [(-abs(F(str(r[f])) - F(t)),) for r in recs]


def test_ranking_labels_rederive_from_the_state_and_the_rule_text(rows):
    """Every ranking label, all three families and all five comparisons, recomputed from
    the rendered JSON and the criteria sentence in exact arithmetic -- no generator code.
    Exact ties on the key would make a question ambiguous and must not occur."""
    checked = 0
    for _t, ex, q in rows:
        f = q.meta["family"]
        if not f.startswith("rank_") or q.meta["shape"] != "json":
            continue
        recs = json.loads(_body(ex))
        names = [r["name"] for r in recs]
        keys = _rank_keys(q.instructions.split("\n")[0], recs)
        assert len(set(keys)) == len(keys), f"tie on the ranking key: {keys}"
        order = sorted(range(len(recs)), key=lambda i: keys[i], reverse=True)
        if f == "rank_pairwise":
            a, b = (names.index(o) for o in q.options)
            want = q.options[0] if keys[a] > keys[b] else q.options[1]
        elif f == "rank_extremum":
            m = re.search(r"(second from last|last|second|first) in the ranking", q.question)
            pos = {"first": 0, "second": 1, "last": -1, "second from last": -2}[m.group(1)]
            want = names[order[pos]]
        else:
            want = ", ".join(names[i] for i in order)
        assert _gold(q) == want, f"{f}/{q.meta['comparator']}: {_gold(q)} != {want}"
        checked += 1
    assert checked > 150


class _FakeBank:
    def __init__(self, phr):
        self.phr = phr

    def pick(self, tid, template, key):
        from lod.phrasings import placeholders
        c = self.phr.get(tid)
        return c if c and placeholders(c) == placeholders(template) else template


def test_a_bank_phrasing_that_changes_the_hop_count_is_refused(monkeypatch):
    """The bank may vary wording, never the quantity asked. A phrasing that loses the
    "{h} steps" count (here "{h} hops", which the singular repair cannot fix and the
    reader cannot count) falls back to the literal stem; a sound one is used."""
    stem = R.LOOKUP_STEMS[0]
    kw = dict(noun="shipment", rid="kestrel-4471", mark="risk band")
    monkeypatch.setattr(R, "_BANK", _FakeBank({"d20.lookup":
                        "Go {h} hops from {noun} {rid}; which {mark} is there?"}))
    assert R.hop_text(stem, 1, "lookup", **kw) == R.hop_text(stem, 1, None, **kw)
    monkeypatch.setattr(R, "_BANK", _FakeBank({"d20.lookup":
                        "From {noun} {rid}, after all {h} steps, what {mark} do you find?"}))
    got = R.hop_text(stem, 3, "lookup", **kw)
    assert got == "From shipment kestrel-4471, after all 3 steps, what risk band do you find?"
    assert R.said_hops(R.hop_text(stem, 1, "lookup", **kw)) == 1


def test_the_phrasing_spec_covers_every_question_template():
    from lod.paths import ASSETS
    spec = json.loads((ASSETS / "phrasing_specs" / "d20.json").read_text())
    from lod.phrasings import placeholders
    fams = {"lookup": R.LOOKUP_STEMS, "bridge": R.BRIDGE_STEMS, "sameend": R.SAMEEND_STEMS,
            "rank_pairwise": R.PAIRWISE_STEMS, "rank_extremum": R.EXTREMUM_STEMS,
            "rank_ordering": R.ORDERING_STEMS}
    assert set(spec) == {f"d20.{f}" for f in fams}
    for f, stems in fams.items():
        e = spec[f"d20.{f}"]
        assert e["template"] in stems
        assert e["state_kind"] and e["answer_meaning"]
        # the bank is sampled with the drawn stem as template; phrasings built from the
        # spec template only apply to stems with the same slots
        assert sum(placeholders(s) == placeholders(e["template"]) for s in stems) >= len(stems) - 1
