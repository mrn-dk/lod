"""Row 36: accessibility trees and navigator decisions."""

from __future__ import annotations

import json

import pytest

from lod.corpus.services.sources.synth import browser


@pytest.fixture(scope="module")
def tasks():
    return browser.tasks()


def test_every_task_yields_what_it_is_asked_for(tasks):
    for t in tasks:
        got = list(t.load(20))
        assert len(got) == 20, f"{t.name} produced {len(got)}"


def test_states_are_valid_json_trees(tasks):
    for t in tasks:
        for ex in list(t.load(5)):
            tree = json.loads(ex.state)
            assert "role" in tree and "children" in tree


def test_targets_are_well_formed(tasks):
    for t in tasks:
        for ex in list(t.load(10)):
            for q in ex.questions:
                probs = q.target_probs()
                assert probs is not None and abs(sum(probs) - 1.0) < 1e-9
                assert len(probs) == len(q.options)
                assert len(set(q.options)) == len(q.options), f"{t.name}: duplicate options"


def test_held_out_layouts_never_reach_training(tasks):
    for t in tasks:
        held = any(f"_{layout}" == t.name[-len(layout) - 1:] for layout in browser.HELDOUT_LAYOUTS)
        dev = any(f"_{layout}" == t.name[-len(layout) - 1:] for layout in browser.DEV_LAYOUTS)
        if held:
            assert t.force_split == "testreal", f"{t.name} is a held-out layout in {t.force_split}"
        elif dev:
            assert t.force_split == "devreal", f"{t.name} is a dev layout in {t.force_split}"
        else:
            assert t.force_split == "train"


def test_dev_layouts_are_neither_trained_nor_tested(tasks):
    """Dev selects checkpoints and fits T, so it holds out layouts of its own: an unseen
    arrangement, as test's are, and never one of test's. Click and goal on one layout are
    one held-out unit and share a family."""
    assert set(browser.DEV_LAYOUTS).isdisjoint(browser.TRAINED_LAYOUTS + browser.HELDOUT_LAYOUTS)
    assert len(browser.DEV_LAYOUTS) >= 2, "goal_reached draws its distractor within the group"
    fam = {}
    for t in tasks:
        fam.setdefault(t.family, set()).add(t.force_split)
    for layout in browser.LAYOUTS:
        want = ("devreal" if layout in browser.DEV_LAYOUTS else
                "testreal" if layout in browser.HELDOUT_LAYOUTS else "train")
        assert fam[f"browser_{layout}"] == {want}, (layout, fam[f"browser_{layout}"])


def test_held_out_layouts_reuse_trained_controls(tasks):
    """A held-out layout must differ in arrangement, not in vocabulary. Otherwise a drop
    on it measures an unseen widget rather than a failure to generalise over structure."""
    import random
    def roles(layout):
        out = set()
        def walk(n):
            out.add(n["role"])
            for c in n.get("children", []):
                walk(c)
        for i in range(30):
            tree, _, _ = browser.build_page(layout, random.Random(f"r:{layout}:{i}"))
            walk(tree)
        return out
    trained = set().union(*(roles(l) for l in browser.TRAINED_LAYOUTS))
    for layout in browser.HELDOUT_LAYOUTS + browser.DEV_LAYOUTS:
        new = roles(layout) - trained
        assert not new, f"{layout} introduces unseen roles {new}"


def test_generation_is_seeded(tasks):
    t = tasks[0]
    assert [e.state for e in t.load(5)] == [e.state for e in t.load(5)]


def test_option_counts_reach_high_cardinality(tasks):
    ks = [len(q.options) for t in tasks for e in t.load(40) for q in e.questions]
    assert max(ks) >= 48, f"max option count {max(ks)} is too low to teach cardinality"
    assert sum(1 for k in ks if k > 8) / len(ks) > 0.15


def _gold(q):
    probs = q.target_probs()
    return q.options[max(range(len(probs)), key=probs.__getitem__)]


def test_answer_is_not_recoverable_by_string_match(tasks):
    """The first version put the goal string verbatim into the option list, so both
    navigation families scored 100 % by exact match and taught nothing. The goal now
    names a sibling attribute and the controls are opaque reference numbers."""
    for family in ("click", "field"):
        hits = total = 0
        for t in tasks:
            if not t.name.startswith(f"browser_{family}_"):
                continue
            for ex in t.load(40):
                for q in ex.questions:
                    total += 1
                    hits += _gold(q) in (q.instructions or "")
        assert total and hits == 0, f"{family}: {hits}/{total} answers appear in the instructions"


def test_goal_reached_is_not_decided_by_the_page_title(tasks):
    """Title equality must predict the label no better than chance."""
    import json
    hits = total = 0
    for t in tasks:
        if not t.name.startswith("browser_goal_"):
            continue
        for ex in t.load(40):
            title = json.loads(ex.state).get("name", "")
            for q in ex.questions:
                total += 1
                hits += (f"'{title}'" in (q.instructions or "")) == (_gold(q) == "yes")
    assert total, "no goal tasks"
    assert 0.40 <= hits / total <= 0.60, f"title equality predicts {hits / total:.2%}"


def test_goal_reached_labels_are_balanced(tasks):
    ys = [_gold(q) for t in tasks if t.name.startswith("browser_goal_")
          for ex in t.load(40) for q in ex.questions]
    assert 0.40 <= ys.count("yes") / len(ys) <= 0.60


def test_every_page_fits_the_training_state_budget(tasks):
    """`ship_v4.sh` trains at `--max-state-tokens 2304` and `packing.Packer` truncates a
    long state from the tail. Before this was enforced, 13 % of `click` examples had their
    gold control cut away and still offered it as an option, which trains guessing."""
    worst = 0
    for t in tasks:
        for ex in t.load(60):
            worst = max(worst, len(ex.state))
    assert worst <= browser.MAX_STATE_CHARS, f"state of {worst} chars exceeds the budget"


def test_gold_control_survives_a_head_truncation(tasks):
    """The character budget is a proxy for tokens; this asserts the thing it is a proxy
    for. 3.0 characters per token is the lowest ratio measured on these trees, so
    MAX_STATE_CHARS // 3 is a pessimistic token count for the whole state."""
    assert browser.MAX_STATE_CHARS // 3 <= 2304
    for t in tasks:
        if "_goal_" in t.name:
            continue
        for ex in t.load(40):
            for q in ex.questions:
                probs = q.target_probs()
                gold = q.options[max(range(len(probs)), key=probs.__getitem__)]
                assert gold in ex.state, f"{t.name}: gold {gold!r} is not in the state"


def test_states_are_rendered_compactly(tasks):
    """Indented JSON cost 2.2x the tokens of the same tree compacted, and the indentation
    carried nothing a navigator reads."""
    for t in tasks:
        for ex in t.load(3):
            assert "\n" not in ex.state          # no indentation
            assert '": ' not in ex.state          # compact separators
            assert '"href"' not in ex.state       # restates the control's own name


def test_layout_content_strings_are_mutually_exclusive():
    """`goal_reached` labels a page against another layout's content string. If two
    layouts could both satisfy one string the label is wrong, not merely hard."""
    assert len(set(browser.LAYOUT_CONTENT.values())) == len(browser.LAYOUT_CONTENT)
    assert set(browser.LAYOUT_CONTENT) == set(browser.LAYOUTS)


# ------------------------------------------------ an audit: what a state-blind reader gets

def _bulk(tasks, family, n=200):
    for t in tasks:
        if t.name.startswith(f"browser_{family}_"):
            for ex in t.load(n):
                yield t, ex, ex.questions[0]


def test_the_option_list_alone_does_not_mark_the_answer():
    """Row 36's founding defect, in the costume it came back in. `field` appended the
    value's kind to the gold field's name and to nothing else, so the option list read
    `['Supplier 0', 'Password 1', 'Event 2 date']`; over all 2,000 questions of an
    earlier build, picking the one option that is not `<Noun> <int>` was correct
    2,000/2,000."""
    import re

    for t in browser.tasks():
        if not t.name.startswith("browser_field_"):
            continue
        for ex in t.load(200):
            for q in ex.questions:
                for opt in q.options:
                    assert re.fullmatch(r"[A-Za-z]+ \d+", opt), (t.name, opt)


def test_no_option_is_the_odd_one_out(tasks):
    """The general form: after masking digits, the gold must not be the only option with
    its shape. This catches a marker added to the answer in any future wording."""
    import re

    for family in ("click", "field"):
        hits = total = 0
        for t, ex, q in _bulk(tasks, family):
            shapes = [re.sub(r"\d+", "#", o) for o in q.options]
            odd = [i for i, sh in enumerate(shapes) if shapes.count(sh) == 1]
            total += 1
            hits += len(odd) == 1 and odd[0] == q.target
        assert total and hits / total <= 0.05, f"{family}: odd-one-out scores {hits/total:.2%}"


def test_every_layout_renders_every_attribute_a_goal_can_name():
    """`_paginated_table` rendered Site and Owner and not Status, so 972 of that task's
    4,000 questions in an earlier build named a value the tree never showed. The
    generator cannot ask about an attribute the layout does not put on the page."""
    import random

    for layout in browser.LAYOUTS:
        if layout == "settings_form":            # carries no records, and no click task
            continue
        for i in range(20):
            tree, entries, _ = browser.build_page(layout, random.Random(f"a:{layout}:{i}"))
            state = browser._state(tree)
            for e in entries:
                for attr in browser.ATTRS:
                    assert e["attrs"][attr] in state, (layout, attr, e["attrs"][attr])


def test_the_named_value_is_in_the_state_and_names_one_record(tasks):
    """Per example, the check the layout sweep makes structurally."""
    import re

    for t, ex, q in _bulk(tasks, "click", n=150):
        attr, value = re.match(
            r"Open the record whose (\w+) is '(.*)'\.", q.instructions).groups()
        assert value in ex.state, (t.name, value)


def _satisfied(tree):
    """Which `LAYOUT_CONTENT` strings a tree actually shows, decided independently of the
    generator -- roles and names only, the way a reader of the JSON would."""
    import re

    nodes = []
    def walk(n):
        nodes.append(n)
        for c in n.get("children", []):
            walk(c)
    walk(tree)
    roles = {}
    for n in nodes:
        roles[n["role"]] = roles.get(n["role"], 0) + 1
    names = {n.get("name", "") for n in nodes}
    out = set()
    if any(n["role"] == "list" and any(c["role"] == "listitem"
                                       for c in n.get("children", [])) for n in nodes):
        out.add("a list of search results")
    if (roles.get("heading", 0) >= 2 and roles.get("group", 0) >= 1
            and not roles.get("table") and not roles.get("list")
            and not roles.get("menu") and not roles.get("tablist")
            and not roles.get("dialog")):
        out.add("an article with section headings")
    if roles.get("form") and roles.get("textbox"):
        out.add("a settings form with editable fields")
    if roles.get("menu") and roles.get("menuitem"):
        out.add("a navigation menu of grouped links")
    if roles.get("columnheader"):
        out.add("a table of records with column headers")
    if "Back" in names and "Next" in names:
        out.add("a multi-step wizard with Back and Next buttons")
    if any(n["role"] == "dialog" and n.get("modal") for n in nodes):
        out.add("a modal dialog over a page that is inert behind it")
    if sum(1 for n in nodes if n["role"] == "region"
           and re.fullmatch(r"Page \d+", n.get("name", "") or "")) >= 2:
        out.add("a log table split across numbered pages")
    tabs = {n.get("name") for t in nodes if t["role"] == "tablist"
            for n in t.get("children", []) if n["role"] == "tab"}
    if sum(1 for n in nodes if n["role"] == "region" and n.get("name") in tabs) >= 2:
        out.add("a row of tabs, each with its own panel of records")
    if sum(1 for n in nodes if n["role"] == "button" and "expanded" in n) >= 2:
        out.add("an accordion of sections that expand and collapse")
    return out


def test_layout_content_is_decidable_from_the_tree_not_just_distinct_as_a_string():
    """Distinct strings were asserted; mutually exclusive *pages* were not. `_modal_dialog`
    built listitems byte-identical to `_search_results`', so a modal page satisfied "a list
    of search results" too and 296 of `browser_goal_modal_dialog`'s 4,000 questions
    in an earlier build said `no` to a goal the page met."""
    import json
    import random

    for layout in browser.LAYOUTS:
        for i in range(20):
            tree, _, _ = browser.build_page(layout, random.Random(f"x:{layout}:{i}"))
            got = _satisfied(json.loads(browser._state(tree)))
            assert got == {browser.LAYOUT_CONTENT[layout]}, (layout, got)


def test_goal_labels_agree_with_an_independent_reading_of_the_tree(tasks):
    import json
    import re

    for t, ex, q in _bulk(tasks, "goal", n=200):
        want = re.match(r"The goal is to reach a page that shows (.*)\.",
                        q.instructions).group(1)
        assert int(want in _satisfied(json.loads(ex.state))) == q.target, (t.name, want)


def test_the_goal_string_alone_does_not_predict_the_answer(tasks):
    """It did. Every content string was `yes` on its own task and `no` on the seven
    others, so the prior depended on which side of the holdout the string lived on: a
    lookup fitted on an earlier build's train split scored 0.1482 on `testreal_eval` -- an
    inverted shortcut, not merely a present one. Drawing the distractor from the layout's
    own group makes each string 50/50 corpus-wide."""
    from collections import Counter, defaultdict

    lut = defaultdict(Counter)
    for t, ex, q in _bulk(tasks, "goal", n=300):
        lut[q.instructions][q.options[q.target]] += 1
    n = sum(sum(v.values()) for v in lut.values())
    ceiling = sum(v.most_common(1)[0][1] for v in lut.values()) / n
    assert ceiling <= 0.56, f"goal string alone scores {ceiling:.2%}"


def test_training_never_names_a_held_out_layouts_content(tasks):
    """A structural holdout that describes the held-out structures in training is not one.
    The three reserved layouts' content strings must not appear in a trained task at all,
    as a goal or as a distractor."""
    reserved = {browser.LAYOUT_CONTENT[l] for l in browser.HELDOUT_LAYOUTS + browser.DEV_LAYOUTS}
    test = {browser.LAYOUT_CONTENT[l] for l in browser.HELDOUT_LAYOUTS}
    for t, ex, q in _bulk(tasks, "goal", n=300):
        if t.force_split == "devreal":        # nor does dev describe test's layouts
            assert not any(r in (q.instructions or "") for r in test), (t.name, q.instructions)
        if t.force_split != "train":
            continue
        assert not any(r in (q.instructions or "") for r in reserved), (t.name, q.instructions)


# A value-shape classifier written independently of `browser.FIELD_KINDS`: regexes over
# the value's surface form, never the generator's lambdas.
_SHAPES = [
    ("email address", r"[a-z]+@example\.com"),
    ("amount in GBP", "\u00a3" + r"\d+\.\d\d"),
    ("amount in EUR", "\u20ac" + r"\d+\.\d\d"),
    ("time of day", r"\d\d:\d\d"),
    ("date", r"\d\d [A-Z][a-z]+ \d{4}"),
    ("phone number", r"01632 \d{6}"),
    ("postcode", r"[A-H]\d{1,2} \dXT"),
    ("percentage", r"\d+(\.\d)?%"),
    ("web address", r"https://[a-z]+\.example\.org/[a-z]+"),
    ("IP address", r"10\.\d{1,3}\.\d{1,3}\.\d{1,3}"),
    ("colour code", r"#[0-9a-f]{6}"),
    ("weight", r"\d+\.\d kg"),
    ("distance", r"\d+\.\d km"),
    ("temperature", r"-?\d+ \u00b0C"),
    ("duration", r"\d+ (days|weeks|hours)"),
    ("software version", r"v\d+\.\d+\.\d+"),
    ("sort code", r"\d\d-\d\d-\d\d"),
    ("file name", r"[a-z]+_\d+\.(pdf|csv|xlsx)"),
    ("day of the week", r"(Mon|Tues|Wednes|Thurs|Fri|Satur|Sun)day"),
    ("map coordinates", r"-?\d+\.\d{3}, -?\d+\.\d{3}"),
]


def _kind_of(value):
    import re

    got = [k for k, rx in _SHAPES if re.fullmatch(rx, value)]
    assert len(got) == 1, (value, got)
    return got[0]


def _nodes(tree):
    out = [tree]
    for c in tree.get("children", []):
        out.extend(_nodes(c))
    return out


def _field_examples(n=600):
    t = next(x for x in browser.tasks() if x.name == "browser_field_settings_form")
    return list(t.load(n))


def test_field_kinds_are_in_the_tree_and_exactly_one_field_expects_the_value():
    """The re-derivation: read the value's kind off its shape, find the one group named
    by the goal, find the one textbox in it declaring that kind, and that is the target --
    with no reference to how the example was generated."""
    import json

    assert {k for k, _ in _SHAPES} == set(browser.FIELD_KINDS)
    n = 0
    for ex in _field_examples():
        q = ex.questions[0]
        value, section = q.meta["value"], q.meta["section"]
        assert f"'{value}'" in q.instructions and section in q.instructions
        tree = json.loads(ex.state)
        boxes = [b for b in _nodes(tree) if b["role"] == "textbox"]
        assert [b["name"] for b in boxes] == q.options
        groups = [g for g in _nodes(tree) if section in (g.get("name") or "")]
        assert len(groups) == 1 and groups[0]["role"] == "group", section
        want = [b["name"] for b in _nodes(groups[0])
                if b["role"] == "textbox" and b.get("expects") == _kind_of(value)]
        assert want == [q.options[q.target]], (q.instructions, want)
        n += 1
    assert n >= 500, n


# ------------------------- the domain-14 audit: value-blind readers must be at chance

def test_field_a_reader_who_ignores_the_value_is_at_chance():
    """THE `field_for_value` LEAK. Version 2 gave every field one of six kinds, the gold's
    exactly once and the rest drawn with replacement, so the gold was the one field whose
    kind nobody else on the page had: "pick a field with a page-unique `expects`" -- which
    never reads the value -- scored 0.68 against a 0.15 chance (an independent audit).
    Every section now holds the same kinds, so that reader must score chance."""
    import json
    from collections import Counter

    score = chance = 0.0
    exs = _field_examples(800)
    for ex in exs:
        q = ex.questions[0]
        boxes = [b for b in _nodes(json.loads(ex.state)) if b["role"] == "textbox"]
        c = Counter(b["expects"] for b in boxes)
        uniq = [i for i, b in enumerate(boxes) if c[b["expects"]] == 1]
        score += (q.target in uniq) / len(uniq) if uniq else 1 / len(boxes)
        chance += 1 / len(boxes)
    score, chance = score / len(exs), chance / len(exs)
    assert score <= chance + 0.03, f"unique-kind reader {score:.3f} vs chance {chance:.3f}"


def test_field_every_kind_on_the_page_occurs_once_per_section():
    """The structural form of the same guarantee: the gold carries no property -- kind
    frequency, kind count, position in its section -- that the other fields lack."""
    import json
    from collections import Counter

    for ex in _field_examples(300):
        tree = json.loads(ex.state)
        groups = [g for g in _nodes(tree) if g["role"] == "group"]
        kinds = [sorted(b["expects"] for b in g["children"]) for g in groups]
        assert all(k == kinds[0] for k in kinds)
        assert all(v == 1 for v in Counter(kinds[0]).values())


def test_field_option_counts_span_two_to_eighty():
    ks = [len(ex.questions[0].options) for ex in _field_examples(600)]
    assert min(ks) == 2 and max(ks) == 80


def _click(tasks, n=150):
    return [(t, ex, ex.questions[0]) for t, ex, _ in _bulk(tasks, "click", n=n)]


def test_click_a_reader_who_ignores_the_value_is_at_chance(tasks):
    """The same leak in `click`: the gold only had to have a page-unique value, so on a
    Status goal the gold was the one record whose status nobody else shared, and "pick
    such a record" scored 0.64 against 0.19 (Owner: 0.49 against 0.14). Every value of
    the named attribute is now distinct on the page."""
    import json
    import re

    for t, ex, q in _click(tasks):
        attr = q.meta["attr"]
        if attr == "Site":                     # distinct by construction; see below
            continue
        tree = json.loads(ex.state)
        per_record = []
        for rec in _records(tree):
            text = " | ".join(n.get("name", "") for n in _nodes(rec))
            if attr == "Status":
                got = [s for s in browser.STATUSES if re.search(rf"(?<!\w){s}(?!\w)", text)]
            else:
                got = re.findall(r"[A-Z]\. [A-Z][a-z]+", text)
            assert len(got) == 1, (t.name, attr, got)
            per_record.append(got[0])
        assert len(per_record) == len(q.options), t.name
        assert len(set(per_record)) == len(per_record), (t.name, attr, per_record)


def _records(tree):
    """The highest ancestor of each link that holds no other link, found without knowing
    the layout."""
    def links(n):
        return sum(x["role"] == "link" for x in _nodes(n))

    out = []

    def visit(n):
        if links(n) == 1:
            out.append(n)
            return
        for c in n.get("children", []):
            visit(c)
    visit(tree)
    return out


def test_click_named_value_occurs_exactly_once(tasks):
    """The attribute a goal names occurs once on the page, case-blind. `_article` and
    `_modal_dialog` named each record's group after its Site and headed it with the Site
    again (8 % of click questions named a value found twice), and Status 'open' matched
    every "Open <ref>" control to a case-blind reader (4.4 %)."""
    import re

    for t, ex, q in _click(tasks):
        v = q.meta["value"]
        assert f"'{v}'" in q.instructions
        hits = re.findall(rf"(?<!\w){re.escape(v)}(?!\w)", ex.state, re.I)
        assert len(hits) == 1, (t.name, v, len(hits))


def test_site_names_carry_no_position():
    """"<city> <i>" put the record's index into the name a goal quotes."""
    import re

    assert not any(re.search(r"\d", s) for s in browser.SITES)
    assert not any(a != b and a in b for a in browser.SITES for b in browser.SITES)


def test_goal_no_word_of_the_title_predicts_the_label(tasks):
    """Titles named the layout ("Billing wizard", "Search results for billing", "Billing
    log"), and the goal describes the layout, so "a title word occurs in the goal" scored
    0.81 on `goal_reached` -- and would have carried to the held-out layouts untouched.
    The same holds for every other *name* on the page, not only the title."""
    import json
    import re

    stop = {"a", "an", "the", "of", "with", "for", "and", "that", "is", "it", "over",
            "across", "behind", "page", "shows", "to", "in", "split", "by"}

    def words(s):
        return {w for w in re.findall(r"[a-z]+", s.lower()) if w not in stop and len(w) > 2}

    title_hits = name_hits = total = 0
    for t, ex, q in _bulk(tasks, "goal", n=200):
        tree = json.loads(ex.state)
        want = words(q.meta["want"])
        yes = q.options[q.target] == "yes"
        title_hits += bool(words(tree.get("name", "")) & want) == yes
        names = set().union(*(words(n.get("name", "")) for n in _nodes(tree)))
        name_hits += bool(names & want) == yes
        total += 1
    assert title_hits / total <= 0.55, f"title words predict {title_hits / total:.3f}"
    assert name_hits / total <= 0.56, f"page names predict {name_hits / total:.3f}"


def test_phrasing_specs_match_the_templates_the_code_emits():
    """Every template the generator sends through the phrasing bank is registered, with
    the exact text and placeholders, in `assets/phrasing_specs/d14.json`."""

    from lod.paths import ASSETS
    spec = json.loads((ASSETS / "phrasing_specs" / "d14.json").read_text(encoding="utf-8"))
    emitted = {tid: tpl for tid, tpl in (browser.CLICK_Q, browser.CLICK_I, browser.GOAL_Q,
                                          browser.GOAL_I, browser.FIELD_Q, browser.FIELD_I)}
    assert {k: v["template"] for k, v in spec.items()} == emitted
