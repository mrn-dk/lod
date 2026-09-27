"""Tests for the row-7 PDF document-structure source.

None of these need `--extra pdf` or a populated cache. The question-derivation path is
exercised against a hand-built outline plan, which is exactly the object `pymupdf` would
have produced, so the label derivation is tested without a PDF anywhere. The two tests
that need the real cache skip when it is absent and assert non-emptiness when it is
present -- a loader that can quietly return `[]` is the failure mode that let row 30 go to
zero unnoticed.
"""

from __future__ import annotations

import random

import pytest

from lod.corpus.services.sources.real import pdfdocs


# ------------------------------------------------------------------- pure functions


@pytest.mark.parametrize("raw,want", [
    ("1 Introduction", "Introduction"),
    ("3.1. Pseudospectral collocation", "Pseudospectral collocation"),
    ("2.The FIV nonlinear energy harvesting system",
     "The FIV nonlinear energy harvesting system"),
    ("IV. Results", "Results"),
    ("A.2 Proof of Theorem 1", "Proof of Theorem 1"),
    ("(3) Discussion", "Discussion"),
    ("  Related  Work ", "Related Work"),
])
def test_numbering_is_stripped(raw, want):
    assert pdfdocs.clean_title(raw) == want


@pytest.mark.parametrize("raw", [
    "A Survey of Tensor Methods",      # a bare capital is a word, not a number
    "3D Convolutional Networks",       # a bare digit glued to a letter is a word
    "COVID-19 Case Counts",
])
def test_numbering_stripper_leaves_words_alone(raw):
    assert pdfdocs.clean_title(raw) == raw


def test_ordering_questions_cannot_be_counted_to():
    """If numbering survived, 'what follows 3.1' is arithmetic and the passage is dead
    weight. This is the single assumption the ordering families rest on."""
    titles = [pdfdocs.clean_title(t) for t in
              ["1 Introduction", "2 Methods", "2.1 Data", "3 Results"]]
    assert not any(any(ch.isdigit() for ch in t) for t in titles), titles


def test_reference_pages_are_not_sections():
    refs = "\n".join(f"[{i}] A. Author, B. Author. Some paper title. Venue, 2019."
                     for i in range(1, 30))
    assert pdfdocs.is_reference_page(refs)
    prose = "\n".join(["We now describe the optimisation procedure in detail."] * 30)
    assert not pdfdocs.is_reference_page(prose)


def test_leaks_only_fires_on_the_gold_option():
    state = "the section on Related Work argued that ..."
    assert pdfdocs.leaks(state, ["Related Work", "Methods"], "Related Work")
    assert not pdfdocs.leaks(state, ["Related Work", "Methods"], "Methods")


# ------------------------------------------------- question derivation, no PDF needed


# Deliberately contains none of the outline titles below: the leak filter drops any
# example whose gold option is readable off the page, which is the behaviour under test
# in `test_no_example_names_its_own_answer` and would otherwise silently thin this one.
PAGE = ("We report the mean absolute deviation on three benchmark collections against "
        "the held-out split, averaged over five seeds. " * 12)

PLAN = {
    "id": "0000.00000", "licence": "CC-BY-4.0", "url": "https://arxiv.org/abs/0000.00000",
    "pages": 20,
    "entries": [
        {"level": 1, "title": "Introduction", "page": 1},
        {"level": 1, "title": "Related Work", "page": 3},
        {"level": 1, "title": "Method", "page": 5},
        {"level": 2, "title": "Estimator", "page": 7},
        {"level": 2, "title": "Optimisation", "page": 9},
        {"level": 1, "title": "Experiments", "page": 11},
        {"level": 1, "title": "Discussion", "page": 14},
        {"level": 1, "title": "Discussion of threats to validity", "page": 17},
    ],
    "interior": [{"page": 2, "entry": 0}, {"page": 4, "entry": 1}, {"page": 6, "entry": 2},
                 {"page": 8, "entry": 3}, {"page": 10, "entry": 4},
                 {"page": 12, "entry": 5}, {"page": 13, "entry": 5}],
    "heading_pages": [1, 3, 5, 7, 9, 11, 14, 17],
    "text": {str(p): "\n".join([PAGE] * 10) for p in range(1, 21)},
}


@pytest.fixture
def fake_plan(monkeypatch):
    monkeypatch.setattr(pdfdocs, "document", lambda rec, converter: dict(PLAN))
    return [{"id": "0000.00000", "path": "x.pdf", "licence": "CC-BY-4.0"}]


def _all(family, recs, n=50):
    return list(pdfdocs._examples(family, pdfdocs.MD, recs, n))


def test_every_family_derives_targets(fake_plan):
    for family in pdfdocs.FAMILIES:
        got = _all(family, fake_plan)
        assert got, family
        for ex in got:
            for q in ex.questions:
                probs = q.target_probs()
                assert probs is not None and abs(sum(probs) - 1.0) < 1e-9
                assert len(q.options) == len(set(q.options)), f"{family}: duplicate options"
                assert pdfdocs.MIN_OPTS <= len(q.options) <= pdfdocs.MAX_OPTS
                assert q.meta["converter"] == pdfdocs.MD
                assert q.meta["licence"]


def test_section_label_is_the_owning_outline_entry(fake_plan):
    by_page = {ex.questions[0].meta["page"]: ex.questions[0] for ex in
               _all("section", fake_plan)}
    assert by_page[2].options[by_page[2].target] == "Introduction"
    assert by_page[8].options[by_page[8].target] == "Estimator"
    assert by_page[12].options[by_page[12].target] == "Experiments"


def test_ordering_follows_the_outline(fake_plan):
    nxt = {q.meta["heading"]: q.options[q.target]
           for ex in _all("next_section", fake_plan) for q in ex.questions}
    assert nxt["Method"] == "Estimator"
    assert nxt["Optimisation"] == "Experiments"
    prv = {q.meta["heading"]: q.options[q.target]
           for ex in _all("prev_section", fake_plan) for q in ex.questions}
    assert prv["Experiments"] == "Optimisation"


def test_parent_is_the_enclosing_top_level_section(fake_plan):
    got = {q.meta["heading"]: q.options[q.target]
           for ex in _all("parent", fake_plan) for q in ex.questions}
    assert got == {"Estimator": "Method", "Optimisation": "Method"}


def test_depth_is_the_outline_level(fake_plan):
    for ex in _all("depth", fake_plan):
        q = ex.questions[0]
        assert q.options[q.target] == pdfdocs.DEPTHS[q.meta["level"] - 1]


def test_no_example_names_its_own_answer(fake_plan):
    """Row 36 shipped at 100 % solvable by string match. Hold the line here."""
    free = 0
    total = 0
    for family in pdfdocs.FAMILIES:
        for ex in _all(family, fake_plan):
            for q in ex.questions:
                total += 1
                gold = q.options[q.target]
                if pdfdocs.normalise(gold) in pdfdocs.normalise(ex.state):
                    free += 1
    assert total and free == 0, f"{free}/{total} answerable by string match"


def test_ordering_never_offers_its_own_anchor(fake_plan):
    """An option set that contains the heading the question names hands over a free
    elimination, and reads like a bug."""
    for family in ("next_section", "prev_section", "parent"):
        for ex in _all(family, fake_plan):
            for q in ex.questions:
                assert q.meta["heading"] not in q.options, (family, q.options)


def test_no_option_contains_another(fake_plan):
    """"Solution strategy" against "Solution strategy and design of the numerical
    framework" is a coin flip, not a structure question."""
    for family in pdfdocs.FAMILIES:
        for ex in _all(family, fake_plan):
            for q in ex.questions:
                ns = [pdfdocs.normalise(o) for o in q.options]
                for i, a in enumerate(ns):
                    for j, b in enumerate(ns):
                        assert i == j or (a not in b), (family, q.options)


def test_whitespace_is_not_content():
    """`pdftotext -layout` renders a full-page figure as a screenful of spaces, which a
    character-count floor waved through."""
    spaces = {"text": {"1": "Figure 3.\n" + "\n".join(" " * 120 for _ in range(60))}}
    assert pdfdocs.passage(spaces, 1) is None
    # A plot's axis labels and legend fragments clear a word count but are not prose.
    plot = {"text": {"1": "\n".join(["0.25   0.50   0.75   F=0.7"] * 60)}}
    assert pdfdocs.passage(plot, 1) is None
    # the real shape of the miss: `pdftotext -layout` lays a plot's labels out across one
    # very wide line, so the line is long in `split()` terms and still not a sentence
    wide = {"text": {"1": "\n".join(
        ["(a)" + " " * 40 + "25" + " " * 30 + "F=0.7   F=0.3   F=0.5" + " " * 20 + "- /2"]
        * 60)}}
    assert pdfdocs.passage(wide, 1) is None
    real = {"text": {"1": "\n".join([" ".join(["word"] * 12)] * 20)}}
    assert pdfdocs.passage(real, 1) is not None
    # one long paragraph on one line is prose too: pymupdf4llm writes them that way
    one_line = {"text": {"1": " ".join(["word"] * (pdfdocs.MIN_WORDS + 10))}}
    assert pdfdocs.passage(one_line, 1) is not None


def test_truncated_outline_entries_are_dropped():
    assert pdfdocs.clean_title("2.2 preserving homeomorphisms")[0].islower()


def test_generation_is_seeded(fake_plan):
    a = [(e.state, e.questions[0].options) for e in _all("section", fake_plan)]
    b = [(e.state, e.questions[0].options) for e in _all("section", fake_plan)]
    assert a == b


# ------------------------------------------------------------------ the real cache


@pytest.fixture
def cached_tasks():
    if not pdfdocs.MANIFEST.exists():
        pytest.skip("no PDF cache; run scripts/fetch_data.py --rows 7")
    pytest.importorskip("pymupdf")
    ts = pdfdocs.tasks()
    if not ts:
        pytest.skip("PDF cache is empty")
    return ts


def test_tasks_are_row_seven_and_name_their_converter(cached_tasks):
    for t in cached_tasks:
        assert t.row == 7 and t.real and t.licence
        assert t.name.rsplit("_", 1)[-1] in pdfdocs.CONVERTERS or \
            t.name.endswith(pdfdocs.MD) or t.name.endswith(pdfdocs.PT)


def test_every_split_carries_both_converters(cached_tasks):
    """A split with one converter in it cannot answer the question this row exists to
    ask: did the model learn to read past damage, or learn one converter's conventions."""
    import collections
    per = collections.defaultdict(set)
    for t in cached_tasks:
        per[t.force_split].add(t.name.rsplit("_", 1)[-1])
    assert set(per) == {"train", "devreal", "testreal"}, dict(per)
    for split, convs in per.items():
        assert convs == {pdfdocs.MD, pdfdocs.PT}, (split, convs)


def test_a_held_out_family_is_paired_with_a_trained_one(cached_tasks):
    trained = {f for f, s in pdfdocs.FORCE_SPLIT.items() if s == "train"}
    assert "next_section" in trained and pdfdocs.FORCE_SPLIT["prev_section"] != "train"
    assert "depth" in trained and pdfdocs.FORCE_SPLIT["parent"] != "train"


def test_task_names_are_unique(cached_tasks):
    names = [t.name for t in cached_tasks]
    assert len(names) == len(set(names))


def test_loaders_are_not_silently_empty(cached_tasks):
    """The row-30 failure mode: a loader that returns `[]` and nobody notices."""
    for t in cached_tasks:
        assert list(t.load(5)), f"{t.name} produced nothing"


def test_documents_are_disjoint_between_tasks(cached_tasks):
    """Two tasks sharing a document would put the same state in two splits, and the
    devreal-against-train dedup would then delete the whole eval side."""
    seen: dict[str, str] = {}
    for t in cached_tasks:
        for ex in t.load(10):
            doc = ex.questions[0].meta["doc"]
            assert seen.setdefault(doc, t.name) == t.name, \
                f"{doc} is in both {seen[doc]} and {t.name}"


# ------------------------------------------------------------ domain-12 verification
#
# Measured on an earlier corpus build and on the eval dump of the model trained on it.
# Domain 12 scores 0.2831 over 166 testreal questions, the lowest in the corpus. Two
# things in this module account for it, and both are fixed above.


def test_no_distractor_is_readable_off_the_passage(fake_plan):
    """The leak filter was one-sided. `leaks()` drops an example whose *gold* is written
    on the page, so gold appeared in its own state 0.0 % of the time across all ten
    tasks of that build -- while a *distractor* did in 9.9 %-34.3 %. For an option-scoring
    model "this option is written on the page" is the loudest surface cue there is, and
    the build had made it a guarantee of being wrong: on the 28 of 166 testreal
    questions carrying a visible distractor the model scored 0.1786 (chance 0.1696)
    against 0.3043 on the rest, and picked a visible distractor 32.1 % of the time."""
    for family in pdfdocs.FAMILIES:
        for ex in _all(family, fake_plan):
            q = ex.questions[0]
            state = pdfdocs.normalise(ex.state)
            for i, option in enumerate(q.options):
                assert pdfdocs.normalise(option) not in state, (family, option)
            assert q.options[q.target] in q.options


def test_a_visible_title_is_dropped_from_the_option_pool():
    page = "As shown in the Related Work section, the estimator is consistent."
    titles = ["Introduction", "Related Work", "Method", "Experiments", "Discussion"]
    opts = pdfdocs.options_for(titles, "Introduction", random.Random(0), state=page)
    assert opts is not None
    assert "Related Work" not in opts
    assert "Introduction" in opts


def test_option_filter_still_returns_none_when_too_few_survive():
    """Filtering must not silently emit a two-option question: too small a pool is a
    dropped example, the same as every other shortfall here."""
    page = "Related Work Method Experiments Discussion all appear on this page."
    titles = ["Introduction", "Related Work", "Method", "Experiments", "Discussion"]
    assert pdfdocs.options_for(titles, "Introduction", random.Random(0), state=page) is None


def test_the_evaluated_split_carries_a_family_the_passage_decides(cached_tasks):
    """That build sent 166 questions here and every one was `prev_section`, whose answer
    is a property of the outline and not of the page: word-overlap ranking puts its gold
    at mean normalised rank 0.54, worse than random, and a lookup on the question text
    alone reproduces 0.967-1.000 of its labels. `section` is the family the passage
    decides, and it was trained-only, so the row's headline measured guessing at outline
    order rather than reading a converted PDF."""
    evaluated = {t.name.rsplit("_", 1)[0].removeprefix("pdfdocs_")
                 for t in cached_tasks if t.force_split != "train"}
    assert "section" in evaluated


def test_section_is_held_out_by_converter_not_by_family(cached_tasks):
    """Holding `section` out whole would leave it untrained, which is worse. One
    converter trains it and the other evaluates it, which is the transfer this row
    exists to ask about."""
    by_split = {t.name: t.force_split for t in cached_tasks}
    section = {n: s for n, s in by_split.items() if n.startswith("pdfdocs_section_")}
    assert set(section.values()) == {"train", "testreal"}, section


def test_families_are_the_held_out_units(cached_tasks):
    """`parent` (dev) and `prev_section` (test) are held out whole, so each is one family
    over both converters and none trains; `section` on the held-out converter is a family
    of its own, not the trained `section`."""
    from collections import defaultdict

    from lod.corpus.services.sources.real.base import family_of
    by: dict[str, set[str]] = defaultdict(set)
    for t in cached_tasks:
        by[t.force_split].add(family_of(t))
    assert "pdfdocs_parent" in by["devreal"] - by["train"] - by["testreal"]
    assert "pdfdocs_prev_section" in by["testreal"] - by["train"] - by["devreal"]
    assert "pdfdocs_section" in by["train"]
    if any(t.name.endswith(pdfdocs.PT) for t in cached_tasks):
        assert f"pdfdocs_section_{pdfdocs.PT}" in by["testreal"] - by["train"]


# ------------------------------------------------------------------ domain-12 audit
#
# An independent audit re-derived every label from the raw outline with its own code.
# Before these fixes it found 9/358 `section`, 17/277 `next_section`, 23/235
# `prev_section` and 3/230 `depth` labels that disagreed with the raw outline, 76/358
# `section` questions with two right answers, and 71/1,275 states that were a
# bibliography labelled "Conclusion".




def test_a_filtered_entry_still_ends_the_section_before_it():
    """"preserving homeomorphisms" starts lower-case and is not offered, but its pages
    are its own: they must not be labelled with the section that precedes it."""
    toc = [[1, "1 Introduction", 1], [1, "2 Method", 3], [2, "2.1 preserving homeomorphisms", 5],
           [1, "3 Experiments", 9], [1, "4 Discussion", 12], [1, "5 Conclusion", 15]]
    plan = pdfdocs.plan_from_toc(toc, 20)
    owner = {d["page"]: plan["entries"][d["entry"]]["title"] for d in plan["interior"]}
    assert owner[4] == "Method"
    assert not {6, 7, 8} & set(owner), owner      # the dropped entry's pages are skipped
    assert owner[10] == "Experiments"


def test_ordering_questions_skip_a_filtered_neighbour():
    toc = [[1, "1 Introduction", 1], [1, "2 Method", 3], [2, "2.1 preserving homeomorphisms", 5],
           [1, "3 Experiments", 9], [1, "4 Discussion", 12], [1, "5 Conclusion", 15]]
    plan = pdfdocs.plan_from_toc(toc, 20)
    titles = [e["title"] for e in plan["entries"]]
    method, exps = titles.index("Method"), titles.index("Experiments")
    assert pdfdocs.neighbour(plan, method, +1) is None     # really "preserving ..."
    assert pdfdocs.neighbour(plan, exps, -1) is None
    assert plan["entries"][pdfdocs.neighbour(plan, exps, +1)]["title"] == "Discussion"


@pytest.mark.parametrize("toc", [
    # the paper's title as the single root: every level is off by one
    [[1, "A Paper Title", 1], [2, "Introduction", 1], [2, "Method", 3], [2, "Results", 6],
     [2, "Conclusion", 9]],
    # figure captions bookmarked as sections
    [[1, "Introduction", 1], [1, "Figure 1: Overview of the system", 3], [1, "Method", 4],
     [1, "Results", 7], [1, "Conclusion", 9]],
    # LaTeX labels bookmarked as sections
    [[1, "Introduction", 1], [1, "Method", 3], [2, "thm:main", 4], [1, "Results", 7],
     [1, "Conclusion", 9]],
    # outline order disagrees with page order
    [[1, "Introduction", 1], [1, "Method", 6], [1, "Results", 3], [1, "Conclusion", 9]],
])
def test_an_outline_that_is_not_a_section_map_is_dropped(toc):
    assert pdfdocs.plan_from_toc(toc, 12) is None


def test_appendix_letters_are_numbering_only_where_the_outline_letters():
    toc = [[1, "1 Introduction", 1], [1, "2 Method", 3], [1, "3 Results", 5],
           [1, "A Code", 8], [1, "B Proof of Proposition 3.3", 10], [1, "C Extra", 12]]
    plan = pdfdocs.plan_from_toc(toc, 14)
    titles = [e["title"] for e in plan["entries"]]
    assert "Code" in titles and "Proof of Proposition 3.3" in titles, titles
    assert pdfdocs.clean_title("A Survey of Tensor Methods") == "A Survey of Tensor Methods"


def test_section_never_offers_the_golds_enclosing_section(fake_plan):
    """The page of "Estimator" is also a page of "Method": offering both is two right
    answers."""
    for ex in _all("section", fake_plan):
        q = ex.questions[0]
        if q.options[q.target] in ("Estimator", "Optimisation"):
            assert "Method" not in q.options, q.options


def test_a_question_never_names_a_duplicated_heading(monkeypatch):
    toc = [[1, "Introduction", 1], [1, "Study one", 3], [2, "Experiments", 4],
           [1, "Study two", 8], [2, "Experiments", 9], [1, "Discussion", 13],
           [1, "Conclusion", 16]]
    plan = pdfdocs.plan_from_toc(toc, 20)
    plan.update(id="0000.00001", licence="CC-BY-4.0", url="",
                text={str(p): "\n".join([PAGE] * 10) for p in range(1, 21)})
    monkeypatch.setattr(pdfdocs, "document", lambda rec, converter: plan)
    recs = [{"id": "0000.00001", "path": "x.pdf"}]
    for family in ("next_section", "prev_section", "depth", "parent"):
        for ex in _all(family, recs):
            assert ex.questions[0].meta["heading"] != "Experiments", family


def test_back_matter_is_not_a_passage():
    prose = ("We evaluate the estimator on three collections and report the mean absolute "
             "deviation over five seeds, comparing against the strongest baseline. ")
    # an un-bookmarked References heading starts on the page
    assert pdfdocs.is_back_matter(prose * 5 + "\n# **References**\n- [1] A. Author. Title.")
    # pymupdf4llm writes a bibliography as a few very long lines, which the line-counting
    # `is_reference_page` waved through
    bib = " ".join(f"- [{i}] Alice Smith and Bob Jones. A title about things. In Proceedings "
                   f"of the Conference on Things, pages 1-9, 20{10 + i % 10}." for i in range(12))
    assert not pdfdocs.is_reference_page(bib)
    assert pdfdocs.is_back_matter(bib)
    # a related-work page is dense in years and is still body text
    related = ("Prior work on sparse recovery (Smith et al., 2019; Jones et al., 2020) "
               "considers the noiseless case, while Lee and Park (2021) extend it. ") * 8
    assert not pdfdocs.is_back_matter(related)


def test_question_text_goes_through_the_phrasing_bank(fake_plan):
    """Every family's question is its template (the bank is empty) with the heading filled
    in, and the spec file registers exactly those templates."""
    import json
    from lod.paths import ASSETS
    spec = json.loads((ASSETS / "phrasing_specs" / "d12.json").read_text())
    for family in pdfdocs.FAMILIES:
        assert spec[pdfdocs.QUESTION_IDS[family]]["template"] == pdfdocs._QUESTIONS[family]
        for ex in _all(family, fake_plan)[:3]:
            q = ex.questions[0]
            assert "{" not in q.question
            if family != "section":
                assert f'"{q.meta["heading"]}"' in q.question


def test_a_figures_own_text_is_not_prose():
    """`pymupdf4llm` writes a plot's axis labels as one `<br>`-joined line inside
    picture-text markers; counted as prose, a page of figures passed the prose floor."""
    fig = ("<!-- Start of picture text --> " +
           "<br>".join(["Along track error", "Cross track error", "time seconds",
                        "heading angle degrees"] * 30) + " <!-- End of picture text -->")
    assert pdfdocs.prose_words(fig) == 0
    assert pdfdocs.passage({"text": {"1": fig + "\nFigure 7.14: Along and cross track errors."}},
                           1) is None


def test_the_prose_floor_applies_after_truncation():
    """`pdftotext -layout` pads with spaces; a cut at MAX_CHARS can keep only blanks."""
    padded = ("Table 2" + " " * 150 + "0.91\n") * (pdfdocs.MAX_CHARS // 150 + 1)
    prose = "\n".join([" ".join(["word"] * 12)] * 40)
    assert pdfdocs.passage({"text": {"1": prose}}, 1) is not None
    assert pdfdocs.passage({"text": {"1": padded + prose}}, 1) is None
