"""Published option definitions, and the ClassLabel names they need.

These are the two halves of the same problem -- an option the model cannot interpret.
`hf.label_value` recovers a name the fetch discarded; `descriptions.attach` adds the
publisher's definition of that name to half the examples. Both are corpus-build code, so
neither needs a GPU or the network (the CWE catalogue is the one exception and is skipped
when it has not been cached).
"""

import pytest

from lod.schema import Example, Question
from lod.corpus.services.sources.real import descriptions
from lod.corpus.services.sources.real.hf import (
    HFSpec,
    class_names,
    detect_label_cols,
    label_value,
)


# ---- ClassLabel names --------------------------------------------------------------

def test_label_value_maps_an_index_to_its_published_name():
    names = ["Adjustments", "Agreements", "Amendments"]
    assert label_value(0, names) == "Adjustments"
    assert label_value(2, names) == "Amendments"


def test_label_value_leaves_everything_else_alone():
    names = ["a", "b"]
    assert label_value("already a string", names) == "already a string"
    assert label_value(7, names) == "7"        # out of range: a count, not a class
    assert label_value(1, None) == "1"         # no names published
    assert label_value(True, names) == "True"  # bool is an int subclass; not an index


def test_numeric_class_names_are_ignored(tmp_path, monkeypatch):
    """LexGLUE SCOTUS publishes "1".."13"; substituting those would change nothing and
    would make the corpus claim a fix that did not happen."""
    from lod.corpus.repositories import raw_store as store
    monkeypatch.setattr(store, "RAW", tmp_path)
    store.save_features("scotusish", {"label": [str(i) for i in range(1, 14)]})
    store.save_features("ledgarish", {"label": ["Adjustments", "Agreements"]})
    assert class_names(HFSpec(row=17, path="scotusish", licence="x")) == {}
    assert "label" in class_names(HFSpec(row=17, path="ledgarish", licence="x"))


def test_detect_label_cols_reports_names_not_indices(tmp_path, monkeypatch):
    from lod.corpus.repositories import raw_store as store
    monkeypatch.setattr(store, "RAW", tmp_path)
    spec = HFSpec(row=17, path="ledgar", licence="x", label_cols=("label",))
    store.save_features(spec.key, {"label": ["Adjustments", "Agreements", "Amendments"]})
    rows = [{"text": "t" * 20, "label": i % 3} for i in range(30)]
    assert detect_label_cols(rows, spec, "text") == {
        "label": ["Adjustments", "Agreements", "Amendments"]}


def test_features_sidecar_round_trips(tmp_path, monkeypatch):
    from lod.corpus.repositories import raw_store as store
    monkeypatch.setattr(store, "RAW", tmp_path)
    store.save_features("k", {"col": ["x", "y"]})
    assert store.load_features("k") == {"col": ["x", "y"]}
    assert store.load_features("never-written") == {}


# ---- descriptions ------------------------------------------------------------------

def _goemotions(task: str, n: int) -> list[Example]:
    return [Example(task=task, state=f"comment {i}",
                    questions=[Question(id=task.split("__")[1],
                                        question="Does this comment express it?",
                                        options=["no", "yes"], target=i % 2)])
            for i in range(n)]


def test_goemotions_definition_lands_on_yes():
    got = descriptions.for_task("goemotions__anger", ["no", "yes"])
    assert got[0] is None
    assert "displeasure" in got[1]


def test_a_task_with_no_published_definitions_gets_none():
    assert descriptions.for_task("gh_label_someone_something", ["bug", "docs"]) is None
    assert descriptions.for_task("nvd_notAMetric", ["LOW", "HIGH"]) is None


def test_cvss_values_are_described_in_the_order_given():
    opts = ["NONE", "LOW", "HIGH"]
    got = descriptions.for_task("nvd_availabilityImpact", opts)
    assert got is not None and len(got) == 3
    assert "unaffected" in got[0] and "sustained" in got[2]
    # order follows the caller's option list, not the table's
    flipped = descriptions.for_task("nvd_availabilityImpact", list(reversed(opts)))
    assert flipped == list(reversed(got))


def test_attach_describes_exactly_half_and_is_deterministic():
    a, b = _goemotions("goemotions__anger", 200), _goemotions("goemotions__anger", 200)
    assert descriptions.attach(a) == 100
    descriptions.attach(b)
    described = lambda xs: {e.state for e in xs
                            if any(q.has_descriptions for q in e.questions)}
    assert described(a) == described(b)      # seed-0 hash, so a rebuild picks the same half


def test_attach_skips_tasks_with_nothing_published():
    ex = [Example(task="gh_label_x", state=f"s{i}",
                  questions=[Question(id="label", question="?", options=["a", "b"],
                                      target=0)]) for i in range(20)]
    assert descriptions.attach(ex) == 0
    assert all(not q.has_descriptions for e in ex for q in e.questions)


def test_attached_descriptions_survive_a_jsonl_round_trip(tmp_path):
    from lod.schema import read_jsonl, write_jsonl
    ex = _goemotions("goemotions__joy", 20)
    n = descriptions.attach(ex)
    p = tmp_path / "x.jsonl"
    write_jsonl(p, ex)
    back = read_jsonl(p)
    assert sum(1 for e in back
               for q in e.questions if q.has_descriptions) == n > 0


def test_cwe_descriptions_come_from_the_catalogue():
    """MITRE's own text, not a paraphrase. Skipped when the catalogue is not cached."""
    from lod.corpus.services.sources.base import CACHE
    if not (CACHE / "cwe_catalogue.json").exists():
        pytest.skip("CWE catalogue not fetched")
    got = descriptions.for_task("nvd_cwe", ["CWE-79", "CWE-99999"])
    assert "Cross-site Scripting" in got[0]
    assert got[1] is None            # not a real CWE: described as nothing, not invented
