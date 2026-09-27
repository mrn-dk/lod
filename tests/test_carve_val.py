"""val is carved by group: twins of a val example must not be in train."""
from lod.corpus.services.generate import carve_val
from lod.schema import Example, Question


def _ex(task, group, i):
    return Example(state=f"s{i}", task=task, questions=[
        Question("q", "?", ["no", "yes"], i % 2, {"group": group})])


def test_twins_land_on_one_side():
    train = [_ex("t", f"ent{g}", g * 3 + k) for g in range(100) for k in range(3)]
    kept, val = carve_val(train, seed=0)
    assert val and kept and len(kept) + len(val) == 300
    vg = {e.questions[0].meta["group"] for e in val}
    kg = {e.questions[0].meta["group"] for e in kept}
    assert vg & kg == set()


def test_ungrouped_examples_still_carve_about_five_percent():
    train = [Example(state=f"s{i}", task="t",
                     questions=[Question("q", "?", ["a", "b"], 0)]) for i in range(400)]
    kept, val = carve_val(train, seed=0)
    assert len(val) == 20 and len(kept) == 380
