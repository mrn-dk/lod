"""The dev split holds out whole structures, disjoint from train and from test.

`devreal` used to carry `_eval` twins -- fresh draws of *trained* structures -- so it
measured recall of a structure while `testreal` measured transfer to an unseen one, and the
two disagreed about which checkpoint and which temperature were right. Every generator
that forces held-out structures to `testreal` also forces a second, disjoint held-out set
to `devreal`; the `select` stage keeps only the dev tasks whose family never trains.

These tests check the three things that make that true: `family_of` names the held-out
unit, each generator's dev held-out families exist and touch neither train nor test, and
`select` refuses a corpus where they would.
"""
from __future__ import annotations

import importlib
import json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from lod.corpus.controllers import build
from lod.corpus.repositories import raw_store as store
from lod.corpus.services.sources.real.base import RealTask, family_of, split_of
from lod.corpus.services.splits import SplitError
from lod.paths import RAW_ROOT

# every source that forces held-out structures to testreal
GENERATORS = (
    "synth.rules", "synth.browser", "synth.chessfacts", "synth.sensors", "synth.toolgate",
    "synth.entityres", "synth.diffs", "synth.scheduling", "synth.relational",
    "synth.entities", "synth.compose", "synth.posterior", "synth.catalog",
    "real.lichess", "real.loghub", "real.pdfdocs", "real.multilingual", "real.sentiment",
    "real.topic", "real.gefs", "real.spf", "real.manifold", "real.tabfact",
)


def _task(name: str, family: str | None = None, **kw) -> RealTask:
    return RealTask(row=33, name=name, licence="x", url="x", load=lambda n: [],
                    real=False, family=family, **kw)


# ---- family_of ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,family", [
    ("rule_threshold", "rule_threshold"),
    ("rule_threshold_eval", "rule_threshold"),
    ("toolgate_gate_cli_eval", "toolgate_gate_cli"),
    ("sched_mx_mn_eval", "sched_mx_mn"),
    ("x_dev", "x"), ("x_test", "x"), ("x_train", "x"), ("x_probe", "x"),
    # only a trailing twin suffix is stripped
    ("tabfact_entailed_dev_tables", "tabfact_entailed_dev_tables"),
    ("x_eval_y", "x_eval_y"),
])
def test_family_of_strips_only_the_twin_suffix(name, family):
    assert family_of(_task(name)) == family


def test_declared_family_wins():
    assert family_of(_task("goemotions__joy", family="goemotions")) == "goemotions"
    assert family_of(_task("rule_x_eval", family="rule_y")) == "rule_y"


# ---- every generator ---------------------------------------------------------------------

@pytest.fixture(autouse=True, scope="module")
def _raw_store():
    """Real sources enumerate their tasks from the fetched cache; point the store at it
    for the module and restore it afterwards."""
    before = store.RAW_ROOT
    store.configure(RAW_ROOT)
    yield
    store.configure(before)


def _tasks(modname: str) -> list[RealTask]:
    mod = importlib.import_module(f"lod.corpus.services.sources.{modname}")
    try:
        tasks = list(mod.tasks())
    except FileNotFoundError as e:          # a real source whose cache is not fetched here
        pytest.skip(f"{modname}: source not cached ({e})")
    if modname.startswith("real.") and not tasks:
        pytest.skip(f"{modname}: source not cached")
    return tasks


def _families(tasks: list[RealTask]) -> dict[str, set[str]]:
    by: dict[str, set[str]] = defaultdict(set)
    for t in tasks:
        by[t.force_split or split_of(t.name, t.row)].add(family_of(t))
    return by


@pytest.mark.parametrize("modname", GENERATORS)
def test_generator_holds_out_dev_structures(modname):
    """Dev carries at least one family that neither trains nor is tested on."""
    fam = _families(_tasks(modname))
    held_dev = fam["devreal"] - fam["train"] - fam["testreal"]
    assert held_dev, f"{modname}: no family-disjoint dev structure ({sorted(fam['devreal'])})"


@pytest.mark.parametrize("modname", GENERATORS)
def test_dev_holdout_is_disjoint_from_test(modname):
    """A dev family that does not train must not be a test family either: test is read
    once, at the end, and nothing that selects a checkpoint may share its structures."""
    fam = _families(_tasks(modname))
    leaked = (fam["devreal"] - fam["train"]) & fam["testreal"]
    assert not leaked, f"{modname}: dev held-out families also in testreal: {sorted(leaked)}"


@pytest.mark.parametrize("modname", GENERATORS)
def test_test_holdout_survives(modname):
    """Adding dev structures must not have emptied test's own held-out set."""
    fam = _families(_tasks(modname))
    assert fam["testreal"] - fam["train"], f"{modname}: no held-out test family left"


@pytest.mark.parametrize("modname", GENERATORS)
def test_an_undeclared_twin_is_a_trained_family(modname):
    """A dev/test task named as a twin (`_eval`, `_dev`, ...) and carrying no declared
    family must fall into a trained family, or devood would count a fresh draw of a
    trained structure as a held-out one -- `sensor_action_trained_dev` stripped to
    `sensor_action_trained`, which no train task was called, and did exactly that."""
    tasks = _tasks(modname)
    fam = _families(tasks)
    bad = [t.name for t in tasks
           if not t.family and (t.force_split or split_of(t.name, t.row)) != "train"
           and family_of(t) != t.name and family_of(t) not in fam["train"]]
    assert not bad, f"{modname}: twins whose family never trains: {bad}"


@pytest.mark.parametrize("modname", GENERATORS)
def test_every_forced_family_lands_in_one_split(modname):
    """Tasks of one family are routed together, or a family could be both trained and
    held out. Twins are the exception by design: a trained family's `_eval` draw sits in
    dev or test *as* that family, which is what excludes it from devood."""
    tasks = _tasks(modname)
    held: dict[str, set[str]] = defaultdict(set)
    fam = _families(tasks)
    for t in tasks:
        f = family_of(t)
        if f not in fam["train"]:
            held[f].add(t.force_split or split_of(t.name, t.row))
    split_twice = {f: s for f, s in held.items() if len(s) > 1}
    assert not split_twice, f"{modname}: held-out family in two splits: {split_twice}"


# ---- the select stage ----------------------------------------------------------------------

def _corpus(tmp: Path, tasks: list[tuple[str, int, str, str | None]], n_each: int = 6) -> Path:
    """tasks: (name, row, split, family or None). Writes meta.json and the eval splits."""
    d = tmp / "corpus"
    d.mkdir(parents=True, exist_ok=True)
    meta = {"tasks": []}
    files: dict[str, list[str]] = defaultdict(list)
    for name, row, split, family in tasks:
        entry = {"name": name, "row": row, "split": split, "examples": n_each}
        if family is not None:
            entry["family"] = family
        meta["tasks"].append(entry)
        for i in range(n_each):
            files[split].append(json.dumps({
                "task": name, "state": f"{name} {i}",
                "questions": [{"id": "q", "question": "?", "options": ["a", "b"],
                               "target": i % 2}]}))
    (d / "meta.json").write_text(json.dumps(meta))
    for split in ("train", "devreal", "testreal"):
        (d / f"{split}.jsonl").write_text("".join(l + "\n" for l in files[split]))
    return d


def _make(corpus: Path, *extra: str) -> SimpleNamespace:
    """Run the select stage; -> returncode (1 on a refused corpus) and the log."""
    lines: list[str] = []
    kw = {"drop_test_overlap": "--drop-test-overlap" in extra}
    if "--sel-questions" in extra:
        kw["sel_questions"] = int(extra[extra.index("--sel-questions") + 1])
    try:
        build.select(corpus, log=lines.append, **kw)
        code, err = 0, ""
    except SplitError as e:
        code, err = 1, str(e)
    return SimpleNamespace(returncode=code, stdout="\n".join(lines), stderr=err)


def _tasks_in(p: Path) -> set[str]:
    return {json.loads(l)["task"] for l in open(p) if l.strip()}


def test_devood_keeps_only_families_that_never_train(tmp_path):
    corpus = _corpus(tmp_path, [
        ("rule_a", 33, "train", "rule_a"),
        ("rule_a_eval", 33, "devreal", "rule_a"),       # a twin: recall, not transfer
        ("rule_dev1", 33, "devreal", "rule_dev1"),      # a dev held-out structure
        ("rule_test1", 33, "testreal", "rule_test1"),
        ("ds__x", 20, "train", "ds"),
        ("ds__y", 20, "devreal", "ds"),                 # a column of a trained dataset
        ("other__y", 20, "devreal", "other"),
    ])
    r = _make(corpus)
    assert r.returncode == 0, r.stderr
    assert _tasks_in(corpus / "devood.jsonl") == {"rule_dev1", "other__y"}
    assert _tasks_in(corpus / "devood_sel.jsonl") <= {"rule_dev1", "other__y"}
    summary = json.loads((corpus / "devood.json").read_text())
    assert summary["devood"]["families"] == ["other", "rule_dev1"]
    assert "Rule application" in r.stdout          # composition by domain is printed


def test_devood_refuses_a_dev_family_that_is_also_tested(tmp_path):
    corpus = _corpus(tmp_path, [
        ("s_train", 37, "train", "s_trained"),
        ("s_held_dev", 37, "devreal", "s_held"),
        ("s_held_test", 37, "testreal", "s_held"),     # one structure in dev AND test
    ])
    r = _make(corpus)
    assert r.returncode != 0
    assert "s_held" in (r.stderr + r.stdout)
    assert not (corpus / "devood.jsonl").exists()
    # the legacy escape hatch drops it instead, and says so
    r = _make(corpus, "--drop-test-overlap")
    assert r.returncode == 0, r.stderr
    assert "WARNING" in r.stdout
    assert _tasks_in(corpus / "devood.jsonl") == set()


def test_devood_recomputes_families_on_a_legacy_corpus(tmp_path):
    """No `family` in meta.json: the name decides, a Hub task at its dataset."""
    corpus = _corpus(tmp_path, [
        ("goemotions__joy", 20, "train", None),
        ("goemotions__sadness", 20, "devreal", None),
        ("toolgate_gate_cli", 38, "train", None),
        ("toolgate_gate_cli_eval", 38, "devreal", None),
        ("sent44_tfns", 44, "devreal", None),
        ("sent44_imdb", 44, "testreal", None),
    ])
    r = _make(corpus)
    assert r.returncode == 0, r.stderr
    assert "legacy" in r.stdout
    assert _tasks_in(corpus / "devood.jsonl") == {"sent44_tfns"}


def test_devood_sel_follows_testreal_domain_shares(tmp_path):
    """Selection quota per domain tracks testreal's share, not the task count: one
    domain with many small tasks must not swamp the subsample."""
    tasks = [(f"gh_{i}", 5, "devreal", f"gh_{i}") for i in range(40)]          # domain 8
    tasks += [(f"sched_d{i}", 41, "devreal", f"sched_d{i}") for i in range(2)]  # domain 19
    tasks += [("gh_t", 5, "testreal", "gh_t"), ("sched_t", 41, "testreal", "sched_t")]
    corpus = _corpus(tmp_path, tasks, n_each=60)
    # testreal: equal questions in both domains -> each should get about half of sel
    r = _make(corpus, "--sel-questions", "100")
    assert r.returncode == 0, r.stderr
    sel = [json.loads(l)["task"] for l in open(corpus / "devood_sel.jsonl") if l.strip()]
    gh = sum(t.startswith("gh_") for t in sel)
    sched = sum(t.startswith("sched_") for t in sel)
    assert 40 <= gh <= 60 and 40 <= sched <= 60, (gh, sched)
