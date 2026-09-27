"""Row 50: broad real-language decisions -- `sources/real/reallang.py`.

Unit tests run on synthetic raw rows in a private store. The `live` tests read the real
raw store (`LOD_RAW_ROOT`, filled by `scripts/fetch_data.py --rows 50`) and the DI 0.2
blocklist, and skip where either is absent.
"""
import json
from collections import defaultdict

import pytest

from lod import sentinel
from lod.corpus.services.sources.real import reallang as rl
from lod.corpus.repositories import raw_store as store
from lod.paths import RAW_ROOT
from lod.corpus.services.sources.real.base import family_of

LIVE_ROOT = RAW_ROOT
# the Decision Index 0.2 member datasets (methodology `benchmarks`); none may be a source
DI_MEMBERS = ("mmlu", "arc", "gsm8k", "hellaswag", "winogrande", "anli", "hle", "gpqa",
              "bbh", "banking77", "clinc", "esci", "acos", "vast", "isarcasm", "humicroedit",
              "finentity", "sata", "toolret", "bright", "bfcl", "api-bank", "apibank",
              "when2call", "ragtruth", "hover", "musr", "cladder", "cruxeval", "nli4ct",
              "contractnli", "routerbench", "sgd", "schema_guided", "chessbench",
              "forecastbench", "pop909", "habermas", "bpomp", "phishnchips", "newyorker")


@pytest.fixture
def raw(tmp_path, monkeypatch):
    (tmp_path / "store").mkdir()
    store.configure(tmp_path)
    monkeypatch.setattr(rl, "_POOLS", {})
    monkeypatch.setattr(rl, "DROPS", {})
    monkeypatch.setattr(rl, "_BLOCK", False)          # no blocklist in unit tests
    monkeypatch.setattr(rl, "DI_DROPS", defaultdict(int))
    yield tmp_path
    store.clear_cache()


def _save(key, rows):
    store.save(f"rl_{key}", rows)


# ---- the table ----------------------------------------------------------------------

def test_table_size_and_names():
    assert 25 <= len(rl.SPECS) <= 40
    names = [s.name for s in rl.SPECS]
    assert len(set(names)) == len(names)
    assert all(n.startswith("rl_") for n in names)


def test_every_fixed_option_has_a_criterion_and_every_task_a_licence():
    for s in rl.SPECS:
        assert s.licence and s.url.startswith("http"), s.name
        if s.options is not None:
            assert s.criteria is not None and len(s.criteria) == len(s.options), s.name
            assert all(c and c.strip() for c in s.criteria), s.name
        else:
            assert s.criteria is None, s.name


def test_no_decision_index_member_is_a_source():
    for s in rl.SPECS:
        hay = f"{s.url} {s.key} {s.task}".lower()
        hits = [m for m in DI_MEMBERS if f"/{m}" in hay or hay.startswith(m) or f"_{m}" in hay
                or f"{m}_" in hay.split("/")[-1]]
        assert not hits, (s.name, hits)
    # SWAG is not HellaSwag, balanced COPA is not WinoGrande, WANLI is not ANLI
    assert {"rl_swag", "rl_copa", "rl_wanli"} <= set(rl.SPEC_BY_NAME)


def test_splits_whole_datasets_and_disjoint_families():
    fam_split = defaultdict(set)
    for s in rl.SPECS:
        assert s.split in ("train", "devreal", "testreal")
        fam_split[s.family].add(s.split)
    assert all(len(v) == 1 for v in fam_split.values()), fam_split
    by = defaultdict(set)
    for fam, sp in fam_split.items():
        by[next(iter(sp))].add(fam)
    assert not (by["train"] & by["devreal"]) and not (by["train"] & by["testreal"])
    assert not (by["devreal"] & by["testreal"])
    n = defaultdict(int)
    for s in rl.SPECS:
        n[s.split] += 1
    share = {k: v / len(rl.SPECS) for k, v in n.items()}
    assert 0.6 <= share["train"] <= 0.8
    assert 0.1 <= share["devreal"] <= 0.2 and 0.1 <= share["testreal"] <= 0.2


def test_tasks_carry_family_split_licence_and_model_label_note(raw):
    for s in rl.SPECS:
        _save(s.key, [{"x": 1}])
    _save("hwu64_intents", [{"id": 0, "name": "alarm_query"}])
    tasks = rl.tasks()
    assert len(tasks) == len(rl.SPECS)
    for t in tasks:
        s = rl.SPEC_BY_NAME[t.name]
        assert t.row == 50 and t.force_split == s.split and family_of(t) == s.family
        assert t.licence == s.licence and t.real
        assert ("data-generating model" in t.notes) == (s.label_source == "model")


def test_tasks_empty_without_fetched_rows(raw):
    assert rl.tasks() == []


# ---- readers ------------------------------------------------------------------------

def test_interp_keeps_the_published_mean():
    pts = (1, 2, 3, 4, 5)
    for v in (1.0, 2.3, 4.5, 5.0):
        t = rl.interp(v, pts)
        assert sum(t) == pytest.approx(1)
        assert sum(p * x for p, x in zip(pts, t)) == pytest.approx(v)
    assert rl.interp(7, pts) == [0, 0, 0, 0, 1]


def test_scruples_soft_target_is_the_vote_share():
    row = {"title": "AITA for x?", "text": "story",
           "label_scores": {"AUTHOR": 1, "OTHER": 6, "EVERYBODY": 0, "NOBODY": 3, "INFO": 0}}
    got = rl._read_scruples(row)
    assert got["target"] == pytest.approx([0.1, 0.6, 0, 0.3, 0]) and got["cls"] == 1
    assert rl._read_scruples(dict(row, label_scores={"AUTHOR": 2, "OTHER": 2})) is None


def test_choice_readers_label_conventions():
    p = rl._read_piqa({"goal": "g", "sol1": "a", "sol2": "b", "label": 1})
    assert p["options"] == ["a", "b"] and p["target"] == 1 and len(p["descriptions"]) == 2
    s = rl._read_siqa({"context": "c", "question": "q?", "answerA": "x", "answerB": "y",
                       "answerC": "z", "label": "3"})
    assert s["target"] == 2 and s["question"] == "q?"
    c = rl._read_copa({"premise": "p", "question": "cause", "choice1": "a",
                       "choice2": "b", "label": "1"})
    assert c["target"] == 1 and "CAUSE" in c["question"]
    assert rl._read_piqa({"goal": "g", "sol1": "a", "sol2": "A", "label": 0}) is None


def test_glaive_call_refusal_and_clarification():
    fn = {"name": "get_news", "description": "Get news",
          "parameters": {"properties": {"country": {}}, "required": ["country"]}}
    sys_ = "SYSTEM: You have functions -\n" + json.dumps(fn, indent=4) + "\n"
    call = {"system": sys_, "chat": 'USER: News for France?\n\n\nASSISTANT: <functioncall> '
            '{"name": "get_news", "arguments": \'{}\'} <|endoftext|>\n\n\n'}
    got = rl._read_glaive(call)
    assert got["options"][got["target"]] == "get_news"
    assert "country (required)" in got["descriptions"][got["target"]]
    ref = dict(call, chat="USER: Book me a flight\n\n\nASSISTANT: I'm sorry, I can't book "
                          "flights. <|endoftext|>\n\n\n")
    got = rl._read_glaive(ref)
    assert sentinel.is_sentinel_key(got["options"][got["target"]])
    assert sentinel.split_of(got["descriptions"][got["target"]]) == "train"
    ask = dict(call, chat="USER: News please\n\n\nASSISTANT: Which country? <|endoftext|>")
    assert rl._read_glaive(ask) is None


def test_toolace_single_call_and_eval_sentinel():
    fns = [{"name": "Get Weather", "description": "weather"},
           {"name": "Get Joke", "description": "a joke"}]
    row = {"system": "Here is a list of functions in JSON format that you can invoke:\n"
                     + json.dumps(fns),
           "conversations": [{"from": "user", "value": "Weather in Oslo?"},
                             {"from": "assistant", "value": '[Get Weather(city="Oslo")]'}]}
    got = rl._read_toolace(row)
    assert got["options"][got["target"]] == "Get Weather" and len(got["options"]) == 3
    two = dict(row, conversations=[row["conversations"][0], {
        "from": "assistant", "value": "[Get Weather(city=\"Oslo\"), Get Joke()]"}])
    assert rl._read_toolace(two) is None
    none = dict(row, conversations=[row["conversations"][0], {
        "from": "assistant", "value": "None of the functions can be used for this."}])
    got = rl._read_toolace(none)
    assert sentinel.split_of(got["descriptions"][got["target"]]) == "eval"


def test_slices_keep_two_tasks_of_one_pool_disjoint(raw):
    rows = [{"text": f"joke number {i}", "label": "humor" if i % 2 else "not humor"}
            for i in range(400)]
    _save("sk_is_humor", rows)
    _save("sk_humor_rating", [{"text": r["text"], "label": "2.0"} for r in rows])
    pools = rl.pools()
    a = {it["state"] for it in pools["rl_humor"]}
    b = {it["state"] for it in pools["rl_humor_rating"]}
    assert a and b and not (a & b) and len(a | b) == 400


def test_examples_have_criteria_on_half_and_label_source(raw):
    rows = [{"premise": f"p {i}", "hypothesis": f"h {i}",
             "gold": rl.NLI3[i % 3]} for i in range(300)]
    _save("wanli", rows)
    ex = list(rl.examples_for(rl.SPEC_BY_NAME["rl_wanli"], 200))
    assert len(ex) == 200
    described = [e for e in ex if e.questions[0].descriptions]
    assert 70 <= len(described) <= 130
    assert described[0].questions[0].descriptions == list(rl.NLI3_CRITERIA)
    assert all(e.questions[0].meta["label_source"] == "human" for e in ex)


def test_class_cap_on_a_large_pool(raw):
    rows = [{"text": f"message {i}", "label": "truth" if i % 10 else "lie"}
            for i in range(3000)]
    _save("sk_deception", rows)
    ex = list(rl.examples_for(rl.SPEC_BY_NAME["rl_deception"], 400))
    lies = sum(e.questions[0].target == 0 for e in ex)
    assert len(ex) == 400 and lies == 200          # capped at half of the sample


def test_blocked_states_are_skipped_and_counted(raw, monkeypatch):
    class Block:
        def blocked(self, text):
            return "blocked" in text
    monkeypatch.setattr(rl, "_BLOCK", Block())
    rows = [{"text": ("blocked " if i < 10 else "") + f"tweet {i}",
             "label": "complaint" if i % 2 else "not complaint"} for i in range(60)]
    _save("sk_complaints", rows)
    ex = list(rl.examples_for(rl.SPEC_BY_NAME["rl_complaints"], 100))
    assert len(ex) == 50 and not any("blocked" in e.state for e in ex)
    assert rl.DI_DROPS["rl_complaints"] == 10


# ---- live: the real store and the DI 0.2 blocklist -----------------------------------

def _live():
    if not (LIVE_ROOT / "v4-raw" / "rl_wanli.jsonl").exists():
        pytest.skip("raw store not fetched (scripts/fetch_data.py --rows 50)")
    if not rl.DI_BLOCKLIST.exists():
        pytest.skip("DI 0.2 blocklist absent")
    store.configure(LIVE_ROOT)


@pytest.mark.parametrize("name", ["rl_cosmosqa", "rl_toolace_tool", "rl_scitail",
                                  "rl_hwu64_intent", "rl_wanli", "rl_swag", "rl_glaive_tool",
                                  "rl_humor_rating"])
def test_live_loaders_return_questions_without_di_overlap(name, monkeypatch):
    _live()
    monkeypatch.setattr(rl, "DI_DROPS", defaultdict(int))
    from lod.corpus.services.decontaminate import Blocklist
    bl = Blocklist.load(rl.DI_BLOCKLIST)
    task = next(t for t in rl.tasks() if t.name == name)
    ex = list(task.load(150))
    assert len(ex) == 150
    for e in ex:
        q = e.questions[0]
        assert len(q.options) >= 2 and q.target is not None
        assert not bl.blocked(e.state)
        if q.descriptions:
            assert len(q.descriptions) == len(q.options) and all(q.descriptions)
    if rl.SPEC_BY_NAME[name].label_source == "model":
        assert all(e.questions[0].meta["label_source"] == "model" for e in ex)
