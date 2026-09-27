"""Row 51, real half: long licensed documents with the dataset's own labels.

What has to stay true: every state is a whole document of 2,304-29,000 Qwen3 tokens (never
truncated), every label is the one the raw file carries, held-out families never share a
document with training, and nothing the Decision Index v0.2 blocklist knows gets through.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict

import pytest

from lod.corpus.services.sources.real import longdocs as L
from lod.corpus.services.sources.real import quality
from lod.corpus.services.sources.real.base import family_of

pytestmark = pytest.mark.skipif(
    not (L.data_dir() / "CUADv1.json").exists(),
    reason="longdocs raw cache absent; run scripts/fetch_data.py --rows 51")


@pytest.fixture(scope="module")
def tasks():
    return L.tasks()


@pytest.fixture(scope="module")
def built(tasks):
    return {t.name: t.load(10**6) for t in tasks}


def test_all_tasks_present_and_licensed(tasks):
    assert len(tasks) == 9
    for t in tasks:
        assert t.row == 51 and t.real and t.name.startswith("longdoc_")
        assert t.licence and t.url and t.force_split in ("train", "devreal", "testreal")
    assert {t.force_split for t in tasks} == {"train", "devreal", "testreal"}


def test_deterministic(tasks):
    for t in tasks:
        a = [e.to_dict() for e in t.load(15)]
        L._BUILT.clear()
        b = [e.to_dict() for e in t.load(15)]
        assert a == b, t.name


def test_volume(built):
    nq = {k: sum(len(e.questions) for e in v) for k, v in built.items()}
    assert all(n >= 50 for n in nq.values()), nq
    assert 3000 <= sum(nq.values()) <= 12000, nq


def test_lengths_in_range_and_recorded(built):
    tok = L.tokenizer()
    rng = random.Random(0)
    for name, exs in built.items():
        for e in exs:
            for q in e.questions:
                g = q.meta["gen"]
                assert L.MIN_TOKENS <= g["state_tokens"] <= L.MAX_TOKENS
                assert g["family"] == name and g["depth_tokens"] is None
        for e in rng.sample(exs, min(5, len(exs))):
            n = len(tok(e.state, add_special_tokens=False)["input_ids"])
            assert n == e.questions[0].meta["gen"]["state_tokens"], name


def test_labels_rederived_from_raw(built):
    gold = {}
    for sp in ("train", "dev"):
        with (L.data_dir() / f"QuALITY.v1.0.1.htmlstripped.{sp}").open() as f:
            for line in f:
                for q in json.loads(line)["questions"]:
                    gold[q["question_unique_id"]] = (q["gold_label"] - 1, q["options"])
    cuad = {c["title"]: c["cats"] for c in L._cuad_contracts()}
    ecthr = {sp: [json.loads(x) for x in (L.data_dir() / f"ecthr_{sp}.jsonl").open()]
             for sp in ("train", "validation", "test")}
    n = 0
    for name, exs in built.items():
        for e in exs:
            for q in e.questions:
                n += 1
                if "quality" in name:
                    g, opts = gold[q.id.removeprefix("longdoc_quality_")]
                    assert q.target == g and q.options == [o.strip() for o in opts]
                elif "cuad" in name:
                    assert q.target == int(cuad[q.meta["contract"]][q.meta["category"]][0])
                    assert L._cuad_split(q.meta["contract"]) == L.SPECS[[s.name for s in L.SPECS].index(name)].split
                else:
                    _, sp, i, art = q.id.rsplit("_", 3)
                    r = ecthr[sp][int(i)]
                    assert art in r["alleged"]
                    assert q.target == int(art in r["violated"])
                    assert "\n".join(p.strip() for p in r["text"]).strip() == e.state
    assert n > 3000


def test_binary_tasks_balanced(built):
    for name, exs in built.items():
        if "quality" in name:
            continue
        c = Counter(q.target for e in exs for q in e.questions)
        assert c[0] == c[1], (name, c)


def test_splits_and_families_disjoint(tasks, built):
    by_split = defaultdict(set)
    for t in tasks:
        by_split[t.force_split].add(family_of(t))
    assert not (by_split["train"] & by_split["devreal"])
    assert not (by_split["train"] & by_split["testreal"])
    assert not (by_split["devreal"] & by_split["testreal"])
    owner = {}
    for name, exs in built.items():
        for e in exs:
            h = hashlib.sha1(e.state.encode()).hexdigest()
            assert owner.setdefault(h, name) == name, f"state shared by {owner[h]} and {name}"
    # held-out categories/articles never trained
    cats = defaultdict(set)
    for name, exs in built.items():
        key = "category" if "cuad" in name else "article" if "ecthr" in name else None
        if key:
            cats[name] = {q.meta[key] for e in exs for q in e.questions}
    assert not (cats["longdoc_cuad_train"] & (cats["longdoc_cuad_dev"] | cats["longdoc_cuad_test"]))
    assert not (cats["longdoc_ecthr_train"] & (cats["longdoc_ecthr_dev"] | cats["longdoc_ecthr_test"]))


def test_di_overlap_zero(built):
    bl = L.blocklist()
    if bl is None:
        pytest.skip("no DI v0.2 blocklist on this machine")
    rng = random.Random(1)
    for name, exs in built.items():
        sample = rng.sample(exs, min(12, len(exs)))
        kept, dropped = bl.filter(sample)
        assert dropped == 0, name
        for e in sample:
            for q in e.questions:
                assert not bl.blocked(q.question + "\n" + "\n".join(q.options))


def test_passes_answerability_gate(built):
    for name, exs in built.items():
        kept, reason = quality.filter_examples(exs, name, described=False)
        assert reason is None, (name, reason)
