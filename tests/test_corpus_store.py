"""The corpus directory: records round-trip, lines stay verbatim, stages rewrite in place."""
import json

from lod.corpus.controllers import build
from lod.corpus.repositories.corpus_store import CorpusStore
from lod.corpus.services.decontaminate import build_blocklist
from lod.schema import Example, Question

ITEM = ("Which of the following best describes the primary function of the enzyme "
        "helicase during the replication of a double stranded DNA molecule in cells")


def _ex(state, task="t"):
    return Example(state=state, task=task, questions=[Question("q", "?", ["a", "b"], 0)])


def test_typed_write_matches_write_jsonl_and_reads_back(tmp_path):
    from lod.schema import write_jsonl

    store = CorpusStore(tmp_path / "c")
    exs = [_ex("café   line"), _ex("two")]
    assert store.write("train", exs) == 2
    write_jsonl(tmp_path / "ref.jsonl", exs)
    assert store.path("train").read_bytes() == (tmp_path / "ref.jsonl").read_bytes()
    assert [e.state for e in store.read("train")] == [e.state for e in exs]
    assert store.present() == ["train"]


def test_lines_are_written_back_verbatim(tmp_path):
    store = CorpusStore(tmp_path)
    raw = '{"task": "t", "state": "x",  "questions": []}\n'     # odd spacing survives
    store.path("devreal").write_text(raw + "\n" + raw.rstrip("\n"))
    lines = store.read_lines("devreal")
    assert lines == [raw, raw]
    store.write_lines("devood", lines)
    assert store.path("devood").read_text() == raw + raw


def test_decontaminate_in_place_and_link_from(tmp_path):
    bl, _ = build_blocklist([("ARC", ITEM)])
    bl_path = bl.save(tmp_path / "bl.npz")
    src = CorpusStore(tmp_path / "c")
    src.write("train", [_ex("Question:\n" + ITEM), _ex("nothing to see")])
    src.write("devreal", [_ex("fine")])
    src.write_meta({"tasks": []})
    dropped = build.decontaminate(src.root, blocklist=bl_path, log=lambda _: None)
    assert dropped == {"train": 1, "devreal": 0}
    assert [e.state for e in src.read("train")] == ["nothing to see"]
    dst = CorpusStore(tmp_path / "d")
    assert dst.link_from(src, skip={"train.jsonl"}) == ["devreal.jsonl", "meta.json"]
    assert (dst.root / "devreal.jsonl").is_symlink()
    assert json.loads((dst.root / "meta.json").read_text()) == {"tasks": []}
