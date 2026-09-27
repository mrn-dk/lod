"""The fetch stage: every source-table row is accounted for, every raw-store key a reader
reads is written by some fetcher, and planning a fetch touches nothing. No network."""

from __future__ import annotations

import pytest

from lod.corpus.controllers import fetch as ctl
from lod.corpus.repositories import raw_store as store
from lod.corpus.repositories import sentiment as sentiment_repo
from lod.corpus.repositories import toolspecs


def test_every_table_row_has_a_fetcher_or_a_reason():
    cov = ctl.coverage()
    assert set(cov) == set(range(1, 52))
    assert not [row for row, what in cov.items() if not what]


def test_rows_without_tasks_are_not_fetched():
    fetched = {r for f in ctl.FETCHERS if not f.opt_in for r in f.rows}
    assert not (set(ctl.NO_TASKS) | set(ctl.BUILD_TIME)) & fetched


def _claimed_keys(tmp_path, monkeypatch) -> set[str]:
    monkeypatch.setattr(store, "RAW_ROOT", tmp_path)
    monkeypatch.setattr(store, "RAW", tmp_path / "store")
    o = ctl.Options()
    return {p.stem for f in ctl.FETCHERS for p in f.outputs(o)
            if p.parent == tmp_path / "store" and p.suffix == ".jsonl"}


def test_every_key_a_reader_loads_is_fetched(tmp_path, monkeypatch):
    from lod.corpus.services.sources.real import (
        grounding, kalshi, lichess, loghub, multilingual, reallang, sentiment, tabfact, topic)
    from lod.corpus.services.sources.synth import chessfacts

    claimed = _claimed_keys(tmp_path, monkeypatch)
    read = {lichess.KEY, kalshi.KEY, grounding.FEVER_KEY, grounding.VITC_KEY, tabfact.KEY,
            loghub.KEY, chessfacts.PUZZLE_KEY, chessfacts.POOL_KEY, "rl_hwu64_intents"}
    read |= {s.key for s in sentiment.SPECS}
    read |= {s.key for s in topic.SOURCES}
    read |= {s.store_key for s in multilingual.SOURCES}
    read |= {reallang.PREFIX + s.key for s in reallang.SPECS}
    assert read - claimed == set()


def test_opt_in_fetchers_run_only_when_named():
    default = {f.name for f in ctl.select()}
    assert "toolgate_http" not in default and "nvd_feeds" not in default
    assert [f.name for f in ctl.select(only={"toolgate_http"})] == ["toolgate_http"]
    with pytest.raises(ValueError):
        ctl.select(only={"no_such_fetcher"})


def test_rows_select_their_fetchers():
    names = {f.name for f in ctl.select(rows={44, 45})}
    assert names == {"sentiment", "topic"}


def test_dry_run_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(ctl, "CACHE", tmp_path)
    monkeypatch.setattr(store, "RAW_ROOT", store.RAW_ROOT)
    monkeypatch.setattr(store, "RAW", store.RAW)
    results = ctl.fetch(raw_root=tmp_path, dry_run=True, row1_limit=2)
    assert results and all(r.status in ("planned", "skipped") for r in results)
    assert list(tmp_path.iterdir()) == []


def test_a_second_raw_root_is_refused(tmp_path):
    with pytest.raises(ValueError, match="LOD_RAW_ROOT"):
        ctl.fetch(raw_root=tmp_path / "elsewhere", dry_run=True)


def test_sst_key_undoes_entities_brackets_and_case():
    assert sentiment_repo.sst_norm("-LRB- Tom &amp; Jerry -RRB-  Rocks") == "( tom & jerry ) rocks"


def test_toolspecs_derives_effect_from_method_and_path():
    ops = {("POST", "/pets"), ("DELETE", "/pets/{id}"), ("POST", "/orders")}
    assert toolspecs.derive("POST", "/pets", ops)["restorable"] == "undo_command"
    assert toolspecs.derive("POST", "/orders", ops)["restorable"] == "backup"
    assert toolspecs.derive("DELETE", "/pets/{id}", ops)["blast"] == "one_item"
    assert toolspecs.derive("GET", "/{tenant}/pets", ops) is None
    assert toolspecs.blast_of("/stores/{s}/items") == "one_collection"
