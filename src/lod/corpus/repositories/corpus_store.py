"""A built corpus on disk: one JSONL file per split, plus `meta.json` and side reports.

    <corpus>/train.jsonl  val.jsonl  devreal.jsonl  testreal.jsonl     the splits
    <corpus>/devreal_sel.jsonl  devood.jsonl  devood_sel.jsonl          selection sets
    <corpus>/meta.json                                                  per-task build record

Every stage of the build reads one corpus directory and writes another (or rewrites one in
place), so this is the only code that knows the layout. A split can be read three ways,
and each stage picks the one that keeps what it must keep:

- `read`         typed `Example`s, for logic that works on questions and targets;
- `read_records` plain dicts, for stages that must pass every field through untouched;
- `read_lines`   the raw lines, for stages that only drop or reorder records -- a line
                 is written back byte for byte, so nothing is re-serialised.

Writes go to a temporary file that replaces the target only once it is complete, so an
interrupted stage never leaves a half file that later reads as a whole one, and a stage
may rewrite the corpus it is reading.
"""

from __future__ import annotations

import json
import os
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, TextIO

from lod.corpus.domains import SPLITS
from lod.schema import Example

META = "meta.json"


class CorpusStore:
    """One corpus directory. A "split" is any `<name>.jsonl` in it."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def __repr__(self) -> str:
        return f"CorpusStore({str(self.root)!r})"

    def path(self, split: str) -> Path:
        return self.root / f"{split}.jsonl"

    def has(self, split: str) -> bool:
        return self.path(split).exists()

    def present(self, splits: Iterable[str] = SPLITS) -> list[str]:
        return [s for s in splits if self.has(s)]

    # ---- reading -------------------------------------------------------------------------

    def iter_lines(self, split: str) -> Iterator[str]:
        """Non-blank lines, each ending in exactly one newline."""
        with open(self.path(split), encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield line if line.endswith("\n") else line + "\n"

    def read_lines(self, split: str) -> list[str]:
        return list(self.iter_lines(split))

    def iter_records(self, split: str) -> Iterator[dict]:
        for line in self.iter_lines(split):
            yield json.loads(line)

    def read_records(self, split: str) -> list[dict]:
        return list(self.iter_records(split))

    def read(self, split: str) -> list[Example]:
        return [Example.from_dict(r) for r in self.iter_records(split)]

    def read_meta(self) -> dict:
        return json.loads((self.root / META).read_text())

    def read_json(self, name: str, default: Any = None) -> Any:
        p = self.root / name
        return json.loads(p.read_text()) if p.exists() else default

    # ---- writing -------------------------------------------------------------------------

    @contextmanager
    def writer(self, name: str) -> Iterator[TextIO]:
        """An open text file for `name` (a split, or a file name with a suffix) that
        replaces the target only when the block exits cleanly."""
        p = self.path(name) if "." not in name else self.root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                yield f
            tmp.replace(p)
        finally:
            tmp.unlink(missing_ok=True)

    def write_lines(self, split: str, lines: Iterable[str]) -> int:
        n = 0
        with self.writer(split) as f:
            for line in lines:
                f.write(line)
                n += 1
        return n

    def write_records(self, split: str, records: Iterable[dict]) -> int:
        return self.write_lines(
            split, (json.dumps(r, ensure_ascii=False) + "\n" for r in records))

    def write(self, split: str, examples: Iterable[Example]) -> int:
        """Typed examples, serialised exactly as `lod.schema.write_jsonl` does."""
        return self.write_records(split, (e.to_dict() for e in examples))

    def write_meta(self, meta: dict) -> None:
        with self.writer(META) as f:
            f.write(json.dumps(meta, indent=2))

    def write_json(self, name: str, obj: Any, **kw) -> Path:
        with self.writer(name) as f:
            f.write(json.dumps(obj, indent=2, **kw))
        return self.root / name

    def write_text(self, name: str, text: str) -> Path:
        with self.writer(name) as f:
            f.write(text)
        return self.root / name

    # ---- between corpora -----------------------------------------------------------------

    def copy_from(self, src: "CorpusStore", name: str) -> bool:
        """Copy one file of `src` verbatim, if it exists there."""
        p = src.root / name
        if not p.exists():
            return False
        self.root.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, self.root / name)
        return True

    def link_from(self, src: "CorpusStore", skip: Iterable[str] = ()) -> list[str]:
        """Symlink every file of `src` not named in `skip` and not already here: the
        splits a stage leaves unchanged are shared, not copied."""
        self.root.mkdir(parents=True, exist_ok=True)
        skip = set(skip)
        linked = []
        for f in sorted(src.root.iterdir()):
            if f.name in skip or (self.root / f.name).exists():
                continue
            os.symlink(f.resolve(), self.root / f.name)
            linked.append(f.name)
        return linked
