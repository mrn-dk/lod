"""Where Lod keeps its data, runs and caches, and how a checkpoint's config file is named.

Every location is an environment variable with a repo-relative default, so a checkout
works as-is and a cluster points the same code at shared storage:

    LOD_ROOT       data/ and runs/ live here                 (default: current directory)
    LOD_RAW_ROOT   raw rows fetched from the sources          (default: ~/.cache/lod-sources)
    LOD_DI_SUITE   a rebuilt Decision Index suite, for the contamination blocklist
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("LOD_ROOT", ".")).expanduser()
DATA = ROOT / "data"
RUNS = ROOT / "runs"
RAW_ROOT = Path(os.environ.get("LOD_RAW_ROOT", "~/.cache/lod-sources")).expanduser()
DI_SUITE = Path(os.environ.get("LOD_DI_SUITE", "decision-index/suite-0.2")).expanduser()

# frozen banks and side tables that ship inside the package
ASSETS = Path(__file__).resolve().parent / "assets"

# the Decision Index contamination blocklist (`scripts/build_corpus.py blocklist`)
DI_BLOCKLIST = DATA / "di_blocklist_v02.npz"

# A checkpoint directory's config file. Checkpoints written before the project took its
# public name carry the same file as `bouncy.json`; `config_path` reads either.
CONFIG_NAME = "lod.json"
LEGACY_CONFIG_NAME = "bouncy.json"


def config_path(ckpt: str | Path) -> Path:
    """`<ckpt>/lod.json` if present, else the legacy file if present, else `lod.json`."""
    ckpt = Path(ckpt)
    for name in (CONFIG_NAME, LEGACY_CONFIG_NAME):
        if (ckpt / name).exists():
            return ckpt / name
    return ckpt / CONFIG_NAME
