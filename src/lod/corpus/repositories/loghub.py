"""LogHub's 2,000-line samples, one file per system, with LogHub's own LICENSE beside them.

LogHub publishes a 2k-line sample of each of its 16 systems at a stable raw GitHub path.
Sixteen line formats rather than one is the point: the parser has to read a line, not
match a timestamp, and a whole family of systems can be held out so a score measures
transfer across log dialect. The families are the reader's (`real/loghub.FAMILIES`), so
the fetch and the holdout cannot drift apart.

LogHub is research data with an attribution-and-citation condition, not OSI-licensed
software, so its LICENSE is stored verbatim next to the logs it covers.

Layout: `<raw root>/loghub/<System>_2k.log`, `LICENSE`, `manifest.json`.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from lod.corpus.repositories import web
from lod.corpus.services.sources.real.loghub import FAMILIES, FAMILY_OF, cache_dir

BASE = "https://raw.githubusercontent.com/logpai/loghub/master"
SYSTEMS: tuple[str, ...] = tuple(s for fam in FAMILIES.values() for s in fam)


def root() -> Path:
    return cache_dir()


def outputs() -> list[Path]:
    return [root() / "LICENSE"] + [root() / f"{s}_2k.log" for s in SYSTEMS]


def fetch_system(system: str, refresh: bool = False) -> dict | None:
    """Download one `<System>_2k.log` -> its manifest record, or None if it failed."""
    url = f"{BASE}/{system}/{system}_2k.log"
    try:
        dest = web.download(url, root() / f"{system}_2k.log", refresh=refresh,
                            min_bytes=1000, timeout=120)
    except Exception as e:  # noqa: BLE001 -- one missing system is not fatal
        print(f"  skip {system}: {type(e).__name__} {e}", flush=True)
        return None
    body = dest.read_bytes()
    lines = [l for l in body.decode("utf-8", "replace").split("\n") if l.strip()]
    return {"system": system, "family": FAMILY_OF[system], "path": dest.name, "url": url,
            "lines": len(lines), "bytes": len(body),
            "sha256": hashlib.sha256(body).hexdigest()}


def fetch(systems: tuple[str, ...] = SYSTEMS, refresh: bool = False,
          delay: float = 0.5) -> int:
    unknown = [s for s in systems if s not in FAMILY_OF]
    if unknown:
        raise ValueError(f"unknown LogHub systems {unknown}; known: {list(SYSTEMS)}")
    d = root()
    web.download(f"{BASE}/LICENSE", d / "LICENSE", refresh=refresh, min_bytes=100)

    manifest = d / "manifest.json"
    records: dict[str, dict] = {}
    if manifest.exists():
        try:
            records = {r["system"]: r for r in json.loads(manifest.read_text())["systems"]}
        except (json.JSONDecodeError, KeyError):
            records = {}
    for system in systems:
        rec = fetch_system(system, refresh)
        if rec is None:
            continue
        records[system] = rec
        print(f"  {system:<13} {rec['lines']:>5} lines  {rec['bytes'] / 1000:>7.1f} kB "
              f"[{rec['family']}]", flush=True)
        time.sleep(delay)
    manifest.write_text(json.dumps(
        {"source": "https://github.com/logpai/loghub",
         "licence": "Loghub research-use (attribution + ISSRE'23 citation)",
         "licence_file": "LICENSE",
         "systems": [records[s] for s in SYSTEMS if s in records]},
        indent=1))
    return len(records)
