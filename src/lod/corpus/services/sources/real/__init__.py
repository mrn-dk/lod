"""The real-task sources.

One module per source-table row (or row group). Each exposes `tasks() -> list[RealTask]`.
`REGISTRY()` is a function rather than a constant so that importing this package does not
touch the network: several loaders resolve option sets from the data itself, and a module
that fetched at import time would make `pytest` and `--help` hit the internet.
"""

from __future__ import annotations

from lod.corpus.services.sources.real import (
    chaosnli,
    clinical,
    crowd,
    gefs,
    gharchive,
    grounding,
    instructions,
    kalshi,
    lichess,
    loghub,
    manifold,
    nvd,
    pdfdocs,
    spdx,
    spf,
    tabfact,
    unfair_tos,
)
from lod.corpus.services.sources.real.base import (
    Accounting,
    RealTask,
    dedup_against_train,
    shape_of,
    split_of,
)

# row -> module providing it. Rows land here as they are implemented; `fetch_data.py --coverage`
# reports how many of the 32 are covered, so a missing row is visible rather than assumed.
MODULES = (chaosnli, clinical, crowd, gefs, gharchive, grounding, instructions,
           kalshi, lichess, loghub, nvd, pdfdocs, spdx, spf, manifold, tabfact,
           unfair_tos)
# row 44: English sentiment, emotion and stance (domain 3)
from lod.corpus.services.sources.real import sentiment  # noqa: E402
MODULES += (sentiment,)

# row 46, multilingual classification. Appended rather than written into the
# tuple above so concurrent additions to it do not collide.
from lod.corpus.services.sources.real import multilingual  # noqa: E402
MODULES = MODULES + (multilingual,)

# row 45: English topic, intent and question-type classification (domain 1)
from lod.corpus.services.sources.real import topic  # noqa: E402
MODULES = MODULES + (topic,)

# row 50 (release round): broad real-language decisions (domain 24); row 51's real long
# documents live in the same package (domain 25)
from lod.corpus.services.sources.real import reallang, longdocs  # noqa: E402
MODULES = MODULES + (reallang, longdocs)


def REGISTRY(include_hf_table: bool = True, include_bigbench: bool = True,
             rows: set[int] | None = None, include_synth: bool = True) -> list[RealTask]:
    out: list[RealTask] = []
    for m in MODULES:
        out += m.tasks()
    if include_synth:
        # Rows 33 and 35: code-labelled rule application and randomised-attribute
        # entity facts. `real=False`, so the build reports their share separately.
        from lod.corpus.services.sources.synth import MODULES as SYNTH
        for m in SYNTH():
            out += m.tasks()
    # bigbench is cheap to enumerate (config names only) and is the corpus's main
    # source of per-example option sets, so it is always included
    if include_bigbench:
        from lod.corpus.services.sources.real import instructions as _ins
        out += _ins.bigbench_tasks()
    if include_hf_table:
        from lod.corpus.services.sources.real import hf as _hf
        from lod.corpus.services.sources.real.hub_sweep import specs_from_survey as _sweep
        from lod.corpus.services.sources.real.table import SPECS as _SPECS
        # row 1 is a query rather than a list, so its specs come from the saved survey;
        # without this the build silently omits the largest single source
        specs = [spec for spec in list(_SPECS) + _sweep()
             if rows is None or spec.row in rows]
        for spec in specs:
            try:
                out += _hf.tasks_for(spec)
            except FileNotFoundError:
                pass          # not fetched yet; scripts/fetch_data.py is stage 1
            except Exception as e:
                print(f"  registry: row {spec.row} {spec.key} unavailable "
                      f"({type(e).__name__}: {str(e)[:60]})")
    names = [t.name for t in out]
    dupes = {n for n in names if names.count(n) > 1}
    if dupes:
        raise ValueError(f"duplicate task names would collide in the split hash: {dupes}")
    return out


__all__ = ["REGISTRY", "RealTask", "Accounting", "split_of", "shape_of",
           "dedup_against_train", "MODULES"]
