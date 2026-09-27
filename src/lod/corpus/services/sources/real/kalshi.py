"""Source-table row 26 — outcome markets with a forecast, grouped into families.

Row 26 names FiveThirtyEight's NBA/NFL/MLB/soccer/election forecasts. FiveThirtyEight was
wound down: its site now serves ABC News HTML and the GitHub raw paths 404. Kalshi is a
live substitute of the same shape -- a question, a settled outcome, and the market's own
price as the forecast -- so row 26's *intent* survives even though its named source does
not. `mirror_of` records that plainly.

One schema per market family rather than one for the whole exchange. This matters for more
than bookkeeping: the corpus caps a schema at 1,000 train rows, so a single `meta.p`
schema can contribute at most 1,000 questions, and the corpus wants 3 % of train to carry
`meta.p`.
Measured, Manifold alone gave 0.86 %. Row 26's own structure is five sports, i.e. five
schemas, and Kalshi's `event_ticker` prefix is exactly that grouping -- ATP matches,
economic releases, weather -- each with its own question wording. The question genuinely
differs per family, so these are distinct schemas rather than one schema counted five
times.

**No `meta.p` from this source.** The snapshot is taken after settlement, so every price
field has already collapsed to the outcome: measured over 3,998 settled markets, only
5.3 % of `last_price` and 5.8 % of `previous_price` sit anywhere between 10c and 90c, and
the bid/ask are at 0.0 %. A "forecast" that equals the answer is not a forecast, and
attaching it as `meta.p` would give a calibration probe a target it can read off the
label. Row 26 therefore contributes outcome *tasks* but not `meta.p`; row 28's Manifold
close probabilities, which are genuinely spread, carry that rule.

Markets that did not settle to a categorical yes/no are dropped: there is no answer to
score.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Iterator

from lod.schema import Example, Question
from lod.corpus.repositories import raw_store as store
from lod.corpus.services.sources.real.base import RealTask

KEY = "thomaswmitch_kalshi_prediction_markets_markets"
PATH = "thomaswmitch/kalshi-prediction-markets-markets"
LICENCE = "CC-BY-4.0 (Kalshi public market data)"
URL = f"https://huggingface.co/datasets/{PATH}"

MIN_PER_FAMILY = 40     # below this a family is not worth its own schema
# Domain-5 verification: the cached HF snapshot (10,012 settled, priced,
# categorical rows) has 26 families clearing MIN_PER_FAMILY, not 12 -- the cap was
# leaving 1,353 already-downloaded rows on the table across 14 extra families (KXBTC,
# KXHIGHCHI, KXHIGHLAX, KXHIGHAUS, KXTHEOPEN, KXFOMEN, KXUFCFIGHT, KXTRUMPMENTION,
# KXHIGHDEN, KXFOWOMEN, KXHIGHMIA, KXUSOPEN, KXHIGHPHIL, KXETHD, KXF). Raised to use all
# of them. This is a family-diversity and outcome-classification fix only -- it does
# nothing for `probability_recovery`, since this source deliberately never sets meta.p
# (see module docstring); "more Kalshi" cannot move the probability-recovery gate.
MAX_FAMILIES = 26
_IN_PROCESS: dict[str, object] = {}


def _family(ticker: str) -> str:
    """The market family: the event ticker's leading alphabetic prefix."""
    m = re.match(r"^([A-Z]+?)(?=[0-9]|-|$)", str(ticker or ""))
    return m.group(1) if m else "OTHER"


def _rows() -> list[dict]:
    if "rows" in _IN_PROCESS:
        return _IN_PROCESS["rows"]           # type: ignore[return-value]
    try:
        raw = store.load(KEY)
    except FileNotFoundError:
        root = store.RAW_ROOT / "huggingface" / PATH / "data"
        files = sorted(root.glob("*.parquet")) if root.exists() else []
        if not files:
            _IN_PROCESS["rows"] = []
            return []
        from datasets import load_dataset
        raw = list(load_dataset("parquet", data_files=[str(f) for f in files], split="train"))
    out = []
    for r in raw:
        result = str(r.get("result") or "").strip().lower()
        if result not in ("yes", "no"):
            continue                          # never settled categorically
        title = str(r.get("title") or "").strip()
        price = r.get("last_price")
        if not title or price is None:
            continue
        try:
            p = float(price) / 100.0
        except (TypeError, ValueError):
            continue
        if not 0.0 <= p <= 1.0:
            continue
        rules = str(r.get("rules_primary") or "")[:1500]
        out.append({"family": _family(r.get("event_ticker")), "title": title,
                    "rules": rules, "result": result, "p": p})
    _IN_PROCESS["rows"] = out
    return out


def families(rows: list[dict] | None = None) -> list[str]:
    rows = rows if rows is not None else _rows()
    counts = Counter(r["family"] for r in rows)
    return [f for f, n in counts.most_common(MAX_FAMILIES) if n >= MIN_PER_FAMILY]


def _loader(family: str):
    def load(n: int) -> Iterator[Example]:
        made = 0
        for r in _rows():
            if made >= n:
                return
            if r["family"] != family:
                continue
            made += 1
            state = r["title"] + ("\n\n" + r["rules"] if r["rules"] else "")
            yield Example(
                task=f"kalshi_{family.lower()}",
                state=state[:4000],
                questions=[Question(
                    id="resolves_yes",
                    question=f"Does this {family} market settle YES?",
                    options=["no", "yes"],
                    target=1 if r["result"] == "yes" else 0,
                )],
            )
    return load


def tasks() -> list[RealTask]:
    """None. Row 26's Kalshi families are dropped from the corpus.

    The target is the settled outcome of a future event -- a baseball game, the S&P 500 at
    4pm, Chicago's high temperature -- and the state is the market's title and rules,
    written before the event. Nothing in the state determines the answer, and the only
    forecast this snapshot carries is a post-settlement price that equals the outcome
    99.2 % of the time, so it cannot be attached either. Measured by a
    domain-5 audit: the best any reader can do is each family's base rate, 0.568
    over 10,012 rows (e.g. MLB game winners 0.500, ATP matches 0.501). A label code cannot
    derive from the state is dropped rather than guessed, and a coin-flip target trains
    nothing but the marginal. `family_tasks()` keeps the old builder for measurement.
    """
    return []


def family_tasks() -> list[RealTask]:
    if not store.has(KEY) and not (store.RAW_ROOT / "huggingface" / PATH).exists():
        return []
    rows = _rows()
    return [RealTask(row=26, name=f"kalshi_{f.lower()}", licence=LICENCE, url=URL,
                     load=_loader(f), meta_p=False,
                     notes=f"Kalshi {f} settled markets; no meta.p -- prices in this "
                           f"snapshot are post-settlement; mirror of FiveThirtyEight, "
                           f"which no longer exists")
            for f in families(rows)]
