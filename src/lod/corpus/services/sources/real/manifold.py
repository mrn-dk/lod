"""Source-table row 28 — resolved binary markets from Manifold.

An *outcome* task: the label is what actually happened, and `meta.p` carries the market's
probability at close. That pairing is the point. It is the only shape in the corpus where
a question has both a realised answer and an independent forecast of it, which is what
lets a probe ask whether the model's confidence tracks a real forecaster's rather than
merely being well-ordered.

Two things are deliberate here:

`sort` is *not* most-popular. Sorted that way the resolved set is dominated by markets
that closed at 0.01 or 0.99 -- "Will Biden be the 2024 Democratic Nominee?" resolved NO at
0.01 -- and a corpus of near-certainties teaches nothing about calibration in the middle
of the range, which is exactly where calibration is hard. Pages are drawn across several
sorts and the resulting spread is reported by `probability_spread`.

CANCEL and MKT resolutions are dropped. A cancelled market has no ground truth, and a
market resolved to a partial probability has no *categorical* answer, so neither can carry
a target for a question whose options are no/yes.

**`meta.p` is the pool price at close, and only for markets that closed before they
resolved** (`usable`). An earlier version used the API's `resolutionProbability`, which is
the price the resolver recorded *when resolving*. Measured on the 2,842 cached markets
(domain-5 audit): 1,254 (44 %) have `resolutionTime` within a minute of
`closeTime` -- Manifold closes a market when it is resolved early, i.e. once the answer is
known -- and those carry a Brier score of 0.035 against the outcome with 64 % of prices at
<=0.02 or >=0.98: a price traders set after the fact, not a forecast. 407 carried a
`resolutionProbability` of exactly 0 or 1, which is the resolution itself typed in as a
probability. Markets that closed on schedule and resolved later score 0.104 -- a real
forecast. The pool price (`probability`) is used rather than `resolutionProbability`
because the latter is the resolver's rounded entry (76 exact 0/1 even among closed
markets; one market resolved YES at a pool price of 0.99 carries 0.09).

**ForecastBench is anti-joined.** The Decision Index benchmark scores ForecastBench, whose
question sets include Manifold markets; any market whose id or question text appears there
is removed (`FORECASTBENCH`, extracted by `build_forecastbench_exclusions`): 93 of 2,842.

**Licence.** Manifold's terms allow its data for AI training only non-commercially;
`LICENCE` says so, and it is carried into every task's metadata.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterator

from lod.schema import Example, Question
from lod.corpus.services.sources.base import CACHE, cache_path, fetch_json
from lod.corpus.services.sources.real.base import RealTask
from lod.paths import ASSETS, DATA

SEARCH = "https://api.manifold.markets/v0/search-markets"
MARKET = "https://api.manifold.markets/v0/market"
LICENCE = ("Manifold Markets API data; non-commercial use only for AI training "
           "(Manifold terms) -- not for commercial model training")
URL = "https://docs.manifold.markets/api"

# several sorts, so the probability distribution is not all 0.01 and 0.99
SORTS = ("newest", "most-popular", "liquidity", "close-date")
PAGE = 100
_CACHE_FILE = CACHE / "manifold_resolved.jsonl"


def download(limit: int | None = None, with_description: bool = True,
             sleep: float = 0.3, refresh: bool = False) -> list[dict]:
    """Resolved binary markets, cached to disk.

    The description needs a second request per market (the search endpoint omits it), so
    the whole set is cached: a corpus rebuild must not re-fetch thousands of markets.
    """
    # the cache is authoritative unless refreshed; see the same note in nvd.download
    if _CACHE_FILE.exists() and not refresh:
        rows = [json.loads(l) for l in _CACHE_FILE.read_text().splitlines() if l.strip()]
        if rows:
            return rows if limit is None else rows[:limit]

    target = limit if limit is not None else 4000
    CACHE.mkdir(parents=True, exist_ok=True)
    seen: dict[str, dict] = {}
    for sort in SORTS:
        offset = 0
        while len(seen) < target:
            url = (f"{SEARCH}?term=&filter=resolved&contractType=BINARY"
                   f"&limit={PAGE}&offset={offset}&sort={sort}")
            try:
                page = fetch_json(url, timeout=90, retries=2, cache=False)
            except Exception:
                break
            if not page:
                break
            for m in page:
                if m.get("id") in seen:
                    continue
                if m.get("resolution") not in ("YES", "NO"):
                    continue
                p = m.get("resolutionProbability")
                if p is None:
                    continue
                groups = m.get("groupSlugs") or []
                seen[m["id"]] = {"id": m["id"], "question": m["question"],
                                 "resolution": m["resolution"], "p": float(p),
                                 "group": str(groups[0]) if groups else "general",
                                 "description": ""}
            offset += PAGE
            time.sleep(sleep)

    rows = list(seen.values())[:target]
    if with_description:
        for r in rows:
            try:
                full = fetch_json(f"{MARKET}/{r['id']}", timeout=60, retries=1)
                r["description"] = (full.get("textDescription") or "")[:2000]
                # groupSlugs is only on the full market, not the search result -- reading
                # it from the search response left every market in one "general" group and
                # so left row 28 as a single schema, which the 1,000-rows-per-schema cap
                # makes structurally unable to reach A6's 3 % meta.p
                groups = full.get("groupSlugs") or []
                if groups:
                    r["group"] = str(groups[0])
                r.update(_timing(full))
            except Exception:
                r["description"] = ""
            time.sleep(sleep)

    _CACHE_FILE.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return rows


# The fields that say *when* the price was read relative to the outcome. `download` keeps
# them on new fetches; for a cache written before that, `full_record` reads them
# from the per-market JSON that `fetch_json` already cached beside it (all 2,842 are there),
# so nothing is re-downloaded.
def _timing(full: dict) -> dict:
    return {"close_time": full.get("closeTime"), "resolution_time": full.get("resolutionTime"),
            "last_bet_time": full.get("lastBetTime"), "p_close": full.get("probability"),
            "slug": full.get("slug"), "created_time": full.get("createdTime")}


def full_record(market_id: str) -> dict | None:
    """The market's full API record, from the local fetch cache only."""
    path = cache_path(f"{MARKET}/{market_id}")
    if not path.exists():
        return None
    try:
        return json.loads(path.read_bytes())
    except (OSError, ValueError):
        return None


FORECASTBENCH = ASSETS / "forecastbench_manifold.json"
# a clone of github.com/forecastingresearch/forecastbench-datasets, only to rebuild FORECASTBENCH
FORECASTBENCH_REPO = DATA / "forecastbench-datasets"


def _norm(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


def _text_hash(text: str) -> str:
    """A normalised question's hash: the exclusion list matches texts without shipping them."""
    return hashlib.sha256(_norm(text).encode("utf-8")).hexdigest()[:20]


def build_forecastbench_exclusions(repo: Path = FORECASTBENCH_REPO) -> dict:
    """Every Manifold market id and question text in ForecastBench's question and
    resolution sets, written to `FORECASTBENCH` (in the repo, so a build needs no clone).
    Question texts are stored as hashes of their normalised form, not as text.

    Question text is matched as well as id: one cached market ("Will the Federal Reserve
    hike interest rates in 2026?", D385361f18) is a different market from ForecastBench's
    NcuQEz998g on the same event, and would carry the same answer.
    """
    ids: set[str] = set()
    texts: set[str] = set()
    files = sorted(repo.glob("datasets/question_sets/*.json")) + \
        sorted(repo.glob("datasets/resolution_sets/*.json"))
    for f in files:
        d = json.loads(f.read_text())
        for q in d.get("questions", []) + d.get("resolutions", []):
            if q.get("source") != "manifold":
                continue
            ids.update(q["id"] if isinstance(q["id"], list) else [q["id"]])
            if q.get("question"):
                texts.add(_text_hash(q["question"]))
    out = {"source": "forecastingresearch/forecastbench-datasets "
                     "(datasets/question_sets + resolution_sets, source == manifold)",
           "files": len(files), "ids": sorted(ids), "question_hashes": sorted(texts)}
    FORECASTBENCH.parent.mkdir(parents=True, exist_ok=True)
    FORECASTBENCH.write_text(json.dumps(out, indent=0) + "\n")
    return out


def _forecastbench() -> tuple[set[str], set[str]]:
    # missing means a build that silently re-admits benchmark markets: fail instead
    d = json.loads(FORECASTBENCH.read_text())
    return set(d["ids"]), set(d["question_hashes"])


RESOLVED_EARLY_MS = 60_000      # closeTime within a minute of resolutionTime
_UPDATE = re.compile(r"Update (\d{4}-\d{2}-\d{2})")


def _strip_post_close(description: str, close_ms: int) -> str:
    """Cut the description at the first creator update dated after the market closed:
    text written once the answer may have been known is not part of the forecast."""
    close_day = datetime.fromtimestamp(close_ms / 1000, timezone.utc).date()
    for m in _UPDATE.finditer(description):
        try:
            if date.fromisoformat(m.group(1)) > close_day:
                return description[:m.start()].rstrip()
        except ValueError:
            continue
    return description


def usable(rows: list[dict] | None = None, report: dict | None = None) -> list[dict]:
    """Rows whose price is a pre-outcome forecast, with `p` replaced by the close price.

    Dropped, in order: non-predictive groups, anything in ForecastBench, markets with no
    cached full record (cannot be timed), markets resolved before their close, and markets
    traded after their recorded close (a close date that was moved)."""
    rows = rows if rows is not None else download()
    fb_ids, fb_texts = _forecastbench()
    out = []
    for r in rows:
        why = None
        full = None
        if (r.get("group") or "general") in EXCLUDED_GROUPS:
            why = "non_predictive_group"
        elif r["id"] in fb_ids or _text_hash(r["question"]) in fb_texts:
            why = "forecastbench"
        else:
            full = full_record(r["id"])
            t = _timing(full) if full else {k: r.get(k) for k in
                                           ("close_time", "resolution_time",
                                            "last_bet_time", "p_close")}
            if t.get("close_time") is None or t.get("resolution_time") is None \
                    or t.get("p_close") is None:
                why = "untimed"
            elif t["resolution_time"] - t["close_time"] <= RESOLVED_EARLY_MS:
                why = "resolved_before_close"
            elif (t.get("last_bet_time") or 0) > t["close_time"]:
                why = "traded_after_close"
        if why:
            if report is not None:
                report[why] = report.get(why, 0) + 1
            continue
        out.append(dict(r, p=float(t["p_close"]), p_resolution=r["p"],
                        close_time=t["close_time"],
                        description=_strip_post_close(r["description"], t["close_time"])))
    if report is not None:
        report["kept"] = len(out)
    return out


QUESTION_ID = "d05.manifold_resolves"
QUESTION = "Does this {topic} market resolve YES?"
INSTRUCTIONS = ("Answer with the probability the market itself put on YES when trading "
                "closed, before the outcome was known -- the traders' forecast, not the "
                "resolution.")
_BANK: list = []


def _question(topic: str, key: str) -> str:
    if not _BANK:
        from lod.phrasings import PhrasingBank
        _BANK.append(PhrasingBank.load())
    return _BANK[0].pick(QUESTION_ID, QUESTION, key).format(topic=topic)


def _example(task: str, topic: str, r: dict) -> Example:
    return Example(
        task=task,
        state=_state(r),
        questions=[Question(
            id="resolves_yes",
            question=_question(topic, r["id"]),
            options=["no", "yes"],
            target=1 if r["resolution"] == "YES" else 0,
            # the forecast, not the answer: probes compare the model against it
            meta={"p": r["p"], "market_id": r["id"]},
            instructions=INSTRUCTIONS,
        )],
    )


def probability_spread(rows: list[dict], bins: int = 10) -> dict[str, int]:
    """How the close probabilities are distributed. A corpus piled up at the ends is
    not much use for calibration, so this is worth printing rather than assuming."""
    out: dict[str, int] = {}
    for r in rows:
        b = min(int(r["p"] * bins), bins - 1)
        key = f"{b/bins:.1f}-{(b+1)/bins:.1f}"
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))


MIN_PER_GROUP = 40
MAX_GROUPS = 14
# Groups whose markets are not forecasts of anything. `fairlyrandom` resolves by RNG and
# `nonpredictive` is Manifold's own label for markets that are explicitly not predictive,
# so their close price is not an estimate of an outcome. Including them would put noise
# into meta.p, which exists precisely to be a forecast worth measuring against.
EXCLUDED_GROUPS = ("fairlyrandom", "nonpredictive", "nonpredictive-profits")


def groups(rows: list[dict] | None = None) -> list[str]:
    """Market groups big enough to be their own schema.

    Row 28 as one schema can contribute at most `per-schema` questions, which is why
    Manifold alone measured 0.86 % against A6's 3 %. Row 26's own structure is five
    sports, i.e. five schemas; a group here is the same thing, and the question text
    genuinely differs per group. Counted over `usable` rows.
    """
    from collections import Counter

    rows = rows if rows is not None else usable()
    counts = Counter(group_of(r) for r in rows if group_of(r) not in EXCLUDED_GROUPS)
    return [g for g, n in counts.most_common(MAX_GROUPS) if n >= MIN_PER_GROUP]


OTHER = "other"
# Groups that are the same markets under another slug. 35 `mlb` markets are the same
# "MLB: Will the X sweep the Y OR will the 3-game series total N or more runs?" template
# as the 43 in `baseball`; with `mlb` pooled into the trained `manifold_other`, the
# shingle dedup deleted 40 of devreal's 43 baseball rows as near-copies of training rows.
GROUP_ALIASES = {"mlb": "baseball"}


def group_of(r: dict) -> str:
    g = r.get("group") or "general"
    return GROUP_ALIASES.get(g, g)


def _state(r: dict) -> str:
    state = r["question"]
    if r["description"]:
        state += "\n\n" + r["description"]
    return state[:4000]


_LINKED: dict = {}


def eval_linked(rows: list[dict]) -> set[str]:
    """Ids of markets that share a 10-word shingle, directly or through a chain, with a
    market in a held-out group. Training tasks skip them.

    Manifold is full of series written from one template ("MLB: Will the X sweep the Y OR
    will the 3-game series total N or more runs?", "Real Betis defeat Getafe ?" + the same
    resolution boilerplate). `dedup_against_train` then deletes the held-out copy, not the
    trained one: 75 of 78 devreal baseball rows and 50 of 230 testreal rows went that way.
    Keeping the whole connected component off the training side keeps the held-out groups
    intact, at the cost of the training copies of those series.
    """
    key = tuple(r["id"] for r in rows)
    if key in _LINKED:
        return _LINKED[key]
    from lod.corpus.services.sources.real.base import shingles

    parent = list(range(len(rows)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    owner: dict[str, int] = {}
    for i, r in enumerate(rows):
        for sh in shingles(_state(r)):
            if sh in owner:
                parent[find(i)] = find(owner[sh])
            else:
                owner[sh] = i
    held = {find(i) for i, r in enumerate(rows) if _force_split(group_of(r)) != "train"}
    out = {r["id"] for i, r in enumerate(rows) if find(i) in held}
    _LINKED[key] = out
    return out


def _load_group(group: str):
    def load(n: int) -> Iterator[Example]:
        made = 0
        rows = usable()
        named = set(groups(rows)) if group == OTHER else set()
        linked = eval_linked(rows) if _force_split(group) == "train" else set()
        for r in rows:
            if made >= n:
                return
            if r["id"] in linked:
                continue
            g = group_of(r)
            if group == OTHER:
                # every usable market whose group is too small to be its own schema --
                # 934 of 1,354 after the timing filter, which would otherwise be dropped
                if g in named:
                    continue
            elif g != group:
                continue
            made += 1
            topic = "prediction" if group == OTHER else group.replace("-", " ")
            yield _example(f"manifold_{group}", topic, r)
    return load


def _load(n: int) -> Iterator[Example]:
    for i, r in enumerate(usable()):
        if i >= n:
            return
        yield _example("manifold_resolved", "prediction", r)


# The `probability_recovery` gate's testreal probe was 2 families/277 rows domain-wide,
# and only one of them (`general`) was Manifold, reaching testreal by hash luck rather
# than by a deliberate holdout. Forced now, on the same "hold out a structure, pair it
# against a trained sibling" pattern as row 13's lichess themes: `us-politics` (the
# narrower slice) is held out against `politics-default` (kept trained) and `ai` is held
# out against `technology-default` (kept trained) -- each pair tests whether a probability
# recovered on the broad topic transfers to its named specialisation. `fun` is held out
# with no sibling, as a genuinely different register (whimsical markets vs. the
# news-driven rest) to keep the probe from being only "narrow slice of a trained topic".
# `general` (the catch-all bucket, already testreal) is kept there rather than trained,
# since it has no single topic to be a trained sibling of. `baseball` and `sports-default`
# stay the devreal check, deliberately rather than by hash. Every other discovered group
# trains. Since the timing filter, `fun` (26 usable) and `sports-default` (22) fall
# under MIN_PER_GROUP and their markets go to `manifold_other`, which trains; the names
# stay here so that a larger cache routes them as designed.
RESERVED_TESTREAL_GROUPS = ("general", "us-politics", "ai", "fun")
RESERVED_DEVREAL_GROUPS = ("baseball", "sports-default")


def _force_split(group: str) -> str:
    if group in RESERVED_TESTREAL_GROUPS:
        return "testreal"
    if group in RESERVED_DEVREAL_GROUPS:
        return "devreal"
    return "train"


def tasks() -> list[RealTask]:
    if not _CACHE_FILE.exists():
        return []
    rows = usable()
    gs = groups(rows)
    if len(gs) <= 1:
        return [RealTask(row=28, name="manifold_resolved", licence=LICENCE, url=URL,
                         load=_load, meta_p=True,
                         notes="resolved binary markets; meta.p = pool price at close")]
    out = [RealTask(row=28, name=f"manifold_{g}", licence=LICENCE, url=URL,
                    load=_load_group(g), meta_p=True, force_split=_force_split(g),
                    notes=f"resolved {g} markets that closed before resolving; meta.p = "
                          f"pool price at close; ForecastBench markets removed; "
                          f"forced to {_force_split(g)}")
           for g in gs]
    if len(rows) > sum(1 for r in rows if group_of(r) in gs):
        out.append(RealTask(row=28, name=f"manifold_{OTHER}", licence=LICENCE, url=URL,
                            load=_load_group(OTHER), meta_p=True, force_split="train",
                            notes="resolved markets from groups too small to be their own "
                                  "schema; meta.p = pool price at close; ForecastBench "
                                  "markets removed; forced to train"))
    return out
