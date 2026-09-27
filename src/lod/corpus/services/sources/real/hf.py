"""A generic Hugging Face adapter for the source table.

Most of the 32 rows reduce to the same operation: load a dataset, pick the text column,
pick the label columns, emit one task per label column. Written out by hand that is 25
near-identical modules; here it is one adapter and a table of `HFSpec` rows, so adding a
source costs a line rather than a file.

It also lets the table's *intent* be met where its literal transport is expensive. Rows
4, 6, 7, 8, 10-14, 29 and 30 name live systems and bulk archives -- arXiv metadata dumps,
the PubMed baseline, SEC EDGAR, the Lichess database. Each has a packaged mirror of the
same underlying records on the Hub, and a mirror of a real system's labels is still a
real system's labels: the definition of a real schema turns on where the *labels* came
from, not on which host served the bytes. Where a mirror is used, `HFSpec.mirror_of`
records what it stands in for so the provenance manifest can say so.

Label detection is shared with the Hub survey behind `lod/assets/hub_sweep_survey.json`
rather than reimplemented, because the two disagreeing would mean the corpus contains
something the survey said was not there. A label column is one with 2-100 distinct values
whose values *repeat* -- an identifier does not repeat -- and whose name is not
identifier-shaped. That guard came from catching `idx` as a label on GLUE.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from lod.schema import Example, Question
from lod.corpus.repositories import raw_store as store
from lod.corpus.services.sources.real.base import RealTask, disjoint_slice

# an identifier is not a label, however few rows are sampled
ID_COL = re.compile(r"(^|_)(id|idx|index|uid|guid|key|hash|url|uri|path|file|filename)(_|$)")
# A column that records *who annotated the row*, or when, is not a property of the text
# and no reader could recover it from the state. The answerability gate in `quality.py`
# does not catch these: it rejects an option set of bare numerals, and these have
# perfectly ordinary options -- `no`/`yes`, a list of worker uuids, `man`/`woman`. It is
# the question that is unanswerable, not the options. Measured on an earlier build: 1,030 of
# 29,900 row-1 questions, 3.4 %, asking things like "Is this text annotated by the
# annotator 0ec6397b-ed29-4f0f-ad7d-7744e53f5b25?" and "What is the gender of the
# annotator who labeled this text?".
#
# Matched against `_snake(col)`, so camelCase and upper case reach it as words: the
# regex used to see `annotatorid` for `annotatorId` and `workerid` for `WorkerID`, and
# neither has an underscore to anchor on (domain-1 audit). `human_annotation` / `annotated`
# record *whether* a person labelled the row -- lmsys/toxic-chat's `human_annotation`
# shipped as "What is the human annotation of this text?" over False/True.
ANNOTATOR_COL = re.compile(
    r"(^|_)(annotator|annotators|worker|workers|submitter|rater|raters|coder|"
    r"turker|contributor|reviewer|assignment|hit|annotated|human_annotation)(_|$)")
# A column recording *when, where or with what* the row was produced is provenance, not
# a property of the text, and the same argument as ANNOTATOR_COL applies: the options are
# ordinary, it is the question that no reader could answer. Measured on an earlier build, the
# row-1 sweep shipped "What calendar date is associated with the provided text?" over
# eleven dates given a Fernet blob, "Which Common Crawl crawl snapshot does this web text
# come from?", "Which quantization precision was used for the language model?" and
# "Which task is this passage an example of?" -- that last one 2,000 questions with a
# constant answer, because the generic adapter read Super-NaturalInstructions' own
# `task_name` as a label while `instructions.py` was already reading the file properly.
# Kept deliberately narrow: `language`, `source` and `publisher` are properties a reader
# can recover from the text and are not listed.
PROVENANCE_COL = re.compile(
    r"(^|_)(date|datetime|timestamp|datestamp|time|created|updated|modified|fetched|"
    r"crawl|dump|snapshot|scraped|ingested|curation|"
    r"version|commit|revision|sha|checksum|seed|run|split|config|subset|"
    r"dataset|task|taskname|extractor|pipeline|generator|quantization|temperature|"
    r"checkpoint|model\d*|modelname)(_|$)")
# a label whose *values* are uuids is an identifier however the column is named
UUID_VALUE = re.compile(r"^[0-9a-f]{8}[-_][0-9a-f]{4}[-_][0-9a-f]{4}", re.I)
MAX_OPTIONS = 255   # row 1's class cap, raised from 100
MIN_OPTIONS = 2
# a ClassLabel whose names are themselves numbers tells us nothing the index did not
NUMERIC = re.compile(r"^-?\d+(\.\d+)?$")
SAMPLE = 400


@dataclass
class HFSpec:
    """One Hub dataset, and how the source table wants it read."""

    row: int
    path: str
    licence: str
    config: str | None = None
    split: str | None = None
    text_cols: tuple[str, ...] = ()        # empty -> longest string column
    label_cols: tuple[str, ...] = ()       # empty -> auto-detect
    skip_cols: tuple[str, ...] = ()
    question: str = ""                      # "" -> derived from the column name
    # column -> its own question, where one spec asks several different things (Civil
    # Comments: toxicity, insult, obscene ...). Wins over `question`. (domain-3 audit)
    questions: dict[str, str] = field(default_factory=dict)
    ordinal_cols: tuple[str, ...] = ()
    soft_cols: tuple[str, ...] = ()        # float columns in [0,1] -> soft bool targets
    # redact the gold label's own wording from the state before emitting it. A clause
    # classifier whose clause is headed by its own class name is a string match, not a
    # reading task. See `_scrub_label`.
    scrub_label: bool = False
    # regexes blanked out of the state before it is emitted, applied line by line with
    # re.MULTILINE. `scrub_label` removes the gold label's own wording; this removes a
    # *structural* header that names the label in a form the label string does not match.
    # Congressional bills open with "[H.R. 3870 Placed on Calendar Senate (PCS)]" and the
    # class is `hr`, so no amount of redacting `hr` reaches it -- measured on an earlier build,
    # 572 of 572 states, 100 %, stated their own answer in the first 200 characters. arXiv
    # papers carry the submission's own "arXiv:1611.03253v1 [cs.DS]" stamp, which equals
    # the gold category on 55.4 % of them.
    strip_state_res: tuple[str, ...] = ()
    # the column naming the *item*, where the dataset publishes one row per annotator
    # rather than one row per item. Set it and `soft_cols` are averaged over every row
    # sharing it before anything is scored. See `_aggregate_soft_rows`.
    soft_aggregate_by: str = ""
    # the column naming the *rater* inside a `soft_aggregate_by` group. Social Bias Frames
    # writes one row per worker *per implied statement*, so a worker who listed three
    # stereotypes had three votes. Set it and each rater counts once. (domain-3 audit)
    soft_rater_by: str = ""
    # fill each soft task's disjoint slice positives-first (`crowd.enriched_slices`), so
    # a rare attribute is not a task a constant "no" answers. Only for a spec whose every
    # task is one of `soft_cols`: the slices partition the pool among those columns.
    soft_enrich: bool = False
    # column -> the split that column's task must land in, whatever the name hashes to.
    # The generic adapter had no equivalent of `lichess.py`'s and `rules.py`'s
    # `force_split`, so a Hub task could only ever go where the hash sent it. That is
    # how domain 9 lost its subject: the hash put *every* entailment-relation task
    # (SNLI `label`, MultiNLI `label`) in devreal/testreal, leaving MultiNLI `genre` --
    # a register classifier solvable from either span alone -- as the only NLI task the
    # model ever trained on. A whole-task holdout is the right default; it is not a
    # reason to be unable to pin one task the other way.
    force_split_cols: dict[str, str] = field(default_factory=dict)
    drop_label_re: str = ""                 # label values matching this are not labels
    # the dataset publishes no train split and may still be read. Off by default: a
    # dataset with only test/validation splits is somebody's held-out benchmark, and the
    # old fall-through read it anyway (MMLU's test set reached training that way). Set it
    # only where the splits partition *data* -- by month, licence, genre -- not a
    # benchmark's evaluation set. See `pick_splits`. (domain-1 audit)
    no_train_split_ok: bool = False
    mirror_of: str = ""                     # what live system this stands in for
    notes: str = ""
    max_rows: int = 4000

    @property
    def key(self) -> str:
        base = re.sub(r"[^A-Za-z0-9]+", "_", self.path).strip("_").lower()
        return f"{base}_{self.config}" if self.config else base


def _question_for(col: str) -> str:
    pretty = col.replace("_", " ").strip()
    return f"What is the {pretty} of this text?"


def load_rows(spec: HFSpec) -> list[dict]:
    """Rows for this spec, from the local raw store. Never touches the network.

    Building the corpus is iterated on far more often than fetching it, so the two are
    separate stages: `fetch_rows` downloads once into `store`, and everything downstream
    reads from there. Before this split a dataset with seven label columns was downloaded
    seven times, once per task.
    """
    try:
        return store.load(spec.key)
    except FileNotFoundError:
        return _load_local_snapshot(spec)


def _load_local_snapshot(spec: HFSpec) -> list[dict]:
    """Read a downloaded Hub snapshot without contacting the Hub."""
    from pathlib import Path

    root = store.RAW_ROOT / "huggingface" / spec.path
    if not root.exists():
        raise FileNotFoundError(f"no local snapshot for {spec.path}: {root}")
    files = sorted(
        path for suffix in ("*.parquet", "*.csv", "*.json", "*.jsonl", "*.jsonl.gz")
        for path in root.rglob(suffix)
        if ".cache" not in path.parts and "README" not in path.name
    )
    if not files:
        raise FileNotFoundError(f"no tabular files in local snapshot: {root}")
    from datasets import load_dataset

    if files[0].suffix == ".parquet":
        import pyarrow.parquet as pq
        columns = set(pq.read_schema(files[0]).names)
        files = [path for path in files if set(pq.read_schema(path).names) == columns]
    suffix = files[0].name.lower()
    if suffix.endswith(".parquet"):
        builder = "parquet"
    elif suffix.endswith(".csv"):
        builder = "csv"
    else:
        builder = "json"
    dataset = load_dataset(builder, data_files=[str(path) for path in files], split="train")
    rows = []
    for row in dataset:
        rows.append(dict(row))
        if len(rows) >= spec.max_rows:
            break
    return rows


class EvalOnlyDataset(RuntimeError):
    """The dataset has no train split, and its spec does not say that is fine."""


# A split name that marks held-out evaluation data. Only consulted when there is no
# train split at all; `train` in the name always wins.
EVAL_SPLIT = re.compile(r"(test|valid|val|dev|eval|heldout|held_out|benchmark)", re.I)
# A domain-1 audit survey: the split names every cached Hub source publishes,
# measured with a date. The raw store keeps rows but not the split they were read from,
# so without this an offline build cannot tell a train sample from a benchmark test set.
SPLIT_SURVEY = Path(__file__).with_name("split_survey.json")
_SURVEY: dict[str, dict] | None = None


def pick_splits(splits: list[str], spec: HFSpec) -> list[str]:
    """The splits this spec may be read from, in order. Raises `EvalOnlyDataset`.

    Train splits only, whenever there is one -- the old order put the rest after them, so
    an empty train split fell through to test. With no train split at all the dataset is
    refused unless `spec.no_train_split_ok`: 48 of the 299 cached Hub sources publish no
    train split, and 31 of those publish nothing but test/validation/dev -- MMLU, GLUE's
    test-only config, Belebele, MathVista, CTI-Bench, LEXam (domain-1 audit).
    """
    train = [s for s in splits if "train" in s.lower()]
    if train:
        return train
    if not spec.no_train_split_ok:
        raise EvalOnlyDataset(f"{spec.path}: no train split (splits: {list(splits)[:6]}); "
                              f"set no_train_split_ok only if they partition data, "
                              f"not a benchmark's held-out set")
    return [s for s in splits if not EVAL_SPLIT.search(s)] or list(splits)


def surveyed_splits(key: str) -> list[str] | None:
    """Split names the survey recorded for this store key, or None if never surveyed."""
    global _SURVEY
    if _SURVEY is None:
        try:
            _SURVEY = json.loads(SPLIT_SURVEY.read_text()).get("sources", {})
        except (OSError, json.JSONDecodeError):
            _SURVEY = {}
    got = _SURVEY.get(key) or {}
    return got.get("splits")


def fetch_rows(spec: HFSpec) -> list[dict]:
    """Download rows from the Hub, with config and split discovered rather than assumed.

    GLUE's first config is `ax`, which has only a test split; assuming config[0] and
    "train" reads as a load failure on datasets that are perfectly usable. A config with
    no train split is skipped, and a dataset with none anywhere is refused -- see
    `pick_splits`.
    """
    from datasets import (get_dataset_config_names, get_dataset_split_names,
                          load_dataset)

    cfgs = [spec.config] if spec.config else None
    if cfgs is None:
        try:
            cfgs = get_dataset_config_names(spec.path) or [None]
        except Exception:
            cfgs = [None]
    last: Exception | None = None
    for cfg in cfgs[:3]:
        try:
            splits = ([spec.split] if spec.split
                      else get_dataset_split_names(spec.path, cfg))
        except Exception as e:
            last = e
            continue
        try:
            order = pick_splits(list(splits), spec)
        except EvalOnlyDataset as e:
            last = e
            continue
        for sp in order[:2]:
            try:
                ds = load_dataset(spec.path, cfg, split=sp, streaming=True)
                out = []
                for i, r in enumerate(ds):
                    if i >= spec.max_rows:
                        break
                    out.append(r)
                if out:
                    return out
            except Exception as e:
                last = e
    if isinstance(last, EvalOnlyDataset):
        raise last
    raise RuntimeError(f"{spec.path}: no loadable config/split "
                       f"({type(last).__name__ if last else 'empty'})")


def _asks_about_the_text(col: str, spec: HFSpec) -> bool:
    """Whether a column is a property of the text rather than of how the row was made.

    Both label detectors ask this, so a column is excluded in one place instead of two,
    and an explicit `label_cols` still wins -- a curated spec has already decided.
    """
    if col in spec.label_cols:
        return True
    low = _snake(col)
    return not (ANNOTATOR_COL.search(low) or PROVENANCE_COL.search(low))


def _snake(col: str) -> str:
    """`annotatorId` / `WorkerID` / `Worker ID` -> `annotator_id` / `worker_id`."""
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", col)
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", s)
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_").lower()


# A text column is free text. Anything else is another label wearing the text's place:
# `nishan-chatterjee/llm_bias_detection`'s longest string column holds fifteen distinct
# tokens (`numeric`, `lowercase_start`) and `rdpahalavan/cic-ids2017`'s holds a timestamp,
# so an earlier build asked which of fourteen languages the string "numeric" is written in --
# 42 questions there, 294 across the seven tasks with a state of that shape.
MIN_TEXT_DISTINCT = 0.5     # free text mostly does not repeat itself
MIN_TEXT_CHARS = 80         # ...or, failing that, it is long enough to be prose


def detect_text_col(rows: list[dict], spec: HFSpec) -> str:
    if spec.text_cols:
        return spec.text_cols[0]
    cands = [c for c in rows[0] if isinstance(rows[0].get(c), str)]
    if not cands:
        raise RuntimeError(f"{spec.path}: no string column")

    def total(col: str) -> int:
        return sum(len(str(r.get(col) or "")) for r in rows)

    def free_text(col: str) -> bool:
        distinct = len({str(r.get(col) or "") for r in rows})
        return (distinct > MIN_TEXT_DISTINCT * len(rows)
                or total(col) / max(1, len(rows)) >= MIN_TEXT_CHARS)

    def serialised(col: str) -> bool:
        """Mostly a JSON array/object in a string: a machine record, not prose."""
        hits = 0
        for r in rows[:200]:
            v = str(r.get(col) or "").strip()
            if v[:1] in "[{" and v[-1:] in "]}":
                try:
                    json.loads(v)
                    hits += 1
                except ValueError:
                    pass
        return hits > 0.5 * min(len(rows), 200)

    named = [c for c in cands if _asks_about_the_text(c, spec)] or cands
    free = [c for c in named if free_text(c)] or named
    # Prose before a serialised record, when there is prose at all (domain-1 audit).
    # `lmsys/toxic-chat` was read through `openai_moderation` -- a JSON list of
    # [category, score] pairs, the longest column -- so its toxicity task was answered
    # from another classifier's scores, and `epibench`'s eleven tasks from a JSON list of
    # figure captions ('[]' -> "Focal").
    free = [c for c in free if not serialised(c)] or free
    return max(free, key=total)


def state_for(row: dict, spec: HFSpec, text_col: str) -> str:
    """The state string for one raw row.

    A single `text_col` is the plain field. Several are rendered as a JSON object keyed
    by column name, because some sources are irreducibly a *pair*: MNLI's label is a
    relation between `premise` and `hypothesis`, and asking for it given only one of them
    is unanswerable -- which is what the generic adapter did, and worse, it picked
    `premise_parse` (a bracketed constituency tree) because that column is the longest.
    It also feeds the corpus budget of >= 25 % object or array states, which a corpus of
    bare strings never reaches.

    Column order follows `spec.text_cols`, so the rendering is stable across rebuilds.
    """
    if len(spec.text_cols) <= 1:
        return _strip_state(str(row.get(text_col) or ""), spec).strip()
    fields = {col: _strip_state(str(row.get(col) or ""), spec).strip()[:4000]
              for col in spec.text_cols}
    fields = {k: v for k, v in fields.items() if v}
    if not fields:
        return ""
    return json.dumps(fields, ensure_ascii=False, indent=2)


def _strip_state(text: str, spec: HFSpec) -> str:
    """Blank `spec.strip_state_res` out of the state. No-op where the spec sets none."""
    for pattern in spec.strip_state_res:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.MULTILINE)
    return text


def _keep_value(spec: HFSpec, value: Any, classlabel: bool = False) -> bool:
    """Whether a cell is a label at all.

    Some corpora interleave structural markers with their labels. PubMed-20k-RCT puts a
    `###<pmid>` document separator in the same column as OBJECTIVE / METHODS / RESULTS,
    which pushed the column to 309 distinct values -- over the 255 cap -- so the whole row
    was rejected and the source silently contributed nothing.

    A missing value is not a label either, whatever the spec says (domain-1 audit): an empty
    string or a NaN became an option `""` on 30 of 653 row-1 tasks and the *target* on
    486 of 48,206 sampled questions -- "What is the doi of this text?" answered `""`.
    And `-1` in a ClassLabel column is the Hub's own "no label" (SNLI's no-consensus
    pairs), so it is dropped there without every spec having to say `drop_label_re`.
    A plain integer column keeps its -1: sentiment scales use it.
    """
    if value is None:
        return False
    if isinstance(value, float) and value != value:
        return False
    if isinstance(value, str) and value.strip().lower() in MISSING_LABELS:
        return False
    if classlabel and not isinstance(value, bool) and str(value).strip() == "-1":
        return False
    if not spec.drop_label_re:
        return True
    return not re.match(spec.drop_label_re, str(value))


MISSING_LABELS = frozenset({"", "nan", "null"})


def _classlabel_cols(spec: HFSpec) -> set[str]:
    """Columns the store's feature sidecar records as ClassLabel, numeric names or not."""
    return set(store.load_features(spec.key))


def class_names(spec: HFSpec) -> dict[str, list[str]]:
    """Published ClassLabel names per column, keeping only the ones that say something.

    Names that are themselves numeric ("1".."13", as LexGLUE SCOTUS publishes them) are
    dropped: substituting them changes nothing, and keeping the map empty means a source
    without real names takes exactly the original path, raw values as options.
    """
    out = {}
    for col, names in store.load_features(spec.key).items():
        if names and not all(NUMERIC.match(n) for n in names):
            out[col] = names
    return out


def label_value(value, names: list[str] | None) -> str:
    """The stored value as an option string, resolved through ClassLabel names.

    A ClassLabel column arrives as an integer index. Only an in-range index is mapped: a
    source that already gave strings, or an integer that is genuinely a count rather than
    a class, must come through untouched.
    """
    if names and isinstance(value, bool) is False and isinstance(value, int):
        if 0 <= value < len(names):
            return names[value]
    return str(value)


def detect_label_cols(rows: list[dict], spec: HFSpec, text_col: str) -> dict[str, list[str]]:
    """Columns that behave like labels: low-cardinality, repeating, not identifiers."""
    named = class_names(spec)
    classlabel = _classlabel_cols(spec)
    if spec.label_cols:
        out = {}
        for c in spec.label_cols:
            # `drop_label_re` applies here too: without it SNLI's -1 (no consensus) was
            # skipped by the loader yet still offered as a live option on every question
            vals = sorted({label_value(r[c], named.get(c))
                           for r in rows
                           if r.get(c) is not None
                           and _keep_value(spec, r[c], c in classlabel)})
            # a declared order is the publisher's, not the alphabet's: WANDS publishes
            # Irrelevant < Partial < Exact, which sorts to Exact, Irrelevant, Partial
            if c in spec.ordinal_cols and named.get(c):
                order = named[c]
                vals.sort(key=lambda v: order.index(v) if v in order else len(order))
            if MIN_OPTIONS <= len(vals) <= MAX_OPTIONS:
                out[c] = vals
        return out

    out: dict[str, list[str]] = {}
    for col in rows[0]:
        if col == text_col or col in spec.skip_cols or ID_COL.search(col.lower()):
            continue
        if not _asks_about_the_text(col, spec):
            continue
        vals = [r.get(col) for r in rows]
        if any(isinstance(v, (list, dict)) for v in vals):
            continue
        if sum(1 for v in vals if isinstance(v, str) and UUID_VALUE.match(v)) > len(vals) // 2:
            continue
        vals = [v for v in vals if v is not None and _keep_value(spec, v, col in classlabel)]
        if not vals:
            continue
        if any(isinstance(v, str) and len(v) > 80 for v in vals):
            continue
        distinct = sorted({label_value(v, named.get(col)) for v in vals})
        # two options that differ only in case or spacing are one label spelled twice,
        # and a question offering both has two right answers: `Linux` / `linux`, `IEEE` /
        # `ieee` -- 16 of 653 row-1 tasks (domain-1 audit). Which spelling is canonical is not
        # something the adapter can know, so the column is not asked.
        if len({_norm_option(d) for d in distinct}) != len(distinct):
            continue
        # a label repeats; an id does not
        if MIN_OPTIONS <= len(distinct) <= MAX_OPTIONS and len(distinct) <= 0.5 * len(vals):
            out[col] = distinct
    return out


def _norm_option(value: str) -> str:
    return " ".join(str(value).split()).casefold()


# Ordinal means "option set with a declared total order (ratings, severities,
# priorities, none/low/high, negative/neutral/positive)". Detecting that from the option
# set is the only way to reach it at scale: row 1 alone contributes ~1,350 schemas, and
# text-classification label sets are full of ordered scales that no table could enumerate.
_ORDERED_VOCABS: tuple[tuple[str, ...], ...] = (
    ("negative", "neutral", "positive"),
    ("low", "medium", "high"),
    ("none", "low", "medium", "high"),
    ("none", "low", "high"),
    ("none", "partial", "complete"),
    ("never", "rarely", "sometimes", "often", "always"),
    ("strongly disagree", "disagree", "neutral", "agree", "strongly agree"),
    ("very negative", "negative", "neutral", "positive", "very positive"),
    ("terrible", "bad", "ok", "good", "great"),
    ("low", "moderate", "high", "critical"),
    ("info", "warning", "error", "critical"),
    ("debug", "info", "warn", "error", "fatal"),
    ("trivial", "minor", "major", "critical", "blocker"),
    ("none", "mild", "moderate", "severe"),
    ("beginner", "intermediate", "advanced"),
)


def is_ordinal_options(options: list[str]) -> bool:
    """Does this option set carry a declared total order?

    Two ways to qualify: the options are a contiguous run of numbers (a rating scale), or
    they match a known ordered vocabulary. Deliberately conservative -- a wrongly tagged
    ordinal inflates the ordinal count with a schema that has no order, which is worse than
    missing one, because the tag is what a future reader trusts.
    """
    # Detection requires at least three levels. A two-value set like 0/1 or spam/ham is a
    # binary class, not a scale, and treating every binary column as ordinal would inflate
    # the ordinal count with schemas that carry no order. The genuinely ordered binaries --
    # row 9's attackComplexity is LOW < HIGH -- are tagged explicitly in the table instead.
    if len(options) < 3:
        return False
    low = [o.strip().lower() for o in options]

    nums = []
    for o in low:
        try:
            nums.append(float(o))
        except ValueError:
            nums = []
            break
    if nums and len(nums) == len(set(nums)):
        ordered = sorted(nums)
        # a contiguous integer run, in any presented order: 1..5, 0..4, "1".."10"
        if all(float(b - a) == 1.0 for a, b in zip(ordered, ordered[1:])):
            return True

    s = set(low)
    for vocab in _ORDERED_VOCABS:
        if s == set(vocab):
            return True
    # star ratings and "N out of M" style
    if all(re.fullmatch(r"\d+\s*(star|stars|out of \d+)", o) for o in low):
        return True
    return False


MAX_MULTILABEL = 20


def detect_multilabel_cols(rows: list[dict], spec: HFSpec,
                           text_col: str) -> dict[str, list[str]]:
    """List-valued columns that are multi-label annotations, and their vocabularies.

    The single-label detector skips any list column, which silently discarded real label
    sets -- LexGLUE's unfair_tos ships its clause types as a list, and it detected nothing
    at all. A multi-label column cannot be one question over its vocabulary (the answer is
    a set, and the model answers with one option), so each label becomes its own yes/no
    question: "is this clause unfair in way X". That is a faithful reading, not a
    reshaping -- it is how the annotation was collected.
    """
    named = class_names(spec)
    out: dict[str, list[str]] = {}
    for col in rows[0]:
        if col == text_col or col in spec.skip_cols or ID_COL.search(col.lower()):
            continue
        if not _asks_about_the_text(col, spec):
            continue
        vals = [r.get(col) for r in rows]
        if not any(isinstance(v, list) for v in vals):
            continue
        vocab: set[str] = set()
        n_lists = 0
        for v in vals:
            if not isinstance(v, list):
                continue
            n_lists += 1
            for item in v:
                if isinstance(item, (str, int)) and len(str(item)) <= 60:
                    # a Sequence(ClassLabel) holds indices; the vocabulary is its names
                    vocab.add(label_value(item, named.get(col)))
        # a list of free text or of unique ids is not a label vocabulary
        if n_lists < len(vals) * 0.5 or not 2 <= len(vocab) <= MAX_MULTILABEL:
            continue
        # ...and nor is a list of numbers. The bare-numeral gate in `quality.py` cannot
        # see these, because each becomes a no/yes question: `mjbommar/shelf` shipped
        # "Does this text carry the label '4000'?" over a word-count range (domain-1 audit).
        if sum(1 for v in vocab if NUMERIC.match(v)) >= 0.5 * len(vocab):
            continue
        out[col] = sorted(vocab)
    return out


def _multilabel_loader(spec: HFSpec, col: str, label: str, index: int, n_tasks: int):
    def load(n: int) -> Iterator[Example]:
        rows = load_rows(spec)
        text_col = detect_text_col(rows, spec)
        names = class_names(spec).get(col)
        # the same text published twice with different label sets answers both ways
        sets: dict[str, set] = defaultdict(set)
        for r, st in zip(rows, _pool_states(rows, spec, text_col)):
            if isinstance(r.get(col), list):
                sets[st].add(frozenset(label_value(x, names) for x in r[col]))
        ambiguous = {st for st, got in sets.items() if len(got) > 1}
        rows = disjoint_slice(_spread(rows), index, n_tasks)
        made = 0
        seen: set[str] = set()
        for r in rows:
            if made >= n:
                return
            state = state_for(r, spec, text_col)
            v = r.get(col)
            if not state or not isinstance(v, list) or state in ambiguous or state in seen:
                continue
            seen.add(state)
            made += 1
            present = label in {label_value(x, names) for x in v}
            yield Example(
                task=f"{spec.key}__{col}__{re.sub(r'[^A-Za-z0-9]+', '_', label)[:30]}",
                state=state[:4000],
                questions=[Question(id=col,
                                    question=f"Does this text carry the label "
                                             f"{label!r}?",
                                    options=["no", "yes"], target=int(present))],
            )
    return load


def _spread(rows: list[dict]) -> list[dict]:
    """Reorder a pool so that any prefix of it covers the whole pool evenly.

    A loader stops at its quota, and `disjoint_slice` hands each task a contiguous block,
    so both read a *prefix of the file order*. Plenty of Hub datasets ship sorted by their
    own label, and then the prefix is not a sample of the dataset, it is a sample of one
    class. `benayas/snips` is 1,873 BookRestaurant, 1,842 AddToPlaylist and 285 GetWeather
    in that order; the first 2,000 rows are 92 % AddToPlaylist, which is exactly the
    majority baseline `benayas_snips__category` had in an earlier build -- 0.921 over three
    options, against 0.47 for the pool it was drawn from.

    Bit-reversal (van der Corput) rather than a shuffle: it is deterministic without a
    seed or a hash, every prefix is evenly spread, and a contiguous block of the reordered
    pool is a *stride* of the original, so `disjoint_slice` keeps its disjointness and
    stops being a prefix at the same time.
    """
    n = len(rows)
    if n < 2:
        return list(rows)
    width = (n - 1).bit_length()
    order = sorted(range(n), key=lambda i: int(f"{i:0{width}b}"[::-1], 2))
    return [rows[i] for i in order]


def _soft_target(value: Any) -> list[float] | None:
    """A rater fraction in [0,1] becomes a soft distribution over ["no","yes"]."""
    try:
        p = float(value)
    except (TypeError, ValueError):
        return None
    if not 0.0 <= p <= 1.0:
        return None
    return [1.0 - p, p]


def _scrub_label(text: str, label: str) -> str:
    """Redact a label's own wording from the state, the way `spdx._scrub` does.

    LEDGAR's classes are clause headings -- "Governing Laws", "Arbitration",
    "Confidentiality" -- and a contract clause very often says its own heading in its
    first line. Measured on an earlier build: 2,156 of 4,000 `ledgar__label` states, **53.9 %**,
    contained their own gold label verbatim, so more than half the task was reading a
    heading rather than a clause. SPDX has had a scrub since it was written; the generic
    adapter had none, so every Hub source whose labels are drawn from the text's own
    vocabulary leaked.

    Singular and plural are both redacted -- a class called "Agreements" appears in prose
    as "agreement" -- and each contiguous multi-word run of a multi-word label as well,
    since "Governing" alone gives the answer away among 100 classes.
    """
    tokens: set[str] = set()
    words = label.split()
    for i in range(len(words)):
        for j in range(i + 1, len(words) + 1):
            run = " ".join(words[i:j])
            if len(run) < 4:      # "To", "Of", "Law" are ordinary English
                continue
            tokens.add(run)
            if run.endswith("s"):
                tokens.add(run[:-1])
            else:
                tokens.add(run + "s")
    out = text
    for token in sorted(tokens, key=len, reverse=True):
        out = re.sub(rf"\b{re.escape(token)}\b", "[REDACTED]", out, flags=re.IGNORECASE)
    return out


MIN_SOFT_RATERS = 3   # below this a fraction is not a distribution (crowd.py's floor)


def _aggregate_soft_rows(rows: list[dict], group_col: str,
                         cols: tuple[str, ...], rater_col: str = "") -> list[dict]:
    """Average each of `cols` over every row sharing `group_col`.

    Some soft-labelled Hub datasets publish one row per *annotator*, not one row per
    item. Social Bias Frames is 4,000 rows over 1,357 posts, each carrying one worker's
    own 0 / 0.5 / 1 judgement and that worker's demographics. Read row by row, each
    annotator's opinion becomes its own "soft" example -- a fraction over a single
    rater, which is a hard label wearing a soft label's clothes -- and the same post
    reaches the corpus several times with *contradictory* targets. Measured on
    an earlier build: 446 duplicate-state groups under `offensiveYN`, 184 of them disagreeing
    with themselves.

    That is the failure `crowd.py` was written to avoid for GoEmotions and Measuring
    Hate Speech; this does the same aggregation for the generic adapter. It is gated
    behind `HFSpec.soft_aggregate_by`, so a source that already publishes one row per
    item is untouched.

    With `rater_col`, each rater counts once (their first row): SBF repeats a worker's
    row once per implied stereotype, so a row mean over-weighted whoever listed more,
    and 220 of 917 posts with >= 3 rows had fewer than 3 distinct workers (domain-3 audit).
    A column is averaged only over raters who answered it (NaN is no answer), and is
    left unset -- no question -- below `MIN_SOFT_RATERS` answers.
    """
    groups: dict[Any, list[dict]] = defaultdict(list)
    seen: set = set()
    for r in rows:
        key = r.get(group_col)
        if key is None:
            continue
        if rater_col:
            if (key, r.get(rater_col)) in seen:
                continue
            seen.add((key, r.get(rater_col)))
        groups[key].append(r)
    out: list[dict] = []
    for grp in groups.values():
        if len(grp) < MIN_SOFT_RATERS:
            continue
        base = dict(grp[0])
        for c in cols:
            vals = [float(r[c]) for r in grp
                    if isinstance(r.get(c), (int, float))
                    or (isinstance(r.get(c), str) and NUMERIC.match(r[c]))]
            vals = [v for v in vals if v == v]
            base[c] = sum(vals) / len(vals) if len(vals) >= MIN_SOFT_RATERS else None
        out.append(base)
    return out


# what counts as a "yes" item when `soft_enrich` fills a slice: a fifth of the raters.
# Not "> 0": Civil Comments items have up to ~10 raters, and its one-in-ten items are
# mostly noise -- claiming them spent the positive half of each slice on near-zeros and
# left `toxicity` with 1.5 % majority-yes items instead of 4.1 % (domain-3 audit).
SOFT_POSITIVE = 0.2


def _soft_positive(row: dict, col: str) -> bool:
    """Enough raters said yes: the item's fraction for `col` is at least SOFT_POSITIVE."""
    t = _soft_target(row.get(col))
    return t is not None and t[1] >= SOFT_POSITIVE


def _labels_by_state(rows: list[dict], spec: HFSpec, text_col: str,
                     col: str) -> dict[str, set[str]]:
    """state -> every label the source gives it in `col`, over the whole pool."""
    names = class_names(spec).get(col)
    is_cl = col in _classlabel_cols(spec)
    out: dict[str, set[str]] = defaultdict(set)
    for r, state in zip(rows, _pool_states(rows, spec, text_col)):
        raw = r.get(col)
        if isinstance(raw, (list, dict)) or not _keep_value(spec, raw, is_cl):
            continue
        if state:
            out[state].add(label_value(raw, names))
    return out


_STATES: dict[tuple, tuple[list[dict], list[str]]] = {}


def _pool_states(rows: list[dict], spec: HFSpec, text_col: str) -> list[str]:
    """`state_for` over a whole pool, memoised: every task of a source asks for it.
    The memo holds the pool itself, so its id cannot be recycled while it is cached."""
    key = (spec.key, text_col, id(rows), len(rows))
    if key not in _STATES:
        _STATES.clear()
        _STATES[key] = (rows, [state_for(r, spec, text_col) for r in rows])
    return _STATES[key][1]


# A column whose texts the source labels more than one way this often is not a property of
# the text. Measured on the row-1 pool (domain-1 audit): `tasksource/folio`'s premises repeat
# once per conclusion and the conclusion is not in the state (77 of 80 sampled states
# carry two labels); `mjbommar/opengloss` asks a *query's* intent given the encyclopedia
# entry every query shares; NLI sets read through one column are the same failure.
AMBIGUOUS_SHARE = 0.2
# Row 1 is not curated, so it also answers for the two shortcuts a curated spec would have
# caught by reading the dataset: a column whose majority class is nearly all of it, and a
# column whose text spells its own label out. `quality.example_rejection` catches only the
# fully constant and the >= 80 % cases, and is shared with sources (unfair_tos, gh_label)
# whose owners set their own bar, so the stricter row-1 bar lives here.
ROW1_MAX_MAJORITY = 0.95
ROW1_MAX_LEAK = 0.5
# (task name -> why it was not built), for the audit; `tasks_for` fills it.
SKIPPED: dict[str, str] = {}


def _column_rejection(by_state: dict[str, set[str]], spec: HFSpec,
                      options: list[str]) -> str | None:
    """Why a hard-label column cannot be asked, judged over the whole pool, or None."""
    total = len(by_state)
    if not total:
        return None
    ambiguous = sum(1 for got in by_state.values() if len(got) > 1)
    if ambiguous >= AMBIGUOUS_SHARE * total and ambiguous >= 3:
        return (f"{ambiguous} of {total} distinct texts carry more than one label: the "
                f"label is not a property of the text")
    if spec.row != 1:
        return None
    from lod.corpus.services.sources.real import quality  # noqa: F401  (TRIVIAL_OPTION_SETS)
    single = [(st, next(iter(got))) for st, got in by_state.items() if len(got) == 1]
    if not single:
        return None
    counts: dict[str, int] = defaultdict(int)
    for _, lab in single:
        counts[lab] += 1
    top = max(counts.values())
    if len(single) >= 20 and top >= ROW1_MAX_MAJORITY * len(single):
        return f"majority class is {top / len(single):.0%} of {len(single)} texts"
    trivial = frozenset(o.strip().lower() for o in options) in quality.TRIVIAL_OPTION_SETS
    if not trivial and len(single) >= 20:
        leaked = 0
        # `quality._spelled_out`'s test, compiled once per option rather than per text
        opts = [(o, str(o).strip().lower()) for o in options]
        pats = {o: re.compile(r"(?<![A-Za-z0-9])" + re.escape(ol) + r"(?![A-Za-z0-9])")
                for o, ol in opts if ol}
        for st, lab in single:
            low = st[:4000].lower()          # what the loader emits
            spelled = {o for o, ol in opts if ol and ol in low and pats[o].search(low)}
            leaked += spelled == {lab}
        if leaked >= ROW1_MAX_LEAK * len(single):
            return (f"the text spells out its own label, and no other, on "
                    f"{leaked / len(single):.0%} of {len(single)} texts")
    return None


def _same_column(rows: list[dict], a: str, b: str) -> bool:
    """`a` and `b` determine each other row by row: one schema published twice."""
    ab: dict[str, str] = {}
    ba: dict[str, str] = {}
    shared = 0
    for r in rows:
        x, y = r.get(a), r.get(b)
        if x is None or y is None or isinstance(x, (list, dict)) or isinstance(y, (list, dict)):
            continue
        x, y = str(x), str(y)
        if ab.setdefault(x, y) != y or ba.setdefault(y, x) != x:
            return False
        shared += 1
    return shared >= MIN_OPTIONS and len(ab) >= MIN_OPTIONS


def _loader(spec: HFSpec, col: str, options: list[str], soft: bool,
            index: int, n_tasks: int):
    def load(n: int) -> Iterator[Example]:
        rows = load_rows(spec)
        if soft and spec.soft_aggregate_by:
            rows = _aggregate_soft_rows(rows, spec.soft_aggregate_by, spec.soft_cols,
                                        spec.soft_rater_by)
        text_col = detect_text_col(rows, spec)
        names = class_names(spec).get(col)
        is_cl = col in _classlabel_cols(spec)
        ambiguous = (set() if soft else
                     {st for st, got in _labels_by_state(rows, spec, text_col, col).items()
                      if len(got) > 1})
        # disjoint slice: several tasks share this dataset's rows, and two tasks sharing
        # states across splits is what makes the devreal/testreal dedup delete an eval set
        if soft and spec.soft_enrich:
            from lod.corpus.services.sources.real.crowd import enriched_slices
            rows = enriched_slices(_spread(rows), spec.soft_cols, _soft_positive)[col]
        else:
            rows = disjoint_slice(_spread(rows), index, n_tasks)
        made = 0
        seen: set[str] = set()
        for r in rows:
            if made >= n:
                return
            state = state_for(r, spec, text_col)
            # one text, one question: a repeat is not a new example, and a text the
            # source labels two ways has two defensible answers (domain-1 audit)
            if not state or state in seen or state in ambiguous:
                continue
            if soft:
                target = _soft_target(r.get(col))
                if target is None:
                    continue
                opts = ["no", "yes"]
            else:
                raw = r.get(col)
                val = label_value(raw, names)
                if val not in options or not _keep_value(spec, raw, is_cl):
                    continue
                target, opts = options.index(val), options
            seen.add(state)
            if not soft and spec.scrub_label:
                state = _scrub_label(state, val)
                if not state.strip():
                    continue
            made += 1
            yield Example(
                task=f"{spec.key}__{col}",
                state=state[:4000],
                questions=[Question(id=col,
                                    question=(spec.questions.get(col) or spec.question
                                              or _question_for(col)),
                                    options=list(opts), target=target)],
            )
    return load


def tasks_for(spec: HFSpec) -> list[RealTask]:
    """One RealTask per label column, derived from already-fetched rows.

    Raises if the spec has not been fetched: schema construction is a local operation and
    silently reaching for the network here is what made a rebuild cost a download.
    """
    splits = surveyed_splits(spec.key)
    if splits is not None and not spec.split:
        pick_splits(splits, spec)          # raises EvalOnlyDataset
    rows = load_rows(spec)
    text_col = detect_text_col(rows, spec)
    detected = detect_label_cols(rows, spec, text_col)
    multi = detect_multilabel_cols(rows, spec, text_col)
    # the same schema under two column names is one task, not two -- the split hash would
    # put one copy in train and the other in eval (MASSIVE's `label` / `label_text`)
    # -- keeping the copy whose options are words: 20 Newsgroups publishes `label` as
    # integers and `label_text` as names, and the numeral copy is gated out anyway
    kept: list[str] = []
    for col in sorted(detected, key=lambda c: sum(1 for o in detected[c] if NUMERIC.match(o))
                      / max(1, len(detected[c]))):
        if col in spec.soft_cols:
            continue
        if any(_same_column(rows, other, col) for other in kept):
            SKIPPED[f"{spec.key}__{col}"] = "duplicates another column"
            del detected[col]
            continue
        kept.append(col)
    for col in list(detected):
        if col in spec.soft_cols:
            continue
        why = _column_rejection(_labels_by_state(rows, spec, text_col, col), spec,
                                detected[col])
        if why:
            SKIPPED[f"{spec.key}__{col}"] = why
            del detected[col]
    if spec.row == 1:
        multi = {col: [lab for lab in vocab if _multilabel_balanced(rows, spec, col, lab)]
                 for col, vocab in multi.items()}
        multi = {col: vocab for col, vocab in multi.items() if vocab}
    cols = list(detected) + [c for c in spec.soft_cols if c not in detected]
    # single-label and multi-label tasks share one numbering, so their row slices are
    # disjoint too: they used to both start at slice 0 (domain-1 audit)
    n_multi = sum(len(v) for v in multi.values())
    n_all = len(cols) + n_multi
    out: list[RealTask] = []
    for i, col in enumerate(cols):
        soft = col in spec.soft_cols
        options = detected.get(col, [])
        out.append(RealTask(
            row=spec.row,
            name=f"{spec.key}__{col}",
            licence=spec.licence,
            url=f"https://huggingface.co/datasets/{spec.path}",
            load=_loader(spec, col, options, soft, i, n_all),
            # tagged by the table where the source declares it, otherwise detected from
            # the option set itself
            ordinal=(col in spec.ordinal_cols) or is_ordinal_options(options),
            soft=soft,
            force_split=spec.force_split_cols.get(col),
            # every column task of one dataset reads the same rows (disjoint slices of
            # them), so the dataset is the held-out unit: `goemotions__joy` in testreal
            # does not measure transfer while `goemotions__anger` trains
            family=spec.key,
            notes=(spec.notes + (f"; mirror of {spec.mirror_of}" if spec.mirror_of else "")
                   ).strip("; "),
        ))

    # one yes/no task per label of each multi-label column
    k = len(cols)
    for col, vocab in multi.items():
        for label in vocab:
            safe = re.sub(r"[^A-Za-z0-9]+", "_", label)[:30]
            out.append(RealTask(
                row=spec.row,
                name=f"{spec.key}__{col}__{safe}",
                licence=spec.licence,
                url=f"https://huggingface.co/datasets/{spec.path}",
                load=_multilabel_loader(spec, col, label, k, n_all),
                family=spec.key,
                notes=(spec.notes + f"; multi-label {col}={label}").strip("; "),
            ))
            k += 1
    return out


def _multilabel_balanced(rows: list[dict], spec: HFSpec, col: str, label: str) -> bool:
    """Row 1: a per-label yes/no task whose answer is not 95 % one way over the pool.

    43 of the 133 row-1 multi-label tasks sampled by the domain-1 audit were >= 95 % `no` --
    a rare tag asked of every text -- and the multi-label variation as a whole had a
    majority baseline of 0.868.
    """
    names = class_names(spec).get(col)
    lists = [r[col] for r in rows if isinstance(r.get(col), list)]
    if not lists:
        return False
    yes = sum(1 for v in lists if label in {label_value(x, names) for x in v})
    share = yes / len(lists)
    return 1 - ROW1_MAX_MAJORITY < share < ROW1_MAX_MAJORITY
