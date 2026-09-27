"""Training data: the dataset, the criteria augmentations and a deterministic,
resumable batch order."""

from __future__ import annotations

import collections
import functools
import hashlib
import random
import re
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from lod import sentinel
from lod.model.packing import Packed, Packer, collate
from lod.schema import Example, Question


class SolvedTracker:
    """Down-weight tasks the model has finished learning.

    A task whose running train accuracy over its last `window` questions passes
    `on` is "solved" and its loss is scaled by `factor` until accuracy falls back
    below `off`. The hysteresis stops tasks flapping in and out each step.

    Why: under the log score a solved task still pays the model to push its
    logits further apart, and that sharpening lands on every question through the
    shared head, which shows up as rising held-out ECE while the solved tasks sit at
    accuracy 1.0 with NLL ~1e-4.
    """

    def __init__(self, factor: float = 0.2, window: int = 500,
                 on: float = 0.99, off: float = 0.97, min_seen: int = 50) -> None:
        self.factor = factor
        self.window = window
        self.on = on
        self.off = off
        self.min_seen = min_seen
        self._recent: dict[str, collections.deque] = collections.defaultdict(
            lambda: collections.deque(maxlen=window))
        self.solved: set[str] = set()
        self._weight_of: dict[str, float] = {}   # cache, invalidated when solved moves

    @property
    def enabled(self) -> bool:
        return self.factor != 1.0

    def update(self, tasks: Sequence[str], correct: Sequence[bool]) -> None:
        touched = set()
        for t, c in zip(tasks, correct):
            self._recent[t].append(1.0 if c else 0.0)
            touched.add(t)
        for t in touched:
            d = self._recent[t]
            if len(d) < self.min_seen:
                continue
            acc = sum(d) / len(d)
            was = t in self.solved
            if was and acc < self.off:
                self.solved.discard(t)
            elif not was and acc > self.on:
                self.solved.add(t)
            if was != (t in self.solved):
                self._weight_of.clear()

    def weights(self, tasks: Sequence[str]) -> list[float]:
        w = self._weight_of
        if not w:
            w.update({t: self.factor for t in self.solved})
        return [w.get(t, 1.0) for t in tasks]

    def weight_tensor(self, tasks: Sequence[str], device) -> "torch.Tensor":
        """The same weights as a tensor, from a dict that only rebuilds when a task
        crosses the hysteresis. The set membership test was running once per question
        per micro-step for a set that changes a few times an hour."""
        return torch.tensor(self.weights(tasks), dtype=torch.float32, device=device)


def shuffle_options(example: Example, rng: random.Random) -> Example:
    """Permute each question's options, remapping its target and its descriptions."""
    questions = []
    for q in example.questions:
        order = list(range(len(q.options)))
        rng.shuffle(order)
        options = [q.options[i] for i in order]
        target = q.target
        if isinstance(target, int):
            target = order.index(target)
        elif isinstance(target, list):
            target = [target[i] for i in order]
        descriptions = None
        if q.descriptions:
            descriptions = [q.description_for(i) for i in order]
        questions.append(Question(q.id, q.question, options, target, q.meta,
                                  q.instructions, descriptions))
    return Example(state=example.state, questions=questions, task=example.task)


# The "none of the above" sentinel is drawn from a grammar (`lod/sentinel.py`) rather than
# a fixed list, so the model has to read its wording instead of memorising a string.
NONE_OF_THE_ABOVE = sentinel.NONE_OF_THE_ABOVE
NONE_OF_THE_ABOVE_DESC = sentinel.NONE_OF_THE_ABOVE_DESC

_OPAQUE_RE = re.compile(r"^opt_\d+$")


def _is_choice(q: Question) -> bool:
    """A choice question: not the two-option noul form the API renders as no/yes."""
    return not (len(q.options) == 2 and [str(o).lower() for o in q.options] == ["no", "yes"])


def opaque_keys(q: Question) -> Question:
    """Replace the option keys with `opt_0..opt_k`, keeping the descriptions.

    The option key is an arbitrary identifier and the description carries the meaning;
    opaque keys force the model to read the description. Without descriptions there is
    nothing left to read, so a question with none is returned untouched rather than made
    unanswerable.
    """
    if not q.has_descriptions:
        return q
    options = [f"opt_{i}" for i in range(len(q.options))]
    return Question(q.id, q.question, options, q.target, q.meta, q.instructions,
                    list(q.descriptions))


def sentinel_eligible(q: Question) -> list[float] | None:
    """The target probabilities if this question can carry a sentinel, else None.

    One precondition for **both** the withheld and the decoy path, which is what makes
    the two produce the same distribution over option counts, target shapes and question
    kinds. Otherwise "does this question carry a sentinel" would be answerable from the
    option count alone.

    Needs >= 3 options so that dropping one still leaves a question, and an unambiguous
    correct option: removing the argmax of a 0.4/0.35/0.25 distribution does not make the
    rest wrong, and renormalising after dropping a 0.25 option would move the label.
    """
    if not _is_choice(q):
        return None
    probs = q.target_probs()
    if probs is None or len(q.options) < 3:
        return None
    best = max(range(len(probs)), key=probs.__getitem__)
    if probs[best] < 0.9:
        return None
    if any(sentinel.is_sentinel_key(o) for o in q.options):
        return None
    return probs


def _sentinel_option(q: Question, rng: random.Random, split: str) -> tuple[str, str | None]:
    """(key, description) for a sentinel appended to `q`, matching its local conventions.

    Two conventions, and both matter more than they look. If the question's keys have
    been made opaque, the sentinel takes the next `opt_N`: being the one readable key in
    a list of `opt_0..opt_k` is a tell that survives every wording change. If no option
    carries a description, neither does the sentinel: being the one described option in a
    bare list is the same tell in the other direction.
    """
    opaque = all(_OPAQUE_RE.match(str(o)) for o in q.options)
    key, desc = sentinel.sample(rng, split,
                                opaque_key=f"opt_{len(q.options)}" if opaque else None)
    return key, (desc if q.has_descriptions else None)


def _with_sentinel(q: Question, keep: list[int], probs: list[float],
                   rng: random.Random, split: str, abstain: bool) -> Question:
    key, desc = _sentinel_option(q, rng, split)
    options = [q.options[i] for i in keep] + [key]
    descriptions = [q.description_for(i) for i in keep] + [desc]
    kept = [probs[i] for i in keep]
    total = sum(kept)
    target = ([0.0] * len(keep) + [1.0] if abstain
              else [p / total for p in kept] + [0.0])
    # Shuffle the sentinel in with the rest: always appended last, it would be findable by
    # position, and a caller who puts "none of these" mid-list would meet an unseen layout.
    order = list(range(len(options)))
    rng.shuffle(order)
    options = [options[i] for i in order]
    descriptions = [descriptions[i] for i in order]
    target = [target[i] for i in order]
    meta = dict(q.meta or {})
    meta["sentinel"] = split
    if abstain:
        # carried rather than inferred from the key: the key is drawn from a grammar, so
        # `packing._is_withheld` cannot recognise one by string match, and the confidence
        # head's target depends on getting this right
        meta["abstain"] = True
    return Question(q.id, q.question, options, target, meta, q.instructions, descriptions)


def withhold_criteria(q: Question, rng: random.Random,
                      split: str = "train") -> Question | None:
    """Drop the correct option and append a sentinel, target 1.0 on it.

    The corpus's labelled "I do not know", and what the confidence head learns to be
    unsure about.
    """
    probs = sentinel_eligible(q)
    if probs is None:
        return None
    best = max(range(len(probs)), key=probs.__getitem__)
    keep = [i for i in range(len(q.options)) if i != best]
    return _with_sentinel(q, keep, probs, rng, split, abstain=True)


def decoy_none(q: Question, rng: random.Random,
               split: str = "train") -> Question | None:
    """Drop a *wrong* option and append a sentinel that is not the answer.

    Without this the phrase is perfectly predictive: every question in training that has
    a none-of-the-above option has it as the correct answer, so the model learns the
    string rather than the situation. Dropping a wrong option rather than simply
    appending is what keeps the option count identical to the withheld path -- see
    `sentinel_eligible`.
    """
    probs = sentinel_eligible(q)
    if probs is None:
        return None
    best = max(range(len(probs)), key=probs.__getitem__)
    drop = rng.choice([i for i in range(len(q.options)) if i != best])
    keep = [i for i in range(len(q.options)) if i != drop]
    return _with_sentinel(q, keep, probs, rng, split, abstain=False)



# --- refilling the gap the withheld option leaves ------------------------------------

_FILLERS: dict | None = None


def filler_pool(path: str | None = None) -> dict:
    """Option strings from other tasks, bucketed by surface shape (`_shape_bucket`): a JSON
    object {bucket: [option, ...]} named by $LOD_FILLER_POOL. Unset -- as in the release
    recipe -- it is empty, and churn only drops options rather than inventing any."""
    global _FILLERS
    if _FILLERS is None:
        import json as _json
        import os as _os
        p = path or _os.environ.get("LOD_FILLER_POOL", "")
        try:
            _FILLERS = _json.loads(open(p, encoding="utf-8").read()) if p else {}
        except OSError:
            _FILLERS = {}
    return _FILLERS



def _sentinel_index_of(q: Question) -> int | None:
    """Which option is the sentinel, or None. By membership in the grammar's output, not
    by a prefix test, because with opaque keys the description is all there is to go on."""
    for i, d in enumerate(q.descriptions or []):
        if sentinel.is_sentinel_description(d):
            return i
    for i, o in enumerate(q.options):
        if sentinel.is_sentinel_key(str(o)):
            return i
    return None


def churn_options(q: Question, rng: random.Random, drop: int, add: int) -> Question:
    """Drop `drop` wrong options, add `add` from other tasks, reshuffle.

    Withholding the correct option leaves, for a task with a fixed option vocabulary, a
    set that is recognisably "the usual list, minus one", so abstention would be readable
    without looking at the state. Dropping wrong options makes a vocabulary member going
    missing the normal case rather than the tell; adding options from other tasks keeps
    the option count from only ever falling (`filler_pool`, empty unless configured).

    Drawn **independently of the sentinel paths and applied to ordinary questions too**:
    churning only the withheld ones would trade a membership tell for an option-count one.
    """
    probs = q.target_probs()
    if probs is None or len(q.options) < 3:
        return q
    best = max(range(len(probs)), key=probs.__getitem__)
    # Only options carrying no meaningful mass may go, or dropping one moves the label --
    # and never the sentinel: a decoy sentinel carries zero mass, and churning it away
    # would undo the rate `decoy_none` exists to hold.
    si = _sentinel_index_of(q)
    spare = [i for i in range(len(q.options))
             if i != best and i != si and probs[i] <= 1e-9]
    drop = min(drop, max(0, len(spare) - 1), len(q.options) - 3)
    gone = set(rng.sample(spare, drop)) if drop > 0 else set()
    keep = [i for i in range(len(q.options)) if i not in gone]

    pool = filler_pool()
    bucket = _shape_bucket(q.options[0])
    have = {str(o).lower() for o in q.options}
    cands = [o for o in pool.get(bucket, ()) if str(o).lower() not in have]
    picked = rng.sample(cands, add) if len(cands) >= add > 0 else []
    if not gone and not picked:
        return q

    options = [q.options[i] for i in keep] + list(picked)
    target = [probs[i] for i in keep] + [0.0] * len(picked)
    descriptions = ([q.description_for(i) for i in keep] + [None] * len(picked)
                    if q.has_descriptions else None)
    total = sum(target)
    if total > 0:
        target = [t / total for t in target]
    order = list(range(len(options)))
    rng.shuffle(order)
    return Question(q.id, q.question, [options[i] for i in order], [target[i] for i in order],
                    q.meta, q.instructions,
                    [descriptions[i] for i in order] if descriptions else None)


_NUM_RE = re.compile(r"^-?\d+(\.\d+)?$")


def _shape_bucket(o) -> str:
    """The surface-shape bucket an option string falls in; the filler pool's keys."""
    s = str(o).strip()
    if _NUM_RE.match(s): return "numeric"
    n = len(s.split())
    if s.lower() in ("yes", "no", "true", "false", "unknown"): return "boolean"
    if s.isupper() and n <= 3: return "upper_short"
    if n == 1: return "word"
    if n <= 4: return "phrase"
    return "sentence"


P_FILLER = 0.45          # share of choice questions whose option set is churned
FILLER_DROP = (1, 2)     # wrong options removed
FILLER_ADD = (0, 3)      # options brought in from other tasks

def strip_criteria(q: Question) -> Question:
    """Drop the descriptions entirely, keeping the bare-option path trained."""
    if not q.descriptions:
        return q
    return Question(q.id, q.question, list(q.options), q.target, q.meta, q.instructions, None)


# Sentinel rates, chosen against a measured sweep: P(the sentinel is correct | a sentinel
# is present) should stay near 0.2 (real traffic rarely has the answer missing) while at
# most ~1/3 of choice questions carry a sentinel. At 0.060 / 0.36 that is about 0.17 and
# 30 %. Withheld rows are also what teach the scorer to *pick* the sentinel when the
# answer really is absent, so p_withheld has a floor and this is near it.
P_WITHHELD = 0.060
P_DECOY = 0.36


# Tasks whose option descriptions ARE the rule being applied -- a band's bounds, a
# routing condition, a gate policy, a matching policy, a semver definition, a coupling
# constraint, a ranking rule. Stripping them leaves a question whose target nobody could
# derive. The other augmentations keep every remaining option's text, so they still apply.
RULE_CRITERIA_PREFIXES = ("rule_", "sensor_", "toolgate_", "entityres_", "diff_",
                          "sched_", "relational_", "compose_")


def augment_criteria(example: Example, rng: random.Random,
                     p_opaque: float = 0.30, p_withheld: float = P_WITHHELD,
                     p_strip: float = 0.10, p_decoy: float = P_DECOY,
                     p_filler: float = P_FILLER,
                     split: str = "train") -> Example:
    """The criteria augmentations, applied per choice question.

    - churn (`p_filler`): drop wrong options (and optionally add some from other tasks);
    - withheld (`p_withheld`): drop the correct option, append a sentinel that is correct;
    - opaque keys (`p_opaque`): replace keys with `opt_0..opt_k`, keep the descriptions;
    - stripped (`p_strip`): drop the descriptions;
    - decoy (`p_decoy`, drawn independently): drop a wrong option, append a sentinel
      that is wrong.

    Withheld / opaque / stripped are mutually exclusive by construction -- one draw per
    question picks at most one -- so the rates are the stated rates and not their
    pairwise products. Not-in-context rows from the sources supply further abstention
    supervision without altering a choice question.
    """
    questions = []
    for q in example.questions:
        if _is_choice(q) and (q.meta or {}).get("ordered_rules"):
            # An ordered policy: rule k says "no lower-numbered rule matches", so
            # withholding, stripping, churning or decoying one rule's text leaves the
            # others undecidable. Opaque keys keep every rule's text, so that one alone
            # still applies.
            if rng.random() < p_opaque:
                q = opaque_keys(q)
            questions.append(q)
            continue
        if _is_choice(q):
            # Churn runs FIRST, before any sentinel exists: the sentinel must never be
            # churned away, so churning afterwards would churn sentinel-carrying
            # questions slightly less often and make the option count a weak tell.
            if rng.random() < p_filler:
                q = churn_options(q, rng, rng.randint(*FILLER_DROP),
                                  rng.randint(*FILLER_ADD))
            u = rng.random()
            if u < p_withheld:
                alt = withhold_criteria(q, rng, split)
                if alt is not None:
                    questions.append(alt)
                    continue
            elif u < p_withheld + p_opaque:
                q = opaque_keys(q)
            elif u < p_withheld + p_opaque + p_strip:
                if not example.task.startswith(RULE_CRITERIA_PREFIXES):
                    questions.append(strip_criteria(q))
                    continue
            # the decoy is drawn independently: it applies to a question that was not
            # withheld, and it is exactly the case the withheld draw never produces
            if rng.random() < p_decoy:
                alt = decoy_none(q, rng, split)
                if alt is not None:
                    q = alt
        questions.append(q)
    return Example(state=example.state, questions=questions, task=example.task)


def reorder_options(example: Example, how: str) -> Example:
    """Deterministic option reorderings, for the order-sensitivity check.

    Options are causal within a question block, so in principle the model could
    learn position rather than content; training-time shuffling should prevent
    that, and this is how we confirm it did.
    """
    if how == "given":
        return example
    questions = []
    for q in example.questions:
        order = list(range(len(q.options)))
        if how == "reversed":
            order.reverse()
        elif how == "rotated":
            order = order[1:] + order[:1]
        else:
            raise ValueError(f"unknown option order {how!r}")
        target = q.target
        if isinstance(target, int):
            target = order.index(target)
        elif isinstance(target, list):
            target = [target[i] for i in order]
        questions.append(Question(q.id, q.question, [q.options[i] for i in order], target, q.meta))
    return Example(state=example.state, questions=questions, task=example.task)


def example_shard(example: Example, n_heads: int) -> int:
    """Which pointer head owns this example, from a stable hash of its content.

    For a K-head ensemble: disjoint shards decorrelate the heads. A hash of the content
    and not of the row index, so the routing survives a reshuffle, a resume, a re-split
    and a corpus rebuild that reorders rows. `hash()` is not usable here -- it is
    randomised per process for strings.
    """
    if n_heads <= 1:
        return -1
    key = "\x1f".join([example.task, str(len(example.state)), example.state[:256]]
                      + [q.id for q in example.questions])
    return int.from_bytes(hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest(),
                          "big") % n_heads


class DecisionDataset(Dataset):
    def __init__(
        self,
        examples: Sequence[Example],
        packer: Packer,
        augment: bool = True,
        seed: int = 0,
        drop_questions_to_fit: bool = True,
        criteria_augment: bool = False,
        sentinel_split: str = "train",
        n_heads: int = 1,
        distill_alpha: float = 0.0,
    ) -> None:
        self.examples = list(examples)
        # Distillation: a question carrying `meta.teacher` (a larger model's distribution
        # over its options) trains on (1 - alpha) * gold + alpha * teacher. Mixed AFTER
        # augmentation, so the sentinel and withheld-criteria draws still see the gold
        # target they were calibrated on.
        self.distill_alpha = float(distill_alpha)
        self.packer = packer
        self.augment = augment
        self.criteria_augment = criteria_augment
        self.sentinel_split = sentinel_split
        self.seed = seed
        self.epoch = 0
        self.drop_questions_to_fit = drop_questions_to_fit
        self.n_heads = int(n_heads)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i: int) -> Packed | None:
        ex = self.examples[i]
        shard = example_shard(ex, self.n_heads)
        if self.augment:
            # The augmentation stream is a function of (seed, epoch, example), so a
            # resumed run sees exactly the draws it would have seen. For a K-head
            # ensemble the owning head is in the key too, so head k never sees the option
            # orders the other heads trained on.
            key = (self.seed, self.epoch, i) if self.n_heads <= 1 else \
                  (self.seed, self.epoch, i, shard)
            rng = random.Random(key.__hash__())
            teachers = ([teacher_by_option(q) for q in ex.questions]
                        if self.distill_alpha > 0 else None)
            if self.criteria_augment:
                ex = augment_criteria(ex, rng, split=self.sentinel_split)
            ex = shuffle_options(ex, rng)
            if teachers is not None:
                ex = mix_teacher(ex, teachers, self.distill_alpha)
        packed = self.packer.pack(ex, drop_questions_to_fit=self.drop_questions_to_fit)
        if packed is not None:
            packed.shard = shard
            packed.example_index = i
        return packed


def teacher_by_option(q: Question) -> dict[str, float] | None:
    t = (q.meta or {}).get("teacher")
    if not t or len(t) != len(q.options):
        return None
    return {str(o): float(p) for o, p in zip(q.options, t)}


def mix_teacher(example: Example, teachers: list, alpha: float) -> Example:
    """(1 - alpha) * gold + alpha * teacher, per question, where the teacher still applies.

    Matched by option key, after augmentation. A question whose options are no longer a
    subset of what the teacher scored -- a sentinel appended, options churned in from
    another task, keys made opaque -- keeps its gold target: the teacher said nothing
    about that option set.
    """
    out = []
    for q, t in zip(example.questions, teachers):
        gold = q.target_probs()
        if t is None or gold is None or not all(str(o) in t for o in q.options):
            out.append(q)
            continue
        tp = [t[str(o)] for o in q.options]
        z = sum(tp)
        if z <= 0:
            out.append(q)
            continue
        mixed = [(1 - alpha) * g + alpha * p / z for g, p in zip(gold, tp)]
        out.append(Question(q.id, q.question, list(q.options), mixed, q.meta,
                            q.instructions, q.descriptions))
    return Example(state=example.state, questions=out, task=example.task)


def _collate_drop_none(items, pad_id: int, mask_dtype: torch.dtype,
                       pad_to_tile: int = 1, dense_mask: bool = True):
    items = [i for i in items if i is not None]
    if not items:
        return None
    return collate(items, pad_id, mask_dtype, pad_to_tile, dense_mask)


def epoch_seed(seed: int, epoch: int) -> int:
    """Stable across processes and runs, unlike hash() on strings."""
    return (seed * 1_000_003 + epoch * 7_919 + 12345) % (2**32)


def token_lengths(examples: Sequence[Example], tokenizer, batch: int = 4096) -> np.ndarray:
    """Exact token counts of state + every question's text, options and criteria, plus a
    small per-question allowance for the template -- what the token budget must cap.

    `token_estimate` (characters / 3) can be off by several times on record-heavy or dense
    states, enough to break a token cap; tokenizing once costs a few minutes at startup.
    """
    texts = []
    for e in examples:
        parts = [e.state]
        for q in e.questions:
            parts.append(q.question)
            parts.extend(str(o) for o in q.options)
            if q.descriptions:
                parts.extend(str(d) for d in q.descriptions if d)
            if getattr(q, "instructions", None):
                parts.append(str(q.instructions))
        texts.append("\n".join(parts))
    out = np.zeros(len(texts), dtype=np.int64)
    for i in range(0, len(texts), batch):
        ids = tokenizer(texts[i: i + batch], add_special_tokens=False)["input_ids"]
        out[i: i + len(ids)] = [len(x) for x in ids]
    n_q = np.array([len(e.questions) for e in examples], dtype=np.int64)
    n_o = np.array([sum(len(q.options) for q in e.questions) for e in examples], dtype=np.int64)
    return out + 16 + 12 * n_q + 3 * n_o


def token_estimate(examples: Sequence[Example], chars_per_token: float = 3.0) -> np.ndarray:
    """`length_key` in approximate tokens, for the token budget. 3 characters a token
    over-estimates English prose (~4) and roughly matches JSON and code, so batches err on
    the small side rather than the out-of-memory side."""
    return (length_key(examples) / chars_per_token).astype(np.int64) + 16


def length_key(examples: Sequence[Example]) -> np.ndarray:
    """A cheap proxy for packed length: characters of state + questions + options.

    Used only to group examples of similar length into the same batch, so it
    needs no tokenizer and costs nothing at startup. It does not have to be
    exact — it only has to correlate with the packed token count.
    """
    return np.array(
        [len(e.state) + sum(len(q.question) + sum(len(o) for o in q.options)
                            for q in e.questions) for e in examples],
        dtype=np.int64)


class EpochBatchSampler(torch.utils.data.Sampler):
    """One deterministic permutation per epoch, chunked into batches.

    `skip_batches` is what makes resume exact: the same epoch permutation is
    rebuilt and the batches already consumed are dropped, so every example is
    still seen exactly once per epoch.

    With `bucket > 1` the permutation is cut into windows of
    `bucket * batch_size`, each window is sorted by `lengths`, batches are taken
    within the window, and the batch *order* is then shuffled again. Every batch
    is still a function of (seed, epoch) alone, so resume is unaffected and each
    example is still seen exactly once per epoch. Why it matters: `collate` pads
    to the longest member of a batch, so random batches are mostly padding.
    """

    def __init__(self, n: int, batch_size: int, seed: int, epoch: int, skip_batches: int = 0,
                 lengths: np.ndarray | None = None, bucket: int = 0,
                 max_tokens: int = 0) -> None:
        rng = np.random.default_rng(epoch_seed(seed, epoch))
        perm = rng.permutation(n)
        if lengths is not None and bucket > 1:
            window = batch_size * bucket
            batches: list[list[int]] = []
            for i in range(0, n, window):
                w = perm[i : i + window]
                w = w[np.argsort(lengths[w], kind="stable")]
                if max_tokens > 0:
                    # Token budget: a batch grows while (members x its longest member),
                    # the padded size collate will build, stays within max_tokens, up to
                    # batch_size members. Sorted ascending, the newest member is the
                    # longest. A 30k-token state then trains in a batch of ~3 and short
                    # examples still go batch_size at a time.
                    cur: list[int] = []
                    for j in w.tolist():
                        if cur and (len(cur) >= batch_size
                                    or (len(cur) + 1) * int(lengths[j]) > max_tokens):
                            batches.append(cur)
                            cur = []
                        cur.append(j)
                    if cur:
                        batches.append(cur)
                else:
                    batches += [w[j : j + batch_size].tolist()
                                for j in range(0, len(w), batch_size)]
            self.all_batches = [batches[i] for i in rng.permutation(len(batches))]
        else:
            self.all_batches = [perm[i : i + batch_size].tolist() for i in range(0, n, batch_size)]
        self.batches = self.all_batches[skip_batches:]

    def __iter__(self):
        return iter(self.batches)

    def __len__(self) -> int:
        return len(self.batches)

    @property
    def n_batches_total(self) -> int:
        return len(self.all_batches)


def make_loader(
    dataset: DecisionDataset,
    pad_id: int,
    mask_dtype: torch.dtype = torch.float32,
    batch_size: int = 8,
    batch_sampler: EpochBatchSampler | None = None,
    num_workers: int = 2,
    shuffle: bool = False,
    pad_to_tile: int = 1,
    dense_mask: bool = True,
    max_tokens: int = 0,
) -> DataLoader:
    """`max_tokens` > 0 (evaluation): batches in length-sorted order, each capped at
    max_tokens padded tokens as well as batch_size members, so long states cannot run the
    card out of memory. Order changes, results do not: consumers join by question ref."""
    fn = functools.partial(_collate_drop_none, pad_id=pad_id, mask_dtype=mask_dtype,
                           pad_to_tile=pad_to_tile, dense_mask=dense_mask)
    kwargs = dict(collate_fn=fn, num_workers=num_workers)
    if num_workers > 0:
        kwargs["persistent_workers"] = False
    if batch_sampler is None and max_tokens > 0 and not shuffle:
        est = token_lengths(dataset.examples, dataset.packer.tok)
        order = np.argsort(est, kind="stable").tolist()
        batches, cur = [], []
        for j in order:
            if cur and (len(cur) >= batch_size or (len(cur) + 1) * int(est[j]) > max_tokens):
                batches.append(cur)
                cur = []
            cur.append(j)
        if cur:
            batches.append(cur)
        return DataLoader(dataset, batch_sampler=batches, **kwargs)
    if batch_sampler is not None:
        return DataLoader(dataset, batch_sampler=batch_sampler, **kwargs)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)
