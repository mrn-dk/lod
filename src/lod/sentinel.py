"""The "not on this list" option, generated from a grammar instead of a list.

Why a grammar. An earlier corpus used one fixed sentinel, `none_of_the_above` /
"None of the above", and every such option in training was the correct answer, so the
phrase was perfectly predictive: the model trained on it answers it at p = 1.0000. The
next corpus replaced the one string with eight and added a decoy, and the bias shrank but
did not go: measured on the model trained on that corpus, holding the state and the two
real options fixed and changing only the third option's description,

    all eight shipped NONE_VARIANTS    win at 0.703 - 0.817, confidence 0.042 - 0.064
    five paraphrases of the same idea  lose at 0.159 - 0.232, confidence 0.179 - 0.330

Eight strings did not remove the memorised surface; it widened it eightfold. The model
learned the *category* of the shipped list, not the meaning of the sentence, and the
evidence for that is the 0.55 gap between strings it had seen and strings it had not.

So the wording is drawn from a grammar with ~25,000 distinct outputs, and the output
space is split by a hash of the string itself into a `train` part and a disjoint `eval`
part. Training never draws from the `eval` part, which is what lets the
`sentinel_wordings` gate measure generalisation to unseen wordings rather than recall of
seen ones. `split_of` classifies any string, including one that arrives over the wire, so
the gate can partition a dump it did not generate.
"""

from __future__ import annotations

import hashlib
import random
import re

# ---- vocabulary -------------------------------------------------------------------
# Slots are filled independently, so the space is the product. Every combination has to
# read as English on its own, which is what bounds the vocabularies: a word that only
# works in one frame goes in that frame, not here.

SUBJECTS = ("options", "choices", "answers", "labels", "categories",
            "alternatives", "entries", "candidates", "selections", "values")
SINGULAR = ("option", "choice", "answer", "label", "category",
            "alternative", "entry", "candidate", "value")
PLACE = ("listed", "above", "shown", "given", "provided", "offered",
         "on this list", "in this list", "here")
VERB = ("applies", "is correct", "fits", "matches", "is right",
        "is appropriate", "describes this", "is the answer")
CORRECT = ("correct", "right", "true", "actual", "appropriate")
AMONG = ("among", "one of", "included in", "present in", "covered by")
COVER = ("covered", "captured", "described", "represented", "matched")
SOMETHING = ("Something else", "Some other value", "Another answer",
             "A different option")
APPEAR = ("on this list", "above", "among these", "in the set given")
NONEWORD = ("None", "Not one")


def _f0(r): return (f"{r.choice(NONEWORD)} of the {r.choice(SUBJECTS)} "
                    f"{r.choice(PLACE)} {r.choice(VERB)}")
def _f1(r): return (f"The {r.choice(CORRECT)} {r.choice(SINGULAR)} is not "
                    f"{r.choice(AMONG)} the {r.choice(SUBJECTS)} {r.choice(PLACE)}")
def _f2(r): return (f"Not {r.choice(COVER)} by any of the {r.choice(SUBJECTS)} "
                    f"{r.choice(PLACE)}")
def _f3(r): return f"No {r.choice(SINGULAR)} {r.choice(PLACE)} {r.choice(VERB)}"
def _f4(r): return (f"The {r.choice(CORRECT)} {r.choice(SINGULAR)} does not appear "
                    f"{r.choice(APPEAR)}")
def _f5(r): return (f"{r.choice(SOMETHING)}, not {r.choice(AMONG)} the "
                    f"{r.choice(SUBJECTS)} {r.choice(PLACE)}")
def _f6(r): return f"{r.choice(SOMETHING)}, not {r.choice(PLACE)}"

# (frame, how many distinct strings it can produce). The frame is drawn in proportion to
# its own size, so the distribution over *strings* is uniform -- drawing the frame
# uniformly instead made the 36-string frame 1 wording in 7 and it recurred 132 times in
# 20,000 draws, which is the exact failure the grammar exists to avoid.
FRAMES = (
    (_f0, len(NONEWORD) * len(SUBJECTS) * len(PLACE) * len(VERB)),
    (_f1, len(CORRECT) * len(SINGULAR) * len(AMONG) * len(SUBJECTS) * len(PLACE)),
    (_f2, len(COVER) * len(SUBJECTS) * len(PLACE)),
    (_f3, len(SINGULAR) * len(PLACE) * len(VERB)),
    (_f4, len(CORRECT) * len(SINGULAR) * len(APPEAR)),
    (_f5, len(SOMETHING) * len(AMONG) * len(SUBJECTS) * len(PLACE)),
    (_f6, len(SOMETHING) * len(PLACE)),
)
_CUM = []
_acc = 0
for _fn, _n in FRAMES:
    _acc += _n
    _CUM.append(_acc)
TOTAL_STRINGS = _acc


def _frames(rng: random.Random) -> str:
    """One description, uniform over the whole string space."""
    u = rng.randrange(TOTAL_STRINGS)
    for (fn, _), edge in zip(FRAMES, _CUM):
        if u < edge:
            return fn(rng)
    return FRAMES[-1][0](rng)


# ---- keys -------------------------------------------------------------------------
# The key is drawn independently of the description, so the pair space is larger than
# either. It is a small, enumerable set on purpose: `KEYS` is what `is_sentinel_key`
# tests against, and a test that has to parse a key would be a second grammar to keep in
# step with the first. The key alone is never the whole signal -- 30 % of choice
# questions replace every key with `opt_N`, the sentinel included.

def _snake(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def _key_set() -> frozenset[str]:
    out = {"none_of_the_above"}                       # the wire contract's own spelling
    for s in SUBJECTS:
        out.add(f"none_of_the_{s}")
        out.add(f"outside_the_{s}")
        out.add(f"no_match_in_{s}")
    for g in SINGULAR:
        out.add(f"other_{g}")
        out.add(f"unlisted_{g}")
        out.add(f"{g}_not_listed")
        for v in VERB:
            out.add(f"no_{g}_{_snake(v)}")
    for c in COVER:
        out.add(f"not_{c}")
    for c in CORRECT:
        for g in SINGULAR:
            out.add(f"{c}_{g}_absent")
    return frozenset(out)


KEYS = _key_set()
_KEYS_SORTED = tuple(sorted(KEYS))

# what the two earlier corpora shipped. Kept so a checkpoint or a dump from those runs is
# still classified as carrying a sentinel, and so the wire contract's spelling still is.
LEGACY_KEYS = frozenset({
    "none_of_the_above", "none_of_these", "no_match", "not_listed", "unlisted",
    "none_apply", "no_option_applies", "other_none",
})
LEGACY_DESCRIPTIONS = frozenset({
    "None of the above", "None of these applies", "No listed option matches",
    "The correct answer is not among the options listed", "Not any of the options given",
    "None of the given options apply", "No option above is correct",
    "Something else, not listed above",
})

NONE_OF_THE_ABOVE = "none_of_the_above"
NONE_OF_THE_ABOVE_DESC = "None of the above"

# share of the wording space reserved to `eval` and never drawn in training
EVAL_SHARE = 0.20


def split_of(description: str) -> str:
    """`train` or `eval` for any wording, from a hash of the string itself.

    Partitioning the *output space* rather than the generator means the two halves stay
    disjoint however either side draws, and a gate can classify a wording it was handed
    rather than one it produced. Legacy strings are forced to `eval`: the shipped
    checkpoints memorised them, so they are the one set a new model must not be
    measured on as if it were fresh.
    """
    if description in LEGACY_DESCRIPTIONS:
        return "eval"
    h = int.from_bytes(hashlib.sha256(description.strip().encode()).digest()[:8], "big")
    return "eval" if (h % 1000) < EVAL_SHARE * 1000 else "train"


def sample(rng: random.Random, split: str = "train",
           opaque_key: str | None = None) -> tuple[str, str]:
    """-> (key, description) drawn from `split`'s half of the space.

    `opaque_key` is passed when the question's own keys have been replaced by `opt_N`:
    the sentinel then takes the next `opt_N` rather than a readable key, because being
    the one readable key in a list of opaque ones is itself a tell, and a tell that
    survives every wording change.
    """
    for _ in range(64):
        desc = _frames(rng)
        if split_of(desc) == split:
            break
    else:                                   # unreachable in practice; never loop forever
        desc = _frames(rng)
    key = opaque_key if opaque_key is not None else rng.choice(_KEYS_SORTED)
    return key, desc


def is_sentinel_key(key: str) -> bool:
    return str(key) in KEYS or str(key) in LEGACY_KEYS


def space_size() -> int:
    """Distinct descriptions the grammar can produce. Reported by the corpus audit."""
    return TOTAL_STRINGS


def all_descriptions() -> frozenset[str]:
    """Every string the grammar can produce, enumerated.

    24,804 strings is small enough to materialise, and an exact membership test is worth
    a lot more than a prefix heuristic: the first version of the corpus audit recognised
    a sentinel by its opening words, and "No " matched every real `no` option in the
    corpus, which reported 152 repeats of a wording the grammar never emitted.
    """
    out = set()
    for noneword in NONEWORD:
        for s in SUBJECTS:
            for pl in PLACE:
                for v in VERB:
                    out.add(f"{noneword} of the {s} {pl} {v}")
    for c in CORRECT:
        for g in SINGULAR:
            for a in AMONG:
                for s in SUBJECTS:
                    for pl in PLACE:
                        out.add(f"The {c} {g} is not {a} the {s} {pl}")
    for cv in COVER:
        for s in SUBJECTS:
            for pl in PLACE:
                out.add(f"Not {cv} by any of the {s} {pl}")
    for g in SINGULAR:
        for pl in PLACE:
            for v in VERB:
                out.add(f"No {g} {pl} {v}")
    for c in CORRECT:
        for g in SINGULAR:
            for ap in APPEAR:
                out.add(f"The {c} {g} does not appear {ap}")
    for so in SOMETHING:
        for a in AMONG:
            for s in SUBJECTS:
                for pl in PLACE:
                    out.add(f"{so}, not {a} the {s} {pl}")
        for pl in PLACE:
            out.add(f"{so}, not {pl}")
    return frozenset(out)


DESCRIPTIONS = all_descriptions()


def is_sentinel_description(d: str | None) -> bool:
    return bool(d) and (d in DESCRIPTIONS or d in LEGACY_DESCRIPTIONS)
