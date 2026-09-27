"""Lichess puzzle adapter (row 13: themes trained, eight reserved to testreal, two to devreal).

Row 13 used to be reserved whole to `testreal`. That made the one domain whose state has
no natural-language shortcut a domain the model had never seen, so the puzzles could not
be generalising anything; the source cap then threw away 32,950 of the 40,000 questions
because a whole reserved family was 53 % of `testreal`.

Now the themes are split the way `synth/rules.py` splits rule combinations: hold out a
*structure*, not a sample, and pair every held-out theme against a trained one, so a drop
on the held-out side is a failure to generalise over structure rather than noise.

    mateIn1    -> mateIn2                 depth
    short      -> long, veryLong          length
    endgame    -> rookEndgame             specialisation
    fork       -> pin, sacrifice          motif
    (trained)  -> defensiveMove, kingsideAttack

The dev split holds out two more themes the same way, disjoint from both (`DEV_THEMES`):

    endgame    -> middlegame              phase
    crushing   -> advantage               size of the edge

Every theme this cache can support is already a task, so there is no unused theme to
define: these two are moved out of training, each with its partner still trained. They
are what checkpoint selection and the temperature fit read, so dev measures transfer to an
unseen theme the way test does, rather than recall of a trained one.

Two things this split forces:

- **Disjoint puzzle pools.** Every theme task asks about the same 4,000 puzzles, so a
  trained task and a reserved task would otherwise share states, and `dedup_against_train`
  -- doing exactly its job -- would delete the reserved eval side. The rows are
  partitioned by a hash of `PuzzleId`: trained tasks draw from one half, reserved tasks
  from the other, and no state ever crosses the split. The dev themes take a fifth of
  the trained half (`_bucket`), so the reserved half -- and every testreal state -- is
  exactly what it was.
- **Real balance.** The old loader padded a task out to its quota with negatives once the
  positives ran out, so `lichess_theme_pin` shipped at 11 % yes and answering "no" scored
  89 % on it. A task now stops at `2 * min(positives, negatives)`: fewer questions, but
  every one of them measures something.

Three things the previous version got wrong, all visible in the built corpus:

- it looked for parquet under `<raw>/huggingface/Lichess/chess-puzzles/data`, which does
  not exist here, so `tasks()` returned nothing and the *generic* Hub adapter claimed the
  rows instead. What reached the model was "What is the Popularity of this text?" over 82
  unsorted integers, given a FEN.
- the theme loader only yielded puzzles that *have* the theme and set `target=1`, so
  every example of every theme task was `yes`. A constant label teaches the prior.
- the state included `themes`, so the answer to the theme question was printed in the
  question's own input.

Here the state is the position and the solution, the themes are held out of it, and each
theme task is balanced against puzzles the theme does not apply to.

A fourth thing the first version of this split got wrong, found in an earlier build:
`master` was a trained task. Lichess's `master` theme records that the *game* the puzzle
came from was played by titled players, which is nowhere in the FEN, the solution or the
opening tag -- 588 trained questions asking for a fact the state does not contain. It is
in `EXCLUDED_THEMES` now, with `opening`, whose answer the state does print.

A fifth: the theme criteria were a restatement of the question for eight of the
nineteen themes, so the only thing carrying what "rookEndgame" or "kingsideAttack" means
was the label word itself -- the memorised-theme-word model row 43 exists to avoid.
`THEME_CRITERIA` holds lichess's published definition of every theme this module ships,
and `tasks()` refuses to ship one it has no definition for.

A sixth and seventh, from a domain-4 audit: six of those definitions said "the side to
move" about a FEN whose side to move is the *opponent* (lichess stores the position before
the opponent's setup move), so the criteria named the wrong side and counted the wrong
moves; they say "the solver" now and every question carries `STATE_INSTRUCTIONS` saying
who that is. And 36 puzzles lichess never finished tagging (a phase tag and nothing else)
were supplying "no" answers they had never been labelled with; `_fully_tagged` keeps them
out of every pool.
"""

from __future__ import annotations

import ast
import hashlib
import json
from collections import Counter
from typing import Iterator

from lod.schema import Example, Question
from lod.corpus.repositories import raw_store as store
from lod.corpus.services.sources.real.base import RealTask

KEY = "lichess_chess_puzzles"
PATH = "Lichess/chess-puzzles"
LICENCE = "CC0-1.0"
URL = f"https://huggingface.co/datasets/{PATH}"
THEME_LIMIT = 24
# Never trained on: forced to `testreal`, paired against a trained theme (see the module
# docstring). `tasks()` raises if the theme list stops producing one of these.
RESERVED_THEMES = ("mateIn2", "long", "veryLong", "rookEndgame", "pin", "sacrifice",
                   "defensiveMove", "kingsideAttack")
# Never trained on and never tested: forced to `devreal`, each paired against a trained
# theme on the same axis (see the module docstring), on a puzzle pool of their own.
DEV_THEMES = ("middlegame", "advantage")
DEV_PAIRS = (("endgame", "middlegame"), ("crushing", "advantage"))
# Themes that reach the top of the count but must never become a task.
#
# `master` ("puzzles from games played by titled players") is *provenance*: who played
# the game is not in the FEN, not in the solution and not in the opening tag, so no code
# could derive it from the state and neither can a reader. It shipped 588 trained
# questions in an earlier build; a depth-3 tree over ten shallow state features reaches
# 0.587 on it against a 0.500 label-permuted floor, which is a confound, not an answer. The
# corpus rule for the seeded rows -- "a scenario whose answer code cannot derive is dropped
# rather than guessed" -- is the same rule here.
#
# `opening` is the opposite failure: `_state` carries `OpeningTags`, and 96.8 % of
# opening-themed puzzles in the cache have one against 22.3 % of the pool, so the answer
# would be printed in the question's own input. It sits at 187 rows in the current cache
# and is excluded by the >= 200 floor by luck alone; name it so a larger fetch cannot
# quietly ship it.
UNANSWERABLE_THEMES = ("master", "masterVsMaster")
STATED_IN_STATE_THEMES = ("opening",)
# `oneMove` carries mateIn1's label on this cache (303 of 304 one-move puzzles are mate):
# a second task with the same answers teaches nothing the first does not.
DUPLICATE_THEMES = ("oneMove",)
EXCLUDED_THEMES = UNANSWERABLE_THEMES + STATED_IN_STATE_THEMES + DUPLICATE_THEMES

# Lichess's own published definition of each theme, as the option criteria. The enricher
# left eight of these as a restatement of the question ("The position and solution do not
# center on a rook endgame theme"), which makes the criteria decoration: a model that has
# to be *told* what a rook endgame is and then check the board is the bet this corpus is
# making, exactly as `chessfacts.GLOSSARY` states it for row 43. A theme this table has
# no entry for cannot be shipped: `tasks()` raises rather than fall back to a restatement.
THEME_CRITERIA: dict[str, str] = {
    "advancedPawn": "one of the pawns is deep into the opponent's half, close enough to "
                    "promotion to matter",
    "advantage": "the solution seizes a clear edge worth roughly two to six pawns, "
                 "short of an outright win",
    "crushing": "the solution punishes a blunder and leaves a crushing, game-ending "
                "advantage",
    "defensiveMove": "the solution needs a precise move that avoids losing material or "
                     "an advantage, rather than one that wins something",
    "endgame": "the position is in the last phase of the game, with few pieces left "
               "beside pawns and kings",
    "fork": "one move of the solution attacks two or more enemy pieces at once",
    "kingsideAttack": "the solution attacks the enemy king on the king's side of the "
                      "board, where it castled",
    "long": "the solver needs three of its own moves, with the opponent replying "
            "between them, before the puzzle is finished",
    # worded without a bare "no": the criteria print for both options, so a "no" inside
    # one makes the string present in every yes/no question and pollutes any string-match
    # measurement with a cue that carries nothing (row 42 read 4.7 % that way, purely
    # from "no" sitting inside "nothing")
    "mate": "the last move of the solution ends the game in checkmate, leaving the "
            "losing king attacked and unable to answer",
    "mateIn1": "the solver plays one move and it is immediate checkmate; the opponent "
               "never gets a reply",
    "mateIn2": "the solver forces checkmate with its second move, whatever the opponent "
               "answers in between",
    "middlegame": "the position is in the second phase of the game, after development "
                  "and before the endgame",
    "oneMove": "the solver plays exactly one move and the puzzle is over; the "
               "opponent's reply, if any, does not count",
    "pin": "the solution turns on a pin: a piece that cannot move without exposing a "
           "more valuable piece behind it on the same line",
    "rookEndgame": "the position is an endgame in which the only pieces beside the kings "
                   "and pawns are rooks",
    "sacrifice": "the solution gives up material for a forced sequence that wins it back "
                 "or better",
    "short": "the solver needs two of its own moves, with the opponent replying "
             "between them, before the puzzle is finished",
    "veryLong": "the solver needs four or more of its own moves before the puzzle "
                "is finished",
}

# Who "the solver" is. A lichess puzzle's FEN is the position *before* the opponent's
# last move, and `Moves` starts with that move, so the side to move in the FEN is the side
# that does NOT solve the puzzle. Six of the criteria above used to say "the side to move
# needs three of its own moves" -- read against the FEN that names the wrong side, and
# counting from the first listed move gives the wrong length. (domain-4 audit: 5,074 of
# 16,891 questions carried that wording.) The criteria now say "the solver" and this says
# who that is.
STATE_INSTRUCTIONS = (
    "The state is a lichess puzzle. `fen` is the position before the opponent's last "
    "move, and `solution_moves` (long algebraic: origin square, destination square, "
    "promotion letter) lists that opponent move first and then the puzzle's solution, "
    "alternating: the solver's move, the opponent's reply, and so on. The solver is the "
    "side that is NOT to move in the FEN; the solver's moves are the 2nd, 4th, 6th, ... "
    "entries.")

THEME_TEMPLATE = "Does this chess puzzle have the '{theme}' theme?"
RATING_TEMPLATE = "How hard is this chess puzzle for the players who have attempted it?"

# Every puzzle lichess has finished tagging carries one length theme (oneMove / short /
# long / veryLong); 36 of the 4,000 cached carry only a phase tag (`['endgame']`). Their
# missing themes are not "no", they are "never tagged": 8 of them were `short = no`
# questions about two-move puzzles, one a `mate = no` about a mating line. They are out of
# every pool, so no negative is drawn from a puzzle nobody labelled.
LENGTH_THEMES = frozenset({"oneMove", "short", "long", "veryLong"})

RATING_OPTIONS = ["under_1200", "1200_to_1599", "1600_to_1999", "2000_to_2399", "2400_plus"]
RATING_DESC = [
    "rated below 1200: a one-move tactic, usually a direct capture or mate",
    "rated 1200-1599: a short forcing sequence a club player finds quickly",
    "rated 1600-1999: needs a quiet or intermediate move to be seen first",
    "rated 2000-2399: several plausible tries, only one of which works",
    "rated 2400 and above: long or counter-intuitive; expert difficulty",
]
_ROWS: list[dict] | None = None


def _themes_of(row: dict) -> list[str]:
    """`Themes` is stored as the *repr* of a Python list, not as JSON."""
    raw = row.get("Themes")
    if isinstance(raw, list):
        return [str(t) for t in raw]
    if not isinstance(raw, str) or not raw.strip():
        return []
    try:
        parsed = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return raw.split()
    return [str(t) for t in parsed] if isinstance(parsed, (list, tuple)) else []


def _rows() -> list[dict]:
    global _ROWS
    if _ROWS is None:
        _ROWS = list(store.load(KEY)) if store.has(KEY) else []
    return _ROWS


def _state(row: dict) -> str:
    """The position and its solution. Themes and rating are the labels, so they are out."""
    return json.dumps({
        "fen": row.get("FEN"),
        "solution_moves": row.get("Moves"),
        "opening": None if row.get("OpeningTags") in (None, "None", "") else row.get("OpeningTags"),
    }, ensure_ascii=False, indent=2)


def _top_themes() -> list[str]:
    counts = Counter(theme for row in _rows() for theme in _themes_of(row)
                     if theme not in EXCLUDED_THEMES)
    # a theme needs enough negatives as well as positives to be worth asking about
    return [t for t, n in counts.most_common(THEME_LIMIT) if n >= 200]


def _reserved_pool(row: dict) -> bool:
    """Which half of the puzzles a row belongs to, stable under rebuilds.

    Trained and reserved tasks must never share a state: the reserved side is evaluated
    against a model trained on the other side, and a shared FEN is both a leak and, once
    `dedup_against_train` sees it, a deletion of the eval row.
    """
    key = str(row.get("PuzzleId") or row.get("FEN") or "")
    return hashlib.sha256(key.encode()).digest()[0] % 2 == 1


def _bucket(row: dict) -> str:
    """`testreal`, `devreal` or `train`: which theme tasks may use this puzzle.

    The reserved half is `_reserved_pool` unchanged. A fifth of the other half, by a
    second byte of the same hash, is the dev themes' own pool: a dev puzzle in the trained
    pool would be the state of some trained theme's question, and `dedup_against_train`
    would delete it from dev.
    """
    if _reserved_pool(row):
        return "testreal"
    key = str(row.get("PuzzleId") or row.get("FEN") or "")
    return "devreal" if hashlib.sha256(key.encode()).digest()[1] % 5 == 0 else "train"


def _fully_tagged(row: dict) -> bool:
    return bool(LENGTH_THEMES & set(_themes_of(row)))


def _pool(which: bool | str) -> list[dict]:
    """The rows a split's tasks draw from. `True`/`False` are the reserved and trained
    pools, as before; `"devreal"` is the dev themes' fifth of the trained half."""
    want = {True: "testreal", False: "train"}.get(which, which)
    return [r for r in _rows() if _bucket(r) == want and _fully_tagged(r)]


def _split_of_theme(theme: str) -> str:
    return ("testreal" if theme in RESERVED_THEMES else
            "devreal" if theme in DEV_THEMES else "train")


_BANK = None


def _phrase(template_id: str, template: str, key: str) -> str:
    """The question wording for one record, from the phrasing bank (loaded once)."""
    global _BANK
    if _BANK is None:
        from lod.phrasings import PhrasingBank
        _BANK = PhrasingBank.load()
    return _BANK.pick(template_id, template, key)


def _theme_criteria(theme: str) -> list[str]:
    """The `no` / `yes` criteria for a theme, from its published definition."""
    what = THEME_CRITERIA[theme]
    return [f"This puzzle is not a '{theme}' puzzle: it is not the case that {what}.",
            f"This puzzle is a '{theme}' puzzle: {what}."]


def _theme_loader(theme: str):
    split = _split_of_theme(theme)
    descs = _theme_criteria(theme)

    def load(n: int) -> Iterator[Example]:
        rows = _pool(split)
        pos = [r for r in rows if theme in _themes_of(r)]
        neg = [r for r in rows if theme not in _themes_of(r)]
        # balanced by construction *and* by size: padding the quota out with negatives
        # once the positives run out is how the old build shipped an 11 %-yes task, on
        # which answering "no" scored 89 %.
        each = min(len(pos), len(neg), n // 2)
        made = 0
        for yes_row, no_row in zip(pos[:each], neg[:each]):
            for row, want_yes in ((yes_row, True), (no_row, False)):
                question = _phrase("d04.lichess_theme", THEME_TEMPLATE,
                                   f"{theme}:{row.get('PuzzleId')}").format(theme=theme)
                yield Example(task=f"lichess_theme_{theme}", state=_state(row), questions=[
                    Question(id=theme, question=question,
                             options=["no", "yes"], target=int(want_yes),
                             instructions=STATE_INSTRUCTIONS,
                             descriptions=list(descs))])
                made += 1
                if made >= n:
                    return
    return load


def _rating_loader():
    def load(n: int) -> Iterator[Example]:
        made = 0
        for row in _pool("train"):
            try:
                rating = int(row.get("Rating"))
            except (TypeError, ValueError):
                continue
            level = (0 if rating < 1200 else 1 if rating < 1600 else
                     2 if rating < 2000 else 3 if rating < 2400 else 4)
            yield Example(task="lichess_rating_bucket", state=_state(row), questions=[
                Question(id="rating_bucket",
                         question=_phrase("d04.lichess_rating", RATING_TEMPLATE,
                                          str(row.get("PuzzleId"))),
                         options=list(RATING_OPTIONS), target=level,
                         instructions=STATE_INSTRUCTIONS,
                         descriptions=list(RATING_DESC))])
            made += 1
            if made >= n:
                return
    return load


def tasks() -> list[RealTask]:
    if not _rows():
        return []
    themes = _top_themes()
    # the theme list is decided at load time from whatever rows are cached, so a reserved
    # theme can silently stop being produced. That would quietly move the holdout, which
    # is worse than a failed build: fail loudly instead.
    missing = [t for t in RESERVED_THEMES + DEV_THEMES if t not in themes]
    if missing:
        raise ValueError(
            f"lichess: reserved themes {missing} are not among the {len(themes)} themes "
            f"this cache produces ({themes}). The trained/reserved split is a stated "
            "holdout, not a sample: fetch more puzzles or amend RESERVED_THEMES / "
            "DEV_THEMES.")
    undefined = [t for t in themes if t not in THEME_CRITERIA]
    if undefined:
        raise ValueError(
            f"lichess: no published definition for themes {undefined}. The criteria are "
            "the definition the model applies, not decoration: add them to "
            "THEME_CRITERIA, or put the theme in EXCLUDED_THEMES if its answer is not "
            "in the state.")
    out = []
    notes = {
        "testreal": "Lichess published puzzle theme, balanced; NEVER trained on -- the "
                    "held-out structure the reserved_families gate reads",
        "devreal": "Lichess published puzzle theme, balanced; held out of train and "
                   "test; a dev structure -- selects checkpoints and fits T",
        "train": "Lichess published puzzle theme, balanced; trained",
    }
    for theme in themes:
        split = _split_of_theme(theme)
        out.append(RealTask(
            row=13, name=f"lichess_theme_{theme}", licence=LICENCE, url=URL,
            load=_theme_loader(theme), force_split=split, notes=notes[split]))
    out.append(RealTask(row=13, name="lichess_rating_bucket", licence=LICENCE, url=URL,
                        load=_rating_loader(), ordinal=True, force_split="train",
                        notes="Lichess puzzle rating, five documented difficulty bands; "
                              "trained"))
    return out
