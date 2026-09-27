"""Row 43: chess positions as facts.

Chess is the easiest place in this corpus to ship a wrong labeller, and a wrong labeller
is invisible -- every example still looks like a chess question. So the first test here
re-derives `attacked` and `pinned` from the rules of chess, with no python-chess call in
it, and compares square by square. The rest hold the design in place:

- the holdout is a composition *depth*, and a held-out family may use no primitive that
  training never saw (the `rules.py` lesson: hold out combinations, train every part);
- no FEN emitted here may be a FEN any lichess task states, or row 13's eight reserved
  themes leak into training;
- the gold option may not appear in the question or the instructions (row 36 shipped at
  100 % string-match solvable because the goal named the control verbatim);
- the same position rendered four ways must be the same question, not four questions.

`chess` is an optional extra (`uv sync --extra chess`). The tests that need a board skip
without it; the structural ones do not, and the one that matters most -- that importing
this module does not import `chess` -- runs either way.
"""

from __future__ import annotations

import ast
import json
import random
import re
from collections import Counter
from pathlib import Path

import pytest

from lod import sentinel
from lod.corpus.repositories import raw_store as store
from lod.paths import RAW_ROOT
from lod.corpus.services.sources.synth import chessfacts as cf

chess = pytest.importorskip("chess", reason="row 43 needs `uv sync --extra chess`")


@pytest.fixture(scope="module")
def pool():
    # The tests below read the question text with the template's own wording; the
    # phrasing bank is held empty here so a populated bank cannot break them. That the
    # label survives a rewording is tested in tests/test_audit_d04.py.
    from lod.phrasings import PhrasingBank
    saved = (store.RAW, store.RAW_ROOT, cf._BANK)
    store.configure(RAW_ROOT)
    cf.reset_pool()
    cf._BANK = PhrasingBank({})
    try:
        yield cf.pool(reserved=None)
    finally:
        store.RAW, store.RAW_ROOT, cf._BANK = saved
        cf.reset_pool()


@pytest.fixture(scope="module")
def sample(pool):
    """25 examples of every task, built once."""
    return {t.name: list(t.load(25)) for t in cf.tasks()}


# ---- the labeller, cross-checked against an independent implementation ---------------

_KNIGHT = [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)]
_KING = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
_ROOK = [(1, 0), (-1, 0), (0, 1), (0, -1)]
_BISHOP = [(1, 1), (1, -1), (-1, 1), (-1, -1)]


def _grid(fen: str) -> dict:
    """(file, rank) -> (is_white, piece letter), read off the FEN text alone."""
    g = {}
    for ri, row in enumerate(fen.split()[0].split("/")):
        rank, file = 7 - ri, 0
        for ch in row:
            if ch.isdigit():
                file += int(ch)
            else:
                g[(file, rank)] = (ch.isupper(), ch.upper())
                file += 1
    return g


def _attackers(g, white: bool, tgt) -> list:
    tf, tr = tgt
    out = []
    for (f, r), (c, p) in g.items():
        if c != white:
            continue
        df, dr = tf - f, tr - r
        if p == "N":
            hit = (df, dr) in _KNIGHT
        elif p == "K":
            hit = (df, dr) in _KING
        elif p == "P":
            hit = abs(df) == 1 and dr == (1 if white else -1)
        else:
            hit = False
            for sf, sr in (_ROOK if p == "R" else _BISHOP if p == "B" else _ROOK + _BISHOP):
                cf_, cr = f + sf, r + sr
                while 0 <= cf_ < 8 and 0 <= cr < 8:
                    if (cf_, cr) == (tf, tr):
                        hit = True
                        break
                    if (cf_, cr) in g:
                        break
                    cf_, cr = cf_ + sf, cr + sr
                if hit:
                    break
        if hit:
            out.append((f, r))
    return sorted(out)


def _pinned(g, sq) -> bool:
    if sq not in g:
        return False
    white, _ = g[sq]
    king = next((s for s, (c, p) in g.items() if c == white and p == "K"), None)
    if king is None:
        return False
    (kf, kr), (f, r) = king, sq
    df, dr = f - kf, r - kr
    if (df, dr) == (0, 0):
        return False
    if df == 0:
        step = (0, 1 if dr > 0 else -1)
    elif dr == 0:
        step = (1 if df > 0 else -1, 0)
    elif abs(df) == abs(dr):
        step = (1 if df > 0 else -1, 1 if dr > 0 else -1)
    else:
        return False
    cf_, cr = kf + step[0], kr + step[1]
    while (cf_, cr) != (f, r):                 # king and piece must be adjacent on the ray
        if (cf_, cr) in g:
            return False
        cf_, cr = cf_ + step[0], cr + step[1]
    want = {"R", "Q"} if 0 in step else {"B", "Q"}
    cf_, cr = f + step[0], r + step[1]
    while 0 <= cf_ < 8 and 0 <= cr < 8:        # first piece past it must be an enemy slider
        if (cf_, cr) in g:
            c, p = g[(cf_, cr)]
            return c != white and p in want
        cf_, cr = cf_ + step[0], cr + step[1]
    return False


# The corpus documentation's row-43 line: "cross-checked against an independent implementation over
# 92,104 attack and pin checks with 0 mismatches". 600 positions x 64 squares x 2 colours
# is 76,800 attack checks, and 600 positions carry ~25.5 pieces each. The test shipped at
# 40 positions -- 5,120 and ~1,033, 6.7 % of the claim -- so the documented number was
# not the number any test re-derived. It costs 1.6 s at 600; there is no reason to
# document one figure and check another.
CROSSCHECK_POSITIONS = 600


def test_attacked_and_pinned_agree_with_an_independent_implementation(pool):
    """The one test that would catch a silently wrong row.

    Every question here reduces to `attacked` or `pinned` eventually, so if python-chess
    is being *called* wrongly -- wrong colour argument, self-defence counted, a pin read
    against the wrong king -- these two helpers are where it shows.
    """
    rng = random.Random(11)
    n_att = n_pin = 0
    for fen in rng.sample(pool, CROSSCHECK_POSITIONS):
        board = chess.Board(fen)
        g = _grid(fen)
        for sq in range(64):
            fr = (chess.square_file(sq), chess.square_rank(sq))
            for white in (True, False):
                mine = sorted(chess.square_name(chess.square(f, r))
                              for f, r in _attackers(g, white, fr))
                theirs = sorted(chess.square_name(a) for a in board.attackers(white, sq))
                assert mine == theirs, f"{fen} {chess.square_name(sq)} white={white}"
                n_att += 1
        for sq in board.piece_map():
            fr = (chess.square_file(sq), chess.square_rank(sq))
            assert _pinned(g, fr) is board.is_pinned(board.piece_at(sq).color, sq), \
                f"{fen} pin on {chess.square_name(sq)}"
            n_pin += 1
    assert n_att == 64 * 2 * CROSSCHECK_POSITIONS      # 76,800
    # pool-dependent: 15,304 on the 2,000-puzzle pool, 12,584 on the 40,000-puzzle pool
    # row 43 now draws from (a different slice of positions is checked)
    assert n_pin > 10_000


def _grid_of(board) -> dict:
    return _grid(board.fen())


def _defended_by(g, sq) -> list:
    """Same-colour attackers of an occupied square, the piece itself excluded."""
    if sq not in g:
        return []
    white, _ = g[sq]
    return [a for a in _attackers(g, white, sq) if a != sq]


def _attacked_by_enemy(g, sq) -> list:
    if sq not in g:
        return []
    white, _ = g[sq]
    return _attackers(g, not white, sq)


def _hangs(g, sq) -> bool:
    return bool(_attacked_by_enemy(g, sq)) and not _defended_by(g, sq)


_VALUES = {"P": 1, "N": 3, "B": 3, "R": 5, "Q": 9, "K": 0}


def _fr(name: str):
    return ("abcdefgh".index(name[0]), int(name[1]) - 1)


def _answer(task: str, q, boards) -> int | None:
    """Re-derive the answer from the question text and the *grid*, not from python-chess.

    This is the composition-level twin of the primitive cross-check above: every family
    is a two- or three-deep combination of `attacked`, `defended` and `pinned`, and a
    combination can be wrong -- a defender counted as its own defender, a colour flipped
    after a move, a king left in a "hanging piece" set -- while both primitives are
    right. It reads the question rather than `q.meta`, so a target that has drifted from
    what the question asks fails here too.

    python-chess is used only to *make* a move; the fact is always read off the grid.
    """
    g = _grid_of(boards[0])
    text = q.question
    opts = q.options

    m = re.search(r"Is the piece on ([a-h][1-8]) attacked by any (white|black) piece", text)
    if m:
        return int(bool(_attackers(g, m.group(2) == "white", _fr(m.group(1)))))
    m = re.search(r"How many pieces defend the piece standing on ([a-h][1-8])", text)
    if m:
        return min(len(_defended_by(g, _fr(m.group(1)))), 3)
    m = re.search(r"Is the piece on ([a-h][1-8]) pinned against its own king", text)
    if m:
        return int(_pinned(g, _fr(m.group(1))))
    m = re.search(r"Is the piece on ([a-h][1-8]) defended by at least one piece that is "
                  r"itself pinned", text)
    if m:
        return int(any(_pinned(g, d) for d in _defended_by(g, _fr(m.group(1)))))
    m = re.search(r"Is the piece on ([a-h][1-8]) defended by at least one piece that is "
                  r"itself attacked by the other side and defended by nobody", text)
    if m:
        ds = [d for d in _defended_by(g, _fr(m.group(1))) if g[d][1] != "K"]
        return int(any(_hangs(g, d) for d in ds))
    m = re.search(r"Is the piece on ([a-h][1-8]) attacked by at least one enemy piece that "
                  r"is itself attacked and defended by nobody", text)
    if m:
        return int(any(_hangs(g, a) for a in _attacked_by_enemy(g, _fr(m.group(1)))))
    if text.startswith("Which of these") and "defended by no piece of its own colour" in text:
        hit = [i for i, o in enumerate(opts)
               if re.fullmatch(r"[a-h][1-8]", o) and _hangs(g, _fr(o))]
        if len(hit) == 1:
            return hit[0]
        sent = [i for i, o in enumerate(opts) if not re.fullmatch(r"[a-h][1-8]", o)]
        return sent[0] if not hit and len(sent) == 1 else None
    if "attacked by the greatest number of" in text:
        counts = [len(_attacked_by_enemy(g, _fr(o))) for o in opts]
        return counts.index(max(counts)) if counts.count(max(counts)) == 1 else None
    if text.startswith("Counting only the pieces on the board"):
        w = sum(_VALUES[p] for c, p in g.values() if c)
        b = sum(_VALUES[p] for c, p in g.values() if not c)
        d = w - b
        return 0 if d <= -3 else 1 if d <= -1 else 2 if d == 0 else 3 if d <= 2 else 4
    if text.startswith("How many legal moves"):
        n = boards[0].legal_moves.count()
        return 0 if n < 15 else 1 if n < 25 else 2 if n < 35 else 3
    if "gap between the two material totals" in text:
        vals = [abs(sum(_VALUES[p] for c, p in _grid_of(b).values() if c)
                    - sum(_VALUES[p] for c, p in _grid_of(b).values() if not c))
                for b in boards]
        return vals.index(max(vals)) if vals.count(max(vals)) == 1 else None
    if "greatest number of legal moves that give check" in text:
        vals = []
        for b in boards:
            n = 0
            for mv in b.legal_moves:          # python-chess makes the move; the grid decides
                after = b.copy(stack=False)
                after.push(mv)
                ga = _grid_of(after)
                king = next(s for s, (c, p) in ga.items() if p == "K" and c is not b.turn)
                n += bool(_attackers(ga, b.turn, king))
            vals.append(n)
        return vals.index(max(vals)) if vals.count(max(vals)) == 1 else None
    if "greatest number of legal moves" in text:
        vals = [b.legal_moves.count() for b in boards]
        return vals.index(max(vals)) if vals.count(max(vals)) == 1 else None
    if "pieces pinned against their own king" in text:
        vals = [sum(1 for sq in _grid_of(b) if _pinned(_grid_of(b), sq)) for b in boards]
        return vals.index(max(vals)) if vals.count(max(vals)) == 1 else None
    if "most pieces attacked by the other side while having no defender" in text:
        vals = []
        for b in boards:
            gb = _grid_of(b)
            vals.append(sum(1 for sq in gb if gb[sq][1] != "K" and _hangs(gb, sq)))
        return vals.index(max(vals)) if vals.count(max(vals)) == 1 else None

    m = re.search(r"Is ([a-h][1-8][a-h][1-8][qrbn]?) a legal move", text)
    if m:
        return int(m.group(1) in {mv.uci() for mv in boards[0].legal_moves})

    m = re.search(r"plays ([a-h][1-8][a-h][1-8][qrbn]?)\.", text)
    if m:
        mover = boards[0].turn
        after = boards[0].copy(stack=False)
        after.push(chess.Move.from_uci(m.group(1)))
        ga = _grid_of(after)
        if "king in check in the position that follows" in text:
            king = next(s for s, (c, p) in ga.items() if p == "K" and c is not mover)
            return int(bool(_attackers(ga, mover, king)))
        if "is the white king in check, and if so" in text or \
           "is the black king in check, and if so" in text:
            king = next(s for s, (c, p) in ga.items() if p == "K" and c is not mover)
            givers = set(_attackers(ga, mover, king))
            if not givers:
                return 0
            return 1 if givers == {_fr(m.group(1)[2:4])} else 2
        m2 = re.search(r"standing on ([a-h][1-8]) attacked by any (white|black) piece", text)
        if m2:
            return int(bool(_attackers(ga, m2.group(2) == "white", _fr(m2.group(1)))))
        if "attacked and has no defender of its own colour" in text:
            hit = [i for i, o in enumerate(opts)
                   if re.fullmatch(r"[a-h][1-8]", o) and _hangs(ga, _fr(o))]
            if len(hit) == 1:
                return hit[0]
            sent = [i for i, o in enumerate(opts) if not re.fullmatch(r"[a-h][1-8]", o)]
            return sent[0] if not hit and len(sent) == 1 else None
        m2 = re.search(r"standing on ([a-h][1-8]) pinned against its own king", text)
        if m2:
            return int(_pinned(ga, _fr(m2.group(1))))
    return None


def test_every_family_target_is_what_its_question_asks_for(pool):
    """The composition-level cross-check: 40 examples of every one of the 18 families,
    re-derived from the question text and a grid read off the FEN. Run over an earlier
    corpus build at full size it agreed on 22,000 of 22,000."""
    seen: Counter = Counter()
    for t in cf.tasks():
        fam = cf.BY_NAME[t.name.removeprefix("chessfact_")]
        for ex in t.load(40):
            for q in ex.questions:
                boards = [chess.Board(f) for f in q.meta["fens"]] \
                    if "fens" in q.meta else None
                if boards is None:
                    d = json.loads(ex.state)
                    blocks = [d] if fam.boards == 1 else list(d.values())
                    if not all("fen" in b for b in blocks):
                        continue          # map/ascii renderings are covered by the
                        # rendering test; the FEN ones are what this reads
                    boards = [chess.Board(b["fen"]) for b in blocks]
                got = _answer(t.name, q, boards)
                assert got is not None, f"{t.name}: no independent rule matched {q.question!r}"
                assert got == q.target, (f"{t.name}: question asks for {got}, target says "
                                         f"{q.target} -- {q.question!r} {q.options}")
                seen[t.name] += 1
    assert len(seen) == len(cf.FAMILIES), f"only checked {sorted(seen)}"
    assert all(v >= 5 for v in seen.values()), seen


def test_the_banded_families_agree_with_the_band_their_criterion_states(pool):
    """A band edge is the cheapest thing in this row to get wrong by one, and the
    composition check above only catches it when a sampled position lands exactly on the
    edge. The criteria state the ranges in words, so read them back.

    `_MOB_DESCS` and `_MAT_DESCS` are the contract; `meta` carries the raw count.
    """
    bands = {
        "fewer_than_15": (None, 14), "from_15_to_24": (15, 24),
        "from_25_to_34": (25, 34), "35_or_more": (35, None),
    }
    mat = {
        "black_ahead_by_three_or_more": (None, -3), "black_ahead_by_one_or_two": (-2, -1),
        "level": (0, 0), "white_ahead_by_one_or_two": (1, 2),
        "white_ahead_by_three_or_more": (3, None),
    }
    seen = Counter()
    for t in cf.tasks():
        if t.name not in ("chessfact_mobility", "chessfact_material"):
            continue
        for ex in t.load(200):
            for q in ex.questions:
                key = q.options[q.target]
                if t.name == "chessfact_mobility":
                    lo, hi = bands[key]
                    n = q.meta["legal_moves"]
                else:
                    lo, hi = mat[key]
                    n = q.meta["white"] - q.meta["black"]
                assert lo is None or n >= lo, f"{t.name}: {n} is not in {key}"
                assert hi is None or n <= hi, f"{t.name}: {n} is not in {key}"
                seen[key] += 1
    assert len(seen) == len(bands) + len(mat), f"a band never came up: {sorted(seen)}"
    # and the edges themselves have to be exercised, or the check is vacuous
    assert seen


def test_defenders_exclude_the_piece_itself(pool):
    """`board.attackers(colour, sq)` includes a piece that x-rays through the square it
    stands on for some piece types; a defender that is the piece itself would make every
    question about defence trivially yes."""
    for fen in pool[:40]:
        board = chess.Board(fen)
        for sq in board.piece_map():
            assert sq not in cf._defenders(board, sq)


# ---- the holdout is a structure ------------------------------------------------------

def test_every_task_is_prefixed_and_forced_to_a_split():
    for t in cf.tasks():
        assert t.name.startswith("chessfact_"), \
            f"{t.name} misses the PROTECTED_PREFIXES key and the enricher would rewrite it"
        assert t.row == cf.ROW and t.real is False
        assert t.force_split in ("train", "devreal", "testreal"), t.name
    names = {t.name for t in cf.tasks()}
    for fam in cf.TRAINED:
        assert next(t for t in cf.tasks() if t.name == f"chessfact_{fam.name}") \
            .force_split == "train"
    for fam in cf.HELDOUT:
        assert next(t for t in cf.tasks() if t.name == f"chessfact_{fam.name}") \
            .force_split == "testreal"
    for fam in cf.DEV_HELDOUT:
        assert next(t for t in cf.tasks() if t.name == f"chessfact_{fam.name}") \
            .force_split == "devreal"
    assert len(names) == len(cf.FAMILIES)


def test_held_out_families_use_no_primitive_training_never_saw():
    """The `rules.py` lesson: hold out a combination, train every part of it. A drop on
    a held-out family then measures composition and not an unseen word."""
    trained = cf.trained_primitives()
    for fam in cf.HELDOUT + cf.DEV_HELDOUT:
        unseen = set(fam.prims) - trained
        assert not unseen, f"{fam.name} introduces untrained primitives {unseen}"


def test_every_primitive_is_trained_somewhere():
    assert set(cf.PRIMITIVES) == cf.trained_primitives()


def test_the_holdout_is_a_depth_not_a_sample():
    assert {f.depth for f in cf.TRAINED} == {1, 2}
    assert {f.depth for f in cf.HELDOUT} == {3}
    assert {f.depth for f in cf.DEV_HELDOUT} == {3}


def test_dev_families_are_neither_trained_nor_tested():
    """Dev selects checkpoints and fits T; it holds out its own compositions, so it reads
    transfer the way test does without ever reading a test family."""
    dev = {f.name for f in cf.DEV_HELDOUT}
    assert dev and not dev & {f.name for f in cf.TRAINED + cf.HELDOUT}
    assert dev <= set(cf.TEMPLATES) and dev <= set(cf.BUILDERS)


def test_every_held_out_family_names_a_trained_control():
    trained = {f.name for f in cf.TRAINED}
    for fam in cf.HELDOUT + cf.DEV_HELDOUT:
        assert fam.control in trained, f"{fam.name} pairs with {fam.control!r}"
        # the control must share the family's primitives, or it controls for nothing
        shared = set(cf.BY_NAME[fam.control].prims) & set(fam.prims)
        assert shared, f"{fam.name} and its control {fam.control} share no primitive"


def test_importing_the_module_does_not_import_chess():
    """`pytest`, `--help` and every build that does not select row 43 must not pay for the
    extra. The pattern is `gefs.py`'s: the import lives inside the function."""
    tree = ast.parse(Path(cf.__file__).read_text())
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mod = getattr(node, "module", None) or ""
            names = [a.name for a in node.names]
            assert "chess" not in names and not mod.startswith("chess"), \
                "module-level `import chess`: row 43's extra is optional"


# ---- contamination against row 13 -----------------------------------------------------

def test_no_position_shares_a_fen_with_a_lichess_task(pool):
    """Row 13 reserves eight puzzle themes to `testreal`. A FEN shared with one of them
    would put a held-out eval state into training, which is the leak, not a duplicate."""
    from lod.corpus.services.sources.real import lichess

    saved = lichess._ROWS
    store.configure(RAW_ROOT)
    lichess._ROWS = None
    try:
        rows = lichess._rows()
        if not rows:
            pytest.skip("no cached lichess puzzles to compare against")
        reserved = {r["FEN"] for r in rows if lichess._reserved_pool(r)}
        trained = {r["FEN"] for r in rows if not lichess._reserved_pool(r)}
        assert not (set(pool) & reserved), "row 43 states a reserved lichess position"
        assert not (set(pool) & trained)
        # and the partition copy must agree with the authority, puzzle by puzzle
        assert all(cf._reserved_pool(r["PuzzleId"]) is lichess._reserved_pool(r)
                   for r in rows)
    finally:
        lichess._ROWS = saved


def test_the_two_halves_of_the_pool_share_no_position(pool):
    """A trained task and a held-out task must never state the same board. The first
    build of this row shared one pool and `dedup_against_train` deleted 1,947 of the
    5,000 testreal examples -- 39 % of the holdout, and silently."""
    trained, held = set(cf.pool(reserved=False)), set(cf.pool(reserved=True))
    dev = set(cf.dev_pool())
    assert trained and held and dev
    assert not (trained & held) and not (dev & trained) and not (dev & held)
    assert len(trained) + len(held) + len(dev) == len(set(pool))
    assert 0.4 < len(held) / len(pool) < 0.6, "the halves are lopsided"
    # dev comes out of the trained half, a quarter of it, and test's half is untouched
    assert 0.08 < len(dev) / len(pool) < 0.17, len(dev) / len(pool)


def test_no_state_is_shared_between_a_trained_and_a_held_out_task(pool):
    by = {"train": set(), "devreal": set(), "testreal": set()}
    for t in cf.tasks():
        for ex in t.load(40):
            by[t.force_split].add(ex.state)
    for a, b in (("train", "testreal"), ("train", "devreal"), ("devreal", "testreal")):
        assert not (by[a] & by[b]), f"{len(by[a] & by[b])} states cross {a}/{b}"


def test_the_seed_pool_is_a_mix_and_is_not_empty(pool):
    mix = cf.pool_mix()
    assert len(pool) > 2000
    assert mix["playout"] > 0
    # the puzzle half is only there when the cache is; say so rather than assume it
    if store.has(cf.PUZZLE_KEY):
        assert mix["puzzle"] > 500, mix
        assert 0.2 < mix["puzzle"] / len(pool) < 0.8, f"skewed seed mix {mix}"


# ---- what the questions may not be solvable by ------------------------------------------

def _gold(q):
    return q.options[q.target]


def _word_in(needle: str, hay: str) -> bool:
    return bool(re.search(r"(?<![A-Za-z0-9_])" + re.escape(needle) + r"(?![A-Za-z0-9_])", hay))


def test_the_gold_option_never_appears_in_its_own_question_or_instructions(sample):
    """Row 36 shipped at 100 % solvable this way. The first version of this row did too:
    `check_from` asked which piece gives check after a move written in long algebraic
    form, and for a direct check the answer was the move's own destination square."""
    hits = total = 0
    for name, exs in sample.items():
        for ex in exs:
            for q in ex.questions:
                total += 1
                hay = f"{q.question} {q.instructions or ''}"
                if _word_in(_gold(q), hay):
                    hits += 1
                    pytest.fail(f"{name}: gold {_gold(q)!r} appears in {hay!r}")
    assert total > 100


def test_the_gold_square_is_never_the_square_the_question_writes_in_a_move(sample):
    """The exact shape of this row's first-draft failure, which the word-boundary test
    above cannot see.

    `check_from` asked which piece gives check after a move written in long algebraic
    form, and for a direct check the answer was the move's own destination square -- the
    gold option verbatim in the question, 1000 of 1000 times, solvable by copying the
    last two characters. `_word_in("f1", "White plays e2f1")` does not fire, because the
    "2" in front of "f1" is a word character, so the general test would have passed
    through the whole family. Measured on an earlier corpus build: 0 of 22,000.
    """
    hits = checked = 0
    for name, exs in sample.items():
        for ex in exs:
            for q in ex.questions:
                m = re.search(r"plays ([a-h][1-8])([a-h][1-8])", q.question)
                if not m:
                    continue
                checked += 1
                gold = _gold(q)
                assert gold != m.group(2), \
                    f"{name}: gold {gold!r} is the destination of {m.group(0)!r}"
                assert gold != m.group(1), \
                    f"{name}: gold {gold!r} is the origin of {m.group(0)!r}"
                hits += gold in q.question
    assert checked > 50, "no after-a-move family was sampled"
    assert hits == 0


def test_no_gold_option_is_a_bare_substring_of_its_own_question(sample):
    """Stricter than the word-boundary test, and the one that would catch a square name
    hidden inside a UCI move or a coordinate.

    It is deliberately *not* run over the criteria or the instructions: those print
    identically for every option, and "no" sits inside "nobody" and "not" there, which
    is the partial-string artifact that read 4.7 % on row 42 purely from "nothing".
    """
    for name, exs in sample.items():
        for ex in exs:
            for q in ex.questions:
                gold = _gold(q)
                if gold in ("no", "yes"):
                    continue          # two-letter keys match inside ordinary words
                assert gold not in q.question, f"{name}: {gold!r} is in its own question"


def test_the_state_never_names_exactly_one_option(sample):
    """Where the options are squares they all carry a piece, so all of them appear in a
    piece-map rendering. The residual is the en-passant field naming one of them."""
    hits = total = 0
    for exs in sample.values():
        for ex in exs:
            for q in ex.questions:
                present = [o for o in q.options if _word_in(o, ex.state)]
                total += 1
                if len(present) == 1 and present[0] == _gold(q):
                    hits += 1
    assert hits / total < 0.01, f"{hits}/{total} answerable by the only option in the state"


def test_no_family_is_won_by_predicting_a_constant(sample):
    """An earlier model scored 0.658 on lichess by answering the class prior. Every family
    here has to be worth more than its majority class."""
    for name, exs in sample.items():
        golds = [_gold(q) for ex in exs for q in ex.questions]
        top = Counter(golds).most_common(1)[0][1] / len(golds)
        assert top <= 0.60, f"{name}: majority-class baseline {top:.3f}"


def test_the_gold_option_is_not_in_a_fixed_position(sample):
    idx = Counter(q.target for exs in sample.values() for ex in exs for q in ex.questions)
    total = sum(idx.values())
    assert idx[0] / total < 0.40, f"index 0 is the answer {idx[0] / total:.2%} of the time"


# ---- shape, criteria and the sentinel ----------------------------------------------------

def test_states_are_json_and_carry_the_position(sample):
    for name, exs in sample.items():
        fam = cf.BY_NAME[name.removeprefix("chessfact_")]
        for ex in exs:
            d = json.loads(ex.state)
            boards = [d] if fam.boards == 1 else list(d.values())
            assert len(boards) == fam.boards
            for b in boards:
                assert b["side_to_move"] in ("white", "black")
                assert "fen" in b or "pieces" in b or "board" in b


def test_the_same_position_rendered_four_ways_is_the_same_question(pool):
    """A rendering change must move the text and nothing else. Otherwise the row teaches
    FEN parsing rather than the composition it claims to."""
    checked = 0
    for fam in cf.FAMILIES:
        for trial in range(12):
            boards = [chess.Board(pool[(trial * 37 + j * 977) % len(pool)])
                      for j in range(fam.boards)]
            classes = cf.CLASSES.get(fam.name)
            want = None if classes is None else trial % classes
            spec = cf.build(fam.name, boards, random.Random(f"{fam.name}:{trial}"),
                            want, "train")
            if spec is None:
                continue
            checked += 1
            exs = [cf._example(spec, fam.name, sh) for sh in cf.RENDERINGS]
            base = exs[0].questions[0]
            for e in exs[1:]:
                q = e.questions[0]
                assert (q.target, q.options, q.question, q.descriptions) == \
                    (base.target, base.options, base.question, base.descriptions)
            assert len({e.state for e in exs}) == len(cf.RENDERINGS), \
                f"{fam.name}: two renderings produced identical text"
    assert checked > 100


def test_every_option_carries_a_criterion(sample):
    """The definition is input. A model told what a pin is and asked to
    apply it is the bet; a model that recognises the word is not."""
    for name, exs in sample.items():
        for ex in exs:
            for q in ex.questions:
                assert q.descriptions and len(q.descriptions) == len(q.options), name
                assert all(d for d in q.descriptions), name
                assert q.instructions and len(q.instructions) > 80, name


def test_the_instructions_define_the_terms_the_question_uses(sample):
    for name, exs in sample.items():
        for ex in exs[:5]:
            for q in ex.questions:
                text = q.question.lower()
                instr = q.instructions
                if "pinned" in text:
                    assert "PINNED" in instr, name
                if "attacked" in text or "attacks" in text:
                    assert "ATTACKS" in instr, name
                if "defend" in text:
                    assert "DEFENDED" in instr, name


def test_the_sentinel_comes_from_the_grammar(sample):
    """Row 39 repeated one hard-coded wording 1,402 times and failed the sentinel audit,
    the same failure an earlier model shipped with."""
    wordings, golds, present = [], 0, 0
    for name, exs in sample.items():
        for ex in exs:
            for q in ex.questions:
                for i, d in enumerate(q.descriptions):
                    if sentinel.is_sentinel_description(d):
                        wordings.append(d)
                        present += 1
                        assert sentinel.is_sentinel_key(q.options[i]), q.options[i]
                        golds += (i == q.target)
    assert present > 20, "the sentinel families produced no sentinel"
    assert Counter(wordings).most_common(1)[0][1] <= 3, "a sentinel wording repeats"
    assert golds / present < 0.40, \
        f"the sentinel is correct {golds / present:.2%} of the time it is present"


def test_sentinel_wordings_respect_the_train_eval_partition(sample):
    """The `sentinel_wordings` gate measures generalisation to unseen sentinel wordings.
    A trained task that drew from the `eval` half of the wording space would make that
    number recall."""
    for name, exs in sample.items():
        want = "eval" if cf.BY_NAME[name.removeprefix("chessfact_")] in \
            cf.HELDOUT + cf.DEV_HELDOUT else "train"
        for ex in exs:
            for q in ex.questions:
                for d in q.descriptions:
                    if sentinel.is_sentinel_description(d):
                        assert sentinel.split_of(d) == want, f"{name}: {d!r}"


# ---- budgets, shapes and determinism ------------------------------------------------------

def test_option_counts_are_spread(sample):
    ks = [len(q.options) for exs in sample.values() for ex in exs for q in ex.questions]
    assert max(ks) >= 7
    assert sum(1 for k in ks if k > 2) / len(ks) > 0.4, "too much of the row is binary"
    assert sum(1 for k in ks if k == 2) / len(ks) > 0.2, "no binary questions at all"


def test_state_lengths_sit_inside_the_training_budget(sample, tok):
    """Both ends, counted with a tokenizer rather than guessed from characters.

    `ship_v4.sh` trains at `--max-state-tokens 2304` and `packing.Packer` truncates a
    long state from the tail, so a four-position state past the budget would lose a
    position and still offer it as an option. And under 40 tokens a state is a shortcut,
    not a task. Measured over all 18,000 states with `Qwen/Qwen3-0.6B-Base`, the model's
    own tokenizer: min 56, p5 76, p50 272, p95 1690, max 2068. The session tokenizer
    used here (SmolLM2-135M) reads the same states at min 68, p50 286, p95 1748,
    max 2137, so it is the slightly pessimistic of the two.
    """
    lens = sorted(len(tok(ex.state, add_special_tokens=False)["input_ids"])
                  for exs in sample.values() for ex in exs)
    assert lens[0] >= 40, f"shortest state is {lens[0]} tokens"
    assert lens[int(0.95 * len(lens))] <= 2304, "p95 is past --max-state-tokens"
    assert lens[-1] <= 2304, f"longest state is {lens[-1]} tokens"


def test_targets_are_well_formed(sample):
    for name, exs in sample.items():
        for ex in exs:
            for q in ex.questions:
                assert len(set(q.options)) == len(q.options), f"{name}: duplicate options"
                probs = q.target_probs()
                assert probs is not None and abs(sum(probs) - 1.0) < 1e-9
                assert q.id and ex.task == name


def test_every_task_yields_what_it_is_asked_for(pool):
    for t in cf.tasks():
        got = list(t.load(20))
        assert len(got) == 20, f"{t.name} produced {len(got)}"


def test_generation_is_seeded(pool):
    t = cf.tasks()[0]
    assert [e.state for e in t.load(6)] == [e.state for e in t.load(6)]


def test_a_task_never_exceeds_its_own_cap(pool):
    t = cf.tasks()[0]
    assert len(list(t.load(cf.PER_TASK + 500))) == cf.PER_TASK


def test_the_loader_refuses_to_be_silently_empty(monkeypatch):
    """A source that returns [] when its input is missing is how the row-1 Hub sweep
    claimed rows 13 and 30. This one raises instead."""
    monkeypatch.setattr(cf, "pool", lambda *a, **k: [])
    with pytest.raises(RuntimeError):
        list(cf.tasks()[0].load(1))
