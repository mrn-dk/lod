"""A larger Lichess puzzle pool for chess facts (`synth/chessfacts.py`, row 43).

Row 43 could draw positions from row 13's cached puzzles plus random playouts, but
playouts almost never contain a pin, so its pin families were built from a few hundred
distinct positions reused many times. Puzzles are tactical: pins and attacked, undefended
pieces are common.

Reads one shard of Lichess/chess-puzzles (CC0), skips every puzzle row 13 already stores
(so the two rows never share a position), drops any position that is a Decision Index
ChessBench position (board, side to move and castling rights, before or after the
puzzle's first move), and stores the rest as `lichess_puzzle_pool`. The ChessBench test
bag is required: without it the exclusion cannot be made, and the pool is not written.
"""

from __future__ import annotations

import mmap
import struct
from pathlib import Path

from lod.corpus.repositories import hub
from lod.corpus.repositories import raw_store as store
from lod.corpus.services.sources.real.lichess import KEY as ROW13_KEY
from lod.corpus.services.sources.real.lichess import PATH
from lod.corpus.services.sources.synth.chessfacts import POOL_KEY

SHARD = "data/train-00000-of-00003.parquet"
N = 40_000


def outputs() -> list[Path]:
    return [store.path_for(POOL_KEY)]


def chessbench_keys(bag: Path) -> set[str]:
    """First three FEN fields of every position in a ChessBench `.bag` file.

    The bag is a flat record file with a trailing offset index; each record starts with a
    varint length followed by the FEN.
    """
    keys: set[str] = set()
    with bag.open("rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as buf:
        index = struct.unpack_from("<Q", buf, len(buf) - 8)[0]
        start = 0
        for off in range(index, len(buf), 8):
            end = struct.unpack_from("<Q", buf, off)[0]
            rec = buf[start:end]
            start = end
            o = length = shift = 0
            while True:
                b = rec[o]
                o += 1
                length |= (b & 127) << shift
                if b < 128:
                    break
                shift += 7
            keys.add(" ".join(bytes(rec[o:o + length]).decode().split()[:3]))
    return keys


def fetch(chessbench: Path | None, n: int = N, refresh: bool = False) -> int:
    if store.has(POOL_KEY) and not refresh:
        return 0
    if chessbench is None or not Path(chessbench).is_file():
        raise FileNotFoundError(
            f"ChessBench test bag not found ({chessbench}); it is needed to keep Decision "
            f"Index positions out of the pool (pass --chessbench)")
    if not store.has(ROW13_KEY):
        raise FileNotFoundError(f"{ROW13_KEY} is not stored yet; fetch row 13 first")

    import chess
    import pyarrow.parquet as pq

    row13 = {r.get("PuzzleId") for r in store.load(ROW13_KEY)}
    bench = chessbench_keys(Path(chessbench))
    table = pq.read_table(hub.file(PATH, SHARD), columns=["PuzzleId", "FEN", "Moves"])
    out, skipped = [], {"row13": 0, "chessbench": 0, "bad": 0}
    for pid, fen, moves in zip(*(table.column(c).to_pylist()
                                 for c in ("PuzzleId", "FEN", "Moves"))):
        if len(out) >= n:
            break
        if pid in row13:
            skipped["row13"] += 1
            continue
        try:
            b = chess.Board(fen)
            b.push(chess.Move.from_uci(str(moves).split()[0]))
        except (ValueError, AssertionError, IndexError):
            skipped["bad"] += 1
            continue
        if " ".join(b.fen().split()[:3]) in bench or " ".join(fen.split()[:3]) in bench:
            skipped["chessbench"] += 1
            continue
        out.append({"PuzzleId": pid, "FEN": fen, "Moves": moves})
    store.save(POOL_KEY, out)
    print(f"  {POOL_KEY}: {len(out):,} puzzles; skipped {skipped}", flush=True)
    return len(out)
