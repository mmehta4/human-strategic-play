"""Research pool: the 3 bonus games at the end of every arena visit.

This replaces a paid study (zero budget) with two kinds of games:

calibration  12 games (24 decision problems) copied exactly from games2p2k, whose players
             were paid a real bonus on Prolific. Comparing arena shares with the published
             shares tests whether free play matches paid play (hypothesis H2). Note: it
             also compares populations (US Prolific workers vs arena players).
research     600 new 3x3 games from a fixed seed, never used in development: the
             confirmatory test of whether the hybrid generalises (hypothesis H1).

The pool is written once by `build()` and then frozen: its checksum goes into the
pre-registration. Bonus games are handed out least-played first, so data fills evenly.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

POOL_PATH = Path(__file__).with_name("research_pool.json")
SEED = 20261023          # fixed before any data is collected; goes into the pre-registration
N_RESEARCH = 600         # 600 games x 2 roles = 1,200 decision problems (power ~0.93 vs ElementaryNet-style)
N_CALIBRATION = 12
BONUS_PER_VISIT = 3
LABELS = {"row": ["A", "B", "C"], "col": ["X", "Y", "Z"]}


def _family(own: np.ndarray, other: np.ndarray) -> str:
    """Same strategic families as games2p2k, generalised to any size."""
    from hsb.games import Game, pure_nash, weakly_dominant_action
    ds_self = weakly_dominant_action(own) is not None
    ds_other = weakly_dominant_action(other) is not None
    if ds_self and ds_other:
        return "both_dominant"
    if ds_self:
        return "self_dominant"
    if ds_other:
        return "other_dominant"
    n = len(pure_nash(Game(own, other)))
    return "coordination" if n >= 2 else ("single_eq" if n == 1 else "no_pure_eq")


def build(path: Path = POOL_PATH) -> dict:
    """Create the frozen pool. Run once; never re-run after data collection starts."""
    from hsb.data import games2p2k
    rng = np.random.default_rng(SEED)
    games = []
    # calibration: whole games (both roles) from games2p2k, spread across families
    recs = games2p2k.load()
    by_game = {}
    for r in recs:
        by_game.setdefault(r.game_id, []).append(r)
    fams = sorted({r.family for r in recs})
    chosen = []
    for k in range(N_CALIBRATION):
        fam = fams[k % len(fams)]
        cands = [gid for gid, rs in by_game.items() if any(r.family == fam for r in rs) and gid not in chosen]
        chosen.append(cands[int(rng.integers(len(cands)))])
    for gid in chosen:
        row_rec = next(r for r in by_game[gid] if r.game.meta["original_role"] == "rowplayer")
        own, other = row_rec.game.own, row_rec.game.other          # row's view: other is (col, row)
        games.append({"id": f"cal-{gid.split('-')[1]}", "kind": "calibration", "source_game": gid,
                      "row": own.astype(int).tolist(), "col": other.T.astype(int).tolist(),
                      "family": _family(own, other)})
    # research: new 3x3 games, payoffs 1-50 like games2p2k
    for k in range(N_RESEARCH):
        row = rng.integers(1, 51, size=(3, 3))
        col = rng.integers(1, 51, size=(3, 3))           # col[i][j]: col's payoff when row i, col j
        games.append({"id": f"res-{k:03d}", "kind": "research", "row": row.tolist(), "col": col.tolist(),
                      "family": _family(row, col.T)})
    pool = {"seed": SEED, "games": games}
    text = json.dumps(pool, indent=1)
    pool["sha256"] = hashlib.sha256(text.encode()).hexdigest()
    path.write_text(json.dumps(pool, indent=1))
    return pool


def load(path: Path = POOL_PATH) -> dict:
    pool = json.loads(path.read_text())
    return {g["id"]: g for g in pool["games"]}


def matrices(game: dict, role: str) -> tuple[list, list]:
    """(my payoffs [mine][theirs], their payoffs [theirs][mine]) for `role`."""
    row, col = np.array(game["row"]), np.array(game["col"])
    if role == "row":
        return row.tolist(), col.T.tolist()
    return col.T.tolist(), row.tolist()


def lines(game: dict, role: str, t) -> list[str]:
    mine, theirs = matrices(game, role)
    other_role = "col" if role == "row" else "row"
    my_l = LABELS[role][: len(mine)]
    their_l = LABELS[other_role][: len(mine[0])]
    out = [t("matrix.intro")]
    for i, a in enumerate(my_l):
        for j, b in enumerate(their_l):
            out.append(t("matrix.cell", mine=a, theirs=b, me=mine[i][j], them=theirs[j][i]))
    return out


def options(game: dict, role: str) -> list[str]:
    mine, _ = matrices(game, role)
    return LABELS[role][: len(mine)]


def payoff(game: dict, role: str, mine: int, theirs: int) -> float:
    my, _ = matrices(game, role)
    return my[int(mine)][int(theirs)]


def checksum(path: Path = POOL_PATH) -> tuple[str, bool]:
    """(stored sha256, whether the file still matches it). Goes into the pre-registration."""
    pool = json.loads(path.read_text())
    stored = pool.pop("sha256")
    return stored, hashlib.sha256(json.dumps(pool, indent=1).encode()).hexdigest() == stored
