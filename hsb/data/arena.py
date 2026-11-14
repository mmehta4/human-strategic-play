"""Turn an arena export (/export.csv) into benchmark records.

Each two-player arena game becomes one decision problem per role: the payoff matrices
come from the arena's own payoff rules (`arena.games.normal_form`), the label is the
share of players choosing each action. Only unflagged choices count, and only each
player's first play of a game, so replays don't double-count anyone.

Games with more than two players (beauty contest, public goods, weakest link),
auctions with private values, the dictator game and the gambles are not normal-form
two-player games, so they are left out here; they are modelled separately.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from ..games import Game
from .schema import Record

FAMILY = {
    "ultimatum_offer": "ultimatum", "ultimatum_min": "ultimatum",
    "trust_send": "trust", "trust_return": "trust",
    "bertrand": "pricing", "money_request": "undercutting", "travelers": "undercutting",
    "m_coord": "matrix", "m_other_dom": "matrix", "pd": "matrix", "stag": "matrix",
    "hawk_dove": "matrix", "hide_seek": "matrix",
}
MIN_CHOICES = 20  # fewer than this and the shares are too noisy to score against


def load(path: str | Path, min_choices: int = MIN_CHOICES, population: str = "arena",
         kinds: tuple = ("fixed", "calibration", "research")) -> list[Record]:
    """kinds: "fixed" = the 20 arena games; "calibration" / "research" = bonus pool games."""
    from arena.games import BY_ID, normal_form, counterpart
    from arena import pool as arena_pool
    from .games2p2k import family_of

    df = pd.read_csv(path)
    is_pool = df.game_id.str.startswith(("cal-", "res-"))
    df = df[(df.flagged == 0) & (df.game_id.isin(FAMILY) | is_pool)]
    df = df.sort_values("session_no").drop_duplicates(["participant_id", "game_id"], keep="first")
    records = []
    pool_games = arena_pool.load() if arena_pool.POOL_PATH.exists() else {}
    for (gid, role), grp in df.groupby(["game_id", "role"]):
        if len(grp) < min_choices:
            continue
        if gid in pool_games:
            pg = pool_games[gid]
            if pg["kind"] not in kinds:
                continue
            own, other = (np.array(m, dtype=float) for m in arena_pool.matrices(pg, role))
            counts = Counter(grp.choice.round(6))
            freq = np.array([counts.get(float(a), 0) for a in range(own.shape[0])], dtype=float)
            game = Game(own, other, {"pool_id": gid, "kind": pg["kind"], "role": role,
                                     "source_game": pg.get("source_game")})
            records.append(Record(source="arena", game_id=f"arena-{gid}", problem_id=f"arena-{gid}-{role}",
                                  game=game, freq=freq / freq.sum(), n=float(freq.sum()),
                                  family=family_of(game), population=population, extra={"kind": pg["kind"]}))
            continue
        if "fixed" not in kinds:
            continue
        mine, theirs, own, other = normal_form(gid, role)
        counts = Counter(grp.choice.round(6))
        freq = np.array([counts.get(round(a, 6), 0) for a in mine], dtype=float)
        if freq.sum() == 0:
            continue
        game_key = "-".join(sorted([gid, counterpart(gid)]))
        records.append(Record(
            source="arena", game_id=f"arena-{game_key}", problem_id=f"arena-{gid}-{role}",
            game=Game(np.array(own), np.array(other), {"arena_game": gid, "role": role,
                                                         "actions": mine, "opponent_actions": theirs}),
            freq=freq / freq.sum(), n=float(freq.sum()), family=FAMILY[gid], population=population,
            extra={"kind": "fixed"},
        ))
    return records


def by_language(path: str | Path, **kw) -> dict[str, list[Record]]:
    """Separate record sets per play language: the first cross-population comparison."""
    df = pd.read_csv(path)
    out = defaultdict(list)
    for lang, grp in df.groupby("lang"):
        tmp = Path(path).with_suffix(f".{lang}.csv")
        grp.to_csv(tmp, index=False)
        out[lang] = load(tmp, population=f"arena-{lang}", **kw)
        tmp.unlink()
    return dict(out)
