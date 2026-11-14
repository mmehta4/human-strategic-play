"""Loader for games2p2k (Zhu, Peterson, Enke & Griffiths): 2,416 two-action decision
problems from 1,208 procedurally generated 2x2 games, ~93k choices from 4,673 people.

Source: https://osf.io/xrvaw/  (CC-BY 4.0). Cite the paper when using it.

The release gives the share of players choosing the first action ("up") in each
decision problem, not raw counts. ~93,460 choices over 2,416 problems means about
39 choices per problem; `n_est` records that average so likelihoods can be scaled
to choice counts when needed. Every problem is shown from the decision maker's
side (the release presents every participant as the row player).
"""
from __future__ import annotations

import ast
import hashlib
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from ..games import Game
from .schema import Record

OSF_FILES = {
    "games2p2k_main.csv": "https://osf.io/download/c7hr8/",
    "README.pdf": "https://osf.io/download/63k9v/",
}
TOTAL_CHOICES = 93_460
DEFAULT_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "games2p2k"

# Robinson & Goforth topology names, as they appear in the release
TOPOLOGY_COL = "topology"


def download(dest: Path = DEFAULT_DIR) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    for name, url in OSF_FILES.items():
        path = dest / name
        if not path.exists():
            urllib.request.urlretrieve(url, path)
    return dest


def load(path: Path = DEFAULT_DIR) -> list[Record]:
    csv = Path(path) / "games2p2k_main.csv"
    if not csv.exists():
        download(Path(path))
    df = pd.read_csv(csv)
    n_est = TOTAL_CHOICES / len(df)
    records = []
    for row in df.itertuples(index=False):
        m = np.array(row.row_form_matrix.strip("[]").split(), dtype=float)
        game = Game.from_2x2(m[:4], m[4:], meta={
            "topology": ast.literal_eval(row.topology),
            "original_role": row.role,
            "mean_logrt_norm": row.logrt_norm,
        })
        p_up = float(row.up_choice)
        records.append(Record(
            source="games2p2k",
            game_id=f"g2p2k-{int(row.game_id)}",
            problem_id=f"g2p2k-{int(row.unique_id)}",
            game=game,
            freq=np.array([p_up, 1 - p_up]),
            n=n_est,
            family=family_of(game),
        ))
    return records


def family_of(game: Game) -> str:
    """Strategic family of a 2x2 decision problem, used for held-out-family tests.

    Dominance is weak dominance, matching the release's `ds_self` / `ds_other`.

    both_dominant  - both players have a dominant action (level-1 suffices)
    self_dominant  - only the decision maker has one
    other_dominant - only the opponent has one (needs level-2 reasoning)
    coordination   - nobody has one; two or more pure equilibria
    single_eq      - nobody has one; exactly one pure equilibrium
    no_pure_eq     - nobody has one; only mixed equilibria
    """
    from ..games import pure_nash, weakly_dominant_action
    ds_self = weakly_dominant_action(game.own) is not None
    ds_other = weakly_dominant_action(game.other) is not None
    if ds_self and ds_other:
        return "both_dominant"
    if ds_self:
        return "self_dominant"
    if ds_other:
        return "other_dominant"
    n_pure = len(pure_nash(game))
    return "coordination" if n_pure >= 2 else ("single_eq" if n_pure == 1 else "no_pure_eq")


def checksum(path: Path = DEFAULT_DIR) -> str:
    return hashlib.sha256((Path(path) / "games2p2k_main.csv").read_bytes()).hexdigest()
