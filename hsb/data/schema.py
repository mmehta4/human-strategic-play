"""Unified record format. Every dataset is converted to a list of `Record`s, one per
decision problem (a game seen from one player's side), so every model and metric
works on every source."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..games import Game


@dataclass
class Record:
    source: str          # dataset name, e.g. "games2p2k"
    game_id: str         # the underlying game; both sides of a game share it
    problem_id: str      # unique per (game, side)
    game: Game           # seen from the decision maker's side
    freq: np.ndarray     # empirical choice shares over the decision maker's actions
    n: float             # number of choices behind `freq` (may be an estimate)
    family: str = ""     # strategic family, for held-out-family evaluation
    population: str = "" # who played (e.g. "prolific-us-2024"), when known
    extra: dict = field(default_factory=dict)

    @property
    def counts(self) -> np.ndarray:
        return self.freq * self.n
