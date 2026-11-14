"""Train/test splits. Every split keeps both sides of a game together: if one
player's side of a game is in the test set, the other side never appears in
training, since the opponent's choices reveal a lot about the game.

random_problems   - random 80/10/10-style split over decision problems, as in
                    Zhu et al.'s released notebook (for comparability only)
grouped_kfold     - k folds over games: tests on games never seen in training
leave_family_out  - train on all strategic families but one, test on the held-out
                    family: the generalisation test Gate 1 is judged on
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..data.schema import Record


def random_problems(records: list[Record], test_frac=0.1, seed=0, n_repeats=5):
    for rep in range(n_repeats):
        rng = np.random.default_rng(seed + rep)
        idx = rng.permutation(len(records))
        cut = int(len(records) * (1 - test_frac))
        yield f"random-{rep}", [records[i] for i in idx[:cut]], [records[i] for i in idx[cut:]]


def grouped_kfold(records: list[Record], k=10, seed=0):
    games = np.array(sorted({r.game_id for r in records}))
    rng = np.random.default_rng(seed)
    rng.shuffle(games)
    folds = np.array_split(games, k)
    for i, fold in enumerate(folds):
        test_games = set(fold)
        yield (f"fold-{i}", [r for r in records if r.game_id not in test_games],
               [r for r in records if r.game_id in test_games])


def leave_family_out(records: list[Record]):
    by_family = defaultdict(list)
    for r in records:
        by_family[r.family].append(r)
    for fam, test in sorted(by_family.items()):
        test_games = {r.game_id for r in test}
        train = [r for r in records if r.game_id not in test_games]
        yield f"holdout-{fam}", train, test
