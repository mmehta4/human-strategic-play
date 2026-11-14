"""Two-player normal-form games and the game-theoretic utilities the models build on.

Convention: a decision problem is always seen from the decision maker's side.
`own[i, j]` is the decision maker's payoff when they play action i and the
opponent plays action j. `other[j, i]` is the opponent's payoff when the
opponent plays j and the decision maker plays i, so each matrix is indexed
(own action, other's action) from its owner's point of view.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np


@dataclass(frozen=True)
class Game:
    own: np.ndarray    # shape (m, n): decision maker's payoffs
    other: np.ndarray  # shape (n, m): opponent's payoffs, opponent as "row"
    meta: dict = field(default_factory=dict, compare=False)

    def __post_init__(self):
        own = np.asarray(self.own, dtype=float)
        other = np.asarray(self.other, dtype=float)
        if own.ndim != 2 or other.shape != own.shape[::-1]:
            raise ValueError(f"payoff shapes {own.shape} and {other.shape} do not match")
        object.__setattr__(self, "own", own)
        object.__setattr__(self, "other", other)

    @property
    def n_actions(self) -> int:
        return self.own.shape[0]

    def swap(self) -> "Game":
        """The same game seen from the opponent's side."""
        return Game(self.other, self.own, dict(self.meta))

    def permute(self, own_perm=None, other_perm=None) -> "Game":
        """Relabel actions. Behaviour should be invariant to this."""
        own_perm = np.arange(self.own.shape[0]) if own_perm is None else np.asarray(own_perm)
        other_perm = np.arange(self.own.shape[1]) if other_perm is None else np.asarray(other_perm)
        return Game(self.own[np.ix_(own_perm, other_perm)],
                    self.other[np.ix_(other_perm, own_perm)], dict(self.meta))

    def scaled(self) -> "Game":
        """Payoffs divided by the largest absolute payoff in the game."""
        s = max(np.abs(self.own).max(), np.abs(self.other).max(), 1e-9)
        return Game(self.own / s, self.other / s, dict(self.meta))

    @classmethod
    def from_2x2(cls, row: list[float], col: list[float], meta: dict | None = None) -> "Game":
        """Build from games2p2k's layout: row=[r11,r12,r21,r22], col=[c11,c12,c21,c22]
        where cell (i, j) is (row action i, col action j)."""
        own = np.array(row, dtype=float).reshape(2, 2)
        col_mat = np.array(col, dtype=float).reshape(2, 2)  # indexed (row action, col action)
        return cls(own, col_mat.T, meta or {})


# ---------------------------------------------------------------- utilities

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    z = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def expected_payoffs(payoff: np.ndarray, opponent_mix: np.ndarray) -> np.ndarray:
    return payoff @ opponent_mix


def logit_response(payoff: np.ndarray, opponent_mix: np.ndarray, lam: float) -> np.ndarray:
    return softmax(lam * expected_payoffs(payoff, opponent_mix))


def best_response(payoff: np.ndarray, opponent_mix: np.ndarray) -> np.ndarray:
    """Uniform over the set of best responses (ties split evenly)."""
    u = expected_payoffs(payoff, opponent_mix)
    best = np.isclose(u, u.max())
    return best / best.sum()


def strictly_dominant_action(payoff: np.ndarray) -> int | None:
    m = payoff.shape[0]
    for i in range(m):
        if all(np.all(payoff[i] > payoff[k]) for k in range(m) if k != i):
            return i
    return None


def weakly_dominant_action(payoff: np.ndarray) -> int | None:
    """An action at least as good as every other against every opponent action,
    and strictly better somewhere (the definition games2p2k uses)."""
    m = payoff.shape[0]
    for i in range(m):
        if all(np.all(payoff[i] >= payoff[k]) and np.any(payoff[i] > payoff[k])
               for k in range(m) if k != i):
            return i
    return None


def nash_equilibria(game: Game, tol: float = 1e-9) -> list[tuple[np.ndarray, np.ndarray]]:
    """All Nash equilibria by support enumeration (fine for the small games used here).
    Returns (own_strategy, other_strategy) pairs, deduplicated. Degenerate games may
    have a continuum; we return the vertices support enumeration finds."""
    A, B = game.own, game.other.T  # both indexed (own action, other action)
    m, n = A.shape
    found: list[tuple[np.ndarray, np.ndarray]] = []
    for k in range(1, min(m, n) + 1):
        for s1 in combinations(range(m), k):
            for s2 in combinations(range(n), k):
                y = _indifference(A[np.ix_(s1, s2)])          # other's mix making own indifferent
                x = _indifference(B[np.ix_(s1, s2)].T)        # own mix making other indifferent
                if x is None or y is None:
                    continue
                xs = np.zeros(m); xs[list(s1)] = x
                ys = np.zeros(n); ys[list(s2)] = y
                if (A @ ys).max() > (A @ ys)[list(s1)].max() + tol:
                    continue
                if (xs @ B).max() > (xs @ B)[list(s2)].max() + tol:
                    continue
                if not any(np.allclose(xs, f[0]) and np.allclose(ys, f[1]) for f in found):
                    found.append((xs, ys))
    return found


def _indifference(M: np.ndarray) -> np.ndarray | None:
    """Mix q over columns of M (k x k) making every row equally good; q >= 0, sums to 1."""
    k = M.shape[0]
    if k == 1:
        return np.ones(1)
    lhs = np.vstack([M[:-1] - M[1:], np.ones(k)])
    rhs = np.zeros(k); rhs[-1] = 1.0
    try:
        q = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        return None
    if np.any(q < -1e-9):
        return None
    q = np.clip(q, 0, None)
    return q / q.sum()


def pure_nash(game: Game) -> list[tuple[int, int]]:
    A, B = game.own, game.other.T
    out = []
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if A[i, j] >= A[:, j].max() and B[i, j] >= B[i, :].max():
                out.append((i, j))
    return out


def structural_features(game: Game) -> np.ndarray:
    """Hand-built strategic features for a 2-action decision problem (used by
    feature-based baselines and to define game families)."""
    g = game.scaled()
    A, O = g.own, g.other
    uni_o = np.full(A.shape[1], 1 / A.shape[1])
    uni_s = np.full(A.shape[0], 1 / A.shape[0])
    l1_self = best_response(A, uni_o)                 # level-1 choice
    l1_other = best_response(O, uni_s)
    l2_self = best_response(A, l1_other)              # level-2 choice
    ds_self = strictly_dominant_action(A) is not None
    ds_other = strictly_dominant_action(O) is not None
    psne = pure_nash(game)
    pne_self = np.zeros(A.shape[0])
    for i, _ in psne:
        pne_self[i] += 1 / max(len(psne), 1)
    gap = A @ uni_o
    return np.concatenate([
        A.ravel(), O.ravel(),
        l1_self, l2_self, pne_self,
        [float(ds_self), float(ds_other), float(len(psne)), gap[0] - gap[-1],
         A.max() - A.min(), O.max() - O.min()],
    ])
