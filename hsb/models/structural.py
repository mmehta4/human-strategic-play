"""Classic behavioural game theory baselines, fitted by maximum likelihood.

Every model predicts a probability distribution over the decision maker's actions
for each decision problem. Fitting minimises mean cross-entropy between the
empirical choice shares and the prediction, weighting every problem equally.

Implemented (in order of the baseline ladder):
    Uniform            - every action equally likely
    NashMix            - average Nash equilibrium strategy plus an error rate
    LogitQRE           - logit quantal response equilibrium (McKelvey & Palfrey 1995)
    QuantalLevelK      - level-k with logit noise and fitted level shares (Stahl & Wilson;
                         Wright & Leyton-Brown's QLk family)
    CognitiveHierarchy - Poisson cognitive hierarchy with logit responses
                         (Camerer, Ho & Chong 2004)

Models batch problems of equal size and fit jointly across sizes (2x2, 3x3, 11x11 ...).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson

from ..data.schema import Record
from ..games import nash_equilibria
from .base import Model, assemble, grouped, grouped_loss

EPS = 1e-9


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def _logit_resp(payoff: np.ndarray, opp_mix: np.ndarray, lam) -> np.ndarray:
    """payoff (G, m, n), opp_mix (G, n) -> (G, m)."""
    lam = np.asarray(lam).reshape(-1, 1) if np.ndim(lam) else lam
    return _softmax(lam * np.einsum("gmn,gn->gm", payoff, opp_mix))


# --------------------------------------------------------------------- Uniform

class Uniform(Model):
    name = "Uniform"

    def fit(self, records):
        return self

    def predict(self, records):
        g = grouped(records)
        return assemble(len(records), g, lambda own, other: np.full(own.shape[:2], 1.0 / own.shape[1]))


# --------------------------------------------------------------------- Nash

class NashMix(Model):
    """Average over all Nash equilibria of the decision maker's strategy, mixed with
    a fitted error rate eps: p = (1 - eps) * NE + eps * uniform."""
    name = "Nash (+ error)"

    def __init__(self):
        self.eps = 0.5
        self._cache: dict[str, np.ndarray] = {}

    def _ne(self, records):
        out = []
        for r in records:
            if r.problem_id not in self._cache:
                eqs = nash_equilibria(r.game)
                self._cache[r.problem_id] = (np.mean([e[0] for e in eqs], axis=0)
                                             if eqs else np.full(r.game.n_actions, 1 / r.game.n_actions))
            out.append(self._cache[r.problem_id])
        return out

    @staticmethod
    def _mix(ne, eps):
        return [(1 - eps) * x + eps / len(x) for x in ne]

    def fit(self, records):
        ne, freq = self._ne(records), [r.freq for r in records]
        def loss(e):
            mixed = self._mix(ne, float(np.clip(e[0], 1e-4, 1)))
            return float(np.mean([-(f * np.log(np.clip(p, EPS, 1))).sum() for f, p in zip(freq, mixed)]))
        res = minimize(loss, x0=[0.3], bounds=[(1e-4, 1.0)], method="L-BFGS-B")
        self.eps = float(res.x[0])
        return self

    def predict(self, records):
        out = self._mix(self._ne(records), self.eps)
        return np.array(out) if len({len(x) for x in out}) == 1 else out

    def params(self):
        return {"eps": self.eps}


# --------------------------------------------------------------------- QRE

def logit_qre(own: np.ndarray, other: np.ndarray, lam: float, steps: int = 40,
              iters: int = 60, damping: float = 0.5) -> np.ndarray:
    """Logit QRE for a batch of games, following the principal branch by homotopy
    from lam = 0 (uniform play) up to `lam`. Returns the decision maker's mix (G, m)."""
    G, m, n = own.shape
    x = np.full((G, m), 1 / m)   # decision maker
    y = np.full((G, n), 1 / n)   # opponent
    for lam_t in np.linspace(0, lam, steps + 1)[1:]:
        for _ in range(iters):
            x_new = _logit_resp(own, y, lam_t)
            y_new = _logit_resp(other, x, lam_t)
            dx = np.abs(x_new - x).max()
            x = damping * x + (1 - damping) * x_new
            y = damping * y + (1 - damping) * y_new
            if dx < 1e-10:
                break
    return x


class LogitQRE(Model):
    name = "Logit QRE"

    def __init__(self, lam: float = 0.1):
        self.lam = lam

    def fit(self, records):
        g = grouped(records)
        grid = np.geomspace(1e-3, 3.0, 25)
        losses = [grouped_loss(g, lambda o, t, lam=lam: logit_qre(o, t, lam, steps=15, iters=30)) for lam in grid]
        g0 = grid[int(np.argmin(losses))]
        res = minimize(lambda v: grouped_loss(g, lambda o, t: logit_qre(o, t, float(np.exp(v[0])))),
                       x0=[np.log(g0)], method="Nelder-Mead",
                       options={"xatol": 1e-3, "fatol": 1e-6, "maxiter": 40})
        self.lam = float(np.exp(res.x[0]))
        return self

    def predict(self, records):
        return assemble(len(records), grouped(records), lambda o, t: logit_qre(o, t, self.lam))

    def params(self):
        return {"lambda": self.lam}


# --------------------------------------------------------------------- level-k

def level_strategies(own, other, lam, K: int):
    """Logit level-k strategies for both sides of each game.
    Returns lists self_k[k] (G, m) and other_k[k] (G, n) for k = 0..K, where level 0
    is uniform and level k logit-responds to the opponent's level k-1."""
    G, m, n = own.shape
    s = [np.full((G, m), 1 / m)]
    o = [np.full((G, n), 1 / n)]
    for k in range(1, K + 1):
        s.append(_logit_resp(own, o[k - 1], lam))
        o.append(_logit_resp(other, s[k - 1], lam))
    return s, o


class QuantalLevelK(Model):
    """Population of levels 0..K with fitted shares; level k >= 1 logit-responds,
    with precision lambda, to the opponent's level k-1."""
    name = "Quantal level-k"

    def __init__(self, K: int = 3):
        self.K = K
        self.w = np.full(K + 1, 1 / (K + 1))
        self.lam = 0.2

    def _predict(self, own, other, w, lam):
        s, _ = level_strategies(own, other, lam, self.K)
        return np.einsum("k,kgm->gm", w, np.stack(s))

    def fit(self, records):
        g = grouped(records)

        def loss(v):
            w = _softmax(v[: self.K + 1][None])[0]
            return grouped_loss(g, lambda o, t: self._predict(o, t, w, float(np.exp(v[-1]))))

        bounds = [(-20, 20)] * (self.K + 1) + [(np.log(1e-4), np.log(50.0))]
        best = None
        for lam0 in (0.05, 0.2, 0.8):
            r = minimize(loss, x0=np.r_[np.zeros(self.K + 1), np.log(lam0)], method="L-BFGS-B",
                         bounds=bounds)
            if best is None or r.fun < best.fun:
                best = r
        self.w = _softmax(best.x[: self.K + 1][None])[0]
        self.lam = float(np.exp(best.x[-1]))
        return self

    def predict(self, records):
        return assemble(len(records), grouped(records), lambda o, t: self._predict(o, t, self.w, self.lam))

    def params(self):
        return {"level_shares": np.round(self.w, 3).tolist(), "lambda": self.lam}


# --------------------------------------------------------------------- cognitive hierarchy

class CognitiveHierarchy(Model):
    """Poisson cognitive hierarchy: level k responds (logit, precision lambda) to the
    normalised Poisson(tau) mix of levels 0..k-1. Prediction = Poisson-weighted mix."""
    name = "Cognitive hierarchy"

    def __init__(self, K: int = 6):
        self.K = K
        self.tau, self.lam = 1.5, 0.2

    def _predict(self, own, other, tau, lam):
        G, m, n = own.shape
        f = poisson.pmf(np.arange(self.K + 1), tau)
        s = [np.full((G, m), 1 / m)]
        o = [np.full((G, n), 1 / n)]
        for k in range(1, self.K + 1):
            wk = f[:k] / f[:k].sum()
            belief_o = np.einsum("k,kgn->gn", wk, np.stack(o))
            belief_s = np.einsum("k,kgm->gm", wk, np.stack(s))
            s.append(_logit_resp(own, belief_o, lam))
            o.append(_logit_resp(other, belief_s, lam))
        f = f / f.sum()
        return np.einsum("k,kgm->gm", f, np.stack(s))

    def fit(self, records):
        g = grouped(records)
        loss = lambda v: grouped_loss(g, lambda o, t: self._predict(o, t, float(np.exp(v[0])), float(np.exp(v[1]))))
        bounds = [(np.log(0.01), np.log(20.0)), (np.log(1e-4), np.log(50.0))]
        best = None
        for t0, l0 in ((0.5, 0.1), (1.5, 0.2), (3.0, 0.5)):
            r = minimize(loss, x0=[np.log(t0), np.log(l0)], method="L-BFGS-B", bounds=bounds)
            if best is None or r.fun < best.fun:
                best = r
        self.tau, self.lam = float(np.exp(best.x[0])), float(np.exp(best.x[1]))
        return self

    def predict(self, records):
        return assemble(len(records), grouped(records), lambda o, t: self._predict(o, t, self.tau, self.lam))

    def params(self):
        return {"tau": self.tau, "lambda": self.lam}
