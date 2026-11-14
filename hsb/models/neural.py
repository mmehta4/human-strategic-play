"""Neural models (PyTorch).

MLP          - black-box network from payoffs to choice probabilities, the kind of
               model Zhu et al. (2025) showed beats behavioural theories.
HybridQLk    - our hybrid v0: a quantal level-k core whose level shares and precision
               are set per game by a network reading the payoffs, plus a small
               penalised neural residual on top. With `residual=False` it is purely
               structural, so every prediction decomposes into "x% level-0,
               y% level-1, ..." with a stated precision.

Both train on all action relabelings of each game (behaviour should not depend on
which action is listed first) and stop early on a validation split drawn from the
training games, never from the test games.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ..data.schema import Record
from .base import Model, stack

torch.set_default_dtype(torch.float64)


def _augment(own, other, freq):
    """All relabelings of own and other actions (2x2 -> 4 copies)."""
    m, n = own.shape[1], own.shape[2]
    outs = []
    for pi in _perms(m):
        for pj in _perms(n):
            outs.append((own[:, pi][:, :, pj], other[:, pj][:, :, pi], freq[:, pi]))
    return [np.concatenate(x) for x in zip(*outs)]


def _perms(k):
    if k == 2:
        return [np.array([0, 1]), np.array([1, 0])]
    return [np.arange(k)]  # larger games: identity only (cheap and still correct)


class _TorchModel(Model):
    def __init__(self, hidden=128, lr=3e-3, weight_decay=1e-5, max_steps=4000,
                 patience=20, val_frac=0.1, seed=0, scale=50.0):
        self.hidden, self.lr, self.wd = hidden, lr, weight_decay
        self.max_steps, self.patience, self.val_frac = max_steps, patience, val_frac
        self.seed, self.scale = seed, scale
        self.net = None

    # subclasses: build(m, n) -> nn.Module ; forward(net, own, other) -> (probs, penalty)
    # Records are grouped by action counts, so a size-agnostic network (HybridQLkV1) can
    # train on 2x2, 3x3 and larger games together. Fixed-size networks need one shape.
    def _groups(self, records, augment):
        by_shape = {}
        for r in records:
            by_shape.setdefault(r.game.own.shape, []).append(r)
        out = []
        for recs in by_shape.values():
            own, other = stack(recs)
            freq = np.array([r.freq for r in recs])
            own, other = own / self.scale, other / self.scale
            if augment:
                own, other, freq = _augment(own, other, freq)
            out.append(tuple(torch.tensor(a) for a in (own, other, freq)))
        return out

    def _loss(self, groups):
        total, n, pen_sum = 0.0, 0, 0.0
        for own, other, freq in groups:
            p, pen = self.forward(self.net, own, other)
            total = total + (-(freq * torch.log(p.clamp_min(1e-9))).sum(1)).sum()
            n += len(freq)
            pen_sum = pen_sum + pen
        return total / n, pen_sum / len(groups)

    def fit(self, records: list[Record]):
        rng = np.random.default_rng(self.seed)
        torch.manual_seed(self.seed)
        games = np.array(sorted({r.game_id for r in records}))
        val_games = set(rng.choice(games, size=max(1, int(len(games) * self.val_frac)), replace=False))
        train = self._groups([r for r in records if r.game_id not in val_games], augment=True)
        val = self._groups([r for r in records if r.game_id in val_games], augment=False)

        own0 = train[0][0]
        self.net = self.build(own0.shape[1], own0.shape[2])
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.wd)
        best, best_state, bad = np.inf, None, 0
        for step in range(self.max_steps):
            self.net.train()
            ce, pen = self._loss(train)
            loss = ce + pen
            opt.zero_grad(); loss.backward(); opt.step()
            if step % 25 == 0:
                self.net.eval()
                with torch.no_grad():
                    vl = float(self._loss(val)[0])
                if vl < best - 1e-6:
                    best, bad = vl, 0
                    best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
                else:
                    bad += 1
                    if bad >= self.patience:
                        break
        self.net.load_state_dict(best_state)
        self.val_loss_, self.steps_ = best, step
        return self

    def predict(self, records):
        """Array (G, m) when every record has m actions; otherwise a list of arrays."""
        self.net.eval()
        out = [None] * len(records)
        by_shape = {}
        for i, r in enumerate(records):
            by_shape.setdefault(r.game.own.shape, []).append(i)
        with torch.no_grad():
            for idx in by_shape.values():
                own, other = stack([records[i] for i in idx])
                p, _ = self.forward(self.net, torch.tensor(own / self.scale), torch.tensor(other / self.scale))
                for i, row in zip(idx, p.numpy()):
                    out[i] = row
        return np.array(out) if len(by_shape) == 1 else out


# ------------------------------------------------------------------- MLP

class MLP(_TorchModel):
    name = "MLP (black box)"

    def build(self, m, n):
        d = 2 * m * n
        return nn.Sequential(nn.Linear(d, self.hidden), nn.ReLU(),
                             nn.Linear(self.hidden, self.hidden), nn.ReLU(),
                             nn.Linear(self.hidden, m))

    def forward(self, net, own, other):
        x = torch.cat([own.flatten(1), other.flatten(1)], 1)
        return torch.softmax(net(x), 1), 0.0


# ------------------------------------------------------------------- Hybrid

class _HybridNet(nn.Module):
    def __init__(self, m, n, hidden, K, residual):
        super().__init__()
        d = 2 * m * n
        self.K, self.residual = K, residual
        self.body = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.levels = nn.Linear(hidden, K + 1)   # level shares
        self.prec = nn.Linear(hidden, 2)         # log precision: own response, belief about opponent
        self.res = nn.Linear(hidden, m) if residual else None
        nn.init.zeros_(self.levels.weight); nn.init.zeros_(self.prec.weight)
        with torch.no_grad():
            self.prec.bias.fill_(np.log(5.0))
        if residual:
            nn.init.zeros_(self.res.weight); nn.init.zeros_(self.res.bias)

    def forward(self, own, other):
        h = self.body(torch.cat([own.flatten(1), other.flatten(1)], 1))
        w = torch.softmax(self.levels(h), 1)                   # (G, K+1)
        lam = torch.exp(self.prec(h).clamp(-6, 6))             # (G, 2)
        lam_s, lam_o = lam[:, :1], lam[:, 1:]
        G, m, n = own.shape
        s = [own.new_full((G, m), 1 / m)]
        o = [own.new_full((G, n), 1 / n)]
        for k in range(1, self.K + 1):
            s.append(torch.softmax(lam_s * torch.einsum("gmn,gn->gm", own, o[k - 1]), 1))
            o.append(torch.softmax(lam_o * torch.einsum("gnm,gm->gn", other, s[k - 1]), 1))
        p = torch.einsum("gk,kgm->gm", w, torch.stack(s))
        r = self.res(h) if self.residual else None
        return p, w, lam, r


class _ElementaryNet(nn.Module):
    """Level-0 scores each action only from that action's own consequences (its payoffs
    and the opponent's payoffs in that row), never from the opponent's incentives, so it
    cannot express strategic reasoning. Levels 1..K then logit-respond as in quantal
    level-k, with global level shares and precision."""

    def __init__(self, hidden, K):
        super().__init__()
        self.K = K
        self.score = nn.Sequential(nn.Linear(8, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.shares = nn.Parameter(torch.zeros(K + 1))
        self.log_lam = nn.Parameter(torch.tensor(np.log(5.0)))

    @staticmethod
    def _feats(own_rows, their_rows):
        # own_rows (G, a, b): my payoff for my action a vs their b
        # their_rows (G, a, b): their payoff when I play a and they play b
        s = lambda x: torch.stack([x.max(-1).values, x.min(-1).values, x.mean(-1)], -1)
        own_mean = own_rows.mean(-1, keepdim=True)
        return torch.cat([s(own_rows), s(their_rows),
                          own_mean - own_rows.mean((1, 2), keepdim=True)[..., 0:1].expand_as(own_mean),
                          (own_rows - their_rows).abs().mean(-1, keepdim=True)], -1)

    def level0(self, own, other):
        # decision maker: rows of own; opponent's payoffs when I play i are other[:, :, i]
        f = self._feats(own, other.transpose(1, 2))
        return torch.softmax(self.score(f).squeeze(-1), 1)

    def forward(self, own, other):
        s = [self.level0(own, other)]
        o = [self.level0(other, own)]
        lam = torch.exp(self.log_lam)
        for k in range(1, self.K + 1):
            s.append(torch.softmax(lam * torch.einsum("gmn,gn->gm", own, o[k - 1]), 1))
            o.append(torch.softmax(lam * torch.einsum("gnm,gm->gn", other, s[k - 1]), 1))
        w = torch.softmax(self.shares, 0)
        return torch.einsum("k,kgm->gm", w, torch.stack(s))


class ElementaryQLk(_TorchModel):
    """Our reimplementation, in the spirit of ElementaryNet (d'Eon, Murad, Leyton-Brown &
    Wright, AAAI 2026): a learned but provably non-strategic level-0, with quantal
    level-k reasoning on top. Not the authors' code; treat its numbers as an
    approximation of that baseline until checked against their release."""
    name = "ElementaryNet-style QLk (reimplementation)"

    def __init__(self, K=3, **kw):
        super().__init__(**kw)
        self.K = K

    def build(self, m, n):
        return _ElementaryNet(self.hidden, self.K)

    def forward(self, net, own, other):
        return net(own, other), 0.0


class HybridQLk(_TorchModel):
    """Context-dependent quantal level-k with an optional penalised residual."""

    def __init__(self, K=3, residual=True, residual_penalty=1e-2, **kw):
        super().__init__(**kw)
        self.K, self.residual, self.residual_penalty = K, residual, residual_penalty
        self.name = "Hybrid v0 (structural + residual)" if residual else "Hybrid v0 (structural only)"

    def build(self, m, n):
        return _HybridNet(m, n, self.hidden, self.K, self.residual)

    def forward(self, net, own, other):
        p, w, lam, r = net(own, other)
        if r is None:
            return p, 0.0
        logits = torch.log(p.clamp_min(1e-9)) + r
        return torch.softmax(logits, 1), self.residual_penalty * (r ** 2).mean()

    def explain(self, records):
        """Per-problem level shares and precisions, for interpretation."""
        own, other = stack(records)
        with torch.no_grad():
            _, w, lam, r = self.net(torch.tensor(own / self.scale), torch.tensor(other / self.scale))
        return {"level_shares": w.numpy(), "precision": lam.numpy(),
                "residual": None if r is None else r.numpy()}


# ------------------------------------------------------------------- Hybrid v1 (any game size)

class _EquivariantLayer(nn.Module):
    """GameNet-style layer: each payoff cell is updated from itself plus the mean of its
    row, its column and the whole matrix. Relabelling actions permutes the output the
    same way, and the same weights work for any number of actions."""

    def __init__(self, c_in, c_out):
        super().__init__()
        self.lin = nn.Linear(4 * c_in, c_out)

    def forward(self, h):  # h: (G, C, m, n)
        G, C, m, n = h.shape
        row = h.mean(3, keepdim=True).expand(-1, -1, -1, n)
        col = h.mean(2, keepdim=True).expand(-1, -1, m, -1)
        glob = h.mean((2, 3), keepdim=True).expand(-1, -1, m, n)
        x = torch.cat([h, row, col, glob], 1).permute(0, 2, 3, 1)
        return torch.relu(self.lin(x)).permute(0, 3, 1, 2)


class _HybridNetV1(nn.Module):
    def __init__(self, hidden, K, residual, layers=3):
        super().__init__()
        self.K, self.residual = K, residual
        dims = [2] + [hidden] * layers
        self.enc = nn.ModuleList(_EquivariantLayer(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.levels = nn.Linear(hidden, K + 1)
        self.prec = nn.Linear(hidden, 2)
        self.res = nn.Linear(hidden, 1) if residual else None
        nn.init.zeros_(self.levels.weight); nn.init.zeros_(self.prec.weight)
        with torch.no_grad():
            self.prec.bias.fill_(np.log(5.0))
        if residual:
            nn.init.zeros_(self.res.weight); nn.init.zeros_(self.res.bias)

    def forward(self, own, other):
        h = torch.stack([own, other.transpose(1, 2)], 1)   # (G, 2, m, n): both payoffs per cell
        for layer in self.enc:
            h = layer(h)
        e = h.mean((2, 3))                                  # game embedding
        w = torch.softmax(self.levels(e), 1)
        lam = torch.exp(self.prec(e).clamp(-6, 6))
        lam_s, lam_o = lam[:, :1], lam[:, 1:]
        G, m, n = own.shape
        s = [own.new_full((G, m), 1 / m)]
        o = [own.new_full((G, n), 1 / n)]
        for k in range(1, self.K + 1):
            s.append(torch.softmax(lam_s * torch.einsum("gmn,gn->gm", own, o[k - 1]), 1))
            o.append(torch.softmax(lam_o * torch.einsum("gnm,gm->gn", other, s[k - 1]), 1))
        p = torch.einsum("gk,kgm->gm", w, torch.stack(s))
        r = self.res(h.mean(3).transpose(1, 2)).squeeze(-1) if self.residual else None  # per own action
        return p, w, lam, r


# ------------------------------------------------------------------- ElementaryNet, closer version

class _PotentialResponse(nn.Module):
    """One ElementaryNet level-0 component: a learned linear potential collapses each outcome's
    two payoffs to one number, then a GameNet-style response network maps that potential
    matrix to a distribution over the decision maker's actions."""

    def __init__(self, hidden, layers=2):
        super().__init__()
        self.coef = nn.Parameter(torch.randn(2) * 0.5 + torch.tensor([1.0, 0.0]))
        dims = [1] + [hidden] * layers
        self.enc = nn.ModuleList(_EquivariantLayer(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.out = nn.Linear(hidden, 1)

    def forward(self, own, opp):  # own, opp: (G, m, n) payoffs to me and to them, my rows
        c = self.coef / self.coef.norm()
        h = (c[0] * own + c[1] * opp).unsqueeze(1)
        for layer in self.enc:
            h = layer(h)
        return torch.softmax(self.out(h.mean(3).transpose(1, 2)).squeeze(-1), 1)


class _ElementaryNetQCH(nn.Module):
    """Level-0 is a convex combination of potential response networks; levels 1..K follow a
    quantal cognitive hierarchy with Poisson level shares: level k responds to the
    renormalised mix of the opponent's levels 0..k-1. Precision and tau are global."""

    def __init__(self, hidden, K, n_potentials=4):
        super().__init__()
        self.K = K
        self.parts = nn.ModuleList(_PotentialResponse(hidden) for _ in range(n_potentials))
        self.mix = nn.Parameter(torch.zeros(n_potentials))
        self.log_tau = nn.Parameter(torch.tensor(np.log(1.5)))
        self.log_lam = nn.Parameter(torch.tensor(np.log(5.0)))

    def level0(self, own, opp):
        w = torch.softmax(self.mix, 0)
        return torch.einsum("p,pgm->gm", w, torch.stack([f(own, opp) for f in self.parts]))

    def forward(self, own, other):
        tau, lam = torch.exp(self.log_tau), torch.exp(self.log_lam)
        k = torch.arange(self.K + 1, dtype=own.dtype)
        f = torch.exp(k * torch.log(tau) - tau - torch.lgamma(k + 1))
        f = f / f.sum()
        s = [self.level0(own, other.transpose(1, 2))]
        o = [self.level0(other, own.transpose(1, 2))]
        for lv in range(1, self.K + 1):
            g = f[:lv] / f[:lv].sum()
            belief_o = torch.einsum("k,kgn->gn", g, torch.stack(o))
            belief_s = torch.einsum("k,kgm->gm", g, torch.stack(s))
            s.append(torch.softmax(lam * torch.einsum("gmn,gn->gm", own, belief_o), 1))
            o.append(torch.softmax(lam * torch.einsum("gnm,gm->gn", other, belief_s), 1))
        return torch.einsum("k,kgm->gm", f, torch.stack(s))


class ElementaryQCH(_TorchModel):
    """A closer reimplementation of ElementaryNet (d'Eon, Murad, Leyton-Brown & Wright): learned
    linear potentials feeding GameNet-style response networks for level-0, and a Poisson
    quantal cognitive hierarchy on top. Still not the authors' code; trained with our NLL
    loss and early stopping like every other neural model here."""
    name = "ElementaryNet-style QCH (closer reimplementation)"

    def __init__(self, K=3, **kw):
        super().__init__(**kw)
        self.K = K

    def build(self, m, n):
        return _ElementaryNetQCH(self.hidden // 2, self.K)

    def forward(self, net, own, other):
        return net(own, other), 0.0


class HybridQLkV1(HybridQLk):
    """Hybrid v1: same structural core and residual as v0, but the encoder works on games
    of any size and treats every action symmetrically, so one model can train on 2x2,
    3x3 and larger generated mechanisms together."""

    def __init__(self, K=3, residual=True, residual_penalty=1e-2, **kw):
        super().__init__(K=K, residual=residual, residual_penalty=residual_penalty, **kw)
        self.name = "Hybrid v1 (any size, structural + residual)" if residual else "Hybrid v1 (any size, structural only)"

    def build(self, m, n):
        return _HybridNetV1(self.hidden, self.K, self.residual)
