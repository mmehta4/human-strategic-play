from __future__ import annotations

import numpy as np

from ..data.schema import Record


class Model:
    """Interface every model follows: fit on training records, then predict a
    probability distribution over the decision maker's actions for each record."""
    name = "Model"

    def fit(self, records: list[Record]) -> "Model":
        raise NotImplementedError

    def predict(self, records: list[Record]) -> np.ndarray:
        raise NotImplementedError

    def params(self) -> dict:
        return {}


def stack(records: list[Record]) -> tuple[np.ndarray, np.ndarray]:
    """Stack payoff matrices: own (G, m, n) and other (G, n, m). All records must
    share an action count."""
    own = np.stack([r.game.own for r in records])
    other = np.stack([r.game.other for r in records])
    return own, other


def grouped(records: list[Record]):
    """Split records by game size: [(indices, own, other, freq), ...]. Lets models that
    work on batches of equal-sized games fit and predict across mixed sizes."""
    by_shape: dict[tuple, list[int]] = {}
    for i, r in enumerate(records):
        by_shape.setdefault(r.game.own.shape, []).append(i)
    out = []
    for idx in by_shape.values():
        own, other = stack([records[i] for i in idx])
        out.append((idx, own, other, np.array([records[i].freq for i in idx])))
    return out


def grouped_loss(groups, fn) -> float:
    """Mean cross-entropy over all problems; fn(own, other) -> predictions for one group."""
    total, n = 0.0, 0
    for idx, own, other, freq in groups:
        p = fn(own, other)
        total += float(-(freq * np.log(np.clip(p, 1e-9, 1))).sum())
        n += len(idx)
    return total / n


def assemble(n_records: int, groups, fn):
    """Predictions back in record order: an array when all games share a size, else a list."""
    out = [None] * n_records
    for idx, own, other, _ in groups:
        for i, row in zip(idx, fn(own, other)):
            out[i] = row
    return np.array(out) if len(groups) == 1 else out
