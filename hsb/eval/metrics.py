"""Scoring. The primary metric is cross-entropy (negative log-likelihood per choice,
every decision problem weighted equally), a strictly proper scoring rule. MSE and
R^2 on the first action's share are also reported because Zhu et al. (2025)
report them, so our numbers can be compared with theirs."""
from __future__ import annotations

import numpy as np

EPS = 1e-9


def cross_entropy(freq: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Per-problem cross-entropy in nats."""
    return -(freq * np.log(np.clip(pred, EPS, 1))).sum(axis=1)


def entropy(freq: np.ndarray) -> np.ndarray:
    return -(freq * np.log(np.clip(freq, EPS, 1))).sum(axis=1)


def score(freq: np.ndarray, pred: np.ndarray, n: np.ndarray | None = None) -> dict:
    ce = cross_entropy(freq, pred)
    uni = np.log(freq.shape[1])
    ent = entropy(freq)
    # Sampling-noise floor: with n choices per problem, even the true distribution
    # scores worse than the empirical entropy by about (m - 1) / (2n) nats.
    floor = ent + ((freq.shape[1] - 1) / (2 * n) if n is not None else 0.0)
    mse = ((freq[:, 0] - pred[:, 0]) ** 2).mean()
    r2 = 1 - mse / freq[:, 0].var()
    return {
        "nll": float(ce.mean()),
        "nll_se": float(ce.std(ddof=1) / np.sqrt(len(ce))),
        "mse": float(mse),
        "r2": float(r2),
        "accuracy": float((pred.argmax(1) == freq.argmax(1)).mean()),
        # share of the gap between guessing uniformly and the noise floor the model closes
        "completeness": float((uni - ce.mean()) / (uni - floor.mean())),
        "n_problems": int(len(ce)),
    }


def paired_difference(freq, pred_a, pred_b) -> dict:
    """Mean NLL difference (a - b) with a paired standard error and a bootstrap 95% CI.
    Negative means model a is better."""
    d = cross_entropy(freq, pred_a) - cross_entropy(freq, pred_b)
    rng = np.random.default_rng(0)
    boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
    return {"mean": float(d.mean()), "se": float(d.std(ddof=1) / np.sqrt(len(d))),
            "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}


def score_ragged(freqs: list, preds: list, ns: list) -> dict:
    """Like `score`, for problems whose action counts differ (2x2 next to 11x11).
    Completeness pools the uniform baseline, the model and the noise floor across problems."""
    ce = np.array([float(-(f * np.log(np.clip(p, EPS, 1))).sum()) for f, p in zip(freqs, preds)])
    uni = np.array([np.log(len(f)) for f in freqs])
    floor = np.array([float(-(f * np.log(np.clip(f, EPS, 1))).sum()) + (len(f) - 1) / (2 * n)
                      for f, n in zip(freqs, ns)])
    return {"nll": float(ce.mean()), "nll_se": float(ce.std(ddof=1) / np.sqrt(len(ce))) if len(ce) > 1 else 0.0,
            "completeness": float((uni.mean() - ce.mean()) / (uni.mean() - floor.mean())),
            "accuracy": float(np.mean([np.argmax(p) == np.argmax(f) for f, p in zip(freqs, preds)])),
            "n_problems": int(len(ce))}
