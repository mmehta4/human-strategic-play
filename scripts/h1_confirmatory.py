"""H1 (confirmatory, docs/evaluation_protocol.md v3): does a model developed only on games2p2k
transfer to twelve other studies of human play?

Trains every model once on all 2,416 games2p2k problems and tests, unchanged, on the 384 decision
problems of the 12-study dataset (`hsb.data.load("wlb")`). Payoffs are standardised per game and
per player in both. Neural models are averaged over seeds 0-4.

    scripts/fetch_wlb_data.sh
    python scripts/h1_confirmatory.py

Rule: hybrid v1's mean NLL must be lower than both the ElementaryNet-style model and quantal
level-k; one-sided paired bootstrap (10,000 resamples, seed 0), Holm over the two comparisons,
alpha 0.05. Run this ONCE. Don't tune anything after.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hsb.data import load  # noqa: E402
from hsb.data.wlb import standardise_records  # noqa: E402
from hsb.eval.metrics import score_ragged  # noqa: E402
from hsb.models.base import Model  # noqa: E402
from hsb.models.neural import ElementaryQCH, ElementaryQLk, HybridQLkV1  # noqa: E402
from hsb.models.structural import CognitiveHierarchy, QuantalLevelK, Uniform  # noqa: E402

TEST = "Hybrid v1 (any size, structural + residual)"
PRIMARY = ["ElementaryNet-style QLk (reimplementation)", "Quantal level-k"]
SECONDARY = ["ElementaryNet-style QCH (closer reimplementation)", "Cognitive hierarchy"]
SEEDS = range(5)
LARGE = 10  # games with more actions than this are dropped in the sensitivity analysis


class SeedAverage(Model):
    """Train one model per seed; predict the average of their predicted distributions."""

    def __init__(self, make):
        self.models = [make(seed) for seed in SEEDS]
        self.name = self.models[0].name

    def fit(self, records):
        for m in self.models:
            m.fit(records)
        return self

    def predict(self, records):
        preds = [m.predict(records) for m in self.models]
        return [np.mean([p[i] for p in preds], axis=0) for i in range(len(records))]


def models():
    return [
        Uniform(), QuantalLevelK(), CognitiveHierarchy(),
        SeedAverage(lambda s: ElementaryQLk(scale=1.0, seed=s)),
        SeedAverage(lambda s: ElementaryQCH(scale=1.0, seed=s)),
        SeedAverage(lambda s: HybridQLkV1(scale=1.0, seed=s)),
    ]


def ce(f, p):
    return float(-(f * np.log(np.clip(p, 1e-9, 1))).sum())


def boot_one_sided(d, n, rng):
    boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return {"mean_diff": float(d.mean()), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "p_one_sided": float((boots >= 0).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=10000)
    args = ap.parse_args()

    train = standardise_records(load("games2p2k"))
    test = load("wlb")
    per_problem, summary, rows = {}, {}, []
    for m in models():
        m.fit(train)
        preds = list(m.predict(test))
        per_problem[m.name] = np.array([ce(r.freq, p) for r, p in zip(test, preds)])
        summary[m.name] = score_ragged([r.freq for r in test], preds, [r.n for r in test])
        rows += [{"model": m.name, "problem_id": r.problem_id, "study": r.family, "n": r.n,
                  "actions": len(r.freq), "nll": v} for r, v in zip(test, per_problem[m.name])]
        print(f"  {m.name:52s} NLL {summary[m.name]['nll']:.4f}", flush=True)

    rng = np.random.default_rng(0)
    primary = {b: boot_one_sided(per_problem[TEST] - per_problem[b], args.boot, rng) for b in PRIMARY}
    ordered = sorted(primary.items(), key=lambda kv: kv[1]["p_one_sided"])
    holm, still = {}, True
    for i, (b, v) in enumerate(ordered):  # step-down Holm
        still = still and v["p_one_sided"] <= 0.05 / (len(ordered) - i)
        holm[b] = still
    n_sig = sum(holm.values())
    verdict = {2: "HOLDS", 1: "HOLDS PARTLY", 0: "FAILS"}[n_sig]

    secondary = {b: boot_one_sided(per_problem[TEST] - per_problem[b], args.boot, rng) for b in SECONDARY}
    w = np.array([r.n for r in test])
    weighted = {k: float((v * w).sum() / w.sum()) for k, v in per_problem.items()}
    small = np.array([len(r.freq) <= LARGE for r in test])
    without_large = {k: float(v[small].mean()) for k, v in per_problem.items()}
    without_large_tests = {b: boot_one_sided(per_problem[TEST][small] - per_problem[b][small], args.boot, rng)
                           for b in PRIMARY}
    studies = sorted({r.family for r in test})
    by_study = {s: {"n_problems": int(sum(r.family == s for r in test)),
                    **{k: float(v[[r.family == s for r in test]].mean()) for k, v in per_problem.items()}}
                for s in studies}

    out = {"protocol": "docs/evaluation_protocol.md v3", "n_problems": len(test), "summary": summary,
           "primary": primary, "holm_significant": holm, "verdict": verdict,
           "secondary": {"comparisons": secondary, "nll_weighted_by_choices": weighted,
                         "without_large_games": {"nll": without_large, "tests": without_large_tests,
                                                 "n_problems": int(small.sum())},
                         "by_study": by_study}}
    (ROOT / "results" / "h1_confirmatory.json").write_text(json.dumps(out, indent=2))
    (ROOT / "results" / "predictions").mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(ROOT / "results" / "predictions" / "h1_wlb.csv.gz", index=False)
    print(json.dumps({"primary": primary, "holm": holm}, indent=2))
    print(f"\nH1 {verdict} on {len(test)} decision problems from 12 studies")


if __name__ == "__main__":
    main()
