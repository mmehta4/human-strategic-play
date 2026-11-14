"""Power for the confirmatory H1 test: how often would a study of N new decision problems
detect the hybrid's advantage, if the true per-problem gains look like the
held-out-family gains seen in development?

    python scripts/power_analysis.py --baseline "ElementaryNet" --n 300 600 900

Resamples development per-problem NLL differences (hybrid minus baseline) under the
held-out-family protocol, then runs the registered test (one-sided bootstrap, alpha 0.05)
on each simulated study. Development effects are optimistic, so the script also reports
power if the true effect is only half as large.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REF = "Hybrid v0 (structural + residual)"


def nll(df):
    return -(df.f0 * np.log(df.p0.clip(1e-9)) + df.f1 * np.log(df.p1.clip(1e-9)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="ElementaryNet")
    ap.add_argument("--n", type=int, nargs="+", default=[150, 300, 600])
    ap.add_argument("--sims", type=int, default=400)
    args = ap.parse_args()
    df = pd.read_csv(ROOT / "results" / "predictions" / "family.csv.gz")
    base_name = next(m for m in df.model.unique() if args.baseline.lower() in m.lower())
    a = df[df.model == REF].set_index("problem_id")
    b = df[df.model == base_name].set_index("problem_id").loc[a.index]
    d = (nll(a) - nll(b)).to_numpy()
    print(f"Development: {len(d)} problems, mean difference {d.mean():+.4f} nats (negative favours the hybrid)")
    rng = np.random.default_rng(0)
    for shrink in (1.0, 0.5):
        dd = d - (1 - shrink) * d.mean()
        for n in args.n:
            hits = 0
            for _ in range(args.sims):
                s = rng.choice(dd, size=n, replace=True)
                boots = rng.choice(s, size=(500, n), replace=True).mean(1)
                hits += np.quantile(boots, 0.95) < 0  # one-sided: hybrid better
            print(f"  effect x{shrink:.1f}, N={n:4d}: power {hits / args.sims:.2f}")


if __name__ == "__main__":
    main()
