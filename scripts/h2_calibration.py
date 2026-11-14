"""H2 (pre-registered): does free arena play match paid play?

Compares arena choice shares on the 12 calibration games (24 decision problems) with the
shares games2p2k's paid Prolific participants produced on the identical games.

    python scripts/h2_calibration.py path/to/arena_export.csv

Rule (docs/evaluation_protocol.md): fit one calibration map on half the problems (split by
game), then on the other half the total variation distance between calibrated arena shares
and paid shares must be at most 0.10 on at least 70% of problems. Repeated over many random
halves; the median pass share is reported. Caveat: this also compares populations
(US Prolific workers vs arena players), so a gap is not purely the effect of money.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hsb.data import arena, games2p2k  # noqa: E402

TV_MAX, PASS_SHARE = 0.10, 0.70


def calibrate(p, beta):
    z = beta * np.log(np.clip(p, 1e-6, 1))
    z = np.exp(z - z.max(axis=1, keepdims=True))
    return z / z.sum(axis=1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("export")
    ap.add_argument("--min-choices", type=int, default=30)
    ap.add_argument("--splits", type=int, default=200)
    args = ap.parse_args()

    paid = {r.problem_id: r for r in games2p2k.load()}
    by_source = {}
    for r in paid.values():
        by_source.setdefault((r.game_id, r.game.meta["original_role"]), r)
    pairs = []
    for a in arena.load(args.export, min_choices=args.min_choices, kinds=("calibration",)):
        role = "rowplayer" if a.game.meta["role"] == "row" else "colplayer"
        p = by_source[(a.game.meta["source_game"], role)]
        assert np.array_equal(a.game.own, p.game.own), f"payoff mismatch for {a.problem_id}"
        pairs.append((a, p))
    if len(pairs) < 8:
        sys.exit(f"Only {len(pairs)} calibration problems have {args.min_choices}+ arena choices; wait for more data.")

    A = np.array([a.freq for a, _ in pairs])
    P = np.array([p.freq for _, p in pairs])
    games = np.array([a.game_id for a, _ in pairs])
    raw_tv = 0.5 * np.abs(A - P).sum(1)
    rng = np.random.default_rng(0)
    uniq = np.unique(games)
    shares, betas = [], []
    for _ in range(args.splits):
        fit_games = set(rng.choice(uniq, size=len(uniq) // 2, replace=False))
        fit = np.array([g in fit_games for g in games])
        loss = lambda b: float(-(P[fit] * np.log(calibrate(A[fit], b))).sum(1).mean())
        beta = minimize_scalar(loss, bounds=(0.05, 5), method="bounded").x
        tv = 0.5 * np.abs(calibrate(A[~fit], beta) - P[~fit]).sum(1)
        shares.append(float((tv <= TV_MAX).mean()))
        betas.append(float(beta))
    out = {"n_problems": len(pairs), "mean_arena_choices": float(np.mean([a.n for a, _ in pairs])),
           "raw_tv_mean": float(raw_tv.mean()), "raw_share_within": float((raw_tv <= TV_MAX).mean()),
           "calibrated_share_within_median": float(np.median(shares)), "beta_median": float(np.median(betas)),
           "pass": bool(np.median(shares) >= PASS_SHARE)}
    (ROOT / "results" / "h2_calibration.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print("\nH2", "PASSES" if out["pass"] else "FAILS",
          f"(median {out['calibrated_share_within_median']:.0%} of held-out problems within TV {TV_MAX}; need {PASS_SHARE:.0%})")


if __name__ == "__main__":
    main()
