"""What does the hybrid's structural core learn? Fit the structural-only hybrid on all
data and report the average level shares and precision it assigns to each strategic
family. This is the interpretable output a regulator or designer would read:
"in games like this, about x% of people reason two steps ahead".

    python scripts/explain_hybrid.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hsb.data import load  # noqa: E402
from hsb.models.neural import HybridQLk  # noqa: E402


def main():
    records = load("games2p2k")
    model = HybridQLk(residual=False).fit(records)
    ex = model.explain(records)
    fams = np.array([r.family for r in records])
    out = {}
    for fam in sorted(set(fams)):
        sel = fams == fam
        w = ex["level_shares"][sel].mean(0)
        lam = ex["precision"][sel].mean(0) / model.scale  # back to "per payoff point" units
        out[fam] = {"n": int(sel.sum()),
                    "level_shares": {f"L{k}": round(float(x), 3) for k, x in enumerate(w)},
                    "mean_level": round(float((w * np.arange(len(w))).sum()), 2),
                    "precision_own": round(float(lam[0]), 3), "precision_belief": round(float(lam[1]), 3)}
    path = ROOT / "results" / "hybrid_explained.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
