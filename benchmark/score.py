"""Score a benchmark submission.

    python benchmark/score.py my_submission.csv
    python benchmark/score.py my_submission.csv --save --name "Team: model" --url https://... --desc "one line"

Submission CSV columns: problem_id, split, p0, p1
  split  - the held-out split the prediction was made for, e.g. "fold-3" or
           "holdout-coordination" (must match benchmark/data/splits.json)
  p0, p1 - predicted probability of the decision maker's first and second action

A complete submission covers every problem once under the kfold protocol and once
under the family protocol. Scores are reported per protocol; NLL is primary.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hsb.eval.metrics import score  # noqa: E402

DATA = Path(__file__).resolve().parent / "data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("submission")
    ap.add_argument("--save", action="store_true", help="write benchmark/submissions/<name>.json for the leaderboard")
    ap.add_argument("--name")
    ap.add_argument("--url", default="")
    ap.add_argument("--desc", default="")
    args = ap.parse_args()
    sub = pd.read_csv(args.submission)
    hidden = os.environ.get("HSB_HIDDEN_LABELS")  # maintainers only: private labels for the hidden set
    labels = pd.read_csv(hidden or DATA / "labels.csv").set_index("problem_id")
    splits = json.loads((DATA / "splits.json").read_text())
    if not np.allclose(sub[["p0", "p1"]].sum(axis=1), 1, atol=1e-4) or (sub[["p0", "p1"]] < 0).any().any():
        sys.exit("Each row's p0 + p1 must equal 1, with no negative probabilities.")
    report = {}
    for protocol in ("kfold", "family"):
        expected = splits[protocol]
        part = sub[sub.split.isin(set(expected.values()))]
        if part.empty:
            continue
        bad = part[part.apply(lambda r: expected.get(r.problem_id) != r.split, axis=1)]
        if len(bad):
            sys.exit(f"{len(bad)} rows in {protocol} name the wrong split for their problem.")
        missing = set(expected) - set(part.problem_id)
        lab = labels.loc[part.problem_id]
        s = score(lab[["f0", "f1"]].to_numpy(), part[["p0", "p1"]].to_numpy(), lab["n"].to_numpy())
        s["coverage"] = 1 - len(missing) / len(expected)
        report[protocol] = s
    print(json.dumps(report, indent=2))
    if args.save:
        if not args.name:
            sys.exit("--save needs --name")
        slug = re.sub(r"[^a-z0-9]+", "-", args.name.lower()).strip("-")
        out = DATA.parent / "submissions" / f"{slug}.json"
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps({"name": args.name, "url": args.url, "description": args.desc,
                                   "date": date.today().isoformat(), "benchmark_version": splits["version"],
                                   "scores": report}, indent=2))
        print(f"saved {out}")


if __name__ == "__main__":
    main()
