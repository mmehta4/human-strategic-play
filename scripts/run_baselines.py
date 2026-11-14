"""Run the full baseline ladder under every evaluation protocol and write results.

    python scripts/run_baselines.py                 # everything (~15 min on a laptop)
    python scripts/run_baselines.py --protocols kfold --quick

Writes results/summary.json, results/RESULTS.md and per-problem predictions in
results/predictions/ so any number in the report can be recomputed.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hsb.data import load  # noqa: E402
from hsb.eval import paired_difference, score, splits  # noqa: E402
from hsb.models import baseline_ladder  # noqa: E402

PROTOCOLS = {
    "random": ("Random problem split (Zhu et al.'s protocol, for comparison)",
               lambda r, q: splits.random_problems(r, n_repeats=2 if q else 5)),
    "kfold": ("Held-out games (10-fold, grouped by game)",
              lambda r, q: splits.grouped_kfold(r, k=3 if q else 10)),
    "family": ("Held-out strategic family (leave one family out)",
               lambda r, q: splits.leave_family_out(r)),
}
REFERENCE = "Hybrid v0 (structural + residual)"


def ladder(only: list[str] | None):
    from hsb.models.neural import ElementaryQCH, HybridQLkV1
    models = baseline_ladder() + [HybridQLkV1(), ElementaryQCH()]
    if not only:
        return baseline_ladder()
    return [m for m in models if any(o.lower() in m.name.lower() for o in only)]


def run(protocol: str, records, quick: bool, only=None):
    rows = []
    for split_name, train, test in PROTOCOLS[protocol][1](records, quick):
        for model in ladder(only):
            t = time.time()
            model.fit(train)
            pred = model.predict(test)
            for r, p in zip(test, pred):
                rows.append({"protocol": protocol, "split": split_name, "model": model.name,
                             "problem_id": r.problem_id, "family": r.family, "n": r.n,
                             "f0": r.freq[0], "f1": r.freq[1], "p0": p[0], "p1": p[1]})
            print(f"  [{protocol}] {split_name:28s} {model.name:36s} {time.time() - t:6.1f}s", flush=True)
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> dict:
    out = {}
    for model, g in df.groupby("model", sort=False):
        out[model] = score(g[["f0", "f1"]].to_numpy(), g[["p0", "p1"]].to_numpy(), g["n"].to_numpy())
    ref = df[df.model == REFERENCE].set_index(["split", "problem_id"])
    for model, g in df.groupby("model", sort=False):
        if model == REFERENCE:
            continue
        g = g.set_index(["split", "problem_id"]).loc[ref.index]
        out[model]["reference_minus_this"] = paired_difference(
            ref[["f0", "f1"]].to_numpy(), ref[["p0", "p1"]].to_numpy(), g[["p0", "p1"]].to_numpy())
    return out


def by_family(df: pd.DataFrame) -> dict:
    out = defaultdict(dict)
    for (fam, model), g in df.groupby(["family", "model"], sort=False):
        out[fam][model] = score(g[["f0", "f1"]].to_numpy(), g[["p0", "p1"]].to_numpy(), g["n"].to_numpy())
    return out


def to_markdown(results: dict, full_fit: dict) -> str:
    lines = ["# Stage 1 baseline results: games2p2k", "",
             "Dataset: games2p2k (Zhu, Peterson, Enke & Griffiths), 2,416 decision problems from",
             "1,208 two-player 2x2 games, ~93k human choices. Lower NLL is better. Completeness is",
             "the share of the gap between uniform guessing and the sampling-noise floor that a",
             "model closes. The last column is the hybrid's NLL minus this model's NLL, paired over",
             "the same test problems, with a bootstrap 95% CI: negative means the hybrid is better.", ""]
    for protocol, block in results.items():
        lines += [f"## {PROTOCOLS[protocol][0]}", "",
                  "| Model | NLL (nats/choice) | R² | MSE | Accuracy | Completeness | Hybrid − model, 95% CI |",
                  "| --- | --- | --- | --- | --- | --- | --- |"]
        for model, s in block["overall"].items():
            d = s.get("reference_minus_this")
            diff = "reference" if d is None else f"{d['mean']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]"
            lines.append(f"| {model} | {s['nll']:.4f} ± {s['nll_se']:.4f} | {s['r2']:.3f} | {s['mse']:.4f} | "
                         f"{s['accuracy']:.3f} | {s['completeness']:.3f} | {diff} |")
        lines.append("")
        if protocol == "family":
            lines += ["### Per held-out family (NLL)", ""]
            models = list(block["overall"].keys())
            lines += ["| Held-out family | " + " | ".join(models) + " |",
                      "| --- |" + " --- |" * len(models)]
            for fam, ms in block["by_family"].items():
                lines.append(f"| {fam} (n={ms[models[0]]['n_problems']}) | " +
                             " | ".join(f"{ms[m]['nll']:.4f}" for m in models) + " |")
            lines.append("")
    lines += ["## Structural parameters fitted on all data", ""]
    for name, p in full_fit.items():
        lines.append(f"- **{name}**: {json.dumps(p)}")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocols", nargs="+", default=list(PROTOCOLS))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", nargs="+", help="run only models whose name contains one of these "
                    "strings and merge them into the existing predictions, e.g. --only 'Hybrid v1'")
    args = ap.parse_args()

    records = load("games2p2k")
    out_dir = ROOT / "results"
    (out_dir / "predictions").mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    results = json.loads(summary_path.read_text())["results"] if args.only and summary_path.exists() else {}
    for protocol in args.protocols:
        print(f"== {PROTOCOLS[protocol][0]}")
        df = run(protocol, records, args.quick, args.only)
        path = out_dir / "predictions" / f"{protocol}.csv.gz"
        if args.only and path.exists():
            old = pd.read_csv(path)
            df = pd.concat([old[~old.model.isin(df.model.unique())], df], ignore_index=True)
        df.to_csv(path, index=False)
        results[protocol] = {"overall": summarise(df)}
        if protocol == "family":
            results[protocol]["by_family"] = by_family(df)

    full_fit = {}
    for m in baseline_ladder(include_neural=False)[1:]:
        full_fit[m.name] = m.fit(records).params()

    (out_dir / "summary.json").write_text(json.dumps({"results": results, "full_fit": full_fit}, indent=2))
    (out_dir / "RESULTS.md").write_text(to_markdown(results, full_fit))
    print((out_dir / "RESULTS.md").read_text())


if __name__ == "__main__":
    main()
