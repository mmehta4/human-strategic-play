"""Score a frontier LLM, zero-shot, on a sample of held-out games and compare it with
the fitted baselines on exactly the same problems.

    python scripts/run_llm_baseline.py --provider groq --model openai/gpt-oss-120b --n 200   # free tier
    python scripts/run_llm_baseline.py --provider gemini --model gemini-3.5-flash-lite      # free tier
    python scripts/run_llm_baseline.py --provider nvidia --model moonshotai/kimi-k3         # free tier
    python scripts/run_llm_baseline.py --provider anthropic --model claude-opus-5-5         # paid
    python scripts/run_llm_baseline.py --dry-run                                            # show the prompt

Uses the first held-out fold of the 10-fold game split (same seed as
run_baselines.py), so the comparison models never saw these games.
Free providers need GROQ_API_KEY / GEMINI_API_KEY / NVIDIA_API_KEY in the environment.
If a free daily limit is hit, re-run tomorrow: cached answers are reused, so it resumes.
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
from hsb.eval import paired_difference, score, splits  # noqa: E402
from hsb.text import describe_matrix_game  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--provider", default="groq", choices=["groq", "gemini", "nvidia", "anthropic"])
    ap.add_argument("--model", default="openai/gpt-oss-120b")
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--workers", type=int, default=2, help="parallel requests (keep low on free tiers)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    records = load("games2p2k")
    _, _, test = next(splits.grouped_kfold(records, k=10))
    rng = np.random.default_rng(0)
    sample = [test[i] for i in sorted(rng.choice(len(test), size=min(args.n, len(test)), replace=False))]

    if args.dry_run:
        from hsb.llm import QUESTION, SYSTEM
        print("SYSTEM:\n" + SYSTEM + "\n\nUSER:\n" + describe_matrix_game(sample[0].game) + "\n\n" + QUESTION)
        print(f"\nWould send {len(sample)} requests to {args.model}.")
        return

    if args.provider == "anthropic":
        from hsb.llm import LLMZeroShot
        llm = LLMZeroShot(model=args.model, effort=args.effort)
    else:
        from hsb.llm_free import FreeLLMZeroShot
        llm = FreeLLMZeroShot(args.provider, args.model, workers=args.workers)
    preds = llm.predict(sample)
    if getattr(llm, "stopped", None):
        print(f"Stopped early ({llm.stopped[:120]}). Re-run tomorrow to finish; answers so far are cached.")
    keep = [i for i, p in enumerate(preds) if p is not None]
    sample = [sample[i] for i in keep]
    p_llm = np.array([preds[i] for i in keep])
    freq = np.array([r.freq for r in sample])
    n = np.array([r.n for r in sample])
    report = {"llm": {llm.name: score(freq, p_llm, n), "answered": len(keep),
                      "unparseable": getattr(llm, "failures", 0), "refusals": getattr(llm, "refusals", 0)}}

    preds = ROOT / "results" / "predictions" / "kfold.csv.gz"
    if preds.exists():
        df = pd.read_csv(preds)
        df = df[df.split == "fold-0"].set_index("problem_id")
        ids = [r.problem_id for r in sample]
        for model, g in df.groupby("model"):
            p = g.loc[ids, ["p0", "p1"]].to_numpy()
            report[model] = {**score(freq, p, n), "llm_minus_this": paired_difference(freq, p_llm, p)}
    slug = args.model.replace("/", "_")
    out = ROOT / "results" / f"llm_baseline_{slug}.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
