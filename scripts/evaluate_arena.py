"""Benchmark v1 on arena data: hold out one game family at a time (ultimatum, trust, pricing,
undercutting, matrix), train on the rest, and score every model on the held-out family.

    python scripts/evaluate_arena.py path/to/arena_export.csv
    python scripts/evaluate_arena.py path/to/arena_export.csv --with-games2p2k   # also train on the 2x2 data

Writes results/arena_RESULTS.md and results/arena_summary.json. Needs at least
`--min-choices` choices (default 20) per game and role.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hsb.data import arena, games2p2k  # noqa: E402
from hsb.eval.metrics import score_ragged  # noqa: E402
from hsb.models.neural import HybridQLkV1  # noqa: E402
from hsb.models.structural import CognitiveHierarchy, LogitQRE, NashMix, QuantalLevelK, Uniform  # noqa: E402


def models(skip_nash: bool):
    ms = [Uniform(), LogitQRE(), QuantalLevelK(), CognitiveHierarchy(), HybridQLkV1()]
    return ms if skip_nash else [NashMix()] + ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("export")
    ap.add_argument("--min-choices", type=int, default=20)
    ap.add_argument("--with-games2p2k", action="store_true")
    ap.add_argument("--skip-nash", action="store_true", help="Nash on 11x11 games is slow")
    args = ap.parse_args()

    recs = arena.load(args.export, min_choices=args.min_choices, kinds=("fixed",))
    extra = games2p2k.load() if args.with_games2p2k else []
    families = sorted({r.family for r in recs})
    print(f"{len(recs)} arena problems in families {families}")
    results = {}
    for fam in families:
        test = [r for r in recs if r.family == fam]
        test_games = {r.game_id for r in test}
        train = [r for r in recs if r.game_id not in test_games] + extra
        if not train:
            continue
        for m in models(args.skip_nash):
            m.fit(train)
            p = m.predict(test)
            results.setdefault(m.name, {"freq": [], "pred": [], "n": []})
            results[m.name]["freq"] += [r.freq for r in test]
            results[m.name]["pred"] += [x for x in p]
            results[m.name]["n"] += [r.n for r in test]
            print(f"  holdout-{fam:14s} {m.name}")
    summary = {name: score_ragged(v["freq"], v["pred"], v["n"]) for name, v in results.items()}
    out = ROOT / "results"
    (out / "arena_summary.json").write_text(json.dumps(summary, indent=2))
    lines = ["# Benchmark v1 (arena data): held-out game family", "",
             f"Families: {', '.join(families)}. Extra training data: {'games2p2k' if extra else 'none'}.", "",
             "| Model | NLL | Completeness | Accuracy | Problems |", "| --- | --- | --- | --- | --- |"]
    for name, s in sorted(summary.items(), key=lambda kv: kv[1]["nll"]):
        lines.append(f"| {name} | {s['nll']:.4f} ± {s['nll_se']:.4f} | {s['completeness']:.3f} | "
                     f"{s['accuracy']:.3f} | {s['n_problems']} |")
    (out / "arena_RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
