"""Freeze benchmark v0: problems, plain-language descriptions, fixed splits, labels.

    python scripts/make_benchmark.py

Writes benchmark/data/{problems.jsonl, splits.json, labels.csv, MANIFEST.json}.
Splits are fixed here once, so every submission is scored on identical folds.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hsb.data import games2p2k, load  # noqa: E402
from hsb.eval import splits  # noqa: E402
from hsb.text import describe_matrix_game  # noqa: E402

VERSION = "v0"


def main():
    out = ROOT / "benchmark" / "data"
    out.mkdir(parents=True, exist_ok=True)
    records = load("games2p2k")
    with (out / "problems.jsonl").open("w") as f:
        for r in records:
            f.write(json.dumps({
                "problem_id": r.problem_id, "game_id": r.game_id, "family": r.family,
                "own_payoffs": r.game.own.tolist(), "other_payoffs": r.game.other.tolist(),
                "description": describe_matrix_game(r.game),
            }) + "\n")
    folds = {}
    for name, _, test in splits.grouped_kfold(records, k=10):
        for r in test:
            folds[r.problem_id] = name
    split_file = {
        "version": VERSION,
        "kfold": folds,
        "family": {r.problem_id: f"holdout-{r.family}" for r in records},
        "notes": "kfold: train on the other 9 folds, test on this one. family: train on problems whose "
                 "game_id never appears in the held-out family, test on that family.",
    }
    (out / "splits.json").write_text(json.dumps(split_file))
    with (out / "labels.csv").open("w") as f:
        f.write("problem_id,f0,f1,n\n")
        for r in records:
            f.write(f"{r.problem_id},{r.freq[0]:.6f},{r.freq[1]:.6f},{r.n:.2f}\n")
    manifest = {
        "version": VERSION, "n_problems": len(records), "n_games": len({r.game_id for r in records}),
        "source": "games2p2k (Zhu, Peterson, Enke & Griffiths), https://osf.io/xrvaw/, CC-BY 4.0",
        "source_sha256": games2p2k.checksum(),
        "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in sorted(out.iterdir()) if p.name != "MANIFEST.json"},
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
