# Benchmark

**Task:** given a two-player game seen from one player's side (both payoff matrices and a
plain-language description), predict the share of people who choose each action.

**Metric:** mean cross-entropy per choice (nats), every decision problem weighted equally.
Secondary: completeness, R², MSE, accuracy.

## Versions

| Version | Data | Status |
| --- | --- | --- |
| v0 | games2p2k: 2,416 problems, 1,208 2x2 games, ~93k choices, public labels | Frozen (`data/MANIFEST.json`) |
| v1 | v0 + arena games (ultimatum, trust, pricing, 11–20, lost luggage, matrix) + a **hidden test set** from the pre-registered Prolific study | Built once that data exists |

## Protocols

- `kfold`: 10 folds over games; both sides of a game are always in the same fold.
- `family`: leave one strategic family out; train only on games with no side in that family.

## Hidden test set (v1)

The hidden labels never go into this repository. They live in a private file that only the
maintainers hold, and `score.py` reads it from the path in the `HSB_HIDDEN_LABELS` environment
variable. Submitters send predictions; maintainers score them and publish only the score.
This stops anyone, including us, from tuning to the test set.

## Files

- `data/problems.jsonl`: one problem per line: ids, family, payoffs, description
- `data/splits.json`: the fixed fold and family assignment
- `data/labels.csv`: observed shares and choice counts (public in v0)
- `data/MANIFEST.json`: checksums; if a file doesn't match, it isn't v0
- `score.py`: scorer (`--save` writes a leaderboard entry)
- `submissions/`: leaderboard entries
