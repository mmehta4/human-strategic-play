# Stage 1 status

What's built, and what needs a person. Items map one-to-one to Stage 1 in the build plan.

## Built and tested

- [x] Unified dataset format, with games2p2k loaded and family-labelled (0 mismatches against the release's own labels)
- [x] Baseline ladder: uniform, Nash + error, logit QRE, quantal level-k, cognitive hierarchy, ElementaryNet-style QLk (reimplementation), black-box MLP
- [x] Hybrid v0 (2x2) and hybrid v1 (any size, action-symmetric)
- [x] Evaluation: three protocols, cross-entropy, completeness, paired bootstrap CIs, per-problem predictions saved
- [x] Exploratory results on games2p2k: `results/RESULTS.md`
- [x] What the structural core learns per family: `results/hybrid_explained.json`
- [x] Procedural generator for 8 mechanism families, with formal spec + text
- [x] Zero-shot LLM baselines on three free tiers: all worse than quantal level-k (H3, exploratory)
- [x] Benchmark v0 frozen: data, fixed splits, checksums, scorer
- [x] Arena v0: consent, 18+ gate, 10 games, profile, percentiles, delete-my-data, export; 2 end-to-end tests
- [x] Protocol v3 frozen and timestamped (OpenTimestamps), power analysis
- [x] 12-study test set loader (data only, pinned commit) and closer ElementaryNet reimplementation
- [x] H1 confirmatory transfer test run once: holds partly (`docs/h1_results.md`)
- [x] Paper updated with the confirmatory result (`paper/main.pdf`)
- [x] 25 automated tests passing

**Stage 1 research is complete.** H2 (free vs. paid play) is paused until the arena has ethics approval.

## Needs the team

Everything the team has to do, for Stage 1 and Stage 2, including what to recheck in this work,
is in `stage1/TASK.md` (and `stage2/TASK.md` for the next stage). This file only records what was built.
