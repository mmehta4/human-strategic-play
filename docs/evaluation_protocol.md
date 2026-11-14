# Evaluation protocol (version 3, frozen 2026-11-02)

**Status: frozen.** This file was committed and its SHA-256 was timestamped with OpenTimestamps
(`docs/evaluation_protocol.md.ots`, anchored in the Bitcoin blockchain) **before any model was
evaluated on the H1 test data**. Anyone can check the timestamp with `ots verify`. Results computed
after that point follow this file exactly; any deviation is reported as a deviation.

Version 2 planned to test H1 and H2 on new data collected in our own arena. Publishing research from
arena players needs ethics approval from an institution, which this project, run by one independent
researcher, doesn't have yet. Version 3 therefore tests H1 on an existing, already-published human
dataset, which needs no new approval, and pauses H2.

## 1. Questions

1. **Transfer (H1, confirmatory).** A model developed only on games2p2k (2x2 games, US online
   participants, 2024) is applied, unchanged, to twelve *other* studies: different decades, labs and
   populations, and mostly 3x3 or larger games. Does the hybrid predict those people's choices better
   than the strongest alternatives?
2. **LLM gap (H3, exploratory, complete).** Do free-tier LLMs, asked zero-shot, predict human play
   worse than quantal level-k?
3. **Free vs. paid play (H2): paused** until the arena has ethics approval. Its design (section 8)
   is unchanged.

## 2. Data

| Role | Source | Status |
| --- | --- | --- |
| Training (development) | games2p2k: 2,416 decision problems from 1,208 2x2 games (Zhu, Peterson, Enke & Griffiths; CC-BY 4.0) | Used throughout development |
| H1 test | The 12-study dataset used by GameNet and ElementaryNet ("all12"): Stahl & Wilson 1994, 1995; Costa-Gomes, Crawford & Broseta 1998; Goeree & Holt 2001; Cooper & Van Huyck 2003; Haruvy, Stahl & Wilson 2001; Haruvy & Stahl 2007; Stahl & Haruvy 2008; Rogers, Palfrey & Camerer 2009; Costa-Gomes, Crawford & Iriberri 2009; Fudenberg & Liang 2019; Chui, Hartline & Wright 2023. Files from github.com/gregdeon/elementarynet at commit `69005d68b367b8cab4a51327ae00061d2c545d90`, read by `hsb/data/wlb.py` | Not used in development |
| H3 test | 200 held-out games2p2k problems | Done (exploratory) |

**H1 test set, as loaded** (`hsb.data.load("wlb")`): 366 games, merged into 336 after removing exact
duplicates (counts summed, as in the ElementaryNet code), giving **384 decision problems** (each side
of a game with at least one observed choice) and 26,553 choices. Sizes: 314 problems 3x3, 30 2x2,
28 2x3/3x2, 6 with 4 or 5 actions, and 4 very large games (61 and 121 actions).

**What was seen before freezing (disclosure).** The test set's choices were never scored by any
model. Building the loader showed only the counts above (games, problems, choices, sizes). The
per-game choice counts of 10 of the 12 studies are written as literals in the source repository's
`datasets.py`, which was read while writing the parser. No statistic of those counts was computed.
The Fudenberg & Liang and Chui et al. counts (219 of the 384 problems) were not viewed at all.

## 3. Payoffs

All payoffs, in training and in test, are standardised per game and per player (mean 0, sample s.d.
1), as in ElementaryNet's main experiments. Predictions therefore ignore stake size, which differs
across these studies in currency and year. This was checked in development only: on games2p2k,
three-fold held-out games, standardised inputs gave NLL 0.5150 for hybrid v1, 0.5182 for the
ElementaryNet-style model and 0.5379 for quantal level-k.

## 4. Metric

Primary: mean cross-entropy (NLL, nats per choice), every decision problem weighted equally.
Secondary: NLL weighted by number of choices (ElementaryNet's weighting), per-study NLL, and the same
analysis without the four very large games.

## 5. Models (frozen at this commit)

Trained once on all 2,416 games2p2k problems (standardised); nothing is tuned on the test data.
- Uniform; quantal level-k (levels 0–3); Poisson cognitive hierarchy.
- ElementaryNet-style QLk (`ElementaryQLk`): our reimplementation, the stronger of the two built in
  development (held-out family NLL 0.5185 vs. 0.5229).
- ElementaryNet-style QCH, closer reimplementation (`ElementaryQCH`).
- **Hybrid v1** (`HybridQLkV1`): the model under test.

Neural models use the development hyperparameters, with `scale=1` for standardised payoffs. Each is
trained with 5 seeds (0–4), and its test prediction is the average of the 5 predicted
distributions. Fixed-size models (MLP, hybrid v0) can't read 3x3 games and are excluded. Logit QRE is
excluded because its homotopy solver isn't reliable on the 121-action games.

## 6. H1 decision rule

`scripts/h1_confirmatory.py`, run once.
- Per-problem NLL differences, hybrid v1 minus baseline, for (a) the ElementaryNet-style model
  (`ElementaryQLk`) and (b) quantal level-k.
- One-sided paired bootstrap over problems, 10,000 resamples, seed 0; Holm correction over (a), (b);
  family-wise alpha 0.05.
- **H1 holds** if both are significant. **Partly** if only one is. **Fails** if neither is.
- Every model's predictions and every secondary analysis are published whatever the outcome.

**Power, stated in advance** (`scripts/power_analysis.py`, resampling development gains, 384
problems): about 1.00 against quantal level-k; **0.61 against the ElementaryNet-style model** (0.23
if the true gain is half the development gain). A non-significant (a) is therefore weak evidence
either way and will be reported as such.

## 7. H3 (exploratory, complete)

All three free-tier models (gpt-oss-120b, Gemini 3.5 Flash Lite, Nemotron 3 Ultra) were worse than
quantal level-k on 200 held-out games2p2k problems (`results/llm_baseline_*.json`). Version 2
planned a confirmatory re-run; it is dropped because the prompt and data are fixed and the free
tiers' daily limits make a repeat slow without changing what is known. H3 is reported as exploratory.

## 8. Paused: H2 (free vs. paid play in the arena)

Unchanged from version 2 and run only after ethics approval: 12 calibration games copied from
games2p2k, played in the arena's bonus rounds. After a single-temperature calibration fitted on half
the games, at least 70% of held-out calibration problems must be within total variation 0.10 of the
paid shares (median over 200 splits). The research pool (`arena/research_pool.json`, SHA-256
`5ec938bcc80c6a85ffcc9f6b1446607fe53d60f81a571589d86471f10b61b9db`) stays frozen for a later study.
