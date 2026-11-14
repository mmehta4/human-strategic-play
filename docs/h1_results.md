# H1 confirmatory transfer test: result and post-freeze notes

Protocol: [`evaluation_protocol.md`](evaluation_protocol.md) v3 (frozen 2026-11-02, SHA-256
`952322830cb330004b22a61e5cbc2eb556b7f7d839c053c91779c2701814bc00`, OpenTimestamps proof in
`evaluation_protocol.md.ots`). Run once with `scripts/h1_confirmatory.py`, exactly as frozen, on
2026-11-02/03. Raw output: `results/h1_confirmatory.json`; every prediction:
`results/predictions/h1_wlb.csv.gz`.

## Verdict: H1 holds partly

Trained only on games2p2k (2x2 games, US online participants) and applied unchanged to 384 decision
problems (26,553 choices) from 12 other published studies:

| Model | NLL | NLL weighted by choices | NLL without 4 very large games | Hybrid − model, 95% CI |
| --- | --- | --- | --- | --- |
| Uniform | 1.0883 | 1.0642 | 1.0528 | |
| **Quantal level-k** | **0.8450** | **0.8629** | **0.8116** | +0.011 [+0.002, +0.020] |
| Cognitive hierarchy | 0.8475 | 0.8640 | 0.8145 | +0.008 [−0.000, +0.018] |
| ElementaryNet-style QLk (reimplementation) | 0.8931 | 0.8978 | 0.8594 | −0.037 [−0.046, −0.029] |
| ElementaryNet-style QCH (closer reimplementation) | 0.8850 | 0.8946 | 0.8505 | −0.029 [−0.037, −0.022] |
| Hybrid v1 (any size) | 0.8559 | 0.8667 | 0.8229 | |

Primary tests (one-sided paired bootstrap, 10,000 resamples, seed 0, Holm):

- (a) Hybrid v1 vs. ElementaryNet-style QLk: hybrid better by 0.037 nats, p < 0.001. **Significant.**
- (b) Hybrid v1 vs. quantal level-k: hybrid *worse* by 0.011 nats, one-sided p = 0.995. **Not significant.**

Secondary analyses agree: same ordering weighted by choices, and without the 61- and 121-action games.

## What it means

Across studies, models rank by how much of their prediction is structure: quantal level-k (four
parameters) transferred best, the hybrid (network-set structural core + residual) second, and the
two neural ElementaryNet-style models last. In development, on held-out families *within*
games2p2k, the hybrid beat quantal level-k; across studies, populations and game sizes the order
reverses. The flexible parts learn regularities of the training population and game size that do
not carry over. Practical rule: use structural models for a genuinely new setting; add learned
flexibility only with training data from the target population.

Per study (secondary, not tested): the hybrid beat quantal level-k in 5 of 12 studies (cvh03, fl19,
rpc08, sh08, chw23), including Fudenberg & Liang, which holds 200 of the 384 problems, and was within
0.005 nats in four more. It lost most in sw94 (by 0.10) and cgcb98 (by 0.13), two studies built to
separate reasoning types, and these losses decide the overall comparison.

## Post-freeze notes (no deviations from the analysis)

1. **Citation erratum.** The protocol's data table lists "Costa-Gomes, Crawford & Iriberri 2009" for
   the `cgw08` study. The data are from Costa-Gomes & Weizsäcker (2008), *Stated beliefs and play in
   normal-form games*. The data loaded and analysed are unchanged; only the citation was wrong. The
   paper uses the correct reference.
2. The analysis ran once, with the frozen code and settings. Nothing was tuned or re-run after
   seeing the result.
3. The comparison with the authors' own ElementaryNet code (trained on these 12 studies) was not
   run. Our ElementaryNet rows are reimplementations trained on games2p2k only, as stated in the
   protocol.
