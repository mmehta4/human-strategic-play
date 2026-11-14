# Stage 1 baseline results: games2p2k

Dataset: games2p2k (Zhu, Peterson, Enke & Griffiths), 2,416 decision problems from
1,208 two-player 2x2 games, ~93k human choices. Lower NLL is better. Completeness is
the share of the gap between uniform guessing and the sampling-noise floor that a
model closes. The last column is the hybrid's NLL minus this model's NLL, paired over
the same test problems, with a bootstrap 95% CI: negative means the hybrid is better.

## Random problem split (Zhu et al.'s protocol, for comparison)

| Model | NLL (nats/choice) | R² | MSE | Accuracy | Completeness | Hybrid − model, 95% CI |
| --- | --- | --- | --- | --- | --- | --- |
| Uniform | 0.6931 ± 0.0000 | -0.066 | 0.0851 | 0.595 | -0.000 | -0.1749 [-0.1842, -0.1659] |
| Nash (+ error) | 0.6280 ± 0.0048 | 0.327 | 0.0537 | 0.688 | 0.361 | -0.1098 [-0.1176, -0.1012] |
| Logit QRE | 0.5734 ± 0.0044 | 0.649 | 0.0281 | 0.881 | 0.664 | -0.0552 [-0.0603, -0.0504] |
| Quantal level-k | 0.5505 ± 0.0052 | 0.761 | 0.0191 | 0.879 | 0.791 | -0.0323 [-0.0364, -0.0284] |
| Cognitive hierarchy | 0.5518 ± 0.0052 | 0.755 | 0.0196 | 0.888 | 0.783 | -0.0336 [-0.0378, -0.0298] |
| ElementaryNet-style QLk (reimplementation) | 0.5228 ± 0.0047 | 0.900 | 0.0080 | 0.929 | 0.944 | -0.0046 [-0.0060, -0.0031] |
| MLP (black box) | 0.5181 ± 0.0047 | 0.922 | 0.0062 | 0.940 | 0.970 | +0.0001 [-0.0007, +0.0009] |
| Hybrid v0 (structural only) | 0.5221 ± 0.0048 | 0.898 | 0.0081 | 0.911 | 0.948 | -0.0038 [-0.0055, -0.0023] |
| Hybrid v0 (structural + residual) | 0.5182 ± 0.0047 | 0.922 | 0.0062 | 0.939 | 0.970 | reference |

## Held-out games (10-fold, grouped by game)

| Model | NLL (nats/choice) | R² | MSE | Accuracy | Completeness | Hybrid − model, 95% CI |
| --- | --- | --- | --- | --- | --- | --- |
| Uniform | 0.6931 ± 0.0000 | -0.070 | 0.0875 | 0.598 | 0.000 | -0.1808 [-0.1874, -0.1744] |
| Nash (+ error) | 0.6311 ± 0.0035 | 0.296 | 0.0575 | 0.681 | 0.333 | -0.1188 [-0.1251, -0.1129] |
| Logit QRE | 0.5656 ± 0.0032 | 0.668 | 0.0271 | 0.884 | 0.684 | -0.0533 [-0.0567, -0.0499] |
| Quantal level-k | 0.5423 ± 0.0036 | 0.782 | 0.0178 | 0.888 | 0.809 | -0.0300 [-0.0326, -0.0274] |
| Cognitive hierarchy | 0.5432 ± 0.0036 | 0.778 | 0.0182 | 0.894 | 0.804 | -0.0309 [-0.0336, -0.0283] |
| ElementaryNet-style QLk (reimplementation) | 0.5156 ± 0.0034 | 0.910 | 0.0073 | 0.934 | 0.952 | -0.0033 [-0.0042, -0.0024] |
| MLP (black box) | 0.5124 ± 0.0034 | 0.924 | 0.0062 | 0.940 | 0.969 | -0.0001 [-0.0006, +0.0004] |
| Hybrid v0 (structural only) | 0.5151 ± 0.0034 | 0.909 | 0.0074 | 0.921 | 0.954 | -0.0028 [-0.0038, -0.0019] |
| Hybrid v0 (structural + residual) | 0.5123 ± 0.0034 | 0.924 | 0.0062 | 0.941 | 0.970 | reference |
| Hybrid v1 (any size, structural + residual) | 0.5132 ± 0.0034 | 0.921 | 0.0065 | 0.942 | 0.965 | -0.0009 [-0.0014, -0.0003] |

## Held-out strategic family (leave one family out)

| Model | NLL (nats/choice) | R² | MSE | Accuracy | Completeness | Hybrid − model, 95% CI |
| --- | --- | --- | --- | --- | --- | --- |
| Uniform | 0.6931 ± 0.0000 | -0.070 | 0.0875 | 0.598 | 0.000 | -0.1771 [-0.1839, -0.1699] |
| Nash (+ error) | 0.6476 ± 0.0050 | 0.224 | 0.0634 | 0.681 | 0.244 | -0.1315 [-0.1396, -0.1236] |
| Logit QRE | 0.5707 ± 0.0033 | 0.651 | 0.0285 | 0.885 | 0.656 | -0.0547 [-0.0586, -0.0509] |
| Quantal level-k | 0.5428 ± 0.0036 | 0.780 | 0.0180 | 0.887 | 0.806 | -0.0268 [-0.0296, -0.0242] |
| Cognitive hierarchy | 0.5439 ± 0.0037 | 0.775 | 0.0184 | 0.895 | 0.800 | -0.0279 [-0.0307, -0.0252] |
| ElementaryNet-style QLk (reimplementation) | 0.5185 ± 0.0035 | 0.895 | 0.0085 | 0.926 | 0.936 | -0.0025 [-0.0036, -0.0014] |
| MLP (black box) | 0.5203 ± 0.0035 | 0.886 | 0.0093 | 0.921 | 0.927 | -0.0043 [-0.0053, -0.0032] |
| Hybrid v0 (structural only) | 0.5283 ± 0.0036 | 0.839 | 0.0132 | 0.888 | 0.884 | -0.0123 [-0.0147, -0.0101] |
| Hybrid v0 (structural + residual) | 0.5160 ± 0.0035 | 0.907 | 0.0076 | 0.931 | 0.950 | reference |
| Hybrid v1 (any size, structural + residual) | 0.5161 ± 0.0034 | 0.906 | 0.0077 | 0.928 | 0.949 | -0.0001 [-0.0008, +0.0007] |
| ElementaryNet-style QCH (closer reimplementation) | 0.5229 ± 0.0034 | 0.876 | 0.0102 | 0.920 | 0.913 | -0.0069 [-0.0082, -0.0055] |

### Per held-out family (NLL)

| Held-out family | Uniform | Nash (+ error) | Logit QRE | Quantal level-k | Cognitive hierarchy | ElementaryNet-style QLk (reimplementation) | MLP (black box) | Hybrid v0 (structural only) | Hybrid v0 (structural + residual) | Hybrid v1 (any size, structural + residual) | ElementaryNet-style QCH (closer reimplementation) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| both_dominant (n=240) | 0.6931 | 0.4925 | 0.4711 | 0.4258 | 0.4266 | 0.4104 | 0.4069 | 0.4056 | 0.4065 | 0.4073 | 0.4119 |
| coordination (n=746) | 0.6931 | 0.7104 | 0.6291 | 0.6042 | 0.6070 | 0.5783 | 0.5773 | 0.5849 | 0.5758 | 0.5761 | 0.5817 |
| other_dominant (n=715) | 0.6931 | 0.8100 | 0.6210 | 0.6137 | 0.6126 | 0.5878 | 0.5961 | 0.6119 | 0.5862 | 0.5850 | 0.5906 |
| self_dominant (n=715) | 0.6931 | 0.4715 | 0.4928 | 0.4471 | 0.4487 | 0.4232 | 0.4231 | 0.4268 | 0.4203 | 0.4212 | 0.4310 |

## Structural parameters fitted on all data

- **Nash (+ error)**: {"eps": 0.5660335164021408}
- **Logit QRE**: {"lambda": 0.0941160676206347}
- **Quantal level-k**: {"level_shares": [0.282, 0.468, 0.25, 0.0], "lambda": 0.3996974363830239}
- **Cognitive hierarchy**: {"tau": 1.2315365854847833, "lambda": 0.3781117433668749}
