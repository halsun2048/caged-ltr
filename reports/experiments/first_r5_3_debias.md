# FIRST R5.3 identity/position debiasing

R5.3 applies a deterministic post-process to the 400 R5.2 first-token logits:
candidate identifiers are mapped back to candidate IDs, each perturbation is
z-scored, and the four aligned score vectors are averaged.

- 100/100 queries had all four perturbations; no model training was used.
- Mean Kendall tau of each raw perturbation ranking against the debiased
  ensemble: `0.7814`.
- Per-variant tau: baseline `0.7823`, reverse `0.7580`, random permutation
  `0.8066`, identifier remap `0.7786`.
- Leave-one-variant-out tau: baseline `0.7008`, reverse `0.6724`, random
  permutation `0.7393`, identifier remap `0.6936`.

The ensemble reduces the spread between perturbation branches (range `0.049`)
relative to the raw baseline comparisons in R5.2, but it is still an
agreement/stability diagnostic rather than a relevance metric. It must be
validated on held-out qrels before being used as a production fusion score.
