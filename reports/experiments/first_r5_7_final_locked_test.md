# FIRST R5.7 final locked NFCorpus test

The dev-only fallback preregistration selected **Always FIRST**. The untouched
NFCorpus test split was then evaluated once with the original frozen candidate
pool and frozen predictions. No threshold search, model update, or test-time
fallback was applied.

| Method | Linear graded NDCG@10 |
|---|---:|
| BM25 | 0.306935 |
| FIRST baseline | 0.343543 |
| FIRST debiased | 0.334447 |

Paired bootstrap 95% CIs:

- FIRST baseline − BM25: `[0.02493, 0.04899]`
- FIRST debiased − BM25: `[0.01366, 0.04223]`
- Debiased − baseline: `[-0.01523, -0.00333]`

The final locked result confirms a significant FIRST gain of `+0.036608` over
BM25. The preregistered fallback is not used, and debiasing remains inferior.
This is the final R5.7 NFCorpus test result.
