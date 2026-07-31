# FIRST R5.7 independent NFCorpus validation

The NFCorpus `dev` split was used as an independent validation set. It was
prepared before opening dev qrels, with 324 queries and a frozen deterministic
BM25 top-20 candidate pool. FIRST inference used the locked checkpoint,
four perturbations, and eight RTX 4090 GPUs; qrels were read only after all
predictions were frozen.

## Overall result

| Method | Linear graded NDCG@10 |
|---|---:|
| BM25 | 0.268063 |
| FIRST baseline | 0.305684 |
| FIRST debiased | 0.300123 |

Paired bootstrap 95% CIs:

- FIRST baseline − BM25: `[0.02698, 0.04947]`
- FIRST debiased − BM25: `[0.01927, 0.04561]`
- Debiased − baseline: `[-0.01113, -0.00020]`

The baseline gain is positive on the independent split. Debiasing is again
slightly harmful, so no fallback threshold is selected from this result.

## Query-length replication

| Query length | Queries | Mean FIRST−BM25 | Positive fraction |
|---|---:|---:|---:|
| 1–2 words | 169 | +0.0127 | 25.4% |
| 3–4 words | 53 | +0.0326 | 41.5% |
| 5+ words | 102 | +0.0813 | 51.0% |

The same monotonic pattern appears on dev: the strongest gains come from
longer queries, while short queries have a small and less reliable gain. This
validates the diagnostic hypothesis, but not yet a production fallback policy.
The next experiment should pre-register a simple threshold on this dev split,
then evaluate it once on the untouched test split.
