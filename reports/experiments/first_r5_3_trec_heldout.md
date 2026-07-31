# FIRST R5.3 held-out TREC-DL evaluation

The identity-aligned/z-score debiasing protocol was evaluated on the frozen
TREC-DL 2019/2020 top-10 candidate snapshot (97 judged queries, 388 prompts).
Qrels were accessed only after all GPU predictions were frozen.

| Method | Linear graded NDCG@10 |
|---|---:|
| BM25 candidate order | 0.491249 |
| FIRST baseline | 0.541815 |
| FIRST debiased ensemble | 0.541286 |

By year, FIRST baseline/debiased were `0.556929/0.554299` (2019) and
`0.529780/0.530923` (2020). Thus FIRST improves over BM25 by about `+0.0506`
overall, while the current four-perturbation debiasing is essentially neutral
(`-0.00053` overall). This is evidence of held-out ranking utility on this
candidate snapshot, not yet a production claim: evaluation uses top-10
candidate lists and a single frozen checkpoint, so larger-candidate and
multi-seed confirmation remain necessary.

Paired bootstrap (10,000 resamples, seed 42) 95% CIs:

- FIRST baseline − BM25: `[0.03697, 0.06415]`
- FIRST debiased − BM25: `[0.03458, 0.06535]`
- FIRST debiased − baseline: `[-0.00598, 0.00398]`

The improvement over BM25 is therefore robust on this sample, while the
debiased post-process is statistically indistinguishable from the raw FIRST
baseline.
