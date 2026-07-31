# FIRST R5.4 held-out TREC-DL top-20 evaluation

To address the top-10 limitation, the same 97 held-out TREC-DL queries were
run with the first 20 BM25 candidates (388 GPU prompts; four perturbations).
The model remains within the registered 26-identifier limit.

| Method | Linear graded NDCG@10 |
|---|---:|
| BM25 top-100 order | 0.491249 |
| FIRST baseline (top-20) | 0.630504 |
| FIRST debiased ensemble (top-20) | 0.633369 |

Paired bootstrap 95% CIs:

- FIRST baseline − BM25: `[0.11390, 0.16586]`
- FIRST debiased − BM25: `[0.11321, 0.17161]`
- FIRST debiased − baseline: `[-0.00498, 0.01049]`

The top-20 run shows a large, robust gain over BM25. Debiasing trends slightly
positive (`+0.00287`) but remains statistically indistinguishable from raw
FIRST.

The matched PRP control is now complete: PRP filtered to the identical top-20
candidate pool scores `0.602779`, versus FIRST `0.630504`. FIRST−PRP is
`+0.027725` with paired bootstrap 95% CI `[0.01154, 0.04473]`, excluding zero.
This supports a statistically significant FIRST advantage over the frozen PRP
control on this top-20 candidate pool.

Year-wise audit shows the overall effect is not uniform: in 2019 FIRST−PRP is
`-0.00345` (95% CI `[-0.02138, 0.01368]`), while in 2020 it is `+0.05255`
(95% CI `[0.02915, 0.07821]`). The aggregate significance is therefore driven
primarily by DL20; a cross-year robustness claim would be premature.
