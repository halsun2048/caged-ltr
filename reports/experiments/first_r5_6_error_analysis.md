# FIRST R5.6 NFCorpus error analysis

This analysis uses the frozen 323-query NFCorpus predictions, the baseline
identifier-logit ranking, and the same top-20 candidate pool used for the
FIRST/PRP comparison. It is descriptive; no additional model fitting or GPU
inference was performed.

## Gain/loss buckets

| Bucket vs BM25 | Queries |
|---|---:|
| Gain | 114 |
| Loss | 49 |
| Tie | 160 |

Mean per-query FIRST−BM25 NDCG@10 is `+0.036243`; FIRST−PRP is `+0.015240`
under this per-query reconstruction (the aggregate report remains the
pre-registered metric).

Gains are associated with longer queries: mean query length is 4.54 words for
gain queries versus 2.98 for loss queries; Spearman correlation between query
length and FIRST−BM25 gain is `0.337`. Relevant-document density is not a
strong explanatory factor (correlation `0.092`), with mean relevant counts of
48.9 for gains and 71.9 for losses.

The pattern suggests that FIRST is most useful when the query contains enough
semantic context to distinguish near-duplicate biomedical passages. Short,
ambiguous queries are the main failure bucket and should be targeted by a
future query-length-aware fallback or calibration study. This is a hypothesis
for a held-out follow-up, not a tuned result.

## Length-bin robustness

Without fitting any threshold, the frozen queries were split into bins:

| Query length | Queries | Mean FIRST−BM25 | Bootstrap 95% CI | Positive fraction |
|---|---:|---:|---:|---:|
| 1–2 words | 172 | +0.0085 | [-0.0026, 0.0208] | 19.8% |
| 3–4 words | 55 | +0.0465 | [0.0120, 0.0862] | 38.2% |
| 5+ words | 96 | +0.0801 | [0.0574, 0.1053] | 61.5% |

The length pattern is therefore not driven only by a few outliers: the 5+
word bin has a clearly positive interval, while the 1–2 word bin is not
significantly different from zero. A fallback policy must be validated on a
separate split before being used in a headline result.

Machine-readable per-query rows and the top gain/loss lists are in
`first_r5_6_error_analysis.json` and `.csv`.
