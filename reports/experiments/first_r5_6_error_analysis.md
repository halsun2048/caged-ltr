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

Machine-readable per-query rows and the top gain/loss lists are in
`first_r5_6_error_analysis.json` and `.csv`.
