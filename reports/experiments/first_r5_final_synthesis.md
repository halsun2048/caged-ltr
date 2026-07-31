# FIRST R5 cross-dataset evidence synthesis

## Frozen results

| Benchmark / protocol | BM25 | FIRST | FIRST − BM25 | PRP control |
|---|---:|---:|---:|---:|
| TREC-DL top-20, 97 queries | 0.491249 | 0.630504 | +0.139255 | 0.602779 |
| NFCorpus top-20, 323 queries | 0.306935 | 0.343543 | +0.036608 | 0.327937 |

The TREC-DL FIRST–PRP paired difference is `+0.027725` (95% CI
`[0.01154, 0.04473]`). The NFCorpus difference is `+0.015606` (95% CI
`[0.00530, 0.02586]`). Both datasets therefore show the same direction:
FIRST improves over the matched PRP control on the frozen candidate pool.

NFCorpus also has an official Pyserini 2.3.0 BM25 reference of `0.321784`
(308/323 queries returned at least one hit; 15 single-token queries had no
hit). This reference is a separate candidate pool and is not substituted for
the matched-pool comparison.

## Conclusions

1. The main FIRST ranking gain replicates beyond TREC-DL, including on a
   biomedical BEIR task with different query and document distributions.
2. The gain is not explained by generic pairwise-teacher strength alone:
   matched FLAN-T5-XL PRP is weaker on both frozen pools.
3. Identity/z-score debiasing is not a universal improvement: it is neutral on
   TREC-DL and significantly harmful on NFCorpus. Raw FIRST remains the default.
4. The evidence supports an offline reranking claim, not a production or
   end-to-end retrieval claim. Candidate recall, latency, and larger-pool
   behavior remain outside this gate.

## Next gate

Freeze these results and perform per-query error analysis (gain/loss buckets,
query length, relevance density, and candidate overlap). A new GPU experiment
is justified only if the analysis identifies a specific mechanism or if an
official-candidate-pool matched rerun is required for the paper.
