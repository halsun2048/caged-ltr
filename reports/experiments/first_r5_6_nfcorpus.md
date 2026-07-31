# FIRST R5.6 BEIR/NFCorpus external validation

NFCorpus was downloaded from the BEIR public release and frozen before qrels
were used for evaluation. A deterministic lexical BM25 implementation produced
20 candidates for each of 323 test queries; four FIRST perturbations were run
on 8 RTX 4090 GPUs (1,292 prompts total).

| Method | Linear graded NDCG@10 |
|---|---:|
| Frozen BM25 top-20 | 0.306935 |
| FIRST baseline | 0.343543 |
| FIRST debiased ensemble | 0.334447 |

Paired bootstrap 95% CIs:

- FIRST baseline − BM25: `[0.02493, 0.04899]`
- FIRST debiased − BM25: `[0.01366, 0.04223]`
- Debiased − baseline: `[-0.01523, -0.00333]`

FIRST baseline therefore improves over the independently prepared BM25 pool;
the identity/z-score debiasing post-process is significantly worse on this
dataset. This is the first external replication beyond TREC-DL, but the BM25
retrieval is a frozen in-project lexical implementation rather than Pyserini.

## Matched PRP control

The real FLAN-T5-XL PRP teacher was run on all 323 queries (122,740 ordered-pair
prompts) using eight RTX 4090 GPUs. PRP was evaluated only after prediction
freeze and was restricted to the identical BM25 top-20 candidate pool used by
FIRST:

| Method | Linear graded NDCG@10 |
|---|---:|
| FIRST baseline | 0.343543 |
| Matched PRP | 0.327937 |

The paired difference FIRST − PRP is `0.015606`, with a paired bootstrap 95% CI
of `[0.00530, 0.02586]`. Thus FIRST is significantly better than this matched
PRP control on NFCorpus. This is an outcome comparison, not evidence that the
two teachers optimize the same objective: PRP is a real FLAN-T5-XL pairwise
teacher, while FIRST is the frozen four-perturbation aggregation.

## Official Pyserini BM25 control

As a retrieval-only control, Pyserini 2.3.0 with the official
`beir-v1.0.0-nfcorpus.flat` index was run over the same 323 frozen queries. It
returned at least one hit for 308 queries (15 single-token queries have no hit
in the official index). Treating the 15 no-hit queries as an empty ranking, the
official BM25 NDCG@10 is `0.321784`. This is a separate top-20 retrieval pool,
so it should not be interpreted as a matched-candidate replacement for the
earlier FIRST/PRP comparison; it closes the baseline provenance gap and is the
reference BM25 number for future reruns.
