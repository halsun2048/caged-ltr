# CAGED-LTR paper-ready summary table

| Evaluation | BM25 | FIRST | CAGED gate/student |
|---|---:|---:|---:|
| NFCorpus strict test NDCG@10 | 0.5683 | 0.6066 | 0.6066 (frozen gate) |
| NFCorpus independent dev2 NDCG@10 | — | 0.6167 | 0.6166 (gate) |
| Independent text-student dev NDCG@10 | — | 0.6167 teacher | 0.4495 TF-IDF+Ridge |
| Independent text-student Brier | — | — | 0.7899 |
| Student fit / score time | — | — | 11.1 s / 0.009 s |

The gate result is not an additional gain over FIRST; it is a frozen routing
reproduction.  The TF-IDF student is an efficiency baseline and should not be
presented as matching the teacher.
