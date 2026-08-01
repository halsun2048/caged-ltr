# NFCorpus R6 final results

## Independent dev

| Model | NDCG@10 | Hit@10 | MRR | Head | Torso | Tail | Latency ms/query |
|---|---:|---:|---:|---:|---:|---:|---:|
| BM25 | 0.5402 | 0.7353 | 0.5743 | 0.5263 | 0.5863 | 0.4631 | 0.00 |
| TF-IDF | 0.3847 | 0.6723 | 0.4082 | 0.3137 | 0.4316 | 0.4418 | 4.49 |
| MIND-only MiniLM | 0.4169 | 0.7038 | 0.4323 | 0.3797 | 0.4665 | 0.4012 | 11.93 |
| NFCorpus-distilled MiniLM | 0.5025 | 0.7227 | 0.5307 | 0.4459 | 0.5549 | 0.5193 | 11.65 |
| FIRST | 0.5664 | 0.7479 | 0.5854 | 0.5465 | 0.6133 | 0.4866 | 427.99 |
| Previous frozen gate | 0.5668 | 0.7437 | 0.5852 | 0.5434 | 0.6134 | 0.4936 | 410.13 |

## Locked test

| Model | NDCG@10 | Hit@10 | MRR | FIRST call rate | Latency ms/query |
|---|---:|---:|---:|---:|---:|
| BM25 | 0.5088 | 0.7186 | 0.5425 | — | 0.00 |
| Distilled MiniLM | 0.4916 | 0.7153 | 0.5295 | — | 12.28 |
| FIRST | 0.5657 | 0.7254 | 0.5872 | 100.0% | 450.15 |
| Previous frozen gate | 0.5653 | 0.7254 | 0.5890 | 97.4% | 438.60 |
| R6 three-way gate | 0.5244 | 0.7254 | 0.5483 | 16.3% | 73.20 |

## Locked conclusion

- The distilled MiniLM significantly beats TF-IDF on dev and is the primary lightweight text encoder.
- The R6 gate is a valid cost-first Pareto point and significantly beats BM25 on locked test.
- The R6 gate is significantly below FIRST on locked test, so it does not replace the previous quality-first frozen gate.
- The locked test must not be reused for further tuning.
