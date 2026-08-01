# FIRST R5 engineering metrics

These measurements are taken from the preserved GPU logs, not estimated from
the model specification.

| Run | GPUs | Prompt records | Per-GPU wall time | Aggregate throughput |
|---|---:|---:|---:|---:|
| NFCorpus test | 8× RTX 4090 | 1,292 | 78.0–80.7 s | ~16.1 records/s |
| NFCorpus dev | 8× RTX 4090 | 1,296 | 85.2–87.8 s | ~14.8 records/s |

Each prompt record is a frozen FIRST variant input; these figures exclude model
download and candidate retrieval. The runs are therefore useful for inference
capacity planning, but they are not an end-to-end production latency claim.
Candidate recall, network overhead, batching under live traffic, and larger
candidate pools remain unmeasured.
