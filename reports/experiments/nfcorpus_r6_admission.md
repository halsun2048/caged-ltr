# R6.0/R6.1 admission and distillation data

```json
{
  "alignment": {
    "duplicate_teacher_keys": 0,
    "extra_teacher_keys": 0,
    "missing_logits": 0,
    "missing_queries": 0
  },
  "bucket_metric_protocol": "conditional NDCG@10 over queries with relevant candidates in bucket; original rank positions retained",
  "dev_candidate_rows": 8303,
  "dev_queries": 476,
  "frequency_boundaries": {
    "head_min": 13.0,
    "tail_max": 4.0
  },
  "frequency_protocol": "unique train-query candidate exposure; 20/80 percentiles; unseen=tail",
  "generated_sha256": {
    "dev_listwise.parquet": "0a1327d361c692f8496e7f693e888b5f4920a8184891b0369e5849eae0f09ad3",
    "dev_queries.txt": "e605df44c8eda7809db1f6b8c642e617ad659781115dfaae26347770430cb211",
    "item_buckets.parquet": "6be5380fa899f6f1db6c98b9ed2f1f36f2c6583fba79ea840d757ea643ab5cb7",
    "train_hard_negatives.parquet": "ee8b19d285a6268228e6737a629428eb8346191694c525aa8b5ea57febd09b0c",
    "train_listwise.parquet": "00b7d4550b2b2ae22f81ab718f6049f90a6d04f3adb89f1a63bfcf97ac62008a",
    "train_pairwise.parquet": "bb750ffa1b0587debe64fb2f2ed250bf5511f2f251453ae5d11d605086284f0c",
    "train_queries.txt": "5e92c22a3ffffb09b8fd3bf7bb9761e0603d4dd4647fed3845869f798cc5bb83"
  },
  "hard_negative_rows": 5360,
  "latency_protocol": {
    "cached_first": "I/O only, never model latency",
    "first_model": "measured prefill/inference",
    "students": "warmup then synchronized end-to-end scoring"
  },
  "mind_role": "pretraining only; never NFCorpus final evidence",
  "pairwise_rows": 8792,
  "schema": "nfcorpus_r6_distillation_v1",
  "seed": 42,
  "source_sha256": {
    "data/processed/nfcorpus_r5/train_independent/candidates.parquet": "8e2fb259133585d6b7bb5ec22db1064a2487d70a0fab4a57cf4f8f9118113ebf",
    "data/processed/nfcorpus_r5/train_independent/qrels.parquet": "628cd613780187b1e93b64f91d92326d68377af0c08c0a4804cdc31790c8ba2b",
    "data/processed/nfcorpus_r5/train_independent/queries.parquet": "e4bbac8c659c1f67aa0941a665764de02c5233c94ca00ac9fabfe94af61a97a3",
    "data/teacher_labels/r5_8_first_train_independent/listwise_logits.parquet": "d5195dfbdaedd5647c951b696f777e28f947414707e9de15d11af58e3d6083b4"
  },
  "split_method": "sha256(query_id), first 80% train, final 20% dev",
  "train_candidate_rows": 33725,
  "train_queries": 1904,
  "untouched_test_accessed": false
}
```
