# R7.7 English MIND MiniLM student

```json
{
  "schema": "mind_r7_7_english_student_v1",
  "identity": "9fd4122a8122cf28a57f4a3d34f48fbb1d1100cd39e3172bff21502d7250aad5",
  "language": "English",
  "model": "artifacts/models/all-MiniLM-L6-v2",
  "device": "cpu",
  "gpu": null,
  "precision": "fp32",
  "train_pairs": 32,
  "dev_queries": 10,
  "epochs_completed": 1,
  "early_stopped": false,
  "first_loss": 0.6572893857955933,
  "last_loss": 0.7019817233085632,
  "loss_decreased": false,
  "best_dev_ndcg10": 0.1774591932830688,
  "history": [
    {
      "epoch": 1,
      "ndcg10": 0.1774591932830688,
      "hit10": 0.7,
      "mrr": 0.16223498723498722,
      "latency_ms_per_query": 53.226815100060776
    }
  ],
  "elapsed_seconds": 1.87,
  "checkpoint": "/tmp/mind_r7_7_smoke_best.pt",
  "boundaries": {
    "dev_used_for_early_stopping_only": true,
    "calibration_accessed": false,
    "mind_holdout_accessed": false,
    "nfcorpus_locked_test_accessed": false
  }
}
```
