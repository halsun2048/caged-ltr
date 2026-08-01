# Text-aware student and efficiency audit

On the independent 2,380-query NFCorpus split, a lightweight TF-IDF word
1–2-gram Ridge student was trained to reproduce frozen FIRST logits.

- Train/validation queries: 1,904 / 476
- Features: 50,000 sparse text features
- Fit time: 11.1 s CPU
- Validation scoring time: 0.009 s
- Student NDCG@10: `0.4495`
- FIRST teacher NDCG@10: `0.6167`
- Brier score of naive sigmoid scores: `0.7899`

The student is extremely fast but substantially below the teacher and poorly
calibrated.  This is a useful efficiency baseline, not a successful distilled
model.  Tail/torso/head rows are reported in the JSON artifact; a proper
production student needs a calibrated objective and richer text/behavior
features.
