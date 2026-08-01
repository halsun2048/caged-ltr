# R7.7 English MIND MiniLM student

- Dataset: frozen English MIND-MTEB train/dev split.
- GPU: NVIDIA GeForce RTX 4090; BF16; batch size 256.
- Training: 137,439 hard pairs; 4,744 dev queries; early stopping at epoch 4.
- Pretrained baseline: NDCG@10 0.301911, Hit@10 0.622470, MRR 0.275637.
- Best epoch 2: NDCG@10 0.389390, Hit@10 0.752319, MRR 0.354818.
- Absolute change: NDCG@10 +0.087479, Hit@10 +0.129849, MRR +0.079181.
- Runtime: 251.41 seconds.
- Checkpoint SHA-256: `253241fba6b18e14b89cc5d59e48a6c5538a3b78106246c62897e5c0025f6bc8`.

The MIND holdout, calibration split, and locked NFCorpus test were not accessed.
