# R5.8 FIRST teacher-label handoff

The completed strict-pool FIRST run is now packaged for local student
experiments.  The handoff contains 295 queries, 1,180 prompt records (four
variants per query), and 20,188 candidate-level logits.  It records the model
revision, frozen prompt/results SHA-256 digests, protocol fingerprint, timing,
entropy, margin, identifier mapping, and output checksums.

Package: `data/teacher_labels/r5_8_first_nfcorpus/`

This closes the reproducible FIRST-to-student handoff for the strict-pool
supplement.  It does not create pairwise PRP labels; those remain optional and
are not required for the locked FIRST-vs-BM25 conclusion.
