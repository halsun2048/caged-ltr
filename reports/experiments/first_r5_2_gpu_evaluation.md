# FIRST R5.2 GPU evaluation

- **Model:** `rryisthebest/First_Model` at revision `64eba9b83c174439d2b6f5d333fbb822b38d73a7`
- **Protocol:** 400 frozen prompts (100 queries × baseline/reverse/random-permutation/identifier-remap), CUDA, no training
- **First-token stage:** 400/400 complete; mean normalized entropy `0.2153` (SD `0.1756`), mean top-1/top-2 margin `2.8758` (SD `2.6119`)
- **Full-generation audit:** fixed 20 baseline prompts, 20/20 parseable; pair agreement mean `0.8687` (SD `0.0667`), range `0.6895–0.9684`
- **Runtime:** mean prefill `0.2766s` for first-token stage; mean prefill `0.2997s` and decoding `2.0154s` for full generation
- **Acceptance:** CUDA used, model identity locked, cache resumable, all selected records complete

These results establish that the R5.2 bounded inference protocol executes at the registered scale. They are not relevance or ranking-quality gains: no qrels are read and no causal/effectiveness claim follows from entropy, margins, or pair agreement alone.
