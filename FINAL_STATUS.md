# Final Status

**Release status:** portfolio/research prototype complete.

CAGED-LTR demonstrates a cost-aware search reranking cascade with a lightweight
Student, a FIRST teacher adapter, deployable routing, failure fallback, offline
A/B replay, and an HTTP-only Streamlit demo.

## Locked claims

- FIRST improves over BM25 on the locked public ranking benchmarks.
- MiniLM distillation produces a much cheaper text-aware Student.
- The frozen R8.9 offline Tail-floor policy reaches `0.65006` NDCG@10 with 40%
  FIRST calls on its locked large-test artifact.
- The feature-consistent R19/R20 deployable Gate improves over Student but does
  not meet the preregistered near-FIRST threshold.
- R15 is offline randomized replay, not a live CTR/CVR experiment.

## Release artifacts

- API: `scripts/run_r16_api.py`
- UI: `app/streamlit_app.py`
- one-command demo: `scripts/run_demo_local.sh`
- final Gate audit: `docs/r20_r24_closeout.md`
- internship walkthrough: `docs/internship_showcase.md`

No further model search is admitted without new independent data or real
behavior logs. See `docs/r30_research_admission.md`.
