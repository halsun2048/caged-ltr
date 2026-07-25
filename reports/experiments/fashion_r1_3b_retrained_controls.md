# Fashion R1.3b retrained semantic controls

Each control is retrained with validation-only early stopping and tested once against the full catalog.

| Initialization | LLMInit NDCG@10 | Fusion NDCG@10 |
|---|---:|---:|
| real | 0.323864 | 0.337715 |
| shuffled | 0.347501 | 0.336702 |
| matched_random | 0.333709 | 0.320860 |

## Interpretation

- The best Overall NDCG system is shuffled LLMInit, not real fusion.
- Real semantic initialization does not beat shuffled or matched-random initialization.
- Real fusion beats both control fusions, but its Overall margin over shuffled fusion is small.
- Relative to shuffled LLMInit, real fusion trades Overall performance for better Tail, Torso, and cold-start NDCG.

## Acceptance

- real_initialization_beats_shuffled_llm_init: fail
- real_initialization_beats_matched_random_llm_init: fail
- real_fusion_beats_shuffled_retrained_fusion: pass
- real_fusion_beats_matched_random_retrained_fusion: pass
- real_fusion_beats_best_control_overall: fail
