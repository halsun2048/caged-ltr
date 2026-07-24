# Fashion R1.3b semantic audit and full-catalog results

Fusion is fixed to per-query full-catalog z-score with semantic weight 0.25. No Fashion metric selected a control or weight.

| Method | H@10 mean ± std | NDCG@10 mean ± std |
|---|---:|---:|
| sasrec | 0.364819 ± 0.012699 | 0.329204 ± 0.011737 |
| llm_init | 0.376182 ± 0.016036 | 0.323864 ± 0.011600 |
| semantic_only_real | 0.371124 ± 0.000000 | 0.306430 ± 0.000000 |
| semantic_only_shuffled | 0.066967 ± 0.000000 | 0.062626 ± 0.000000 |
| semantic_only_matched_random | 0.079173 ± 0.000000 | 0.077340 ± 0.000000 |
| fusion_real | 0.391504 ± 0.003176 | 0.337715 ± 0.005694 |
| fusion_shuffled | 0.378675 ± 0.014087 | 0.323788 ± 0.007741 |
| fusion_matched_random | 0.372700 ± 0.012541 | 0.320439 ± 0.005569 |

## Item-frequency NDCG@10

| Method | Head | Torso | Tail | Cold-start |
|---|---:|---:|---:|---:|
| sasrec | 0.563693 ± 0.014973 | 0.047771 ± 0.005572 | 0.071269 ± 0.021864 | 0.000000 ± 0.000000 |
| llm_init | 0.560180 ± 0.015000 | 0.049878 ± 0.006728 | 0.050964 ± 0.025808 | 0.000000 ± 0.000000 |
| semantic_only_real | 0.465884 ± 0.000000 | 0.110069 ± 0.000000 | 0.135009 ± 0.000000 | 0.080865 ± 0.000000 |
| semantic_only_shuffled | 0.104800 ± 0.000000 | 0.014801 ± 0.000000 | 0.013226 ± 0.000000 | 0.005056 ± 0.000000 |
| semantic_only_matched_random | 0.128273 ± 0.000000 | 0.022676 ± 0.000000 | 0.018496 ± 0.000000 | 0.005994 ± 0.000000 |
| fusion_real | 0.571469 ± 0.008375 | 0.060823 ± 0.006362 | 0.089057 ± 0.003171 | 0.000817 ± 0.000322 |
| fusion_shuffled | 0.562009 ± 0.011507 | 0.048510 ± 0.004617 | 0.044747 ± 0.011375 | 0.000437 ± 0.000064 |
| fusion_matched_random | 0.558220 ± 0.008971 | 0.052769 ± 0.007242 | 0.035963 ± 0.015453 | 0.000365 ± 0.000043 |

## Completed checks

- full_catalog_overall_direction_positive_all_seeds: pass
- full_catalog_tail_direction_positive_all_seeds: pass
- semantic_only_real_beats_shuffled_overall: pass
- semantic_only_real_beats_matched_random_overall: pass
- fusion_real_beats_shuffled_overall: pass
- fusion_real_beats_matched_random_overall: pass

## Provenance boundary

- Status: `not_fully_verifiable`.
- Item prompts use title, brand, date, price, feature, and description.
- No interaction history is directly included in item prompts.
- The raw metadata snapshot and generation timestamp are absent from the bundle.
- User embeddings are not used by these experiments.

## Remaining

- retrain LLMInit with shuffled semantic initialization for three seeds
- retrain LLMInit with matched-random initialization for three seeds
- regenerate embeddings from a versioned pre-cutoff metadata snapshot if available
