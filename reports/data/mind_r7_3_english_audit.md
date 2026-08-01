# R7.3 English behavior-data admission

All model inputs and external behavior data are frozen to **English**. Non-English and translated MIND variants are rejected.

```json
{
  "schema": "mind_r7_3_english_admission_v1",
  "language_policy": {
    "required_language": "English",
    "reason": "NFCorpus, FIRST prompts, and the MiniLM distillation/evaluation pipeline are English.",
    "translated_or_non_english_data_allowed": false
  },
  "preferred_source": {
    "dataset": "mteb/MindSmallReranking",
    "provenance": "English MIND-small reranking derivative",
    "use": "external behavior pretraining/evaluation only; never NFCorpus final evidence"
  },
  "official_candidates": [
    {
      "path": "data/external/mind/official_small/mindsmall_top_ranked_00001.parquet",
      "bytes": 10113239,
      "sha256": "da9b36d75fdf2215c2d4eac99e44dcb6ae0e64ef45fa274c022bae41c85b71db",
      "valid_parquet": true,
      "rows": 232166,
      "columns": [
        "query-id",
        "corpus-ids"
      ],
      "text_columns_checked": [
        "query-id",
        "corpus-ids"
      ],
      "behavior_fields": [],
      "language": {
        "sample_characters": 1000000,
        "latin_alphabetic_ratio": 1.0,
        "english_stopword_ratio": 0.09551546204879699,
        "tokens": 165502
      },
      "english": true,
      "admitted": false
    }
  ],
  "validated_relational_bundle": {
    "schema": "mind_r7_5_english_bundle_v1",
    "source": "mteb/MindSmallReranking test configuration derived from English MIND",
    "license": "other; Microsoft Research License terms must be retained",
    "language_policy": "English only; non-English/translated variants rejected",
    "raw_groups": {
      "corpus": {
        "files": 1,
        "bytes": 486691,
        "rows": 5277,
        "columns": [
          "id",
          "text",
          "title"
        ],
        "expected": {
          "files": 1,
          "rows": 5277,
          "columns": [
            "id",
            "text",
            "title"
          ]
        },
        "matches_expected": true,
        "file_sha256": {
          "test-00000-of-00001.parquet": "4d4c58ef932f9b77c284f9501d40ca5f2789f7133865aae38872f66339793d2b"
        }
      },
      "queries": {
        "files": 1,
        "bytes": 124721520,
        "rows": 2362514,
        "columns": [
          "id",
          "text"
        ],
        "expected": {
          "files": 1,
          "rows": 2362514,
          "columns": [
            "id",
            "text"
          ]
        },
        "matches_expected": true,
        "file_sha256": {
          "test-00000-of-00001.parquet": "ff6cc644acdc4fe23092d7920a017abfe5db60d7b94455539ad11b49fccd8b95"
        }
      },
      "data": {
        "files": 16,
        "bytes": 501004044,
        "rows": 97006943,
        "columns": [
          "query-id",
          "corpus-id",
          "score"
        ],
        "expected": {
          "files": 16,
          "rows": 97006943,
          "columns": [
            "query-id",
            "corpus-id",
            "score"
          ]
        },
        "matches_expected": true,
        "file_sha256": {
          "test-00000-of-00016.parquet": "72259c1f87e82799dfa1421ac8f876dcc0f462d159ca6f10e95c818ea0f334c6",
          "test-00001-of-00016.parquet": "1db6560c309632c1f3fb141af68dce36d2a955e201f51d3b4e0db0375a296ff9",
          "test-00002-of-00016.parquet": "920c0b4393a90e965500a95bb811a33ae9425e73724382e239ebcf4351b811ca",
          "test-00003-of-00016.parquet": "041c0764debfdb931b061b5041dddcabea3083d98a0f6e564c68c41b817cd995",
          "test-00004-of-00016.parquet": "e5e04ae949ad03505d9f04d49d5cc3b7d928993388c619b2335de47c07e8ad3f",
          "test-00005-of-00016.parquet": "8fc8989f1ce91c5f47c835bc449ab49494f9306089ed865de16835c93d4f63ce",
          "test-00006-of-00016.parquet": "6be3bba61ddabb39daef737f3d367d373247d6f72ba17d8cee34ab8c337236a1",
          "test-00007-of-00016.parquet": "8b09f066b43a9120675df87d7ebf4c6070fdce33b8a21652c9b75f5f117183d2",
          "test-00008-of-00016.parquet": "fca8dbb0eb681999189ef580abf077d3298a58eb9aa26166d4a348ad80d4534c",
          "test-00009-of-00016.parquet": "af1d978d8b89ff40c8f5fc5dbe77e821c14ff225dca3b5dd7eecc75a3281781e",
          "test-00010-of-00016.parquet": "0d8549be3ed1bbe75de7c922421bbdcc977bf4ab788450ca6ead6c5b8e9f8a2f",
          "test-00011-of-00016.parquet": "96890d9b83c4b4026f4482fbeb2446ddc2f1ebd826dc88af3e740651c484d959",
          "test-00012-of-00016.parquet": "1a29fee21a3d3acfb6a1e55cef7aa6918fcc795bcf6d2f1a98853b379a2a8426",
          "test-00013-of-00016.parquet": "b3dc1c6e671eff55cb5c33d5bbe5ce37b798fae77ae9c3eb9717410e8f4d741c",
          "test-00014-of-00016.parquet": "6979dc041fefa7357ca49a23e7d799be9478008039afb4ac17460f1b45ecf848",
          "test-00015-of-00016.parquet": "e5e457f2353db2c70e0d5de60f4e75d0716ca51f18f349b66f839d807e0ed399"
        }
      },
      "top_ranked": {
        "files": 11,
        "bytes": 76164665,
        "rows": 2362514,
        "columns": [
          "query-id",
          "corpus-ids"
        ],
        "expected": {
          "files": 11,
          "rows": 2362514,
          "columns": [
            "query-id",
            "corpus-ids"
          ]
        },
        "matches_expected": true,
        "file_sha256": {
          "test-00000-of-00011.parquet": "8e7461106ec0c3a7a1148778e36222f5781361656dfdd4fe48af5fc226b4519d",
          "test-00001-of-00011.parquet": "6292f7b676e5a5db721fcf4c5daaf10ba960050d0a2cebfb18309cbf1f275d7e",
          "test-00002-of-00011.parquet": "09ca2a26345c432d25462671845f701f21763ef9e930e426de4d69b78b2af928",
          "test-00003-of-00011.parquet": "156fb669104fbaf8ad75f39323a467f1a4435d3234336926be900602fbcc2f20",
          "test-00004-of-00011.parquet": "951e1d401b8ec4b379f0b3cb9e4fbe0ecd9925d9920b2f57ac4addcb3e82e419",
          "test-00005-of-00011.parquet": "02f1c5a5c89f1192e9d090492a388282b87063342ef12c17f3d5b9e4bec4a715",
          "test-00006-of-00011.parquet": "13e785dda50cf063200a0b4c697492932908044bf0a9f7f77d01271cb53c8ec0",
          "test-00007-of-00011.parquet": "882a9d481f0d3b156c34ce1b6547ae1d35ed07431f49592df17bc9cb8459f234",
          "test-00008-of-00011.parquet": "8d07075da51f44e34a9f0270c72d2b2e4725e41cf6e4c99c85ba8f1991efb1c3",
          "test-00009-of-00011.parquet": "d60312b056386f2deb422ae01aecfe77b9ea1ff4cccd419221b3420fa9287c70",
          "test-00010-of-00011.parquet": "9e0afffeb8a5cd6f043e06a07c8594af2b8f5d06f5916e7b788eb8b8d9d705e5"
        }
      }
    },
    "language": {
      "characters": 1727475,
      "tokens": 283372,
      "latin_alphabetic_ratio": 0.9999345359534298,
      "english_stopword_ratio": 0.1371553999689454,
      "english_gate_passed": true
    },
    "split_policy": {
      "unit": "query_id",
      "seed": 20260801,
      "hash": "BLAKE2b-64 over mind-r7.5:{seed}:{query_id}",
      "buckets": {
        "train": "0-199",
        "dev": "200-219",
        "calibration": "220-239",
        "holdout": "240-9999"
      },
      "counts": {
        "calibration": 4686,
        "dev": 4744,
        "train": 47124
      },
      "holdout_queries": 2305960
    },
    "processed": {
      "selected_queries": 56554,
      "selected_labels": 2310729,
      "selected_candidate_lists": 56554,
      "files": {
        "data/processed/mind_r7_5/queries_selected.parquet": {
          "bytes": 826668,
          "sha256": "b20d9f477d084ad7911693e3f3e660feecd0ef3ccc26c51cd81dff55ddb96bec"
        },
        "data/processed/mind_r7_5/labels_selected.parquet": {
          "bytes": 3275822,
          "sha256": "236396c6e8dd6d044bb1c33bdee86535b92f0dd0b12ed6079605bfb682460cbe"
        },
        "data/processed/mind_r7_5/candidates_selected.parquet": {
          "bytes": 3485234,
          "sha256": "73f3c2396f3ca6a39f653dd825eace30fb33dfa235c07f2cbf08a578b234bb6d"
        }
      }
    },
    "boundaries": {
      "external_pretraining_or_independent_dev_only": true,
      "nfcorpus_final_evidence": false,
      "mind_holdout_accessed": false,
      "nfcorpus_locked_test_accessed": false
    },
    "acceptance": {
      "all_raw_files_match_manifest": true,
      "english_only": true,
      "query_splits_disjoint": true,
      "all_selected_queries_have_candidate_lists": true,
      "all_selected_queries_have_labels": true,
      "large_holdout_preserved": true
    }
  },
  "explicitly_rejected_non_official_files": [
    "data/external/mind/sproos_mindsmall_tr/README.md",
    "data/external/mind/sproos_mindsmall_tr/train.parquet"
  ],
  "download_status": "complete",
  "blocker": null,
  "acceptance": {
    "english_only_policy_frozen": true,
    "at_least_one_complete_english_behavior_shard": true,
    "complete_relational_bundle": true,
    "non_english_data_used": false,
    "nfcorpus_test_accessed": false
  }
}
```
