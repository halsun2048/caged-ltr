# R7.9 fixed-config three-seed stability

```json
{
  "schema": "mind_r7_9_multiseed_v1",
  "configuration": {
    "fixed_without_search": true,
    "seeds": [
      42,
      2024,
      3407
    ],
    "model": "sentence-transformers/all-MiniLM-L6-v2",
    "epochs_max": 5,
    "batch_size": 256,
    "learning_rate": 2e-05,
    "precision": "bf16",
    "patience": 2
  },
  "runs": [
    {
      "seed": 42,
      "dev": {
        "epoch": 2,
        "ndcg10": 0.38938956874870184,
        "hit10": 0.752318718381113,
        "mrr": 0.3548176724213419
      },
      "calibration": {
        "baseline": {
          "ndcg10": 0.3033396632495585,
          "hit10": 0.6348698250106701,
          "mrr": 0.27804294037337063,
          "latency_ms_per_query": 0.19408342127604358
        },
        "trained": {
          "ndcg10": 0.3874072900426088,
          "hit10": 0.7460520699957319,
          "mrr": 0.35329553167692856,
          "latency_ms_per_query": 0.11582990395397304
        },
        "ndcg10_delta": {
          "mean": 0.08406762679305037,
          "ci95_low": 0.0752171998672855,
          "ci95_high": 0.09281052475259391,
          "probability_positive": 1.0
        },
        "bucket_ndcg10_delta": {
          "head": {
            "mean": 0.0846555922643599,
            "ci95_low": 0.07561039402675478,
            "ci95_high": 0.09364696572709859,
            "probability_positive": 1.0
          },
          "tail": {
            "mean": 0.10339345600504561,
            "ci95_low": 0.03990275691037656,
            "ci95_high": 0.16108695815688087,
            "probability_positive": 0.9998
          },
          "torso": {
            "mean": 0.06583413156051379,
            "ci95_low": 0.024332721633826118,
            "ci95_high": 0.10770944109926985,
            "probability_positive": 0.9986
          }
        }
      },
      "elapsed_seconds": 251.41,
      "peak_gpu_memory_gib": 4.1,
      "files": {
        "train_report": {
          "path": "reports/experiments/mind_r7_7_english_student.json",
          "sha256": "3281fa2a9e551032cdc184b4c4d8c2db455f6fceb24ce79dfa49d9ab9e9a2b03"
        },
        "calibration_report": {
          "path": "reports/experiments/mind_r7_8_calibration.json",
          "sha256": "3ec75f44d448f39b3c15a03fd745527ad2a84b0f480083bf6e53eee9acb4a515"
        },
        "train_log_sha256": "95c84e7aa5b3044ee05849f0b7f5e5884a07954af0659a8ab56bff947572c419"
      }
    },
    {
      "seed": 2024,
      "dev": {
        "epoch": 4,
        "ndcg10": 0.3895283507187225,
        "hit10": 0.7474704890387859,
        "mrr": 0.3553363035736372
      },
      "calibration": {
        "baseline": {
          "ndcg10": 0.3033396632495585,
          "hit10": 0.6348698250106701,
          "mrr": 0.27804294037337063,
          "latency_ms_per_query": 0.19153400557413794
        },
        "trained": {
          "ndcg10": 0.384760622455074,
          "hit10": 0.7464788732394366,
          "mrr": 0.34855269618698925,
          "latency_ms_per_query": 0.11977037195095612
        },
        "ndcg10_delta": {
          "mean": 0.08142095920551548,
          "ci95_low": 0.07236357780472033,
          "ci95_high": 0.09059697974875731,
          "probability_positive": 1.0
        },
        "bucket_ndcg10_delta": {
          "head": {
            "mean": 0.08156614888112804,
            "ci95_low": 0.07220155895946317,
            "ci95_high": 0.09117642485961715,
            "probability_positive": 1.0
          },
          "tail": {
            "mean": 0.11852201468921959,
            "ci95_low": 0.051555042045116385,
            "ci95_high": 0.1844182433772582,
            "probability_positive": 0.9996
          },
          "torso": {
            "mean": 0.06484718259261159,
            "ci95_low": 0.020624023500466333,
            "ci95_high": 0.10850606349421608,
            "probability_positive": 0.9984
          }
        }
      },
      "elapsed_seconds": 315.13,
      "peak_gpu_memory_gib": 4.1,
      "files": {
        "train_report": {
          "path": "reports/experiments/mind_r7_9_seed2024_train.json",
          "sha256": "9e08c857bdd177c19bc90f646828e0282516b7147b86465583b6db9c48df0189"
        },
        "calibration_report": {
          "path": "reports/experiments/mind_r7_9_seed2024_calibration.json",
          "sha256": "c17c1baf140029d8cac0e9b50b09a27b8b8f88804c9d151fcf13c7f4dfafec08"
        },
        "train_log_sha256": "b125a98739a8323e90514e823413956d0ec00404efa38845d6932f8b9a77b8cf"
      }
    },
    {
      "seed": 3407,
      "dev": {
        "epoch": 2,
        "ndcg10": 0.3899851023146444,
        "hit10": 0.7485244519392917,
        "mrr": 0.3558657982020182
      },
      "calibration": {
        "baseline": {
          "ndcg10": 0.3033396632495585,
          "hit10": 0.6348698250106701,
          "mrr": 0.27804294037337063,
          "latency_ms_per_query": 0.18934006015704552
        },
        "trained": {
          "ndcg10": 0.3877871755610217,
          "hit10": 0.7473324797268459,
          "mrr": 0.3530603195118036,
          "latency_ms_per_query": 0.11610773644826247
        },
        "ndcg10_delta": {
          "mean": 0.08444751231146319,
          "ci95_low": 0.07594059152702055,
          "ci95_high": 0.09316585849649262,
          "probability_positive": 1.0
        },
        "bucket_ndcg10_delta": {
          "head": {
            "mean": 0.08514585265731776,
            "ci95_low": 0.0760799826364436,
            "ci95_high": 0.09408630037671627,
            "probability_positive": 1.0
          },
          "tail": {
            "mean": 0.09359171328276018,
            "ci95_low": 0.028961918781311956,
            "ci95_high": 0.15197003040985385,
            "probability_positive": 0.9986
          },
          "torso": {
            "mean": 0.0679475147384849,
            "ci95_low": 0.025748298924088565,
            "ci95_high": 0.10975437572723527,
            "probability_positive": 0.9992
          }
        }
      },
      "elapsed_seconds": 266.89,
      "peak_gpu_memory_gib": 4.1,
      "files": {
        "train_report": {
          "path": "reports/experiments/mind_r7_9_seed3407_train.json",
          "sha256": "11516dd6ec8fabc2941e71a8a2b5b81fc4b8ed3f31f503a45d39ea55afca249e"
        },
        "calibration_report": {
          "path": "reports/experiments/mind_r7_9_seed3407_calibration.json",
          "sha256": "9267349b795fa2b1c39cb480e63449c14f15c56a175ac2e74d0e5cc4ef58b0c5"
        },
        "train_log_sha256": "3381b379ef2f49d932438949a2a5030e8c4853ac84fa3b8b19976a430dea5176"
      }
    }
  ],
  "summary": {
    "dev_ndcg10": {
      "mean": 0.38963434059402285,
      "std": 0.0003115934005891513,
      "min": 0.38938956874870184,
      "max": 0.3899851023146444
    },
    "dev_hit10": {
      "mean": 0.7494378864530636,
      "std": 0.0025499223315941376,
      "min": 0.7474704890387859,
      "max": 0.752318718381113
    },
    "dev_mrr": {
      "mean": 0.3553399247323324,
      "std": 0.0005240722732813712,
      "min": 0.3548176724213419,
      "max": 0.3558657982020182
    },
    "calibration_ndcg10": {
      "mean": 0.3866516960195681,
      "std": 0.0016486957486211767,
      "min": 0.384760622455074,
      "max": 0.3877871755610217
    },
    "calibration_hit10": {
      "mean": 0.7466211409873381,
      "std": 0.0006519527237097615,
      "min": 0.7460520699957319,
      "max": 0.7473324797268459
    },
    "calibration_mrr": {
      "mean": 0.3516361824585738,
      "std": 0.002672965933367561,
      "min": 0.34855269618698925,
      "max": 0.35329553167692856
    },
    "calibration_ndcg10_delta": {
      "mean": 0.08331203277000969,
      "std": 0.0016486957486211765,
      "min": 0.08142095920551548,
      "max": 0.08444751231146319
    },
    "elapsed_seconds": {
      "mean": 277.81,
      "std": 33.23393446464021,
      "min": 251.41,
      "max": 315.13
    },
    "peak_gpu_memory_gib": {
      "mean": 4.1,
      "std": 0.0,
      "min": 4.1,
      "max": 4.1
    }
  },
  "calibration_bucket_delta_summary": {
    "head": {
      "mean": 0.08378919793426857,
      "std": 0.001940759936913258,
      "min": 0.08156614888112804,
      "max": 0.08514585265731776
    },
    "torso": {
      "mean": 0.06620960963053676,
      "std": 0.001583904255150854,
      "min": 0.06484718259261159,
      "max": 0.0679475147384849
    },
    "tail": {
      "mean": 0.10516906132567512,
      "std": 0.012559640231519925,
      "min": 0.09359171328276018,
      "max": 0.11852201468921959
    }
  },
  "boundary": {
    "calibration_is_fixed_confirmation_not_tuning": true,
    "mind_large_test_accessed": false,
    "nfcorpus_locked_test_accessed": false
  },
  "acceptance": {
    "all_seeds_calibration_beat_pretrained": true,
    "all_seed_bootstrap_ci95_exclude_zero": true,
    "head_direction_positive_all_seeds": true,
    "torso_direction_positive_all_seeds": true,
    "tail_direction_positive_all_seeds": true,
    "large_test_not_accessed": true
  }
}
```
