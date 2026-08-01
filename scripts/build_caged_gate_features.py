"""Build leakage-safe gate features and a frozen threshold preregistration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--candidates", type=Path, required=True)
    ap.add_argument("--output-features", type=Path, required=True)
    ap.add_argument("--output-prereg", type=Path, required=True)
    args = ap.parse_args()
    queries = pd.read_parquet(args.queries)
    candidates = pd.read_parquet(args.candidates).copy()
    candidates["passage_chars"] = candidates["passage"].fillna("").str.len()
    item_freq = candidates.groupby("passage_id")["query_id"].nunique()
    candidates["item_candidate_frequency"] = candidates["passage_id"].map(item_freq)
    agg = (
        candidates.groupby("query_id")
        .agg(
            candidate_count=("passage_id", "count"),
            mean_passage_chars=("passage_chars", "mean"),
            short_passage_rate=("passage_chars", lambda x: float((x < 80).mean())),
            mean_item_candidate_frequency=("item_candidate_frequency", "mean"),
            max_item_candidate_frequency=("item_candidate_frequency", "max"),
            mean_bm25_rank=("bm25_rank", "mean"),
        )
        .reset_index()
    )
    q = queries[["query_id", "query"]].copy()
    q["query_chars"] = q["query"].fillna("").str.len()
    q["query_tokens"] = q["query"].fillna("").str.split().str.len()
    features = q.merge(agg, on="query_id", how="inner")
    features["behavior_uncertainty"] = np.nan
    features["behavior_uncertainty_status"] = "unavailable_public_NFCorpus"
    args.output_features.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(args.output_features, index=False)
    prereg = {
        "protocol": "CAGED-LTR gate v2",
        "feature_file": str(args.output_features),
        "features": [
            "entropy",
            "permutation_stability",
            "query_chars",
            "query_tokens",
            "mean_passage_chars",
            "short_passage_rate",
            "mean_item_candidate_frequency",
            "max_item_candidate_frequency",
            "behavior_uncertainty",
        ],
        "behavior_uncertainty": (
            "reserved; unavailable on public NFCorpus and must not be imputed from qrels"
        ),
        "threshold_grid": {
            "entropy_max": [0.3, 0.5, 0.7],
            "stability_min": [0.2, 0.4, 0.6],
            "short_passage_rate_max": [0.25, 0.5, 0.75],
        },
        "selection_split": "new independent dev with qrels required; do not use untouched test",
        "test_evaluation": "one-time only after threshold lock",
    }
    args.output_prereg.write_text(json.dumps(prereg, indent=2, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "queries": len(features),
                "features": list(features.columns),
                "prereg": str(args.output_prereg),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
