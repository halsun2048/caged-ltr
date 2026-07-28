"""Train SASRec or frozen-semantic late fusion on author-processed Yelp."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from caged_ltr.sequential import (
    YelpSASRecRunConfig,
    evaluate_yelp_test_checkpoint,
    run_yelp_sasrec,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/reproduction/yelp_sasrec_smoke.yaml"),
    )
    parser.add_argument(
        "--model",
        choices=(
            "sasrec",
            "llm_init",
            "semantic_only",
            "late_fusion",
            "dual_view",
            "dual_view_no_ca",
            "dual_view_unshared",
            "dual_view_capacity",
            "raw_semantic_only",
        ),
    )
    parser.add_argument("--raw-semantic-path", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-users", type=int)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--semantic-weight", type=float)
    parser.add_argument(
        "--validation-only",
        action="store_true",
        help="Never construct or evaluate the test split",
    )
    parser.add_argument(
        "--test-checkpoint",
        type=Path,
        help="Test an externally selected validation-only checkpoint without retraining",
    )
    args = parser.parse_args()
    config = YelpSASRecRunConfig.from_yaml(
        args.config,
        model=args.model,
        output_dir=args.output_dir,
        seed=args.seed,
        max_users=args.max_users,
        max_epochs=args.max_epochs,
        semantic_weight=args.semantic_weight,
        raw_semantic_path=args.raw_semantic_path,
        test_after_selection=False if args.validation_only else None,
    )
    if args.test_checkpoint is not None:
        metrics = evaluate_yelp_test_checkpoint(
            config,
            checkpoint_path=args.test_checkpoint,
        )
        print(json.dumps({"test": metrics["item_frequency"]["overall"]}, indent=2))
        return
    summary = run_yelp_sasrec(config)
    print(
        json.dumps(
            {
                "model": summary["model"],
                "best_epoch": summary["best_epoch"],
                "validation": summary["validation"]["item_frequency"]["overall"],
                "test": (
                    summary["test"]["item_frequency"]["overall"]
                    if summary["test"] is not None
                    else None
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
