"""Train SASRec or frozen-semantic late fusion on author-processed Yelp."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from caged_ltr.sequential import YelpSASRecRunConfig, run_yelp_sasrec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/reproduction/yelp_sasrec_smoke.yaml"),
    )
    parser.add_argument("--model", choices=("sasrec", "llm_init", "late_fusion"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-users", type=int)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--semantic-weight", type=float)
    args = parser.parse_args()
    config = YelpSASRecRunConfig.from_yaml(
        args.config,
        model=args.model,
        output_dir=args.output_dir,
        seed=args.seed,
        max_users=args.max_users,
        max_epochs=args.max_epochs,
        semantic_weight=args.semantic_weight,
    )
    summary = run_yelp_sasrec(config)
    print(
        json.dumps(
            {
                "model": summary["model"],
                "best_epoch": summary["best_epoch"],
                "validation": summary["validation"]["item_frequency"]["overall"],
                "test": summary["test"]["item_frequency"]["overall"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
