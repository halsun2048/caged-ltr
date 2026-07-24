"""Convert one dataset from the LLM-ESR authors' bundle to common Parquet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from caged_ltr.data import LLMESRAuthorPreparationConfig, prepare_llmesr_author


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("yelp", "fashion", "beauty"), required=True)
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("data/raw/yelp/LLMESR_author_processed.zip"),
    )
    parser.add_argument("--processed-dir", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    processed_dir = args.processed_dir or Path(
        f"data/processed/{args.dataset}_llmesr_author"
    )
    report = args.report or Path(
        f"reports/data/{args.dataset}_llmesr_author_summary.json"
    )
    manifest = prepare_llmesr_author(
        LLMESRAuthorPreparationConfig(
            archive=args.archive,
            processed_dir=processed_dir,
            report_path=report,
            dataset_name=args.dataset,
        )
    )
    print(json.dumps(manifest["statistics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
