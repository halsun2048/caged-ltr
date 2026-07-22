"""Convert the LLM-ESR authors' Yelp bundle to the common Parquet schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from caged_ltr.data.yelp_author import YelpAuthorPreparationConfig, prepare_yelp_author


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("data/raw/yelp/LLMESR_author_processed.zip"),
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed/yelp_llmesr_author"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/data/yelp_llmesr_author_summary.json"),
    )
    args = parser.parse_args()
    manifest = prepare_yelp_author(
        YelpAuthorPreparationConfig(
            archive=args.archive,
            processed_dir=args.processed_dir,
            report_path=args.report,
        )
    )
    print(json.dumps(manifest["statistics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
