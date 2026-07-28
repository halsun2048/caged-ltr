"""Safely prepare the public RLMRec Yelp split and semantic embeddings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from caged_ltr.data.rlmrec import RLMRecYelpPreparationConfig, prepare_rlmrec_yelp


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw/rlmrec/official/yelp"),
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("data/raw/rlmrec/data.zip"),
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed/rlmrec_yelp_author"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/data/rlmrec_yelp_author_summary.json"),
    )
    args = parser.parse_args()
    report = prepare_rlmrec_yelp(
        RLMRecYelpPreparationConfig(
            raw_dir=args.raw_dir,
            processed_dir=args.processed_dir,
            report_path=args.report,
            archive=args.archive,
        )
    )
    print(json.dumps(report["statistics"], ensure_ascii=False, indent=2))
    print(json.dumps(report["semantic_provenance_audit"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
