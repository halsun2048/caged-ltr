"""Prepare the official Yelp archive into leakage-aware Parquet tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from caged_ltr.data.yelp import YelpPreparationConfig, prepare_yelp


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("data/raw/yelp/Yelp-JSON.zip"))
    parser.add_argument("--interim-dir", type=Path, default=Path("data/interim/yelp"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed/yelp_current"))
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/data/yelp_current_summary.json"),
    )
    parser.add_argument("--min-user-interactions", type=int, default=3)
    parser.add_argument("--min-item-interactions", type=int, default=3)
    parser.add_argument("--event-time-min", default="2000-01-01 00:00:00")
    parser.add_argument("--event-time-max", default="2019-12-31 00:00:00")
    parser.add_argument("--memory-limit", default="24GB")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    manifest = prepare_yelp(
        YelpPreparationConfig(
            archive=args.archive,
            interim_dir=args.interim_dir,
            processed_dir=args.processed_dir,
            report_path=args.report,
            min_user_interactions=args.min_user_interactions,
            min_item_interactions=args.min_item_interactions,
            event_time_min=args.event_time_min,
            event_time_max=args.event_time_max,
            memory_limit=args.memory_limit,
            threads=args.threads,
        )
    )
    print(json.dumps(manifest["statistics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
