"""Replay real FLAN-T5 Sliding-10 exactly from the completed Allpair cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from caged_ltr.teachers.prp_sliding_replay import (
    run_sliding10_cached_replay,
)

ALLPAIR_MANIFEST_ID = (
    "6f632b5968b6b014f68c9edbac54006a3355b50275228593597a4412d7d70c76"
)
OVERLAY_MANIFEST_ID = (
    "947865f82ef67a0ec996165fbc14ab6e517ee6b19ec7fb0a87a162d9ba33d1b9"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--teacher-input",
        type=Path,
        default=Path(
            "data/processed/prp_trec_dl_top100/teacher_inputs.jsonl"
        ),
    )
    parser.add_argument(
        "--qrels",
        type=Path,
        default=Path("data/processed/prp_trec_dl_top100/qrels.parquet"),
    )
    parser.add_argument(
        "--allpair-output-dir",
        type=Path,
        default=Path("runs/prp_r3_1c_flan_t5_xl_top100"),
    )
    parser.add_argument(
        "--truncation-overlay",
        type=Path,
        default=Path(
            "runs/prp_r3_1d_truncation_audit/"
            "rescored_truncated_responses.jsonl"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/prp_r3_2_sliding10_cached_replay"),
    )
    parser.add_argument("--passes", type=int, default=10)
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()
    if args.passes != 10:
        parser.error("R3.2 pre-registers exactly 10 backward passes")
    if args.random_seed != 42:
        parser.error("R3.2 pre-registers random seed 42")
    print(
        "[1/3] fixing BM25, reverse-BM25, and seed-42 random Sliding-10 "
        "rankings from cache",
        flush=True,
    )
    summary = run_sliding10_cached_replay(
        teacher_input_path=args.teacher_input,
        qrels_path=args.qrels,
        allpair_output_dir=args.allpair_output_dir,
        truncation_overlay_path=args.truncation_overlay,
        output_dir=args.output_dir,
        passes=args.passes,
        random_seed=args.random_seed,
        expected_allpair_manifest_identity=ALLPAIR_MANIFEST_ID,
        expected_overlay_manifest_identity=OVERLAY_MANIFEST_ID,
    )
    print("[2/3] all rankings fixed before qrels evaluation", flush=True)
    print(
        "[3/3] "
        + json.dumps(
            {
                "stage": summary["stage"],
                "acceptance": summary["acceptance"],
                "report": str(args.output_dir / "summary.json"),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
