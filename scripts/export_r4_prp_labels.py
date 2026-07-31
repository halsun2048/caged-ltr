"""Export frozen R4 PRP Allpair rankings as grouped RankNet labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from caged_ltr.data.instruction_distillation import export_prp_teacher_labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--teacher-summary",
        type=Path,
        default=Path("runs/r4_teacher_flan_t5_xl/summary.json"),
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=Path("data/processed/r4_msmarco_1k/candidates.parquet"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/teacher_labels/r4_prp_allpair_1k.parquet"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/data/r4_prp_allpair_1k_summary.json"),
    )
    args = parser.parse_args()
    report = export_prp_teacher_labels(
        args.teacher_summary,
        args.candidates,
        args.output,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_suffix(args.report.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.report)
    print(json.dumps({**report, "report": str(args.report)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
