"""Run resumable FLAN-T5-XL PRP likelihood inference on frozen TREC-DL inputs."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from caged_ltr.teachers import (
    DEFAULT_FLAN_T5_XL_REVISION,
    FlanT5PairwiseTeacher,
)
from caged_ltr.teachers.prp_real import run_prp_r3_1b


def _duration(seconds: float) -> str:
    total = max(0, round(seconds))
    minutes, seconds = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    return (
        f"{hours:d}h{minutes:02d}m"
        if hours
        else f"{minutes:02d}m{seconds:02d}s"
    )


class _Progress:
    def __init__(self, *, enabled: bool) -> None:
        self.enabled = enabled
        self.started = time.monotonic()
        self.line_open = False

    def model_loading(self, model: str, revision: str) -> None:
        if not self.enabled:
            return
        print(
            f"[model] loading {model}@{revision[:12]} on CUDA; "
            "the first run downloads and verifies model files",
            file=sys.stderr,
            flush=True,
        )

    def __call__(self, event: dict[str, Any]) -> None:
        if not self.enabled:
            return
        done = int(event["prompt_done"])
        total = int(event["prompt_total"])
        width = 24
        filled = round(width * done / max(total, 1))
        bar = "#" * filled + "-" * (width - filled)
        elapsed = time.monotonic() - self.started
        rate = done / elapsed if elapsed > 0 else 0.0
        sys.stderr.write("\r\033[2K")
        sys.stderr.write(
            f"[{bar}] prompts={done:>4}/{total:<4} "
            f"stage={event['stage']!s:<10} "
            f"batch={int(event['batch_size']):>2} "
            f"rate={rate:>5.2f}/s elapsed={_duration(elapsed)}"
        )
        sys.stderr.flush()
        self.line_open = True
        if done == total:
            sys.stderr.write("\n")
            sys.stderr.flush()
            self.line_open = False

    def finish(self) -> None:
        if self.enabled and self.line_open:
            sys.stderr.write("\n")
            sys.stderr.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--teacher-input",
        type=Path,
        default=Path(
            "data/processed/prp_trec_dl_top10/teacher_inputs.jsonl"
        ),
    )
    parser.add_argument(
        "--qrels",
        type=Path,
        default=Path("data/processed/prp_trec_dl_top10/qrels.parquet"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/prp_r3_1b_flan_t5_xl_1q"),
    )
    parser.add_argument("--model", default="google/flan-t5-xl")
    parser.add_argument("--revision", default=DEFAULT_FLAN_T5_XL_REVISION)
    parser.add_argument("--cache-dir", type=Path, default=Path(".hf-cache"))
    parser.add_argument("--queries", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--max-ordered-prompts",
        type=int,
        default=32,
        help="Admission cap; use 0 to complete every selected query.",
    )
    parser.add_argument(
        "--scoring-mode",
        choices=("likelihood", "generation"),
        default="likelihood",
    )
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float16",
    )
    parser.add_argument("--max-input-tokens", type=int, default=512)
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()
    progress = _Progress(enabled=args.progress)
    progress.model_loading(args.model, args.revision)
    teacher = FlanT5PairwiseTeacher.from_pretrained(
        model_name=args.model,
        model_revision=args.revision,
        device="cuda",
        dtype=args.dtype,
        scoring_mode=args.scoring_mode,
        max_input_tokens=args.max_input_tokens,
        cache_dir=str(args.cache_dir),
    )
    summary = run_prp_r3_1b(
        teacher,
        teacher_input_path=args.teacher_input,
        output_dir=args.output_dir,
        qrels_path=args.qrels,
        query_limit=args.queries,
        batch_size=args.batch_size,
        max_ordered_prompts=(
            args.max_ordered_prompts
            if args.max_ordered_prompts > 0
            else None
        ),
        progress_callback=progress,
    )
    progress.finish()
    print(
        json.dumps(
            {
                "stage": summary["stage"],
                "queries": summary["query_count"],
                "prompts": summary["cached_ordered_prompts"],
                "expected_prompts": summary["expected_ordered_prompts"],
                "qrels_accessed": summary["qrels_accessed"],
                "report": str(args.output_dir / "summary.json"),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
