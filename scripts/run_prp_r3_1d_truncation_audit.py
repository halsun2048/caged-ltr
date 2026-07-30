"""Rescore only R3.1c prompts truncated at 512 tokens, then reaggregate."""

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
from caged_ltr.teachers.prp_truncation import (
    discover_truncated_requests,
    run_truncation_sensitivity_audit,
)


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
        self.inference_started = self.started
        self.initial_done = 0
        self.line_open = False

    def model_loading(self, targets: int, model: str, revision: str) -> None:
        if self.enabled:
            print(
                f"[audit] targets={targets}; loading {model}@{revision[:12]} "
                "on CUDA with max_input_tokens=1024",
                file=sys.stderr,
                flush=True,
            )

    def __call__(self, event: dict[str, Any]) -> None:
        if not self.enabled:
            return
        done = int(event["prompt_done"])
        total = int(event["prompt_total"])
        if event["stage"] == "resume":
            self.initial_done = done
            self.inference_started = time.monotonic()
        newly_done = max(done - self.initial_done, 0)
        inference_elapsed = time.monotonic() - self.inference_started
        rate = newly_done / inference_elapsed if inference_elapsed > 0 else 0.0
        remaining = max(total - done, 0)
        eta = remaining / rate if rate > 0 else 0.0
        width = 24
        filled = round(width * done / max(total, 1))
        bar = "#" * filled + "-" * (width - filled)
        sys.stderr.write("\r\033[2K")
        sys.stderr.write(
            f"[{bar}] prompts={done:>3}/{total:<3} "
            f"stage={event['stage']!s:<9} batch={int(event['batch_size']):>2} "
            f"rate={rate:>5.2f}/s "
            f"ETA={_duration(eta) if rate > 0 else '--'}"
        )
        sys.stderr.flush()
        self.line_open = True
        if done == total:
            self.finish()

    def finish(self) -> None:
        if self.enabled and self.line_open:
            sys.stderr.write("\n")
            sys.stderr.flush()
            self.line_open = False


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
        "--baseline-output-dir",
        type=Path,
        default=Path("runs/prp_r3_1c_flan_t5_xl_top100"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/prp_r3_1d_truncation_audit"),
    )
    parser.add_argument("--model", default="google/flan-t5-xl")
    parser.add_argument("--revision", default=DEFAULT_FLAN_T5_XL_REVISION)
    parser.add_argument("--cache-dir", type=Path, default=Path(".hf-cache"))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-input-tokens", type=int, default=1024)
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()
    if args.max_input_tokens != 1024:
        parser.error("R3.1d pre-registers --max-input-tokens=1024")

    _, targets = discover_truncated_requests(
        teacher_input_path=args.teacher_input,
        baseline_output_dir=args.baseline_output_dir,
    )
    progress = _Progress(enabled=args.progress)
    progress.model_loading(len(targets), args.model, args.revision)
    teacher = FlanT5PairwiseTeacher.from_pretrained(
        model_name=args.model,
        model_revision=args.revision,
        device="cuda",
        dtype="float16",
        scoring_mode="likelihood",
        max_input_tokens=args.max_input_tokens,
        cache_dir=str(args.cache_dir),
    )
    summary = run_truncation_sensitivity_audit(
        teacher,
        teacher_input_path=args.teacher_input,
        baseline_output_dir=args.baseline_output_dir,
        output_dir=args.output_dir,
        qrels_path=args.qrels,
        batch_size=args.batch_size,
        progress_callback=progress,
    )
    progress.finish()
    print(
        json.dumps(
            {
                "stage": summary["stage"],
                "targets": summary["target_ordered_prompts"],
                "remaining_truncated": summary["remaining_truncated_inputs"],
                "qrels_accessed": summary["qrels_accessed"],
                "report": str(args.output_dir / "summary.json"),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
