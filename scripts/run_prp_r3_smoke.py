"""Run the resumable 100-query synthetic PRP protocol smoke test."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from caged_ltr.teachers.prp_smoke import run_prp_smoke


def _duration(seconds: float) -> str:
    total = max(0, round(seconds))
    minutes, seconds = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:d}h{minutes:02d}m" if hours else f"{minutes:02d}m{seconds:02d}s"


class _Progress:
    def __init__(self, *, enabled: bool) -> None:
        self.enabled = enabled
        self.started = time.monotonic()
        self.last_rendered_prompt = -1
        self.line_open = False

    def __call__(self, event: dict[str, Any]) -> None:
        if not self.enabled:
            return
        prompt_done = int(event["prompt_done"])
        prompt_total = int(event["prompt_total"])
        stage = str(event["stage"])
        refresh_step = max(prompt_total // 1000, 1)
        if (
            stage not in {"resume", "query_complete"}
            and prompt_done - self.last_rendered_prompt < refresh_step
        ):
            return
        self.last_rendered_prompt = prompt_done
        width = 24
        filled = round(width * prompt_done / max(prompt_total, 1))
        bar = "#" * filled + "-" * (width - filled)
        sys.stderr.write("\r\033[2K")
        sys.stderr.write(
            f"[{bar}] queries={int(event['query_done']):>3}/{int(event['query_total'])} "
            f"prompts={prompt_done:>6}/{prompt_total} "
            f"stage={stage:<24} "
            f"elapsed={_duration(time.monotonic() - self.started)}"
        )
        self.line_open = True
        if stage == "query_complete" and int(event["query_done"]) == int(
            event["query_total"]
        ):
            sys.stderr.write("\n")
            self.line_open = False
        sys.stderr.flush()

    def finish(self) -> None:
        if self.enabled and self.line_open:
            sys.stderr.write("\n")
            sys.stderr.flush()
            self.line_open = False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/prp_r3_smoke"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/experiments/prp_r3_smoke.json"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--candidates", type=int, default=10)
    parser.add_argument("--sliding-passes", type=int, default=3)
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()
    progress = _Progress(enabled=args.progress)
    summary = run_prp_smoke(
        args.output_dir,
        seed=args.seed,
        query_count=args.queries,
        candidates_per_query=args.candidates,
        sliding_passes=args.sliding_passes,
        progress_callback=progress,
    )
    progress.finish()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary_report = args.report.with_suffix(args.report.suffix + ".tmp")
    temporary_report.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_report.replace(args.report)
    print(
        json.dumps(
            {
                "stage": summary["stage"],
                "result_type": summary["result_type"],
                "report": str(args.report),
                "acceptance": summary["acceptance"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
