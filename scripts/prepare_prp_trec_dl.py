"""Download and prepare official DL19/DL20 Top-K passage candidates for PRP."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from caged_ltr.data.prp_trec_dl import prepare_prp_trec_dl

EXPECTED_SNAPSHOT_SHA256 = (
    "e3ffe3a166beee7b70c7859a2af1ab0c8b0b7a1012bfe330ae1ed09545484f55"
)


def _duration(seconds: float) -> str:
    total = max(0, round(seconds))
    minutes, seconds = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:d}h{minutes:02d}m" if hours else f"{minutes:02d}m{seconds:02d}s"


def _amount(done: int, total: int | None, stage: str) -> str:
    if stage == "download":
        done_value = done / (1024 * 1024)
        if total:
            return f"{done_value:7.1f}/{total / (1024 * 1024):.1f} MiB"
        return f"{done_value:7.1f} MiB"
    return f"{done:>7}/{total if total is not None else '?'} records"


class _Progress:
    def __init__(self, *, enabled: bool) -> None:
        self.enabled = enabled
        self.started = time.monotonic()
        self.files: list[str] = []
        self.line_open = False

    def __call__(self, event: dict[str, Any]) -> None:
        if not self.enabled:
            return
        filename = str(event["filename"])
        if filename not in self.files:
            self.files.append(filename)
        stage = str(event["stage"])
        done = int(event["done"])
        raw_total = event.get("total")
        total = int(raw_total) if raw_total is not None else None
        width = 24
        ratio = done / total if total else 0.0
        filled = round(width * min(max(ratio, 0.0), 1.0))
        bar = "#" * filled + "-" * (width - filled)
        status = "cached" if bool(event.get("cached")) else stage
        sys.stderr.write("\r\033[2K")
        sys.stderr.write(
            f"[{len(self.files):>2}] {status:<8} {filename:<43} [{bar}] "
            f"{_amount(done, total, stage)} "
            f"elapsed={_duration(time.monotonic() - self.started)}"
        )
        self.line_open = True
        if total is not None and done >= total:
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
        "--raw-dir",
        type=Path,
        default=Path("data/raw/prp_trec_dl"),
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed/prp_trec_dl_top10"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/data/prp_trec_dl_summary.json"),
    )
    parser.add_argument(
        "--candidate-snapshot",
        type=Path,
        default=Path("data/raw/prp_trec_dl/pyserini_bm25_top10.jsonl.gz"),
    )
    parser.add_argument(
        "--candidate-snapshot-sha256",
        default=None,
        help=(
            "Required immutable snapshot digest for non-default snapshots. "
            "The default Top10 snapshot keeps its repository-pinned digest."
        ),
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()
    default_snapshot = Path(
        "data/raw/prp_trec_dl/pyserini_bm25_top10.jsonl.gz"
    )
    expected_snapshot_sha256 = args.candidate_snapshot_sha256
    if expected_snapshot_sha256 is None:
        if args.candidate_snapshot != default_snapshot:
            raise ValueError(
                "--candidate-snapshot-sha256 is required for a non-default snapshot"
            )
        expected_snapshot_sha256 = EXPECTED_SNAPSHOT_SHA256
    progress = _Progress(enabled=args.progress)
    manifest = prepare_prp_trec_dl(
        args.raw_dir,
        args.processed_dir,
        top_k=args.top_k,
        candidate_snapshot=args.candidate_snapshot,
        candidate_snapshot_sha256=expected_snapshot_sha256,
        progress_callback=progress,
    )
    progress.finish()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_suffix(args.report.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.report)
    print(
        json.dumps(
            {
                "stage": manifest["stage"],
                "queries": manifest["audit"]["queries"],
                "candidates": manifest["audit"]["candidates"],
                "report": str(args.report),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
