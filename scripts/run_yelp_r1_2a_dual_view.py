"""Run the R1.2a Yelp dual-view architecture and validation-only controls."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from caged_ltr.sequential import YelpSASRecRunConfig, run_yelp_sasrec

VARIANTS = (
    "dual_view_no_ca",
    "dual_view",
    "dual_view_unshared",
    "dual_view_capacity",
)


def _duration(seconds: float) -> str:
    total = max(0, round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return (
        f"{hours:d}h{minutes:02d}m"
        if hours
        else f"{minutes:02d}m{secs:02d}s"
    )


class _Progress:
    def __init__(self, total: int, *, enabled: bool) -> None:
        self.total = total
        self.enabled = enabled
        self.index = 0
        self.config: YelpSASRecRunConfig | None = None
        self.started = 0.0

    @staticmethod
    def _clear() -> None:
        sys.stderr.write("\r\033[2K")
        sys.stderr.flush()

    def start(self, index: int, config: YelpSASRecRunConfig) -> None:
        self.index = index
        self.config = config
        self.started = time.monotonic()
        if self.enabled:
            self._render(0, 0.0, 0, None)

    def epoch(self, record: dict[str, Any]) -> None:
        if not self.enabled:
            print(json.dumps(record), flush=True)
            return
        assert self.config is not None
        best_key = next(key for key in record if key.startswith("best_NDCG@"))
        self._render(
            int(record["epoch"]),
            float(record[best_key]),
            int(record["stale_epochs"]),
            float(record["train_bpr"]),
        )

    def _render(
        self,
        epoch: int,
        best: float,
        stale: int,
        loss: float | None,
    ) -> None:
        assert self.config is not None
        width = 24
        filled = round(width * min(epoch / self.config.max_epochs, 1.0))
        bar = "#" * filled + "-" * (width - filled)
        elapsed = time.monotonic() - self.started
        loss_text = "" if loss is None else f" loss={loss:.4f}"
        eta_text = ""
        if epoch:
            remaining = max(self.config.patience - stale, 0)
            eta_text = f" stop-ETA~{_duration(elapsed / epoch * remaining)}"
        self._clear()
        sys.stderr.write(
            f"[{self.index}/{self.total}] {self.config.model:<20} "
            f"[{bar}] epoch={epoch:>3}/{self.config.max_epochs} "
            f"best={best:.6f} stale={stale:>2}/{self.config.patience}"
            f"{loss_text} elapsed={_duration(elapsed)}{eta_text}"
        )
        sys.stderr.flush()

    def cached(self, index: int, config: YelpSASRecRunConfig, summary: dict[str, Any]) -> None:
        if self.enabled:
            self._clear()
        ndcg = summary["validation"]["item_frequency"]["overall"][
            f"NDCG@{config.top_k}"
        ]
        print(
            f"[{index}/{self.total}] cached {config.model:<20} "
            f"best_epoch={summary['best_epoch']:<3} valid NDCG@{config.top_k}={ndcg:.6f}",
            file=sys.stderr,
            flush=True,
        )

    def finish(self, summary: dict[str, Any]) -> None:
        assert self.config is not None
        if self.enabled:
            self._clear()
        ndcg = summary["validation"]["item_frequency"]["overall"][
            f"NDCG@{self.config.top_k}"
        ]
        print(
            f"[{self.index}/{self.total}] done   {self.config.model:<20} "
            f"best_epoch={summary['best_epoch']:<3} "
            f"valid NDCG@{self.config.top_k}={ndcg:.6f} "
            f"time={_duration(time.monotonic() - self.started)}",
            file=sys.stderr,
            flush=True,
        )


def _metric(summary: dict[str, Any], bucket: str, top_k: int) -> float:
    return float(
        summary["validation"]["item_frequency"][bucket][f"NDCG@{top_k}"]
    )


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Yelp R1.2a dual-view structural controls",
        "",
        "Seed 42 validation only; the test split was not evaluated.",
        "",
        "| Variant | Overall | Head | Torso | Tail | Cold-start | Trainable params |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        row = report["variants"][variant]
        groups = row["validation"]["item_frequency"]
        values = [
            f"{groups[bucket]['NDCG@10']:.6f}"
            for bucket in ("overall", "head", "torso", "tail", "cold_start")
        ]
        lines.append(
            f"| {variant} | {' | '.join(values)} | "
            f"{row['parameters']['trainable']} |"
        )
    lines.extend(
        [
            "",
            "Cross attention uses causal plus padding masks. This is a "
            "leakage-safe correction to the author implementation's padding-only "
            "cross-attention mask.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/reproduction/yelp_sasrec.yaml"),
    )
    parser.add_argument("--run-root", type=Path, default=Path("runs/r1_2a/yelp"))
    parser.add_argument(
        "--report-json",
        type=Path,
        default=Path("reports/experiments/yelp_r1_2a_dual_view_validation.json"),
    )
    parser.add_argument(
        "--report-markdown",
        type=Path,
        default=Path("reports/experiments/yelp_r1_2a_dual_view_validation.md"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-users", type=int)
    parser.add_argument("--max-eval-users", type=int)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show a live epoch progress bar",
    )
    args = parser.parse_args()

    base = YelpSASRecRunConfig.from_yaml(args.config)
    progress = _Progress(len(VARIANTS), enabled=args.progress)
    summaries: dict[str, dict[str, Any]] = {}
    for index, variant in enumerate(VARIANTS, start=1):
        config = replace(
            base,
            model=variant,
            seed=args.seed,
            output_dir=args.run_root / f"{variant}_seed{args.seed}",
            max_users=args.max_users if args.max_users is not None else base.max_users,
            max_eval_users=(
                args.max_eval_users
                if args.max_eval_users is not None
                else base.max_eval_users
            ),
            max_epochs=(
                args.max_epochs if args.max_epochs is not None else base.max_epochs
            ),
            test_after_selection=False,
        )
        summary_path = config.output_dir / "summary.json"
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if (
                summary.get("model") != variant
                or summary.get("seed") != args.seed
                or summary.get("test") is not None
            ):
                raise ValueError(f"incompatible cached result: {summary_path}")
            progress.cached(index, config, summary)
        else:
            progress.start(index, config)
            summary = run_yelp_sasrec(config, epoch_callback=progress.epoch)
            progress.finish(summary)
        summaries[variant] = summary

    top_k = base.top_k
    report = {
        "experiment": "R1.2a-dual-view-structural-controls",
        "dataset": "yelp",
        "status": "validation_locked",
        "seed": args.seed,
        "split": "validation",
        "test_accessed": False,
        "variants": summaries,
        "comparisons": {
            "cross_attention_minus_no_ca": {
                bucket: _metric(summaries["dual_view"], bucket, top_k)
                - _metric(summaries["dual_view_no_ca"], bucket, top_k)
                for bucket in ("overall", "head", "torso", "tail", "cold_start")
            },
            "shared_minus_unshared": {
                bucket: _metric(summaries["dual_view"], bucket, top_k)
                - _metric(summaries["dual_view_unshared"], bucket, top_k)
                for bucket in ("overall", "head", "torso", "tail", "cold_start")
            },
            "cross_attention_minus_capacity_control": {
                bucket: _metric(summaries["dual_view"], bucket, top_k)
                - _metric(summaries["dual_view_capacity"], bucket, top_k)
                for bucket in ("overall", "head", "torso", "tail", "cold_start")
            },
        },
        "acceptance": {
            "test_not_accessed": all(
                summary["test"] is None for summary in summaries.values()
            ),
            "cross_attention_improves_over_no_ca": (
                _metric(summaries["dual_view"], "overall", top_k)
                > _metric(summaries["dual_view_no_ca"], "overall", top_k)
            ),
            "cross_attention_improves_tail_over_no_ca": (
                _metric(summaries["dual_view"], "tail", top_k)
                > _metric(summaries["dual_view_no_ca"], "tail", top_k)
            ),
            "cross_attention_beats_capacity_control": (
                _metric(summaries["dual_view"], "overall", top_k)
                > _metric(summaries["dual_view_capacity"], "overall", top_k)
            ),
        },
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_markdown(report, args.report_markdown)
    print(
        json.dumps(
            {
                "stage": "complete",
                "report": str(args.report_json),
                "acceptance": report["acceptance"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
