"""Run resumable LightGCN/RLMRec-Con controls on the public RLMRec Yelp split."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from caged_ltr.graph import RLMRecRunConfig, run_rlmrec

DEFAULT_VARIANTS = ("lightgcn", "semantic_only", "rlmrec_con", "shuffled_con")


def _duration(seconds: float) -> str:
    total = max(0, round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:d}h{minutes:02d}m" if hours else f"{minutes:02d}m{seconds:02d}s"


class _Progress:
    def __init__(self, total: int, *, enabled: bool) -> None:
        self.total = total
        self.enabled = enabled
        self.index = 0
        self.label = ""
        self.config: RLMRecRunConfig | None = None
        self.started = 0.0

    @staticmethod
    def _clear() -> None:
        sys.stderr.write("\r\033[2K")
        sys.stderr.flush()

    def start(self, index: int, label: str, config: RLMRecRunConfig) -> None:
        self.index = index
        self.label = label
        self.config = config
        self.started = time.monotonic()
        self._render_epoch(
            {
                "epoch": 0,
                "best_Recall@20": 0.0,
                "stale_evaluations": 0,
                "train_loss": 0.0,
            }
        )

    def _render_epoch(self, record: dict[str, Any]) -> None:
        if not self.enabled:
            return
        assert self.config is not None
        epoch = int(record["epoch"])
        width = 24
        filled = round(width * min(epoch / self.config.max_epochs, 1.0))
        bar = "#" * filled + "-" * (width - filled)
        self._clear()
        sys.stderr.write(
            f"[{self.index}/{self.total}] {self.label:<24} [{bar}] "
            f"epoch={epoch:>3}/{self.config.max_epochs} "
            f"best R@20={float(record['best_Recall@20']):.6f} "
            f"stale={int(record['stale_evaluations'])}/{self.config.patience} "
            f"loss={float(record['train_loss']):.4f} "
            f"elapsed={_duration(time.monotonic() - self.started)}"
        )
        sys.stderr.flush()

    def epoch(self, record: dict[str, Any]) -> None:
        if self.enabled:
            self._render_epoch(record)
        else:
            print(json.dumps(record), flush=True)

    def evaluation(self, stage: str, done: int, total: int) -> None:
        if not self.enabled:
            return
        width = 24
        filled = round(width * done / max(total, 1))
        bar = "#" * filled + "-" * (width - filled)
        unit = "batches" if stage.startswith("train") else "users"
        self._clear()
        sys.stderr.write(
            f"[{self.index}/{self.total}] {self.label:<24} [{bar}] "
            f"{stage} {unit}={done:>5}/{total} "
            f"elapsed={_duration(time.monotonic() - self.started)}"
        )
        sys.stderr.flush()

    def finish(self, summary: dict[str, Any], *, cached: bool = False) -> None:
        self._clear()
        status = "cached" if cached else "done"
        test = summary.get("test")
        metric = (
            float(test["overall"]["NDCG@20"])
            if test is not None
            else float(summary["validation"]["overall"]["NDCG@20"])
        )
        split = "test" if test is not None else "valid"
        print(
            f"[{self.index}/{self.total}] {status:<6} {self.label:<24} "
            f"best_epoch={summary['best_epoch']:<3} "
            f"{split} NDCG@20={metric:.6f} "
            f"time={_duration(time.monotonic() - self.started)}",
            file=sys.stderr,
            flush=True,
        )


def _mean_std(values: list[float]) -> dict[str, float]:
    import numpy as np

    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
    }


def _aggregate(summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    variants = sorted({summary["variant"] for summary in summaries.values()})
    aggregate: dict[str, Any] = {}
    for variant in variants:
        selected = [
            summary for summary in summaries.values() if summary["variant"] == variant
        ]
        aggregate[variant] = {}
        for split in ("validation", "test"):
            if any(summary[split] is None for summary in selected):
                aggregate[variant][split] = None
                continue
            aggregate[variant][split] = {
                bucket: {
                    metric: _mean_std(
                        [
                            float(summary[split][bucket][metric])
                            for summary in selected
                        ]
                    )
                    for metric in ("Recall@20", "NDCG@20")
                }
                for bucket in ("overall", "head", "torso", "tail")
            }
    return aggregate


def _cached_summary(path: Path, *, variant: str, seed: int) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary.get("variant") != variant or summary.get("seed") != seed:
        raise ValueError(f"cached result identity mismatch: {path}")
    if summary.get("stage") != "complete":
        return None
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=DEFAULT_VARIANTS,
        default=list(DEFAULT_VARIANTS),
    )
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed/rlmrec_yelp_author"),
    )
    parser.add_argument("--output-root", type=Path, default=Path("runs/rlmrec_r2"))
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/experiments/rlmrec_r2.json"),
    )
    args = parser.parse_args()

    base = RLMRecRunConfig(
        processed_dir=args.processed_dir,
        output_dir=args.output_root,
        variant="lightgcn",
    )
    if args.smoke:
        base = replace(
            base,
            output_dir=Path("runs/rlmrec_r2_smoke"),
            max_epochs=1,
            evaluation_interval=1,
            patience=1,
            batch_size=256,
            evaluation_batch_size=128,
            max_batches_per_epoch=2,
            max_eval_users=128,
        )
    total = len(args.seeds) * len(args.variants)
    progress = _Progress(total, enabled=args.progress)
    summaries: dict[str, dict[str, Any]] = {}
    index = 0
    for seed in args.seeds:
        for variant in args.variants:
            index += 1
            label = f"{variant} seed={seed}"
            output_dir = base.output_dir / f"{variant}_seed{seed}"
            config = replace(
                base,
                output_dir=output_dir,
                variant=variant,
                seed=seed,
            )
            progress.start(index, label, config)
            cached = _cached_summary(
                output_dir / "summary.json",
                variant=variant,
                seed=seed,
            )
            if cached is not None:
                summaries[f"{variant}_seed{seed}"] = cached
                progress.finish(cached, cached=True)
                continue
            summary = run_rlmrec(
                config,
                epoch_callback=progress.epoch,
                evaluation_callback=progress.evaluation,
            )
            summaries[f"{variant}_seed{seed}"] = summary
            progress.finish(summary)

    report = {
        "stage": "complete",
        "seeds": args.seeds,
        "variants": args.variants,
        "smoke": args.smoke,
        "runs": summaries,
        "aggregate": _aggregate(summaries),
        "interpretation_boundary": (
            "Public profile embeddings have no verifiable train-only cutoff; results are "
            "a structure reproduction and cannot establish leakage-free semantic gains."
        ),
    }
    report_path = (
        Path("reports/experiments/rlmrec_r2_smoke.json") if args.smoke else args.report
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"stage": "complete", "report": str(report_path)}))


if __name__ == "__main__":
    main()
