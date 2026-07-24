"""Run and aggregate the frozen R1.1 Yelp multi-seed experiment matrix."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from caged_ltr.sequential import YelpSASRecRunConfig, run_yelp_sasrec

SEEDS = (42, 2024, 3407)
MODELS = ("sasrec", "llm_init", "late_fusion")
FIXED_SEMANTIC_WEIGHT = 2.0
PAPER_REFERENCE = {"Hit@10": 0.5940, "NDCG@10": 0.3597}


def _duration(seconds: float) -> str:
    total = max(0, round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m"
    return f"{minutes:02d}m{secs:02d}s"


class _TerminalProgress:
    def __init__(self, total_runs: int, *, width: int = 24) -> None:
        self.total_runs = total_runs
        self.width = width
        self.run_index = 0
        self.model = ""
        self.seed = 0
        self.max_epochs = 0
        self.patience = 0
        self.started = 0.0

    @staticmethod
    def _clear() -> None:
        sys.stderr.write("\r\033[2K")
        sys.stderr.flush()

    def skip(self, index: int, model: str, seed: int, summary: dict[str, Any]) -> None:
        overall = summary["test"]["item_frequency"]["overall"]
        self._clear()
        print(
            f"[{index}/{self.total_runs}] cached {model:<11} seed={seed:<4} "
            f"test NDCG@10={overall['NDCG@10']:.6f}",
            file=sys.stderr,
            flush=True,
        )

    def start(self, index: int, config: YelpSASRecRunConfig) -> None:
        self.run_index = index
        self.model = config.model
        self.seed = config.seed
        self.max_epochs = config.max_epochs
        self.patience = config.patience
        self.started = time.monotonic()
        self._render(epoch=0, best=0.0, stale=0, loss=None)

    def epoch(self, record: dict[str, Any]) -> None:
        top_k = next(
            key.removeprefix("best_NDCG@")
            for key in record
            if key.startswith("best_NDCG@")
        )
        self._render(
            epoch=int(record["epoch"]),
            best=float(record[f"best_NDCG@{top_k}"]),
            stale=int(record["stale_epochs"]),
            loss=float(record["train_bpr"]),
        )

    def _render(
        self,
        *,
        epoch: int,
        best: float,
        stale: int,
        loss: float | None,
    ) -> None:
        fraction = min(epoch / self.max_epochs, 1.0)
        filled = round(self.width * fraction)
        bar = "#" * filled + "-" * (self.width - filled)
        elapsed = time.monotonic() - self.started
        if epoch:
            average_epoch = elapsed / epoch
            remaining_patience = max(self.patience - stale, 0)
            eta = f" stop-ETA~{_duration(average_epoch * remaining_patience)}"
            loss_text = f" loss={loss:.4f}"
        else:
            eta = ""
            loss_text = ""
        line = (
            f"[{self.run_index}/{self.total_runs}] {self.model:<11} seed={self.seed:<4} "
            f"[{bar}] epoch {epoch:>3}/{self.max_epochs} best={best:.6f} "
            f"stale={stale:>2}/{self.patience}{loss_text} "
            f"elapsed={_duration(elapsed)}{eta}"
        )
        self._clear()
        sys.stderr.write(line)
        sys.stderr.flush()

    def finish(self, summary: dict[str, Any]) -> None:
        overall = summary["test"]["item_frequency"]["overall"]
        elapsed = time.monotonic() - self.started
        self._clear()
        print(
            f"[{self.run_index}/{self.total_runs}] done   {self.model:<11} "
            f"seed={self.seed:<4} best_epoch={summary['best_epoch']:<3} "
            f"test NDCG@10={overall['NDCG@10']:.6f} time={_duration(elapsed)}",
            file=sys.stderr,
            flush=True,
        )

    def abort(self) -> None:
        self._clear()


def _output_dir(root: Path, model: str, seed: int) -> Path:
    if model == "late_fusion" and seed == 42:
        return root / "late_fusion_weight_search" / "weight_2p0"
    return root / f"{model}_seed{seed}"


def _validate_existing(
    summary: dict[str, Any],
    *,
    config: YelpSASRecRunConfig,
    summary_path: Path,
) -> None:
    if summary.get("model") != config.model or summary.get("seed") != config.seed:
        raise ValueError(f"existing result does not match requested run: {summary_path}")
    if summary.get("test") is None:
        raise ValueError(f"existing result has not completed its one test: {summary_path}")
    protocol = summary.get("protocol", {})
    if protocol.get("evaluation_seed") != config.evaluation_seed:
        raise ValueError(f"evaluation seed mismatch in {summary_path}")
    resolved_path = summary_path.with_name("resolved_config.yaml")
    resolved = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
    if config.model == "late_fusion" and not np.isclose(
        float(resolved["semantic_weight"]), FIXED_SEMANTIC_WEIGHT
    ):
        raise ValueError(f"semantic weight mismatch in {resolved_path}")


def _mean_std(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)),
    }


def _aggregate_model(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "seeds": [int(summary["seed"]) for summary in summaries],
        "per_seed": {},
        "test": {},
    }
    for summary in summaries:
        overall = summary["test"]["item_frequency"]["overall"]
        result["per_seed"][str(summary["seed"])] = {
            "best_epoch": int(summary["best_epoch"]),
            "epochs_ran": int(summary["epochs_ran"]),
            "Hit@10": float(overall["Hit@10"]),
            "NDCG@10": float(overall["NDCG@10"]),
        }
    for family in ("item_frequency", "item_paper", "user_frequency", "user_paper"):
        result["test"][family] = {}
        buckets = summaries[0]["test"][family]
        for bucket in buckets:
            result["test"][family][bucket] = {
                metric: _mean_std(
                    [
                        float(summary["test"][family][bucket][metric])
                        for summary in summaries
                    ]
                )
                for metric in ("Hit@10", "NDCG@10")
            }
            result["test"][family][bucket]["count_per_seed"] = [
                int(summary["test"][family][bucket]["count"]) for summary in summaries
            ]
    return result


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Yelp R1.1 multi-seed results",
        "",
        "Fixed evaluation seed: `20240722`; late-fusion weight: `2.0`.",
        "",
        "| Model | H@10 mean ± std | NDCG@10 mean ± std |",
        "|---|---:|---:|",
    ]
    for model in MODELS:
        overall = report["models"][model]["test"]["item_frequency"]["overall"]
        hit = overall["Hit@10"]
        ndcg = overall["NDCG@10"]
        lines.append(
            f"| {model} | {hit['mean']:.6f} ± {hit['std']:.6f} | "
            f"{ndcg['mean']:.6f} ± {ndcg['std']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Item-frequency NDCG@10",
            "",
            "| Model | Head | Torso | Tail | Cold-start |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for model in MODELS:
        groups = report["models"][model]["test"]["item_frequency"]
        cells = [
            f"{groups[bucket]['NDCG@10']['mean']:.6f} ± "
            f"{groups[bucket]['NDCG@10']['std']:.6f}"
            for bucket in ("head", "torso", "tail", "cold_start")
        ]
        lines.append(f"| {model} | {' | '.join(cells)} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/reproduction/yelp_sasrec.yaml"),
    )
    parser.add_argument("--run-root", type=Path, default=Path("runs/r1_1"))
    parser.add_argument(
        "--semantic-only-summary",
        type=Path,
        default=Path("runs/yelp_semantic_only_seed42/summary.json"),
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=Path("reports/experiments/yelp_r1_1_multiseed.json"),
    )
    parser.add_argument(
        "--report-markdown",
        type=Path,
        default=Path("reports/experiments/yelp_r1_1_multiseed.md"),
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show a live terminal progress bar (use --no-progress for JSON epoch logs)",
    )
    args = parser.parse_args()

    base = YelpSASRecRunConfig.from_yaml(args.config)
    summaries: dict[str, list[dict[str, Any]]] = {model: [] for model in MODELS}
    total_runs = len(SEEDS) * len(MODELS)
    progress = _TerminalProgress(total_runs) if args.progress else None
    run_index = 0
    for seed in SEEDS:
        for model in MODELS:
            run_index += 1
            output_dir = _output_dir(args.run_root, model, seed)
            config = replace(
                base,
                model=model,
                seed=seed,
                output_dir=output_dir,
                semantic_weight=FIXED_SEMANTIC_WEIGHT,
                test_after_selection=True,
            )
            summary_path = output_dir / "summary.json"
            if summary_path.is_file():
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                _validate_existing(summary, config=config, summary_path=summary_path)
                if progress is not None:
                    progress.skip(run_index, model, seed, summary)
                else:
                    print(
                        json.dumps(
                            {
                                "stage": "skip_completed",
                                "model": model,
                                "seed": seed,
                                "output_dir": str(output_dir),
                            }
                        ),
                        flush=True,
                    )
            else:
                if progress is not None:
                    progress.start(run_index, config)
                else:
                    print(
                        json.dumps(
                            {
                                "stage": "train",
                                "model": model,
                                "seed": seed,
                                "output_dir": str(output_dir),
                            }
                        ),
                        flush=True,
                    )
                try:
                    summary = run_yelp_sasrec(
                        config,
                        epoch_callback=progress.epoch if progress is not None else None,
                    )
                except BaseException:
                    if progress is not None:
                        progress.abort()
                    raise
                if progress is not None:
                    progress.finish(summary)
            summaries[model].append(summary)

    semantic_only = json.loads(args.semantic_only_summary.read_text(encoding="utf-8"))
    report = {
        "experiment": "R1.1",
        "status": "complete",
        "seeds": list(SEEDS),
        "evaluation_seed": base.evaluation_seed,
        "late_fusion_weight": FIXED_SEMANTIC_WEIGHT,
        "weight_selected_on": "seed 42 validation NDCG@10 only",
        "test_protocol": "once per seed after validation-selected checkpoint",
        "paper_sasrec_reference": PAPER_REFERENCE,
        "models": {
            model: _aggregate_model(model_summaries)
            for model, model_summaries in summaries.items()
        },
        "semantic_only": {
            "deterministic_single_run": True,
            "test": semantic_only["test"],
        },
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.report_markdown.parent.mkdir(parents=True, exist_ok=True)
    _write_markdown(report, args.report_markdown)
    print(json.dumps({"stage": "complete", "report": str(args.report_json)}), flush=True)


if __name__ == "__main__":
    main()
