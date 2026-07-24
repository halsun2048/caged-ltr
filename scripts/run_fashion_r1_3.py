"""Run the locked R1.3 Fashion external-confirmation experiment."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from caged_ltr.reproducibility import sha256_file
from caged_ltr.sequential import (
    YelpSASRecRunConfig,
    calibrated_scores,
    evaluate_yelp_test_checkpoint,
    export_locked_test_scores,
    load_validation_scores,
    run_yelp_sasrec,
    save_validation_scores,
    validation_metrics,
)

DEFAULT_SEEDS = (42, 2024, 3407)
TRAIN_MODELS = ("sasrec", "llm_init")
FIXED_METHOD = "zscore"
FIXED_SEMANTIC_WEIGHT = 0.25
FINAL_EVALUATION_NEGATIVES = 1000


def _duration(seconds: float) -> str:
    total = max(0, round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:d}h{minutes:02d}m" if hours else f"{minutes:02d}m{secs:02d}s"


class _Progress:
    def __init__(self, total: int) -> None:
        self.total = total
        self.index = 0
        self.model = ""
        self.seed = 0
        self.max_epochs = 0
        self.patience = 0
        self.started = 0.0

    @staticmethod
    def _clear() -> None:
        sys.stderr.write("\r\033[2K")
        sys.stderr.flush()

    def cached(self, index: int, model: str, seed: int, summary: dict[str, Any]) -> None:
        self._clear()
        print(
            f"[{index}/{self.total}] cached {model:<8} seed={seed:<4} "
            f"best_epoch={summary['best_epoch']}",
            file=sys.stderr,
            flush=True,
        )

    def start(self, index: int, config: YelpSASRecRunConfig) -> None:
        self.index = index
        self.model = config.model
        self.seed = config.seed
        self.max_epochs = config.max_epochs
        self.patience = config.patience
        self.started = time.monotonic()
        self._render(epoch=0, best=0.0, stale=0, loss=None)

    def epoch(self, record: dict[str, Any]) -> None:
        key = next(name for name in record if name.startswith("best_NDCG@"))
        self._render(
            epoch=int(record["epoch"]),
            best=float(record[key]),
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
        width = 24
        filled = round(width * min(epoch / self.max_epochs, 1.0))
        bar = "#" * filled + "-" * (width - filled)
        elapsed = time.monotonic() - self.started
        loss_text = "" if loss is None else f" loss={loss:.4f}"
        self._clear()
        sys.stderr.write(
            f"[{self.index}/{self.total}] {self.model:<8} seed={self.seed:<4} "
            f"[{bar}] epoch {epoch:>3}/{self.max_epochs} best={best:.6f} "
            f"stale={stale:>2}/{self.patience}{loss_text} "
            f"elapsed={_duration(elapsed)}"
        )
        sys.stderr.flush()

    def finish(self, summary: dict[str, Any]) -> None:
        self._clear()
        print(
            f"[{self.index}/{self.total}] done   {self.model:<8} seed={self.seed:<4} "
            f"best_epoch={summary['best_epoch']:<3} "
            f"time={_duration(time.monotonic() - self.started)}",
            file=sys.stderr,
            flush=True,
        )

    def abort(self) -> None:
        self._clear()


def _run_dir(root: Path, model: str, seed: int) -> Path:
    return root / f"{model}_seed{seed}"


def _load_training_summary(
    path: Path,
    *,
    model: str,
    seed: int,
    evaluation_seed: int,
) -> dict[str, Any]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary.get("model") != model or summary.get("seed") != seed:
        raise ValueError(f"cached training result does not match {model} seed {seed}: {path}")
    if summary.get("protocol", {}).get("evaluation_seed") != evaluation_seed:
        raise ValueError(f"cached evaluation seed mismatch: {path}")
    return summary


def _load_or_score_test(
    config: YelpSASRecRunConfig,
    *,
    cache_path: Path,
    num_negatives: int,
):
    checkpoint = config.output_dir / "best_model.pt"
    expected_checkpoint = sha256_file(checkpoint)
    if config.semantic_path is None:
        raise ValueError("semantic_path is required")
    expected_semantic = sha256_file(config.semantic_path)
    if cache_path.is_file():
        bundle = load_validation_scores(cache_path)
        if (
            bundle.seed == config.seed
            and bundle.evaluation_seed == config.evaluation_seed
            and bundle.checkpoint_sha256 == expected_checkpoint
            and bundle.semantic_sha256 == expected_semantic
            and bundle.collaborative.shape[1] == num_negatives + 1
        ):
            print(
                f"[test] cached llm_init+fusion seed={config.seed} "
                f"candidates={num_negatives + 1}",
                file=sys.stderr,
                flush=True,
            )
            return bundle
    print(
        f"[test] scoring llm_init+fusion seed={config.seed} "
        f"candidates={num_negatives + 1}",
        file=sys.stderr,
        flush=True,
    )
    bundle = export_locked_test_scores(
        config,
        num_negatives=num_negatives,
        checkpoint_path=checkpoint,
    )
    save_validation_scores(cache_path, bundle)
    return bundle


def _mean_std(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
    }


def _aggregate(per_seed: dict[str, dict[str, Any]], method: str) -> dict[str, Any]:
    seeds = list(per_seed)
    output: dict[str, Any] = {}
    for family in ("item_frequency", "item_paper", "user_frequency", "user_paper"):
        output[family] = {}
        for bucket in per_seed[seeds[0]][method][family]:
            output[family][bucket] = {
                metric: _mean_std(
                    [
                        float(per_seed[seed][method][family][bucket][metric])
                        for seed in seeds
                    ]
                )
                for metric in ("Hit@10", "NDCG@10")
            }
            output[family][bucket]["count_per_seed"] = [
                int(per_seed[seed][method][family][bucket]["count"]) for seed in seeds
            ]
    return output


def _metric(metrics: dict[str, Any], bucket: str, metric: str = "NDCG@10") -> float:
    return float(metrics["item_frequency"][bucket][metric])


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Fashion R1.3 locked external confirmation",
        "",
        "No Fashion validation or test metric was used to select the fusion rule. "
        "The rule was locked on Yelp: per-query z-score with semantic weight 0.25.",
        "",
        f"Final evaluation uses one target plus "
        f"{report['protocol']['final_evaluation_negatives']} fixed unseen negatives.",
        "",
        "| Method | H@10 mean ± std | NDCG@10 mean ± std |",
        "|---|---:|---:|",
    ]
    for method in ("sasrec", "llm_init", "semantic_only", "calibrated_fusion"):
        overall = report["aggregate_test"][method]["item_frequency"]["overall"]
        hit = overall["Hit@10"]
        ndcg = overall["NDCG@10"]
        lines.append(
            f"| {method} | {hit['mean']:.6f} ± {hit['std']:.6f} | "
            f"{ndcg['mean']:.6f} ± {ndcg['std']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Item-frequency NDCG@10",
            "",
            "| Method | Head | Torso | Tail | Cold-start |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for method in ("sasrec", "llm_init", "semantic_only", "calibrated_fusion"):
        groups = report["aggregate_test"][method]["item_frequency"]
        cells = [
            f"{groups[bucket]['NDCG@10']['mean']:.6f} ± "
            f"{groups[bucket]['NDCG@10']['std']:.6f}"
            for bucket in ("head", "torso", "tail", "cold_start")
        ]
        lines.append(f"| {method} | {' | '.join(cells)} |")
    lines.extend(
        [
            "",
            "## Acceptance",
            "",
            *[
                f"- {name}: {'pass' if passed else 'fail'}"
                for name, passed in report["acceptance"].items()
            ],
            "",
            "## Data audit",
            "",
            f"- Author bundle: {report['data_audit']['author_bundle_users']} users, "
            f"{report['data_audit']['author_bundle_items']} items.",
            f"- Paper table: {report['data_audit']['paper_table_users']} users, "
            f"{report['data_audit']['paper_table_items']} items.",
            "- The 45-user discrepancy is retained and explicitly reported.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/reproduction/fashion_sasrec.yaml"),
    )
    parser.add_argument("--run-root", type=Path, default=Path("runs/r1_3/fashion"))
    parser.add_argument(
        "--report-json",
        type=Path,
        default=Path("reports/experiments/fashion_r1_3.json"),
    )
    parser.add_argument(
        "--report-markdown",
        type=Path,
        default=Path("reports/experiments/fashion_r1_3.md"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--final-negatives",
        type=int,
        default=FINAL_EVALUATION_NEGATIVES,
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        raise ValueError("seeds must be unique and non-negative")
    if args.final_negatives <= 0:
        raise ValueError("final-negatives must be positive")

    base = YelpSASRecRunConfig.from_yaml(args.config)
    progress = _Progress(len(args.seeds) * len(TRAIN_MODELS)) if args.progress else None
    summaries: dict[tuple[str, int], dict[str, Any]] = {}
    index = 0
    for seed in args.seeds:
        for model in TRAIN_MODELS:
            index += 1
            output_dir = _run_dir(args.run_root, model, seed)
            config = replace(
                base,
                model=model,
                seed=seed,
                output_dir=output_dir,
                test_after_selection=False,
            )
            summary_path = output_dir / "summary.json"
            if summary_path.is_file():
                summary = _load_training_summary(
                    summary_path,
                    model=model,
                    seed=seed,
                    evaluation_seed=base.evaluation_seed,
                )
                if progress is not None:
                    progress.cached(index, model, seed, summary)
            else:
                if progress is not None:
                    progress.start(index, config)
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
            summaries[(model, seed)] = summary

    per_seed: dict[str, dict[str, Any]] = {}
    for seed in args.seeds:
        sasrec_dir = _run_dir(args.run_root, "sasrec", seed)
        sasrec_config = replace(
            base,
            model="sasrec",
            seed=seed,
            output_dir=sasrec_dir,
            evaluation_negatives=args.final_negatives,
            test_after_selection=False,
        )
        sasrec_summary_path = sasrec_dir / "summary.json"
        sasrec_summary = json.loads(sasrec_summary_path.read_text(encoding="utf-8"))
        if sasrec_summary.get("test") is None:
            print(
                f"[test] scoring sasrec seed={seed} candidates={args.final_negatives + 1}",
                file=sys.stderr,
                flush=True,
            )
            sasrec_metrics = evaluate_yelp_test_checkpoint(sasrec_config)
            sasrec_summary = json.loads(sasrec_summary_path.read_text(encoding="utf-8"))
            sasrec_summary["protocol"]["final_test_evaluation"] = (
                f"target plus {args.final_negatives} fixed unseen negatives"
            )
            sasrec_summary_path.write_text(
                json.dumps(sasrec_summary, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        else:
            sasrec_metrics = sasrec_summary["test"]
            print(f"[test] cached sasrec seed={seed}", file=sys.stderr, flush=True)

        llm_dir = _run_dir(args.run_root, "llm_init", seed)
        llm_config = replace(
            base,
            model="llm_init",
            seed=seed,
            output_dir=llm_dir,
            test_after_selection=False,
        )
        bundle = _load_or_score_test(
            llm_config,
            cache_path=llm_dir / f"test_scores_sampled{args.final_negatives}.npz",
            num_negatives=args.final_negatives,
        )
        llm_metrics = validation_metrics(bundle, bundle.collaborative, top_k=base.top_k)
        semantic_metrics = validation_metrics(bundle, bundle.semantic, top_k=base.top_k)
        fusion = calibrated_scores(
            bundle.collaborative,
            bundle.semantic,
            method=FIXED_METHOD,
            semantic_weight=FIXED_SEMANTIC_WEIGHT,
        )
        fusion_metrics = validation_metrics(bundle, fusion, top_k=base.top_k)
        per_seed[str(seed)] = {
            "sasrec": sasrec_metrics,
            "llm_init": llm_metrics,
            "semantic_only": semantic_metrics,
            "calibrated_fusion": fusion_metrics,
        }

    aggregate = {
        method: _aggregate(per_seed, method)
        for method in ("sasrec", "llm_init", "semantic_only", "calibrated_fusion")
    }
    tail_gains = [
        _metric(per_seed[str(seed)]["calibrated_fusion"], "tail")
        - _metric(per_seed[str(seed)]["llm_init"], "tail")
        for seed in args.seeds
    ]
    overall_gains = [
        _metric(per_seed[str(seed)]["calibrated_fusion"], "overall")
        - _metric(per_seed[str(seed)]["llm_init"], "overall")
        for seed in args.seeds
    ]
    data_manifest = json.loads(base.report_path.read_text(encoding="utf-8"))
    report = {
        "experiment": "R1.3",
        "dataset": "fashion",
        "status": "complete",
        "seeds": args.seeds,
        "protocol": {
            "checkpoint_selection": (
                f"validation NDCG@{base.top_k} with {base.evaluation_negatives} negatives"
            ),
            "fusion_selection_dataset": "Yelp validation only",
            "fusion_method": FIXED_METHOD,
            "fusion_semantic_weight": FIXED_SEMANTIC_WEIGHT,
            "fashion_tuning": "none",
            "final_evaluation_negatives": args.final_negatives,
            "test_usage": "once after all checkpoints and fusion settings were locked",
        },
        "data_audit": {
            "author_bundle_users": data_manifest["statistics"]["users"],
            "author_bundle_items": data_manifest["statistics"]["items"],
            "paper_table_users": data_manifest["paper_reference"]["users"],
            "paper_table_items": data_manifest["paper_reference"]["items"],
            "paper_reference_match": data_manifest["paper_reference_match"],
            "processed_fingerprint": data_manifest["processed_fingerprint"],
        },
        "per_seed_test": per_seed,
        "aggregate_test": aggregate,
        "gains_calibrated_minus_llm_init": {
            "overall_ndcg_per_seed": overall_gains,
            "tail_ndcg_per_seed": tail_gains,
            "overall_ndcg_mean": float(np.mean(overall_gains)),
            "tail_ndcg_mean": float(np.mean(tail_gains)),
        },
        "acceptance": {
            "overall_direction_positive_all_seeds": all(gain > 0 for gain in overall_gains),
            "tail_direction_positive_all_seeds": all(gain > 0 for gain in tail_gains),
            "tail_mean_absolute_gain_at_least_0p005": float(np.mean(tail_gains)) >= 0.005,
            "semantic_only_weaker_than_llm_init_all_seeds": all(
                _metric(per_seed[str(seed)]["semantic_only"], "overall")
                < _metric(per_seed[str(seed)]["llm_init"], "overall")
                for seed in args.seeds
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
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
