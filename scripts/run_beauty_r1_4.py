"""Run the locked R1.4 Beauty confirmation with resumable training."""

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
    ValidationScoreBundle,
    YelpSASRecRunConfig,
    calibrated_scores,
    confidence_aware_scores,
    evaluate_yelp_test_checkpoint,
    export_locked_test_scores,
    load_validation_scores,
    run_yelp_sasrec,
    save_validation_scores,
    semantic_control,
    validation_metrics,
)

DEFAULT_SEEDS = (42, 2024, 3407)
TRAIN_VARIANTS = ("sasrec", "llm_init", "shuffled_init")
CONTROL_SEED = 20240725
BASE_SEMANTIC_WEIGHT = 0.25
GATED_RESIDUAL_WEIGHT = 0.1
FINAL_EVALUATION_NEGATIVES = 1000


def _duration(seconds: float) -> str:
    total = max(0, round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:d}h{minutes:02d}m" if hours else f"{minutes:02d}m{seconds:02d}s"


class _TrainingProgress:
    def __init__(self, total: int) -> None:
        self.total = total
        self.index = 0
        self.label = ""
        self.max_epochs = 0
        self.patience = 0
        self.started = 0.0

    @staticmethod
    def _clear() -> None:
        sys.stderr.write("\r\033[2K")
        sys.stderr.flush()

    def start(
        self,
        index: int,
        label: str,
        config: YelpSASRecRunConfig,
    ) -> None:
        self.index = index
        self.label = label
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
        loss_text = "" if loss is None else f" loss={loss:.4f}"
        self._clear()
        sys.stderr.write(
            f"[{self.index}/{self.total}] {self.label:<22} [{bar}] "
            f"epoch={epoch:>3}/{self.max_epochs} best={best:.6f} "
            f"stale={stale:>2}/{self.patience}{loss_text} "
            f"elapsed={_duration(time.monotonic() - self.started)}"
        )
        sys.stderr.flush()

    def finish(self, summary: dict[str, Any], *, cached: bool = False) -> None:
        self._clear()
        status = "cached" if cached else "done"
        print(
            f"[{self.index}/{self.total}] {status:<6} {self.label:<22} "
            f"best_epoch={summary['best_epoch']}",
            file=sys.stderr,
            flush=True,
        )

    def abort(self) -> None:
        self._clear()


class _EvaluationProgress:
    def __init__(self, total: int) -> None:
        self.total = total
        self.index = 0
        self.label = ""

    def start(self, label: str) -> None:
        self.index += 1
        self.label = label
        self.update(0, 1)

    def update(self, done: int, total: int) -> None:
        width = 24
        filled = round(width * done / max(total, 1))
        bar = "#" * filled + "-" * (width - filled)
        sys.stderr.write(
            f"\r\033[2K[test {self.index}/{self.total}] {self.label:<25} "
            f"[{bar}] users={done:>5}/{total}"
        )
        sys.stderr.flush()

    def finish(self, *, cached: bool = False) -> None:
        print(" cached" if cached else " done", file=sys.stderr, flush=True)


def _run_dir(root: Path, variant: str, seed: int) -> Path:
    return root / f"{variant}_seed{seed}"


def _variant_config(
    base: YelpSASRecRunConfig,
    *,
    root: Path,
    variant: str,
    seed: int,
    shuffled_path: Path,
) -> YelpSASRecRunConfig:
    if variant not in TRAIN_VARIANTS:
        raise ValueError(f"unknown training variant: {variant}")
    return replace(
        base,
        model="sasrec" if variant == "sasrec" else "llm_init",
        semantic_path=(
            shuffled_path if variant == "shuffled_init" else base.semantic_path
        ),
        seed=seed,
        output_dir=_run_dir(root, variant, seed),
        test_after_selection=False,
    )


def _load_training_summary(
    path: Path,
    *,
    config: YelpSASRecRunConfig,
) -> dict[str, Any]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    expected_semantic = (
        sha256_file(config.semantic_path)
        if config.model == "llm_init" and config.semantic_path is not None
        else None
    )
    if (
        summary.get("model") != config.model
        or summary.get("seed") != config.seed
        or summary.get("semantic_sha256") != expected_semantic
        or summary.get("protocol", {}).get("evaluation_seed")
        != config.evaluation_seed
    ):
        raise ValueError(f"cached training result does not match: {path}")
    return summary


def _load_or_score(
    config: YelpSASRecRunConfig,
    *,
    checkpoint: Path,
    path: Path,
    negatives: int,
    progress: _EvaluationProgress,
    label: str,
) -> ValidationScoreBundle:
    progress.start(label)
    checkpoint_hash = sha256_file(checkpoint)
    if config.semantic_path is None:
        raise ValueError("semantic_path is required")
    semantic_hash = sha256_file(config.semantic_path)
    if path.is_file():
        bundle = load_validation_scores(path)
        if (
            bundle.seed == config.seed
            and bundle.evaluation_seed == config.evaluation_seed
            and bundle.checkpoint_sha256 == checkpoint_hash
            and bundle.semantic_sha256 == semantic_hash
            and bundle.collaborative.shape[1] == negatives + 1
        ):
            progress.finish(cached=True)
            return bundle
    bundle = export_locked_test_scores(
        config,
        num_negatives=negatives,
        checkpoint_path=checkpoint,
        progress_callback=progress.update,
    )
    save_validation_scores(path, bundle)
    progress.finish()
    return bundle


def _candidate_buckets(bundle: ValidationScoreBundle) -> np.ndarray:
    return bundle.item_frequency_table[bundle.candidates - 1]


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
    return output


def _metric(metrics: dict[str, Any], bucket: str) -> float:
    return float(metrics["item_frequency"][bucket]["NDCG@10"])


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Beauty R1.4 locked confidence-aware fusion",
        "",
        "All fusion settings were locked on Yelp validation before Beauty test access.",
        "",
        "| Method | H@10 mean ± std | NDCG@10 mean ± std |",
        "|---|---:|---:|",
    ]
    for method in report["aggregate_test"]:
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
    for method in report["aggregate_test"]:
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
                f"- {name}: {'pass' if value else 'fail'}"
                for name, value in report["acceptance"].items()
            ],
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/reproduction/beauty_sasrec.yaml"),
    )
    parser.add_argument("--run-root", type=Path, default=Path("runs/r1_4/beauty"))
    parser.add_argument(
        "--yelp-lock",
        type=Path,
        default=Path("reports/experiments/yelp_r1_4_gate_validation.json"),
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=Path("reports/experiments/beauty_r1_4.json"),
    )
    parser.add_argument(
        "--report-markdown",
        type=Path,
        default=Path("reports/experiments/beauty_r1_4.md"),
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
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Stop after validation-only training; do not access Beauty test.",
    )
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        raise ValueError("seeds must be unique and non-negative")
    if args.final_negatives <= 0:
        raise ValueError("final-negatives must be positive")

    lock = json.loads(args.yelp_lock.read_text(encoding="utf-8"))
    locked_weight = float(lock["selection"]["semantic_weight"])
    if (
        not lock["selection"]["feasible"]
        or locked_weight != GATED_RESIDUAL_WEIGHT
        or lock["beauty_transfer"]["test_accessed"]
    ):
        raise ValueError("Yelp gate lock is missing or inconsistent")

    base = YelpSASRecRunConfig.from_yaml(args.config)
    if base.semantic_path is None:
        raise ValueError("semantic_path is required")
    real_semantics = np.load(base.semantic_path, allow_pickle=False).astype(np.float32)
    shuffled_path = (
        base.processed_dir
        / "semantic_controls"
        / f"shuffled_seed{CONTROL_SEED}.npy"
    )
    shuffled_path.parent.mkdir(parents=True, exist_ok=True)
    if not shuffled_path.is_file():
        np.save(
            shuffled_path,
            semantic_control(
                real_semantics,
                kind="shuffled",
                seed=CONTROL_SEED,
            ),
            allow_pickle=False,
        )

    training_progress = (
        _TrainingProgress(len(args.seeds) * len(TRAIN_VARIANTS))
        if args.progress
        else None
    )
    index = 0
    for seed in args.seeds:
        for variant in TRAIN_VARIANTS:
            index += 1
            config = _variant_config(
                base,
                root=args.run_root,
                variant=variant,
                seed=seed,
                shuffled_path=shuffled_path,
            )
            summary_path = config.output_dir / "summary.json"
            if summary_path.is_file():
                summary = _load_training_summary(summary_path, config=config)
                if training_progress is not None:
                    training_progress.index = index
                    training_progress.label = f"{variant} seed={seed}"
                    training_progress.finish(summary, cached=True)
                continue
            if training_progress is not None:
                training_progress.start(index, f"{variant} seed={seed}", config)
            try:
                summary = run_yelp_sasrec(
                    config,
                    epoch_callback=(
                        training_progress.epoch
                        if training_progress is not None
                        else None
                    ),
                )
            except BaseException:
                if training_progress is not None:
                    training_progress.abort()
                raise
            if training_progress is not None:
                training_progress.finish(summary)

    if args.train_only:
        print(
            json.dumps(
                {
                    "stage": "training_complete",
                    "test_accessed": False,
                    "run_root": str(args.run_root),
                }
            ),
            flush=True,
        )
        return

    evaluation_progress = _EvaluationProgress(len(args.seeds) * 4)
    per_seed: dict[str, dict[str, Any]] = {}
    for seed in args.seeds:
        sasrec = _variant_config(
            base,
            root=args.run_root,
            variant="sasrec",
            seed=seed,
            shuffled_path=shuffled_path,
        )
        sasrec = replace(sasrec, evaluation_negatives=args.final_negatives)
        summary_path = sasrec.output_dir / "summary.json"
        sasrec_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        evaluation_progress.start(f"sasrec seed={seed}")
        if sasrec_summary.get("test") is None:
            sasrec_metrics = evaluate_yelp_test_checkpoint(
                sasrec,
                progress_callback=evaluation_progress.update,
            )
            evaluation_progress.finish()
        else:
            if (
                sasrec_summary.get("protocol", {}).get(
                    "final_test_evaluation_negatives"
                )
                != args.final_negatives
            ):
                raise ValueError(
                    "cached SASRec test used a different negative count: "
                    f"{summary_path}"
                )
            sasrec_metrics = sasrec_summary["test"]
            evaluation_progress.finish(cached=True)

        real_config = _variant_config(
            base,
            root=args.run_root,
            variant="llm_init",
            seed=seed,
            shuffled_path=shuffled_path,
        )
        real_checkpoint = real_config.output_dir / "best_model.pt"
        real_bundle = _load_or_score(
            real_config,
            checkpoint=real_checkpoint,
            path=real_config.output_dir
            / f"test_scores_sampled{args.final_negatives}.npz",
            negatives=args.final_negatives,
            progress=evaluation_progress,
            label=f"real branches seed={seed}",
        )
        shuffled_branch_bundle = _load_or_score(
            replace(real_config, semantic_path=shuffled_path),
            checkpoint=real_checkpoint,
            path=real_config.output_dir
            / f"shuffled_test_scores_sampled{args.final_negatives}.npz",
            negatives=args.final_negatives,
            progress=evaluation_progress,
            label=f"shuffled branch seed={seed}",
        )
        shuffled_init_config = _variant_config(
            base,
            root=args.run_root,
            variant="shuffled_init",
            seed=seed,
            shuffled_path=shuffled_path,
        )
        shuffled_init_bundle = _load_or_score(
            shuffled_init_config,
            checkpoint=shuffled_init_config.output_dir / "best_model.pt",
            path=shuffled_init_config.output_dir
            / f"test_scores_sampled{args.final_negatives}.npz",
            negatives=args.final_negatives,
            progress=evaluation_progress,
            label=f"shuffled init seed={seed}",
        )
        if not np.array_equal(real_bundle.candidates, shuffled_branch_bundle.candidates):
            raise ValueError("real and shuffled semantic candidates do not align")

        fixed = calibrated_scores(
            real_bundle.collaborative,
            real_bundle.semantic,
            method="zscore",
            semantic_weight=BASE_SEMANTIC_WEIGHT,
        )
        gate = confidence_aware_scores(
            real_bundle.collaborative,
            real_bundle.semantic,
            _candidate_buckets(real_bundle),
            semantic_weight=locked_weight,
            base_semantic_weight=BASE_SEMANTIC_WEIGHT,
        )
        shuffled_gate = confidence_aware_scores(
            shuffled_branch_bundle.collaborative,
            shuffled_branch_bundle.semantic,
            _candidate_buckets(shuffled_branch_bundle),
            semantic_weight=locked_weight,
            base_semantic_weight=BASE_SEMANTIC_WEIGHT,
        )
        per_seed[str(seed)] = {
            "sasrec": sasrec_metrics,
            "llm_init": validation_metrics(
                real_bundle,
                real_bundle.collaborative,
                top_k=base.top_k,
            ),
            "shuffled_init": validation_metrics(
                shuffled_init_bundle,
                shuffled_init_bundle.collaborative,
                top_k=base.top_k,
            ),
            "semantic_only": validation_metrics(
                real_bundle,
                real_bundle.semantic,
                top_k=base.top_k,
            ),
            "fixed_fusion": validation_metrics(
                real_bundle,
                fixed,
                top_k=base.top_k,
            ),
            "confidence_gate": validation_metrics(
                real_bundle,
                gate,
                top_k=base.top_k,
            ),
            "shuffled_gate": validation_metrics(
                shuffled_branch_bundle,
                shuffled_gate,
                top_k=base.top_k,
            ),
        }

    methods = tuple(next(iter(per_seed.values())))
    aggregate = {method: _aggregate(per_seed, method) for method in methods}
    strongest = {
        str(seed): max(
            ("sasrec", "llm_init", "shuffled_init"),
            key=lambda method: _metric(per_seed[str(seed)][method], "overall"),
        )
        for seed in args.seeds
    }
    report = {
        "experiment": "R1.4",
        "dataset": "beauty",
        "status": "complete",
        "seeds": args.seeds,
        "protocol": {
            "checkpoint_selection": (
                f"Beauty validation NDCG@{base.top_k} with "
                f"{base.evaluation_negatives} negatives"
            ),
            "fusion_selection_dataset": "Yelp validation only",
            "base_semantic_weight": BASE_SEMANTIC_WEIGHT,
            "gated_residual_weight": locked_weight,
            "beauty_tuning": "none",
            "final_evaluation_negatives": args.final_negatives,
            "test_usage": "once after checkpoints and fusion settings were locked",
            "evaluation_scope": "sampled negatives; not full-catalog",
        },
        "data_audit": json.loads(base.report_path.read_text(encoding="utf-8"))[
            "statistics"
        ],
        "strongest_collaborative_per_seed": strongest,
        "per_seed_test": per_seed,
        "aggregate_test": aggregate,
        "acceptance": {
            "gate_beats_strongest_collaborative_all_seeds": all(
                _metric(per_seed[str(seed)]["confidence_gate"], "overall")
                > _metric(per_seed[str(seed)][strongest[str(seed)]], "overall")
                for seed in args.seeds
            ),
            "gate_tail_beats_fixed_fusion_all_seeds": all(
                _metric(per_seed[str(seed)]["confidence_gate"], "tail")
                > _metric(per_seed[str(seed)]["fixed_fusion"], "tail")
                for seed in args.seeds
            ),
            "real_gate_tail_beats_shuffled_gate_all_seeds": all(
                _metric(per_seed[str(seed)]["confidence_gate"], "tail")
                > _metric(per_seed[str(seed)]["shuffled_gate"], "tail")
                for seed in args.seeds
            ),
            "semantic_only_weaker_than_strongest_collaborative_all_seeds": all(
                _metric(per_seed[str(seed)]["semantic_only"], "overall")
                < _metric(per_seed[str(seed)][strongest[str(seed)]], "overall")
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
