"""Run paired user-level bootstrap inference for the Beauty R1.4 test."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from caged_ltr.evaluation import paired_bootstrap_mean
from caged_ltr.sequential import (
    calibrated_scores,
    confidence_aware_scores,
)

DEFAULT_SEEDS = (42, 2024, 3407)
COMPARISONS = (
    ("gate_minus_llm_init", "confidence_gate", "llm_init"),
    ("gate_minus_fixed_fusion", "confidence_gate", "fixed_fusion"),
    ("real_gate_minus_shuffled_gate", "confidence_gate", "shuffled_gate"),
)
METRICS = ("Hit@10", "NDCG@10")
BUCKETS = ("overall", "head", "torso", "tail", "cold_start")


class _Progress:
    @staticmethod
    def _render(prefix: str, done: int, total: int) -> None:
        width = 24
        filled = round(width * done / max(total, 1))
        bar = "#" * filled + "-" * (width - filled)
        sys.stderr.write(
            f"\r\033[2K{prefix} [{bar}] {done:>5}/{total}"
        )
        sys.stderr.flush()

    def scores(self, index: int, total_seeds: int, done: int, total: int) -> None:
        self._render(f"[scores {index}/{total_seeds}]", done, total)

    def bootstrap(self, index: int, bucket: str, done: int, total: int) -> None:
        self._render(f"[bootstrap {index}/{len(BUCKETS)}] {bucket:<10}", done, total)

    @staticmethod
    def finish() -> None:
        print(file=sys.stderr, flush=True)


def _target_contributions(scores: np.ndarray, *, top_k: int) -> np.ndarray:
    values = np.asarray(scores)
    if values.ndim != 2 or values.shape[1] < top_k or not np.isfinite(values).all():
        raise ValueError("scores must be a finite candidate matrix")
    ranks = (values[:, 1:] > values[:, :1]).sum(axis=1)
    hits = (ranks < top_k).astype(np.float64)
    ndcg = np.where(hits > 0.0, 1.0 / np.log2(ranks + 2.0), 0.0)
    return np.column_stack((hits, ndcg))


def _load_seed_differences(
    *,
    run_root: Path,
    seed: int,
    negatives: int,
    top_k: int,
    base_weight: float,
    gate_weight: float,
    batch_size: int,
    progress: _Progress,
    index: int,
    total_seeds: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    real_path = (
        run_root
        / f"llm_init_seed{seed}"
        / f"test_scores_sampled{negatives}.npz"
    )
    shuffled_path = (
        run_root
        / f"llm_init_seed{seed}"
        / f"shuffled_test_scores_sampled{negatives}.npz"
    )
    with np.load(real_path, allow_pickle=False) as real:
        collaborative = real["collaborative"]
        semantic = real["semantic"]
        candidates = real["candidates"]
        target_buckets = real["target_item_frequency"]
        item_bucket_table = real["item_frequency_table"]
        user_offsets = real["user_offsets"]
        targets = real["targets"]
        checkpoint_hash = str(real["checkpoint_sha256"].item())
        real_semantic_hash = str(real["semantic_sha256"].item())
        evaluation_seed = int(real["evaluation_seed"].item())
        stored_seed = int(real["seed"].item())
    with np.load(shuffled_path, allow_pickle=False) as shuffled:
        shuffled_semantic = shuffled["semantic"]
        shuffled_checkpoint_hash = str(shuffled["checkpoint_sha256"].item())
        shuffled_semantic_hash = str(shuffled["semantic_sha256"].item())
        shuffled_evaluation_seed = int(shuffled["evaluation_seed"].item())
        shuffled_seed = int(shuffled["seed"].item())

    expected_shape = (targets.size, negatives + 1)
    matrices = (collaborative, semantic, shuffled_semantic, candidates)
    if any(matrix.shape != expected_shape for matrix in matrices):
        raise ValueError(f"score cache shape mismatch for seed {seed}")
    if (
        stored_seed != seed
        or shuffled_seed != seed
        or evaluation_seed != shuffled_evaluation_seed
        or checkpoint_hash != shuffled_checkpoint_hash
    ):
        raise ValueError(f"real and shuffled cache metadata mismatch for seed {seed}")

    method_contributions = {
        method: np.empty((targets.size, len(METRICS)), dtype=np.float64)
        for _, method, _ in COMPARISONS
    }
    method_contributions.update(
        {
            baseline: np.empty((targets.size, len(METRICS)), dtype=np.float64)
            for _, _, baseline in COMPARISONS
        }
    )
    for start in range(0, targets.size, batch_size):
        stop = min(start + batch_size, targets.size)
        collaborative_batch = collaborative[start:stop]
        semantic_batch = semantic[start:stop]
        shuffled_semantic_batch = shuffled_semantic[start:stop]
        candidate_buckets = item_bucket_table[candidates[start:stop] - 1]
        fixed_scores = calibrated_scores(
            collaborative_batch,
            semantic_batch,
            method="zscore",
            semantic_weight=base_weight,
        )
        gate_scores = confidence_aware_scores(
            collaborative_batch,
            semantic_batch,
            candidate_buckets,
            semantic_weight=gate_weight,
            base_semantic_weight=base_weight,
        )
        shuffled_gate_scores = confidence_aware_scores(
            collaborative_batch,
            shuffled_semantic_batch,
            candidate_buckets,
            semantic_weight=gate_weight,
            base_semantic_weight=base_weight,
        )
        method_contributions["llm_init"][start:stop] = _target_contributions(
            collaborative_batch,
            top_k=top_k,
        )
        method_contributions["fixed_fusion"][start:stop] = _target_contributions(
            fixed_scores,
            top_k=top_k,
        )
        method_contributions["confidence_gate"][start:stop] = _target_contributions(
            gate_scores,
            top_k=top_k,
        )
        method_contributions["shuffled_gate"][start:stop] = _target_contributions(
            shuffled_gate_scores,
            top_k=top_k,
        )
        progress.scores(index, total_seeds, stop, targets.size)
    progress.finish()

    columns: list[np.ndarray] = []
    for _, treatment, control in COMPARISONS:
        difference = method_contributions[treatment] - method_contributions[control]
        columns.extend((difference[:, 0], difference[:, 1]))
    return (
        np.column_stack(columns),
        target_buckets,
        {
            "seed": seed,
            "evaluation_seed": evaluation_seed,
            "rows": int(targets.size),
            "checkpoint_sha256": checkpoint_hash,
            "real_semantic_sha256": real_semantic_hash,
            "shuffled_semantic_sha256": shuffled_semantic_hash,
            "user_offsets": user_offsets,
            "targets": targets,
        },
    )


def _column_names() -> list[str]:
    return [
        f"{comparison}.{metric}"
        for comparison, _, _ in COMPARISONS
        for metric in METRICS
    ]


def _point_estimates(
    differences: np.ndarray,
    target_buckets: np.ndarray,
) -> dict[str, dict[str, float]]:
    columns = _column_names()
    output: dict[str, dict[str, float]] = {}
    for bucket in BUCKETS:
        mask = (
            np.ones(target_buckets.shape, dtype=bool)
            if bucket == "overall"
            else target_buckets == bucket
        )
        output[bucket] = {
            name: float(value)
            for name, value in zip(
                columns,
                differences[mask].mean(axis=0),
                strict=True,
            )
        }
    return output


def _reshape_bootstrap(result: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for column, name in enumerate(_column_names()):
        comparison, metric = name.split(".", maxsplit=1)
        output.setdefault(comparison, {})[metric] = {
            "mean": result["mean"][column],
            "bootstrap_standard_error": result["bootstrap_standard_error"][column],
            "ci95_percentile": {
                "lower": result["ci95_percentile"]["lower"][column],
                "upper": result["ci95_percentile"]["upper"][column],
            },
            "probability_positive": result["probability_positive"][column],
            "two_sided_p": result["two_sided_p"][column],
        }
    return output


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Beauty R1.4 paired bootstrap audit",
        "",
        "Post-hoc uncertainty audit of the already locked sampled-1000 test. "
        "No method or weight was selected from these intervals.",
        "",
        "| Bucket | Comparison | Mean ΔNDCG@10 | 95% CI | p (two-sided) |",
        "|---|---|---:|---:|---:|",
    ]
    for bucket in BUCKETS:
        for comparison, _, _ in COMPARISONS:
            values = report["seed_averaged_paired_bootstrap"][bucket][comparison][
                "NDCG@10"
            ]
            interval = values["ci95_percentile"]
            lines.append(
                f"| {bucket} | {comparison} | {values['mean']:+.6f} | "
                f"[{interval['lower']:+.6f}, {interval['upper']:+.6f}] | "
                f"{values['two_sided_p']:.6f} |"
            )
    lines.extend(
        [
            "",
            "## Acceptance",
            "",
            *[
                f"- {name}: {'pass' if passed else 'fail'}"
                for name, passed in report["acceptance"].items()
            ],
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path("runs/r1_4/beauty"))
    parser.add_argument(
        "--source-report",
        type=Path,
        default=Path("reports/experiments/beauty_r1_4.json"),
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=Path("reports/experiments/beauty_r1_4_paired_bootstrap.json"),
    )
    parser.add_argument(
        "--report-markdown",
        type=Path,
        default=Path("reports/experiments/beauty_r1_4_paired_bootstrap.md"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20240727)
    parser.add_argument("--score-batch-size", type=int, default=512)
    parser.add_argument("--bootstrap-batch-size", type=int, default=25)
    args = parser.parse_args()

    source = json.loads(args.source_report.read_text(encoding="utf-8"))
    if source["status"] != "complete" or args.seeds != source["seeds"]:
        raise ValueError("source report and requested seeds do not match")
    protocol = source["protocol"]
    negatives = int(protocol["final_evaluation_negatives"])
    base_weight = float(protocol["base_semantic_weight"])
    gate_weight = float(protocol["gated_residual_weight"])
    top_k = 10

    progress = _Progress()
    aggregate: np.ndarray | None = None
    reference_buckets: np.ndarray | None = None
    reference_users: np.ndarray | None = None
    reference_targets: np.ndarray | None = None
    per_seed: dict[str, Any] = {}
    inputs: list[dict[str, Any]] = []
    for index, seed in enumerate(args.seeds, start=1):
        differences, target_buckets, metadata = _load_seed_differences(
            run_root=args.run_root,
            seed=seed,
            negatives=negatives,
            top_k=top_k,
            base_weight=base_weight,
            gate_weight=gate_weight,
            batch_size=args.score_batch_size,
            progress=progress,
            index=index,
            total_seeds=len(args.seeds),
        )
        users = metadata.pop("user_offsets")
        targets = metadata.pop("targets")
        if reference_buckets is None:
            reference_buckets = target_buckets.copy()
            reference_users = users.copy()
            reference_targets = targets.copy()
            aggregate = np.zeros_like(differences)
        elif (
            not np.array_equal(target_buckets, reference_buckets)
            or not np.array_equal(users, reference_users)
            or not np.array_equal(targets, reference_targets)
        ):
            raise ValueError("paired users, targets, or buckets differ across seeds")
        if aggregate is None:
            raise RuntimeError("aggregate was not initialized")
        aggregate += differences
        per_seed[str(seed)] = _point_estimates(differences, target_buckets)
        inputs.append(metadata)

    if aggregate is None or reference_buckets is None:
        raise RuntimeError("no seed differences were loaded")
    aggregate /= len(args.seeds)
    bootstrap: dict[str, Any] = {}
    for index, bucket in enumerate(BUCKETS, start=1):
        mask = (
            np.ones(reference_buckets.shape, dtype=bool)
            if bucket == "overall"
            else reference_buckets == bucket
        )
        result = paired_bootstrap_mean(
            aggregate[mask],
            iterations=args.iterations,
            seed=args.bootstrap_seed + index,
            batch_size=args.bootstrap_batch_size,
            progress_callback=lambda done, total, i=index, b=bucket: (
                progress.bootstrap(i, b, done, total)
            ),
        )
        progress.finish()
        bootstrap[bucket] = _reshape_bootstrap(result)

    def lower(bucket: str, comparison: str) -> float:
        return float(
            bootstrap[bucket][comparison]["NDCG@10"]["ci95_percentile"]["lower"]
        )

    report = {
        "experiment": "R1.4a-paired-bootstrap",
        "dataset": "beauty",
        "status": "complete",
        "analysis_type": "post-hoc uncertainty audit; no model selection",
        "protocol": {
            "paired_unit": "test user with identical target and sampled candidates",
            "seed_aggregation": "average each user's paired difference across seeds",
            "bootstrap": "nonparametric row resampling with replacement",
            "iterations": args.iterations,
            "bootstrap_seed": args.bootstrap_seed,
            "confidence_interval": "two-sided percentile 95%",
            "final_evaluation_negatives": negatives,
            "base_semantic_weight": base_weight,
            "gated_residual_weight": gate_weight,
        },
        "inputs": inputs,
        "per_seed_point_estimates": per_seed,
        "seed_averaged_paired_bootstrap": bootstrap,
        "acceptance": {
            "overall_gate_beats_llm_ci_excludes_zero": (
                lower("overall", "gate_minus_llm_init") > 0.0
            ),
            "overall_gate_beats_fixed_ci_excludes_zero": (
                lower("overall", "gate_minus_fixed_fusion") > 0.0
            ),
            "tail_gate_beats_fixed_ci_excludes_zero": (
                lower("tail", "gate_minus_fixed_fusion") > 0.0
            ),
            "tail_real_gate_beats_shuffled_ci_excludes_zero": (
                lower("tail", "real_gate_minus_shuffled_gate") > 0.0
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
