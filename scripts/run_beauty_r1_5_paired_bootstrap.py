"""Run paired user-level bootstrap inference for Beauty full-catalog ranks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from caged_ltr.data.sequential import load_yelp_author_sequences
from caged_ltr.evaluation import paired_bootstrap_mean

COMPARISONS = (
    ("gate_minus_llm_init", "confidence_gate_real", "llm_init"),
    ("gate_minus_fixed_fusion", "confidence_gate_real", "fusion_real"),
    (
        "real_gate_minus_shuffled_gate",
        "confidence_gate_real",
        "confidence_gate_shuffled",
    ),
    ("gate_minus_semantic_only", "confidence_gate_real", "semantic_only_real"),
)
METRICS = ("Hit@10", "NDCG@10")
BUCKETS = ("overall", "head", "torso", "tail", "cold_start")


def _contributions(ranks: np.ndarray, *, top_k: int) -> np.ndarray:
    values = np.asarray(ranks)
    if values.ndim != 1 or (values < 0).any():
        raise ValueError("ranks must be a non-negative vector")
    hits = (values < top_k).astype(np.float64)
    ndcg = np.where(hits > 0.0, 1.0 / np.log2(values + 2.0), 0.0)
    return np.column_stack((hits, ndcg))


def _columns() -> list[str]:
    return [
        f"{comparison}.{metric}"
        for comparison, _, _ in COMPARISONS
        for metric in METRICS
    ]


def _differences(path: Path, *, top_k: int) -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        methods = {
            method
            for _, treatment, control in COMPARISONS
            for method in (treatment, control)
        }
        contributions = {
            method: _contributions(payload[method], top_k=top_k)
            for method in methods
        }
    columns: list[np.ndarray] = []
    for _, treatment, control in COMPARISONS:
        delta = contributions[treatment] - contributions[control]
        columns.extend((delta[:, 0], delta[:, 1]))
    return np.column_stack(columns)


def _point_estimates(
    differences: np.ndarray,
    target_buckets: np.ndarray,
) -> dict[str, dict[str, float]]:
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
                _columns(),
                differences[mask].mean(axis=0),
                strict=True,
            )
        }
    return output


def _reshape(result: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for column, name in enumerate(_columns()):
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


def _progress(index: int, bucket: str, done: int, total: int) -> None:
    width = 24
    filled = round(width * done / max(total, 1))
    bar = "#" * filled + "-" * (width - filled)
    sys.stderr.write(
        f"\r\033[2K[{index}/{len(BUCKETS)}] {bucket:<10} "
        f"[{bar}] bootstrap={done:>5}/{total}"
    )
    sys.stderr.flush()


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Beauty R1.5 full-catalog paired bootstrap",
        "",
        "Post-hoc uncertainty audit. No method or weight was selected from "
        "full-catalog results.",
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
    parser.add_argument(
        "--source-report",
        type=Path,
        default=Path("reports/experiments/beauty_r1_5_full_catalog.json"),
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed/beauty_llmesr_author"),
    )
    parser.add_argument(
        "--data-report",
        type=Path,
        default=Path("reports/data/beauty_llmesr_author_summary.json"),
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=Path(
            "reports/experiments/beauty_r1_5_full_catalog_paired_bootstrap.json"
        ),
    )
    parser.add_argument(
        "--report-markdown",
        type=Path,
        default=Path(
            "reports/experiments/beauty_r1_5_full_catalog_paired_bootstrap.md"
        ),
    )
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20240727)
    parser.add_argument("--bootstrap-batch-size", type=int, default=25)
    args = parser.parse_args()

    source = json.loads(args.source_report.read_text(encoding="utf-8"))
    if source["status"] != "complete":
        raise ValueError("full-catalog source report is incomplete")
    data = load_yelp_author_sequences(
        args.processed_dir,
        report_path=args.data_report,
    )
    eligible = np.flatnonzero(data.test_targets > 0)
    selected_users = int(source["protocol"]["evaluated_users_per_seed"])
    user_offsets = eligible[:selected_users]
    targets = data.test_targets[user_offsets]
    target_buckets = np.asarray(data.item_frequency_buckets)[targets - 1]

    aggregate: np.ndarray | None = None
    per_seed: dict[str, Any] = {}
    inputs_by_seed = {
        int(item["seed"]): item for item in source["inputs"]
    }
    for seed in source["seeds"]:
        ranks_path = Path(inputs_by_seed[seed]["ranks"])
        differences = _differences(ranks_path, top_k=10)
        if differences.shape[0] != selected_users:
            raise ValueError(f"rank count mismatch for seed {seed}")
        if aggregate is None:
            aggregate = np.zeros_like(differences)
        aggregate += differences
        per_seed[str(seed)] = _point_estimates(differences, target_buckets)
    if aggregate is None:
        raise RuntimeError("no rank inputs were loaded")
    aggregate /= len(source["seeds"])

    bootstrap: dict[str, Any] = {}
    for index, bucket in enumerate(BUCKETS, start=1):
        mask = (
            np.ones(target_buckets.shape, dtype=bool)
            if bucket == "overall"
            else target_buckets == bucket
        )
        result = paired_bootstrap_mean(
            aggregate[mask],
            iterations=args.iterations,
            seed=args.bootstrap_seed + index,
            batch_size=args.bootstrap_batch_size,
            progress_callback=lambda done, total, i=index, b=bucket: _progress(
                i,
                b,
                done,
                total,
            ),
        )
        print(file=sys.stderr, flush=True)
        bootstrap[bucket] = _reshape(result)

    def interval(bucket: str, comparison: str) -> tuple[float, float]:
        values = bootstrap[bucket][comparison]["NDCG@10"]["ci95_percentile"]
        return float(values["lower"]), float(values["upper"])

    overall_fixed = interval("overall", "gate_minus_fixed_fusion")
    tail_fixed = interval("tail", "gate_minus_fixed_fusion")
    tail_shuffled = interval("tail", "real_gate_minus_shuffled_gate")
    overall_semantic = interval("overall", "gate_minus_semantic_only")
    report = {
        "experiment": "R1.5-full-catalog-paired-bootstrap",
        "dataset": "beauty",
        "status": "complete",
        "analysis_type": "post-hoc uncertainty audit; no model selection",
        "protocol": {
            "paired_unit": "test user over the identical complete catalog",
            "seed_aggregation": "average each user's paired difference across seeds",
            "bootstrap": "nonparametric row resampling with replacement",
            "iterations": args.iterations,
            "bootstrap_seed": args.bootstrap_seed,
            "confidence_interval": "two-sided percentile 95%",
            "selected_users": selected_users,
            "catalog_items": source["protocol"]["catalog_items"],
        },
        "inputs": [
            {
                "seed": seed,
                "ranks": inputs_by_seed[seed]["ranks"],
            }
            for seed in source["seeds"]
        ],
        "per_seed_point_estimates": per_seed,
        "seed_averaged_paired_bootstrap": bootstrap,
        "acceptance": {
            "overall_gate_beats_fixed_ci_excludes_zero": overall_fixed[0] > 0.0,
            "tail_gate_beats_fixed_ci_excludes_zero": tail_fixed[0] > 0.0,
            "tail_real_gate_beats_shuffled_ci_excludes_zero": (
                tail_shuffled[0] > 0.0
            ),
            "overall_gate_beats_semantic_only_ci_excludes_zero": (
                overall_semantic[0] > 0.0
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
