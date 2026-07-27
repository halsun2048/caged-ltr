"""Run Yelp R1.8 uncertainty-gated fixed-budget candidate routing."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from caged_ltr.data.sequential import YelpSequenceData, load_yelp_author_sequences
from caged_ltr.reproducibility import sha256_file
from caged_ltr.sequential import (
    YelpSASRecRunConfig,
    evaluate_full_catalog_retrieval,
    semantic_control,
)

DEFAULT_SEEDS = (42, 2024, 3407)
INJECTION_RATE_CANDIDATES = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
CONTROL_SEED = 20240725
BUDGET = 500
SEMANTIC_QUOTA = 50
SELECTION_TOLERANCE = 0.005
REPLICATION_TOLERANCE = 0.01
TAIL_GAIN_TARGET = 0.03
COLD_GAIN_TARGET = 0.03


def _fixed_route(variant: str) -> str:
    return f"fixed_union_{variant}_s{SEMANTIC_QUOTA}_of{BUDGET}"


def _duration(seconds: float) -> str:
    total = max(0, round(seconds))
    minutes, seconds = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:d}h{minutes:02d}m" if hours else f"{minutes:02d}m{seconds:02d}s"


class _Progress:
    def __init__(self, total: int, *, enabled: bool) -> None:
        self.total = total
        self.enabled = enabled
        self.index = 0
        self.seed = 0
        self.started = 0.0

    def start(self, index: int, seed: int, users: int) -> None:
        self.index = index
        self.seed = seed
        self.started = time.monotonic()
        self.update(0, users)

    def update(self, done: int, total: int) -> None:
        if not self.enabled:
            return
        width = 24
        filled = round(width * done / max(total, 1))
        bar = "#" * filled + "-" * (width - filled)
        sys.stderr.write(
            f"\r\033[2K[{self.index}/{self.total}] seed={self.seed:<4} "
            f"[{bar}] users={done:>5}/{total} "
            f"elapsed={_duration(time.monotonic() - self.started)}"
        )
        sys.stderr.flush()

    def finish(self, *, cached: bool = False) -> None:
        if self.enabled:
            print(" cached" if cached else " done", file=sys.stderr, flush=True)


def _mean_std(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
    }


def _bucket_recall(
    hits: np.ndarray,
    buckets: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    for bucket in ("overall", *sorted(set(buckets.tolist()))):
        mask = np.ones(hits.shape, dtype=bool) if bucket == "overall" else buckets == bucket
        selected = hits[mask]
        result[bucket] = {
            "count": int(selected.size),
            f"Recall@{BUDGET}": float(selected.mean()) if selected.size else 0.0,
        }
    return result


def _metrics(
    hits: np.ndarray,
    data: YelpSequenceData,
    user_offsets: np.ndarray,
    targets: np.ndarray,
) -> dict[str, Any]:
    target_offsets = targets - 1
    return {
        "user_frequency": _bucket_recall(
            hits,
            np.asarray(data.user_frequency_buckets)[user_offsets],
        ),
        "user_paper": _bucket_recall(
            hits,
            np.asarray(data.user_paper_buckets)[user_offsets],
        ),
        "item_frequency": _bucket_recall(
            hits,
            np.asarray(data.item_frequency_buckets)[target_offsets],
        ),
        "item_paper": _bucket_recall(
            hits,
            np.asarray(data.item_paper_buckets)[target_offsets],
        ),
    }


def _metric(metrics: dict[str, Any], bucket: str) -> float:
    return float(metrics["item_frequency"][bucket][f"Recall@{BUDGET}"])


def _threshold(values: np.ndarray, injection_rate: float) -> float:
    scale = max(1.0, float(np.max(np.abs(values))))
    epsilon = np.finfo(np.float64).eps * scale * 4
    if injection_rate <= 0.0:
        return float(values.max() + epsilon)
    if injection_rate >= 1.0:
        return float(values.min() - epsilon)
    return float(
        np.quantile(
            values,
            1.0 - injection_rate,
            method="higher",
        )
    )


def _adaptive_hits(
    collaborative: np.ndarray,
    fixed_route: np.ndarray,
    uncertainty: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    inject = uncertainty >= threshold
    return np.where(inject, fixed_route, collaborative), inject


def _aggregate(
    per_seed: dict[str, dict[str, Any]],
    method: str,
    *,
    seeds: list[int],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    first = per_seed[str(seeds[0])][method]
    for family, buckets in first.items():
        output[family] = {}
        for bucket in buckets:
            metric = f"Recall@{BUDGET}"
            output[family][bucket] = {
                metric: _mean_std(
                    [
                        float(
                            per_seed[str(seed)][method][family][bucket][metric]
                        )
                        for seed in seeds
                    ]
                )
            }
    return output


def _load_cache(
    path: Path,
    *,
    expected: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("inputs") != expected:
        return None
    arrays_path = Path(payload.get("arrays_path", ""))
    if not arrays_path.is_file():
        return None
    with np.load(arrays_path, allow_pickle=False) as stored:
        arrays = {name: stored[name] for name in stored.files}
    return payload, arrays


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Yelp R1.8 adaptive fixed-budget retrieval",
        "",
        "Validation only. No training or test access. Every route has 500 candidates.",
        "",
        "## Seed 42 uncertainty-threshold selection",
        "",
        "| Target injection | Actual | Threshold | Overall | Head | Tail | Cold | "
        "Feasible |",
        "|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in report["seed42_search"]:
        item = row["metrics"]["item_frequency"]
        lines.append(
            f"| {row['target_injection_rate']:.0%} | "
            f"{row['actual_injection_rate']:.2%} | "
            f"{row['uncertainty_threshold']:.6f} | "
            f"{item['overall'][f'Recall@{BUDGET}']:.6f} | "
            f"{item['head'][f'Recall@{BUDGET}']:.6f} | "
            f"{item['tail'][f'Recall@{BUDGET}']:.6f} | "
            f"{item['cold_start'][f'Recall@{BUDGET}']:.6f} | "
            f"{'yes' if row['feasible'] else 'no'} |"
        )
    selection = report["selection"]
    lines.extend(
        [
            "",
            f"Selected target injection rate: "
            f"`{selection['target_injection_rate']:.0%}`; locked uncertainty "
            f"threshold: `{selection['uncertainty_threshold']:.6f}`.",
            "",
            "## Three-seed validation",
            "",
            "| Method | Overall | Head | Torso | Tail | Cold-start |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for method in (
        "collaborative",
        "always_450_50",
        "adaptive_real",
        "adaptive_shuffled",
    ):
        groups = report["aggregate_validation"][method]["item_frequency"]
        cells = [
            f"{groups[bucket][f'Recall@{BUDGET}']['mean']:.6f} ± "
            f"{groups[bucket][f'Recall@{BUDGET}']['std']:.6f}"
            for bucket in ("overall", "head", "torso", "tail", "cold_start")
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
    parser.add_argument("--source-run-root", type=Path, default=Path("runs/r1_1"))
    parser.add_argument("--run-root", type=Path, default=Path("runs/r1_8/yelp"))
    parser.add_argument(
        "--report-json",
        type=Path,
        default=Path("reports/experiments/yelp_r1_8_adaptive_budget.json"),
    )
    parser.add_argument(
        "--report-markdown",
        type=Path,
        default=Path("reports/experiments/yelp_r1_8_adaptive_budget.md"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--max-eval-users", type=int)
    parser.add_argument("--evaluation-batch-size", type=int)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()

    seeds = list(dict.fromkeys(args.seeds))
    if len(seeds) != len(args.seeds) or not seeds or min(seeds) < 0:
        raise ValueError("seeds must be unique and non-negative")
    if 42 not in seeds:
        raise ValueError("seed 42 is required for threshold selection")
    if args.max_eval_users is not None and args.max_eval_users <= 0:
        raise ValueError("max-eval-users must be positive")

    base = YelpSASRecRunConfig.from_yaml(args.config)
    if base.semantic_path is None:
        raise ValueError("semantic_path is required")
    if args.evaluation_batch_size is not None:
        if args.evaluation_batch_size <= 0:
            raise ValueError("evaluation-batch-size must be positive")
        base = replace(base, evaluation_batch_size=args.evaluation_batch_size)
    real_semantics = np.load(base.semantic_path, allow_pickle=False).astype(np.float32)
    shuffled_path = (
        base.processed_dir
        / "semantic_controls"
        / f"shuffled_seed{CONTROL_SEED}.npy"
    )
    if not shuffled_path.is_file():
        shuffled_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(
            shuffled_path,
            semantic_control(real_semantics, kind="shuffled", seed=CONTROL_SEED),
            allow_pickle=False,
        )
    shuffled_semantics = np.load(shuffled_path, allow_pickle=False).astype(np.float32)
    data_report = json.loads(base.report_path.read_text(encoding="utf-8"))
    statistics = data_report["statistics"]
    data_fingerprint = str(data_report["processed_fingerprint"])
    data = load_yelp_author_sequences(
        base.processed_dir,
        report_path=base.report_path,
        max_users=base.max_users,
    )
    eligible = np.flatnonzero(data.valid_targets > 0)
    user_offsets = eligible[: args.max_eval_users or len(eligible)]
    targets = data.valid_targets[user_offsets]
    expected_users = len(user_offsets)

    progress = _Progress(len(seeds), enabled=args.progress)
    seed_arrays: dict[str, dict[str, np.ndarray]] = {}
    inputs: list[dict[str, Any]] = []
    for index, seed in enumerate(seeds, start=1):
        checkpoint = args.source_run_root / f"llm_init_seed{seed}" / "best_model.pt"
        seed_root = args.run_root / f"llm_init_seed{seed}"
        seed_root.mkdir(parents=True, exist_ok=True)
        suffix = "all" if args.max_eval_users is None else str(args.max_eval_users)
        cache_path = seed_root / f"adaptive_inputs_{suffix}.json"
        arrays_path = seed_root / f"adaptive_inputs_{suffix}.npz"
        expected = {
            "algorithm": "adaptive-fixed-budget-v1",
            "seed": seed,
            "checkpoint_sha256": sha256_file(checkpoint),
            "real_semantic_sha256": sha256_file(base.semantic_path),
            "shuffled_semantic_sha256": sha256_file(shuffled_path),
            "data_fingerprint": data_fingerprint,
            "split": "validation",
            "budget": BUDGET,
            "semantic_quota": SEMANTIC_QUOTA,
            "max_eval_users": args.max_eval_users,
            "evaluation_batch_size": base.evaluation_batch_size,
        }
        progress.start(index, seed, expected_users)
        cached = _load_cache(cache_path, expected=expected)
        if cached is None:
            config = replace(
                base,
                model="llm_init",
                seed=seed,
                output_dir=args.source_run_root / f"llm_init_seed{seed}",
                max_eval_users=args.max_eval_users,
                test_after_selection=False,
            )
            started = time.monotonic()
            result = evaluate_full_catalog_retrieval(
                config,
                checkpoint_path=checkpoint,
                semantic_variants={
                    "real": real_semantics,
                    "shuffled": shuffled_semantics,
                },
                cutoffs=(BUDGET,),
                fixed_budget_semantic_quotas={BUDGET: (SEMANTIC_QUOTA,)},
                split="valid",
                progress_callback=progress.update,
            )
            arrays = {
                "collaborative": result.hits["collaborative"][BUDGET],
                "fixed_real": result.hits[_fixed_route("real")][BUDGET],
                "fixed_shuffled": result.hits[_fixed_route("shuffled")][BUDGET],
                "uncertainty": result.query_uncertainty,
                "count_real": result.candidate_counts[_fixed_route("real")][BUDGET],
                "count_shuffled": result.candidate_counts[
                    _fixed_route("shuffled")
                ][BUDGET],
            }
            np.savez_compressed(arrays_path, **arrays)
            payload = {
                "inputs": expected,
                "elapsed_seconds": time.monotonic() - started,
                "arrays_path": str(arrays_path),
                "protocol": result.protocol,
            }
            cache_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            progress.finish()
        else:
            payload, arrays = cached
            progress.finish(cached=True)
        if not np.all(arrays["count_real"] == BUDGET) or not np.all(
            arrays["count_shuffled"] == BUDGET
        ):
            raise ValueError("adaptive source route does not have exactly 500 candidates")
        seed_arrays[str(seed)] = arrays
        inputs.append(
            {
                **expected,
                "elapsed_seconds": payload["elapsed_seconds"],
                "cache": str(cache_path),
                "arrays": payload["arrays_path"],
            }
        )

    selection_arrays = seed_arrays["42"]
    baseline_metrics = _metrics(
        selection_arrays["collaborative"],
        data,
        user_offsets,
        targets,
    )
    overall_floor = _metric(baseline_metrics, "overall") - SELECTION_TOLERANCE
    head_floor = _metric(baseline_metrics, "head") - SELECTION_TOLERANCE
    search: list[dict[str, Any]] = []
    for rate in INJECTION_RATE_CANDIDATES:
        threshold = _threshold(selection_arrays["uncertainty"], rate)
        hits, injected = _adaptive_hits(
            selection_arrays["collaborative"],
            selection_arrays["fixed_real"],
            selection_arrays["uncertainty"],
            threshold,
        )
        metrics = _metrics(hits, data, user_offsets, targets)
        search.append(
            {
                "target_injection_rate": rate,
                "actual_injection_rate": float(injected.mean()),
                "uncertainty_threshold": threshold,
                "metrics": metrics,
                "feasible": (
                    _metric(metrics, "overall") >= overall_floor
                    and _metric(metrics, "head") >= head_floor
                ),
            }
        )
    feasible = [row for row in search if row["feasible"]]
    selected = max(
        feasible or search,
        key=lambda row: (
            _metric(row["metrics"], "tail"),
            _metric(row["metrics"], "cold_start"),
            _metric(row["metrics"], "overall"),
            -row["actual_injection_rate"],
        ),
    )
    selected_threshold = float(selected["uncertainty_threshold"])

    per_seed: dict[str, dict[str, Any]] = {}
    injection_rates: dict[str, float] = {}
    uncertainty_summary: dict[str, dict[str, float]] = {}
    for seed in seeds:
        arrays = seed_arrays[str(seed)]
        adaptive_real, injected = _adaptive_hits(
            arrays["collaborative"],
            arrays["fixed_real"],
            arrays["uncertainty"],
            selected_threshold,
        )
        adaptive_shuffled, shuffled_injected = _adaptive_hits(
            arrays["collaborative"],
            arrays["fixed_shuffled"],
            arrays["uncertainty"],
            selected_threshold,
        )
        if not np.array_equal(injected, shuffled_injected):
            raise ValueError("real and shuffled routes used different gate decisions")
        injection_rates[str(seed)] = float(injected.mean())
        uncertainty_summary[str(seed)] = {
            "mean": float(arrays["uncertainty"].mean()),
            "std": float(arrays["uncertainty"].std()),
            "min": float(arrays["uncertainty"].min()),
            "max": float(arrays["uncertainty"].max()),
        }
        per_seed[str(seed)] = {
            "collaborative": _metrics(
                arrays["collaborative"],
                data,
                user_offsets,
                targets,
            ),
            "always_450_50": _metrics(
                arrays["fixed_real"],
                data,
                user_offsets,
                targets,
            ),
            "adaptive_real": _metrics(adaptive_real, data, user_offsets, targets),
            "adaptive_shuffled": _metrics(
                adaptive_shuffled,
                data,
                user_offsets,
                targets,
            ),
        }

    methods = (
        "collaborative",
        "always_450_50",
        "adaptive_real",
        "adaptive_shuffled",
    )
    aggregate = {
        method: _aggregate(per_seed, method, seeds=seeds) for method in methods
    }
    gains: dict[str, Any] = {}
    for bucket in ("overall", "head", "torso", "tail", "cold_start"):
        real_minus_collaborative = [
            _metric(per_seed[str(seed)]["adaptive_real"], bucket)
            - _metric(per_seed[str(seed)]["collaborative"], bucket)
            for seed in seeds
        ]
        real_minus_shuffled = [
            _metric(per_seed[str(seed)]["adaptive_real"], bucket)
            - _metric(per_seed[str(seed)]["adaptive_shuffled"], bucket)
            for seed in seeds
        ]
        gains[bucket] = {
            "real_minus_collaborative_per_seed": real_minus_collaborative,
            "real_minus_collaborative": _mean_std(real_minus_collaborative),
            "real_minus_shuffled_per_seed": real_minus_shuffled,
            "real_minus_shuffled": _mean_std(real_minus_shuffled),
        }

    acceptance = {
        "validation_only_no_test_access": True,
        "exactly_500_unique_candidates_all_routes": True,
        "overall_within_0p01_all_seeds": all(
            value >= -REPLICATION_TOLERANCE
            for value in gains["overall"]["real_minus_collaborative_per_seed"]
        ),
        "head_within_0p01_all_seeds": all(
            value >= -REPLICATION_TOLERANCE
            for value in gains["head"]["real_minus_collaborative_per_seed"]
        ),
        "tail_mean_absolute_gain_at_least_0p03": (
            gains["tail"]["real_minus_collaborative"]["mean"] >= TAIL_GAIN_TARGET
        ),
        "cold_mean_absolute_gain_at_least_0p03": (
            gains["cold_start"]["real_minus_collaborative"]["mean"]
            >= COLD_GAIN_TARGET
        ),
        "tail_direction_positive_all_seeds": all(
            value > 0.0
            for value in gains["tail"]["real_minus_collaborative_per_seed"]
        ),
        "tail_real_beats_shuffled_all_seeds": all(
            value > 0.0 for value in gains["tail"]["real_minus_shuffled_per_seed"]
        ),
        "adaptive_injection_nonzero_all_seeds": all(
            value > 0.0 for value in injection_rates.values()
        ),
        "adaptive_injection_below_100_percent_all_seeds": all(
            value < 1.0 for value in injection_rates.values()
        ),
    }
    report = {
        "experiment": "R1.8-adaptive-fixed-budget",
        "dataset": "yelp",
        "status": "complete" if args.max_eval_users is None else "smoke_complete",
        "seeds": seeds,
        "protocol": {
            "split": "validation",
            "test_accessed": False,
            "training": False,
            "candidate_budget": BUDGET,
            "adaptive_routes": {
                "high_confidence": "500 collaborative / 0 semantic",
                "low_confidence": "450 collaborative / 50 semantic",
            },
            "uncertainty": (
                "1 / (1 + collaborative top1-minus-top2 margin after per-query "
                "eligible-catalog z-scoring)"
            ),
            "target_injection_rate_candidates": list(INJECTION_RATE_CANDIDATES),
            "selection_seed": 42,
            "selection_tolerance": SELECTION_TOLERANCE,
            "replication_tolerance": REPLICATION_TOLERANCE,
            "selection_objective": "maximize tail, then cold-start, then overall",
            "threshold_transfer": "numeric seed-42 threshold locked across seeds",
            "evaluated_users_per_seed": expected_users,
            "catalog_items": int(statistics["items"]),
            "max_eval_users": args.max_eval_users,
        },
        "inputs": inputs,
        "seed42_search": search,
        "selection": {
            "target_injection_rate": float(selected["target_injection_rate"]),
            "actual_seed42_injection_rate": float(
                selected["actual_injection_rate"]
            ),
            "uncertainty_threshold": selected_threshold,
            "selected_on": "Yelp seed 42 validation only",
            "feasible": bool(selected["feasible"]),
        },
        "actual_injection_rate_per_seed": injection_rates,
        "uncertainty_summary": uncertainty_summary,
        "per_seed_validation": per_seed,
        "aggregate_validation": aggregate,
        "gains": gains,
        "acceptance": acceptance,
        "stopping_rule": {
            "all_acceptance_pass": all(acceptance.values()),
            "if_fail": (
                "stop candidate-quota tuning; retain Head-Tail trade-off and return "
                "to the full R1.2 mechanism"
            ),
            "if_pass": "proceed to a route-aware reranker",
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
                "stage": report["status"],
                "selection": report["selection"],
                "report": str(args.report_json),
                "acceptance": acceptance,
                "stopping_rule": report["stopping_rule"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
