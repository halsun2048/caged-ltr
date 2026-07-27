"""Run Yelp R1.7 fixed-budget collaborative/semantic candidate routing."""

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
    evaluate_full_catalog_retrieval,
    semantic_control,
)

DEFAULT_SEEDS = (42, 2024, 3407)
DEFAULT_SEMANTIC_QUOTAS = (0, 50, 100, 200, 250)
CONTROL_SEED = 20240725
BUDGET = 500
OVERALL_TOLERANCE = 0.01
HEAD_TOLERANCE = 0.01
TAIL_GAIN_TARGET = 0.03
COLD_GAIN_TARGET = 0.03


def _route(variant: str, semantic_quota: int) -> str:
    return f"fixed_union_{variant}_s{semantic_quota}_of{BUDGET}"


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


def _metric(metrics: dict[str, Any], bucket: str) -> float:
    return float(metrics["item_frequency"][bucket][f"Recall@{BUDGET}"])


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
        "# Yelp R1.7 fixed-budget dual-route retrieval",
        "",
        "Validation only. Every route contains exactly 500 unique candidates.",
        "",
        "## Seed 42 quota selection",
        "",
        "| Collaborative / semantic | Overall | Head | Torso | Tail | Cold | "
        "Overlap | Feasible |",
        "|---|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in report["seed42_search"]:
        item = row["metrics"]["item_frequency"]
        semantic_quota = row["semantic_quota"]
        lines.append(
            f"| {BUDGET - semantic_quota}/{semantic_quota} | "
            f"{item['overall'][f'Recall@{BUDGET}']:.6f} | "
            f"{item['head'][f'Recall@{BUDGET}']:.6f} | "
            f"{item['torso'][f'Recall@{BUDGET}']:.6f} | "
            f"{item['tail'][f'Recall@{BUDGET}']:.6f} | "
            f"{item['cold_start'][f'Recall@{BUDGET}']:.6f} | "
            f"{row['mean_overlap_count']:.1f} | "
            f"{'yes' if row['feasible'] else 'no'} |"
        )
    selected = report["selection"]["semantic_quota"]
    lines.extend(
        [
            "",
            f"Selected quota: collaborative `{BUDGET - selected}`, semantic "
            f"`{selected}`.",
            "",
            "## Three-seed validation",
            "",
            "| Method | Overall | Head | Torso | Tail | Cold-start |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for method in (
        "collaborative",
        "semantic_only",
        "fixed_real",
        "fixed_shuffled",
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
    parser.add_argument("--run-root", type=Path, default=Path("runs/r1_7/yelp"))
    parser.add_argument(
        "--report-json",
        type=Path,
        default=Path("reports/experiments/yelp_r1_7_fixed_budget.json"),
    )
    parser.add_argument(
        "--report-markdown",
        type=Path,
        default=Path("reports/experiments/yelp_r1_7_fixed_budget.md"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--semantic-quotas",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEMANTIC_QUOTAS),
    )
    parser.add_argument("--max-eval-users", type=int)
    parser.add_argument("--evaluation-batch-size", type=int)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()

    seeds = list(dict.fromkeys(args.seeds))
    quotas = sorted(set(args.semantic_quotas))
    if len(seeds) != len(args.seeds) or not seeds or min(seeds) < 0:
        raise ValueError("seeds must be unique and non-negative")
    if 42 not in seeds:
        raise ValueError("seed 42 is required for quota selection")
    if not quotas or quotas[0] < 0 or quotas[-1] > BUDGET:
        raise ValueError(f"semantic quotas must lie in [0, {BUDGET}]")
    if 0 not in quotas:
        raise ValueError("semantic quotas must include the collaborative-only control 0")
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
    available_users = int(statistics.get("evaluable_users", statistics["users"]))
    expected_users = min(available_users, args.max_eval_users or available_users)

    progress = _Progress(len(seeds), enabled=args.progress)
    raw_metrics: dict[str, dict[str, Any]] = {}
    prefix_summaries: dict[str, dict[str, dict[str, float]]] = {}
    inputs: list[dict[str, Any]] = []
    for index, seed in enumerate(seeds, start=1):
        checkpoint = args.source_run_root / f"llm_init_seed{seed}" / "best_model.pt"
        seed_root = args.run_root / f"llm_init_seed{seed}"
        seed_root.mkdir(parents=True, exist_ok=True)
        suffix = "all" if args.max_eval_users is None else str(args.max_eval_users)
        cache_path = seed_root / f"fixed_budget_{suffix}.json"
        arrays_path = seed_root / f"fixed_budget_{suffix}.npz"
        expected = {
            "algorithm": "fixed-budget-v1",
            "seed": seed,
            "checkpoint_sha256": sha256_file(checkpoint),
            "real_semantic_sha256": sha256_file(base.semantic_path),
            "shuffled_semantic_sha256": sha256_file(shuffled_path),
            "data_fingerprint": data_fingerprint,
            "split": "validation",
            "budget": BUDGET,
            "semantic_quotas": quotas,
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
                fixed_budget_semantic_quotas={BUDGET: quotas},
                split="valid",
                progress_callback=progress.update,
            )
            arrays = {}
            for route, by_cutoff in result.hits.items():
                arrays[f"hit__{route}"] = by_cutoff[BUDGET]
            for route, by_cutoff in result.candidate_counts.items():
                if route.startswith("fixed_union_"):
                    arrays[f"count__{route}"] = by_cutoff[BUDGET]
            for route, by_cutoff in result.collaborative_prefix_lengths.items():
                arrays[f"prefix__{route}"] = by_cutoff[BUDGET]
            np.savez_compressed(arrays_path, **arrays)
            payload = {
                "inputs": expected,
                "elapsed_seconds": time.monotonic() - started,
                "arrays_path": str(arrays_path),
                "protocol": result.protocol,
                "metrics": result.metrics,
            }
            cache_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            progress.finish()
        else:
            payload, arrays = cached
            progress.finish(cached=True)
        raw_metrics[str(seed)] = payload["metrics"]
        prefix_summaries[str(seed)] = {}
        for variant in ("real", "shuffled"):
            prefix_summaries[str(seed)][variant] = {}
            for quota in quotas:
                route = _route(variant, quota)
                prefixes = arrays[f"prefix__{route}"].astype(np.float64)
                overlap = prefixes - (BUDGET - quota)
                prefix_summaries[str(seed)][variant][str(quota)] = {
                    "mean_collaborative_prefix": float(prefixes.mean()),
                    "mean_overlap_count": float(overlap.mean()),
                    "mean_overlap_fraction_of_semantic_quota": (
                        float(overlap.mean() / quota) if quota else 0.0
                    ),
                }
                counts = arrays[f"count__{route}"]
                if not np.all(counts == BUDGET):
                    raise ValueError("fixed-budget route did not contain exactly 500 items")
        inputs.append(
            {
                **expected,
                "elapsed_seconds": payload["elapsed_seconds"],
                "cache": str(cache_path),
                "arrays": payload["arrays_path"],
            }
        )

    seed42 = raw_metrics["42"]
    collaborative42 = seed42["collaborative"][str(BUDGET)]
    overall_floor = _metric(collaborative42, "overall") - OVERALL_TOLERANCE
    head_floor = _metric(collaborative42, "head") - HEAD_TOLERANCE
    search: list[dict[str, Any]] = []
    for quota in quotas:
        metrics = seed42[_route("real", quota)][str(BUDGET)]
        feasible = (
            _metric(metrics, "overall") >= overall_floor
            and _metric(metrics, "head") >= head_floor
        )
        search.append(
            {
                "semantic_quota": quota,
                "collaborative_quota": BUDGET - quota,
                "metrics": metrics,
                "mean_overlap_count": prefix_summaries["42"]["real"][str(quota)][
                    "mean_overlap_count"
                ],
                "feasible": feasible,
            }
        )
    feasible = [row for row in search if row["feasible"]]
    selected = max(
        feasible or search,
        key=lambda row: (
            _metric(row["metrics"], "tail"),
            _metric(row["metrics"], "cold_start"),
            _metric(row["metrics"], "overall"),
        ),
    )
    selected_quota = int(selected["semantic_quota"])

    per_seed: dict[str, dict[str, Any]] = {}
    for seed in seeds:
        raw = raw_metrics[str(seed)]
        per_seed[str(seed)] = {
            "collaborative": raw["collaborative"][str(BUDGET)],
            "semantic_only": raw["semantic_real"][str(BUDGET)],
            "fixed_real": raw[_route("real", selected_quota)][str(BUDGET)],
            "fixed_shuffled": raw[_route("shuffled", selected_quota)][str(BUDGET)],
        }
    methods = ("collaborative", "semantic_only", "fixed_real", "fixed_shuffled")
    aggregate = {
        method: _aggregate(per_seed, method, seeds=seeds) for method in methods
    }

    gains: dict[str, Any] = {}
    for bucket in ("overall", "head", "torso", "tail", "cold_start"):
        real_minus_collaborative = [
            _metric(per_seed[str(seed)]["fixed_real"], bucket)
            - _metric(per_seed[str(seed)]["collaborative"], bucket)
            for seed in seeds
        ]
        real_minus_shuffled = [
            _metric(per_seed[str(seed)]["fixed_real"], bucket)
            - _metric(per_seed[str(seed)]["fixed_shuffled"], bucket)
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
            value >= -OVERALL_TOLERANCE
            for value in gains["overall"]["real_minus_collaborative_per_seed"]
        ),
        "head_within_0p01_all_seeds": all(
            value >= -HEAD_TOLERANCE
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
    }
    report = {
        "experiment": "R1.7-fixed-budget-dual-route",
        "dataset": "yelp",
        "status": "complete" if args.max_eval_users is None else "smoke_complete",
        "seeds": seeds,
        "protocol": {
            "split": "validation",
            "test_accessed": False,
            "training": False,
            "candidate_budget": BUDGET,
            "semantic_quota_candidates": quotas,
            "selection_seed": 42,
            "selection_constraints": {
                "overall_recall_tolerance": OVERALL_TOLERANCE,
                "head_recall_tolerance": HEAD_TOLERANCE,
            },
            "selection_objective": "maximize tail, then cold-start, then overall recall",
            "deduplication_and_fill": (
                "semantic quota plus shortest collaborative ranking prefix yielding "
                "exactly 500 unique candidates"
            ),
            "evaluated_users_per_seed": expected_users,
            "catalog_items": int(statistics["items"]),
            "max_eval_users": args.max_eval_users,
        },
        "inputs": inputs,
        "seed42_search": search,
        "selection": {
            "semantic_quota": selected_quota,
            "collaborative_quota": BUDGET - selected_quota,
            "selected_on": "Yelp seed 42 validation only",
            "feasible": bool(selected["feasible"]),
        },
        "prefix_and_overlap_summary": prefix_summaries,
        "per_seed_validation": per_seed,
        "aggregate_validation": aggregate,
        "gains": gains,
        "acceptance": acceptance,
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
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
