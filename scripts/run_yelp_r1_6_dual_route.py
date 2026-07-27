"""Run the validation-only Yelp R1.6 dual-route candidate-recall diagnostic."""

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
DEFAULT_CUTOFFS = (100, 500)
CONTROL_SEED = 20240725
PRIMARY_CUTOFF = 500
TAIL_GAIN_TARGET = 0.01
ROUTES = (
    "collaborative",
    "semantic_real",
    "semantic_shuffled",
    "union_real",
    "union_shuffled",
)


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


def _recall(
    metrics: dict[str, Any],
    route: str,
    cutoff: int,
    bucket: str,
) -> float:
    return float(
        metrics[route][str(cutoff)]["item_frequency"][bucket][f"Recall@{cutoff}"]
    )


def _aggregate(
    per_seed: dict[str, dict[str, Any]],
    *,
    seeds: list[int],
    cutoffs: list[int],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for route in ROUTES:
        output[route] = {}
        for cutoff in cutoffs:
            output[route][str(cutoff)] = {}
            first = per_seed[str(seeds[0])][route][str(cutoff)]
            for family, buckets in first.items():
                output[route][str(cutoff)][family] = {}
                for bucket in buckets:
                    metric = f"Recall@{cutoff}"
                    output[route][str(cutoff)][family][bucket] = {
                        metric: _mean_std(
                            [
                                float(
                                    per_seed[str(seed)][route][str(cutoff)][family][
                                        bucket
                                    ][metric]
                                )
                                for seed in seeds
                            ]
                        ),
                    }
    return output


def _candidate_count_summary(values: np.ndarray) -> dict[str, float | int]:
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": int(values.min()),
        "max": int(values.max()),
    }


def _cache_payload(
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
        "# Yelp R1.6 dual-route candidate recall",
        "",
        "Validation only. No Yelp, Fashion, or Beauty test split was accessed.",
        "",
        "Each route is the set union of collaborative Top-K and semantic Top-K, so its "
        "candidate budget lies between K and 2K.",
    ]
    for cutoff in report["protocol"]["cutoffs"]:
        lines.extend(
            [
                "",
                f"## Recall@{cutoff}",
                "",
                "| Route | Overall | Head | Torso | Tail | Cold-start |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for route in ROUTES:
            groups = report["aggregate_validation"][route][str(cutoff)][
                "item_frequency"
            ]
            cells = [
                f"{groups[bucket][f'Recall@{cutoff}']['mean']:.6f} ± "
                f"{groups[bucket][f'Recall@{cutoff}']['std']:.6f}"
                for bucket in ("overall", "head", "torso", "tail", "cold_start")
            ]
            lines.append(f"| {route} | {' | '.join(cells)} |")
        real_budget = np.mean(
            [
                seed["union_real"][str(cutoff)]["mean"]
                for seed in report["candidate_count_summary"].values()
            ]
        )
        shuffled_budget = np.mean(
            [
                seed["union_shuffled"][str(cutoff)]["mean"]
                for seed in report["candidate_count_summary"].values()
            ]
        )
        lines.extend(
            [
                "",
                f"Mean union candidate count: real `{real_budget:.1f}`, "
                f"shuffled `{shuffled_budget:.1f}`.",
            ]
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
        "--config",
        type=Path,
        default=Path("configs/reproduction/yelp_sasrec.yaml"),
    )
    parser.add_argument("--source-run-root", type=Path, default=Path("runs/r1_1"))
    parser.add_argument("--run-root", type=Path, default=Path("runs/r1_6/yelp"))
    parser.add_argument(
        "--report-json",
        type=Path,
        default=Path("reports/experiments/yelp_r1_6_dual_route.json"),
    )
    parser.add_argument(
        "--report-markdown",
        type=Path,
        default=Path("reports/experiments/yelp_r1_6_dual_route.md"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--cutoffs", type=int, nargs="+", default=list(DEFAULT_CUTOFFS))
    parser.add_argument("--max-eval-users", type=int)
    parser.add_argument("--evaluation-batch-size", type=int)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()

    seeds = list(dict.fromkeys(args.seeds))
    cutoffs = sorted(set(args.cutoffs))
    if len(seeds) != len(args.seeds) or not seeds or min(seeds) < 0:
        raise ValueError("seeds must be unique and non-negative")
    if not cutoffs or min(cutoffs) <= 0:
        raise ValueError("cutoffs must be positive")
    if PRIMARY_CUTOFF not in cutoffs and args.max_eval_users is None:
        raise ValueError(f"formal run must include primary cutoff {PRIMARY_CUTOFF}")
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
    data_fingerprint = str(data_report["processed_fingerprint"])
    statistics = data_report["statistics"]
    available_users = int(statistics.get("evaluable_users", statistics["users"]))
    expected_users = min(available_users, args.max_eval_users or available_users)

    progress = _Progress(len(seeds), enabled=args.progress)
    per_seed: dict[str, dict[str, Any]] = {}
    count_summaries: dict[str, dict[str, Any]] = {}
    inputs: list[dict[str, Any]] = []
    for index, seed in enumerate(seeds, start=1):
        checkpoint = args.source_run_root / f"llm_init_seed{seed}" / "best_model.pt"
        seed_root = args.run_root / f"llm_init_seed{seed}"
        seed_root.mkdir(parents=True, exist_ok=True)
        suffix = "all" if args.max_eval_users is None else str(args.max_eval_users)
        cache_path = seed_root / f"validation_retrieval_{suffix}.json"
        arrays_path = seed_root / f"validation_retrieval_{suffix}.npz"
        expected = {
            "seed": seed,
            "checkpoint_sha256": sha256_file(checkpoint),
            "real_semantic_sha256": sha256_file(base.semantic_path),
            "shuffled_semantic_sha256": sha256_file(shuffled_path),
            "data_fingerprint": data_fingerprint,
            "split": "validation",
            "cutoffs": cutoffs,
            "max_eval_users": args.max_eval_users,
            "evaluation_batch_size": base.evaluation_batch_size,
        }
        progress.start(index, seed, expected_users)
        cached = _cache_payload(cache_path, expected=expected)
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
                cutoffs=cutoffs,
                split="valid",
                progress_callback=progress.update,
            )
            arrays = {}
            for route, by_cutoff in result.hits.items():
                for cutoff, values in by_cutoff.items():
                    arrays[f"hit__{route}__{cutoff}"] = values
            for route, by_cutoff in result.candidate_counts.items():
                for cutoff, values in by_cutoff.items():
                    arrays[f"count__{route}__{cutoff}"] = values
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
        per_seed[str(seed)] = payload["metrics"]
        count_summaries[str(seed)] = {
            route: {
                str(cutoff): _candidate_count_summary(
                    arrays[f"count__{route}__{cutoff}"]
                )
                for cutoff in cutoffs
            }
            for route in ("union_real", "union_shuffled")
        }
        inputs.append(
            {
                **expected,
                "elapsed_seconds": payload["elapsed_seconds"],
                "cache": str(cache_path),
                "arrays": payload["arrays_path"],
            }
        )

    aggregate = _aggregate(per_seed, seeds=seeds, cutoffs=cutoffs)
    gains: dict[str, Any] = {}
    for cutoff in cutoffs:
        gains[str(cutoff)] = {}
        for bucket in ("overall", "head", "torso", "tail", "cold_start"):
            real_minus_collaborative = [
                _recall(per_seed[str(seed)], "union_real", cutoff, bucket)
                - _recall(per_seed[str(seed)], "collaborative", cutoff, bucket)
                for seed in seeds
            ]
            real_minus_shuffled = [
                _recall(per_seed[str(seed)], "union_real", cutoff, bucket)
                - _recall(per_seed[str(seed)], "union_shuffled", cutoff, bucket)
                for seed in seeds
            ]
            gains[str(cutoff)][bucket] = {
                "real_union_minus_collaborative_per_seed": real_minus_collaborative,
                "real_union_minus_collaborative": _mean_std(
                    real_minus_collaborative
                ),
                "real_union_minus_shuffled_union_per_seed": real_minus_shuffled,
                "real_union_minus_shuffled_union": _mean_std(real_minus_shuffled),
            }

    primary = gains[str(PRIMARY_CUTOFF)] if PRIMARY_CUTOFF in cutoffs else None
    acceptance = {
        "validation_only_no_test_access": True,
        f"tail_union_absolute_gain_at_{PRIMARY_CUTOFF}_at_least_0p01": (
            primary is not None
            and primary["tail"]["real_union_minus_collaborative"]["mean"]
            >= TAIL_GAIN_TARGET
        ),
        f"tail_real_union_beats_shuffled_all_seeds_at_{PRIMARY_CUTOFF}": (
            primary is not None
            and all(
                value > 0.0
                for value in primary["tail"][
                    "real_union_minus_shuffled_union_per_seed"
                ]
            )
        ),
        f"tail_union_direction_positive_all_seeds_at_{PRIMARY_CUTOFF}": (
            primary is not None
            and all(
                value > 0.0
                for value in primary["tail"][
                    "real_union_minus_collaborative_per_seed"
                ]
            )
        ),
        f"head_union_not_below_collaborative_at_{PRIMARY_CUTOFF}": (
            primary is not None
            and all(
                value >= 0.0
                for value in primary["head"][
                    "real_union_minus_collaborative_per_seed"
                ]
            )
        ),
    }
    report = {
        "experiment": "R1.6-dual-route-candidate-recall",
        "dataset": "yelp",
        "status": "complete" if args.max_eval_users is None else "smoke_complete",
        "seeds": seeds,
        "protocol": {
            "split": "validation",
            "test_accessed": False,
            "selection_or_training": False,
            "source_checkpoint": "converged Yelp R1.1 LLMInit",
            "collaborative_route": "LLMInit sequential score",
            "semantic_route": "frozen prefix-mean cosine score",
            "control": f"row-shuffled semantic mapping, seed {CONTROL_SEED}",
            "cutoffs": cutoffs,
            "primary_cutoff": PRIMARY_CUTOFF,
            "union_definition": "collaborative Top-K set union semantic Top-K",
            "union_budget": "between K and 2K; actual counts reported",
            "evaluated_users_per_seed": expected_users,
            "catalog_items": int(statistics["items"]),
            "max_eval_users": args.max_eval_users,
        },
        "inputs": inputs,
        "candidate_count_summary": count_summaries,
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
                "report": str(args.report_json),
                "acceptance": acceptance,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
