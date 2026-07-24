"""Run validation-only diagnostics and calibrated fusion for Yelp R1.2."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from caged_ltr.reproducibility import sha256_file
from caged_ltr.sequential import (
    ValidationScoreBundle,
    YelpSASRecRunConfig,
    calibrated_scores,
    export_validation_scores,
    load_validation_scores,
    save_validation_scores,
    score_diagnostics,
    validation_metrics,
)

SEEDS = (42, 2024, 3407)
METHODS = ("zscore", "rank")
WEIGHTS = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
OVERALL_NDCG_TOLERANCE = 0.002
TAIL_NDCG_TARGET_GAIN = 0.005


def _metric(metrics: dict[str, Any], bucket: str, name: str = "NDCG@10") -> float:
    return float(metrics["item_frequency"][bucket][name])


def _load_or_export(
    config: YelpSASRecRunConfig,
    *,
    cache_path: Path,
) -> ValidationScoreBundle:
    checkpoint = config.output_dir / "best_model.pt"
    expected_checkpoint = sha256_file(checkpoint)
    if config.semantic_path is None:
        raise ValueError("semantic_path is required")
    expected_semantic = sha256_file(config.semantic_path)
    if cache_path.is_file():
        bundle = load_validation_scores(cache_path)
        valid = (
            bundle.seed == config.seed
            and bundle.evaluation_seed == config.evaluation_seed
            and bundle.checkpoint_sha256 == expected_checkpoint
            and bundle.semantic_sha256 == expected_semantic
            and bundle.collaborative.shape[1] == config.evaluation_negatives + 1
        )
        if valid:
            print(
                json.dumps(
                    {
                        "stage": "cached_validation_scores",
                        "seed": config.seed,
                        "path": str(cache_path),
                    }
                ),
                flush=True,
            )
            return bundle
    print(
        json.dumps(
            {
                "stage": "export_validation_scores",
                "seed": config.seed,
                "checkpoint": str(checkpoint),
            }
        ),
        flush=True,
    )
    bundle = export_validation_scores(config, checkpoint_path=checkpoint)
    save_validation_scores(cache_path, bundle)
    return bundle


def _mean_std(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(array.mean()), "std": float(array.std(ddof=1))}


def _aggregate(
    per_seed: dict[str, dict[str, Any]],
    key: str,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    first = per_seed[str(SEEDS[0])][key]["item_frequency"]
    for bucket in first:
        output[bucket] = {
            metric: _mean_std(
                [
                    float(per_seed[str(seed)][key]["item_frequency"][bucket][metric])
                    for seed in SEEDS
                ]
            )
            for metric in ("Hit@10", "NDCG@10")
        }
    return output


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    selection = report["selection"]
    lines = [
        "# Yelp R1.2 validation-only calibrated fusion",
        "",
        "No test split was scored or used.",
        "",
        f"Selected on seed 42 validation: `{selection['method']}` with semantic "
        f"weight `{selection['semantic_weight']}`.",
        "",
        "| Method | Overall NDCG@10 | Torso | Tail | Cold-start | Feasible |",
        "|---|---:|---:|---:|---:|:---:|",
    ]
    for row in report["seed42_search"]:
        item = row["metrics"]["item_frequency"]
        lines.append(
            f"| {row['method']} weight={row['semantic_weight']:g} | "
            f"{item['overall']['NDCG@10']:.6f} | "
            f"{item['torso']['NDCG@10']:.6f} | "
            f"{item['tail']['NDCG@10']:.6f} | "
            f"{item['cold_start']['NDCG@10']:.6f} | "
            f"{'yes' if row['constraints']['fully_feasible'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Three-seed validation replication",
            "",
            "| System | Overall NDCG@10 | Head | Torso | Tail | Cold-start |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for key in ("llm_init", "calibrated"):
        groups = report["aggregate_validation"][key]
        cells = [
            f"{groups[bucket]['NDCG@10']['mean']:.6f} ± "
            f"{groups[bucket]['NDCG@10']['std']:.6f}"
            for bucket in ("overall", "head", "torso", "tail", "cold_start")
        ]
        lines.append(f"| {key} | {' | '.join(cells)} |")
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
    parser.add_argument("--r1-run-root", type=Path, default=Path("runs/r1_1"))
    parser.add_argument("--run-root", type=Path, default=Path("runs/r1_2"))
    parser.add_argument(
        "--report-json",
        type=Path,
        default=Path("reports/experiments/yelp_r1_2_calibrated_validation.json"),
    )
    parser.add_argument(
        "--report-markdown",
        type=Path,
        default=Path("reports/experiments/yelp_r1_2_calibrated_validation.md"),
    )
    args = parser.parse_args()

    base = YelpSASRecRunConfig.from_yaml(args.config)
    bundles: dict[int, ValidationScoreBundle] = {}
    diagnostics: dict[str, Any] = {}
    for seed in SEEDS:
        output_dir = args.r1_run_root / f"llm_init_seed{seed}"
        config = replace(
            base,
            model="llm_init",
            seed=seed,
            output_dir=output_dir,
            test_after_selection=False,
        )
        bundle = _load_or_export(
            config,
            cache_path=args.run_root / f"validation_scores_seed{seed}.npz",
        )
        bundles[seed] = bundle
        diagnostics[str(seed)] = score_diagnostics(bundle, top_k=base.top_k)

    selection_bundle = bundles[42]
    baseline_metrics = validation_metrics(
        selection_bundle,
        selection_bundle.collaborative,
        top_k=base.top_k,
    )
    semantic_metrics = validation_metrics(
        selection_bundle,
        selection_bundle.semantic,
        top_k=base.top_k,
    )
    overall_floor = (
        _metric(baseline_metrics, "overall") - OVERALL_NDCG_TOLERANCE
    )
    torso_floor = _metric(baseline_metrics, "torso")
    search: list[dict[str, Any]] = []
    for method in METHODS:
        for weight in WEIGHTS:
            scores = calibrated_scores(
                selection_bundle.collaborative,
                selection_bundle.semantic,
                method=method,
                semantic_weight=weight,
            )
            metrics = validation_metrics(selection_bundle, scores, top_k=base.top_k)
            overall_pass = _metric(metrics, "overall") >= overall_floor
            torso_pass = _metric(metrics, "torso") >= torso_floor
            search.append(
                {
                    "method": method,
                    "semantic_weight": weight,
                    "metrics": metrics,
                    "constraints": {
                        "overall_floor": overall_floor,
                        "overall_pass": overall_pass,
                        "torso_floor": torso_floor,
                        "torso_pass": torso_pass,
                        "fully_feasible": overall_pass and torso_pass,
                    },
                }
            )
    feasible = [row for row in search if row["constraints"]["fully_feasible"]]
    candidate_pool = feasible or [
        row for row in search if row["constraints"]["overall_pass"]
    ]
    if not candidate_pool:
        candidate_pool = search
    selected = max(
        candidate_pool,
        key=lambda row: (
            _metric(row["metrics"], "tail"),
            _metric(row["metrics"], "cold_start"),
            _metric(row["metrics"], "overall"),
        ),
    )
    print(
        json.dumps(
            {
                "stage": "selected_on_seed42_validation",
                "method": selected["method"],
                "semantic_weight": selected["semantic_weight"],
                "fully_feasible": selected["constraints"]["fully_feasible"],
            }
        ),
        flush=True,
    )

    per_seed: dict[str, dict[str, Any]] = {}
    for seed, bundle in bundles.items():
        calibrated = calibrated_scores(
            bundle.collaborative,
            bundle.semantic,
            method=str(selected["method"]),
            semantic_weight=float(selected["semantic_weight"]),
        )
        per_seed[str(seed)] = {
            "llm_init": validation_metrics(
                bundle, bundle.collaborative, top_k=base.top_k
            ),
            "semantic_only": validation_metrics(
                bundle, bundle.semantic, top_k=base.top_k
            ),
            "calibrated": validation_metrics(bundle, calibrated, top_k=base.top_k),
        }
    aggregate = {
        key: _aggregate(per_seed, key)
        for key in ("llm_init", "semantic_only", "calibrated")
    }
    baseline_aggregate = aggregate["llm_init"]
    calibrated_aggregate = aggregate["calibrated"]
    tail_gain = (
        calibrated_aggregate["tail"]["NDCG@10"]["mean"]
        - baseline_aggregate["tail"]["NDCG@10"]["mean"]
    )
    acceptance = {
        "seed42_candidate_fully_feasible": bool(
            selected["constraints"]["fully_feasible"]
        ),
        "overall_within_tolerance_all_seeds": all(
            _metric(per_seed[str(seed)]["calibrated"], "overall")
            >= _metric(per_seed[str(seed)]["llm_init"], "overall")
            - OVERALL_NDCG_TOLERANCE
            for seed in SEEDS
        ),
        "torso_non_decrease_all_seeds": all(
            _metric(per_seed[str(seed)]["calibrated"], "torso")
            >= _metric(per_seed[str(seed)]["llm_init"], "torso")
            for seed in SEEDS
        ),
        "tail_direction_positive_all_seeds": all(
            _metric(per_seed[str(seed)]["calibrated"], "tail")
            > _metric(per_seed[str(seed)]["llm_init"], "tail")
            for seed in SEEDS
        ),
        "tail_mean_absolute_gain_at_least_0p005": tail_gain
        >= TAIL_NDCG_TARGET_GAIN,
        "cold_start_mean_ndcg_nonzero": calibrated_aggregate["cold_start"][
            "NDCG@10"
        ]["mean"]
        > 0.0,
    }
    report = {
        "experiment": "R1.2",
        "status": "validation_complete",
        "split": "validation",
        "test_accessed": False,
        "seeds": list(SEEDS),
        "evaluation_seed": base.evaluation_seed,
        "selection_seed": 42,
        "methods": list(METHODS),
        "semantic_weights": list(WEIGHTS),
        "selection_rule": {
            "overall_ndcg_tolerance": OVERALL_NDCG_TOLERANCE,
            "torso_non_decrease": True,
            "objective": "maximize tail NDCG, then cold-start NDCG, then overall NDCG",
            "fallback_if_no_fully_feasible": (
                "maximize the same objective among overall-feasible candidates"
            ),
        },
        "seed42_baselines": {
            "llm_init": baseline_metrics,
            "semantic_only": semantic_metrics,
        },
        "seed42_search": search,
        "selection": {
            "method": selected["method"],
            "semantic_weight": selected["semantic_weight"],
            "constraints": selected["constraints"],
        },
        "per_seed_validation": per_seed,
        "aggregate_validation": aggregate,
        "diagnostics": diagnostics,
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
                "stage": "complete",
                "test_accessed": False,
                "selection": report["selection"],
                "acceptance": acceptance,
                "report": str(args.report_json),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
