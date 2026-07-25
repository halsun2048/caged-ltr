"""Select one confidence-aware semantic strength on Yelp validation only."""

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
    confidence_aware_scores,
    export_validation_scores,
    load_validation_scores,
    save_validation_scores,
    semantic_control,
    validation_metrics,
)

SEEDS = (42, 2024, 3407)
CONTROL_SEED = 20240725
FIXED_WEIGHT = 0.25
WEIGHTS = (0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0)
OVERALL_TOLERANCE = 0.002


def _metric(metrics: dict[str, Any], bucket: str, metric: str = "NDCG@10") -> float:
    return float(metrics["item_frequency"][bucket][metric])


def _load_or_export(
    config: YelpSASRecRunConfig,
    *,
    cache_path: Path,
) -> ValidationScoreBundle:
    checkpoint = config.output_dir / "best_model.pt"
    if config.semantic_path is None:
        raise ValueError("semantic_path is required")
    checkpoint_hash = sha256_file(checkpoint)
    semantic_hash = sha256_file(config.semantic_path)
    if cache_path.is_file():
        bundle = load_validation_scores(cache_path)
        if (
            bundle.seed == config.seed
            and bundle.checkpoint_sha256 == checkpoint_hash
            and bundle.semantic_sha256 == semantic_hash
            and bundle.collaborative.shape[1] == config.evaluation_negatives + 1
        ):
            print(
                json.dumps(
                    {
                        "stage": "cached_validation_scores",
                        "seed": config.seed,
                        "semantic": str(config.semantic_path),
                    }
                ),
                flush=True,
            )
            return bundle
    bundle = export_validation_scores(config, checkpoint_path=checkpoint)
    save_validation_scores(cache_path, bundle)
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
    output: dict[str, Any] = {}
    for bucket in per_seed["42"][method]["item_frequency"]:
        output[bucket] = {
            metric: _mean_std(
                [
                    float(per_seed[str(seed)][method]["item_frequency"][bucket][metric])
                    for seed in SEEDS
                ]
            )
            for metric in ("Hit@10", "NDCG@10")
        }
    return output


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Yelp R1.4 confidence-aware gate selection",
        "",
        "Validation only. No Yelp or Beauty test split was accessed.",
        "",
        f"Selected gated residual strength: `{report['selection']['semantic_weight']}`.",
        "",
        "| Weight | Overall NDCG@10 | Head | Torso | Tail | Feasible |",
        "|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in report["seed42_search"]:
        item = row["metrics"]["item_frequency"]
        lines.append(
            f"| {row['semantic_weight']:g} | {item['overall']['NDCG@10']:.6f} | "
            f"{item['head']['NDCG@10']:.6f} | "
            f"{item['torso']['NDCG@10']:.6f} | "
            f"{item['tail']['NDCG@10']:.6f} | "
            f"{'yes' if row['feasible'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Three-seed validation",
            "",
            "| Method | Overall | Head | Torso | Tail | Cold-start |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for method in ("llm_init", "fixed_fusion", "confidence_gate", "shuffled_gate"):
        groups = report["aggregate_validation"][method]
        cells = [
            f"{groups[bucket]['NDCG@10']['mean']:.6f} ± "
            f"{groups[bucket]['NDCG@10']['std']:.6f}"
            for bucket in ("overall", "head", "torso", "tail", "cold_start")
        ]
        lines.append(f"| {method} | {' | '.join(cells)} |")
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
    parser.add_argument("--run-root", type=Path, default=Path("runs/r1_4/yelp"))
    parser.add_argument(
        "--report-json",
        type=Path,
        default=Path("reports/experiments/yelp_r1_4_gate_validation.json"),
    )
    parser.add_argument(
        "--report-markdown",
        type=Path,
        default=Path("reports/experiments/yelp_r1_4_gate_validation.md"),
    )
    args = parser.parse_args()
    base = YelpSASRecRunConfig.from_yaml(args.config)
    if base.semantic_path is None:
        raise ValueError("semantic_path is required")

    real_semantics = np.load(base.semantic_path, allow_pickle=False).astype(np.float32)
    shuffled = semantic_control(
        real_semantics,
        kind="shuffled",
        seed=CONTROL_SEED,
    )
    control_path = (
        base.processed_dir
        / "semantic_controls"
        / f"shuffled_seed{CONTROL_SEED}.npy"
    )
    control_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(control_path, shuffled, allow_pickle=False)

    real_bundles: dict[int, ValidationScoreBundle] = {}
    shuffled_bundles: dict[int, ValidationScoreBundle] = {}
    for seed in SEEDS:
        output_dir = args.r1_run_root / f"llm_init_seed{seed}"
        real_config = replace(
            base,
            model="llm_init",
            seed=seed,
            output_dir=output_dir,
            test_after_selection=False,
        )
        real_bundles[seed] = _load_or_export(
            real_config,
            cache_path=args.run_root / f"real_validation_seed{seed}.npz",
        )
        shuffled_bundles[seed] = _load_or_export(
            replace(real_config, semantic_path=control_path),
            cache_path=args.run_root / f"shuffled_validation_seed{seed}.npz",
        )

    selection_bundle = real_bundles[42]
    fixed_scores = calibrated_scores(
        selection_bundle.collaborative,
        selection_bundle.semantic,
        method="zscore",
        semantic_weight=FIXED_WEIGHT,
    )
    fixed_metrics = validation_metrics(
        selection_bundle,
        fixed_scores,
        top_k=base.top_k,
    )
    overall_floor = _metric(fixed_metrics, "overall") - OVERALL_TOLERANCE
    search: list[dict[str, Any]] = []
    for weight in WEIGHTS:
        scores = confidence_aware_scores(
            selection_bundle.collaborative,
            selection_bundle.semantic,
            _candidate_buckets(selection_bundle),
            semantic_weight=weight,
        )
        metrics = validation_metrics(selection_bundle, scores, top_k=base.top_k)
        search.append(
            {
                "semantic_weight": weight,
                "metrics": metrics,
                "feasible": _metric(metrics, "overall") >= overall_floor,
            }
        )
    feasible = [row for row in search if row["feasible"]]
    pool = feasible or search
    selected = max(
        pool,
        key=lambda row: (
            _metric(row["metrics"], "tail"),
            _metric(row["metrics"], "torso"),
            _metric(row["metrics"], "overall"),
        ),
    )
    selected_weight = float(selected["semantic_weight"])

    per_seed: dict[str, dict[str, Any]] = {}
    for seed in SEEDS:
        real = real_bundles[seed]
        shuffled_bundle = shuffled_bundles[seed]
        real_gate = confidence_aware_scores(
            real.collaborative,
            real.semantic,
            _candidate_buckets(real),
            semantic_weight=selected_weight,
        )
        shuffled_gate = confidence_aware_scores(
            shuffled_bundle.collaborative,
            shuffled_bundle.semantic,
            _candidate_buckets(shuffled_bundle),
            semantic_weight=selected_weight,
        )
        fixed = calibrated_scores(
            real.collaborative,
            real.semantic,
            method="zscore",
            semantic_weight=FIXED_WEIGHT,
        )
        per_seed[str(seed)] = {
            "llm_init": validation_metrics(real, real.collaborative, top_k=base.top_k),
            "fixed_fusion": validation_metrics(real, fixed, top_k=base.top_k),
            "confidence_gate": validation_metrics(
                real,
                real_gate,
                top_k=base.top_k,
            ),
            "shuffled_gate": validation_metrics(
                shuffled_bundle,
                shuffled_gate,
                top_k=base.top_k,
            ),
        }
    aggregate = {
        method: _aggregate(per_seed, method)
        for method in ("llm_init", "fixed_fusion", "confidence_gate", "shuffled_gate")
    }
    report = {
        "experiment": "R1.4-gate-selection",
        "dataset": "yelp",
        "status": "validation_locked",
        "split": "validation",
        "test_accessed": False,
        "seeds": list(SEEDS),
        "control_seed": CONTROL_SEED,
        "fixed_fusion_weight": FIXED_WEIGHT,
        "weight_candidates": list(WEIGHTS),
        "gate_definition": {
            "base_semantic_weight": FIXED_WEIGHT,
            "gated_residual_weight": "selected from weight_candidates",
            "query_uncertainty": "1 / (1 + top1_minus_top2_collaborative_zscore)",
            "item_rarity": {
                "head": 0.0,
                "torso": 0.5,
                "tail": 1.0,
                "cold_start": 1.0,
            },
            "train_only_inputs": True,
            "trainable_parameters": 0,
        },
        "selection_rule": {
            "overall_floor": overall_floor,
            "overall_tolerance_from_fixed_fusion": OVERALL_TOLERANCE,
            "objective": "maximize tail, then torso, then overall NDCG@10",
        },
        "seed42_fixed_fusion": fixed_metrics,
        "seed42_search": search,
        "selection": {
            "semantic_weight": selected_weight,
            "feasible": bool(selected["feasible"]),
            "selected_on": "Yelp seed 42 validation only",
        },
        "per_seed_validation": per_seed,
        "aggregate_validation": aggregate,
        "acceptance": {
            "overall_within_tolerance_all_seeds": all(
                _metric(per_seed[str(seed)]["confidence_gate"], "overall")
                >= _metric(per_seed[str(seed)]["fixed_fusion"], "overall")
                - OVERALL_TOLERANCE
                for seed in SEEDS
            ),
            "tail_beats_fixed_fusion_all_seeds": all(
                _metric(per_seed[str(seed)]["confidence_gate"], "tail")
                > _metric(per_seed[str(seed)]["fixed_fusion"], "tail")
                for seed in SEEDS
            ),
            "real_gate_beats_shuffled_gate_tail_all_seeds": all(
                _metric(per_seed[str(seed)]["confidence_gate"], "tail")
                > _metric(per_seed[str(seed)]["shuffled_gate"], "tail")
                for seed in SEEDS
            ),
        },
        "beauty_transfer": {
            "method_locked": True,
            "semantic_weight": selected_weight,
            "test_accessed": False,
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
                "selection": report["selection"],
                "acceptance": report["acceptance"],
                "report": str(args.report_json),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
