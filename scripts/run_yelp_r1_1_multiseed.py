"""Run and aggregate the frozen R1.1 Yelp multi-seed experiment matrix."""

from __future__ import annotations

import argparse
import json
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
    args = parser.parse_args()

    base = YelpSASRecRunConfig.from_yaml(args.config)
    summaries: dict[str, list[dict[str, Any]]] = {model: [] for model in MODELS}
    for seed in SEEDS:
        for model in MODELS:
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
                summary = run_yelp_sasrec(config)
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
