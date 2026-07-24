"""Run Fashion semantic controls, drift analysis, and full-catalog evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from caged_ltr.reproducibility import sha256_file
from caged_ltr.sequential import (
    YelpSASRecRunConfig,
    checkpoint_embedding_drift,
    evaluate_full_catalog,
    semantic_control,
)

DEFAULT_SEEDS = (42, 2024, 3407)
CONTROL_SEED = 20240725
SEMANTIC_WEIGHT = 0.25
METHODS = (
    "sasrec",
    "llm_init",
    "semantic_only_real",
    "semantic_only_shuffled",
    "semantic_only_matched_random",
    "fusion_real",
    "fusion_shuffled",
    "fusion_matched_random",
)


def _array_sha256(array: np.ndarray) -> str:
    values = np.ascontiguousarray(array)
    return hashlib.sha256(values.view(np.uint8)).hexdigest()


class _Progress:
    def __init__(self, total_jobs: int) -> None:
        self.total_jobs = total_jobs
        self.job = 0
        self.label = ""

    def start(self, label: str) -> None:
        self.job += 1
        self.label = label
        self.update(0, 1)

    def update(self, done: int, total: int) -> None:
        width = 24
        filled = round(width * done / max(total, 1))
        bar = "#" * filled + "-" * (width - filled)
        sys.stderr.write(
            f"\r\033[2K[{self.job}/{self.total_jobs}] {self.label:<22} "
            f"[{bar}] users={done:>4}/{total}"
        )
        sys.stderr.flush()

    def finish(self, *, cached: bool = False) -> None:
        suffix = " cached" if cached else " done"
        sys.stderr.write(suffix + "\n")
        sys.stderr.flush()


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


def _aggregate_metric(
    metrics: dict[str, Any],
    bucket: str,
    metric: str = "NDCG@10",
) -> float:
    return float(metrics["item_frequency"][bucket][metric]["mean"])


def _cached_result(
    path: Path,
    *,
    checkpoint_sha256: str,
    variant_hashes: dict[str, str],
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("checkpoint_sha256") == checkpoint_sha256
        and payload.get("variant_hashes") == variant_hashes
    ):
        return payload
    return None


def _evaluate_cached(
    config: YelpSASRecRunConfig,
    *,
    checkpoint: Path,
    variants: dict[str, np.ndarray],
    cache_path: Path,
    progress: _Progress,
    label: str,
) -> dict[str, Any]:
    checkpoint_hash = sha256_file(checkpoint)
    variant_hashes = {name: _array_sha256(values) for name, values in variants.items()}
    progress.start(label)
    cached = _cached_result(
        cache_path,
        checkpoint_sha256=checkpoint_hash,
        variant_hashes=variant_hashes,
    )
    if cached is not None:
        progress.finish(cached=True)
        return cached
    result = evaluate_full_catalog(
        config,
        checkpoint_path=checkpoint,
        semantic_variants=variants,
        semantic_weight=SEMANTIC_WEIGHT,
        progress_callback=progress.update,
    )
    payload = {
        "checkpoint_sha256": checkpoint_hash,
        "variant_hashes": variant_hashes,
        "metrics": result.metrics,
        "protocol": result.protocol,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    progress.finish()
    return payload


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Fashion R1.3b semantic audit and full-catalog results",
        "",
        "Fusion is fixed to per-query full-catalog z-score with semantic weight 0.25. "
        "No Fashion metric selected a control or weight.",
        "",
        "| Method | H@10 mean ± std | NDCG@10 mean ± std |",
        "|---|---:|---:|",
    ]
    for method in METHODS:
        overall = report["aggregate_full_catalog"][method]["item_frequency"]["overall"]
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
    for method in METHODS:
        groups = report["aggregate_full_catalog"][method]["item_frequency"]
        cells = [
            f"{groups[bucket]['NDCG@10']['mean']:.6f} ± "
            f"{groups[bucket]['NDCG@10']['std']:.6f}"
            for bucket in ("head", "torso", "tail", "cold_start")
        ]
        lines.append(f"| {method} | {' | '.join(cells)} |")
    lines.extend(
        [
            "",
            "## Completed checks",
            "",
            *[
                f"- {name}: {'pass' if passed else 'fail'}"
                for name, passed in report["completed_checks"].items()
            ],
            "",
            "## Provenance boundary",
            "",
            f"- Status: `{report['source_audit']['provenance_status']}`.",
            "- Item prompts use title, brand, date, price, feature, and description.",
            "- No interaction history is directly included in item prompts.",
            "- The raw metadata snapshot and generation timestamp are absent from the bundle.",
            "- User embeddings are not used by these experiments.",
            "",
            "## Remaining",
            "",
            *[f"- {item}" for item in report["remaining_work"]],
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
    parser.add_argument("--cache-root", type=Path, default=Path("runs/r1_3b/fashion"))
    parser.add_argument(
        "--report-json",
        type=Path,
        default=Path("reports/experiments/fashion_r1_3b_semantic_audit.json"),
    )
    parser.add_argument(
        "--report-markdown",
        type=Path,
        default=Path("reports/experiments/fashion_r1_3b_semantic_audit.md"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--control-seed", type=int, default=CONTROL_SEED)
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        raise ValueError("seeds must be unique and non-negative")

    base = YelpSASRecRunConfig.from_yaml(args.config)
    if base.semantic_path is None:
        raise ValueError("semantic_path is required")
    real = np.load(base.semantic_path, allow_pickle=False).astype(np.float32)
    variants = {
        kind: semantic_control(real, kind=kind, seed=args.control_seed)
        for kind in ("real", "shuffled", "matched_random")
    }
    progress = _Progress(len(args.seeds) * 2)
    per_seed: dict[str, dict[str, Any]] = {}
    drift: dict[str, Any] = {}
    protocols: dict[str, Any] = {}
    for seed in args.seeds:
        seed_key = str(seed)
        sasrec_dir = args.run_root / f"sasrec_seed{seed}"
        sasrec = _evaluate_cached(
            replace(
                base,
                model="sasrec",
                seed=seed,
                output_dir=sasrec_dir,
                test_after_selection=False,
            ),
            checkpoint=sasrec_dir / "best_model.pt",
            variants={},
            cache_path=args.cache_root / f"sasrec_full_catalog_seed{seed}.json",
            progress=progress,
            label=f"sasrec seed={seed}",
        )
        llm_dir = args.run_root / f"llm_init_seed{seed}"
        llm = _evaluate_cached(
            replace(
                base,
                model="llm_init",
                seed=seed,
                output_dir=llm_dir,
                test_after_selection=False,
            ),
            checkpoint=llm_dir / "best_model.pt",
            variants=variants,
            cache_path=args.cache_root / f"llm_full_catalog_seed{seed}.json",
            progress=progress,
            label=f"llm+controls seed={seed}",
        )
        per_seed[seed_key] = {**sasrec["metrics"], **llm["metrics"]}
        protocols[seed_key] = {
            "sasrec": sasrec["protocol"],
            "llm_init_and_controls": llm["protocol"],
        }
        drift[seed_key] = checkpoint_embedding_drift(
            llm_dir / "best_model.pt",
            real,
        )

    aggregate = {method: _aggregate(per_seed, method) for method in METHODS}
    fusion_overall_gains = [
        _metric(per_seed[str(seed)]["fusion_real"], "overall")
        - _metric(per_seed[str(seed)]["llm_init"], "overall")
        for seed in args.seeds
    ]
    fusion_tail_gains = [
        _metric(per_seed[str(seed)]["fusion_real"], "tail")
        - _metric(per_seed[str(seed)]["llm_init"], "tail")
        for seed in args.seeds
    ]
    report = {
        "experiment": "R1.3b",
        "dataset": "fashion",
        "status": "inference_audit_complete",
        "seeds": args.seeds,
        "control_seed": args.control_seed,
        "semantic_weight": SEMANTIC_WEIGHT,
        "source_audit": {
            "repository": "https://github.com/Applied-Machine-Learning-Lab/LLM-ESR",
            "audited_commit": "e5dc388c",
            "item_notebook": "data/fashion/get_item_embedding.ipynb",
            "embedding_model_declared": "text-embedding-ada-002",
            "item_prompt_fields": [
                "title",
                "brand",
                "date",
                "price",
                "feature",
                "description",
            ],
            "interaction_fields_in_item_prompt": [],
            "pca": "PCA(n_components=64), fitted over the full item catalog",
            "user_embeddings_used": False,
            "prompt_issue": "the date placeholder is described as score",
            "provenance_status": "not_fully_verifiable",
            "unresolved": (
                "author bundle omits raw item metadata snapshot, timestamps, and raw ID map"
            ),
        },
        "control_definitions": {
            "real": "unaltered PCA64 author item embeddings",
            "shuffled": "fixed row permutation; exact vector and norm distribution preserved",
            "matched_random": (
                "deterministic Gaussian values standardized to each real dimension's "
                "mean and standard deviation"
            ),
            "inference_scope": (
                "semantic-only and frozen inference branch; LLMInit checkpoints were "
                "trained with real semantic initialization"
            ),
        },
        "protocols": protocols,
        "per_seed_full_catalog": per_seed,
        "aggregate_full_catalog": aggregate,
        "embedding_drift": drift,
        "gains_fusion_real_minus_llm_init": {
            "overall_ndcg_per_seed": fusion_overall_gains,
            "tail_ndcg_per_seed": fusion_tail_gains,
            "overall_ndcg_mean": float(np.mean(fusion_overall_gains)),
            "tail_ndcg_mean": float(np.mean(fusion_tail_gains)),
        },
        "completed_checks": {
            "full_catalog_overall_direction_positive_all_seeds": all(
                gain > 0 for gain in fusion_overall_gains
            ),
            "full_catalog_tail_direction_positive_all_seeds": all(
                gain > 0 for gain in fusion_tail_gains
            ),
            "semantic_only_real_beats_shuffled_overall": (
                _aggregate_metric(aggregate["semantic_only_real"], "overall")
                > _aggregate_metric(aggregate["semantic_only_shuffled"], "overall")
            ),
            "semantic_only_real_beats_matched_random_overall": (
                _aggregate_metric(aggregate["semantic_only_real"], "overall")
                > _aggregate_metric(
                    aggregate["semantic_only_matched_random"], "overall"
                )
            ),
            "fusion_real_beats_shuffled_overall": (
                _aggregate_metric(aggregate["fusion_real"], "overall")
                > _aggregate_metric(aggregate["fusion_shuffled"], "overall")
            ),
            "fusion_real_beats_matched_random_overall": (
                _aggregate_metric(aggregate["fusion_real"], "overall")
                > _aggregate_metric(aggregate["fusion_matched_random"], "overall")
            ),
        },
        "remaining_work": [
            "retrain LLMInit with shuffled semantic initialization for three seeds",
            "retrain LLMInit with matched-random initialization for three seeds",
            "regenerate embeddings from a versioned pre-cutoff metadata snapshot if available",
        ],
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
                "status": report["status"],
                "completed_checks": report["completed_checks"],
                "report": str(args.report_json),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
