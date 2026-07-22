"""Run the complete R0 pipeline on deterministic synthetic candidate lists."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from caged_ltr.calibration.plots import save_reliability_diagram
from caged_ltr.data import CandidateDataset, CandidateList, collate_candidate_lists
from caged_ltr.evaluation.buckets import FrequencyBucketer
from caged_ltr.evaluation.efficiency import benchmark_inference, count_parameters
from caged_ltr.evaluation.predictions import prediction_records, write_predictions
from caged_ltr.evaluation.reporting import build_bucket_report
from caged_ltr.losses import pointwise_bce_loss
from caged_ltr.models import DCNv2Student, LambdaMARTRanker, MLPStudent
from caged_ltr.reproducibility import seed_everything, write_environment


def synthetic_candidate_lists(
    *,
    seed: int,
    num_requests: int = 30,
    candidates_per_request: int = 6,
    feature_dim: int = 8,
) -> list[CandidateList]:
    """Build easy but non-trivial request groups with one positive per request."""
    generator = np.random.default_rng(seed)
    true_weights = np.linspace(1.2, -0.4, feature_dim, dtype=np.float32)
    frequencies = np.geomspace(1, 500, num_requests).astype(np.int64)
    examples: list[CandidateList] = []
    for request_index in range(num_requests):
        features = generator.normal(size=(candidates_per_request, feature_dim)).astype(np.float32)
        latent_scores = features @ true_weights + generator.normal(
            scale=0.05, size=candidates_per_request
        )
        labels = np.zeros(candidates_per_request, dtype=np.float32)
        labels[int(np.argmax(latent_scores))] = 1.0
        examples.append(
            CandidateList(
                request_id=f"request-{request_index:03d}",
                candidate_ids=[
                    f"request-{request_index:03d}-candidate-{candidate_index}"
                    for candidate_index in range(candidates_per_request)
                ],
                features=features,
                labels=labels,
                user_id=f"user-{request_index % 10:02d}",
                query_frequency=int(frequencies[request_index]),
                candidate_frequencies=generator.integers(
                    1, 1_000, size=candidates_per_request, dtype=np.int64
                ),
                user_frequency=int(1 + (request_index % 10) ** 2),
            )
        )
    return examples


def _train_neural_model(
    model: torch.nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    steps: int = 100,
) -> torch.nn.Module:
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    model.train()
    for _ in range(steps):
        optimizer.zero_grad()
        loss = pointwise_bce_loss(model(features), labels)
        loss.backward()
        optimizer.step()
    return model.eval()


def _fingerprint_examples(examples: list[CandidateList]) -> str:
    digest = hashlib.sha256()
    for example in examples:
        digest.update(example.request_id.encode("utf-8"))
        digest.update("\0".join(example.candidate_ids).encode("utf-8"))
        digest.update(example.features.tobytes())
        digest.update(example.labels.tobytes())
    return digest.hexdigest()


def run_smoke(output_dir: Path, *, seed: int = 42) -> dict[str, object]:
    """Execute R0 and write all raw and generated artifacts."""
    seed_everything(seed)
    examples = synthetic_candidate_lists(seed=seed)
    test_indices = {0, 1, 14, 15, 28, 29}
    train_examples = [
        example for index, example in enumerate(examples) if index not in test_indices
    ]
    test_examples = [example for index, example in enumerate(examples) if index in test_indices]
    train_batch = next(
        iter(
            DataLoader(
                CandidateDataset(train_examples),
                batch_size=len(train_examples),
                shuffle=False,
                collate_fn=collate_candidate_lists,
            )
        )
    )
    test_batch = collate_candidate_lists(test_examples)

    query_bucketer = FrequencyBucketer().fit(train_batch.query_frequencies.numpy())
    query_buckets = query_bucketer.transform(test_batch.query_frequencies.numpy()).tolist()

    neural_models: dict[str, torch.nn.Module] = {
        "mlp": MLPStudent(train_batch.features.shape[1], hidden_dims=(32, 16), dropout=0.0),
        "dcn_v2": DCNv2Student(
            train_batch.features.shape[1],
            num_cross_layers=2,
            deep_dims=(32, 16),
            dropout=0.0,
        ),
    }
    for model in neural_models.values():
        _train_neural_model(model, train_batch.features, train_batch.labels)

    lambda_mart = LambdaMARTRanker(
        seed=seed,
        n_estimators=40,
        learning_rate=0.08,
        num_leaves=15,
        min_child_samples=2,
    ).fit(
        train_batch.features.numpy(),
        train_batch.labels.numpy(),
        train_batch.group_sizes.tolist(),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_config = {
        "run": {"name": "r0_smoke", "seed": seed},
        "data": {
            "source": "deterministic_synthetic",
            "split": "frequency-stratified holdout for R0 bucket coverage",
            "fingerprint": _fingerprint_examples(examples),
            "train_requests": len(train_examples),
            "test_requests": len(test_examples),
            "candidates_per_request": 6,
            "feature_dim": 8,
        },
        "models": {
            "mlp": {"hidden_dims": [32, 16], "dropout": 0.0, "training_steps": 100},
            "dcn_v2": {
                "num_cross_layers": 2,
                "deep_dims": [32, 16],
                "dropout": 0.0,
                "training_steps": 100,
            },
            "lambda_mart": {
                "n_estimators": 40,
                "learning_rate": 0.08,
                "num_leaves": 15,
                "min_child_samples": 2,
            },
        },
        "evaluation": {"cutoffs": [1, 3, 5], "calibration_bins": 5},
        "prompt": {"name": None, "version": None},
        "notes": {
            "lambda_mart_probability": "uncalibrated sigmoid of LambdaMART ranking score"
        },
    }
    (output_dir / "config.yaml").write_text(
        yaml.safe_dump(resolved_config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    report_rows: list[dict[str, object]] = []
    efficiency: dict[str, dict[str, float | int]] = {}

    with torch.no_grad():
        for model_name, model in neural_models.items():
            scores = model(test_batch.features)
            records = prediction_records(test_batch, scores, query_buckets=query_buckets)
            write_predictions(records, output_dir / f"{model_name}_predictions.parquet")
            save_reliability_diagram(
                [record.label for record in records],
                [record.probability for record in records],
                output_dir / f"{model_name}_reliability.png",
                num_bins=5,
                title=f"{model_name} reliability",
            )
            report_rows.extend(
                {"model": model_name, **row}
                for row in build_bucket_report(records, cutoffs=(1, 3, 5), num_bins=5)
            )
            benchmark = benchmark_inference(
                lambda model=model: model(test_batch.features),
                examples_per_call=test_batch.features.shape[0],
                warmup=2,
                repeats=10,
            )
            efficiency[model_name] = {
                **asdict(benchmark),
                "parameters": count_parameters(model),
            }

    lambda_scores = lambda_mart.predict(test_batch.features.numpy())
    lambda_records = prediction_records(test_batch, lambda_scores, query_buckets=query_buckets)
    write_predictions(lambda_records, output_dir / "lambda_mart_predictions.parquet")
    save_reliability_diagram(
        [record.label for record in lambda_records],
        [record.probability for record in lambda_records],
        output_dir / "lambda_mart_reliability.png",
        num_bins=5,
        title="lambda_mart reliability (uncalibrated proxy)",
    )
    report_rows.extend(
        {"model": "lambda_mart", **row}
        for row in build_bucket_report(lambda_records, cutoffs=(1, 3, 5), num_bins=5)
    )
    lambda_benchmark = benchmark_inference(
        lambda: lambda_mart.predict(test_batch.features.numpy()),
        examples_per_call=test_batch.features.shape[0],
        warmup=2,
        repeats=10,
    )
    efficiency["lambda_mart"] = asdict(lambda_benchmark)

    pd.DataFrame(report_rows).to_csv(output_dir / "metrics.csv", index=False)
    (output_dir / "metrics.json").write_text(
        json.dumps(report_rows, ensure_ascii=False, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "efficiency.json").write_text(
        json.dumps(efficiency, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_environment(output_dir / "environment.json", Path.cwd())
    artifact_names = {path.name for path in output_dir.iterdir()}
    artifact_names.add("run_summary.json")
    summary: dict[str, object] = {
        "seed": seed,
        "data_fingerprint": resolved_config["data"]["fingerprint"],
        "train_requests": len(train_examples),
        "test_requests": len(test_examples),
        "models": [*neural_models, "lambda_mart"],
        "artifacts": sorted(artifact_names),
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/r0_smoke"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    summary = run_smoke(args.output_dir, seed=args.seed)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
