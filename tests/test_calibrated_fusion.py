from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as parquet
import pytest
import torch

from caged_ltr.models import SASRec, SASRecConfig
from caged_ltr.sequential import (
    YelpSASRecRunConfig,
    calibrated_scores,
    export_locked_test_scores,
    export_validation_scores,
    load_validation_scores,
    normalize_branch_scores,
    save_validation_scores,
    score_diagnostics,
    validation_metrics,
)


def _fixture(root: Path) -> tuple[YelpSASRecRunConfig, Path]:
    processed = root / "processed"
    processed.mkdir()
    parquet.write_table(
        pa.Table.from_pylist(
            [
                {
                    "user_idx": user,
                    "user_frequency_bucket": "head" if user < 2 else "tail",
                    "user_paper_bucket": "head" if user < 2 else "tail",
                    "train_item_ids": [user % 4, (user + 1) % 6],
                    "valid_item_id": (user + 2) % 8,
                    "test_item_id": (user + 3) % 8,
                }
                for user in range(6)
            ]
        ),
        processed / "sequences.parquet",
    )
    parquet.write_table(
        pa.Table.from_pylist(
            [
                {
                    "item_idx": item,
                    "frequency_bucket": "head" if item < 3 else "tail",
                    "paper_bucket": "head" if item < 2 else "tail",
                }
                for item in range(8)
            ]
        ),
        processed / "items.parquet",
    )
    report = root / "report.json"
    report.write_text(
        json.dumps({"processed_fingerprint": "calibration-fixture"}),
        encoding="utf-8",
    )
    semantic_path = root / "semantic.npy"
    semantic = np.arange(32, dtype=np.float32).reshape(8, 4) + 1.0
    np.save(semantic_path, semantic)
    output = root / "run"
    output.mkdir()
    model_config = SASRecConfig(
        num_items=8,
        max_length=4,
        hidden_dim=4,
        num_blocks=1,
        num_heads=1,
        dropout=0.0,
    )
    torch.save(
        {"state_dict": SASRec(model_config, item_initialization=semantic).state_dict()},
        output / "best_model.pt",
    )
    return (
        YelpSASRecRunConfig(
            processed_dir=processed,
            report_path=report,
            output_dir=output,
            model="llm_init",
            semantic_path=semantic_path,
            max_length=4,
            hidden_dim=4,
            num_blocks=1,
            num_heads=1,
            dropout=0.0,
            evaluation_batch_size=3,
            evaluation_negatives=2,
            top_k=2,
            test_after_selection=False,
        ),
        semantic_path,
    )


def test_validation_score_export_cache_diagnostics_and_metrics(tmp_path: Path) -> None:
    config, _ = _fixture(tmp_path)
    bundle = export_validation_scores(config)

    assert bundle.collaborative.shape == bundle.semantic.shape == (6, 3)
    assert np.array_equal(bundle.candidates[:, 0], bundle.targets)
    assert bundle.data_fingerprint == "calibration-fixture"
    assert bundle.seed == 42
    assert bundle.evaluation_seed == 20240722

    cache_path = tmp_path / "cache" / "validation.npz"
    save_validation_scores(cache_path, bundle)
    cached = load_validation_scores(cache_path)
    np.testing.assert_array_equal(cached.candidates, bundle.candidates)
    np.testing.assert_allclose(cached.collaborative, bundle.collaborative)
    assert cached.checkpoint_sha256 == bundle.checkpoint_sha256
    assert cached.semantic_sha256 == bundle.semantic_sha256

    zscores = normalize_branch_scores(bundle.collaborative, "zscore")
    np.testing.assert_allclose(zscores.mean(axis=1), 0.0, atol=1e-9)
    source_stds = bundle.collaborative.std(axis=1)
    normalized_stds = zscores.std(axis=1)
    np.testing.assert_allclose(normalized_stds[source_stds > 1e-12], 1.0, atol=1e-12)
    np.testing.assert_array_equal(
        zscores[source_stds <= 1e-12],
        np.zeros_like(zscores[source_stds <= 1e-12]),
    )
    rank_scores = normalize_branch_scores(bundle.semantic, "rank")
    assert set(np.unique(rank_scores)) == {0.0, 0.5, 1.0}

    fused = calibrated_scores(
        bundle.collaborative,
        bundle.semantic,
        method="rank",
        semantic_weight=0.5,
    )
    metrics = validation_metrics(bundle, fused, top_k=2)
    assert metrics["item_frequency"]["overall"]["count"] == 6
    assert 0.0 <= metrics["item_frequency"]["overall"]["NDCG@2"] <= 1.0

    diagnostics = score_diagnostics(bundle, top_k=2)
    assert diagnostics["split"] == "validation"
    assert diagnostics["rows"] == 6
    assert diagnostics["candidates_per_row"] == 3
    assert diagnostics["branch_statistics"]["semantic"]["all_candidates"]["count"] == 18
    assert sum(diagnostics["top_2_item_frequency_exposure"]["semantic"].values()) == pytest.approx(
        1.0
    )

    test_bundle = export_locked_test_scores(config, num_negatives=2)
    data_targets = np.asarray([(user + 3) % 8 + 1 for user in range(6)])
    np.testing.assert_array_equal(test_bundle.targets, data_targets)
    assert test_bundle.collaborative.shape == (6, 3)


def test_calibrated_fusion_rejects_invalid_inputs(tmp_path: Path) -> None:
    config, semantic_path = _fixture(tmp_path)
    with pytest.raises(ValueError, match="llm_init"):
        export_validation_scores(
            YelpSASRecRunConfig(
                processed_dir=config.processed_dir,
                report_path=config.report_path,
                output_dir=config.output_dir,
                model="sasrec",
                semantic_path=semantic_path,
                max_length=4,
                hidden_dim=4,
                num_blocks=1,
                num_heads=1,
                evaluation_negatives=2,
                top_k=2,
            )
        )

    scores = np.asarray([[1.0, 1.0, 1.0], [3.0, 2.0, 1.0]])
    normalized = normalize_branch_scores(scores, "zscore")
    np.testing.assert_array_equal(normalized[0], np.zeros(3))
    with pytest.raises(ValueError, match="method"):
        normalize_branch_scores(scores, "unknown")
    with pytest.raises(ValueError, match="non-negative"):
        calibrated_scores(scores, scores, method="rank", semantic_weight=-1.0)
    with pytest.raises(ValueError, match="align"):
        calibrated_scores(scores, scores[:, :2], method="rank", semantic_weight=1.0)
    with pytest.raises(ValueError, match="test_after_selection=false"):
        export_locked_test_scores(
            replace(config, test_after_selection=True),
            num_negatives=2,
        )
