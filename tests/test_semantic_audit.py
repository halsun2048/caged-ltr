from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as parquet
import pytest
import torch

from caged_ltr.models import SASRec, SASRecConfig
from caged_ltr.sequential import (
    YelpSASRecRunConfig,
    checkpoint_embedding_drift,
    evaluate_full_catalog,
    evaluate_full_catalog_retrieval,
    semantic_control,
)


def _fixture(root: Path) -> tuple[YelpSASRecRunConfig, np.ndarray, Path]:
    processed = root / "processed"
    processed.mkdir()
    sequence_rows = [
        {
            "user_idx": user,
            "user_frequency_bucket": "head" if user < 2 else "tail",
            "user_paper_bucket": "head" if user < 2 else "tail",
            "train_item_ids": [user % 3, (user + 1) % 4],
            "valid_item_id": (user + 2) % 5,
            "test_item_id": user % 3 if user == 0 else (user + 3) % 6,
        }
        for user in range(4)
    ]
    sequence_rows.append(
        {
            "user_idx": 4,
            "user_frequency_bucket": "tail",
            "user_paper_bucket": "tail",
            "train_item_ids": [0, 1],
            "valid_item_id": None,
            "test_item_id": None,
        }
    )
    parquet.write_table(
        pa.Table.from_pylist(sequence_rows),
        processed / "sequences.parquet",
    )
    parquet.write_table(
        pa.Table.from_pylist(
            [
                {
                    "item_idx": item,
                    "frequency_bucket": "head" if item < 2 else "tail",
                    "paper_bucket": "head" if item < 2 else "tail",
                }
                for item in range(6)
            ]
        ),
        processed / "items.parquet",
    )
    report = root / "report.json"
    report.write_text(
        json.dumps({"processed_fingerprint": "full-catalog-fixture"}),
        encoding="utf-8",
    )
    semantics = np.arange(24, dtype=np.float32).reshape(6, 4) + 1.0
    semantic_path = root / "semantics.npy"
    np.save(semantic_path, semantics)
    model_config = SASRecConfig(
        num_items=6,
        max_length=4,
        hidden_dim=4,
        num_blocks=1,
        num_heads=1,
        dropout=0.0,
    )
    checkpoint = root / "best_model.pt"
    torch.save(
        {"state_dict": SASRec(model_config, item_initialization=semantics).state_dict()},
        checkpoint,
    )
    config = YelpSASRecRunConfig(
        processed_dir=processed,
        report_path=report,
        output_dir=root,
        model="llm_init",
        semantic_path=semantic_path,
        max_length=4,
        hidden_dim=4,
        num_blocks=1,
        num_heads=1,
        dropout=0.0,
        evaluation_batch_size=2,
        evaluation_negatives=2,
        top_k=2,
        device="cpu",
        test_after_selection=False,
    )
    return config, semantics, checkpoint


def test_semantic_controls_and_checkpoint_drift(tmp_path: Path) -> None:
    config, semantics, checkpoint = _fixture(tmp_path)
    real = semantic_control(semantics, kind="real", seed=7)
    shuffled = semantic_control(semantics, kind="shuffled", seed=7)
    matched = semantic_control(semantics, kind="matched_random", seed=7)

    np.testing.assert_array_equal(real, semantics)
    assert not np.array_equal(shuffled, semantics)
    np.testing.assert_array_equal(
        np.sort(shuffled, axis=0),
        np.sort(semantics, axis=0),
    )
    np.testing.assert_allclose(matched.mean(axis=0), semantics.mean(axis=0), atol=1e-5)
    np.testing.assert_allclose(matched.std(axis=0), semantics.std(axis=0), atol=1e-5)

    drift = checkpoint_embedding_drift(checkpoint, semantics)
    assert drift["cosine_initial_to_checkpoint"]["mean"] == pytest.approx(1.0)
    assert drift["relative_l2_displacement"]["max"] == pytest.approx(0.0)

    with pytest.raises(ValueError, match="kind"):
        semantic_control(semantics, kind="unknown", seed=7)
    with pytest.raises(ValueError, match="non-negative"):
        semantic_control(semantics, kind="real", seed=-1)
    with pytest.raises(ValueError, match="finite matrix"):
        semantic_control(np.asarray([1.0, 2.0]), kind="real", seed=7)
    assert config.model == "llm_init"


def test_full_catalog_masks_history_and_scores_semantic_controls(tmp_path: Path) -> None:
    config, semantics, checkpoint = _fixture(tmp_path)
    progress: list[tuple[int, int]] = []
    result = evaluate_full_catalog(
        config,
        checkpoint_path=checkpoint,
        semantic_variants={
            "real": semantics,
            "shuffled": semantic_control(semantics, kind="shuffled", seed=9),
        },
        semantic_weight=0.25,
        gated_residual_weight=0.1,
        progress_callback=lambda done, total: progress.append((done, total)),
    )

    assert set(result.ranks) == {
        "llm_init",
        "semantic_only_real",
        "semantic_only_shuffled",
        "fusion_real",
        "fusion_shuffled",
        "confidence_gate_real",
        "confidence_gate_shuffled",
    }
    assert progress == [(2, 4), (4, 4)]
    assert result.protocol["candidate_catalog_size"] == 6
    assert result.protocol["repeated_test_target_in_history"] >= 1
    assert result.protocol["gated_residual_weight"] == pytest.approx(0.1)
    assert result.protocol["exclusion"].startswith("observed history only")
    for method, ranks in result.ranks.items():
        assert ranks.shape == (4,), method
        assert (ranks < 6).all(), method
        assert result.metrics[method]["item_frequency"]["overall"]["count"] == 4

    sasrec_checkpoint = tmp_path / "sasrec.pt"
    torch.save(
        {
            "state_dict": SASRec(
                SASRecConfig(
                    num_items=6,
                    max_length=4,
                    hidden_dim=4,
                    num_blocks=1,
                    num_heads=1,
                    dropout=0.0,
                )
            ).state_dict()
        },
        sasrec_checkpoint,
    )
    sasrec = evaluate_full_catalog(
        YelpSASRecRunConfig(
            processed_dir=config.processed_dir,
            report_path=config.report_path,
            output_dir=config.output_dir,
            model="sasrec",
            max_length=4,
            hidden_dim=4,
            num_blocks=1,
            num_heads=1,
            dropout=0.0,
            evaluation_batch_size=4,
            evaluation_negatives=2,
            top_k=2,
            device="cpu",
            test_after_selection=False,
        ),
        checkpoint_path=sasrec_checkpoint,
    )
    assert set(sasrec.ranks) == {"sasrec"}

    with pytest.raises(ValueError, match="gated_residual_weight"):
        evaluate_full_catalog(
            config,
            checkpoint_path=checkpoint,
            gated_residual_weight=-0.1,
        )


def test_validation_candidate_retrieval_routes_and_union(tmp_path: Path) -> None:
    config, semantics, checkpoint = _fixture(tmp_path)
    progress: list[tuple[int, int]] = []
    result = evaluate_full_catalog_retrieval(
        config,
        checkpoint_path=checkpoint,
        semantic_variants={
            "real": semantics,
            "shuffled": semantic_control(semantics, kind="shuffled", seed=9),
        },
        cutoffs=(1, 2),
        fixed_budget_semantic_quotas={2: (0, 1, 2)},
        progress_callback=lambda done, total: progress.append((done, total)),
    )

    assert set(result.hits) == {
        "collaborative",
        "semantic_real",
        "semantic_shuffled",
        "union_real",
        "union_shuffled",
        "fixed_union_real_s0_of2",
        "fixed_union_real_s1_of2",
        "fixed_union_real_s2_of2",
        "fixed_union_shuffled_s0_of2",
        "fixed_union_shuffled_s1_of2",
        "fixed_union_shuffled_s2_of2",
    }
    assert set(result.candidate_counts) == {
        "union_real",
        "union_shuffled",
        "fixed_union_real_s0_of2",
        "fixed_union_real_s1_of2",
        "fixed_union_real_s2_of2",
        "fixed_union_shuffled_s0_of2",
        "fixed_union_shuffled_s1_of2",
        "fixed_union_shuffled_s2_of2",
    }
    assert progress == [(2, 4), (4, 4)]
    assert result.protocol["split"] == "validation"
    assert result.protocol["test_accessed"] is False
    assert result.protocol["history"] == "training interactions only"
    for cutoff in (1, 2):
        collaborative = result.hits["collaborative"][cutoff]
        assert collaborative.shape == (4,)
        for variant in ("real", "shuffled"):
            union = result.hits[f"union_{variant}"][cutoff]
            semantic = result.hits[f"semantic_{variant}"][cutoff]
            assert np.all(union >= collaborative)
            assert np.all(union >= semantic)
            counts = result.candidate_counts[f"union_{variant}"][cutoff]
            assert np.all(counts >= cutoff)
            assert np.all(counts <= 2 * cutoff)
        metric = result.metrics["union_real"][str(cutoff)]["item_frequency"][
            "overall"
        ][f"Recall@{cutoff}"]
        assert 0.0 <= metric <= 1.0

    for variant in ("real", "shuffled"):
        collaborative = result.hits["collaborative"][2]
        semantic = result.hits[f"semantic_{variant}"][2]
        zero = f"fixed_union_{variant}_s0_of2"
        one = f"fixed_union_{variant}_s1_of2"
        two = f"fixed_union_{variant}_s2_of2"
        np.testing.assert_array_equal(result.hits[zero][2], collaborative)
        np.testing.assert_array_equal(result.hits[two][2], semantic)
        np.testing.assert_array_equal(result.candidate_counts[one][2], 2)
        np.testing.assert_array_equal(
            result.collaborative_prefix_lengths[zero][2],
            2,
        )
        np.testing.assert_array_equal(
            result.collaborative_prefix_lengths[two][2],
            0,
        )
        prefixes = result.collaborative_prefix_lengths[one][2]
        assert np.all((prefixes >= 1) & (prefixes <= 2))

    with pytest.raises(ValueError, match="restricted to validation"):
        evaluate_full_catalog_retrieval(
            config,
            checkpoint_path=checkpoint,
            semantic_variants={"real": semantics},
            cutoffs=(1,),
            split="test",
        )
    with pytest.raises(ValueError, match="positive"):
        evaluate_full_catalog_retrieval(
            config,
            checkpoint_path=checkpoint,
            semantic_variants={"real": semantics},
            cutoffs=(0,),
        )
    with pytest.raises(ValueError, match="cannot exceed"):
        evaluate_full_catalog_retrieval(
            config,
            checkpoint_path=checkpoint,
            semantic_variants={"real": semantics},
            cutoffs=(2,),
            fixed_budget_semantic_quotas={2: (3,)},
        )
