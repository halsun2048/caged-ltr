from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as parquet
import torch

from caged_ltr.data.sequential import (
    SASRecEvaluationDataset,
    SASRecTrainingDataset,
    load_yelp_author_sequences,
)
from caged_ltr.models import (
    DualViewSASRec,
    FrozenRawSemanticSASRec,
    FrozenSemanticLateFusion,
    FrozenSemanticOnly,
    SASRec,
    SASRecConfig,
)
from caged_ltr.sequential import (
    YelpSASRecRunConfig,
    evaluate_yelp_test_checkpoint,
    run_yelp_sasrec,
)


def _write_sequence_fixture(root: Path) -> tuple[Path, Path]:
    processed = root / "processed"
    processed.mkdir()
    sequence_rows = []
    for user in range(6):
        sequence_rows.append(
            {
                "user_idx": user,
                "user_frequency_bucket": "head" if user < 2 else "tail",
                "user_paper_bucket": "head" if user < 2 else "tail",
                "train_item_ids": [user % 4, (user + 1) % 6],
                "valid_item_id": (user + 2) % 8,
                "test_item_id": (user + 3) % 8,
            }
        )
    sequence_rows.append(
        {
            "user_idx": 6,
            "user_frequency_bucket": "tail",
            "user_paper_bucket": "tail",
            "train_item_ids": [0, 1],
            "valid_item_id": None,
            "test_item_id": None,
        }
    )
    item_rows = [
        {
            "item_idx": item,
            "frequency_bucket": "head" if item < 3 else "tail",
            "paper_bucket": "head" if item < 2 else "tail",
        }
        for item in range(8)
    ]
    parquet.write_table(pa.Table.from_pylist(sequence_rows), processed / "sequences.parquet")
    parquet.write_table(pa.Table.from_pylist(item_rows), processed / "items.parquet")
    report = root / "report.json"
    report.write_text(json.dumps({"processed_fingerprint": "fixture-fingerprint"}))
    return processed, report


def test_sequence_datasets_keep_holdouts_out_of_history_and_negatives(tmp_path: Path) -> None:
    processed, report = _write_sequence_fixture(tmp_path)
    data = load_yelp_author_sequences(processed, report_path=report)
    training = SASRecTrainingDataset(data, max_length=4, seed=42)
    sequence, positive, negative = training[0]

    assert sequence.tolist() == [0, 0, 0, 1]
    assert positive.tolist() == [0, 0, 0, 2]
    known = {
        *data.train_histories[0].tolist(),
        int(data.valid_targets[0]),
        int(data.test_targets[0]),
    }
    assert int(negative[-1]) not in known

    valid = SASRecEvaluationDataset(
        data, split="valid", max_length=4, num_negatives=2, seed=42
    )
    test = SASRecEvaluationDataset(
        data, split="test", max_length=4, num_negatives=2, seed=42
    )
    valid_sequence, valid_candidates, _, _ = valid[0]
    test_sequence, test_candidates, _, _ = test[0]
    assert len(valid) == len(test) == 6
    assert len(data.train_histories) == 7
    assert valid_sequence.tolist() == [0, 0, 1, 2]
    assert test_sequence.tolist() == [0, 1, 2, int(data.valid_targets[0])]
    assert int(valid_candidates[0]) == int(data.valid_targets[0])
    assert int(test_candidates[0]) == int(data.test_targets[0])
    assert set(valid_candidates[1:].tolist()).isdisjoint(known)


def test_sasrec_is_causal_and_semantic_table_stays_frozen() -> None:
    config = SASRecConfig(
        num_items=8,
        max_length=4,
        hidden_dim=4,
        num_blocks=1,
        num_heads=1,
        dropout=0.0,
    )
    model = SASRec(config).eval()
    first = torch.tensor([[0, 1, 2, 3]])
    changed_future = torch.tensor([[0, 1, 2, 4]])
    with torch.no_grad():
        first_states = model.encode(first)
        changed_states = model.encode(changed_future)
    torch.testing.assert_close(first_states[:, :3], changed_states[:, :3])

    semantic = np.arange(24, dtype=np.float32).reshape(8, 3) + 1.0
    late_fusion = FrozenSemanticLateFusion(config, semantic)
    before = late_fusion.semantic_items.clone()
    positives = torch.tensor([[0, 0, 2, 3]])
    negatives = torch.tensor([[0, 0, 5, 6]])
    loss = late_fusion.loss(first, positives, negatives)
    loss.backward()
    assert loss.isfinite()
    assert "semantic_items" not in dict(late_fusion.named_parameters())
    torch.testing.assert_close(before, late_fusion.semantic_items)

    semantic_only = FrozenSemanticOnly(config, semantic)
    scores = semantic_only.score_candidates(first, torch.tensor([[4, 5, 6]]))
    assert scores.shape == (1, 3)
    assert sum(parameter.numel() for parameter in semantic_only.parameters()) == 0
    catalog = torch.arange(1, config.num_items + 1).unsqueeze(0)
    with torch.no_grad():
        torch.testing.assert_close(
            model.score_catalog(first),
            model.score_candidates(first, catalog),
        )
        torch.testing.assert_close(
            semantic_only.score_catalog(first),
            semantic_only.score_candidates(first, catalog),
        )


def test_dual_view_is_causal_shared_and_keeps_raw_semantics_frozen() -> None:
    config = SASRecConfig(
        num_items=8,
        max_length=4,
        hidden_dim=4,
        num_blocks=1,
        num_heads=1,
        dropout=0.0,
    )
    raw = np.arange(48, dtype=np.float32).reshape(8, 6) / 48.0
    collaborative = np.arange(32, dtype=np.float32).reshape(8, 4) / 32.0
    model = DualViewSASRec(config, raw, collaborative).eval()

    assert model.semantic_encoder is model.collaborative_encoder
    assert [type(layer) for layer in model.semantic_adapter] == [
        torch.nn.Linear,
        torch.nn.Linear,
    ]
    assert not model.semantic_items.weight.requires_grad
    before = model.semantic_items.weight.detach().clone()
    first = torch.tensor([[0, 1, 2, 3]])
    changed_future = torch.tensor([[0, 1, 2, 4]])
    with torch.no_grad():
        first_views = model.encode_views(first)
        changed_views = model.encode_views(changed_future)
    for first_states, changed_states in zip(
        first_views, changed_views, strict=True
    ):
        torch.testing.assert_close(first_states[:, :3], changed_states[:, :3])

    positives = torch.tensor([[0, 0, 2, 3]])
    negatives = torch.tensor([[0, 0, 5, 6]])
    loss = model.loss(first, positives, negatives)
    loss.backward()
    assert loss.isfinite()
    torch.testing.assert_close(before, model.semantic_items.weight)

    candidates = torch.arange(1, config.num_items + 1).unsqueeze(0)
    with torch.no_grad():
        torch.testing.assert_close(
            model.score_catalog(first),
            model.score_candidates(first, candidates),
        )


def test_dual_view_controls_isolate_sharing_and_cross_attention_capacity() -> None:
    config = SASRecConfig(
        num_items=8,
        max_length=4,
        hidden_dim=4,
        num_blocks=1,
        num_heads=1,
        dropout=0.0,
    )
    raw = np.ones((8, 6), dtype=np.float32)
    collaborative = np.ones((8, 4), dtype=np.float32)
    cross = DualViewSASRec(config, raw, collaborative)
    no_cross = DualViewSASRec(
        config, raw, collaborative, use_cross_attention=False
    )
    unshared = DualViewSASRec(
        config, raw, collaborative, share_encoder=False
    )
    capacity = DualViewSASRec(
        config,
        raw,
        collaborative,
        use_cross_attention=False,
        capacity_control=True,
    )

    assert not hasattr(no_cross, "semantic_from_collaborative")
    assert unshared.semantic_encoder is not unshared.collaborative_encoder
    assert sum(
        parameter.numel() for parameter in cross.parameters() if parameter.requires_grad
    ) == sum(
        parameter.numel()
        for parameter in capacity.parameters()
        if parameter.requires_grad
    )


def test_raw_semantic_only_is_causal_frozen_and_scores_catalog() -> None:
    config = SASRecConfig(
        num_items=8,
        max_length=4,
        hidden_dim=4,
        num_blocks=1,
        num_heads=1,
        dropout=0.0,
    )
    raw = np.arange(48, dtype=np.float32).reshape(8, 6) / 48.0
    model = FrozenRawSemanticSASRec(config, raw).eval()
    assert not model.semantic_items.weight.requires_grad
    assert [type(layer) for layer in model.semantic_adapter] == [
        torch.nn.Linear,
        torch.nn.Linear,
    ]
    before = model.semantic_items.weight.detach().clone()
    first = torch.tensor([[0, 1, 2, 3]])
    changed_future = torch.tensor([[0, 1, 2, 4]])
    with torch.no_grad():
        torch.testing.assert_close(
            model.encode(first)[:, :3],
            model.encode(changed_future)[:, :3],
        )
    positives = torch.tensor([[0, 0, 2, 3]])
    negatives = torch.tensor([[0, 0, 5, 6]])
    loss = model.loss(first, positives, negatives)
    loss.backward()
    assert loss.isfinite()
    torch.testing.assert_close(before, model.semantic_items.weight)
    catalog = torch.arange(1, config.num_items + 1).unsqueeze(0)
    with torch.no_grad():
        torch.testing.assert_close(
            model.score_catalog(first),
            model.score_candidates(first, catalog),
        )


def test_yelp_sasrec_runner_writes_a_leakage_aware_smoke_run(tmp_path: Path) -> None:
    processed, report = _write_sequence_fixture(tmp_path)
    semantic_path = tmp_path / "semantic.npy"
    np.save(semantic_path, np.arange(24, dtype=np.float32).reshape(8, 3) + 1.0)
    output = tmp_path / "run"
    epoch_records: list[dict[str, object]] = []
    summary = run_yelp_sasrec(
        YelpSASRecRunConfig(
            processed_dir=processed,
            report_path=report,
            output_dir=output,
            model="late_fusion",
            semantic_path=semantic_path,
            max_length=4,
            hidden_dim=4,
            num_blocks=1,
            num_heads=1,
            dropout=0.0,
            batch_size=3,
            evaluation_batch_size=3,
            max_epochs=1,
            patience=1,
            evaluation_negatives=2,
            top_k=2,
            test_after_selection=False,
        ),
        epoch_callback=epoch_records.append,
    )

    assert len(epoch_records) == 1
    assert set(epoch_records[0]) == {
        "epoch",
        "train_bpr",
        "valid_NDCG@2",
        "best_NDCG@2",
        "best_epoch",
        "stale_epochs",
    }
    assert epoch_records[0]["epoch"] == epoch_records[0]["best_epoch"] == 1
    assert epoch_records[0]["stale_epochs"] == 0
    assert summary["data_fingerprint"] == "fixture-fingerprint"
    assert summary["semantic_sha256"] is not None
    assert summary["parameters"]["frozen_semantic_values"] == 27
    assert summary["protocol"]["test_usage"] == "not evaluated; validation-only run"
    assert summary["test"] is None
    assert (output / "best_model.pt").is_file()
    assert (output / "predictions.parquet").is_file()
    assert set(parquet.read_table(output / "predictions.parquet")["split"].to_pylist()) == {
        "valid"
    }

    test_metrics = evaluate_yelp_test_checkpoint(
        YelpSASRecRunConfig(
            processed_dir=processed,
            report_path=report,
            output_dir=output,
            model="late_fusion",
            semantic_path=semantic_path,
            max_length=4,
            hidden_dim=4,
            num_blocks=1,
            num_heads=1,
            dropout=0.0,
            batch_size=3,
            evaluation_batch_size=3,
            max_epochs=1,
            patience=1,
            evaluation_negatives=2,
            top_k=2,
            test_after_selection=False,
        )
    )
    assert test_metrics["item_frequency"]["overall"]["count"] == 6
    persisted = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert persisted["protocol"]["test_usage"].startswith("once after external")
    assert persisted["protocol"]["final_test_evaluation_negatives"] == 2


def test_yelp_runner_supports_validation_only_dual_view(tmp_path: Path) -> None:
    processed, report = _write_sequence_fixture(tmp_path)
    collaborative_path = tmp_path / "collaborative.npy"
    raw_path = tmp_path / "raw.npy"
    np.save(
        collaborative_path,
        np.arange(32, dtype=np.float32).reshape(8, 4) / 32.0,
    )
    np.save(raw_path, np.arange(48, dtype=np.float32).reshape(8, 6) / 48.0)
    output = tmp_path / "dual"
    summary = run_yelp_sasrec(
        YelpSASRecRunConfig(
            processed_dir=processed,
            report_path=report,
            output_dir=output,
            model="dual_view",
            semantic_path=collaborative_path,
            raw_semantic_path=raw_path,
            max_length=4,
            hidden_dim=4,
            num_blocks=1,
            num_heads=1,
            dropout=0.0,
            batch_size=3,
            evaluation_batch_size=3,
            max_epochs=1,
            patience=1,
            evaluation_negatives=2,
            top_k=2,
            test_after_selection=False,
        ),
        epoch_callback=lambda _: None,
    )

    assert summary["test"] is None
    assert summary["raw_semantic_sha256"] is not None
    assert summary["parameters"]["frozen_semantic_values"] == 54
    architecture = summary["protocol"]["architecture"]
    assert architecture["shared_sequence_encoder"] is True
    assert architecture["bidirectional_cross_attention"] is True
    assert architecture["cross_attention_mask"] == "causal plus padding"


def test_yelp_runner_supports_raw_semantic_only_ablation(tmp_path: Path) -> None:
    processed, report = _write_sequence_fixture(tmp_path)
    raw_path = tmp_path / "raw.npy"
    np.save(raw_path, np.arange(48, dtype=np.float32).reshape(8, 6) / 48.0)
    output = tmp_path / "raw-only"
    summary = run_yelp_sasrec(
        YelpSASRecRunConfig(
            processed_dir=processed,
            report_path=report,
            output_dir=output,
            model="raw_semantic_only",
            raw_semantic_path=raw_path,
            max_length=4,
            hidden_dim=4,
            num_blocks=1,
            num_heads=1,
            dropout=0.0,
            batch_size=3,
            evaluation_batch_size=3,
            max_epochs=1,
            patience=1,
            evaluation_negatives=2,
            top_k=2,
            test_after_selection=False,
        ),
        epoch_callback=lambda _: None,
    )

    assert summary["test"] is None
    assert summary["semantic_sha256"] is None
    assert summary["raw_semantic_sha256"] is not None
    assert summary["parameters"]["frozen_semantic_values"] == 54
    assert summary["protocol"]["architecture"]["views"].endswith("only")
