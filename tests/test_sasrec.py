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
