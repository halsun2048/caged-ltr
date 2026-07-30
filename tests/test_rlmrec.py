from __future__ import annotations

import pickle
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.sparse import coo_matrix, csr_matrix

from caged_ltr.data import rlmrec as rlmrec_data
from caged_ltr.data.rlmrec import RLMRecYelpPreparationConfig, prepare_rlmrec_yelp
from caged_ltr.graph.rlmrec_runner import (
    RLMRecRunConfig,
    _sample_negatives,
    normalized_bipartite_adjacency,
    run_rlmrec,
)
from caged_ltr.models.rlmrec import RLMRecLightGCN
from caged_ltr.reproducibility import sha256_file


def _write_bundle(raw_dir: Path) -> dict[str, str]:
    raw_dir.mkdir()
    users, items = 40, 30
    train_pairs = [(user, user % items) for user in range(users)]
    train_pairs += [(user, (user + 7) % items) for user in range(users)]
    validation_pairs = [(user, (user + 1) % items) for user in range(users)]
    test_pairs = [(user, (user + 2) % items) for user in range(users)]
    matrices = {
        "trn_mat.pkl": train_pairs,
        "val_mat.pkl": validation_pairs,
        "tst_mat.pkl": test_pairs,
    }
    for name, pairs in matrices.items():
        row, column = zip(*pairs, strict=True)
        matrix = coo_matrix(
            (
                np.ones(len(pairs), dtype=np.float64),
                (
                    np.asarray(row, dtype=np.int32),
                    np.asarray(column, dtype=np.int32),
                ),
            ),
            shape=(users, items),
        )
        (raw_dir / name).write_bytes(pickle.dumps(matrix, protocol=4))
    rng = np.random.default_rng(7)
    arrays = {
        "usr_emb_np.pkl": rng.normal(size=(users, 66)),
        "itm_emb_np.pkl": rng.normal(size=(items, 66)),
    }
    for name, array in arrays.items():
        (raw_dir / name).write_bytes(
            pickle.dumps(array.astype(np.float64), protocol=4)
        )
    (raw_dir / "usr_prf.pkl").write_bytes(b"profile audit fixture")
    (raw_dir / "itm_prf.pkl").write_bytes(b"profile audit fixture")
    return {path.name: sha256_file(path) for path in raw_dir.iterdir()}


def test_prepare_and_run_rlmrec_structure_reproduction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_dir = tmp_path / "raw"
    hashes = _write_bundle(raw_dir)
    monkeypatch.setattr(rlmrec_data, "RLMREC_YELP_HASHES", hashes)
    processed_dir = tmp_path / "processed"
    report = prepare_rlmrec_yelp(
        RLMRecYelpPreparationConfig(
            raw_dir=raw_dir,
            processed_dir=processed_dir,
            report_path=tmp_path / "data_report.json",
        )
    )
    assert report["statistics"]["users"] == 40
    assert report["statistics"]["local_semantic_dimension"] == 64
    assert report["semantic_provenance_audit"]["train_only_provenance_verifiable"] is False

    base = RLMRecRunConfig(
        processed_dir=processed_dir,
        output_dir=tmp_path / "run",
        variant="lightgcn",
        seed=3,
        embedding_dim=8,
        layer_count=1,
        keep_rate=1.0,
        batch_size=16,
        evaluation_batch_size=10,
        max_epochs=1,
        evaluation_interval=1,
        patience=1,
        cutoffs=(5, 10, 20),
        device="cpu",
        max_batches_per_epoch=1,
        max_eval_users=10,
    )
    summary = run_rlmrec(base)
    assert summary["best_epoch"] == 0
    assert summary["test"]["overall"]["users"] == 10
    assert 0 <= summary["validation"]["overall"]["Recall@20"] <= 1

    resumed = run_rlmrec(base)
    assert resumed["best_epoch"] == 0
    with pytest.raises(ValueError, match="checkpoint configuration mismatch"):
        run_rlmrec(replace(base, max_epochs=2))
    with pytest.raises(ValueError, match="semantic artifact SHA-256 mismatch"):
        run_rlmrec(
            replace(
                base,
                output_dir=tmp_path / "bad-hash",
                expected_user_semantic_sha256="0" * 64,
            )
        )

    semantic = run_rlmrec(
        replace(
            base,
            output_dir=tmp_path / "semantic",
            variant="semantic_only",
            user_semantic_filename="user_semantics.npy",
            item_semantic_filename="item_semantics.npy",
            reproduction_type="raw semantic test",
            test_after_selection=False,
        )
    )
    assert semantic["test"] is None
    assert semantic["protocol"]["semantic_files"]["user"] == "user_semantics.npy"
    assert "user_semantics.npy" in semantic["artifacts"]
    con = run_rlmrec(
        replace(base, output_dir=tmp_path / "con", variant="rlmrec_con")
    )
    assert np.isfinite(con["validation"]["overall"]["NDCG@20"])
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="no CUDA support"):
            run_rlmrec(replace(base, output_dir=tmp_path / "cuda", device="cuda"))


def test_lightgcn_controls_and_negative_sampling() -> None:
    train = csr_matrix(
        (
            np.ones(4, dtype=np.float32),
            (
                np.array([0, 0, 1, 2]),
                np.array([0, 1, 2, 3]),
            ),
        ),
        shape=(3, 4),
    )
    adjacency = normalized_bipartite_adjacency(train, device=torch.device("cpu"))
    assert adjacency.shape == (7, 7)
    dense = adjacency.to_dense()
    torch.testing.assert_close(dense, dense.T)

    rows = np.array([0, 0, 1, 2])
    negatives = _sample_negatives(train, rows, np.random.default_rng(9))
    assert not np.asarray(train[rows, negatives]).ravel().any()

    semantics_u = torch.randn(3, 6)
    semantics_i = torch.randn(4, 6)
    model = RLMRecLightGCN(
        num_users=3,
        num_items=4,
        adjacency=adjacency,
        variant="shuffled_con",
        embedding_dim=4,
        layer_count=1,
        keep_rate=0.5,
        user_semantics=semantics_u,
        item_semantics=semantics_i,
        contrastive_chunk_size=2,
    )
    model.train()
    loss, parts = model.loss(
        torch.tensor([0, 1]),
        torch.tensor([1, 2]),
        torch.tensor([3, 0]),
    )
    assert torch.isfinite(loss)
    assert parts["alignment_unweighted"] > 0


def test_rlmrec_rejects_invalid_model_settings() -> None:
    adjacency = torch.sparse_coo_tensor(
        torch.empty((2, 0), dtype=torch.long),
        torch.empty(0),
        (2, 2),
    )
    with pytest.raises(ValueError, match="unsupported"):
        RLMRecLightGCN(
            num_users=1,
            num_items=1,
            adjacency=adjacency,
            variant="invalid",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="requires"):
        RLMRecLightGCN(
            num_users=1,
            num_items=1,
            adjacency=adjacency,
            variant="semantic_only",
        )
