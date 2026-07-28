"""Preparation helpers for the public RLMRec Yelp bundle."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import coo_matrix, save_npz
from sklearn.decomposition import PCA

from caged_ltr.data.safe_numpy_pickle import decode_numpy_ndarray_pickle
from caged_ltr.data.safe_scipy_pickle import decode_scipy_coo_pickle
from caged_ltr.reproducibility import sha256_file

RLMREC_YELP_HASHES = {
    "trn_mat.pkl": "2306f5b9de8f3dd3342744d937a45b04abd22ede422304a549fb324e12e5f6bd",
    "val_mat.pkl": "33aaf3bd27bd25d7d3e8223700fbd2d5400feec217c9ce09c87785460bcb355b",
    "tst_mat.pkl": "3007de4f5955223bf7752f59cadb3b069d92acb879987e7eee286f252772a564",
    "usr_emb_np.pkl": "963b39db0b7f10fd6a673c61cdc620fe981a6d7acc6cf9515f419679013b4520",
    "itm_emb_np.pkl": "1fe3195f90ae39428eb491d80339d3e520cfe2a837f0c8175f71ebdf143a9315",
    "usr_prf.pkl": "bb3736724cfe8cce89b4872eeb901ff2a59777554c61a76dd2742a4318eb3424",
    "itm_prf.pkl": "73851996b2bd7f4d210c1c16d166b522abe84fb7f88d149736e7b4760836a7c8",
}
RLMREC_ARCHIVE_SHA256 = "f7d283fc4b296764294ab836952ff9dda3556e743423da6cc538657c205ab997"


@dataclass(frozen=True)
class RLMRecYelpPreparationConfig:
    raw_dir: Path
    processed_dir: Path
    report_path: Path
    archive: Path | None = None


def _binary_matrix(matrix: coo_matrix) -> coo_matrix:
    result = matrix.copy()
    result.data = np.ones(result.nnz, dtype=np.float32)
    result.sum_duplicates()
    result.data[:] = 1.0
    return result


def _pairs(matrix: coo_matrix) -> set[tuple[int, int]]:
    return set(zip(matrix.row.tolist(), matrix.col.tolist(), strict=True))


def prepare_rlmrec_yelp(config: RLMRecYelpPreparationConfig) -> dict[str, Any]:
    """Safely convert the official public bundle and record its provenance boundary."""
    raw_dir = config.raw_dir
    for name, expected_hash in RLMREC_YELP_HASHES.items():
        path = raw_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"missing official RLMRec Yelp artifact: {path}")
        if sha256_file(path) != expected_hash:
            raise ValueError(f"SHA-256 mismatch for {name}")
    if (
        config.archive is not None
        and sha256_file(config.archive) != RLMREC_ARCHIVE_SHA256
    ):
        raise ValueError("RLMRec public data.zip SHA-256 mismatch")

    matrices = {
        split: _binary_matrix(
            decode_scipy_coo_pickle(
                raw_dir / filename,
                expected_sha256=RLMREC_YELP_HASHES[filename],
            )
        )
        for split, filename in {
            "train": "trn_mat.pkl",
            "validation": "val_mat.pkl",
            "test": "tst_mat.pkl",
        }.items()
    }
    shapes = {matrix.shape for matrix in matrices.values()}
    if len(shapes) != 1:
        raise ValueError("official RLMRec matrices do not share one shape")
    user_count, item_count = next(iter(shapes))

    interaction_sets = {name: _pairs(matrix) for name, matrix in matrices.items()}
    overlaps = {
        "train_validation": len(interaction_sets["train"] & interaction_sets["validation"]),
        "train_test": len(interaction_sets["train"] & interaction_sets["test"]),
        "validation_test": len(
            interaction_sets["validation"] & interaction_sets["test"]
        ),
    }
    if any(overlaps.values()):
        raise ValueError(f"official RLMRec split matrices overlap: {overlaps}")

    user_semantics = decode_numpy_ndarray_pickle(
        raw_dir / "usr_emb_np.pkl",
        expected_sha256=RLMREC_YELP_HASHES["usr_emb_np.pkl"],
    ).astype(np.float32)
    item_semantics = decode_numpy_ndarray_pickle(
        raw_dir / "itm_emb_np.pkl",
        expected_sha256=RLMREC_YELP_HASHES["itm_emb_np.pkl"],
    ).astype(np.float32)
    if user_semantics.shape[0] != user_count or item_semantics.shape[0] != item_count:
        raise ValueError("semantic row counts do not match the interaction matrices")
    if user_semantics.shape[1] != item_semantics.shape[1]:
        raise ValueError("user and item semantic dimensions differ")

    config.processed_dir.mkdir(parents=True, exist_ok=True)
    for split, matrix in matrices.items():
        save_npz(config.processed_dir / f"{split}.npz", matrix.tocsr(), compressed=True)
    np.save(config.processed_dir / "user_semantics.npy", user_semantics, allow_pickle=False)
    np.save(config.processed_dir / "item_semantics.npy", item_semantics, allow_pickle=False)
    combined_semantics = np.concatenate([user_semantics, item_semantics], axis=0)
    pca = PCA(n_components=64, svd_solver="randomized", random_state=20240728)
    combined_pca = pca.fit_transform(combined_semantics).astype(np.float32)
    user_pca, item_pca = np.split(combined_pca, [user_count])
    np.save(config.processed_dir / "user_semantics_pca64.npy", user_pca, allow_pickle=False)
    np.save(config.processed_dir / "item_semantics_pca64.npy", item_pca, allow_pickle=False)

    item_train_degree = np.asarray(matrices["train"].tocsc().sum(axis=0)).ravel()
    user_train_degree = np.asarray(matrices["train"].tocsr().sum(axis=1)).ravel()
    report: dict[str, Any] = {
        "dataset": "RLMRec public Yelp",
        "source": {
            "repository": "https://github.com/HKUDS/RLMRec",
            "repository_commit_audited": "22413752246de3dee8ab0d509f7f7a8889080f95",
            "archive": "https://archive.org/download/rlmrec_data/data.zip",
            "archive_sha256": RLMREC_ARCHIVE_SHA256,
            "artifact_sha256": RLMREC_YELP_HASHES,
        },
        "statistics": {
            "users": user_count,
            "items": item_count,
            "semantic_dimension": int(user_semantics.shape[1]),
            "local_semantic_dimension": int(user_pca.shape[1]),
            "pca64_explained_variance_ratio": float(
                pca.explained_variance_ratio_.sum()
            ),
            "interactions": {
                split: int(matrix.nnz) for split, matrix in matrices.items()
            },
            "users_with_no_train_interaction": int(np.count_nonzero(user_train_degree == 0)),
            "items_with_no_train_interaction": int(np.count_nonzero(item_train_degree == 0)),
            "split_pair_overlaps": overlaps,
        },
        "protocol": {
            "validation_selection_metric": "Recall@20",
            "test_access": "once after checkpoint selection",
            "ranking": "full catalog with training interactions masked",
            "official_lightgcn_layers": 3,
            "official_embedding_dimension": 32,
        },
        "semantic_provenance_audit": {
            "profile_generation_inputs": "raw interaction and review text",
            "split_cutoff_documented": False,
            "train_only_provenance_verifiable": False,
            "status": "temporally_unverified",
            "consequence": (
                "Use only as an official-structure reproduction. Do not interpret semantic "
                "gains as strict leakage-free evidence."
            ),
        },
        "local_structure_reproduction": {
            "enabled": True,
            "adaptation": (
                "Joint deterministic PCA reduces the public 1536-dimensional user/item "
                "profile embeddings to 64 dimensions for CPU training."
            ),
            "faithfulness": (
                "This changes the published semantic encoder output and is therefore a "
                "structure reproduction, not an exact numerical reproduction."
            ),
            "pca_random_state": 20240728,
        },
        "safety": {
            "pickle_load_used": False,
            "decoder": "allowlisted pickletools bytecode extraction",
        },
    }
    config.report_path.parent.mkdir(parents=True, exist_ok=True)
    config.report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report
