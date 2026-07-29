"""Training and full-catalog evaluation for the local RLMRec structure reproduction."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.sparse import csr_matrix, load_npz

from caged_ltr.models.rlmrec import RLMRecLightGCN, RLMRecVariant
from caged_ltr.reproducibility import seed_everything, sha256_file

EpochCallback = Callable[[dict[str, Any]], None]
EvaluationCallback = Callable[[str, int, int], None]


@dataclass(frozen=True)
class RLMRecRunConfig:
    processed_dir: Path
    output_dir: Path
    variant: RLMRecVariant
    seed: int = 42
    embedding_dim: int = 32
    layer_count: int = 3
    keep_rate: float = 0.8
    batch_size: int = 1024
    evaluation_batch_size: int = 512
    learning_rate: float = 1e-3
    baseline_regularization_weight: float = 1e-6
    con_regularization_weight: float = 1e-7
    alignment_weight: float = 1e-2
    temperature: float = 0.2
    max_epochs: int = 300
    evaluation_interval: int = 3
    patience: int = 5
    cutoffs: tuple[int, ...] = (5, 10, 20)
    device: str = "auto"
    test_after_selection: bool = True
    max_batches_per_epoch: int | None = None
    max_eval_users: int | None = None
    control_seed: int = 20240728


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def normalized_bipartite_adjacency(
    train: csr_matrix,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Construct the symmetric degree-normalized LightGCN adjacency."""
    coo = train.tocoo()
    user_degree = np.asarray(train.sum(axis=1)).ravel()
    item_degree = np.asarray(train.sum(axis=0)).ravel()
    weights = np.power(user_degree[coo.row] * item_degree[coo.col], -0.5)
    users = coo.row.astype(np.int64, copy=False)
    items = coo.col.astype(np.int64, copy=False) + train.shape[0]
    indices = np.stack(
        [np.concatenate([users, items]), np.concatenate([items, users])],
        axis=0,
    )
    values = np.concatenate([weights, weights]).astype(np.float32)
    return torch.sparse_coo_tensor(
        torch.from_numpy(indices).to(device),
        torch.from_numpy(values).to(device),
        (sum(train.shape), sum(train.shape)),
        device=device,
    ).coalesce()


def _sample_negatives(
    train: csr_matrix,
    rows: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    negatives = rng.integers(0, train.shape[1], size=rows.size, dtype=np.int64)
    collision = np.asarray(train[rows, negatives]).ravel() != 0
    while np.any(collision):
        negatives[collision] = rng.integers(
            0,
            train.shape[1],
            size=int(collision.sum()),
            dtype=np.int64,
        )
        collision = np.asarray(train[rows, negatives]).ravel() != 0
    return negatives


def _item_buckets(train: csr_matrix) -> tuple[np.ndarray, dict[str, Any]]:
    degree = np.asarray(train.sum(axis=0)).ravel()
    order = np.argsort(degree, kind="stable")
    count = degree.size
    tail_end = max(1, int(np.floor(count * 0.2)))
    head_start = min(count - 1, int(np.ceil(count * 0.8)))
    labels = np.full(count, 1, dtype=np.int8)
    labels[order[:tail_end]] = 0
    labels[order[head_start:]] = 2
    return labels, {
        "definition": "bottom/top 20% of item identities ranked by train interaction count",
        "tail_items": int(np.count_nonzero(labels == 0)),
        "torso_items": int(np.count_nonzero(labels == 1)),
        "head_items": int(np.count_nonzero(labels == 2)),
        "tail_max_train_degree": int(degree[labels == 0].max()),
        "head_min_train_degree": int(degree[labels == 2].min()),
    }


def _empty_accumulator(cutoffs: tuple[int, ...]) -> dict[str, dict[str, Any]]:
    return {
        bucket: {
            "users": 0,
            "recall": np.zeros(len(cutoffs), dtype=np.float64),
            "ndcg": np.zeros(len(cutoffs), dtype=np.float64),
        }
        for bucket in ("overall", "head", "torso", "tail")
    }


def _add_user_metrics(
    accumulator: dict[str, dict[str, Any]],
    *,
    bucket: str,
    ranked: np.ndarray,
    targets: np.ndarray,
    cutoffs: tuple[int, ...],
) -> None:
    if targets.size == 0:
        return
    hits = np.isin(ranked, targets, assume_unique=False)
    accumulator[bucket]["users"] += 1
    for index, cutoff in enumerate(cutoffs):
        cutoff_hits = hits[:cutoff]
        accumulator[bucket]["recall"][index] += cutoff_hits.sum() / targets.size
        discounts = 1.0 / np.log2(np.arange(2, cutoff + 2))
        dcg = float(np.dot(cutoff_hits, discounts))
        ideal = int(min(cutoff, targets.size))
        idcg = float(discounts[:ideal].sum())
        accumulator[bucket]["ndcg"][index] += dcg / idcg


def evaluate_full_catalog(
    model: RLMRecLightGCN,
    *,
    train: csr_matrix,
    targets: csr_matrix,
    item_buckets: np.ndarray,
    cutoffs: tuple[int, ...],
    batch_size: int,
    device: torch.device,
    stage: str,
    callback: EvaluationCallback | None = None,
    max_users: int | None = None,
) -> dict[str, Any]:
    """Evaluate all items while masking only official training interactions."""
    target_users = np.flatnonzero(np.diff(targets.indptr) > 0)
    if max_users is not None:
        target_users = target_users[:max_users]
    accumulator = _empty_accumulator(cutoffs)
    maximum_k = max(cutoffs)
    model.eval()
    with torch.no_grad():
        user_embeddings, item_embeddings = model.ranking_embeddings()
        for start in range(0, target_users.size, batch_size):
            users = target_users[start : start + batch_size]
            user_tensor = torch.from_numpy(users).long().to(device)
            scores = user_embeddings[user_tensor] @ item_embeddings.T
            for local_index, user in enumerate(users):
                train_items = train.indices[train.indptr[user] : train.indptr[user + 1]]
                scores[local_index, torch.from_numpy(train_items).long().to(device)] = -torch.inf
            ranked = torch.topk(scores, k=maximum_k, dim=1).indices.cpu().numpy()
            for local_index, user in enumerate(users):
                user_targets = targets.indices[
                    targets.indptr[user] : targets.indptr[user + 1]
                ]
                _add_user_metrics(
                    accumulator,
                    bucket="overall",
                    ranked=ranked[local_index],
                    targets=user_targets,
                    cutoffs=cutoffs,
                )
                for bucket_index, bucket in enumerate(("tail", "torso", "head")):
                    bucket_targets = user_targets[
                        item_buckets[user_targets] == bucket_index
                    ]
                    _add_user_metrics(
                        accumulator,
                        bucket=bucket,
                        ranked=ranked[local_index],
                        targets=bucket_targets,
                        cutoffs=cutoffs,
                    )
            if callback is not None:
                callback(stage, min(start + batch_size, target_users.size), target_users.size)

    result: dict[str, Any] = {}
    for bucket, values in accumulator.items():
        user_count = int(values["users"])
        denominator = max(user_count, 1)
        metrics = {"users": user_count}
        for index, cutoff in enumerate(cutoffs):
            metrics[f"Recall@{cutoff}"] = float(values["recall"][index] / denominator)
            metrics[f"NDCG@{cutoff}"] = float(values["ndcg"][index] / denominator)
        result[bucket] = metrics
    return result


def _artifact_hashes(processed_dir: Path) -> dict[str, str]:
    return {
        name: sha256_file(processed_dir / name)
        for name in (
            "train.npz",
            "validation.npz",
            "test.npz",
            "user_semantics_pca64.npy",
            "item_semantics_pca64.npy",
        )
    }


def _config_payload(config: RLMRecRunConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["processed_dir"] = str(config.processed_dir)
    payload["output_dir"] = str(config.output_dir)
    payload["cutoffs"] = list(config.cutoffs)
    return payload


def run_rlmrec(
    config: RLMRecRunConfig,
    *,
    epoch_callback: EpochCallback | None = None,
    evaluation_callback: EvaluationCallback | None = None,
) -> dict[str, Any]:
    """Train with validation-only checkpoint selection and test the best checkpoint once."""
    if config.max_epochs <= 0 or config.batch_size <= 0:
        raise ValueError("training sizes must be positive")
    if config.evaluation_interval <= 0 or config.patience <= 0:
        raise ValueError("early-stopping settings must be positive")
    seed_everything(config.seed)
    device = _resolve_device(config.device)
    train = load_npz(config.processed_dir / "train.npz").tocsr().astype(np.float32)
    validation = (
        load_npz(config.processed_dir / "validation.npz").tocsr().astype(np.float32)
    )
    test = load_npz(config.processed_dir / "test.npz").tocsr().astype(np.float32)
    user_semantics: torch.Tensor | None = None
    item_semantics: torch.Tensor | None = None
    if config.variant != "lightgcn":
        user_array = np.load(
            config.processed_dir / "user_semantics_pca64.npy",
            allow_pickle=False,
        )
        item_array = np.load(
            config.processed_dir / "item_semantics_pca64.npy",
            allow_pickle=False,
        )
        if config.variant == "shuffled_con":
            control_rng = np.random.default_rng(config.control_seed)
            user_array = user_array[control_rng.permutation(user_array.shape[0])].copy()
            item_array = item_array[control_rng.permutation(item_array.shape[0])].copy()
        user_semantics = torch.from_numpy(user_array).to(device)
        item_semantics = torch.from_numpy(item_array).to(device)

    adjacency = normalized_bipartite_adjacency(train, device=device)
    regularization_weight = (
        config.baseline_regularization_weight
        if config.variant == "lightgcn"
        else config.con_regularization_weight
    )
    model = RLMRecLightGCN(
        num_users=train.shape[0],
        num_items=train.shape[1],
        adjacency=adjacency,
        variant=config.variant,
        embedding_dim=config.embedding_dim,
        layer_count=config.layer_count,
        keep_rate=config.keep_rate,
        regularization_weight=regularization_weight,
        alignment_weight=config.alignment_weight,
        temperature=config.temperature,
        user_semantics=user_semantics,
        item_semantics=item_semantics,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    item_buckets, bucket_definition = _item_buckets(train)
    train_coo = train.tocoo()
    rows = train_coo.row.astype(np.int64, copy=False)
    columns = train_coo.col.astype(np.int64, copy=False)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    latest_path = config.output_dir / "latest.pt"
    best_path = config.output_dir / "best.pt"
    start_epoch = 0
    best_epoch = -1
    best_recall = -np.inf
    stale_evaluations = 0
    validation_result: dict[str, Any] | None = None
    epochs_ran = 0
    config_signature = json.dumps(
        _config_payload(config),
        ensure_ascii=False,
        sort_keys=True,
    )
    if latest_path.is_file():
        checkpoint = torch.load(latest_path, map_location=device, weights_only=True)
        checkpoint_signature = checkpoint.get("config_signature")
        if (
            checkpoint_signature is not None
            and checkpoint_signature != config_signature
        ):
            raise ValueError(
                f"resume checkpoint configuration mismatch: {latest_path}"
            )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_epoch = int(checkpoint["best_epoch"])
        best_recall = float(checkpoint["best_recall"])
        stale_evaluations = int(checkpoint["stale_evaluations"])
        epochs_ran = start_epoch

    started = time.monotonic()
    for epoch in range(start_epoch, config.max_epochs):
        if stale_evaluations >= config.patience:
            break
        epoch_started = time.monotonic()
        epoch_rng = np.random.default_rng(config.seed * 1_000_003 + epoch)
        order = epoch_rng.permutation(rows.size)
        negatives = _sample_negatives(train, rows, epoch_rng)
        torch.manual_seed(config.seed * 1_000_003 + epoch)
        model.train()
        loss_sums = {"loss": 0.0, "bpr": 0.0, "alignment_unweighted": 0.0}
        batch_count = 0
        total_batches = int(np.ceil(order.size / config.batch_size))
        if config.max_batches_per_epoch is not None:
            total_batches = min(total_batches, config.max_batches_per_epoch)
        for start in range(0, order.size, config.batch_size):
            if (
                config.max_batches_per_epoch is not None
                and batch_count >= config.max_batches_per_epoch
            ):
                break
            indices = order[start : start + config.batch_size]
            users = torch.from_numpy(rows[indices]).long().to(device)
            positives = torch.from_numpy(columns[indices]).long().to(device)
            negative_items = torch.from_numpy(negatives[indices]).long().to(device)
            optimizer.zero_grad(set_to_none=True)
            loss, parts = model.loss(users, positives, negative_items)
            loss.backward()
            optimizer.step()
            loss_sums["loss"] += float(loss.detach())
            loss_sums["bpr"] += parts["bpr"]
            loss_sums["alignment_unweighted"] += parts["alignment_unweighted"]
            batch_count += 1
            if evaluation_callback is not None:
                evaluation_callback(
                    f"train epoch {epoch + 1}",
                    batch_count,
                    total_batches,
                )
        if batch_count == 0:
            raise RuntimeError("training epoch did not contain a batch")
        epochs_ran = epoch + 1

        evaluated = epoch % config.evaluation_interval == 0
        if evaluated:
            validation_result = evaluate_full_catalog(
                model,
                train=train,
                targets=validation,
                item_buckets=item_buckets,
                cutoffs=config.cutoffs,
                batch_size=config.evaluation_batch_size,
                device=device,
                stage="validation",
                callback=evaluation_callback,
                max_users=config.max_eval_users,
            )
            score = float(validation_result["overall"]["Recall@20"])
            if score > best_recall:
                best_recall = score
                best_epoch = epoch
                stale_evaluations = 0
                torch.save(model.state_dict(), best_path)
            else:
                stale_evaluations += 1

        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best_epoch": best_epoch,
                "best_recall": best_recall,
                "stale_evaluations": stale_evaluations,
                "config_signature": config_signature,
            },
            latest_path,
        )
        if epoch_callback is not None:
            epoch_callback(
                {
                    "epoch": epoch + 1,
                    "best_epoch": best_epoch,
                    "best_Recall@20": best_recall,
                    "stale_evaluations": stale_evaluations,
                    "train_loss": loss_sums["loss"] / batch_count,
                    "train_bpr": loss_sums["bpr"] / batch_count,
                    "train_alignment_unweighted": (
                        loss_sums["alignment_unweighted"] / batch_count
                    ),
                    "evaluated": evaluated,
                    "epoch_seconds": time.monotonic() - epoch_started,
                }
            )

    if best_epoch < 0 or not best_path.is_file():
        raise RuntimeError("no validation checkpoint was selected")
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    validation_result = evaluate_full_catalog(
        model,
        train=train,
        targets=validation,
        item_buckets=item_buckets,
        cutoffs=config.cutoffs,
        batch_size=config.evaluation_batch_size,
        device=device,
        stage="validation_final",
        callback=evaluation_callback,
        max_users=config.max_eval_users,
    )
    test_result = None
    if config.test_after_selection:
        test_result = evaluate_full_catalog(
            model,
            train=train,
            targets=test,
            item_buckets=item_buckets,
            cutoffs=config.cutoffs,
            batch_size=config.evaluation_batch_size,
            device=device,
            stage="test_once",
            callback=evaluation_callback,
            max_users=config.max_eval_users,
        )

    summary = {
        "stage": "complete",
        "variant": config.variant,
        "seed": config.seed,
        "best_epoch": best_epoch,
        "epochs_ran": epochs_ran,
        "elapsed_seconds": time.monotonic() - started,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "device": str(device),
        "config": _config_payload(config),
        "artifacts": _artifact_hashes(config.processed_dir),
        "bucket_definition": bucket_definition,
        "validation": validation_result,
        "test": test_result,
        "protocol": {
            "selection": "validation Recall@20 only",
            "ranking": "full catalog",
            "masked_during_validation_and_test": "training interactions only",
            "test_access": (
                "once after best checkpoint selection"
                if config.test_after_selection
                else "not accessed"
            ),
            "semantic_asset_status": "temporally_unverified",
            "reproduction_type": "CPU structure reproduction with joint PCA64",
            "official_code_alignment_weight": config.alignment_weight,
        },
    }
    (config.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
