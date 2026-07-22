"""Leakage-aware sequence datasets for the author-processed Yelp snapshot."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as parquet
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True, slots=True)
class YelpSequenceData:
    """In-memory author sequences with model IDs offset by one for padding."""

    train_histories: tuple[np.ndarray, ...]
    valid_targets: np.ndarray
    test_targets: np.ndarray
    user_indices: np.ndarray
    user_frequency_buckets: tuple[str, ...]
    user_paper_buckets: tuple[str, ...]
    item_frequency_buckets: tuple[str, ...]
    item_paper_buckets: tuple[str, ...]
    num_items: int
    fingerprint: str

    def __post_init__(self) -> None:
        users = len(self.train_histories)
        if users == 0 or any(len(history) == 0 for history in self.train_histories):
            raise ValueError("every selected user must have a non-empty training history")
        if self.valid_targets.shape != (users,) or self.test_targets.shape != (users,):
            raise ValueError("validation and test targets must align with users")
        if self.user_indices.shape != (users,):
            raise ValueError("user indices must align with histories")
        if len(self.user_frequency_buckets) != users or len(self.user_paper_buckets) != users:
            raise ValueError("user buckets must align with histories")
        if len(self.item_frequency_buckets) != self.num_items:
            raise ValueError("item frequency buckets must cover the catalog")
        if len(self.item_paper_buckets) != self.num_items:
            raise ValueError("item paper buckets must cover the catalog")
        all_ids = [*self.train_histories, self.valid_targets, self.test_targets]
        if any((values < 1).any() or (values > self.num_items).any() for values in all_ids):
            raise ValueError("model item IDs must lie in [1, num_items]")


def load_yelp_author_sequences(
    processed_dir: Path,
    *,
    report_path: Path | None = None,
    max_users: int | None = None,
) -> YelpSequenceData:
    """Load the paper-faithful author split and add one to item IDs for padding."""
    if max_users is not None and max_users <= 0:
        raise ValueError("max_users must be positive when provided")
    sequence_path = processed_dir / "sequences.parquet"
    item_path = processed_dir / "items.parquet"
    if not sequence_path.is_file() or not item_path.is_file():
        raise FileNotFoundError("run scripts/prepare_yelp_author.py before model training")

    sequence_rows = parquet.read_table(
        sequence_path,
        columns=[
            "user_idx",
            "user_frequency_bucket",
            "user_paper_bucket",
            "train_item_ids",
            "valid_item_id",
            "test_item_id",
        ],
    ).to_pylist()
    if max_users is not None:
        sequence_rows = sequence_rows[:max_users]
    item_rows = parquet.read_table(
        item_path,
        columns=["item_idx", "frequency_bucket", "paper_bucket"],
    ).to_pylist()
    if [int(row["item_idx"]) for row in item_rows] != list(range(len(item_rows))):
        raise ValueError("item_idx must be contiguous and zero-based")
    user_indices = np.asarray([int(row["user_idx"]) for row in sequence_rows], dtype=np.int64)
    if not np.array_equal(user_indices, np.sort(user_indices)):
        raise ValueError("sequences must be ordered by user_idx")

    report = report_path or processed_dir / "manifest.json"
    manifest = json.loads(report.read_text(encoding="utf-8"))
    return YelpSequenceData(
        train_histories=tuple(
            np.asarray(row["train_item_ids"], dtype=np.int64) + 1 for row in sequence_rows
        ),
        valid_targets=np.asarray(
            [int(row["valid_item_id"]) + 1 for row in sequence_rows], dtype=np.int64
        ),
        test_targets=np.asarray(
            [int(row["test_item_id"]) + 1 for row in sequence_rows], dtype=np.int64
        ),
        user_indices=user_indices,
        user_frequency_buckets=tuple(
            str(row["user_frequency_bucket"]) for row in sequence_rows
        ),
        user_paper_buckets=tuple(str(row["user_paper_bucket"]) for row in sequence_rows),
        item_frequency_buckets=tuple(str(row["frequency_bucket"]) for row in item_rows),
        item_paper_buckets=tuple(str(row["paper_bucket"]) for row in item_rows),
        num_items=len(item_rows),
        fingerprint=str(manifest["processed_fingerprint"]),
    )


def _left_pad(values: np.ndarray, max_length: int) -> np.ndarray:
    output = np.zeros(max_length, dtype=np.int64)
    selected = values[-max_length:]
    if len(selected):
        output[-len(selected) :] = selected
    return output


def _sample_negative(
    generator: np.random.Generator,
    num_items: int,
    forbidden: set[int],
    used: set[int] | None = None,
) -> int:
    if len(forbidden) >= num_items:
        raise ValueError("no negative item remains after excluding known interactions")
    for _ in range(num_items * 2):
        candidate = int(generator.integers(1, num_items + 1))
        if candidate not in forbidden and (used is None or candidate not in used):
            return candidate
    available = [
        item
        for item in range(1, num_items + 1)
        if item not in forbidden and (used is None or item not in used)
    ]
    if not available:
        raise ValueError("not enough unique negative items")
    return available[int(generator.integers(0, len(available)))]


class SASRecTrainingDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """One seq-to-seq training example per user with deterministic negatives."""

    def __init__(self, data: YelpSequenceData, *, max_length: int, seed: int) -> None:
        if max_length <= 0 or seed < 0:
            raise ValueError("max_length must be positive and seed must be non-negative")
        self.data = data
        self.max_length = max_length
        self.seed = seed
        self.epoch = 0
        self.indices = [
            index for index, history in enumerate(data.train_histories) if len(history) >= 2
        ]
        if not self.indices:
            raise ValueError("at least one user needs two training interactions")

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        user_index = self.indices[index]
        history = self.data.train_histories[user_index]
        sequence = _left_pad(history[:-1], self.max_length)
        positives = _left_pad(history[1:], self.max_length)
        known = {
            *history.tolist(),
            int(self.data.valid_targets[user_index]),
            int(self.data.test_targets[user_index]),
        }
        generator = np.random.default_rng(
            np.random.SeedSequence([self.seed, self.epoch, user_index])
        )
        negatives = np.zeros(self.max_length, dtype=np.int64)
        for position in np.flatnonzero(positives):
            negatives[position] = _sample_negative(generator, self.data.num_items, known)
        return (
            torch.from_numpy(sequence),
            torch.from_numpy(positives),
            torch.from_numpy(negatives),
        )


class SASRecEvaluationDataset(
    Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]
):
    """Deterministic sampled-negative validation or test examples."""

    def __init__(
        self,
        data: YelpSequenceData,
        *,
        split: str,
        max_length: int,
        num_negatives: int,
        seed: int,
        max_users: int | None = None,
    ) -> None:
        if split not in {"valid", "test"}:
            raise ValueError("split must be 'valid' or 'test'")
        if max_length <= 0 or num_negatives <= 0 or seed < 0:
            raise ValueError("lengths must be positive and seed must be non-negative")
        self.data = data
        self.split = split
        self.max_length = max_length
        self.num_negatives = num_negatives
        self.seed = seed
        self.size = min(len(data.train_histories), max_users or len(data.train_histories))

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        train = self.data.train_histories[index]
        if self.split == "valid":
            history = train
            target = int(self.data.valid_targets[index])
            split_offset = 0
        else:
            history = np.append(train, self.data.valid_targets[index])
            target = int(self.data.test_targets[index])
            split_offset = 1
        known = {
            *train.tolist(),
            int(self.data.valid_targets[index]),
            int(self.data.test_targets[index]),
        }
        generator = np.random.default_rng(
            np.random.SeedSequence([self.seed, 10_000 + split_offset, index])
        )
        negatives: list[int] = []
        used: set[int] = set()
        for _ in range(self.num_negatives):
            negative = _sample_negative(generator, self.data.num_items, known, used)
            negatives.append(negative)
            used.add(negative)
        candidates = np.asarray([target, *negatives], dtype=np.int64)
        return (
            torch.from_numpy(_left_pad(history, self.max_length)),
            torch.from_numpy(candidates),
            torch.tensor(index, dtype=torch.int64),
            torch.tensor(target, dtype=torch.int64),
        )
