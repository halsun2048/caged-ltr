"""Canonical request-level candidate-list schema used throughout the project."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(slots=True)
class CandidateList:
    """One request and all candidates considered by a ranking model.

    Features and labels are request-time values. Callers are responsible for enforcing
    temporal cutoffs before constructing this object.
    """

    request_id: str
    candidate_ids: Sequence[str]
    features: np.ndarray
    labels: np.ndarray
    user_id: str | None = None
    query_frequency: int | None = None
    candidate_frequencies: np.ndarray | None = None
    user_frequency: int | None = None

    def __post_init__(self) -> None:
        self.candidate_ids = tuple(str(item) for item in self.candidate_ids)
        self.features = np.asarray(self.features, dtype=np.float32)
        self.labels = np.asarray(self.labels, dtype=np.float32)
        if self.candidate_frequencies is not None:
            self.candidate_frequencies = np.asarray(self.candidate_frequencies, dtype=np.int64)

        size = len(self.candidate_ids)
        if not self.request_id:
            raise ValueError("request_id must not be empty")
        if size == 0:
            raise ValueError("a candidate list must contain at least one candidate")
        if len(set(self.candidate_ids)) != size:
            raise ValueError("candidate_ids must be unique within a request")
        if self.features.ndim != 2 or self.features.shape[0] != size:
            raise ValueError("features must have shape [num_candidates, num_features]")
        if self.labels.shape != (size,):
            raise ValueError("labels must have shape [num_candidates]")
        if not np.isfinite(self.features).all() or not np.isfinite(self.labels).all():
            raise ValueError("features and labels must contain only finite values")
        if self.candidate_frequencies is not None and self.candidate_frequencies.shape != (size,):
            raise ValueError("candidate_frequencies must have shape [num_candidates]")
        if self.query_frequency is not None and self.query_frequency < 0:
            raise ValueError("query_frequency must be non-negative")
        if self.user_frequency is not None and self.user_frequency < 0:
            raise ValueError("user_frequency must be non-negative")
        if self.candidate_frequencies is not None and (self.candidate_frequencies < 0).any():
            raise ValueError("candidate_frequencies must be non-negative")


@dataclass(slots=True)
class CandidateBatch:
    """Flattened candidates plus request group boundaries."""

    features: torch.Tensor
    labels: torch.Tensor
    group_sizes: torch.Tensor
    request_ids: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    user_ids: tuple[str | None, ...]
    query_frequencies: torch.Tensor
    candidate_frequencies: torch.Tensor
    user_frequencies: torch.Tensor

    def __post_init__(self) -> None:
        candidate_count = int(self.group_sizes.sum().item())
        group_count = int(self.group_sizes.numel())
        if self.features.ndim != 2 or self.features.shape[0] != candidate_count:
            raise ValueError("features do not match group_sizes")
        if self.labels.shape != (candidate_count,):
            raise ValueError("labels do not match group_sizes")
        if len(self.candidate_ids) != candidate_count:
            raise ValueError("candidate_ids do not match group_sizes")
        if len(self.request_ids) != group_count or len(self.user_ids) != group_count:
            raise ValueError("request-level identifiers do not match group_sizes")
        if self.query_frequencies.shape != (group_count,):
            raise ValueError("query_frequencies do not match group_sizes")
        if self.user_frequencies.shape != (group_count,):
            raise ValueError("user_frequencies do not match group_sizes")
        if self.candidate_frequencies.shape != (candidate_count,):
            raise ValueError("candidate_frequencies do not match group_sizes")

    def group_slices(self) -> Iterator[slice]:
        """Yield slices into flattened candidate tensors for every request."""
        start = 0
        for size in self.group_sizes.tolist():
            end = start + int(size)
            yield slice(start, end)
            start = end

    def to(self, device: torch.device | str) -> CandidateBatch:
        """Move tensor fields while retaining request and candidate identifiers."""
        return CandidateBatch(
            features=self.features.to(device),
            labels=self.labels.to(device),
            group_sizes=self.group_sizes.to(device),
            request_ids=self.request_ids,
            candidate_ids=self.candidate_ids,
            user_ids=self.user_ids,
            query_frequencies=self.query_frequencies.to(device),
            candidate_frequencies=self.candidate_frequencies.to(device),
            user_frequencies=self.user_frequencies.to(device),
        )


class CandidateDataset(Dataset[CandidateList]):
    """Thin request-level dataset that keeps candidate lists intact."""

    def __init__(self, examples: Sequence[CandidateList]) -> None:
        if not examples:
            raise ValueError("examples must not be empty")
        feature_dims = {example.features.shape[1] for example in examples}
        if len(feature_dims) != 1:
            raise ValueError("all examples must use the same feature dimension")
        request_ids = [example.request_id for example in examples]
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("request_ids must be unique")
        self._examples = tuple(examples)

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, index: int) -> CandidateList:
        return self._examples[index]


def collate_candidate_lists(examples: Sequence[CandidateList]) -> CandidateBatch:
    """Collate complete candidate lists without padding or losing group boundaries."""
    if not examples:
        raise ValueError("cannot collate an empty batch")
    feature_dims = {example.features.shape[1] for example in examples}
    if len(feature_dims) != 1:
        raise ValueError("all examples in a batch must use the same feature dimension")

    group_sizes = [len(example.candidate_ids) for example in examples]
    candidate_frequencies = [
        example.candidate_frequencies
        if example.candidate_frequencies is not None
        else np.full(group_sizes[index], -1, dtype=np.int64)
        for index, example in enumerate(examples)
    ]
    return CandidateBatch(
        features=torch.from_numpy(np.concatenate([item.features for item in examples])),
        labels=torch.from_numpy(np.concatenate([item.labels for item in examples])),
        group_sizes=torch.tensor(group_sizes, dtype=torch.int64),
        request_ids=tuple(item.request_id for item in examples),
        candidate_ids=tuple(
            candidate_id for item in examples for candidate_id in item.candidate_ids
        ),
        user_ids=tuple(item.user_id for item in examples),
        query_frequencies=torch.tensor(
            [item.query_frequency if item.query_frequency is not None else -1 for item in examples],
            dtype=torch.int64,
        ),
        candidate_frequencies=torch.from_numpy(np.concatenate(candidate_frequencies)),
        user_frequencies=torch.tensor(
            [item.user_frequency if item.user_frequency is not None else -1 for item in examples],
            dtype=torch.int64,
        ),
    )
