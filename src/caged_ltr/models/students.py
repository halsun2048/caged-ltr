"""Local CPU-friendly baseline students required by R0."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from torch import nn


def _mlp(input_dim: int, hidden_dims: Sequence[int], dropout: float) -> nn.Sequential:
    if input_dim <= 0 or not hidden_dims or any(dimension <= 0 for dimension in hidden_dims):
        raise ValueError("input and hidden dimensions must be positive")
    if not 0.0 <= dropout < 1.0:
        raise ValueError("dropout must lie in [0, 1)")
    layers: list[nn.Module] = []
    previous = input_dim
    for dimension in hidden_dims:
        layers.extend((nn.Linear(previous, dimension), nn.ReLU(), nn.Dropout(dropout)))
        previous = dimension
    return nn.Sequential(*layers)


class MLPStudent(nn.Module):
    """Pointwise multilayer perceptron baseline returning one logit per candidate."""

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dims: Sequence[int] = (128, 64),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.backbone = _mlp(input_dim, hidden_dims, dropout)
        self.output = nn.Linear(hidden_dims[-1], 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2:
            raise ValueError("features must have shape [num_candidates, input_dim]")
        return self.output(self.backbone(features)).squeeze(-1)


class CrossLayerV2(nn.Module):
    """Full-rank DCN-v2 cross layer: x_(l+1) = x0 * (W xl + b) + xl."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, input_dim)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x0: torch.Tensor, xl: torch.Tensor) -> torch.Tensor:
        return x0 * self.linear(xl) + xl


class DCNv2Student(nn.Module):
    """Stacked cross network and deep tower joined by a pointwise output head."""

    def __init__(
        self,
        input_dim: int,
        *,
        num_cross_layers: int = 3,
        deep_dims: Sequence[int] = (128, 64),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_cross_layers <= 0:
            raise ValueError("num_cross_layers must be positive")
        self.cross_layers = nn.ModuleList(
            CrossLayerV2(input_dim) for _ in range(num_cross_layers)
        )
        self.deep = _mlp(input_dim, deep_dims, dropout)
        self.output = nn.Linear(input_dim + deep_dims[-1], 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2:
            raise ValueError("features must have shape [num_candidates, input_dim]")
        crossed = features
        for layer in self.cross_layers:
            crossed = layer(features, crossed)
        deep = self.deep(features)
        return self.output(torch.cat((crossed, deep), dim=-1)).squeeze(-1)


class LambdaMARTRanker:
    """Deterministic LightGBM LambdaMART wrapper with explicit query groups."""

    def __init__(
        self,
        *,
        seed: int = 42,
        n_estimators: int = 100,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        min_child_samples: int = 10,
        n_jobs: int = 1,
    ) -> None:
        from lightgbm import LGBMRanker

        self.model: Any = LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            random_state=seed,
            data_random_seed=seed,
            feature_fraction_seed=seed,
            bagging_seed=seed,
            deterministic=True,
            force_col_wise=True,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            min_child_samples=min_child_samples,
            n_jobs=n_jobs,
            verbosity=-1,
        )

    @staticmethod
    def _validate(
        features: np.ndarray,
        labels: np.ndarray,
        group_sizes: Sequence[int],
    ) -> tuple[np.ndarray, np.ndarray, list[int]]:
        feature_array = np.asarray(features, dtype=np.float32)
        label_array = np.asarray(labels, dtype=np.float32)
        sizes = [int(size) for size in group_sizes]
        if feature_array.ndim != 2 or label_array.shape != (feature_array.shape[0],):
            raise ValueError("features and labels have incompatible shapes")
        if any(size <= 0 for size in sizes) or sum(sizes) != feature_array.shape[0]:
            raise ValueError("group_sizes must be positive and cover every candidate")
        return feature_array, label_array, sizes

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        group_sizes: Sequence[int],
    ) -> LambdaMARTRanker:
        feature_array, label_array, sizes = self._validate(features, labels, group_sizes)
        self.model.fit(feature_array, label_array, group=sizes)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        feature_array = np.asarray(features, dtype=np.float32)
        if feature_array.ndim != 2:
            raise ValueError("features must be two-dimensional")
        return np.asarray(self.model.booster_.predict(feature_array), dtype=np.float64)
