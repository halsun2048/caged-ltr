"""Causal SASRec and a frozen-semantic late-fusion baseline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from caged_ltr.losses import bpr_loss


@dataclass(frozen=True, slots=True)
class SASRecConfig:
    num_items: int
    max_length: int = 200
    hidden_dim: int = 64
    num_blocks: int = 2
    num_heads: int = 1
    dropout: float = 0.5
    semantic_weight: float = 1.0

    def __post_init__(self) -> None:
        if min(self.num_items, self.max_length, self.hidden_dim, self.num_blocks) <= 0:
            raise ValueError("item count and model dimensions must be positive")
        if self.num_heads <= 0 or self.hidden_dim % self.num_heads != 0:
            raise ValueError("num_heads must positively divide hidden_dim")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        if self.semantic_weight < 0.0:
            raise ValueError("semantic_weight must be non-negative")


class _SASRecBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.feed_forward_norm = nn.LayerNorm(hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        padding_mask: torch.Tensor,
        causal_mask: torch.Tensor,
    ) -> torch.Tensor:
        normalized = self.attention_norm(hidden)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            attn_mask=causal_mask,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        hidden = hidden + self.attention_dropout(attended)
        hidden = hidden + self.feed_forward(self.feed_forward_norm(hidden))
        return hidden.masked_fill(padding_mask.unsqueeze(-1), 0.0)


class SASRec(nn.Module):
    """ID-only causal self-attentive sequential recommender trained with BPR."""

    def __init__(
        self,
        config: SASRecConfig,
        *,
        item_initialization: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.item_embedding = nn.Embedding(
            config.num_items + 1,
            config.hidden_dim,
            padding_idx=0,
        )
        self.position_embedding = nn.Embedding(config.max_length + 1, config.hidden_dim)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            _SASRecBlock(config.hidden_dim, config.num_heads, config.dropout)
            for _ in range(config.num_blocks)
        )
        self.final_norm = nn.LayerNorm(config.hidden_dim)
        self.reset_parameters(item_initialization)

    def reset_parameters(self, item_initialization: np.ndarray | None = None) -> None:
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.item_embedding.weight[0].zero_()
        if item_initialization is not None:
            array = np.asarray(item_initialization, dtype=np.float32)
            expected = (self.config.num_items, self.config.hidden_dim)
            if array.shape != expected or not np.isfinite(array).all():
                raise ValueError(f"item initialization must be finite with shape {expected}")
            with torch.no_grad():
                self.item_embedding.weight[1:].copy_(torch.from_numpy(array))

    def encode(self, sequences: torch.Tensor) -> torch.Tensor:
        if sequences.ndim != 2 or sequences.shape[1] != self.config.max_length:
            raise ValueError("sequences must have shape [batch, max_length]")
        if sequences.dtype != torch.long:
            raise ValueError("sequences must use torch.long item IDs")
        if ((sequences < 0) | (sequences > self.config.num_items)).any():
            raise ValueError("sequence item ID is outside the catalog")
        padding_mask = sequences.eq(0)
        if padding_mask.all(dim=1).any():
            raise ValueError("each sequence must contain at least one non-padding item")
        positions = (~padding_mask).long().cumsum(dim=1)
        hidden = self.item_embedding(sequences) * self.config.hidden_dim**0.5
        hidden = self.embedding_dropout(hidden + self.position_embedding(positions))
        hidden = hidden.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        causal_mask = torch.triu(
            torch.ones(
                self.config.max_length,
                self.config.max_length,
                dtype=torch.bool,
                device=sequences.device,
            ),
            diagonal=1,
        )
        for block in self.blocks:
            hidden = block(hidden, padding_mask=padding_mask, causal_mask=causal_mask)
        return self.final_norm(hidden).masked_fill(padding_mask.unsqueeze(-1), 0.0)

    def _collaborative_scores(
        self,
        user_states: torch.Tensor,
        candidate_items: torch.Tensor,
    ) -> torch.Tensor:
        return (user_states * self.item_embedding(candidate_items)).sum(dim=-1)

    def training_scores(
        self,
        sequences: torch.Tensor,
        positive_items: torch.Tensor,
        negative_items: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if positive_items.shape != sequences.shape or negative_items.shape != sequences.shape:
            raise ValueError("training item tensors must match sequence shape")
        states = self.encode(sequences)
        return (
            self._collaborative_scores(states, positive_items),
            self._collaborative_scores(states, negative_items),
        )

    def loss(
        self,
        sequences: torch.Tensor,
        positive_items: torch.Tensor,
        negative_items: torch.Tensor,
    ) -> torch.Tensor:
        positive_scores, negative_scores = self.training_scores(
            sequences, positive_items, negative_items
        )
        return bpr_loss(positive_scores, negative_scores, mask=positive_items.ne(0))

    def score_candidates(
        self,
        sequences: torch.Tensor,
        candidate_items: torch.Tensor,
    ) -> torch.Tensor:
        if candidate_items.ndim != 2 or candidate_items.shape[0] != sequences.shape[0]:
            raise ValueError("candidate_items must have shape [batch, candidates]")
        user_states = self.encode(sequences)[:, -1, :]
        candidates = self.item_embedding(candidate_items)
        return torch.einsum("bd,bcd->bc", user_states, candidates)


class FrozenSemanticLateFusion(SASRec):
    """SASRec collaborative score plus a frozen semantic prefix-mean score."""

    def __init__(self, config: SASRecConfig, semantic_items: np.ndarray) -> None:
        super().__init__(config)
        array = np.asarray(semantic_items, dtype=np.float32)
        if array.ndim != 2 or array.shape[0] != config.num_items:
            raise ValueError("semantic_items must have one row per catalog item")
        if not np.isfinite(array).all():
            raise ValueError("semantic_items must contain finite values")
        tensor = torch.from_numpy(array)
        tensor = torch.nn.functional.normalize(tensor, dim=-1)
        table = torch.cat((torch.zeros(1, tensor.shape[1]), tensor), dim=0)
        self.register_buffer("semantic_items", table, persistent=True)

    def _semantic_states(self, sequences: torch.Tensor) -> torch.Tensor:
        mask = sequences.ne(0).unsqueeze(-1)
        embeddings = self.semantic_items[sequences] * mask
        counts = mask.cumsum(dim=1).clamp_min(1)
        means = embeddings.cumsum(dim=1) / counts
        return torch.nn.functional.normalize(means, dim=-1)

    def _semantic_scores(
        self,
        semantic_states: torch.Tensor,
        candidate_items: torch.Tensor,
    ) -> torch.Tensor:
        return (semantic_states * self.semantic_items[candidate_items]).sum(dim=-1)

    def training_scores(
        self,
        sequences: torch.Tensor,
        positive_items: torch.Tensor,
        negative_items: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positive_scores, negative_scores = super().training_scores(
            sequences, positive_items, negative_items
        )
        semantic_states = self._semantic_states(sequences)
        weight = self.config.semantic_weight
        return (
            positive_scores
            + weight * self._semantic_scores(semantic_states, positive_items),
            negative_scores
            + weight * self._semantic_scores(semantic_states, negative_items),
        )

    def score_candidates(
        self,
        sequences: torch.Tensor,
        candidate_items: torch.Tensor,
    ) -> torch.Tensor:
        collaborative = super().score_candidates(sequences, candidate_items)
        semantic_user = self._semantic_states(sequences)[:, -1, :]
        semantic_candidates = self.semantic_items[candidate_items]
        semantic = torch.einsum("bd,bcd->bc", semantic_user, semantic_candidates)
        return collaborative + self.config.semantic_weight * semantic
