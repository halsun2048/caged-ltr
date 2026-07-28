"""Leakage-safe dual-view sequential recommender based on LLM-ESR."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from caged_ltr.losses import bpr_loss
from caged_ltr.models.sasrec import SASRecConfig, _SASRecBlock


class _CausalCrossAttention(nn.Module):
    """Cross attention with both padding and future-position masking."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("cross-attention heads must divide hidden_dim")
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, length, _ = tensor.shape
        return tensor.reshape(batch, length, self.num_heads, self.head_dim).transpose(
            1, 2
        )

    def forward(
        self,
        query_states: torch.Tensor,
        key_value_states: torch.Tensor,
        *,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        query = self._split_heads(self.query(query_states))
        key = self._split_heads(self.key(key_value_states))
        value = self._split_heads(self.value(key_value_states))
        scores = query @ key.transpose(-2, -1) / math.sqrt(self.head_dim)
        length = query_states.shape[1]
        future_mask = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=scores.device),
            diagonal=1,
        )
        blocked = future_mask[None, None] | padding_mask[:, None, None, :]
        scores = scores.masked_fill(blocked, -1e9)
        attention = self.dropout(torch.softmax(scores, dim=-1))
        attended = attention @ value
        attended = attended.transpose(1, 2).contiguous().reshape(
            query_states.shape[0], length, -1
        )
        return self.output(attended).masked_fill(padding_mask.unsqueeze(-1), 0.0)


class _CausalSequenceEncoder(nn.Module):
    def __init__(self, config: SASRecConfig) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            _SASRecBlock(config.hidden_dim, config.num_heads, config.dropout)
            for _ in range(config.num_blocks)
        )
        self.final_norm = nn.LayerNorm(config.hidden_dim)

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        length = hidden.shape[1]
        causal_mask = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=hidden.device),
            diagonal=1,
        )
        for block in self.blocks:
            hidden = block(
                hidden,
                padding_mask=padding_mask,
                causal_mask=causal_mask,
            )
        return self.final_norm(hidden).masked_fill(padding_mask.unsqueeze(-1), 0.0)


class _PositionwiseCapacityControl(nn.Module):
    """No-cross-token control with the same parameter count as one CA direction."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.output = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, hidden: torch.Tensor, *, padding_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.output(self.value(self.key(self.query(hidden))))
        return hidden.masked_fill(padding_mask.unsqueeze(-1), 0.0)


class DualViewSASRec(nn.Module):
    """Frozen semantic and trainable collaborative views with optional cross attention."""

    def __init__(
        self,
        config: SASRecConfig,
        raw_semantic_items: np.ndarray,
        collaborative_initialization: np.ndarray,
        *,
        use_cross_attention: bool = True,
        share_encoder: bool = True,
        capacity_control: bool = False,
        cross_attention_heads: int = 2,
    ) -> None:
        super().__init__()
        if use_cross_attention and capacity_control:
            raise ValueError("cross attention and capacity control are mutually exclusive")
        self.config = config
        self.use_cross_attention = use_cross_attention
        self.share_encoder = share_encoder
        self.capacity_control = capacity_control

        raw = np.array(raw_semantic_items, dtype=np.float32, copy=True)
        collaborative = np.asarray(collaborative_initialization, dtype=np.float32)
        if (
            raw.ndim != 2
            or raw.shape[0] != config.num_items
            or not np.isfinite(raw).all()
        ):
            raise ValueError("raw semantics must be finite with one row per item")
        expected_collaborative = (config.num_items, config.hidden_dim)
        if collaborative.shape != expected_collaborative or not np.isfinite(
            collaborative
        ).all():
            raise ValueError(
                "collaborative initialization must be finite with shape "
                f"{expected_collaborative}"
            )

        raw_with_padding = torch.cat(
            (torch.zeros(1, raw.shape[1]), torch.from_numpy(raw)),
            dim=0,
        )
        self.semantic_items = nn.Embedding.from_pretrained(
            raw_with_padding,
            freeze=True,
            padding_idx=0,
        )
        adapter_dim = max(1, raw.shape[1] // 2)
        self.semantic_adapter = nn.Sequential(
            nn.Linear(raw.shape[1], adapter_dim),
            nn.Linear(adapter_dim, config.hidden_dim),
        )
        self.collaborative_items = nn.Embedding(
            config.num_items + 1,
            config.hidden_dim,
            padding_idx=0,
        )
        with torch.no_grad():
            self.collaborative_items.weight.zero_()
            self.collaborative_items.weight[1:].copy_(
                torch.from_numpy(collaborative)
            )

        self.position_embedding = nn.Embedding(config.max_length + 1, config.hidden_dim)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        self.embedding_dropout = nn.Dropout(config.dropout)

        self.semantic_encoder = _CausalSequenceEncoder(config)
        self._collaborative_encoder = (
            None if share_encoder else _CausalSequenceEncoder(config)
        )
        if use_cross_attention:
            self.semantic_from_collaborative = _CausalCrossAttention(
                config.hidden_dim, cross_attention_heads, config.dropout
            )
            self.collaborative_from_semantic = _CausalCrossAttention(
                config.hidden_dim, cross_attention_heads, config.dropout
            )
        if capacity_control:
            self.semantic_capacity = _PositionwiseCapacityControl(config.hidden_dim)
            self.collaborative_capacity = _PositionwiseCapacityControl(config.hidden_dim)

    @property
    def collaborative_encoder(self) -> _CausalSequenceEncoder:
        if self._collaborative_encoder is None:
            return self.semantic_encoder
        return self._collaborative_encoder

    @property
    def frozen_semantic_values(self) -> int:
        return self.semantic_items.weight.numel()

    def _validate_sequences(self, sequences: torch.Tensor) -> torch.Tensor:
        if sequences.ndim != 2 or sequences.shape[1] != self.config.max_length:
            raise ValueError("sequences must have shape [batch, max_length]")
        if sequences.dtype != torch.long:
            raise ValueError("sequences must use torch.long item IDs")
        if ((sequences < 0) | (sequences > self.config.num_items)).any():
            raise ValueError("sequence item ID is outside the catalog")
        padding_mask = sequences.eq(0)
        if padding_mask.all(dim=1).any():
            raise ValueError("each sequence must contain at least one non-padding item")
        return padding_mask

    def encode_views(
        self, sequences: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        padding_mask = self._validate_sequences(sequences)
        positions = (~padding_mask).long().cumsum(dim=1)
        position = self.position_embedding(positions)
        semantic = self.semantic_adapter(self.semantic_items(sequences))
        collaborative = self.collaborative_items(sequences)
        scale = self.config.hidden_dim**0.5
        semantic = self.embedding_dropout(semantic * scale + position)
        collaborative = self.embedding_dropout(collaborative * scale + position)
        semantic = semantic.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        collaborative = collaborative.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        if self.use_cross_attention:
            semantic, collaborative = (
                self.semantic_from_collaborative(
                    semantic,
                    collaborative,
                    padding_mask=padding_mask,
                ),
                self.collaborative_from_semantic(
                    collaborative,
                    semantic,
                    padding_mask=padding_mask,
                ),
            )
        elif self.capacity_control:
            semantic = self.semantic_capacity(semantic, padding_mask=padding_mask)
            collaborative = self.collaborative_capacity(
                collaborative, padding_mask=padding_mask
            )

        return (
            self.semantic_encoder(semantic, padding_mask=padding_mask),
            self.collaborative_encoder(collaborative, padding_mask=padding_mask),
        )

    def _item_views(
        self, item_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.semantic_adapter(self.semantic_items(item_ids)),
            self.collaborative_items(item_ids),
        )

    def training_scores(
        self,
        sequences: torch.Tensor,
        positive_items: torch.Tensor,
        negative_items: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if positive_items.shape != sequences.shape or negative_items.shape != sequences.shape:
            raise ValueError("training item tensors must match sequence shape")
        semantic_states, collaborative_states = self.encode_views(sequences)
        positive_semantic, positive_collaborative = self._item_views(positive_items)
        negative_semantic, negative_collaborative = self._item_views(negative_items)
        return (
            (semantic_states * positive_semantic).sum(dim=-1)
            + (collaborative_states * positive_collaborative).sum(dim=-1),
            (semantic_states * negative_semantic).sum(dim=-1)
            + (collaborative_states * negative_collaborative).sum(dim=-1),
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
        semantic_states, collaborative_states = self.encode_views(sequences)
        semantic_items, collaborative_items = self._item_views(candidate_items)
        return torch.einsum(
            "bd,bcd->bc", semantic_states[:, -1], semantic_items
        ) + torch.einsum(
            "bd,bcd->bc", collaborative_states[:, -1], collaborative_items
        )

    def score_catalog(self, sequences: torch.Tensor) -> torch.Tensor:
        semantic_states, collaborative_states = self.encode_views(sequences)
        item_ids = torch.arange(
            1,
            self.config.num_items + 1,
            dtype=torch.long,
            device=sequences.device,
        )
        semantic_items, collaborative_items = self._item_views(item_ids)
        return (
            semantic_states[:, -1] @ semantic_items.transpose(0, 1)
            + collaborative_states[:, -1] @ collaborative_items.transpose(0, 1)
        )


class FrozenRawSemanticSASRec(nn.Module):
    """Trainable causal semantic route without a collaborative item-ID route."""

    def __init__(
        self,
        config: SASRecConfig,
        raw_semantic_items: np.ndarray,
    ) -> None:
        super().__init__()
        self.config = config
        raw = np.array(raw_semantic_items, dtype=np.float32, copy=True)
        if (
            raw.ndim != 2
            or raw.shape[0] != config.num_items
            or not np.isfinite(raw).all()
        ):
            raise ValueError("raw semantics must be finite with one row per item")
        table = torch.cat(
            (torch.zeros(1, raw.shape[1]), torch.from_numpy(raw)),
            dim=0,
        )
        self.semantic_items = nn.Embedding.from_pretrained(
            table,
            freeze=True,
            padding_idx=0,
        )
        adapter_dim = max(1, raw.shape[1] // 2)
        self.semantic_adapter = nn.Sequential(
            nn.Linear(raw.shape[1], adapter_dim),
            nn.Linear(adapter_dim, config.hidden_dim),
        )
        self.position_embedding = nn.Embedding(config.max_length + 1, config.hidden_dim)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.encoder = _CausalSequenceEncoder(config)

    @property
    def frozen_semantic_values(self) -> int:
        return self.semantic_items.weight.numel()

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
        hidden = (
            self.semantic_adapter(self.semantic_items(sequences))
            * self.config.hidden_dim**0.5
            + self.position_embedding(positions)
        )
        hidden = self.embedding_dropout(hidden)
        hidden = hidden.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return self.encoder(hidden, padding_mask=padding_mask)

    def _scores(
        self,
        states: torch.Tensor,
        item_ids: torch.Tensor,
    ) -> torch.Tensor:
        items = self.semantic_adapter(self.semantic_items(item_ids))
        return (states * items).sum(dim=-1)

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
            self._scores(states, positive_items),
            self._scores(states, negative_items),
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
        states = self.encode(sequences)[:, -1]
        candidates = self.semantic_adapter(self.semantic_items(candidate_items))
        return torch.einsum("bd,bcd->bc", states, candidates)

    def score_catalog(self, sequences: torch.Tensor) -> torch.Tensor:
        states = self.encode(sequences)[:, -1]
        item_ids = torch.arange(
            1,
            self.config.num_items + 1,
            dtype=torch.long,
            device=sequences.device,
        )
        items = self.semantic_adapter(self.semantic_items(item_ids))
        return states @ items.transpose(0, 1)
