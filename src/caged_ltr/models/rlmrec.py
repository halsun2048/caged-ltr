"""CPU-capable LightGCN and RLMRec-Con structure reproduction."""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

RLMRecVariant = Literal["lightgcn", "semantic_only", "rlmrec_con", "shuffled_con"]


def _chunked_infonce(
    anchors: torch.Tensor,
    positives: torch.Tensor,
    candidates: torch.Tensor,
    *,
    temperature: float,
    chunk_size: int,
) -> torch.Tensor:
    anchors = F.normalize(anchors, dim=-1)
    positives = F.normalize(positives, dim=-1)
    candidates = F.normalize(candidates, dim=-1)
    numerator = (anchors * positives).sum(dim=-1) / temperature
    denominator: torch.Tensor | None = None
    for chunk in candidates.split(chunk_size):
        chunk_lse = torch.logsumexp(anchors @ chunk.T / temperature, dim=1)
        denominator = (
            chunk_lse
            if denominator is None
            else torch.logaddexp(denominator, chunk_lse)
        )
    if denominator is None:
        raise ValueError("InfoNCE candidate set must not be empty")
    return (denominator - numerator).mean()


class RLMRecLightGCN(nn.Module):
    """LightGCN with optional frozen-profile contrastive alignment."""

    def __init__(
        self,
        *,
        num_users: int,
        num_items: int,
        adjacency: torch.Tensor,
        variant: RLMRecVariant,
        embedding_dim: int = 32,
        layer_count: int = 3,
        keep_rate: float = 0.8,
        regularization_weight: float = 1e-6,
        alignment_weight: float = 1e-2,
        temperature: float = 0.2,
        user_semantics: torch.Tensor | None = None,
        item_semantics: torch.Tensor | None = None,
        contrastive_chunk_size: int = 2048,
    ) -> None:
        super().__init__()
        if variant not in {
            "lightgcn",
            "semantic_only",
            "rlmrec_con",
            "shuffled_con",
        }:
            raise ValueError(f"unsupported RLMRec variant: {variant}")
        if layer_count < 0 or embedding_dim <= 0:
            raise ValueError("invalid LightGCN dimensions")
        if not 0 < keep_rate <= 1:
            raise ValueError("keep_rate must be in (0, 1]")
        self.num_users = num_users
        self.num_items = num_items
        self.variant = variant
        self.embedding_dim = embedding_dim
        self.layer_count = layer_count
        self.keep_rate = keep_rate
        self.regularization_weight = regularization_weight
        self.alignment_weight = alignment_weight
        self.temperature = temperature
        self.contrastive_chunk_size = contrastive_chunk_size
        self.register_buffer("adjacency", adjacency.coalesce(), persistent=False)

        if variant != "semantic_only":
            self.user_embeddings = nn.Parameter(torch.empty(num_users, embedding_dim))
            self.item_embeddings = nn.Parameter(torch.empty(num_items, embedding_dim))
            nn.init.xavier_uniform_(self.user_embeddings)
            nn.init.xavier_uniform_(self.item_embeddings)
        else:
            self.register_parameter("user_embeddings", None)
            self.register_parameter("item_embeddings", None)

        if variant != "lightgcn":
            if user_semantics is None or item_semantics is None:
                raise ValueError(f"{variant} requires user and item semantics")
            if user_semantics.shape[0] != num_users:
                raise ValueError("user semantic row count mismatch")
            if item_semantics.shape[0] != num_items:
                raise ValueError("item semantic row count mismatch")
            if user_semantics.shape[1] != item_semantics.shape[1]:
                raise ValueError("semantic dimensions differ")
            self.register_buffer(
                "user_semantics", user_semantics.float(), persistent=False
            )
            self.register_buffer(
                "item_semantics", item_semantics.float(), persistent=False
            )
            semantic_dim = int(user_semantics.shape[1])
            hidden_dim = (semantic_dim + embedding_dim) // 2
            self.semantic_mlp = nn.Sequential(
                nn.Linear(semantic_dim, hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(hidden_dim, embedding_dim),
            )
            for module in self.semantic_mlp:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
        else:
            self.register_buffer("user_semantics", None, persistent=False)
            self.register_buffer("item_semantics", None, persistent=False)
            self.semantic_mlp = None

    def _drop_edges(self) -> torch.Tensor:
        if not self.training or self.keep_rate == 1.0:
            return self.adjacency
        values = self.adjacency.values()
        mask = torch.rand_like(values).add(self.keep_rate).floor().bool()
        return torch.sparse_coo_tensor(
            self.adjacency.indices()[:, mask],
            values[mask],
            self.adjacency.shape,
            device=values.device,
        ).coalesce()

    def collaborative_embeddings(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.user_embeddings is None or self.item_embeddings is None:
            raise RuntimeError("semantic-only model has no collaborative embeddings")
        embeddings = torch.cat([self.user_embeddings, self.item_embeddings], dim=0)
        layers = [embeddings]
        adjacency = self._drop_edges()
        for _ in range(self.layer_count):
            embeddings = torch.sparse.mm(adjacency, embeddings)
            layers.append(embeddings)
        combined = torch.stack(layers, dim=0).sum(dim=0)
        return combined[: self.num_users], combined[self.num_users :]

    def semantic_embeddings(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.semantic_mlp is None:
            raise RuntimeError("LightGCN baseline has no semantic projection")
        return (
            self.semantic_mlp(self.user_semantics),
            self.semantic_mlp(self.item_semantics),
        )

    def ranking_embeddings(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.variant == "semantic_only":
            return self.semantic_embeddings()
        return self.collaborative_embeddings()

    def loss(
        self,
        users: torch.Tensor,
        positives: torch.Tensor,
        negatives: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        user_embeddings, item_embeddings = self.ranking_embeddings()
        anchor = user_embeddings[users]
        positive = item_embeddings[positives]
        negative = item_embeddings[negatives]
        bpr = F.softplus(
            (anchor * negative).sum(dim=-1) - (anchor * positive).sum(dim=-1)
        ).mean()
        regularization = self.regularization_weight * sum(
            parameter.square().sum() for parameter in self.parameters()
        )
        alignment = torch.zeros((), device=bpr.device)
        if self.variant in {"rlmrec_con", "shuffled_con"}:
            semantic_users, semantic_items = self.semantic_embeddings()
            alignment = _chunked_infonce(
                anchor,
                semantic_users[users],
                semantic_users,
                temperature=self.temperature,
                chunk_size=self.contrastive_chunk_size,
            )
            alignment = alignment + _chunked_infonce(
                positive,
                semantic_items[positives],
                semantic_items[positives],
                temperature=self.temperature,
                chunk_size=self.contrastive_chunk_size,
            )
            alignment = alignment + _chunked_infonce(
                negative,
                semantic_items[negatives],
                semantic_items[negatives],
                temperature=self.temperature,
                chunk_size=self.contrastive_chunk_size,
            )
        total = bpr + regularization + self.alignment_weight * alignment
        return total, {
            "bpr": float(bpr.detach()),
            "regularization": float(regularization.detach()),
            "alignment_unweighted": float(alignment.detach()),
        }
