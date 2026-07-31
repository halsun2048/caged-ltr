"""Author-faithful pointwise cross-encoder student for R4."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import nn

DEFAULT_DEBERTA_V3_BASE = "microsoft/deberta-v3-base"
DEFAULT_DEBERTA_V3_BASE_REVISION = (
    "8ccc9b6f36199bec6961081d44eb72fb3f7353f3"
)


def tokenize_query_passages(
    tokenizer: Any,
    queries: Sequence[str],
    passages: Sequence[str],
    *,
    max_length: int = 500,
) -> dict[str, torch.Tensor]:
    """Tokenize query/passage pairs exactly as a pointwise cross-encoder."""
    if len(queries) != len(passages) or not queries:
        raise ValueError("queries and passages must have equal non-zero length")
    if max_length <= 0:
        raise ValueError("max_length must be positive")
    encoded = tokenizer(
        list(queries),
        list(passages),
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=max_length,
    )
    if "input_ids" not in encoded:
        raise ValueError("tokenizer output must contain input_ids")
    return dict(encoded)


class PointwiseCrossEncoder(nn.Module):
    """Single-logit DeBERTa-style cross encoder used by the author code."""

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    @classmethod
    def from_pretrained(
        cls,
        *,
        model_name: str = DEFAULT_DEBERTA_V3_BASE,
        revision: str = DEFAULT_DEBERTA_V3_BASE_REVISION,
        cache_dir: str | None = None,
    ) -> PointwiseCrossEncoder:
        """Load a pinned pretrained encoder with one scalar relevance head."""
        from transformers import AutoConfig, AutoModelForSequenceClassification

        config = AutoConfig.from_pretrained(
            model_name,
            revision=revision,
            cache_dir=cache_dir,
        )
        config.num_labels = 1
        backbone = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            revision=revision,
            cache_dir=cache_dir,
            config=config,
        )
        return cls(backbone)

    def forward(self, **encoded: torch.Tensor) -> torch.Tensor:
        if "input_ids" not in encoded:
            raise ValueError("encoded inputs must contain input_ids")
        output = self.backbone(**encoded)
        logits = output.logits
        if logits.ndim != 2 or logits.shape[1] != 1:
            raise ValueError("pointwise backbone must return logits with shape [batch, 1]")
        return logits[:, 0]
