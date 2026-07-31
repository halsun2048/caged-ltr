from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from caged_ltr.models.pointwise import PointwiseCrossEncoder, tokenize_query_passages


class _Backbone(nn.Module):
    def forward(self, *, input_ids: torch.Tensor, **_: torch.Tensor) -> SimpleNamespace:
        return SimpleNamespace(logits=input_ids.float().sum(dim=1, keepdim=True))


class _Tokenizer:
    def __call__(
        self,
        queries: list[str],
        passages: list[str],
        **kwargs: object,
    ) -> dict[str, torch.Tensor]:
        assert queries == ["q1", "q2"]
        assert passages == ["p1", "p2"]
        assert kwargs == {
            "padding": True,
            "truncation": True,
            "return_tensors": "pt",
            "max_length": 500,
        }
        return {"input_ids": torch.tensor([[1, 2], [3, 4]])}


def test_pointwise_cross_encoder_returns_one_score_per_pair() -> None:
    model = PointwiseCrossEncoder(_Backbone())

    scores = model(input_ids=torch.tensor([[1, 2], [3, 4]]))

    assert scores.tolist() == [3.0, 7.0]


def test_pointwise_tokenization_keeps_query_passage_pairs() -> None:
    encoded = tokenize_query_passages(_Tokenizer(), ["q1", "q2"], ["p1", "p2"])

    assert encoded["input_ids"].shape == (2, 2)


def test_pointwise_tokenization_rejects_unpaired_inputs() -> None:
    with pytest.raises(ValueError, match="equal non-zero"):
        tokenize_query_passages(_Tokenizer(), ["q1"], [])
