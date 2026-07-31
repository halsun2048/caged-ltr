from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch

from caged_ltr.distillation import (
    TextRankingGroup,
    collate_text_ranking_groups,
    load_text_ranking_groups,
    teacher_ndcg_at_k,
)


class _Tokenizer:
    def __call__(
        self,
        queries: list[str],
        passages: list[str],
        **_: object,
    ) -> dict[str, torch.Tensor]:
        assert len(queries) == len(passages)
        return {"input_ids": torch.ones((len(queries), 2), dtype=torch.int64)}


def test_load_and_collate_grouped_distillation_control(tmp_path: Path) -> None:
    candidates_path = tmp_path / "candidates.parquet"
    labels_path = tmp_path / "labels.parquet"
    pd.DataFrame(
        [
            {
                "query_id": "q1",
                "split": "train",
                "query": "query",
                "passage_id": passage_id,
                "passage": f"passage {passage_id}",
                "bm25_rank": rank,
            }
            for rank, passage_id in enumerate(("a", "b", "c"), start=1)
        ]
    ).to_parquet(candidates_path, index=False)
    pd.DataFrame(
        [
            {
                "query_id": "q1",
                "passage_id": passage_id,
                "bm25_teacher_label": label,
            }
            for passage_id, label in (("a", 3.0), ("b", 2.0), ("c", 1.0))
        ]
    ).to_parquet(labels_path, index=False)

    groups = load_text_ranking_groups(
        candidates_path,
        labels_path,
        control="bm25",
    )
    batch = collate_text_ranking_groups(_Tokenizer(), groups)

    assert groups[0].passage_ids == ("a", "b", "c")
    assert batch["labels"].tolist() == [3.0, 2.0, 1.0]
    assert batch["group_sizes"].tolist() == [3]


def test_teacher_ndcg_rewards_the_exact_teacher_order() -> None:
    labels = torch.tensor([3.0, 2.0, 1.0])
    groups = torch.tensor([3])

    exact = teacher_ndcg_at_k(labels, labels, groups)
    reverse = teacher_ndcg_at_k(torch.flip(labels, dims=(0,)), labels, groups)

    assert exact == pytest.approx(1.0)
    assert reverse < exact


def test_text_ranking_group_requires_a_strict_teacher_order() -> None:
    with pytest.raises(ValueError, match="strict total order"):
        TextRankingGroup(
            query_id="q",
            split="train",
            query="query",
            passage_ids=("a", "b"),
            passages=("first", "second"),
            labels=(1.0, 1.0),
        )
