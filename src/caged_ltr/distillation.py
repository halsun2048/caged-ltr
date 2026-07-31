"""Grouped text-ranking utilities for instruction-distillation students."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import torch

from caged_ltr.models.pointwise import tokenize_query_passages

DistillationControl = Literal["bm25", "random", "prp"]


@dataclass(frozen=True, slots=True)
class TextRankingGroup:
    """One query and its complete text candidate list."""

    query_id: str
    split: str
    query: str
    passage_ids: tuple[str, ...]
    passages: tuple[str, ...]
    labels: tuple[float, ...]

    def __post_init__(self) -> None:
        size = len(self.passage_ids)
        if not self.query_id or not self.query or size <= 1:
            raise ValueError("ranking group must have a query and at least two candidates")
        if self.split not in {"train", "validation"}:
            raise ValueError("ranking group split must be train or validation")
        if len(self.passages) != size or len(self.labels) != size:
            raise ValueError("candidate IDs, passages, and labels must have equal length")
        if len(set(self.passage_ids)) != size:
            raise ValueError("passage IDs must be unique within a ranking group")
        if len(set(self.labels)) != size:
            raise ValueError("distillation labels must define a strict total order")


def load_text_ranking_groups(
    candidates_path: Path,
    labels_path: Path,
    *,
    control: DistillationControl,
) -> list[TextRankingGroup]:
    """Load one pseudo-label control and preserve complete query groups."""
    label_columns = {
        "bm25": "bm25_teacher_label",
        "random": "random_teacher_label",
        "prp": "prp_teacher_label",
    }
    if control not in label_columns:
        raise ValueError(f"unsupported distillation control: {control}")
    candidates = pd.read_parquet(candidates_path)
    labels = pd.read_parquet(labels_path)
    label_column = label_columns[control]
    required = {"query_id", "passage_id", label_column}
    missing = required - set(labels)
    if missing:
        raise ValueError(f"label file is missing columns: {sorted(missing)}")
    joined = candidates.merge(
        labels[["query_id", "passage_id", label_column]],
        on=["query_id", "passage_id"],
        how="left",
        validate="one_to_one",
    )
    if joined[label_column].isna().any() or len(joined) != len(candidates):
        raise ValueError("labels do not cover every candidate exactly once")

    groups = []
    for query_id, frame in joined.groupby("query_id", sort=False):
        frame = frame.sort_values("bm25_rank")
        splits = frame["split"].astype(str).unique()
        queries = frame["query"].astype(str).unique()
        if len(splits) != 1 or len(queries) != 1:
            raise ValueError(f"inconsistent request fields for query {query_id}")
        groups.append(
            TextRankingGroup(
                query_id=str(query_id),
                split=str(splits[0]),
                query=str(queries[0]),
                passage_ids=tuple(frame["passage_id"].astype(str)),
                passages=tuple(frame["passage"].astype(str)),
                labels=tuple(frame[label_column].astype(float)),
            )
        )
    if not groups:
        raise ValueError("distillation dataset contains no query groups")
    return groups


def collate_text_ranking_groups(
    tokenizer: Any,
    groups: Sequence[TextRankingGroup],
    *,
    max_length: int = 500,
) -> dict[str, object]:
    """Flatten complete query groups while retaining RankNet boundaries."""
    if not groups:
        raise ValueError("cannot collate an empty ranking-group batch")
    queries = [group.query for group in groups for _ in group.passages]
    passages = [passage for group in groups for passage in group.passages]
    encoded = tokenize_query_passages(
        tokenizer,
        queries,
        passages,
        max_length=max_length,
    )
    return {
        "encoded": encoded,
        "labels": torch.tensor(
            [label for group in groups for label in group.labels],
            dtype=torch.float32,
        ),
        "group_sizes": torch.tensor(
            [len(group.passage_ids) for group in groups],
            dtype=torch.int64,
        ),
        "query_ids": tuple(group.query_id for group in groups),
        "passage_ids": tuple(
            passage_id for group in groups for passage_id in group.passage_ids
        ),
    }


def teacher_ndcg_at_k(
    scores: torch.Tensor,
    labels: torch.Tensor,
    group_sizes: torch.Tensor,
    *,
    cutoff: int = 10,
) -> float:
    """Macro linear-gain NDCG against pseudo-teacher order."""
    if scores.ndim != 1 or labels.shape != scores.shape:
        raise ValueError("scores and labels must be equal one-dimensional tensors")
    if group_sizes.ndim != 1 or int(group_sizes.sum()) != scores.numel():
        raise ValueError("group_sizes must cover every score")
    if cutoff <= 0:
        raise ValueError("cutoff must be positive")
    values = []
    start = 0
    for raw_size in group_sizes.tolist():
        size = int(raw_size)
        group_scores = scores[start : start + size]
        group_labels = labels[start : start + size]
        order = torch.argsort(group_scores, descending=True, stable=True)
        ranked = group_labels[order][:cutoff].tolist()
        ideal = torch.sort(group_labels, descending=True).values[:cutoff].tolist()
        dcg = sum(
            float(relevance) / math.log2(rank + 1)
            for rank, relevance in enumerate(ranked, start=1)
        )
        idcg = sum(
            float(relevance) / math.log2(rank + 1)
            for rank, relevance in enumerate(ideal, start=1)
        )
        values.append(dcg / idcg if idcg else 0.0)
        start += size
    return sum(values) / len(values)
