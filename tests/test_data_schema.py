from __future__ import annotations

import numpy as np
import pytest
from torch.utils.data import DataLoader

from caged_ltr.data import (
    CandidateDataset,
    CandidateList,
    collate_candidate_lists,
)


def _example(request_id: str, candidate_count: int) -> CandidateList:
    return CandidateList(
        request_id=request_id,
        candidate_ids=[f"{request_id}-{index}" for index in range(candidate_count)],
        features=np.arange(candidate_count * 2).reshape(candidate_count, 2),
        labels=np.asarray([1.0] + [0.0] * (candidate_count - 1)),
        query_frequency=candidate_count,
        candidate_frequencies=np.arange(1, candidate_count + 1),
    )


def test_candidate_dataloader_preserves_group_boundaries() -> None:
    examples = [_example("q1", 2), _example("q2", 3)]
    loader = DataLoader(
        CandidateDataset(examples),
        batch_size=2,
        collate_fn=collate_candidate_lists,
        shuffle=False,
    )

    batch = next(iter(loader))

    assert batch.features.shape == (5, 2)
    assert batch.group_sizes.tolist() == [2, 3]
    assert batch.request_ids == ("q1", "q2")
    assert [(item.start, item.stop) for item in batch.group_slices()] == [(0, 2), (2, 5)]


def test_candidate_list_rejects_duplicate_candidate_ids() -> None:
    with pytest.raises(ValueError, match="unique"):
        CandidateList(
            request_id="q1",
            candidate_ids=["same", "same"],
            features=np.ones((2, 1)),
            labels=np.asarray([1.0, 0.0]),
        )
