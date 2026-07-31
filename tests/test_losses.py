from __future__ import annotations

import math

import pytest
import torch

from caged_ltr.losses import bpr_loss, listwise_kl_loss, pointwise_bce_loss, ranknet_loss


def test_bpr_rewards_a_larger_positive_margin_and_honors_mask() -> None:
    good = bpr_loss(
        torch.tensor([2.0, -100.0]),
        torch.tensor([0.0, 100.0]),
        mask=torch.tensor([True, False]),
    )
    bad = bpr_loss(torch.tensor([0.0]), torch.tensor([2.0]))

    assert good < bad


def test_pointwise_bce_has_known_value_and_gradient() -> None:
    logits = torch.zeros(2, requires_grad=True)
    loss = pointwise_bce_loss(logits, torch.tensor([0.0, 1.0]))

    assert loss.item() == pytest.approx(math.log(2.0))
    loss.backward()
    assert logits.grad is not None


def test_ranknet_rewards_correct_ordering() -> None:
    labels = torch.tensor([2.0, 1.0, 0.0])
    groups = torch.tensor([3])

    correct = ranknet_loss(torch.tensor([2.0, 1.0, 0.0]), labels, groups)
    reversed_order = ranknet_loss(torch.tensor([0.0, 1.0, 2.0]), labels, groups)

    assert correct < reversed_order


def test_ranknet_gradient_pushes_the_teacher_preferred_item_up() -> None:
    scores = torch.zeros(2, requires_grad=True)
    loss = ranknet_loss(scores, torch.tensor([2.0, 1.0]), torch.tensor([2]))

    loss.backward()

    assert scores.grad is not None
    assert scores.grad[0] < 0
    assert scores.grad[1] > 0


def test_ranknet_handles_groups_without_pairs() -> None:
    scores = torch.tensor([0.1, 0.2], requires_grad=True)
    loss = ranknet_loss(scores, torch.ones(2), torch.tensor([2]))

    assert loss.item() == 0.0
    loss.backward()
    assert scores.grad is not None


def test_listwise_kl_is_zero_only_for_matching_distributions() -> None:
    teacher = torch.tensor([2.0, 1.0, 0.0])
    groups = torch.tensor([3])

    matching = listwise_kl_loss(teacher.clone(), teacher, groups)
    reversed_order = listwise_kl_loss(torch.flip(teacher, dims=(0,)), teacher, groups)

    assert matching.item() == pytest.approx(0.0, abs=1e-7)
    assert reversed_order.item() > 0.1
