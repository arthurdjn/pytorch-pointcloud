import pytest
import torch

from torch_pointcloud.losses import hungarian_match


def test_hungarian_match_minimizes_total_cost() -> None:
    cost = torch.tensor([[0.9, 0.1], [0.2, 0.8], [0.5, 0.5]])
    pred, target = hungarian_match(cost)
    assert pred.tolist() == [0, 1]
    assert target.tolist() == [1, 0]
    assert pred.dtype == torch.long and target.dtype == torch.long


def test_hungarian_match_more_targets_than_predictions() -> None:
    cost = torch.tensor([[0.1, 0.9, 0.5], [0.9, 0.1, 0.5]])
    pred, target = hungarian_match(cost)
    assert pred.tolist() == [0, 1]
    assert target.tolist() == [0, 1]


def test_hungarian_match_empty_sides() -> None:
    for shape in ((0, 3), (3, 0), (0, 0)):
        pred, target = hungarian_match(torch.zeros(shape))
        assert pred.numel() == 0 and target.numel() == 0


def test_hungarian_match_rejects_non_finite_and_non_matrix_costs() -> None:
    with pytest.raises(ValueError, match="finite"):
        hungarian_match(torch.tensor([[0.1, float("nan")]]))
    with pytest.raises(ValueError, match="matrix"):
        hungarian_match(torch.zeros(2, 2, 2))


def test_hungarian_match_does_not_track_gradients() -> None:
    cost = torch.rand(3, 3, requires_grad=True)
    pred, target = hungarian_match(cost)
    assert not pred.requires_grad and not target.requires_grad
