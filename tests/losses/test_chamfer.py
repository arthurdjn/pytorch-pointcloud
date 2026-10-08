from typing import Literal

import pytest
import torch

from torch_pointcloud.losses import chamfer_distance


@pytest.mark.parametrize(
    ("norm", "expected"),
    [
        pytest.param("l2", 0.5, id="l2-sums-mean-squared-distances"),
        pytest.param("l1", 0.25, id="l1-averages-mean-euclidean-distances"),
    ],
)
def test_chamfer_distance_hand_computed(norm: Literal["l1", "l2"], expected: float) -> None:
    pred = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    target = torch.tensor([[0.0, 0.0, 0.0]])
    assert chamfer_distance(pred, target, norm=norm).item() == pytest.approx(expected)


@pytest.mark.parametrize(
    ("norm", "expected"),
    [
        pytest.param("l2", 50.0, id="l2-no-sqrt"),
        pytest.param("l1", 5.0, id="l1-euclidean"),
    ],
)
def test_chamfer_distance_single_pair(norm: Literal["l1", "l2"], expected: float) -> None:
    pred = torch.tensor([[0.0, 0.0, 0.0]])
    target = torch.tensor([[3.0, 4.0, 0.0]])
    assert chamfer_distance(pred, target, norm=norm).item() == pytest.approx(expected)


@pytest.mark.parametrize(
    "norm",
    [
        pytest.param("l2", id="l2"),
        pytest.param("l1", id="l1"),
    ],
)
def test_chamfer_distance_is_symmetric(norm: Literal["l1", "l2"]) -> None:
    torch.manual_seed(0)
    pred, pred_batch = torch.randn(16, 3), torch.arange(2).repeat_interleave(8)
    target, target_batch = torch.randn(12, 3), torch.arange(2).repeat_interleave(6)
    forward = chamfer_distance(pred, target, pred_batch=pred_batch, target_batch=target_batch, norm=norm)
    backward = chamfer_distance(target, pred, pred_batch=target_batch, target_batch=pred_batch, norm=norm)
    assert forward.item() == pytest.approx(backward.item())


@pytest.mark.parametrize(
    "norm",
    [
        pytest.param("l2", id="l2"),
        pytest.param("l1", id="l1"),
    ],
)
def test_chamfer_distance_identical_clouds_is_zero(norm: Literal["l1", "l2"]) -> None:
    torch.manual_seed(0)
    pred, batch = torch.randn(32, 3), torch.arange(2).repeat_interleave(16)
    assert chamfer_distance(
        pred, pred.clone(), pred_batch=batch, target_batch=batch, norm=norm
    ).item() == pytest.approx(0.0)


@pytest.mark.parametrize(
    "norm",
    [
        pytest.param("l2", id="l2"),
        pytest.param("l1", id="l1"),
    ],
)
def test_chamfer_distance_matches_within_each_set(norm: Literal["l1", "l2"]) -> None:
    torch.manual_seed(0)
    pred_a, target_a = torch.randn(8, 3), torch.randn(6, 3)
    pred_b, target_b = torch.randn(8, 3), torch.randn(6, 3)
    packed = chamfer_distance(
        torch.cat([pred_a, pred_b]),
        torch.cat([target_a, target_b]),
        pred_batch=torch.arange(2).repeat_interleave(8),
        target_batch=torch.arange(2).repeat_interleave(6),
        norm=norm,
    )
    loss_a = chamfer_distance(pred_a, target_a, norm=norm)
    loss_b = chamfer_distance(pred_b, target_b, norm=norm)
    assert packed.item() == pytest.approx((loss_a + loss_b).item() / 2)


def test_chamfer_distance_sets_of_different_sizes() -> None:
    torch.manual_seed(0)
    pred, pred_batch = torch.randn(13, 3), torch.tensor([0] * 4 + [1] * 9)
    target, target_batch = torch.randn(10, 3), torch.tensor([0] * 7 + [1] * 3)
    packed = chamfer_distance(pred, target, pred_batch=pred_batch, target_batch=target_batch)
    per_set = [chamfer_distance(pred[pred_batch == b], target[target_batch == b]) for b in range(2)]
    # the mean runs over all points, so each set weighs by its point count in each direction
    sq_pred = [(pred[pred_batch == b][:, None] - target[target_batch == b][None]).pow(2).sum(-1) for b in range(2)]
    expected = (
        torch.cat([d.min(1).values for d in sq_pred]).mean() + torch.cat([d.min(0).values for d in sq_pred]).mean()
    )
    assert packed.item() == pytest.approx(expected.item())
    assert all(torch.isfinite(loss) for loss in per_set)


@pytest.mark.parametrize(
    "norm",
    [
        pytest.param("l2", id="l2"),
        pytest.param("l1", id="l1"),
    ],
)
def test_chamfer_distance_backward_flows(norm: Literal["l1", "l2"]) -> None:
    torch.manual_seed(0)
    pred, batch = torch.randn(16, 3, requires_grad=True), torch.arange(2).repeat_interleave(8)
    target, target_batch = torch.randn(12, 3), torch.arange(2).repeat_interleave(6)
    chamfer_distance(pred, target, pred_batch=batch, target_batch=target_batch, norm=norm).backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()
    assert pred.grad.abs().sum() > 0


def test_chamfer_distance_unknown_norm_raises() -> None:
    pred = torch.randn(4, 3)
    with pytest.raises(ValueError, match="norm"):
        chamfer_distance(pred, pred, norm="linf")  # type: ignore[arg-type]
