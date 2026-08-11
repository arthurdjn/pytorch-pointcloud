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
    pred = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
    target = torch.tensor([[[0.0, 0.0, 0.0]]])
    assert chamfer_distance(pred, target, norm=norm).item() == pytest.approx(expected)


@pytest.mark.parametrize(
    ("norm", "expected"),
    [
        pytest.param("l2", 50.0, id="l2-no-sqrt"),
        pytest.param("l1", 5.0, id="l1-euclidean"),
    ],
)
def test_chamfer_distance_single_pair(norm: Literal["l1", "l2"], expected: float) -> None:
    pred = torch.tensor([[[0.0, 0.0, 0.0]]])
    target = torch.tensor([[[3.0, 4.0, 0.0]]])
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
    pred = torch.randn(2, 8, 3)
    target = torch.randn(2, 6, 3)
    forward = chamfer_distance(pred, target, norm=norm)
    backward = chamfer_distance(target, pred, norm=norm)
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
    pred = torch.randn(2, 16, 3)
    assert chamfer_distance(pred, pred.clone(), norm=norm).item() == pytest.approx(0.0)


@pytest.mark.parametrize(
    "norm",
    [
        pytest.param("l2", id="l2"),
        pytest.param("l1", id="l1"),
    ],
)
def test_chamfer_distance_batch_is_mean_of_samples(norm: Literal["l1", "l2"]) -> None:
    torch.manual_seed(0)
    pred_a, target_a = torch.randn(1, 8, 3), torch.randn(1, 6, 3)
    pred_b, target_b = torch.randn(1, 8, 3), torch.randn(1, 6, 3)
    batched = chamfer_distance(torch.cat([pred_a, pred_b]), torch.cat([target_a, target_b]), norm=norm)
    loss_a = chamfer_distance(pred_a, target_a, norm=norm)
    loss_b = chamfer_distance(pred_b, target_b, norm=norm)
    assert batched.item() == pytest.approx((loss_a + loss_b).item() / 2)


@pytest.mark.parametrize(
    "norm",
    [
        pytest.param("l2", id="l2"),
        pytest.param("l1", id="l1"),
    ],
)
def test_chamfer_distance_backward_flows(norm: Literal["l1", "l2"]) -> None:
    torch.manual_seed(0)
    pred = torch.randn(2, 8, 3, requires_grad=True)
    target = torch.randn(2, 6, 3)
    chamfer_distance(pred, target, norm=norm).backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()
    assert pred.grad.abs().sum() > 0


def test_chamfer_distance_unknown_norm_raises() -> None:
    pred = torch.randn(1, 4, 3)
    with pytest.raises(ValueError, match="norm"):
        chamfer_distance(pred, pred, norm="linf")  # type: ignore[arg-type]
