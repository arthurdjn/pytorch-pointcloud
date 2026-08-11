import pytest
import torch

from torch_pointcloud.losses import LovaszLoss


def test_lovasz_loss_perfect_prediction_is_near_zero() -> None:
    labels = torch.tensor([0, 1, 2, 0, 1, 2])
    logits = torch.full((6, 3), -10.0)
    logits[torch.arange(6), labels] = 10.0
    assert LovaszLoss()(logits, labels).item() < 1e-3


def test_lovasz_loss_wrong_prediction_is_positive() -> None:
    labels = torch.tensor([0, 0, 0, 1, 1, 1])
    logits = torch.zeros(6, 3)
    logits[:, 2] = 10.0  # always predicts the absent class 2
    assert LovaszLoss()(logits, labels).item() > 0.5


def test_lovasz_loss_ignores_index() -> None:
    labels = torch.tensor([0, 1, -1, -1])
    logits = torch.full((4, 3), -10.0)
    logits[0, 0] = 10.0
    logits[1, 1] = 10.0  # rows 2-3 are garbage but ignored
    assert LovaszLoss(ignore_index=-1)(logits, labels).item() < 1e-3


def test_lovasz_loss_empty_input_is_zero() -> None:
    logits = torch.zeros(0, 3)
    labels = torch.zeros(0, dtype=torch.long)
    out = LovaszLoss()(logits, labels)
    assert out.item() == 0.0
    assert torch.isfinite(out)


def test_lovasz_loss_all_ignored_is_zero() -> None:
    logits = torch.randn(4, 3)
    labels = torch.full((4,), -1)
    out = LovaszLoss(ignore_index=-1)(logits, labels)
    assert out.item() == 0.0


def test_lovasz_loss_backward_flows() -> None:
    logits = torch.randn(10, 4, requires_grad=True)
    labels = torch.randint(0, 4, (10,))
    LovaszLoss()(logits, labels).backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_lovasz_loss_invalid_classes_raises() -> None:
    with pytest.raises(ValueError, match="'present' or 'all'"):
        LovaszLoss(classes="presnt")  # type: ignore[arg-type]


def test_lovasz_loss_no_present_class_is_zero() -> None:
    logits = torch.randn(4, 3, requires_grad=True)
    labels = torch.full((4,), 5)  # every label outside [0, C)
    out = LovaszLoss(classes="present")(logits, labels)
    assert out.item() == 0.0
    out.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
