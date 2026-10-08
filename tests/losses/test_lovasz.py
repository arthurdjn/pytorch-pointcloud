from typing import Literal

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


def _lovasz_softmax_loop(
    probas: torch.Tensor,
    labels: torch.Tensor,
    *,
    classes: str = "present",
    ignore_index: int = -1,
) -> torch.Tensor:
    """The per-class loop the batched `lovasz_softmax` replaced, kept as the reference."""
    valid = labels != ignore_index
    probas, labels = probas[valid], labels[valid]
    if probas.numel() == 0:
        return probas.sum()

    losses = []
    for c in range(probas.size(1)):
        fg = (labels == c).to(probas.dtype)
        if classes == "present" and fg.sum() == 0:
            continue

        errors = (fg - probas[:, c]).abs()
        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
        gt_sorted = fg[perm]
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.cumsum(0)
        union = gts + (1.0 - gt_sorted).cumsum(0)
        jaccard = 1.0 - intersection / union
        jaccard[1:] = jaccard[1:] - jaccard[:-1]
        losses.append(torch.dot(errors_sorted, jaccard))
    if not losses:
        return 0.0 * probas.sum()
    return torch.stack(losses).mean()


@pytest.mark.parametrize("classes", ["present", "all"])
@pytest.mark.parametrize("seed", [0, 1])
def test_lovasz_softmax_matches_per_class_loop(classes: Literal["present", "all"], seed: int) -> None:
    """Value and gradient of the class-batched sort equal the per-class loop (ignored points, absent classes)."""
    from torch_pointcloud.losses import lovasz_softmax

    torch.manual_seed(seed)
    num, num_classes = 5000, 7
    logits = torch.randn(num, num_classes, requires_grad=True)
    labels = torch.randint(0, num_classes, (num,))
    labels[labels == 3] = 0  # class 3 absent
    labels[torch.rand(num) < 0.1] = -1
    reference = _lovasz_softmax_loop(logits.softmax(1), labels, classes=classes)
    result = lovasz_softmax(logits.softmax(1), labels, classes=classes)
    assert torch.allclose(result, reference, atol=1e-6, rtol=1e-6)
    grad_reference = torch.autograd.grad(reference, logits)[0]
    grad_result = torch.autograd.grad(result, logits)[0]
    assert torch.allclose(grad_result, grad_reference, atol=1e-6, rtol=1e-5)


def test_lovasz_softmax_batches_classes_in_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    """A tiny chunk budget sorts the classes a few at a time and still gives the single-call value."""
    from torch_pointcloud.losses import lovasz, lovasz_softmax

    torch.manual_seed(0)
    probas = torch.randn(200, 9).softmax(1)
    labels = torch.randint(-1, 9, (200,))
    whole = lovasz_softmax(probas, labels)
    monkeypatch.setattr(lovasz, "_SORT_CHUNK_ELEMENTS", 400)  # two classes per sort
    assert torch.allclose(lovasz_softmax(probas, labels), whole)
