import pytest
import torch
from torch import Tensor

from torch_pointcloud.metrics import accuracy, confusion_matrix, intersection_over_union


@pytest.fixture
def perfect_preds() -> tuple[Tensor, Tensor]:
    target = torch.tensor([0, 0, 1, 1, 1, 2, 2])
    preds = target.clone()
    return preds, target


@pytest.fixture
def mixed_preds() -> tuple[Tensor, Tensor]:
    # 3 classes; target/pred designed so each (i, j) cell is hand-checked below.
    target = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])
    preds = torch.tensor([0, 0, 1, 1, 2, 2, 2, 2, 0])
    return preds, target


def test_confusion_matrix_perfect(perfect_preds: tuple[Tensor, Tensor]) -> None:
    preds, target = perfect_preds
    cm = confusion_matrix(preds, target, num_classes=3)
    assert torch.equal(cm.diag(), torch.tensor([2, 3, 2]))
    assert cm.sum().item() == target.numel()
    # All off-diagonal is zero.
    assert (cm - torch.diag(cm.diag())).sum().item() == 0


def test_confusion_matrix_mixed(mixed_preds: tuple[Tensor, Tensor]) -> None:
    preds, target = mixed_preds
    cm = confusion_matrix(preds, target, num_classes=3)
    # Row i = true class i, column j = predicted class j.
    expected = torch.tensor(
        [
            [2, 1, 0],  # true 0: 2 -> 0, 1 -> 1
            [0, 1, 2],  # true 1: 1 -> 1, 2 -> 2
            [1, 0, 2],  # true 2: 1 -> 0, 2 -> 2
        ],
        dtype=torch.long,
    )
    assert torch.equal(cm, expected)


def test_confusion_matrix_ignore_index() -> None:
    target = torch.tensor([0, 1, 2, 255])
    preds = torch.tensor([0, 1, 2, 0])
    cm = confusion_matrix(preds, target, num_classes=3, ignore_index=255)
    assert torch.equal(cm, torch.eye(3, dtype=torch.long))


def test_confusion_matrix_ignores_several_indices() -> None:
    preds = torch.tensor([0, 1, 2, 2])
    target = torch.tensor([0, 1, 2, 3])
    cm = confusion_matrix(preds, target, num_classes=4, ignore_index=[1, 3])
    assert cm.sum() == 2
    assert cm[0, 0] == 1 and cm[2, 2] == 1


def test_confusion_matrix_shape_and_dtype() -> None:
    preds = torch.tensor([0, 1, 2])
    target = torch.tensor([0, 1, 2])
    cm = confusion_matrix(preds, target, num_classes=5)
    assert cm.shape == (5, 5)
    assert cm.dtype == torch.long


def test_overall_accuracy_perfect(perfect_preds: tuple[Tensor, Tensor]) -> None:
    preds, target = perfect_preds
    cm = confusion_matrix(preds, target, num_classes=3)
    assert accuracy(cm) == pytest.approx(1.0)


def test_overall_accuracy_partial() -> None:
    preds = torch.tensor([0, 1, 1, 0])
    target = torch.tensor([0, 1, 0, 0])
    cm = confusion_matrix(preds, target, num_classes=2)
    assert accuracy(cm) == pytest.approx(0.75)


def test_overall_accuracy_ignore_index() -> None:
    preds = torch.tensor([0, 1, 0])
    target = torch.tensor([0, 1, 255])
    cm = confusion_matrix(preds, target, num_classes=2, ignore_index=255)
    assert accuracy(cm) == pytest.approx(1.0)


def test_overall_accuracy_fully_ignored_returns_zero() -> None:
    preds = torch.tensor([0, 1])
    target = torch.tensor([255, 255])
    cm = confusion_matrix(preds, target, num_classes=2, ignore_index=255)
    assert accuracy(cm) == 0.0


def test_per_class_accuracy_perfect(perfect_preds: tuple[Tensor, Tensor]) -> None:
    preds, target = perfect_preds
    cm = confusion_matrix(preds, target, num_classes=3)
    assert torch.allclose(accuracy(cm, average="none"), torch.ones(3), atol=1e-6)


def test_per_class_accuracy_partial() -> None:
    # class 0: 2/3 correct, class 1: 2/2 correct.
    preds = torch.tensor([0, 0, 1, 1, 1])
    target = torch.tensor([0, 0, 0, 1, 1])
    cm = confusion_matrix(preds, target, num_classes=2)
    assert torch.allclose(accuracy(cm, average="none"), torch.tensor([2.0 / 3.0, 1.0]), atol=1e-6)


def test_per_class_accuracy_ignore_index_zeros_class() -> None:
    preds = torch.tensor([0, 1, 1])
    target = torch.tensor([0, 1, 1])
    cm = confusion_matrix(preds, target, num_classes=3, ignore_index=2)
    acc = accuracy(cm, average="none", ignore_index=2)
    assert acc[2].item() == 0.0
    assert acc[0].item() == pytest.approx(1.0)
    assert acc[1].item() == pytest.approx(1.0)


def test_confusion_matrix_metrics_accumulate_batches() -> None:
    """Confusion matrices add up: the sum over batches is the matrix of all the points, so every metric agrees."""
    generator = torch.Generator().manual_seed(0)
    num_classes = 6
    batches = []
    for _ in range(4):
        target = torch.randint(-1, num_classes - 1, (500,), generator=generator)  # the last class occurs nowhere
        noise = torch.randint(0, num_classes - 1, (500,), generator=generator)
        preds = torch.where(torch.rand(500, generator=generator) < 0.7, target.clamp_min(0), noise)
        batches.append((preds, target))
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for preds, target in batches:
        cm += confusion_matrix(preds, target, num_classes, ignore_index=-1)
    all_preds = torch.cat([preds for preds, _ in batches])
    all_target = torch.cat([target for _, target in batches])

    assert torch.equal(cm, confusion_matrix(all_preds, all_target, num_classes, ignore_index=-1))
    assert 0.0 < intersection_over_union(cm) < 1.0
    assert accuracy(cm) == pytest.approx((all_preds == all_target)[all_target != -1].float().mean().item())
    assert intersection_over_union(cm, average="none")[-1] == 0.0
    assert intersection_over_union(cm, average="none", zero_division=1.0)[-1] == 1.0


def test_accuracy_empty_matrix_is_zero_division() -> None:
    cm = torch.zeros(3, 3, dtype=torch.long)
    assert accuracy(cm) == 0.0
    assert accuracy(cm, zero_division=1.0) == 1.0


def test_accuracy_averages() -> None:
    # class 0: 1/2 correct, class 1: 3/3 correct, class 2: no point.
    cm = torch.tensor([[1, 1, 0], [0, 3, 0], [0, 0, 0]])

    assert accuracy(cm) == pytest.approx(4 / 5)
    assert accuracy(cm, average="macro") == pytest.approx((0.5 + 1.0 + 0.0) / 3)
    assert accuracy(cm, average="none").tolist() == pytest.approx([0.5, 1.0, 0.0])
    assert accuracy(cm, average="none", zero_division=1.0).tolist() == pytest.approx([0.5, 1.0, 1.0])


def test_label_metrics_class_names() -> None:
    """`class_names` turns the per-class tensor into a `{name: value}` dict and must match the matrix."""
    cm = torch.tensor([[1, 1, 0], [0, 3, 0], [0, 0, 0]])
    names = ["wall", "floor", "chair"]

    ious = intersection_over_union(cm, average="none", class_names=names)
    assert list(ious) == names
    assert ious == pytest.approx({"wall": 0.5, "floor": 0.75, "chair": 0.0})
    assert accuracy(cm, average="none", class_names=names) == pytest.approx({"wall": 0.5, "floor": 1.0, "chair": 0.0})
    # averaged results stay floats, whatever the names
    assert intersection_over_union(cm, class_names=names) == intersection_over_union(cm)
    assert accuracy(cm, average="macro", class_names=names) == accuracy(cm, average="macro")
    with pytest.raises(ValueError, match="class_names"):
        intersection_over_union(cm, average="none", class_names=names[:2])
    with pytest.raises(ValueError, match="class_names"):
        accuracy(cm, average="none", class_names=names[:2])


def test_metrics_ignore_several_indices() -> None:
    """Ignoring a class drops the points it truly labels, even in a matrix that still counts them."""
    cm = torch.tensor([[4, 0, 1, 0], [2, 3, 0, 1], [0, 0, 5, 0], [6, 0, 0, 2]])
    kept = torch.tensor([[0, 0, 0, 0], [2, 3, 0, 1], [0, 0, 5, 0], [0, 0, 0, 0]])

    assert accuracy(cm, ignore_index=[0, 3]) == pytest.approx(8 / 11)
    assert accuracy(cm, average="macro", ignore_index=[0, 3]) == pytest.approx((3 / 6 + 5 / 5) / 2)
    assert intersection_over_union(cm, ignore_index=[0, 3]) == pytest.approx((3 / 6 + 5 / 5) / 2)
    assert torch.equal(
        intersection_over_union(cm, average="none", ignore_index=[0, 3]),
        intersection_over_union(kept, average="none", ignore_index=[0, 3]),
    )
    assert intersection_over_union(cm, ignore_index=255) == intersection_over_union(cm)
