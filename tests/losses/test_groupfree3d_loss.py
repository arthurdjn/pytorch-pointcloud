from typing import Any, Dict, Tuple

import pytest
import torch
from torch import Tensor

from torch_pointcloud.losses import GroupFree3DLoss
from torch_pointcloud.losses.focal import sigmoid_focal_loss
from torch_pointcloud.utils.data import DataKeys


def _create_loss(size_class_agnostic: bool = False, **overrides: Any) -> GroupFree3DLoss:
    kwargs: Dict[str, Any] = dict(
        num_classes=4,
        num_heading_bins=12,
        mean_sizes=None
        if size_class_agnostic
        else [[1.0, 1.0, 1.0], [2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 2.0]],
        sampling_weight=0.8,
        objectness_weight=0.1,
        center_delta=0.04,
        size_delta=0.111,
        heading_delta=1.0,
    )
    kwargs.update(overrides)
    return GroupFree3DLoss(**kwargs)


def _create_data(
    size_class_agnostic: bool = False,
    batch_size: int = 2,
    num_seeds: int = 32,
    num_queries: int = 8,
    num_stages: int = 3,
) -> Tuple[Dict[str, Tensor], Dict[str, Any]]:
    torch.manual_seed(0)
    t, b, q = num_stages, batch_size, num_queries
    output: Dict[str, Tensor] = {
        "objectness_scores": torch.randn(t, b, q, 1),
        "center": torch.rand(t, b, q, 3) * 4,
        "heading_scores": torch.randn(t, b, q, 12),
        "heading_residuals_normalized": torch.randn(t, b, q, 12),
        "size": torch.rand(t, b, q, 3) + 0.5,
        "sem_cls_scores": torch.randn(t, b, q, 4),
        "seed_objectness_logits": torch.randn(b * num_seeds),
        "pos_seed": torch.rand(b * num_seeds, 3) * 4,
        "batch_seed": torch.arange(b).repeat_interleave(num_seeds),
    }
    if not size_class_agnostic:
        output["size_scores"] = torch.randn(t, b, q, 4)
        output["size_residuals_normalized"] = torch.randn(t, b, q, 4, 3)
    # The first seed of each scene sits on a box center, so every scene has a positive candidate.
    centers = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    output["pos_seed"][num_seeds * torch.arange(b)] = centers[:b]
    output["query_indices"] = torch.arange(q).repeat(b, 1) + num_seeds * torch.arange(b).unsqueeze(1)
    data: Dict[str, Any] = {
        DataKeys.BOX: torch.tensor(
            [
                [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 0.3],
                [3.0, 2.0, 1.0, 1.5, 1.0, 1.0, -0.5],
                [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0],
            ]
        ),
        DataKeys.LABEL: torch.tensor([0, 3, 1]),
        DataKeys.BATCH_BOX: torch.tensor([0, 0, 1]),
    }
    return output, data


@pytest.mark.parametrize(
    "size_class_agnostic", [pytest.param(False, id="templates"), pytest.param(True, id="agnostic")]
)
def test_groupfree3d_loss_returns_finite_scalars(size_class_agnostic: bool) -> None:
    loss_fn = _create_loss(size_class_agnostic)
    output, data = _create_data(size_class_agnostic)
    out = loss_fn(output, data)
    for key in ("loss", "sampling_loss", "objectness_loss", "box_loss", "center_loss", "size_res_loss", "sem_cls_loss"):
        assert out[key].ndim == 0, key
        assert torch.isfinite(out[key]), key
    if size_class_agnostic:
        assert out["size_cls_loss"] == 0


def test_groupfree3d_loss_backward_reaches_every_stage_and_the_seeds() -> None:
    loss_fn = _create_loss()
    output, data = _create_data()
    for value in output.values():
        if value.is_floating_point():
            value.requires_grad_(True)
    loss_fn(output, data)["loss"].backward()
    for key in ("center", "objectness_scores", "size_residuals_normalized", "seed_objectness_logits"):
        grad = output[key].grad
        assert grad is not None, key
        assert torch.isfinite(grad).all(), key
    center_grad = output["center"].grad
    assert center_grad is not None
    assert (center_grad.flatten(1).abs().sum(dim=1) > 0).all()


def test_groupfree3d_loss_without_boxes_is_finite() -> None:
    loss_fn = _create_loss()
    output, data = _create_data()
    data = {key: value[:0] for key, value in data.items()}
    out = loss_fn(output, data)
    assert torch.isfinite(out["loss"])
    assert out["center_loss"] == 0


def test_groupfree3d_loss_sampling_marks_the_closest_seeds_of_each_object() -> None:
    loss_fn = _create_loss(num_positive_seeds=2)
    output, data = _create_data(batch_size=1, num_seeds=6, num_queries=2)
    output["pos_seed"] = torch.tensor(
        [
            [1.0, 1.0, 1.0],  # center of the box
            [1.4, 1.0, 1.0],
            [1.8, 1.0, 1.0],
            [1.1, 1.0, 1.0],
            [5.0, 5.0, 5.0],  # background
            [0.5, 1.0, 1.0],
        ]
    )
    data = {
        DataKeys.BOX: torch.tensor([[1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 0.0]]),
        DataKeys.LABEL: torch.tensor([0]),
        DataKeys.BATCH_BOX: torch.tensor([0]),
    }
    out = loss_fn(output, data)

    label = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    logits = output["seed_objectness_logits"]
    expected = sigmoid_focal_loss(logits.unsqueeze(-1), label.unsqueeze(-1), torch.full((6,), 1 / 6)).sum()
    assert torch.allclose(out["sampling_loss"], expected)


def test_groupfree3d_loss_exact_centers_give_no_center_loss() -> None:
    loss_fn = _create_loss()
    output, data = _create_data(batch_size=1, num_seeds=4, num_queries=2)
    output["pos_seed"] = torch.tensor([[1.0, 1.0, 1.0], [3.0, 2.0, 1.0], [9.0, 9.0, 9.0], [1.2, 1.0, 1.0]])
    output["query_indices"] = torch.tensor([[0, 1]])
    output["center"] = torch.tensor([[1.0, 1.0, 1.0], [3.0, 2.0, 1.0]]).expand(3, 1, 2, 3).clone()
    out = loss_fn(output, data)
    assert out["center_loss"] == 0

    output["center"] = output["center"] + 0.5
    assert loss_fn(output, data)["center_loss"] > 0


def test_groupfree3d_loss_requires_one_size_template_per_class() -> None:
    with pytest.raises(ValueError, match="mean_sizes"):
        _create_loss(mean_sizes=[[1.0, 1.0, 1.0]])
