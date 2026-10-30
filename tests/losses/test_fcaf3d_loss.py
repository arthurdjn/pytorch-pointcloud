from typing import Dict

import pytest
import torch
from torch import Tensor

from torch_pointcloud.losses import FCAF3DLoss

STRIDES = (0.08, 0.16, 0.32, 0.64)


def _pyramid(num_scenes: int = 1, extent: float = 4.0) -> Dict[str, Tensor]:
    """Locations on the grid of every level over a cube of side `extent`, one block per scene."""
    pos, level, batch = [], [], []
    for scene in range(num_scenes):
        for i, stride in enumerate(STRIDES):
            axis = torch.arange(0.0, extent, stride)
            grid = torch.cartesian_prod(axis, axis, axis)
            pos.append(grid)
            level.append(torch.full((len(grid),), i))
            batch.append(torch.full((len(grid),), scene))
    return {"pos": torch.cat(pos), "level": torch.cat(level), "batch": torch.cat(batch)}


def _output(pyramid: Dict[str, Tensor], num_classes: int) -> Dict[str, Tensor]:
    generator = torch.Generator().manual_seed(0)
    m = len(pyramid["pos"])
    boxes = torch.cat([pyramid["pos"], torch.full((m, 3), 0.5), torch.zeros(m, 1)], dim=1)
    return {
        "cls_logits": torch.randn(m, num_classes, generator=generator).requires_grad_(),
        "centerness": torch.randn(m, 1, generator=generator).requires_grad_(),
        "boxes": (boxes + 0.05 * torch.randn(m, 7, generator=generator)).requires_grad_(),
        **pyramid,
    }


def test_fcaf3d_loss_assigns_a_box_to_one_level_and_its_most_central_locations() -> None:
    pyramid = _pyramid()
    criterion = FCAF3DLoss(num_classes=3, min_points_per_level=27, num_positives=18)
    boxes = torch.tensor([[2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 0.0]])
    center, box, cls = criterion.assign(pyramid["pos"], pyramid["level"], boxes, torch.tensor([2]), num_levels=4)

    positive = cls >= 0
    assert 0 < int(positive.sum()) <= 18
    assert (cls[positive] == 2).all() and (box[positive] == boxes).all()
    # A 1 m box holds 27 or more locations down to stride 0.32 and fewer at 0.64, so it goes to level 2.
    assert set(pyramid["level"][positive].tolist()) == {2}
    inside = ((pyramid["pos"][positive] - 2.0).abs() < 0.5).all(dim=1)
    assert inside.all()
    assert (center[positive] > 0).all() and (center[positive] <= 1).all()


def test_fcaf3d_loss_gives_a_location_to_the_smallest_box() -> None:
    pyramid = _pyramid()
    criterion = FCAF3DLoss(num_classes=2, min_points_per_level=1, num_positives=1000)
    boxes = torch.tensor([[2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.0], [2.0, 2.0, 2.0, 0.6, 0.6, 0.6, 0.0]])
    _, box, cls = criterion.assign(pyramid["pos"], pyramid["level"], boxes, torch.tensor([0, 1]), num_levels=4)
    both = (cls >= 0) & ((pyramid["pos"] - 2.0).abs() < 0.3).all(dim=1)
    assert both.any()
    assert (cls[both] == 1).all()


def test_fcaf3d_loss_rotated_boxes_use_their_own_frame() -> None:
    pos = torch.tensor([[0.0, 0.9, 0.0], [0.9, 0.0, 0.0]])
    criterion = FCAF3DLoss(num_classes=1, min_points_per_level=1, num_positives=10)
    boxes = torch.tensor([[0.0, 0.0, 0.0, 2.0, 0.5, 1.0, torch.pi / 2]])
    _, _, cls = criterion.assign(pos, torch.zeros(2, dtype=torch.long), boxes, torch.tensor([0]), num_levels=1)
    assert cls.tolist() == [0, -1]


def test_fcaf3d_loss_terms_are_finite_and_train_every_branch() -> None:
    output = _output(_pyramid(num_scenes=2), num_classes=3)
    data = {
        "box": torch.tensor([[2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 0.0], [1.0, 3.0, 1.0, 0.8, 1.2, 0.6, 0.0]]),
        "label": torch.tensor([2, 0]),
        "batch_box": torch.tensor([0, 1]),
    }
    losses = FCAF3DLoss(num_classes=3, min_points_per_level=27, num_positives=18)(output, data)

    assert set(losses) == {"loss", "cls_loss", "center_loss", "box_loss"}
    assert torch.isfinite(losses["loss"]) and not losses["cls_loss"].requires_grad
    assert losses["loss"].item() == pytest.approx(
        sum(losses[k].item() for k in ("cls_loss", "center_loss", "box_loss"))
    )
    losses["loss"].backward()
    for key in ("cls_logits", "centerness", "boxes"):
        grad = output[key].grad
        assert grad is not None and grad.abs().sum() > 0


def test_fcaf3d_loss_scene_without_boxes_only_classifies() -> None:
    output = _output(_pyramid(), num_classes=3)
    empty = {
        "box": torch.zeros(0, 7),
        "label": torch.zeros(0, dtype=torch.long),
        "batch_box": torch.zeros(0, dtype=torch.long),
    }
    losses = FCAF3DLoss(num_classes=3, min_points_per_level=27, num_positives=18)(output, empty)
    assert losses["center_loss"] == 0 and losses["box_loss"] == 0 and losses["cls_loss"] > 0
