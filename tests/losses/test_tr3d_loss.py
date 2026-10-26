from typing import Dict

import pytest
import torch
from torch import Tensor

from torch_pointcloud.losses import TR3DLoss

STRIDES = (0.16, 0.32)


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


def test_tr3d_loss_assigns_the_nearest_locations_of_the_class_level() -> None:
    pyramid = _pyramid()
    criterion = TR3DLoss(num_classes=2, label_levels=[0, 1], num_positives=6)
    boxes = torch.tensor([[2.01, 2.10, 2.20, 1.0, 1.0, 1.0, 0.0]])
    box, cls = criterion.assign(pyramid["pos"], pyramid["level"], boxes, torch.tensor([1]))

    positive = cls >= 0
    assert int(positive.sum()) == 6
    assert set(pyramid["level"][positive].tolist()) == {1}
    distances = (pyramid["pos"] - boxes[:, :3]).pow(2).sum(dim=1)
    level_one = pyramid["level"] == 1
    assert distances[positive].max() <= distances[level_one & ~positive].min()
    assert (box[positive] == boxes).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_tr3d_loss_assigns_on_the_device_of_its_inputs() -> None:
    pyramid = _pyramid()
    criterion = TR3DLoss(num_classes=2, label_levels=[0, 1], num_positives=6)
    boxes = torch.tensor([[2.01, 2.10, 2.20, 1.0, 1.0, 1.0, 0.0]], device="cuda")
    _, cls = criterion.assign(pyramid["pos"].cuda(), pyramid["level"].cuda(), boxes, torch.tensor([1], device="cuda"))

    assert cls.is_cuda
    assert int((cls >= 0).sum()) == 6


def test_tr3d_loss_rejects_a_level_table_of_the_wrong_size() -> None:
    with pytest.raises(ValueError, match="one level per class"):
        TR3DLoss(num_classes=3, label_levels=[0, 1], num_positives=6)


def test_tr3d_loss_terms_are_finite_and_train_every_branch() -> None:
    pyramid = _pyramid(num_scenes=2)
    m = len(pyramid["pos"])
    generator = torch.Generator().manual_seed(0)
    boxes = torch.cat([pyramid["pos"], torch.full((m, 3), 0.5), torch.zeros(m, 1)], dim=1)
    output = {
        "cls_logits": torch.randn(m, 2, generator=generator).requires_grad_(),
        "boxes": (boxes + 0.05 * torch.randn(m, 7, generator=generator)).requires_grad_(),
        **pyramid,
    }
    data = {
        "box": torch.tensor([[2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 0.0], [1.0, 3.0, 1.0, 0.8, 1.2, 0.6, 0.0]]),
        "label": torch.tensor([1, 0]),
        "batch_box": torch.tensor([0, 1]),
    }
    losses = TR3DLoss(num_classes=2, label_levels=[0, 1], num_positives=6)(output, data)

    assert set(losses) == {"loss", "cls_loss", "box_loss"}
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()
    for key in ("cls_logits", "boxes"):
        grad = output[key].grad
        assert grad is not None and grad.abs().sum() > 0
