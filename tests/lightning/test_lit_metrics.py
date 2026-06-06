import pytest
import torch

from torch_pointcloud.lightning.metrics import MeanAveragePrecision3D, boxes_from_packed
from torch_pointcloud.utils.imports import _LIGHTNING_AVAILABLE
from torch_pointcloud.utils.types import Boxes3D, Detection3D

pytestmark = pytest.mark.skipif(not _LIGHTNING_AVAILABLE, reason="lightning is not installed")


def test_boxes_from_packed_doubles_half_extents() -> None:
    """The packed `box` (half-extents, label at column 7) becomes a full-extent `Boxes3D`."""
    box = torch.tensor([[1.0, 2.0, 3.0, 0.5, 1.0, 1.5, 0.7, 4.0]])
    batch = torch.tensor([0])
    out = boxes_from_packed(box, batch)
    assert torch.allclose(out["boxes"][0], torch.tensor([1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 0.7]))
    assert out["labels"].tolist() == [4]
    assert out["batch"] is batch


def test_mean_average_precision3d_perfect_match_scores_one() -> None:
    """A prediction identical to the ground truth yields `mAP@t = 1.0` at every threshold."""
    metric = MeanAveragePrecision3D(iou_thresholds=(0.25, 0.5))
    preds: Detection3D = {
        "boxes": torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]]),
        "scores": torch.tensor([0.9]),
        "labels": torch.tensor([0]),
        "batch": torch.tensor([0]),
    }
    target: Boxes3D = {
        "boxes": torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]]),
        "labels": torch.tensor([0]),
        "batch": torch.tensor([0]),
    }
    metric.update(preds, target)
    out = metric.compute()
    assert out["mAP@0.25"] == pytest.approx(1.0)
    assert out["mAP@0.5"] == pytest.approx(1.0)


def test_mean_average_precision3d_reset_clears_state() -> None:
    """`reset` empties the accumulated per-batch lists."""
    metric = MeanAveragePrecision3D()
    pred: Detection3D = {
        "boxes": torch.empty(0, 7),
        "scores": torch.empty(0),
        "labels": torch.empty(0),
        "batch": torch.empty(0),
    }
    target: Boxes3D = {"boxes": torch.empty(0, 7), "labels": torch.empty(0), "batch": torch.empty(0)}
    metric.update(pred, target)
    assert len(metric.preds) == 1
    metric.reset()
    assert metric.preds == []
