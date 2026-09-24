import torch

from torch_pointcloud.metrics import box_average_precision, box_matches, kitti_average_precision
from torch_pointcloud.metrics.detection import BoxMatches
from torch_pointcloud.utils.types import Boxes3D, Detection3D


def _matches(offset: float, label: int) -> BoxMatches:
    """One ground-truth box of `label` and a prediction shifted along x by `offset`."""
    box = torch.tensor([[0.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0]])
    shifted = box + torch.tensor([offset, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    labels = torch.tensor([label])
    zero = torch.tensor([0])
    preds: Detection3D = {"boxes": shifted, "scores": torch.tensor([0.9]), "labels": labels, "batch": zero}
    target: Boxes3D = {"boxes": box, "labels": labels, "batch": zero}
    return box_matches(preds, target)


def test_kitti_average_precision_matches_the_explicit_protocol() -> None:
    matches = [_matches(0.0, 0), _matches(0.3, 1), _matches(0.1, 2), _matches(1.0, 0)]
    thresholds = {0: 0.7, 1: 0.5, 2: 0.5}
    per_class = box_average_precision(
        matches,
        iou_threshold=thresholds,
        average="none",
        class_names=("Car", "Pedestrian", "Cyclist"),
        interpolation="r11",
    )
    expected = {f"AP/{name}": ap for name, ap in per_class.items()}
    expected["mAP"] = box_average_precision(matches, iou_threshold=thresholds, interpolation="r11")
    assert kitti_average_precision(matches) == expected


def test_kitti_average_precision_thresholds_follow_the_class_names() -> None:
    # A 4 m box shifted by 1 m overlaps its target at IoU 0.6: a true positive at 0.5, a miss at 0.7.
    matches = [_matches(1.0, 0), _matches(1.0, 1)]
    metrics = kitti_average_precision(matches, class_names=("Pedestrian", "Car"))
    assert metrics["AP/Pedestrian"] > 0.0
    assert metrics["AP/Car"] == 0.0
