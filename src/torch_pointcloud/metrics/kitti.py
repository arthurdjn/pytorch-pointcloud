"""The KITTI 3D object detection protocol: per-class AP at the official IoU thresholds."""

from typing import Dict, Sequence

from .detection import BoxMatches, box_average_precision

KITTI_IOU_THRESHOLDS = {"Car": 0.7, "Pedestrian": 0.5, "Cyclist": 0.5}


def kitti_average_precision(
    matches: Sequence[BoxMatches],
    class_names: Sequence[str] = ("Car", "Pedestrian", "Cyclist"),
) -> Dict[str, float]:
    r"""The KITTI 3D detection metrics: per-class AP at the official IoU thresholds, and their mean.

    Each class is matched at its KITTI IoU threshold (`Car` at $0.7$, `Pedestrian` and `Cyclist` at $0.5$), and AP
    interpolates precision at the 11 recall points $0.0, 0.1, \ldots, 1.0$ of the KITTI benchmark. `mAP` averages the
    per-class APs.

    Args:
        matches: The `box_matches` records of every evaluated scene, with labels indexing `class_names`.
        class_names: Class name per label index, each one of `Car`, `Pedestrian` and `Cyclist`.

    Returns:
        `AP/<class>` for every class, and `mAP`.

    Example:
        ```pycon
        >>> import torch
        >>> from torch_pointcloud.metrics import box_matches, kitti_average_precision
        >>> zero = torch.tensor([0])
        >>> boxes = torch.tensor([[0.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0]])
        >>> preds = {"boxes": boxes, "scores": torch.tensor([0.9]), "labels": zero, "batch": zero}
        >>> target = {"boxes": boxes, "labels": zero, "batch": zero}
        >>> metrics = kitti_average_precision([box_matches(preds, target)])
        >>> sorted(metrics)
        ['AP/Car', 'AP/Cyclist', 'AP/Pedestrian', 'mAP']

        ```
    """
    iou_threshold = {index: KITTI_IOU_THRESHOLDS[name] for index, name in enumerate(class_names)}
    per_class = box_average_precision(
        matches,
        iou_threshold=iou_threshold,
        average="none",
        class_names=class_names,
        interpolation="r11",
    )
    metrics = {f"AP/{name}": ap for name, ap in per_class.items()}
    metrics["mAP"] = box_average_precision(matches, iou_threshold=iou_threshold, interpolation="r11")
    return metrics
