from typing import Any, Dict, List, Sequence

import torch
from torch import Tensor
from torchmetrics import Metric

from torch_pointcloud.utils.metrics import mean_average_precision3d
from torch_pointcloud.utils.types import Boxes3D, Detection3D


def boxes_from_packed(box: Tensor, batch: Tensor) -> Boxes3D:
    r"""Repo detection ground truth `box` to a `Boxes3D` with full edge lengths.

    Reads the SUN RGB-D layout: rows $[c_x, c_y, c_z, d_x, d_y, d_z, \text{heading}, \text{cls}]$ of shape
    $(K, 8)$ with half-extents $d$, doubled to the full edge lengths `mean_average_precision3d` expects.

    Args:
        box: Packed ground-truth boxes, shape $(K, 8)$.
        batch: Per-box scene index, shape $(K,)$ (the collate's `batch_box`, not the per-point `batch`).

    Returns:
        Packed `Boxes3D` with boxes of shape $(K, 7)$ as $(c_x, c_y, c_z, d_x, d_y, d_z, \text{heading})$.
    """
    boxes = torch.cat([box[:, :3], 2 * box[:, 3:6], box[:, 6:7]], dim=1)
    return {"boxes": boxes, "labels": box[:, 7].long(), "batch": batch}


class MeanAveragePrecision3D(Metric):
    r"""Packed 3D-detection mean average precision as a `torchmetrics` metric.

    A stateful wrapper of `mean_average_precision3d`: each `update` appends one batch's packed predictions
    and ground truth, and `compute` returns `{"mAP@t": ...}` (averaged over the classes present in the
    targets) for each IoU threshold. The state is a per-process list of batches (not gathered across
    processes), so run detection validation on a single device.

    Args:
        iou_thresholds: IoU thresholds at which `mAP@t` is reported.
        kwargs: Forwarded to `torchmetrics.Metric`.
    """

    preds: List[Detection3D]
    targets: List[Boxes3D]
    higher_is_better = True
    full_state_update = False

    def __init__(self, *, iou_thresholds: Sequence[float] = (0.25, 0.5), **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.iou_thresholds = tuple(iou_thresholds)
        self.add_state("preds", default=[], dist_reduce_fx=None)
        self.add_state("targets", default=[], dist_reduce_fx=None)

    def update(self, preds: Detection3D, target: Boxes3D) -> None:
        self.preds.append(preds)
        self.targets.append(target)

    def compute(self) -> Dict[str, float]:
        return mean_average_precision3d(self.preds, self.targets, iou_thresholds=self.iou_thresholds)
