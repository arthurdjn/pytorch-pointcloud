from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence

import torch
from torch import Tensor

from torch_pointcloud.datasets.shapenetpart import ShapeNetPart
from torch_pointcloud.utils.imports import _TORCHMETRICS_GITHUB_URL, optional_import
from torch_pointcloud.utils.metrics import average_precision3d, mean_average_precision3d, part_iou
from torch_pointcloud.utils.types import Boxes3D, Detection3D

if TYPE_CHECKING:
    from torchmetrics import Metric
else:
    Metric, _ = optional_import("torchmetrics", "Metric", url=_TORCHMETRICS_GITHUB_URL)


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


class AveragePrecision3D(Metric):
    r"""Packed 3D-detection per-class average precision as a `torchmetrics` metric.

    A stateful wrapper of `average_precision3d`: each `update` appends one batch's packed predictions and
    ground truth, and `compute` returns one `AP/<class>` entry per class plus their mean as `mAP`, each
    class matched at its own IoU threshold (the KITTI / nuScenes convention, e.g. Car@0.7 and
    Pedestrian/Cyclist@0.5). Targets may carry an `ignore_mask` so predictions overlapping an ignore region
    are not counted as false positives. The state is a per-process list of batches (not gathered across
    processes), so run detection validation on a single device.

    Args:
        iou_per_class: Mapping of class index to the IoU threshold used to match its boxes. Keys are
            coerced to `int` (YAML / OmegaConf mappings may arrive with string keys).
        class_names: Optional class names used in the returned keys (defaults to the class index).
        kwargs: Forwarded to `torchmetrics.Metric`.
    """

    preds: List[Detection3D]
    targets: List[Boxes3D]
    higher_is_better = True
    full_state_update = False

    def __init__(
        self,
        *,
        iou_per_class: Mapping[int, float],
        class_names: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.iou_per_class = {int(key): float(value) for key, value in iou_per_class.items()}
        self.class_names = list(class_names) if class_names is not None else None
        self.add_state("preds", default=[], dist_reduce_fx=None)
        self.add_state("targets", default=[], dist_reduce_fx=None)

    def update(self, preds: Detection3D, target: Boxes3D) -> None:
        self.preds.append(preds)
        self.targets.append(target)

    def compute(self) -> Dict[str, float]:
        return average_precision3d(
            self.preds, self.targets, iou_per_class=self.iou_per_class, class_names=self.class_names
        )


class InstancePartMeanIoU(Metric):
    r"""ShapeNetPart instance / class mean IoU as a `torchmetrics` metric.

    A stateful wrapper of `part_iou`: each shape is scored only over the part labels its category owns
    (a part absent from both the prediction and the target counts as IoU $1$), per-category IoU sums and
    shape counts accumulate across `update` calls (summed across processes), and `compute` returns the
    protocol's two numbers: `ins_mIoU` (mean over shapes) and `cls_mIoU` (mean per category, then over
    the categories seen). The category of each shape is read off its target labels, since every category
    owns a disjoint part range.

    Args:
        part_ids: Part labels owned by each category; defaults to the 16-category / 50-part
            ShapeNetPart table (`ShapeNetPart.seg_ids`).
        kwargs: Forwarded to `torchmetrics.Metric`.
    """

    iou_sum: Tensor
    count: Tensor
    part_to_category: Tensor
    higher_is_better = True
    full_state_update = False

    def __init__(self, *, part_ids: Optional[Sequence[Sequence[int]]] = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.part_ids = [list(ids) for ids in (part_ids if part_ids is not None else ShapeNetPart.seg_ids.values())]
        num_classes = max(max(ids) for ids in self.part_ids) + 1
        part_to_category = torch.full((num_classes,), -1, dtype=torch.long)
        for c, ids in enumerate(self.part_ids):
            part_to_category[ids] = c
        self.register_buffer("part_to_category", part_to_category)
        self.add_state("iou_sum", default=torch.zeros(len(self.part_ids)), dist_reduce_fx="sum")
        self.add_state("count", default=torch.zeros(len(self.part_ids), dtype=torch.long), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor, batch: Tensor) -> None:
        r"""Score one packed batch of shapes.

        Args:
            preds: Predicted part indices $(N,)$, or logits / probabilities $(N, \text{num_classes})$.
            target: Ground truth part indices, shape $(N,)$.
            batch: Per-point shape index, shape $(N,)$.
        """
        if preds.dim() == 2:
            preds = preds.argmax(dim=1)
        target = target.long()
        batch = batch.long()
        category = torch.zeros(int(batch.max().item()) + 1, dtype=torch.long, device=target.device)
        category[batch] = self.part_to_category[target]
        ious = part_iou(preds, target, self.part_ids, category, batch)
        self.iou_sum.index_add_(0, category, ious)
        self.count += torch.bincount(category, minlength=self.count.numel())

    def compute(self) -> Dict[str, Tensor]:
        present = self.count > 0
        ins_miou = self.iou_sum.sum() / self.count.sum()
        cls_miou = (self.iou_sum[present] / self.count[present]).mean()
        return {"ins_mIoU": ins_miou, "cls_mIoU": cls_miou}
