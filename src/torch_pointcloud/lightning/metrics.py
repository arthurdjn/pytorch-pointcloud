"""Evaluation metrics for 3D detection, instance segmentation, and part segmentation."""

from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence

import torch
from torch import Tensor

from torch_pointcloud.datasets.shapenetpart import ShapeNetPart
from torch_pointcloud.utils.imports import _TORCHMETRICS_GITHUB_URL, optional_import
from torch_pointcloud.utils.metrics import (
    Interpolation,
    average_precision3d,
    instance_average_precision,
    mean_average_precision3d,
    nuscenes_detection_metrics,
    nuscenes_velocity_attributes,
    part_iou,
)
from torch_pointcloud.utils.types import Boxes3D, Detection3D, OptTensor

if TYPE_CHECKING:
    from torchmetrics import Metric
else:
    Metric, _ = optional_import("torchmetrics", "Metric", url=_TORCHMETRICS_GITHUB_URL)


class MeanAveragePrecision3D(Metric):
    r"""Packed 3D-detection mean average precision as a `torchmetrics` metric.

    A stateful wrapper of `mean_average_precision3d`: each `update` appends one batch's packed predictions
    and ground truth, and `compute` returns `{"mAP@t": ...}` (averaged over the classes present in the
    targets) for each IoU threshold. An `ignore_mask` passed to `update` is stored as the predictions'
    `ignore_mask` entry, excluding the flagged predictions from scoring entirely (the KITTI min-height
    rule). The state is a per-process list of batches (not gathered across processes), so run detection
    validation on a single device.

    Args:
        iou_thresholds: IoU thresholds at which `mAP@t` is reported.
        interpolation: AP interpolation: `"all"` integrates the full precision-recall curve; `"r11"` /
            `"r40"` sample the KITTI 11- / 40-point recall grids.
        kwargs: Forwarded to `torchmetrics.Metric`.
    """

    preds: List[Detection3D]
    targets: List[Boxes3D]
    higher_is_better = True
    full_state_update = False

    def __init__(
        self,
        *,
        iou_thresholds: Sequence[float] = (0.25, 0.5),
        interpolation: Interpolation = "all",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.iou_thresholds = tuple(iou_thresholds)
        self.interpolation: Interpolation = interpolation
        self.add_state("preds", default=[], dist_reduce_fx=None)
        self.add_state("targets", default=[], dist_reduce_fx=None)

    def update(self, preds: Detection3D, target: Boxes3D, ignore_mask: OptTensor = None) -> None:
        r"""Append one batch's packed predictions and ground truth.

        Args:
            preds: Packed predictions (one `decode` output), `{"boxes", "scores", "labels", "batch"}`.
            target: Packed ground truth aligned to `preds`, `{"boxes", "labels", "batch"}`.
            ignore_mask: Optional per-prediction ignore mask, shape $(N,)$ bool, stored as the predictions'
                `ignore_mask` entry; flagged predictions are excluded from scoring entirely.
        """
        if ignore_mask is not None:
            preds = {**preds, "ignore_mask": ignore_mask}

        self.preds.append(preds)
        self.targets.append(target)

    def compute(self) -> Dict[str, float]:
        """Score the accumulated batches and return one `mAP@t` entry per IoU threshold."""
        return mean_average_precision3d(
            self.preds,
            self.targets,
            iou_thresholds=self.iou_thresholds,
            interpolation=self.interpolation,
        )


class AveragePrecision3D(Metric):
    r"""Packed 3D-detection per-class average precision as a `torchmetrics` metric.

    A stateful wrapper of `average_precision3d`: each `update` appends one batch's packed predictions and
    ground truth, and `compute` returns one `AP/<class>` entry per class plus their mean as `mAP`, each
    class matched at its own IoU threshold (the KITTI / nuScenes convention, e.g. Car@0.7 and
    Pedestrian/Cyclist@0.5). Targets may carry an `ignore_mask` so predictions overlapping an ignore region
    are not counted as false positives. An `ignore_mask` passed to `update` is stored as the predictions'
    `ignore_mask` entry, excluding the flagged predictions from scoring entirely (the KITTI min-height
    rule). The state is a per-process list of batches (not gathered across processes), so run detection
    validation on a single device.

    Args:
        iou_per_class: Mapping of class index to the IoU threshold used to match its boxes. Keys are
            coerced to `int` (YAML / OmegaConf mappings may arrive with string keys).
        class_names: Optional class names used in the returned keys (defaults to the class index).
        interpolation: AP interpolation: `"all"` integrates the full precision-recall curve; `"r11"` /
            `"r40"` sample the KITTI 11- / 40-point recall grids.
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
        interpolation: Interpolation = "all",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.iou_per_class = {int(key): float(value) for key, value in iou_per_class.items()}
        self.class_names = list(class_names) if class_names is not None else None
        self.interpolation: Interpolation = interpolation
        self.add_state("preds", default=[], dist_reduce_fx=None)
        self.add_state("targets", default=[], dist_reduce_fx=None)

    def update(self, preds: Detection3D, target: Boxes3D, ignore_mask: OptTensor = None) -> None:
        r"""Append one batch's packed predictions and ground truth.

        Args:
            preds: Packed predictions (one `decode` output), `{"boxes", "scores", "labels", "batch"}`.
            target: Packed ground truth aligned to `preds`, `{"boxes", "labels", "batch"}` with an
                optional `ignore_mask`.
            ignore_mask: Optional per-prediction ignore mask, shape $(N,)$ bool, stored as the predictions'
                `ignore_mask` entry; flagged predictions are excluded from scoring entirely.
        """
        if ignore_mask is not None:
            preds = {**preds, "ignore_mask": ignore_mask}

        self.preds.append(preds)
        self.targets.append(target)

    def compute(self) -> Dict[str, float]:
        """Score the accumulated batches and return one `AP/<class>` entry per class plus their `mAP`."""
        return average_precision3d(
            self.preds,
            self.targets,
            iou_per_class=self.iou_per_class,
            class_names=self.class_names,
            interpolation=self.interpolation,
        )


class NuScenesDetection(Metric):
    r"""The official nuScenes detection metrics as a `torchmetrics` metric.

    A stateful wrapper of `nuscenes_detection_metrics`: each `update` appends one batch's packed
    predictions and ground truth together with the optional velocity, attribute and point-count extras,
    and `compute` returns the functional's flat dict (`AP/<class>`, `mAP`, the five TP errors and the
    `NDS`). The predictions' `velocity` entry and the `velocity` argument are appended to the boxes as
    $(v_x, v_y)$ columns, the $(M, 9)$ layout the functional scores; prediction attributes are derived
    from the accumulated prediction velocities in `compute` with the standard speed heuristic
    (`nuscenes_velocity_attributes`). Sample indices are offset per update by the number of samples seen
    so far (read off the batch tensors), so scenes of different updates never collide. The state is a
    per-process list of batches (not gathered across processes), so run detection validation on a
    single device.

    Args:
        class_names: Class name per label index; `barrier` and `traffic_cone` get their official special
            handling by name.
        class_ranges: Maximum BEV evaluation range per class name; defaults to the official ranges.
        dist_thresholds: Matching thresholds in meters the AP is averaged over.
        tp_threshold: Matching threshold in meters of the TP-error metrics.
        max_boxes_per_sample: Per-sample cap on scored predictions (highest scores kept).
        min_recall: Recall up to which the AP and TP-error curves are clipped.
        min_precision: Precision subtracted before the AP mean.
        kwargs: Forwarded to `torchmetrics.Metric`.
    """

    pred_boxes: List[Tensor]
    pred_scores: List[Tensor]
    pred_labels: List[Tensor]
    pred_batch: List[Tensor]
    gt_boxes: List[Tensor]
    gt_labels: List[Tensor]
    gt_batch: List[Tensor]
    gt_num_points: List[Tensor]
    gt_attributes: List[Tensor]
    num_samples: Tensor
    higher_is_better = True
    full_state_update = False

    def __init__(
        self,
        *,
        class_names: Sequence[str],
        class_ranges: Optional[Mapping[str, float]] = None,
        dist_thresholds: Sequence[float] = (0.5, 1.0, 2.0, 4.0),
        tp_threshold: float = 2.0,
        max_boxes_per_sample: int = 500,
        min_recall: float = 0.1,
        min_precision: float = 0.1,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.class_names = list(class_names)
        self.class_ranges = dict(class_ranges) if class_ranges is not None else None
        self.dist_thresholds = tuple(dist_thresholds)
        self.tp_threshold = tp_threshold
        self.max_boxes_per_sample = max_boxes_per_sample
        self.min_recall = min_recall
        self.min_precision = min_precision
        for state in ("pred_boxes", "pred_scores", "pred_labels", "pred_batch"):
            self.add_state(state, default=[], dist_reduce_fx=None)
        for state in ("gt_boxes", "gt_labels", "gt_batch", "gt_num_points", "gt_attributes"):
            self.add_state(state, default=[], dist_reduce_fx=None)
        self.add_state("num_samples", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(
        self,
        preds: Detection3D,
        target: Boxes3D,
        *,
        velocity: OptTensor = None,
        num_points: OptTensor = None,
        attribute: OptTensor = None,
    ) -> None:
        r"""Append one batch's packed predictions and ground truth with the optional nuScenes extras.

        The keyword names match the dataset's ground-truth keys (`velocity`, `num_points`, `attribute`),
        so a Lightning module's `metric_input_keys` passthrough feeds them directly; the prediction-side
        velocity lives inside `preds`.

        Args:
            preds: Packed predictions (one `decode` output), `{"boxes", "scores", "labels", "batch"}`; an
                optional `velocity` entry $(M, 2)$ is appended to the boxes as $(v_x, v_y)$ columns.
            target: Packed ground truth aligned to `preds`, `{"boxes", "labels", "batch"}`.
            velocity: Optional ground-truth per-box BEV velocity, shape $(K, 2)$, appended to the target
                boxes.
            num_points: Optional ground-truth per-box point count, shape $(K,)$; boxes with exactly $0$
                points are removed (unknown counts of $-1$ are kept).
            attribute: Optional ground-truth per-box attribute id, shape $(K,)$; a negative id marks a
                box without an attribute.
        """
        # Validate the all-or-none rule before touching any state, so a raising update leaves the
        # metric usable instead of permanently length-mismatched until `reset()`.
        num_updates = len(self.gt_boxes) + 1
        for name, state, value in (
            ("num_points", self.gt_num_points, num_points),
            ("attribute", self.gt_attributes, attribute),
        ):
            count = len(state) + (value is not None)
            if count not in (0, num_updates):
                raise ValueError(
                    f"`{name}` must be passed on every update or on none; got it on {count} of {num_updates} updates."
                )
        offset = int(self.num_samples.item())
        boxes = preds["boxes"]
        pred_velocity = preds.get("velocity")
        if pred_velocity is not None:
            boxes = torch.cat([boxes, pred_velocity], dim=1)
        self.pred_boxes.append(boxes)
        self.pred_scores.append(preds["scores"])
        self.pred_labels.append(preds["labels"])
        self.pred_batch.append(preds["batch"] + offset)
        boxes = target["boxes"]
        if velocity is not None:
            boxes = torch.cat([boxes, velocity], dim=1)
        self.gt_boxes.append(boxes)
        self.gt_labels.append(target["labels"])
        self.gt_batch.append(target["batch"] + offset)
        if num_points is not None:
            self.gt_num_points.append(num_points)
        if attribute is not None:
            self.gt_attributes.append(attribute)
        pred_samples = int(preds["batch"].max().item()) + 1 if preds["batch"].numel() else 0
        gt_samples = int(target["batch"].max().item()) + 1 if target["batch"].numel() else 0
        self.num_samples += max(pred_samples, gt_samples)

    def compute(self) -> Dict[str, float]:
        """Score the accumulated samples and return the official nuScenes metrics, `NDS` included."""
        device = self.num_samples.device
        empty_boxes = torch.zeros((0, 7), device=device)
        empty_ids = torch.zeros((0,), dtype=torch.long, device=device)
        pred_boxes = torch.cat(self.pred_boxes) if self.pred_boxes else empty_boxes
        pred_labels = torch.cat(self.pred_labels) if self.pred_labels else empty_ids
        pred_attributes = None
        if pred_boxes.shape[1] >= 9:
            pred_attributes = nuscenes_velocity_attributes(
                pred_labels, pred_boxes[:, 7:9], class_names=self.class_names
            )

        return nuscenes_detection_metrics(
            pred_boxes,
            torch.cat(self.pred_scores) if self.pred_scores else empty_boxes.new_zeros((0,)),
            pred_labels,
            torch.cat(self.pred_batch) if self.pred_batch else empty_ids,
            torch.cat(self.gt_boxes) if self.gt_boxes else empty_boxes,
            torch.cat(self.gt_labels) if self.gt_labels else empty_ids,
            torch.cat(self.gt_batch) if self.gt_batch else empty_ids,
            class_names=self.class_names,
            gt_num_points=torch.cat(self.gt_num_points) if self.gt_num_points else None,
            pred_attributes=pred_attributes,
            gt_attributes=torch.cat(self.gt_attributes) if self.gt_attributes else None,
            class_ranges=self.class_ranges,
            dist_thresholds=self.dist_thresholds,
            tp_threshold=self.tp_threshold,
            max_boxes_per_sample=self.max_boxes_per_sample,
            min_recall=self.min_recall,
            min_precision=self.min_precision,
        )


class InstanceAveragePrecision(Metric):
    r"""Point-mask instance-segmentation AP as a `torchmetrics` metric.

    A stateful wrapper of `instance_average_precision`: each `update` appends one scene's
    `instance_matches` record (the compact per-scene reduction of predicted masks against ground-truth
    instances), and `compute` returns `{"AP/<class>": ..., "mAP": ..., "mAP@0.5": ..., "mAP@0.25": ...}`
    following the standard indoor instance-segmentation protocol. The state is a per-process list of
    records (not gathered across processes), so run instance validation on a single device.

    Args:
        num_classes: Number of instance classes.
        class_names: Optional names for the `AP/<class>` keys; falls back to the class index.
        min_points: Minimum point count for a prediction or ground-truth instance to be scored; smaller
            ground-truth instances count as ignore regions.
        kwargs: Forwarded to `torchmetrics.Metric`.
    """

    matches: List[Mapping[str, Tensor]]
    higher_is_better = True
    full_state_update = False

    def __init__(
        self,
        *,
        num_classes: int,
        class_names: Optional[Sequence[str]] = None,
        min_points: int = 100,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.class_names = list(class_names) if class_names is not None else None
        self.min_points = min_points
        self.add_state("matches", default=[], dist_reduce_fx=None)

    def update(self, match: Mapping[str, Tensor]) -> None:
        r"""Append one scene's `instance_matches` record.

        Args:
            match: The per-scene record returned by `instance_matches` (per-instance counts, labels,
                scores and same-class pairwise intersections).
        """
        self.matches.append(match)

    def compute(self) -> Dict[str, float]:
        """Score the accumulated scene records and return the per-class average precisions and their mean."""
        return instance_average_precision(
            self.matches,
            num_classes=self.num_classes,
            class_names=self.class_names,
            min_points=self.min_points,
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
        restrict_to_category: If `True`, the argmax of 2-D `preds` is taken over the shape's own category
            parts only (logits of the other parts are masked out), the protocol of PointNet, Point-MAE and
            Point-M2AE. The default is the global argmax over all parts (DGCNN, PointNeXt).
        kwargs: Forwarded to `torchmetrics.Metric`.
    """

    iou_sum: Tensor
    count: Tensor
    part_to_category: Tensor
    higher_is_better = True
    full_state_update = False

    def __init__(
        self,
        *,
        part_ids: Optional[Sequence[Sequence[int]]] = None,
        restrict_to_category: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.part_ids = [list(ids) for ids in (part_ids if part_ids is not None else ShapeNetPart.seg_ids.values())]
        self.restrict_to_category = restrict_to_category
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
            preds: Predicted part indices $(N,)$, or logits / probabilities $(N, \text{num\_classes})$.
            target: Ground truth part indices, shape $(N,)$.
            batch: Per-point shape index, shape $(N,)$.
        """
        target = target.long()
        batch = batch.long()
        category = torch.zeros(int(batch.max().item()) + 1, dtype=torch.long, device=target.device)
        category[batch] = self.part_to_category[target]

        if preds.dim() == 2:
            if self.restrict_to_category:
                # Only the shape's own parts compete: mask every other part's score before the argmax.
                allowed = self.part_to_category[: preds.size(1)][None, :] == category[batch][:, None]
                preds = preds.masked_fill(~allowed, float("-inf"))
            preds = preds.argmax(dim=1)

        ious = part_iou(preds, target, self.part_ids, category, batch)
        self.iou_sum.index_add_(0, category, ious)
        self.count += torch.bincount(category, minlength=self.count.numel())

    def compute(self) -> Dict[str, Tensor]:
        """Reduce the accumulated per-category IoU sums into `ins_mIoU` and `cls_mIoU`."""
        present = self.count > 0
        if not bool(present.any()):
            zero = self.iou_sum.sum()
            return {"ins_mIoU": zero, "cls_mIoU": zero.clone()}
        ins_miou = self.iou_sum.sum() / self.count.sum()
        cls_miou = (self.iou_sum[present] / self.count[present]).mean()
        return {"ins_mIoU": ins_miou, "cls_mIoU": cls_miou}
