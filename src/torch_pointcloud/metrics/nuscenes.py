"""The nuScenes detection metrics: center-distance average precision, true-positive errors and the NDS."""

import math
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

_TP_KEYS = ("trans", "scale", "orient", "vel", "attr")


def _cummean(values: np.ndarray) -> np.ndarray:
    """Cumulative mean over the non-NaN entries; an all-NaN input yields the full-error sentinel of ones."""
    valid = ~np.isnan(values)
    if not valid.any():
        return np.ones(len(values))
    count = np.cumsum(valid)
    total = np.nancumsum(values)
    return np.divide(total, count, out=np.zeros_like(total), where=count > 0)


def _top_score_mask(scores: Tensor, batch: Tensor, max_boxes: int) -> Tensor:
    """Boolean mask keeping each sample's `max_boxes` highest-scoring entries."""
    order = torch.argsort(scores, descending=True, stable=True)
    grouped = order[torch.argsort(batch[order], stable=True)]
    counts = torch.bincount(batch)
    starts = torch.cumsum(counts, dim=0) - counts
    rank = torch.arange(batch.numel(), device=batch.device) - starts.repeat_interleave(counts)
    keep = torch.zeros(batch.numel(), dtype=torch.bool, device=batch.device)
    keep[grouped[rank < max_boxes]] = True
    return keep


def _nuscenes_accumulate(
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    pred_batch: np.ndarray,
    gt_boxes: np.ndarray,
    gt_batch: np.ndarray,
    pred_attributes: Optional[np.ndarray],
    gt_attributes: Optional[np.ndarray],
    dist_threshold: float,
    period: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    r"""Greedy center-distance matching for one class: 101-point precision, confidence and TP-error curves.

    Predictions in descending score order each take the closest not-yet-matched ground-truth box of their
    sample by BEV center distance, a match requiring a distance strictly below `dist_threshold`. The
    cumulative precision, confidence and per-match error curves are interpolated at the 101 recall points
    $0.00, 0.01, \ldots, 1.00$; without any match the sentinel curves are returned (zero precision and
    confidence, all errors $1$).
    """
    sentinel: Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]] = (
        np.zeros(101),
        np.zeros(101),
        {key: np.ones(101) for key in _TP_KEYS},
    )
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return sentinel
    has_velocity = pred_boxes.shape[1] >= 9 and gt_boxes.shape[1] >= 9

    order = np.argsort(-pred_scores, kind="stable")
    matched = np.zeros(len(gt_boxes), dtype=bool)
    tp = np.zeros(len(order))
    errors: Dict[str, List[float]] = {key: [] for key in _TP_KEYS}
    match_conf: List[float] = []
    for position, index in enumerate(order):
        candidates = np.flatnonzero((gt_batch == pred_batch[index]) & ~matched)
        if len(candidates) == 0:
            continue
        dist = np.linalg.norm(gt_boxes[candidates, :2] - pred_boxes[index, :2], axis=1)
        best = int(np.argmin(dist))
        if float(dist[best]) >= dist_threshold:
            continue
        gt, pred = gt_boxes[candidates[best]], pred_boxes[index]
        matched[candidates[best]] = True
        tp[position] = 1.0
        errors["trans"].append(float(dist[best]))
        intersection = float(np.prod(np.minimum(gt[3:6], pred[3:6])))
        union = float(np.prod(gt[3:6]) + np.prod(pred[3:6])) - intersection
        errors["scale"].append(1.0 - intersection / union)
        yaw = float(gt[6] - pred[6])
        errors["orient"].append(abs((yaw + period / 2) % period - period / 2))
        errors["vel"].append(float(np.linalg.norm(gt[7:9] - pred[7:9])) if has_velocity else float("nan"))
        if pred_attributes is None or gt_attributes is None or int(gt_attributes[candidates[best]]) < 0:
            errors["attr"].append(float("nan"))
        else:
            errors["attr"].append(float(int(gt_attributes[candidates[best]]) != int(pred_attributes[index])))
        match_conf.append(float(pred_scores[index]))
    if not match_conf:
        return sentinel

    grid = np.linspace(0.0, 1.0, 101)
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(1.0 - tp)
    recall = tp_cum / len(gt_boxes)
    precision = np.interp(grid, recall, tp_cum / (tp_cum + fp_cum), right=0)
    confidence = np.interp(grid, recall, pred_scores[order], right=0)
    conf_match = np.array(match_conf)
    curves: Dict[str, np.ndarray] = {}
    for key, values in errors.items():
        cumulative = _cummean(np.array(values, dtype=np.float64))
        curves[key] = np.interp(confidence[::-1], conf_match[::-1], cumulative[::-1])[::-1]
    return precision, confidence, curves


def _nuscenes_ap(precision: np.ndarray, min_recall: float, min_precision: float) -> float:
    r"""AP over a 101-point interpolated precision curve: drop recalls $\le$ `min_recall`, subtract
    `min_precision`, clamp at $0$, average and rescale by the remaining precision span."""
    clipped = np.clip(precision[round(100 * min_recall) + 1 :] - min_precision, 0.0, None)
    return float(np.mean(clipped) / (1.0 - min_precision))


def _nuscenes_tp_error(curve: np.ndarray, confidence: np.ndarray, min_recall: float) -> float:
    """Mean of a 101-point TP-error curve from just above `min_recall` to the highest achieved recall
    (the last nonzero confidence); `1.0` when that recall does not exceed `min_recall`."""
    first = round(100 * min_recall) + 1
    nonzero = np.flatnonzero(confidence)
    last = int(nonzero[-1]) if len(nonzero) else 0
    if last < first:
        return 1.0
    return float(np.mean(curve[first : last + 1]))


def filter_boxes_by_range(boxes: Tensor, labels: Tensor, ranges: Sequence[float]) -> Tensor:
    r"""Mask of boxes whose BEV center distance from the sensor origin is strictly below their class range.

    Args:
        boxes: Boxes $(N, 7)$ or $(N, 9)$ of $(c_x, c_y, c_z, d_x, d_y, d_z, \theta[, v_x, v_y])$.
        labels: Per-box class index into `ranges`, shape $(N,)$.
        ranges: Maximum BEV range per class index, in the coordinate unit.

    Returns:
        Boolean keep mask of shape $(N,)$.

    Example:
        ```pycon
        >>> boxes = torch.tensor([[3.0, 4, 0, 4, 2, 1.5, 0], [0, 41, 0, 0.5, 0.5, 1, 0]])
        >>> filter_boxes_by_range(boxes, torch.tensor([0, 1]), ranges=[50.0, 40.0])
        tensor([ True, False])

        ```
    """
    limits = boxes.new_tensor(list(ranges))[labels.long()]
    return torch.linalg.norm(boxes[:, :2], dim=1) < limits


def nuscenes_detection_metrics(
    pred_boxes: Tensor,
    pred_scores: Tensor,
    pred_labels: Tensor,
    pred_batch: Tensor,
    gt_boxes: Tensor,
    gt_labels: Tensor,
    gt_batch: Tensor,
    *,
    class_names: Sequence[str],
    gt_num_points: Optional[Tensor] = None,
    pred_attributes: Optional[Tensor] = None,
    gt_attributes: Optional[Tensor] = None,
    class_ranges: Optional[Mapping[str, float]] = None,
    dist_thresholds: Sequence[float] = (0.5, 1.0, 2.0, 4.0),
    tp_threshold: float = 2.0,
    max_boxes_per_sample: int = 500,
    min_recall: float = 0.1,
    min_precision: float = 0.1,
) -> Dict[str, float]:
    r"""The nuScenes detection metrics: per-class AP, mAP, the five TP errors and the NDS.

    Follows the official protocol of the nuScenes benchmark
    ([nuScenes: A Multimodal Dataset for Autonomous Driving](https://arxiv.org/abs/1903.11027)).
    Predictions are matched per sample and class by BEV center distance: in descending score order each
    prediction greedily takes the closest still-unmatched ground-truth box strictly below the threshold.
    AP interpolates precision at 101 recall points $0.00, 0.01, \ldots, 1.00$, drops recalls up to
    `min_recall`, subtracts `min_precision`, clamps at $0$, averages and rescales by the remaining
    precision span; `mAP` averages over `class_names` and `dist_thresholds`. The TP errors ATE (BEV
    center distance), ASE ($1 - $ IoU of center- and yaw-aligned boxes), AOE (absolute yaw difference,
    modulo $\pi$ for `barrier`), AVE (L2 xy-velocity difference) and AAE ($1 - $ attribute accuracy)
    average the cumulative-mean error curve of the `tp_threshold` matches from `min_recall` to the
    highest achieved recall; a class without matches scores the full error of $1$. The officially
    excluded pairs (`traffic_cone`: AOE/AVE/AAE, `barrier`: AVE/AAE) are left out of the per-metric
    means, and $\text{NDS} = (5 \cdot \text{mAP} + \sum_\text{tp} (1 - \min(1, \text{err}))) / 10$.

    Boxes are filtered before scoring: each sample keeps its `max_boxes_per_sample` highest-scoring
    predictions, boxes farther from the sensor origin (BEV) than their class range are dropped on both
    sides, and ground-truth boxes with `gt_num_points == 0` are removed. When velocity columns or
    attributes are absent (on either side), AVE / AAE fall back to the full penalty of $1.0$ per class.

    Args:
        pred_boxes: Predicted boxes $(M, 7)$ of $(c_x, c_y, c_z, d_x, d_y, d_z, \theta)$, or $(M, 9)$
            with $(v_x, v_y)$ velocity columns appended.
        pred_scores: Per-box confidence, shape $(M,)$.
        pred_labels: Per-box class index into `class_names`, shape $(M,)$.
        pred_batch: Per-box sample index, shape $(M,)$.
        gt_boxes: Ground-truth boxes $(K, 7)$ or $(K, 9)$, like `pred_boxes`.
        gt_labels: Per-box class index into `class_names`, shape $(K,)$.
        gt_batch: Per-box sample index, shape $(K,)$.
        class_names: Class name per label index; `barrier` and `traffic_cone` get their official special
            handling by name.
        gt_num_points: Optional per-box point count, shape $(K,)$; boxes with exactly $0$ points are
            removed (unknown counts of $-1$ are kept).
        pred_attributes: Optional per-box attribute id, shape $(M,)$. Without it AAE is $1.0$.
        gt_attributes: Optional per-box attribute id, shape $(K,)$; a negative id marks a box without an
            attribute, which is skipped in the AAE mean. Without it AAE is $1.0$.
        class_ranges: Maximum BEV evaluation range per class name; defaults to the official ranges (50 m
            car/truck/bus/trailer/construction_vehicle, 40 m pedestrian/motorcycle/bicycle, 30 m
            traffic_cone/barrier). A name missing from the mapping is not range-filtered.
        dist_thresholds: Matching thresholds in meters the AP is averaged over.
        tp_threshold: Matching threshold in meters of the TP-error metrics.
        max_boxes_per_sample: Per-sample cap on scored predictions (highest scores kept).
        min_recall: Recall up to which the AP and TP-error curves are clipped.
        min_precision: Precision subtracted before the AP mean.

    Returns:
        A flat dict with `AP/<class>` (averaged over `dist_thresholds`), `mAP`, `mATE`, `mASE`, `mAOE`,
        `mAVE`, `mAAE` and `NDS`.

    Example:
        ```pycon
        >>> zero = torch.tensor([0])
        >>> pred_boxes = torch.tensor([[0.25, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0]])
        >>> gt_boxes = torch.tensor([[0.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0]])
        >>> metrics = nuscenes_detection_metrics(
        ...     pred_boxes, torch.tensor([0.9]), zero, zero, gt_boxes, zero, zero, class_names=["car"]
        ... )
        >>> f"{metrics['AP/car']:.2f} {metrics['mATE']:.2f} {metrics['NDS']:.3f}"
        '1.00 0.25 0.775'

        ```
    """
    if class_ranges is None:
        class_ranges = {
            "car": 50.0,
            "truck": 50.0,
            "bus": 50.0,
            "trailer": 50.0,
            "construction_vehicle": 50.0,
            "pedestrian": 40.0,
            "motorcycle": 40.0,
            "bicycle": 40.0,
            "traffic_cone": 30.0,
            "barrier": 30.0,
        }
    limits = [float(class_ranges.get(name, math.inf)) for name in class_names]

    pred_keep = _top_score_mask(pred_scores, pred_batch, max_boxes_per_sample)
    pred_keep &= filter_boxes_by_range(pred_boxes, pred_labels, limits)
    gt_keep = filter_boxes_by_range(gt_boxes, gt_labels, limits)
    if gt_num_points is not None:
        gt_keep &= gt_num_points != 0

    def to_numpy(tensor: Tensor) -> np.ndarray:
        return tensor.detach().cpu().numpy()

    pred_box, pred_score = to_numpy(pred_boxes[pred_keep]), to_numpy(pred_scores[pred_keep])
    pred_label, pred_sample = to_numpy(pred_labels[pred_keep]), to_numpy(pred_batch[pred_keep])
    gt_box, gt_label, gt_sample = to_numpy(gt_boxes[gt_keep]), to_numpy(gt_labels[gt_keep]), to_numpy(gt_batch[gt_keep])
    pred_attribute = to_numpy(pred_attributes[pred_keep]) if pred_attributes is not None else None
    gt_attribute = to_numpy(gt_attributes[gt_keep]) if gt_attributes is not None else None

    out: Dict[str, float] = {}
    aps: List[float] = []
    tp_errors: Dict[str, List[float]] = {key: [] for key in _TP_KEYS}
    for index, name in enumerate(class_names):
        pred_mask = pred_label == index
        gt_mask = gt_label == index
        period = math.pi if name == "barrier" else 2 * math.pi
        pred_attribute_cls = pred_attribute[pred_mask] if pred_attribute is not None else None
        gt_attribute_cls = gt_attribute[gt_mask] if gt_attribute is not None else None

        class_aps: List[float] = []
        for threshold in dist_thresholds:
            precision, _, _ = _nuscenes_accumulate(
                pred_box[pred_mask],
                pred_score[pred_mask],
                pred_sample[pred_mask],
                gt_box[gt_mask],
                gt_sample[gt_mask],
                pred_attribute_cls,
                gt_attribute_cls,
                float(threshold),
                period,
            )
            class_aps.append(_nuscenes_ap(precision, min_recall, min_precision))

        out[f"AP/{name}"] = float(np.mean(class_aps)) if class_aps else 0.0
        aps.extend(class_aps)

        _, confidence, error_curves = _nuscenes_accumulate(
            pred_box[pred_mask],
            pred_score[pred_mask],
            pred_sample[pred_mask],
            gt_box[gt_mask],
            gt_sample[gt_mask],
            pred_attribute_cls,
            gt_attribute_cls,
            tp_threshold,
            period,
        )

        excluded: Tuple[str, ...]
        if name == "traffic_cone":
            excluded = ("orient", "vel", "attr")
        elif name == "barrier":
            excluded = ("vel", "attr")
        else:
            excluded = ()
        for key in _TP_KEYS:
            if key not in excluded:
                tp_errors[key].append(_nuscenes_tp_error(error_curves[key], confidence, min_recall))

    out["mAP"] = float(np.mean(aps)) if aps else 0.0
    score_sum = 0.0
    for key, metric_name in zip(_TP_KEYS, ("mATE", "mASE", "mAOE", "mAVE", "mAAE")):
        error = float(np.mean(tp_errors[key])) if tp_errors[key] else 1.0
        out[metric_name] = error
        score_sum += max(0.0, 1.0 - error)
    out["NDS"] = (5.0 * out["mAP"] + score_sum) / 10.0
    return out


def nuscenes_velocity_attributes(
    labels: Tensor,
    velocity: Tensor,
    *,
    class_names: Sequence[str],
    speed_threshold: float = 1.0,
) -> Tensor:
    r"""Derive per-box nuScenes attribute ids from predicted velocities (the standard speed heuristic).

    A box moving faster than `speed_threshold` (BEV speed, m/s) gets its class's moving attribute, a
    slower box the parked / stopped / standing default; `barrier` and `traffic_cone` carry no attribute
    (id $-1$). The returned ids index the official 8-entry attribute table (`attribute.json` order), the
    id space of the `pred_attributes` / `gt_attributes` arguments of `nuscenes_detection_metrics`.

    Args:
        labels: Per-box class index into `class_names`, shape $(M,)$ long.
        velocity: Per-box BEV velocity $(v_x, v_y)$, shape $(M, 2)$.
        class_names: Class name per label index (the official 10 detection class names).
        speed_threshold: BEV speed in m/s above which a box counts as moving.

    Returns:
        Per-box attribute id, shape $(M,)$ long, $-1$ for classes without attributes.

    Shape:
        - labels: $(M,)$
        - velocity: $(M, 2)$
        - output: $(M,)$

    Example:
        ```pycon
        >>> labels = torch.tensor([0, 0, 1])
        >>> velocity = torch.tensor([[3.0, 0.0], [0.5, 0.0], [2.0, 0.0]])
        >>> nuscenes_velocity_attributes(labels, velocity, class_names=("car", "barrier")).tolist()
        [0, 2, -1]

        ```
    """
    attribute_names = (
        "vehicle.moving",
        "vehicle.stopped",
        "vehicle.parked",
        "cycle.with_rider",
        "cycle.without_rider",
        "pedestrian.sitting_lying_down",
        "pedestrian.standing",
        "pedestrian.moving",
    )
    moving_attribute = {
        "car": "vehicle.moving",
        "truck": "vehicle.moving",
        "construction_vehicle": "vehicle.moving",
        "bus": "vehicle.moving",
        "trailer": "vehicle.moving",
        "motorcycle": "cycle.with_rider",
        "bicycle": "cycle.with_rider",
        "pedestrian": "pedestrian.moving",
    }
    stopped_attribute = {
        "car": "vehicle.parked",
        "truck": "vehicle.parked",
        "construction_vehicle": "vehicle.parked",
        "bus": "vehicle.stopped",
        "trailer": "vehicle.parked",
        "motorcycle": "cycle.without_rider",
        "bicycle": "cycle.without_rider",
        "pedestrian": "pedestrian.standing",
    }

    moving = torch.linalg.norm(velocity, dim=1) > speed_threshold
    attributes = torch.full_like(labels, -1)
    for index, name in enumerate(class_names):
        if name not in moving_attribute:
            continue

        mask = labels == index
        attributes[mask & moving] = attribute_names.index(moving_attribute[name])
        attributes[mask & ~moving] = attribute_names.index(stopped_attribute[name])

    return attributes
