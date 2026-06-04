r"""Detection-AP evaluation helpers shared by the VoteNet benchmarks and the training harness.

Faithful NumPy port of the evaluation pipeline from :github:
[facebookresearch/votenet](https://github.com/facebookresearch/votenet)
(`models/ap_helper.py`, `utils/box_util.py`, `utils/nms.py`, `utils/eval_det.py`),
adapted to consume the dense $(B, K, \cdot)$ proposal dict produced by
`torch_pointcloud.models.VoteNet`. Predictions and ground truth are scored in the
original upright-camera convention so the numbers match the reference.

Used by `examples/votenet_benchmark_scannet.py`, `examples/votenet_benchmark_sunrgbd.py`
and `torch_pointcloud.lightning.LitDetectionModel`.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.spatial import ConvexHull, Delaunay
from torch import Tensor


@dataclass
class DatasetConfig:
    r"""Minimal box-decode config (`class2angle` / `class2size`) for one dataset.

    Args:
        num_class: Number of semantic classes.
        num_heading_bin: Number of heading-angle bins.
        num_size_cluster: Number of size templates.
        mean_size_arr: Per-template mean box size, shape $(\text{num\_size\_cluster}, 3)$.
        oriented: Whether boxes carry a heading. SUN RGB-D boxes are oriented, ScanNet boxes are axis-aligned.
    """

    num_class: int
    num_heading_bin: int
    num_size_cluster: int
    mean_size_arr: np.ndarray
    oriented: bool

    def class2angle(self, pred_cls: int, residual: float) -> float:
        if self.num_heading_bin == 1:
            return 0.0
        angle_per_class = 2 * np.pi / float(self.num_heading_bin)
        angle = pred_cls * angle_per_class + residual
        if angle > np.pi:
            angle = angle - 2 * np.pi
        return float(angle)

    def class2size(self, pred_cls: int, residual: np.ndarray) -> np.ndarray:
        return self.mean_size_arr[pred_cls] + residual


def flip_axis_to_camera(pc: np.ndarray) -> np.ndarray:
    """Depth (X right, Y forward, Z up) to camera (X right, Y down, Z forward)."""
    pc2 = np.copy(pc)
    pc2[..., [0, 1, 2]] = pc2[..., [0, 2, 1]]
    pc2[..., 1] *= -1
    return pc2


def flip_axis_to_depth(pc: np.ndarray) -> np.ndarray:
    """Inverse of `flip_axis_to_camera`: camera to depth coords."""
    pc2 = np.copy(pc)
    pc2[..., [0, 1, 2]] = pc2[..., [0, 2, 1]]
    pc2[..., 2] *= -1
    return pc2


def roty(t: float) -> np.ndarray:
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def get_3d_box(box_size: np.ndarray, heading_angle: float, center: np.ndarray) -> np.ndarray:
    """8x3 corners of a box from $(l, w, h)$, heading (rad about up axis) and center."""
    r = roty(heading_angle)
    length, width, height = box_size
    x_corners = [length / 2, length / 2, -length / 2, -length / 2, length / 2, length / 2, -length / 2, -length / 2]
    y_corners = [height / 2, height / 2, height / 2, height / 2, -height / 2, -height / 2, -height / 2, -height / 2]
    z_corners = [width / 2, -width / 2, -width / 2, width / 2, width / 2, -width / 2, -width / 2, width / 2]
    corners = np.dot(r, np.vstack([x_corners, y_corners, z_corners]))
    corners[0, :] += center[0]
    corners[1, :] += center[1]
    corners[2, :] += center[2]
    return np.transpose(corners)


def polygon_clip(subject_polygon: List[Any], clip_polygon: List[Any]) -> Optional[List[Any]]:
    """Sutherland-Hodgman clip of `subject_polygon` by convex `clip_polygon` (CCW)."""

    def inside(p: Any) -> bool:
        return (cp2[0] - cp1[0]) * (p[1] - cp1[1]) > (cp2[1] - cp1[1]) * (p[0] - cp1[0])

    def intersection() -> List[float]:
        dc = [cp1[0] - cp2[0], cp1[1] - cp2[1]]
        dp = [s[0] - e[0], s[1] - e[1]]
        n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
        n2 = s[0] * e[1] - s[1] * e[0]
        n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
        return [(n1 * dp[0] - n2 * dc[0]) * n3, (n1 * dp[1] - n2 * dc[1]) * n3]

    output_list = subject_polygon
    cp1 = clip_polygon[-1]
    for clip_vertex in clip_polygon:
        cp2 = clip_vertex
        input_list = output_list
        output_list = []
        s = input_list[-1]
        for subject_vertex in input_list:
            e = subject_vertex
            if inside(e):
                if not inside(s):
                    output_list.append(intersection())
                output_list.append(e)
            elif inside(s):
                output_list.append(intersection())
            s = e
        cp1 = cp2
        if len(output_list) == 0:
            return None
    return output_list


def convex_hull_intersection(p1: List[Any], p2: List[Any]) -> Tuple[Any, float]:
    inter_p = polygon_clip(p1, p2)
    if inter_p is not None and len(inter_p) >= 3 and np.isfinite(np.asarray(inter_p)).all():
        try:
            return inter_p, float(ConvexHull(inter_p).volume)
        except Exception:
            return None, 0.0
    return None, 0.0


def box3d_vol(corners: np.ndarray) -> float:
    a = np.sqrt(np.sum((corners[0] - corners[1]) ** 2))
    b = np.sqrt(np.sum((corners[1] - corners[2]) ** 2))
    c = np.sqrt(np.sum((corners[0] - corners[4]) ** 2))
    return float(a * b * c)


def box3d_iou(corners1: np.ndarray, corners2: np.ndarray) -> float:
    """3D IoU of two 8x3 boxes in camera coords (up = -Y), matching the reference."""
    rect1 = [(corners1[i, 0], corners1[i, 2]) for i in range(3, -1, -1)]
    rect2 = [(corners2[i, 0], corners2[i, 2]) for i in range(3, -1, -1)]
    _, inter_area = convex_hull_intersection(rect1, rect2)
    ymax = min(corners1[0, 1], corners2[0, 1])
    ymin = max(corners1[4, 1], corners2[4, 1])
    inter_vol = inter_area * max(0.0, ymax - ymin)
    vol1 = box3d_vol(corners1)
    vol2 = box3d_vol(corners2)
    return float(inter_vol / (vol1 + vol2 - inter_vol))


def in_hull(p: np.ndarray, hull: np.ndarray) -> np.ndarray:
    delaunay = hull if isinstance(hull, Delaunay) else Delaunay(hull)
    return delaunay.find_simplex(p) >= 0


def extract_pc_in_box3d(pc: np.ndarray, box3d: np.ndarray) -> int:
    return int(np.count_nonzero(in_hull(pc[:, 0:3], box3d)))


def nms_3d_faster_samecls(boxes: np.ndarray, overlap_threshold: float) -> List[int]:
    """Axis-aligned 3D NMS that only suppresses boxes of the same class."""
    x1, y1, z1 = boxes[:, 0], boxes[:, 1], boxes[:, 2]
    x2, y2, z2 = boxes[:, 3], boxes[:, 4], boxes[:, 5]
    score, cls = boxes[:, 6], boxes[:, 7]
    area = (x2 - x1) * (y2 - y1) * (z2 - z1)

    order = np.argsort(score)
    pick: List[int] = []
    while order.size != 0:
        last = order.size
        i = order[-1]
        pick.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[: last - 1]])
        yy1 = np.maximum(y1[i], y1[order[: last - 1]])
        zz1 = np.maximum(z1[i], z1[order[: last - 1]])
        xx2 = np.minimum(x2[i], x2[order[: last - 1]])
        yy2 = np.minimum(y2[i], y2[order[: last - 1]])
        zz2 = np.minimum(z2[i], z2[order[: last - 1]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1) * np.maximum(0, zz2 - zz1)
        iou = inter / (area[i] + area[order[: last - 1]] - inter)
        iou = iou * (cls[i] == cls[order[: last - 1]])
        order = np.delete(order, np.concatenate(([last - 1], np.where(iou > overlap_threshold)[0])))
    return pick


def voc_ap(rec: np.ndarray, prec: np.ndarray) -> float:
    """VOC average precision (the all-points / area-under-PR variant)."""
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def eval_det_cls(
    pred: Dict[int, List[Tuple[np.ndarray, float]]],
    gt: Dict[int, List[np.ndarray]],
    ovthresh: float,
) -> float:
    """Average precision for a single class given per-scene predictions and ground truth."""
    class_recs: Dict[int, Dict[str, Any]] = {}
    npos = 0
    for img_id in gt:
        bbox = np.array(gt[img_id])
        class_recs[img_id] = {"bbox": bbox, "det": [False] * len(bbox)}
        npos += len(bbox)
    for img_id in pred:
        if img_id not in gt:
            class_recs[img_id] = {"bbox": np.array([]), "det": []}

    image_ids: List[int] = []
    confidence: List[float] = []
    boxes: List[np.ndarray] = []
    for img_id in pred:
        for box, score in pred[img_id]:
            image_ids.append(img_id)
            confidence.append(score)
            boxes.append(box)
    if len(boxes) == 0:
        return 0.0
    confidence_arr = np.array(confidence)
    boxes_arr = np.array(boxes)

    order = np.argsort(-confidence_arr)
    boxes_arr = boxes_arr[order, ...]
    image_ids = [image_ids[x] for x in order]

    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        rec = class_recs[image_ids[d]]
        bb = boxes_arr[d, ...].astype(float)
        ovmax = -np.inf
        bbgt = rec["bbox"].astype(float)
        jmax = -1
        for j in range(bbgt.shape[0]):
            iou = box3d_iou(bb, bbgt[j, ...])
            if iou > ovmax:
                ovmax = iou
                jmax = j
        if ovmax > ovthresh and not rec["det"][jmax]:
            tp[d] = 1.0
            rec["det"][jmax] = True
        else:
            fp[d] = 1.0

    fp_cum = np.cumsum(fp)
    tp_cum = np.cumsum(tp)
    rec_arr = tp_cum / float(max(npos, 1))
    prec_arr = tp_cum / np.maximum(tp_cum + fp_cum, np.finfo(np.float64).eps)
    return voc_ap(rec_arr, prec_arr)


def eval_det(
    pred_all: Dict[int, List[Tuple[int, np.ndarray, float]]],
    gt_all: Dict[int, List[Tuple[int, np.ndarray]]],
    ovthresh: float,
) -> Dict[int, float]:
    """Per-class AP across all scenes. Returns `{class: ap}`."""
    pred: Dict[int, Dict[int, List[Tuple[np.ndarray, float]]]] = {}
    gt: Dict[int, Dict[int, List[np.ndarray]]] = {}
    for img_id in pred_all:
        for classname, bbox, score in pred_all[img_id]:
            pred.setdefault(classname, {}).setdefault(img_id, []).append((bbox, score))
    for img_id in gt_all:
        for classname, bbox in gt_all[img_id]:
            gt.setdefault(classname, {}).setdefault(img_id, []).append(bbox)

    ap: Dict[int, float] = {}
    for classname in gt:
        ap[classname] = eval_det_cls(pred.get(classname, {}), gt[classname], ovthresh) if classname in pred else 0.0
    return ap


EVAL_CONFIG: Dict[str, Any] = {
    "remove_empty_box": True,
    "nms_iou": 0.25,
    "conf_thresh": 0.05,
    "min_points_in_box": 5,
}


def parse_predictions(
    output: Dict[str, Tensor],
    point_clouds: Tensor,
    config: DatasetConfig,
    eval_config: Dict[str, Any] = EVAL_CONFIG,
) -> List[List[Tuple[int, np.ndarray, float]]]:
    """Decode boxes, run 3D per-class NMS, and emit `[(sem_cls, corners(8,3), score)]` per scene."""
    pred_center = output["center"].detach().cpu().numpy()
    heading_class = torch.argmax(output["heading_scores"], -1).detach().cpu().numpy()
    heading_residual = (
        torch.gather(output["heading_residuals"], 2, torch.argmax(output["heading_scores"], -1, keepdim=True))
        .squeeze(2)
        .detach()
        .cpu()
        .numpy()
    )
    size_class = torch.argmax(output["size_scores"], -1)
    size_residual = (
        torch.gather(output["size_residuals"], 2, size_class[:, :, None, None].repeat(1, 1, 1, 3))
        .squeeze(2)
        .detach()
        .cpu()
        .numpy()
    )
    size_class_np = size_class.detach().cpu().numpy()
    sem_cls = torch.argmax(output["sem_cls_scores"], -1).detach().cpu().numpy()
    sem_cls_probs = _softmax(output["sem_cls_scores"].detach().cpu().numpy())
    obj_prob = _softmax(output["objectness_scores"].detach().cpu().numpy())[:, :, 1]

    bsize, num_proposal = pred_center.shape[0], pred_center.shape[1]
    batch_pc = point_clouds.detach().cpu().numpy()[:, :, 0:3]
    pred_corners = np.zeros((bsize, num_proposal, 8, 3))
    center_cam = flip_axis_to_camera(pred_center)
    for i in range(bsize):
        for j in range(num_proposal):
            angle = config.class2angle(int(heading_class[i, j]), float(heading_residual[i, j]))
            box_size = config.class2size(int(size_class_np[i, j]), size_residual[i, j])
            pred_corners[i, j] = get_3d_box(box_size, angle, center_cam[i, j])

    results: List[List[Tuple[int, np.ndarray, float]]] = []
    for i in range(bsize):
        nonempty = np.ones(num_proposal, dtype=bool)
        if eval_config["remove_empty_box"]:
            pc = batch_pc[i]
            for j in range(num_proposal):
                box_depth = flip_axis_to_depth(pred_corners[i, j])
                if extract_pc_in_box3d(pc, box_depth) < eval_config["min_points_in_box"]:
                    nonempty[j] = False

        boxes_with_prob = np.zeros((num_proposal, 8))
        boxes_with_prob[:, 0] = pred_corners[i, :, :, 0].min(axis=1)
        boxes_with_prob[:, 1] = pred_corners[i, :, :, 1].min(axis=1)
        boxes_with_prob[:, 2] = pred_corners[i, :, :, 2].min(axis=1)
        boxes_with_prob[:, 3] = pred_corners[i, :, :, 0].max(axis=1)
        boxes_with_prob[:, 4] = pred_corners[i, :, :, 1].max(axis=1)
        boxes_with_prob[:, 5] = pred_corners[i, :, :, 2].max(axis=1)
        boxes_with_prob[:, 6] = obj_prob[i]
        boxes_with_prob[:, 7] = sem_cls[i]
        keep_idx = np.where(nonempty)[0]
        pred_mask = np.zeros(num_proposal, dtype=bool)
        if keep_idx.size > 0:
            pick = nms_3d_faster_samecls(boxes_with_prob[nonempty], eval_config["nms_iou"])
            pred_mask[keep_idx[pick]] = True

        scene_pred: List[Tuple[int, np.ndarray, float]] = []
        for ii in range(config.num_class):
            for j in range(num_proposal):
                if pred_mask[j] and obj_prob[i, j] > eval_config["conf_thresh"]:
                    scene_pred.append((ii, pred_corners[i, j], float(sem_cls_probs[i, j, ii] * obj_prob[i, j])))
        results.append(scene_pred)
    return results


def parse_groundtruths(
    center_label: np.ndarray,
    size_class_label: np.ndarray,
    size_residual_label: np.ndarray,
    heading_class_label: np.ndarray,
    heading_residual_label: np.ndarray,
    sem_cls_label: np.ndarray,
    box_label_mask: np.ndarray,
    config: DatasetConfig,
) -> List[Tuple[int, np.ndarray]]:
    """Decode one scene's ground-truth boxes into `[(sem_cls, corners(8,3))]`."""
    center_cam = flip_axis_to_camera(center_label)
    out: List[Tuple[int, np.ndarray]] = []
    for j in range(center_label.shape[0]):
        if box_label_mask[j] == 0:
            continue
        angle = config.class2angle(int(heading_class_label[j]), float(heading_residual_label[j]))
        box_size = config.class2size(int(size_class_label[j]), size_residual_label[j])
        corners = get_3d_box(box_size, angle, center_cam[j])
        out.append((int(sem_cls_label[j]), corners))
    return out


def corners_from_boxes(boxes: np.ndarray, half_sizes: bool = True) -> List[Tuple[int, np.ndarray]]:
    r"""Convert detection ground-truth boxes to `[(sem_cls, corners(8, 3))]` for AP scoring.

    Args:
        boxes: $(K, 8)$ rows $[c_x, c_y, c_z, d_x, d_y, d_z, \text{heading}, \text{cls}]$ in the upright
            depth frame, with `heading` in radians about the up axis.
        half_sizes: Whether $d_x, d_y, d_z$ are half extents (doubled to full edge lengths) or already full.

    Returns:
        One `(sem_cls, corners(8, 3))` tuple per box, in the camera frame the IoU uses.
    """
    centers_cam = flip_axis_to_camera(boxes[:, 0:3])
    scale = 2.0 if half_sizes else 1.0
    out: List[Tuple[int, np.ndarray]] = []
    for i in range(boxes.shape[0]):
        corners = get_3d_box(boxes[i, 3:6] * scale, float(boxes[i, 6]), centers_cam[i])
        out.append((int(boxes[i, 7]), corners))
    return out


class APCalculator:
    """Accumulates per-scene predictions / ground truth and computes mean AP."""

    def __init__(self, ap_iou_thresh: float = 0.25) -> None:
        self.ap_iou_thresh = ap_iou_thresh
        self.pred_map_cls: Dict[int, List[Tuple[int, np.ndarray, float]]] = {}
        self.gt_map_cls: Dict[int, List[Tuple[int, np.ndarray]]] = {}
        self.scan_cnt = 0

    def step(
        self,
        batch_pred: List[List[Tuple[int, np.ndarray, float]]],
        batch_gt: List[List[Tuple[int, np.ndarray]]],
    ) -> None:
        for pred, gt in zip(batch_pred, batch_gt):
            self.pred_map_cls[self.scan_cnt] = pred
            self.gt_map_cls[self.scan_cnt] = gt
            self.scan_cnt += 1

    def compute(self) -> Tuple[float, Dict[int, float]]:
        ap = eval_det(self.pred_map_cls, self.gt_map_cls, ovthresh=self.ap_iou_thresh)
        mean_ap = float(np.mean(list(ap.values()))) if ap else 0.0
        return mean_ap, ap


def _softmax(x: np.ndarray) -> np.ndarray:
    probs = np.exp(x - np.max(x, axis=-1, keepdims=True))
    probs /= np.sum(probs, axis=-1, keepdims=True)
    return probs


def predict_packed(model: Callable[..., Dict[str, Tensor]], point_clouds: Tensor, device: str) -> Dict[str, Tensor]:
    """Run a packed `VoteNet` on a dense $(B, N, 3 + C)$ input (xyz first, then features)."""
    b, n, c = point_clouds.shape
    pos = point_clouds[:, :, 0:3].reshape(b * n, 3).to(device)
    x = point_clouds[:, :, 3:].reshape(b * n, c - 3).to(device) if c > 3 else None
    batch = torch.arange(b, device=device).repeat_interleave(n)
    return model(x, pos, batch)
