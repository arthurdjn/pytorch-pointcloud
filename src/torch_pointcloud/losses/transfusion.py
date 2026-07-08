r"""TransFusion detection loss: Hungarian-matched query targets and a dense center-heatmap objective."""

from typing import Any, Dict, List, Sequence, Tuple

import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn

from torch_pointcloud.losses.anchor import sigmoid_focal_loss
from torch_pointcloud.utils.box3d import boxes_iou3d
from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.heatmap import draw_heatmap_targets

_EPS = 1e-4
_LOG_EPS = 1e-12


def _clip_sigmoid(x: Tensor) -> Tensor:
    r"""Sigmoid clamped to $[\varepsilon, 1 - \varepsilon]$ so the focal $\log$ terms stay finite."""
    return torch.clamp(x.sigmoid(), min=_EPS, max=1 - _EPS)


def _gaussian_focal_loss(pred: Tensor, target: Tensor) -> Tensor:
    r"""Penalty-reduced center focal loss over the dense heatmap, normalized by the peak-cell count.

    The positive term is $-\log(p)(1 - p)^2$ at cells whose Gaussian target is exactly $1$; every other
    cell is a soft negative down-weighted by $(1 - y)^4$. Normalized by the number of positive cells.

    Args:
        pred: Clamped sigmoid probabilities, shape $(B, C, H, W)$.
        target: Gaussian heatmap target of the same shape, values in $[0, 1]$.

    Returns:
        Scalar focal loss.
    """
    pos_weights = target.eq(1).float()
    neg_weights = (1 - target).pow(4)
    pos_loss = -(pred + _LOG_EPS).log() * (1 - pred).pow(2) * pos_weights
    neg_loss = -(1 - pred + _LOG_EPS).log() * pred.pow(2) * neg_weights
    num_pos = pos_weights.sum().clamp(min=1.0)
    return (pos_loss + neg_loss).sum() / num_pos


class TransFusionLoss(nn.Module):
    r"""Query-based TransFusion detection loss (dense heatmap, matched classification, box, IoU rescore).

    Reference: :arxiv: [Bai et al., 2022](https://arxiv.org/abs/2203.11496).

    The head predicts a dense per-class BEV heatmap plus a fixed set of object queries, each carrying a
    class logit vector and a box code. Four terms are summed:

    - **Heatmap:** the ground-truth box centers are splatted onto a per-class BEV Gaussian map and the
      dense heatmap is supervised by the penalty-reduced center focal loss.
    - **Classification:** every scene's queries are decoded to boxes and matched to the ground truth by a
      per-scene Hungarian assignment (cost: focal classification + normalized center $L_1$ + 3D IoU). The
      per-query class logits are then trained by sigmoid focal loss over one-hot targets (background for
      unmatched queries), normalized by the positive count.
    - **Box regression:** code-weighted $L_1$ over the $10$-dim box code
      $(x, y, z, \log d_x, \log d_y, \log d_z, \sin\theta, \cos\theta, v_x, v_y)$ at the matched queries.
    - **IoU rescore:** an $L_1$ term regressing the per-query `iou` branch toward $2 \cdot \text{IoU}_{3D} - 1$
      between each matched query's decoded box and its ground-truth box.

    The loss holds no reference to the model: the grid geometry is rebuilt from the constructor params.

    !!! note
        nuScenes ground-truth boxes carry no velocity ($(K, 7)$), so the velocity targets are zero. The
        default `code_weights` zeroes the last two (velocity) codes to leave that branch unsupervised.

    Args:
        num_classes: Number of foreground classes (heatmap channels and query logits).
        point_cloud_range: Range $(x_\min, y_\min, z_\min, x_\max, y_\max, z_\max)$.
        voxel_size: Voxel size $(v_x, v_y, v_z)$.
        feature_map_stride: Stride from the voxel grid to the BEV feature map.
        num_proposals: Number of object queries per scene.
        gaussian_overlap: Min-overlap passed to the Gaussian-radius solver.
        min_radius: Lower clamp on the integer splat radius.
        hungarian_cls_cost: Weight of the focal classification term in the matching cost.
        hungarian_reg_cost: Weight of the normalized center-$L_1$ term in the matching cost.
        hungarian_iou_cost: Weight of the 3D-IoU term in the matching cost.
        code_weights: Per-code regression weight, length $10$; the last two (velocity) default to $0$.
        cls_weight: Multiplier on the classification term.
        bbox_weight: Multiplier on the box-regression term.
        hm_weight: Multiplier on the heatmap term.
        iou_weight: Multiplier on the IoU-rescore term.
        focal_alpha: Focal positive/negative balance (classification loss and matching cost).
        focal_gamma: Focal focusing exponent (classification loss and matching cost).
    """

    code_weights: Tensor

    def __init__(
        self,
        num_classes: int,
        point_cloud_range: Sequence[float],
        voxel_size: Sequence[float],
        feature_map_stride: int,
        *,
        num_proposals: int = 200,
        gaussian_overlap: float = 0.1,
        min_radius: int = 2,
        hungarian_cls_cost: float = 0.15,
        hungarian_reg_cost: float = 0.25,
        hungarian_iou_cost: float = 0.25,
        code_weights: Sequence[float] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0),
        cls_weight: float = 1.0,
        bbox_weight: float = 0.25,
        hm_weight: float = 1.0,
        iou_weight: float = 0.5,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ) -> None:
        super().__init__()
        if len(code_weights) != 10:
            raise ValueError(f"`code_weights` must have length 10, got {len(code_weights)}.")

        self.num_classes = num_classes
        self.point_cloud_range = tuple(float(p) for p in point_cloud_range)
        self.voxel_size = tuple(float(v) for v in voxel_size)
        self.feature_map_stride = feature_map_stride
        self.num_proposals = num_proposals
        self.gaussian_overlap = gaussian_overlap
        self.min_radius = min_radius
        self.hungarian_cls_cost = hungarian_cls_cost
        self.hungarian_reg_cost = hungarian_reg_cost
        self.hungarian_iou_cost = hungarian_iou_cost
        self.cls_weight = cls_weight
        self.bbox_weight = bbox_weight
        self.hm_weight = hm_weight
        self.iou_weight = iou_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

        pcr, vs = self.point_cloud_range, self.voxel_size
        grid = [int(round((pcr[i + 3] - pcr[i]) / vs[i])) for i in range(3)]
        self.feature_map_size = (grid[0] // feature_map_stride, grid[1] // feature_map_stride)

        self.register_buffer("code_weights", torch.tensor(list(code_weights), dtype=torch.float32))

    def forward(self, output: Dict[str, Tensor], batch: Dict[str, Any]) -> Dict[str, Tensor]:
        r"""Compute the TransFusion loss and its components.

        Args:
            output: The head's raw output: per-query `center` $(B, 2, Q)$, `height` $(B, 1, Q)$,
                `dim` $(B, 3, Q)$, `rot` $(B, 2, Q)$, `vel` $(B, 2, Q)$, `iou` $(B, 1, Q)$ and `heatmap`
                $(B, C, Q)$ class logits, plus the dense `dense_heatmap` $(B, C, H, W)$.
            batch: Packed ground truth (`DataKeys.BOX` $(K, 7)$ full-extent, `DataKeys.LABEL` $(K,)$
                $0$-based, `DataKeys.BATCH_BOX` $(K,)$ per-box scene index).

        Returns:
            A dict with the scalar `loss` (to backprop) and detached `heatmap_loss`, `cls_loss`,
            `bbox_loss`, `iou_loss` diagnostics.
        """
        center = output["center"]
        batch_size, _, num_queries = center.shape
        device = center.device
        width, height = self.feature_map_size

        boxes_per_scene, labels_per_scene = self._densify_gt(batch, batch_size, device)

        hm_target = torch.stack(
            [
                draw_heatmap_targets(
                    boxes_per_scene[b],
                    labels_per_scene[b],
                    self.num_classes,
                    self.feature_map_size,
                    self.voxel_size,
                    self.point_cloud_range,
                    self.feature_map_stride,
                    gaussian_overlap=self.gaussian_overlap,
                    min_radius=self.min_radius,
                )[0]
                for b in range(batch_size)
            ]
        )
        heatmap_loss = _gaussian_focal_loss(_clip_sigmoid(output["dense_heatmap"]), hm_target) * self.hm_weight

        labels = center.new_full((batch_size, num_queries), self.num_classes, dtype=torch.long)
        bbox_targets = center.new_zeros((batch_size, num_queries, 10))
        bbox_weights = center.new_zeros((batch_size, num_queries, 10))
        iou_terms: List[Tensor] = []
        num_pos = 0
        for b in range(batch_size):
            decoded = self._decode_queries(output, b)
            pos_inds, pos_gt = self._match(decoded, output["heatmap"][b].t(), boxes_per_scene[b], labels_per_scene[b])
            num_pos += int(pos_inds.numel())
            if pos_inds.numel() == 0:
                continue
            labels[b, pos_inds] = labels_per_scene[b][pos_gt]
            bbox_targets[b, pos_inds] = self._encode(boxes_per_scene[b][pos_gt])
            bbox_weights[b, pos_inds] = 1.0
            iou = boxes_iou3d(decoded[pos_inds], boxes_per_scene[b][pos_gt, :7]).diagonal()
            iou_terms.append((output["iou"][b, 0, pos_inds] - (iou * 2 - 1)).abs().sum())

        cls_loss = self._cls_loss(output["heatmap"], labels, num_pos) * self.cls_weight
        bbox_loss = self._bbox_loss(output, bbox_targets, bbox_weights, num_pos) * self.bbox_weight
        iou_sum = torch.stack(iou_terms).sum() if iou_terms else center.new_zeros(())
        iou_loss = iou_sum / max(num_pos, 1) * self.iou_weight

        total = heatmap_loss + cls_loss + bbox_loss + iou_loss
        return {
            "loss": total,
            "heatmap_loss": heatmap_loss.detach(),
            "cls_loss": cls_loss.detach(),
            "bbox_loss": bbox_loss.detach(),
            "iou_loss": iou_loss.detach(),
        }

    def _densify_gt(
        self, batch: Dict[str, Any], batch_size: int, device: torch.device
    ) -> Tuple[List[Tensor], List[Tensor]]:
        r"""Split the packed ground truth into per-scene box / zero-based-label lists, dropping empty boxes."""
        box: Tensor = batch[DataKeys.BOX]
        label: Tensor = batch[DataKeys.LABEL].long()  # 0-based class index (nuScenes `class_to_idx`)
        box_batch: Tensor = batch[DataKeys.BATCH_BOX]

        boxes_per_scene: List[Tensor] = []
        labels_per_scene: List[Tensor] = []
        for b in range(batch_size):
            scene = box_batch == b
            scene_boxes = box[scene].to(device)
            scene_labels = label[scene].to(device)
            keep = (scene_boxes[:, 3] > 0) & (scene_boxes[:, 4] > 0)
            boxes_per_scene.append(scene_boxes[keep])
            labels_per_scene.append(scene_labels[keep])
        return boxes_per_scene, labels_per_scene

    def _decode_queries(self, output: Dict[str, Tensor], b: int) -> Tensor:
        r"""Decode one scene's per-query predictions to oriented boxes $(Q, 7)$ in metric coordinates."""
        vs, pcr, stride = self.voxel_size, self.point_cloud_range, self.feature_map_stride
        center = output["center"][b]
        dim = output["dim"][b].exp()
        rot = output["rot"][b]
        x = center[0] * stride * vs[0] + pcr[0]
        y = center[1] * stride * vs[1] + pcr[1]
        z = output["height"][b][0]
        angle = torch.atan2(rot[0], rot[1])
        return torch.stack([x, y, z, dim[0], dim[1], dim[2], angle], dim=-1)

    def _match(self, decoded: Tensor, cls_logits: Tensor, gt_boxes: Tensor, gt_labels: Tensor) -> Tuple[Tensor, Tensor]:
        r"""Per-scene Hungarian match of queries to ground truth (focal cls + center $L_1$ + 3D-IoU cost)."""
        if gt_boxes.shape[0] == 0 or decoded.shape[0] == 0:
            empty = decoded.new_zeros((0,), dtype=torch.long)
            return empty, empty

        prob = cls_logits.sigmoid()
        neg = -(1 - prob + _LOG_EPS).log() * (1 - self.focal_alpha) * prob.pow(self.focal_gamma)
        pos = -(prob + _LOG_EPS).log() * self.focal_alpha * (1 - prob).pow(self.focal_gamma)
        cls_cost = (pos[:, gt_labels] - neg[:, gt_labels]) * self.hungarian_cls_cost

        pc_start = decoded.new_tensor(self.point_cloud_range[0:2])
        pc_range = decoded.new_tensor(self.point_cloud_range[3:5]) - pc_start
        reg_cost = torch.cdist((decoded[:, :2] - pc_start) / pc_range, (gt_boxes[:, :2] - pc_start) / pc_range, p=1)
        reg_cost = reg_cost * self.hungarian_reg_cost

        iou_cost = -boxes_iou3d(decoded, gt_boxes[:, :7]) * self.hungarian_iou_cost

        cost = cls_cost + reg_cost + iou_cost
        row, col = linear_sum_assignment(cost.detach().cpu().numpy())
        return (
            torch.as_tensor(row, dtype=torch.long, device=decoded.device),
            torch.as_tensor(col, dtype=torch.long, device=decoded.device),
        )

    def _encode(self, boxes: Tensor) -> Tensor:
        r"""Encode matched ground-truth boxes into the $10$-dim query box code (grid center, log size, sincos)."""
        vs, pcr, stride = self.voxel_size, self.point_cloud_range, self.feature_map_stride
        targets = boxes.new_zeros((boxes.shape[0], 10))
        targets[:, 0] = (boxes[:, 0] - pcr[0]) / (stride * vs[0])
        targets[:, 1] = (boxes[:, 1] - pcr[1]) / (stride * vs[1])
        targets[:, 2] = boxes[:, 2]
        targets[:, 3:6] = boxes[:, 3:6].log()
        targets[:, 6] = torch.sin(boxes[:, 6])
        targets[:, 7] = torch.cos(boxes[:, 6])
        if boxes.shape[1] >= 9:
            targets[:, 8:10] = boxes[:, 7:9]
        return targets

    def _cls_loss(self, cls_logits: Tensor, labels: Tensor, num_pos: int) -> Tensor:
        r"""Sigmoid focal classification over one-hot query targets (background maps to an all-zero row)."""
        batch_size, num_queries = labels.shape
        cls_score = cls_logits.permute(0, 2, 1)
        one_hot = cls_score.new_zeros((batch_size, num_queries, self.num_classes + 1))
        one_hot.scatter_(-1, labels.unsqueeze(-1), 1.0)
        one_hot = one_hot[..., : self.num_classes]
        weights = cls_score.new_ones((batch_size, num_queries))
        loss = sigmoid_focal_loss(cls_score, one_hot, weights, alpha=self.focal_alpha, gamma=self.focal_gamma)
        return loss.sum() / max(num_pos, 1)

    def _bbox_loss(self, output: Dict[str, Tensor], bbox_targets: Tensor, bbox_weights: Tensor, num_pos: int) -> Tensor:
        r"""Code-weighted $L_1$ box regression over the matched queries."""
        preds = torch.cat([output[k] for k in ("center", "height", "dim", "rot", "vel")], dim=1).permute(0, 2, 1)
        reg_weights = bbox_weights * self.code_weights
        loss = (preds - bbox_targets).abs() * reg_weights
        return loss.sum() / max(num_pos, 1)
