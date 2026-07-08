r"""Two-stage PointRCNN detection loss: stage-1 per-point head and stage-2 ROI refinement."""

import math
from typing import Any, Dict, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from torch_pointcloud.utils.box3d import encode_box_residuals
from torch_pointcloud.utils.data import DataKeys

_CORNER_TEMPLATE = torch.tensor(
    [
        [1.0, 1.0, -1.0],
        [1.0, -1.0, -1.0],
        [-1.0, -1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [1.0, 1.0, 1.0],
        [1.0, -1.0, 1.0],
        [-1.0, -1.0, 1.0],
        [-1.0, 1.0, 1.0],
    ]
)


def _smooth_l1(diff: Tensor, beta: float) -> Tensor:
    r"""Element-wise smooth-$L_1$: $0.5 x^2 / \beta$ for $|x| < \beta$, else $|x| - 0.5\beta$."""
    n = diff.abs()
    return torch.where(n < beta, 0.5 * n**2 / beta, n - 0.5 * beta)


def _encode_point_residuals(boxes: Tensor, points: Tensor, classes: Tensor, mean_sizes: Tensor) -> Tensor:
    r"""Encode ground-truth boxes into per-point residuals against class mean-size anchors.

    The inverse of [`decode_point_residuals`][torch_pointcloud.models.pointrcnn.decode_point_residuals]:
    the center offset is normalized by the point-anchor base diagonal, sizes become log ratios against the
    class mean size, and the heading becomes a $(\cos, \sin)$ pair.

    Args:
        boxes: Ground-truth boxes $(c_x, c_y, c_z, d_x, d_y, d_z, \theta)$, shape $(N, 7)$.
        points: Anchor point coordinates, shape $(N, 3)$.
        classes: Per-point class index ($1 \ldots \text{num\_classes}$), shape $(N,)$.
        mean_sizes: Per-class mean box size $(d_x, d_y, d_z)$, shape $(\text{num\_classes}, 3)$.

    Returns:
        Residual encodings $(x_t, y_t, z_t, d_{x,t}, d_{y,t}, d_{z,t}, \cos, \sin)$, shape $(N, 8)$.

    Shape:
        - boxes: $(N, 7)$
        - points: $(N, 3)$
        - output: $(N, 8)$
    """
    xg, yg, zg, dxg, dyg, dzg, rg = torch.split(boxes, 1, dim=-1)
    xa, ya, za = torch.split(points, 1, dim=-1)

    anchor = mean_sizes[classes - 1]
    dxa, dya, dza = torch.split(anchor.clamp_min(1e-5), 1, dim=-1)
    dxg, dyg, dzg = dxg.clamp_min(1e-5), dyg.clamp_min(1e-5), dzg.clamp_min(1e-5)
    diagonal = torch.sqrt(dxa**2 + dya**2)

    xt = (xg - xa) / diagonal
    yt = (yg - ya) / diagonal
    zt = (zg - za) / dza
    dxt = torch.log(dxg / dxa)
    dyt = torch.log(dyg / dya)
    dzt = torch.log(dzg / dza)
    return torch.cat([xt, yt, zt, dxt, dyt, dzt, torch.cos(rg), torch.sin(rg)], dim=-1)


class PointRCNNLoss(nn.Module):
    r"""Two-stage PointRCNN detection loss (per-point proposal head + ROI refinement head).

    Reference: :arxiv: [Shi et al., 2019](https://arxiv.org/abs/1812.04244).

    Stage 1 supervises the per-point head that generates proposals: every point inside a ground-truth box
    is foreground (points in the gap between a box and its enlarged copy are ignored), driving a sigmoid
    focal classification loss over the per-point class logits and a code-weighted smooth-$L_1$ over the
    residual box encoding (center offset normalized by the class mean-size diagonal, log extents,
    $(\cos, \sin)$ heading). Stage 2 supervises the refinement head on the sampled ROIs the model forward
    produces: a binary cross-entropy on the confidence logit against an IoU-thresholded label, a
    code-weighted smooth-$L_1$ on the ROI-canonical box residual, and an optional corner regularization
    (the mean smooth-$L_1$ over the eight box corners, robust to the heading flip).

    The stage-1 point targets are assigned inside the loss (points-in-box matching + mean-size residual
    encoding) from the packed ground truth; the stage-2 ROI-to-ground-truth matching (which is random) is
    done by the model forward, which passes the per-ROI max IoU and the canonically transformed matched
    box in its training-mode output. The loss holds no reference to the model.

    Args:
        num_classes: Number of foreground classes.
        mean_sizes: Per-class mean box size $(d_x, d_y, d_z)$, shape $(\text{num\_classes}, 3)$.
        gt_extra_width: Per-axis enlargement of a box when marking ignored points around it.
        point_cls_weight: Weight of the stage-1 classification term.
        point_box_weight: Weight of the stage-1 box-regression term.
        point_code_weights: Per-code stage-1 regression weights, shape $(8,)$.
        reg_fg_thresh: ROI-to-GT IoU at or above which a ROI's box regression is supervised.
        cls_fg_thresh: ROI-to-GT IoU above which a ROI's confidence label is $1$.
        cls_bg_thresh: ROI-to-GT IoU below which a ROI's confidence label is $0$; the band between
            `cls_bg_thresh` and `cls_fg_thresh` is ignored.
        rcnn_cls_weight: Weight of the stage-2 confidence term.
        rcnn_reg_weight: Weight of the stage-2 box-regression term.
        rcnn_corner_weight: Weight of the stage-2 corner regularization ($0$ disables it).
        rcnn_code_weights: Per-code stage-2 regression weights, shape $(7,)$.
        focal_alpha: Stage-1 focal-loss positive/negative balance.
        focal_gamma: Stage-1 focal-loss focusing exponent.
        smooth_l1_beta: Smooth-$L_1$ transition point $\beta$ for both regression terms.
    """

    mean_sizes: Tensor
    point_code_weights: Tensor
    rcnn_code_weights: Tensor
    corner_template: Tensor

    def __init__(
        self,
        num_classes: int,
        *,
        mean_sizes: Union[Tensor, Sequence[Sequence[float]]],
        gt_extra_width: Sequence[float] = (0.2, 0.2, 0.2),
        point_cls_weight: float = 1.0,
        point_box_weight: float = 1.0,
        point_code_weights: Sequence[float] = (1.0,) * 8,
        reg_fg_thresh: float = 0.55,
        cls_fg_thresh: float = 0.6,
        cls_bg_thresh: float = 0.45,
        rcnn_cls_weight: float = 1.0,
        rcnn_reg_weight: float = 1.0,
        rcnn_corner_weight: float = 1.0,
        rcnn_code_weights: Sequence[float] = (1.0,) * 7,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        smooth_l1_beta: float = 1.0 / 9.0,
    ) -> None:
        super().__init__()
        mean = torch.as_tensor(mean_sizes, dtype=torch.float32)
        if mean.shape != (num_classes, 3):
            raise ValueError(f"`mean_sizes` must have shape ({num_classes}, 3), got {tuple(mean.shape)}.")

        self.num_classes = num_classes
        self.gt_extra_width = tuple(gt_extra_width)
        self.point_cls_weight = point_cls_weight
        self.point_box_weight = point_box_weight
        self.reg_fg_thresh = reg_fg_thresh
        self.cls_fg_thresh = cls_fg_thresh
        self.cls_bg_thresh = cls_bg_thresh
        self.rcnn_cls_weight = rcnn_cls_weight
        self.rcnn_reg_weight = rcnn_reg_weight
        self.rcnn_corner_weight = rcnn_corner_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.smooth_l1_beta = smooth_l1_beta

        self.register_buffer("mean_sizes", mean, persistent=False)
        self.register_buffer(
            "point_code_weights", torch.tensor(list(point_code_weights), dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "rcnn_code_weights", torch.tensor(list(rcnn_code_weights), dtype=torch.float32), persistent=False
        )
        self.register_buffer("corner_template", _CORNER_TEMPLATE / 2, persistent=False)

    def forward(self, output: Dict[str, Tensor], batch: Dict[str, Any]) -> Dict[str, Tensor]:
        r"""Compute the two-stage PointRCNN loss and its components.

        Args:
            output: The model's training-mode output: stage-1 `point_cls_preds` $(N, C)$,
                `point_box_preds` $(N, 8)$, `point_coords` $(N, 3)$, `point_batch` $(N,)$; stage-2
                `rcnn_cls` $(M, 1)$, `rcnn_reg` $(M, 7)$, `rcnn_boxes` $(M, 7)$, `rois` $(M, 7)$,
                `gt_of_rois` $(M, 7)$ (ROI-canonical matched box), `gt_of_rois_src` $(M, 7)$ (lidar-frame
                matched box) and `roi_ious` $(M,)$.
            batch: Packed ground truth: `DataKeys.BOX` $(K, 7)$ full-extent, `DataKeys.LABEL` $(K,)$
                ($0$-based classes) and `DataKeys.BATCH_BOX` $(K,)$ per-box scene index.

        Returns:
            A dict with the scalar `loss` (to backprop) and detached `point_cls_loss`, `point_box_loss`,
            `rcnn_cls_loss`, `rcnn_box_loss`.
        """
        point_cls_labels, point_box_labels = self._assign_point_targets(
            output["point_coords"], output["point_batch"], batch
        )
        reg_valid_mask = (output["roi_ious"] > self.reg_fg_thresh).long()
        rcnn_cls_labels = self._rcnn_cls_labels(output["roi_ious"])

        point_cls_loss = self._point_cls_loss(output["point_cls_preds"], point_cls_labels)
        point_box_loss = self._point_box_loss(output["point_box_preds"], point_box_labels, point_cls_labels)
        rcnn_cls_loss = self._rcnn_cls_loss(output["rcnn_cls"], rcnn_cls_labels)
        rcnn_box_loss = self._rcnn_box_loss(
            output["rcnn_reg"],
            output["rcnn_boxes"],
            output["rois"],
            output["gt_of_rois"],
            output["gt_of_rois_src"],
            reg_valid_mask,
        )
        total = point_cls_loss + point_box_loss + rcnn_cls_loss + rcnn_box_loss

        return {
            "loss": total,
            "point_cls_loss": point_cls_loss.detach(),
            "point_box_loss": point_box_loss.detach(),
            "rcnn_cls_loss": rcnn_cls_loss.detach(),
            "rcnn_box_loss": rcnn_box_loss.detach(),
        }

    def _rcnn_cls_labels(self, roi_ious: Tensor) -> Tensor:
        r"""Binary ROI confidence label from the per-ROI max IoU ($-1$ in the ignore band)."""
        labels = (roi_ious > self.cls_fg_thresh).long()
        ignore = (roi_ious > self.cls_bg_thresh) & (roi_ious < self.cls_fg_thresh)
        labels[ignore] = -1
        return labels

    def _assign_point_targets(
        self, point_coords: Tensor, point_batch: Tensor, batch: Dict[str, Any]
    ) -> Tuple[Tensor, Tensor]:
        r"""Assign per-point foreground labels and box-residual targets by points-in-box matching.

        Each point inside a ground-truth box is foreground (labelled with that box's $1$-based class); a
        point in the gap between a box and its `gt_extra_width`-enlarged copy is ignored ($-1$); every other
        point is background ($0$). Foreground points get the mean-size residual encoding of their box.

        Returns:
            `(point_cls_labels, point_box_labels)` of shapes $(N,)$ (long) and $(N, 8)$.
        """
        gt_boxes: Tensor = batch[DataKeys.BOX]
        gt_labels: Tensor = batch[DataKeys.LABEL].long() + 1
        gt_batch: Tensor = batch[DataKeys.BATCH_BOX]
        extra = point_coords.new_tensor(self.gt_extra_width)

        num_points = point_coords.shape[0]
        point_cls_labels = point_coords.new_zeros(num_points, dtype=torch.long)
        point_box_labels = point_coords.new_zeros(num_points, 8)

        batch_size = int(point_batch.max().item()) + 1 if num_points else 0
        for b in range(batch_size):
            point_mask = point_batch == b
            box_mask = gt_batch == b
            scene_points = point_coords[point_mask]
            scene_boxes = gt_boxes[box_mask]
            scene_labels = gt_labels[box_mask]
            if scene_boxes.shape[0] == 0 or scene_points.shape[0] == 0:
                continue

            in_box = _points_in_boxes(scene_points, scene_boxes)
            enlarged = scene_boxes.clone()
            enlarged[:, 3:6] = enlarged[:, 3:6] + extra
            in_ext = _points_in_boxes(scene_points, enlarged)

            fg = in_box.any(dim=1)
            box_idx = in_box.float().argmax(dim=1)
            ignore = in_ext.any(dim=1) & ~fg

            cls_single = point_cls_labels.new_zeros(scene_points.shape[0])
            cls_single[fg] = scene_labels[box_idx[fg]]
            cls_single[ignore] = -1
            point_cls_labels[point_mask] = cls_single

            if fg.any():
                fg_boxes = scene_boxes[box_idx[fg]]
                fg_classes = scene_labels[box_idx[fg]]
                box_single = point_box_labels.new_zeros(scene_points.shape[0], 8)
                box_single[fg] = _encode_point_residuals(fg_boxes, scene_points[fg], fg_classes, self.mean_sizes)
                point_box_labels[point_mask] = box_single

        return point_cls_labels, point_box_labels

    def _point_cls_loss(self, preds: Tensor, labels: Tensor) -> Tensor:
        r"""Stage-1 sigmoid focal classification loss over one-hot foreground labels."""
        labels = labels.view(-1)
        preds = preds.view(-1, self.num_classes)
        positives = labels > 0
        cared = labels >= 0
        cls_weights = cared.to(preds.dtype) / positives.sum().clamp(min=1).to(preds.dtype)

        one_hot = preds.new_zeros(labels.shape[0], self.num_classes + 1)
        one_hot.scatter_(-1, (labels * cared.long()).unsqueeze(-1), 1.0)
        one_hot = one_hot[..., 1:]

        pred_sigmoid = preds.sigmoid()
        alpha_weight = one_hot * self.focal_alpha + (1.0 - one_hot) * (1.0 - self.focal_alpha)
        pt = one_hot * (1.0 - pred_sigmoid) + (1.0 - one_hot) * pred_sigmoid
        focal_weight = alpha_weight * pt.pow(self.focal_gamma)
        bce = preds.clamp(min=0) - preds * one_hot + torch.log1p(torch.exp(-preds.abs()))
        loss = focal_weight * bce * cls_weights.unsqueeze(-1)
        return loss.sum() * self.point_cls_weight

    def _point_box_loss(self, preds: Tensor, targets: Tensor, labels: Tensor) -> Tensor:
        r"""Stage-1 code-weighted smooth-$L_1$ box-regression loss over foreground points."""
        pos_mask = labels.view(-1) > 0
        reg_weights = pos_mask.to(preds.dtype) / pos_mask.sum().clamp(min=1).to(preds.dtype)
        diff = (preds - targets) * self.point_code_weights.view(1, -1)
        loss = _smooth_l1(diff, self.smooth_l1_beta) * reg_weights.unsqueeze(-1)
        return loss.sum() * self.point_box_weight

    def _rcnn_cls_loss(self, rcnn_cls: Tensor, labels: Tensor) -> Tensor:
        r"""Stage-2 binary cross-entropy on the confidence logit, masked to non-ignored ROIs."""
        flat = rcnn_cls.view(-1)
        valid = (labels >= 0).to(flat.dtype)
        bce = F.binary_cross_entropy(torch.sigmoid(flat), labels.clamp(min=0).to(flat.dtype), reduction="none")
        loss = (bce * valid).sum() / valid.sum().clamp(min=1.0)
        return loss * self.rcnn_cls_weight

    def _rcnn_box_loss(
        self,
        rcnn_reg: Tensor,
        rcnn_boxes: Tensor,
        rois: Tensor,
        gt_of_rois: Tensor,
        gt_of_rois_src: Tensor,
        reg_valid_mask: Tensor,
    ) -> Tensor:
        r"""Stage-2 code-weighted smooth-$L_1$ box residual plus optional corner regularization."""
        fg_mask = reg_valid_mask.view(-1) > 0
        fg_sum = fg_mask.sum()

        rois_anchor = rois.clone()
        rois_anchor[:, 0:3] = 0
        rois_anchor[:, 6] = 0
        reg_targets = encode_box_residuals(gt_of_rois[..., 0:7], rois_anchor)
        diff = (rcnn_reg - reg_targets) * self.rcnn_code_weights.view(1, -1)
        smooth = _smooth_l1(diff, self.smooth_l1_beta)
        loss = (smooth * fg_mask.unsqueeze(-1).to(smooth.dtype)).sum() / fg_sum.clamp(min=1).to(smooth.dtype)
        loss = loss * self.rcnn_reg_weight

        if self.rcnn_corner_weight > 0 and fg_sum > 0:
            corner = self._corner_loss(rcnn_boxes[fg_mask], gt_of_rois_src[fg_mask][..., 0:7])
            loss = loss + corner.mean() * self.rcnn_corner_weight
        return loss

    def _corner_loss(self, pred_boxes: Tensor, gt_boxes: Tensor) -> Tensor:
        r"""Per-box mean smooth-$L_1$ over the eight box corners, robust to the $\pi$ heading flip."""
        pred_corners = self._boxes_to_corners(pred_boxes)
        gt_corners = self._boxes_to_corners(gt_boxes)
        gt_flip = gt_boxes.clone()
        gt_flip[:, 6] = gt_flip[:, 6] + math.pi
        gt_corners_flip = self._boxes_to_corners(gt_flip)

        dist = torch.min(
            torch.norm(pred_corners - gt_corners, dim=2),
            torch.norm(pred_corners - gt_corners_flip, dim=2),
        )
        return _smooth_l1(dist, beta=1.0).mean(dim=1)

    def _boxes_to_corners(self, boxes: Tensor) -> Tensor:
        r"""Convert boxes $(N, 7)$ to their 8 corners $(N, 8, 3)$ (heading rotates $x \to y$)."""
        corners = boxes[:, None, 3:6] * self.corner_template[None, :, :]
        cos, sin = torch.cos(boxes[:, 6]), torch.sin(boxes[:, 6])
        x = corners[..., 0] * cos[:, None] - corners[..., 1] * sin[:, None]
        y = corners[..., 0] * sin[:, None] + corners[..., 1] * cos[:, None]
        rotated = torch.stack([x, y, corners[..., 2]], dim=-1)
        return rotated + boxes[:, None, 0:3]


def _points_in_boxes(points: Tensor, boxes: Tensor) -> Tensor:
    r"""Boolean containment test of every point against every oriented box, $(N, 3), (G, 7) \to (N, G)$."""
    offset = points[:, None, :] - boxes[None, :, 0:3]
    half = boxes[:, 3:6] / 2.0
    cos, sin = torch.cos(boxes[:, 6]), torch.sin(boxes[:, 6])
    local_x = offset[..., 0] * cos[None, :] + offset[..., 1] * sin[None, :]
    local_y = -offset[..., 0] * sin[None, :] + offset[..., 1] * cos[None, :]
    inside_x = local_x.abs() <= half[None, :, 0]
    inside_y = local_y.abs() <= half[None, :, 1]
    inside_z = offset[..., 2].abs() <= half[None, :, 2]
    return inside_x & inside_y & inside_z
