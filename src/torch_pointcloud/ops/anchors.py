r"""Axis-aligned anchor generation and target assignment for the anchor-based BEV detectors.

A packed-format port of the anchor generator and target assigner of
:github: [open-mmlab/OpenPCDet](https://github.com/open-mmlab/OpenPCDet), shared by the
[`AnchorHead`][torch_pointcloud.layers.anchors.AnchorHead] (which decodes against the anchors) and the
[`AnchorHeadLoss`][torch_pointcloud.losses.anchor.AnchorHeadLoss] (which assigns targets to them).
"""

from typing import Optional, Sequence, Tuple, TypedDict

import torch
from torch import Tensor

from torch_pointcloud.ops.box3d import boxes_iou3d, boxes_iou_nearest_bev, encode_box_residuals

__all__ = ["AnchorTargets", "assign_anchor_targets", "generate_anchors"]


def generate_anchors(
    point_cloud_range: Sequence[float],
    feature_map_size: Tuple[int, int],
    anchor_sizes: Sequence[Sequence[float]],
    anchor_rotations: Sequence[float],
    anchor_bottom_heights: Sequence[float],
    *,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    r"""Generate axis-aligned anchors for a single class over a BEV feature map.

    A port of OpenPCDet's `AnchorGenerator` with `align_center=False`: anchor centers are placed on a
    grid spanning `point_cloud_range` (endpoints inclusive), then `anchor_bottom_heights` are lifted
    by half the box height to box centers.

    Args:
        point_cloud_range: Range $(x_\min, y_\min, z_\min, x_\max, y_\max, z_\max)$.
        feature_map_size: BEV feature map size $(n_x, n_y)$.
        anchor_sizes: Box sizes $(d_x, d_y, d_z)$, one row per size template.
        anchor_rotations: Yaw angles (radians).
        anchor_bottom_heights: Anchor bottom $z$ per height template.
        dtype: Anchor dtype.

    Returns:
        Anchors $(1, n_y, n_x, n_\text{size}, n_\text{rot}, 7)$ as $(x, y, z, dx, dy, dz, \theta)$.
    """
    nx, ny = feature_map_size
    x_min, y_min, _, x_max, y_max, _ = point_cloud_range
    x_stride = (x_max - x_min) / (nx - 1)
    y_stride = (y_max - y_min) / (ny - 1)

    x_shifts = torch.arange(x_min, x_max + 1e-5, step=x_stride, dtype=dtype)
    y_shifts = torch.arange(y_min, y_max + 1e-5, step=y_stride, dtype=dtype)
    z_shifts = torch.tensor(list(anchor_bottom_heights), dtype=dtype)

    sizes = torch.tensor([list(s) for s in anchor_sizes], dtype=dtype)
    rotations = torch.tensor(list(anchor_rotations), dtype=dtype)
    num_size, num_rot = sizes.shape[0], rotations.shape[0]

    xs, ys, zs = torch.meshgrid([x_shifts, y_shifts, z_shifts], indexing="ij")
    anchors = torch.stack((xs, ys, zs), dim=-1)
    anchors = anchors[:, :, :, None, :].repeat(1, 1, 1, num_size, 1)
    sizes = sizes.view(1, 1, 1, -1, 3).repeat(*anchors.shape[:3], 1, 1)
    anchors = torch.cat((anchors, sizes), dim=-1)
    anchors = anchors[:, :, :, :, None, :].repeat(1, 1, 1, 1, num_rot, 1)
    rotations = rotations.view(1, 1, 1, 1, -1, 1).repeat(*anchors.shape[:3], num_size, 1, 1)
    anchors = torch.cat((anchors, rotations), dim=-1)

    anchors = anchors.permute(2, 1, 0, 3, 4, 5).contiguous()
    # lift anchor bottom heights to box-center heights
    anchors[..., 2] += anchors[..., 5] / 2
    return anchors


class AnchorTargets(TypedDict):
    r"""Per-anchor training targets from [`assign_anchor_targets`][torch_pointcloud.ops.anchors.assign_anchor_targets]."""

    cls_labels: Tensor
    box_reg_targets: Tensor


def _assign_from_overlap(
    overlap: Tensor,
    anchors: Tensor,
    gt_boxes: Tensor,
    gt_labels: Tensor,
    matched_threshold: float,
    unmatched_threshold: float,
) -> Tuple[Tensor, Tensor]:
    r"""Labels $(A,)$ and residual targets $(A, 7 + C)$ of one scene from its $(A, G)$ IoU matrix."""
    num_anchors, num_gt = overlap.shape
    labels = anchors.new_full((num_anchors,), -1, dtype=torch.long)
    box_reg_targets = anchors.new_zeros((num_anchors, gt_boxes.shape[-1]))

    anchor_to_gt_argmax = overlap.argmax(dim=1)
    anchor_to_gt_max = overlap[torch.arange(num_anchors, device=anchors.device), anchor_to_gt_argmax]

    # Every box force-matches its best anchor (unless nothing overlaps it at all).
    gt_to_anchor_argmax = overlap.argmax(dim=0)
    gt_to_anchor_max = overlap[gt_to_anchor_argmax, torch.arange(num_gt, device=anchors.device)]
    gt_to_anchor_max[gt_to_anchor_max == 0] = -1
    anchors_with_max_overlap = (overlap == gt_to_anchor_max).nonzero()[:, 0]
    gt_inds_force = anchor_to_gt_argmax[anchors_with_max_overlap]

    pos_inds = anchor_to_gt_max >= matched_threshold
    labels[pos_inds] = gt_labels[anchor_to_gt_argmax[pos_inds]]
    bg_inds = anchor_to_gt_max < unmatched_threshold
    labels[bg_inds] = 0
    labels[anchors_with_max_overlap] = gt_labels[gt_inds_force]

    # The extra box columns (a velocity) have no anchor counterpart: zero-padded anchors encode them as they are.
    fg_inds = (labels > 0).nonzero()[:, 0]
    fg_anchors = anchors[fg_inds]
    padding = fg_anchors.new_zeros((fg_anchors.shape[0], gt_boxes.shape[-1] - 7))
    fg_anchors = torch.cat([fg_anchors, padding], dim=-1)
    box_reg_targets[fg_inds] = encode_box_residuals(gt_boxes[anchor_to_gt_argmax[fg_inds]], fg_anchors)
    return labels, box_reg_targets


def assign_anchor_targets(
    anchors: Tensor,
    gt_boxes: Tensor,
    gt_labels: Tensor,
    *,
    matched_threshold: float,
    unmatched_threshold: float,
    match_height: bool = False,
    gt_batch: Optional[Tensor] = None,
    batch_size: Optional[int] = None,
) -> AnchorTargets:
    r"""Assign classification and box-regression targets to a single class group of axis-aligned anchors.

    Each anchor is matched to the ground-truth box of highest IoU: IoU $\ge$ `matched_threshold` makes it a
    positive carrying that box's label, IoU $<$ `unmatched_threshold` makes it background, and anything in
    between is ignored. Each ground-truth box additionally force-matches its single highest-IoU anchor, so a
    box with no anchor above threshold still receives one positive. Positive anchors' regression targets are
    the residual encoding of their matched box against the anchor (the inverse of
    `decode_box_residuals`); the columns a box carries beyond its geometry (a velocity) are copied into the
    targets as they are.

    Callers with several class groups (one anchor set per class) invoke this once per group with that class's
    anchors, ground truth, and thresholds, then concatenate the results. With `gt_batch` (the scene index of
    every box) and `batch_size`, one call matches the anchors in every scene of a batch: the IoU runs once
    over the whole batch and the outputs gain a leading scene dimension.

    Args:
        anchors: Anchors $(x, y, z, d_x, d_y, d_z, \theta)$ for one class group, shape $(A, 7)$.
        gt_boxes: Ground-truth boxes $(c_x, c_y, c_z, d_x, d_y, d_z, \theta, \ldots)$, shape $(G, 7 + C)$.
        gt_labels: Ground-truth class labels ($1$-based foreground indices), shape $(G,)$.
        matched_threshold: IoU at or above which an anchor becomes a positive.
        unmatched_threshold: IoU below which an anchor becomes background.
        match_height: Match by the rotated 3D IoU when `True`, otherwise by the bird's-eye IoU of the boxes
            snapped to their nearest axis-aligned orientation
            ([`boxes_iou_nearest_bev`][torch_pointcloud.ops.box3d.boxes_iou_nearest_bev]).
        gt_batch: Scene index of every ground-truth box, shape $(G,)$, to match a whole batch at once.
        batch_size: Number of scenes $B$ when `gt_batch` is given.

    Returns:
        A `TypedDict` with `cls_labels` $(A,)$ ($-1$ ignore, $0$ background, $\ge 1$ foreground class) and
        `box_reg_targets` $(A, 7 + C)$ (residual encodings, zero for non-positive anchors); $(B, A)$ and
        $(B, A, 7 + C)$ with `gt_batch`.

    Shape:
        - anchors: $(A, 7)$
        - gt_boxes: $(G, 7 + C)$
        - gt_labels: $(G,)$
        - cls_labels: $(A,)$
        - box_reg_targets: $(A, 7 + C)$

    Example:
        ```pycon
        >>> anchors = torch.tensor([[0.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0], [20.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0]])
        >>> gt_boxes = torch.tensor([[0.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0]])
        >>> gt_labels = torch.tensor([1])
        >>> out = assign_anchor_targets(anchors, gt_boxes, gt_labels, matched_threshold=0.6, unmatched_threshold=0.45)
        >>> out["cls_labels"].tolist()
        [1, 0]

        ```
    """
    if (gt_batch is None) != (batch_size is None):
        raise ValueError("`gt_batch` and `batch_size` go together.")

    num_anchors, num_gt = anchors.shape[0], gt_boxes.shape[0]
    num_scenes = 1 if batch_size is None else batch_size
    labels = anchors.new_full((num_scenes, num_anchors), -1, dtype=torch.long)
    box_reg_targets = anchors.new_zeros((num_scenes, num_anchors, gt_boxes.shape[-1]))

    if num_gt > 0 and num_anchors > 0:
        geometry = gt_boxes[:, :7]
        overlap = boxes_iou3d(anchors, geometry) if match_height else boxes_iou_nearest_bev(anchors, geometry)
        # The columns of each scene, as contiguous slices of the boxes sorted by scene (one sync for the counts).
        if gt_batch is None:
            order, counts = torch.arange(num_gt, device=anchors.device), [num_gt]
        else:
            order = torch.argsort(gt_batch, stable=True)
            counts = torch.bincount(gt_batch, minlength=num_scenes).tolist()

        offset = 0
        for b, count in enumerate(counts):
            if count == 0:
                labels[b] = 0
                continue

            cols = order[offset : offset + count]
            offset += count
            labels[b], box_reg_targets[b] = _assign_from_overlap(
                overlap[:, cols], anchors, gt_boxes[cols], gt_labels[cols], matched_threshold, unmatched_threshold
            )
    else:
        labels[:] = 0

    if gt_batch is None:
        labels, box_reg_targets = labels[0], box_reg_targets[0]
    return {
        "cls_labels": labels,
        "box_reg_targets": box_reg_targets,
    }
