r"""RoI operations of two-stage 3D detectors: residual coders, grid points, pooling and training-time sampling.

RoIs and boxes are $(c_x, c_y, c_z, d_x, d_y, d_z, \theta)$ with full extents and the heading counter-clockwise about
$+z$ from $+x$, as in [`box3d`][torch_pointcloud.ops.box3d].
"""

import math
from typing import Literal, TypedDict

import torch
from torch import Tensor
from torch_geometric.utils import scatter

from torch_pointcloud.ops.box3d import boxes_iou3d, decode_box_residuals, encode_box_residuals
from torch_pointcloud.utils.types import OptTensor


def decode_roi_residuals(encodings: Tensor, rois: Tensor) -> Tensor:
    r"""Decode box residuals predicted in the frame of their RoI.

    The residuals are decoded by [`decode_box_residuals`][torch_pointcloud.ops.box3d.decode_box_residuals] against
    each RoI moved to the origin (heading and extents kept); the decoded center is then rotated by the RoI heading
    and moved back to the RoI center.

    Args:
        encodings: Residuals $(x_t, y_t, z_t, d_{x,t}, d_{y,t}, d_{z,t}, \theta_t)$, shape $(R, 7)$.
        rois: RoIs $(c_x, c_y, c_z, d_x, d_y, d_z, \theta)$, shape $(R, 7)$.

    Returns:
        The decoded boxes, shape $(R, 7)$.

    Example:
        ```pycon
        >>> rois = torch.tensor([[10.0, 5.0, 0.0, 4.0, 2.0, 1.5, math.pi / 2]])
        >>> encodings = torch.tensor([[0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        >>> decode_roi_residuals(encodings, rois)[0, :2]
        tensor([10.0000,  5.4472])

        ```
    """
    local = rois.clone()
    local[:, 0:3] = 0
    boxes = decode_box_residuals(encodings, local)
    cos, sin = torch.cos(rois[:, 6]), torch.sin(rois[:, 6])
    x = boxes[:, 0] * cos - boxes[:, 1] * sin
    y = boxes[:, 0] * sin + boxes[:, 1] * cos
    center = torch.stack([x, y, boxes[:, 2]], dim=1) + rois[:, 0:3]
    return torch.cat([center, boxes[:, 3:]], dim=1)


def encode_roi_residuals(boxes: Tensor, rois: Tensor) -> Tensor:
    r"""Encode boxes as residuals in the frame of their RoI (inverse of `decode_roi_residuals`).

    Every box is moved into the frame of its RoI (center at the origin, heading along $+x$), its heading is folded
    into $[-\pi/2, \pi/2]$ by a half turn when it faces away from the RoI, and the result is encoded by
    [`encode_box_residuals`][torch_pointcloud.ops.box3d.encode_box_residuals] against the RoI extents. Decoding the
    residuals returns the box, up to that half turn of the heading.

    Args:
        boxes: Boxes $(c_x, c_y, c_z, d_x, d_y, d_z, \theta)$, shape $(R, 7)$.
        rois: RoIs in the same layout, shape $(R, 7)$.

    Returns:
        The residuals $(x_t, y_t, z_t, d_{x,t}, d_{y,t}, d_{z,t}, \theta_t)$, shape $(R, 7)$.

    Example:
        ```pycon
        >>> rois = torch.tensor([[10.0, 5.0, 0.0, 4.0, 2.0, 1.5, math.pi / 2]])
        >>> boxes = torch.tensor([[10.0, 5.4472, 0.0, 4.0, 2.0, 1.5, math.pi / 2]])
        >>> encode_roi_residuals(boxes, rois)[0, 0]
        tensor(0.1000)

        ```
    """
    heading = rois[:, 6]
    offset = boxes[:, 0:3] - rois[:, 0:3]
    cos, sin = torch.cos(heading), torch.sin(heading)
    x = offset[:, 0] * cos + offset[:, 1] * sin
    y = -offset[:, 0] * sin + offset[:, 1] * cos

    local_heading = (boxes[:, 6] - heading) % (2 * math.pi)
    opposite = (local_heading > math.pi / 2) & (local_heading < math.pi * 3 / 2)
    local_heading = torch.where(opposite, (local_heading + math.pi) % (2 * math.pi), local_heading)
    local_heading = torch.where(local_heading > math.pi, local_heading - 2 * math.pi, local_heading)
    local_heading = local_heading.clamp(-math.pi / 2, math.pi / 2)

    local = torch.cat([torch.stack([x, y, offset[:, 2]], dim=1), boxes[:, 3:6], local_heading[:, None]], dim=1)
    anchors = torch.cat([torch.zeros_like(rois[:, 0:3]), rois[:, 3:6], torch.zeros_like(rois[:, 6:7])], dim=1)
    return encode_box_residuals(local, anchors)


def roi_grid_points(rois: Tensor, grid_size: int) -> Tensor:
    r"""Place a regular $G \times G \times G$ grid of points inside every oriented box.

    The points sit at the cell centers of the box, enumerated with $x$ slowest and $z$ fastest.

    Args:
        rois: Boxes $(c_x, c_y, c_z, d_x, d_y, d_z, \theta)$, shape $(R, 7)$.
        grid_size: Points per box side $G$.

    Returns:
        The grid points, shape $(R, G^3, 3)$.

    Example:
        ```pycon
        >>> rois = torch.tensor([[1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 0.0]])
        >>> roi_grid_points(rois, 2)[0, :2]
        tensor([[0.5000, 1.0000, 1.5000],
                [0.5000, 1.0000, 4.5000]])

        ```
    """
    steps = (torch.arange(grid_size, device=rois.device, dtype=rois.dtype) + 0.5) / grid_size - 0.5
    local = torch.stack(torch.meshgrid(steps, steps, steps, indexing="ij"), dim=-1).view(1, -1, 3)
    local = local * rois[:, None, 3:6]

    cos, sin = torch.cos(rois[:, 6:7]), torch.sin(rois[:, 6:7])
    x = local[..., 0] * cos - local[..., 1] * sin
    y = local[..., 0] * sin + local[..., 1] * cos
    points = torch.stack([x, y, local[..., 2]], dim=-1)
    return points + rois[:, None, 0:3]


def roi_aware_pool3d(
    x: Tensor,
    pos: Tensor,
    boxes: Tensor,
    out_size: int,
    *,
    pos_batch: OptTensor = None,
    box_batch: OptTensor = None,
    max_points_per_cell: int = 127,
    reduce: Literal["max", "mean"] = "max",
) -> Tensor:
    r"""Pool point features into a regular grid of cells inside every oriented box.

    Every box is split into $S \times S \times S$ cells along its own axes. A point lies in a box when its height is
    within half the box height of the center and its offset, rotated into the box frame by $-\theta$, is strictly
    within half the box length and width; it falls into the cell its offset from the box's lower corner lands in,
    clamped to the grid. Every cell keeps its first `max_points_per_cell` points in index order and reduces their
    features with `reduce`; an empty cell is zero. Boxes are $(c_x, c_y, c_z, d_x, d_y, d_z, \theta)$ with full
    extents.

    Args:
        x: Point features, shape $(N, C)$.
        pos: Point coordinates, shape $(N, 3)$.
        boxes: Boxes, shape $(K, 7)$.
        out_size: Cells per box side $S$.
        pos_batch: Optional per-point scene index, shape $(N,)$.
        box_batch: Optional per-box scene index, shape $(K,)$.
        max_points_per_cell: Points kept per cell.
        reduce: Reduction over the points of a cell, `"max"` or `"mean"`.

    Returns:
        The pooled features, shape $(K, S, S, S, C)$, cells indexed $(x, y, z)$ along the box axes.

    Shape:
        - x: $(N, C)$
        - pos: $(N, 3)$
        - boxes: $(K, 7)$
        - output: $(K, S, S, S, C)$

    Example:
        ```pycon
        >>> x = torch.tensor([[1.0], [3.0], [5.0]])
        >>> pos = torch.tensor([[-0.5, 0.0, 0.0], [-0.4, 0.0, 0.0], [0.5, 0.0, 0.0]])
        >>> boxes = torch.tensor([[0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0]])
        >>> roi_aware_pool3d(x, pos, boxes, 2, reduce="mean")[0, :, 1, 1, 0].tolist()
        [2.0, 5.0]

        ```
    """
    if (pos_batch is None) != (box_batch is None):
        raise ValueError("`pos_batch` and `box_batch` must be given together; got exactly one of them.")

    if pos_batch is None or box_batch is None:
        pos_batch = pos.new_zeros(pos.shape[0], dtype=torch.long)
        box_batch = boxes.new_zeros(boxes.shape[0], dtype=torch.long)

    num_cells = out_size**3
    out = x.new_zeros(boxes.shape[0], num_cells, x.shape[1])
    for scene in box_batch.unique().tolist():
        scene_points = (pos_batch == scene).nonzero(as_tuple=False).squeeze(-1)
        scene_boxes = (box_batch == scene).nonzero(as_tuple=False).squeeze(-1)
        # Boxes go through in chunks so the (N, K) point-in-box test stays within 2^24 pairs.
        chunk_size = max(1, (1 << 24) // max(scene_points.shape[0], 1))
        for index in scene_boxes.split(chunk_size):
            box = boxes[index]
            offset = pos[scene_points, None, :] - box[None, :, 0:3]
            cos, sin = torch.cos(box[:, 6]), torch.sin(box[:, 6])
            local_x = offset[..., 0] * cos + offset[..., 1] * sin
            local_y = offset[..., 1] * cos - offset[..., 0] * sin
            size = box[:, 3:6]
            inside = (
                (offset[..., 2].abs() <= size[:, 2] / 2)
                & (local_x > -size[:, 0] / 2)
                & (local_x < size[:, 0] / 2)
                & (local_y > -size[:, 1] / 2)
                & (local_y < size[:, 1] / 2)
            )
            point, slot = inside.nonzero(as_tuple=True)

            cell = size[slot] / out_size
            local = torch.stack([local_x[point, slot], local_y[point, slot], offset[point, slot, 2]], dim=1)
            ijk = ((local + size[slot] / 2) / cell).long().clamp(0, out_size - 1)
            key = slot * num_cells + (ijk[:, 0] * out_size + ijk[:, 1]) * out_size + ijk[:, 2]

            # Pairs come point-major, so a stable sort by cell keeps every cell's points in index order.
            key, order = key.sort(stable=True)
            point = point[order]
            first = torch.searchsorted(key, key, side="left")
            keep = torch.arange(key.numel(), device=key.device) - first < max_points_per_cell
            key, point = key[keep], point[keep]

            pooled = scatter(x[scene_points[point]], key, dim=0, dim_size=index.numel() * num_cells, reduce=reduce)
            out[index] = pooled.view(index.numel(), num_cells, x.shape[1])

    return out.view(boxes.shape[0], out_size, out_size, out_size, x.shape[1])


class RoISamples(TypedDict):
    r"""RoIs sampled for the refinement stage of a two-stage detector, matched to the ground truth.

    Attributes:
        index: Index of every sampled RoI into the input RoIs, scene by scene, shape $(B S,)$.
        target_index: Index of the matched ground-truth box, $-1$ when the RoI overlaps no box of its class,
            shape $(B S,)$.
        iou: 3D IoU of every sampled RoI with its matched box, shape $(B S,)$.
    """

    index: Tensor
    target_index: Tensor
    iou: Tensor


def _sample_background(hard: Tensor, easy: Tensor, num_samples: int, hard_ratio: float) -> Tensor:
    """Draw `num_samples` background RoIs with replacement, `hard_ratio` of them among the hard ones when both exist."""
    if hard.numel() > 0 and easy.numel() > 0:
        num_hard = min(int(num_samples * hard_ratio), hard.numel())
        hard = hard[torch.randint(hard.numel(), (num_hard,), device=hard.device)]
        easy = easy[torch.randint(easy.numel(), (num_samples - num_hard,), device=easy.device)]
        return torch.cat([hard, easy])

    pool = hard if hard.numel() > 0 else easy
    return pool[torch.randint(pool.numel(), (num_samples,), device=pool.device)]


@torch.no_grad()
def sample_rois(
    rois: Tensor,
    roi_labels: Tensor,
    roi_batch: Tensor,
    boxes: Tensor,
    labels: Tensor,
    batch: Tensor,
    *,
    num_samples: int,
    foreground_ratio: float,
    foreground_threshold: float,
    hard_background_threshold: float,
    hard_background_ratio: float,
) -> RoISamples:
    r"""Sample a fixed number of RoIs per scene and match each one to a ground-truth box of its class.

    Every RoI is matched to the box of its class with the highest 3D IoU. A RoI is foreground when that IoU is at
    least `foreground_threshold`, hard background when it is at least `hard_background_threshold`, and easy
    background otherwise. Every scene draws `num_samples` RoIs: up to `foreground_ratio` of them among the
    foreground without replacement, the rest among the background with replacement, `hard_background_ratio` of them
    hard. A scene without background fills all its samples from the foreground with replacement.

    Args:
        rois: RoIs $(c_x, c_y, c_z, d_x, d_y, d_z, \theta)$, shape $(R, 7)$.
        roi_labels: Class of every RoI, shape $(R,)$.
        roi_batch: Scene index of every RoI, shape $(R,)$.
        boxes: Ground-truth boxes in the same layout, shape $(K, 7)$.
        labels: Class of every box, shape $(K,)$.
        batch: Scene index of every box, shape $(K,)$.
        num_samples: RoIs drawn per scene $S$.
        foreground_ratio: Largest share of foreground RoIs among the samples of a scene.
        foreground_threshold: IoU at or above which a RoI is foreground.
        hard_background_threshold: IoU at or above which a background RoI is hard.
        hard_background_ratio: Share of hard RoIs among the background samples, when the scene has both kinds.

    Returns:
        The sampled RoIs, their matched boxes and IoUs, ordered by scene.

    Shape:
        - rois: $(R, 7)$
        - boxes: $(K, 7)$
        - output: $(B S,)$ each

    Example:
        ```pycon
        >>> boxes = torch.tensor([[0.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0]])
        >>> rois = torch.cat([boxes, boxes + torch.tensor([[20.0, 0, 0, 0, 0, 0, 0]])])
        >>> zero = torch.zeros(2, dtype=torch.long)
        >>> samples = sample_rois(
        ...     rois,
        ...     zero,
        ...     zero,
        ...     boxes,
        ...     zero[:1],
        ...     zero[:1],
        ...     num_samples=2,
        ...     foreground_ratio=0.5,
        ...     foreground_threshold=0.55,
        ...     hard_background_threshold=0.1,
        ...     hard_background_ratio=0.8,
        ... )
        >>> samples["index"].tolist(), samples["target_index"].tolist(), samples["iou"].round(decimals=2).tolist()
        ([0, 1], [0, -1], [1.0, 0.0])

        ```
    """
    num_foreground = int(round(foreground_ratio * num_samples))
    index, target_index, ious = [], [], []
    for b in torch.unique(roi_batch):
        scene = (roi_batch == b).nonzero().view(-1)
        targets = (batch == b).nonzero().view(-1)
        iou = rois.new_zeros(scene.numel())
        target = scene.new_full((scene.numel(),), -1)
        if targets.numel() > 0:
            overlaps = boxes_iou3d(rois[scene], boxes[targets])
            overlaps = overlaps * (roi_labels[scene, None] == labels[None, targets])
            iou, best = overlaps.max(dim=1)
            target = torch.where(iou > 0, targets[best], target)

        foreground = (iou >= foreground_threshold).nonzero().view(-1)
        hard = ((iou < foreground_threshold) & (iou >= hard_background_threshold)).nonzero().view(-1)
        easy = (iou < hard_background_threshold).nonzero().view(-1)
        if hard.numel() + easy.numel() == 0:
            sampled = foreground[torch.randint(foreground.numel(), (num_samples,), device=foreground.device)]
        else:
            take = min(num_foreground, foreground.numel())
            foreground = foreground[torch.randperm(foreground.numel(), device=foreground.device)[:take]]
            background = _sample_background(hard, easy, num_samples - take, hard_background_ratio)
            sampled = torch.cat([foreground, background])

        index.append(scene[sampled])
        target_index.append(target[sampled])
        ious.append(iou[sampled])

    return {"index": torch.cat(index), "target_index": torch.cat(target_index), "iou": torch.cat(ious)}
