r"""Generic 3D oriented-box geometry shared by detection models and their evaluation.

Boxes are parameterized as $(c_x, c_y, c_z, d_x, d_y, d_z, \theta)$: gravity-aligned center, full
extents, and heading $\theta$ (radians) counter-clockwise about $+z$ from $+x$. Axis-aligned boxes set
$\theta = 0$.
Corners are $(\ldots, 8, 3)$ with the top face (max $z$) first; IoU is frame-invariant so these work
in any right-handed frame.
"""

import math
from typing import Tuple

import numpy as np
import torch
from scipy.spatial import ConvexHull
from torch import Tensor

import torch_pointcloud.transforms.functional as F
from torch_pointcloud.utils.types import OptTensor

_CORNER_X = torch.tensor([1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0])
_CORNER_Y = torch.tensor([1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0])
_CORNER_Z = torch.tensor([1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0])


def box_corners(boxes: Tensor) -> Tensor:
    r"""Convert parameterized boxes to their 8 corners.

    The heading is counter-clockwise about $+z$ from $+x$; boxes are $(c_x, c_y, c_z, d_x, d_y, d_z, \theta)$
    with full extents.

    Args:
        boxes: Boxes $(c_x, c_y, c_z, d_x, d_y, d_z, \theta)$, shape $(\ldots, 7)$.

    Returns:
        Corner coordinates, shape $(\ldots, 8, 3)$, with the top face (max $z$) as corners $0..3$.

    Shape:
        - boxes: $(\ldots, 7)$
        - output: $(\ldots, 8, 3)$
    """
    center, extent, heading = boxes[..., :3], boxes[..., 3:6], boxes[..., 6]
    signs = torch.stack([_CORNER_X, _CORNER_Y, _CORNER_Z], dim=-1).to(boxes)
    corners = 0.5 * signs * extent[..., None, :]
    cos, sin = torch.cos(heading)[..., None], torch.sin(heading)[..., None]
    x = corners[..., 0] * cos - corners[..., 1] * sin
    y = corners[..., 0] * sin + corners[..., 1] * cos
    rotated = torch.stack([x, y, corners[..., 2]], dim=-1)
    return rotated + center[..., None, :]


def decode_box_residuals(encodings: Tensor, anchors: Tensor, *, angle_by_sincos: bool = False) -> Tensor:
    r"""Decode predicted box residuals against anchors (OpenPCDet's `ResidualCoder`).

    Residuals encode the center offset normalized by the anchor base diagonal, log-size ratios, and an
    angle term: a plain delta by default, or a $(\cos, \sin)$ pair when `angle_by_sincos` (one extra
    channel). Trailing channels (e.g. nuScenes velocity) decode as plain deltas.

    Args:
        encodings: Predicted residuals, shape $(\ldots, 7 + C)$, or $(\ldots, 8 + C)$ with `angle_by_sincos`.
        anchors: Matching anchors $(x, y, z, d_x, d_y, d_z, \theta, \ldots)$, shape $(\ldots, 7 + C)$.
        angle_by_sincos: Whether the heading residual is encoded as $(\cos, \sin)$.

    Returns:
        Decoded boxes $(c_x, c_y, c_z, d_x, d_y, d_z, \theta, \ldots)$, shape $(\ldots, 7 + C)$.

    Shape:
        - encodings: $(\ldots, 7 + C)$ or $(\ldots, 8 + C)$
        - anchors: $(\ldots, 7 + C)$
        - output: $(\ldots, 7 + C)$

    Example:
        >>> anchors = torch.tensor([[0.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0]])
        >>> decode_box_residuals(torch.zeros(1, 7), anchors)
        tensor([[0.0000, 0.0000, 0.0000, 4.0000, 2.0000, 1.5000, 0.0000]])
    """
    xa, ya, za, dxa, dya, dza, ra, *cas = torch.split(anchors, 1, dim=-1)
    if not angle_by_sincos:
        xt, yt, zt, dxt, dyt, dzt, rt, *cts = torch.split(encodings, 1, dim=-1)
    else:
        xt, yt, zt, dxt, dyt, dzt, cost, sint, *cts = torch.split(encodings, 1, dim=-1)

    diagonal = torch.sqrt(dxa**2 + dya**2)
    xg = xt * diagonal + xa
    yg = yt * diagonal + ya
    zg = zt * dza + za
    dxg = torch.exp(dxt) * dxa
    dyg = torch.exp(dyt) * dya
    dzg = torch.exp(dzt) * dza
    if angle_by_sincos:
        rg = torch.atan2(sint + torch.sin(ra), cost + torch.cos(ra))
    else:
        rg = rt + ra
    cgs = [t + a for t, a in zip(cts, cas)]
    return torch.cat([xg, yg, zg, dxg, dyg, dzg, rg, *cgs], dim=-1)


def encode_box_residuals(boxes: Tensor, anchors: Tensor, *, angle_by_sincos: bool = False) -> Tensor:
    r"""Encode ground-truth boxes into anchor-relative residuals (inverse of `decode_box_residuals`).

    The exact inverse of [`decode_box_residuals`][torch_pointcloud.utils.box3d.decode_box_residuals]: the
    center offset is normalized by the anchor base diagonal, sizes become log ratios, and the heading
    becomes a plain delta or a $(\cos, \sin)$ pair. Extents are clamped to $10^{-5}$ before the log so a
    degenerate box does not produce a non-finite target.

    Args:
        boxes: Ground-truth boxes $(c_x, c_y, c_z, d_x, d_y, d_z, \theta, \ldots)$, shape $(\ldots, 7 + C)$.
        anchors: Matching anchors $(x, y, z, d_x, d_y, d_z, \theta, \ldots)$, shape $(\ldots, 7 + C)$.
        angle_by_sincos: Whether to encode the heading residual as $(\cos, \sin)$ (one extra channel).

    Returns:
        Residual encodings, shape $(\ldots, 7 + C)$, or $(\ldots, 8 + C)$ with `angle_by_sincos`.

    Shape:
        - boxes: $(\ldots, 7 + C)$
        - anchors: $(\ldots, 7 + C)$
        - output: $(\ldots, 7 + C)$ or $(\ldots, 8 + C)$

    Example:
        >>> anchors = torch.tensor([[0.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0]])
        >>> boxes = torch.tensor([[1.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0]])
        >>> torch.allclose(decode_box_residuals(encode_box_residuals(boxes, anchors), anchors), boxes)
        True
    """
    xa, ya, za, dxa, dya, dza, ra, *cas = torch.split(anchors, 1, dim=-1)
    xg, yg, zg, dxg, dyg, dzg, rg, *cgs = torch.split(boxes, 1, dim=-1)

    dxa, dya, dza = dxa.clamp_min(1e-5), dya.clamp_min(1e-5), dza.clamp_min(1e-5)
    dxg, dyg, dzg = dxg.clamp_min(1e-5), dyg.clamp_min(1e-5), dzg.clamp_min(1e-5)

    diagonal = torch.sqrt(dxa**2 + dya**2)
    xt = (xg - xa) / diagonal
    yt = (yg - ya) / diagonal
    zt = (zg - za) / dza
    dxt = torch.log(dxg / dxa)
    dyt = torch.log(dyg / dya)
    dzt = torch.log(dzg / dza)
    if angle_by_sincos:
        rts = [torch.cos(rg) - torch.cos(ra), torch.sin(rg) - torch.sin(ra)]
    else:
        rts = [rg - ra]
    cts = [g - a for g, a in zip(cgs, cas)]
    return torch.cat([xt, yt, zt, dxt, dyt, dzt, *rts, *cts], dim=-1)


def limit_period(val: Tensor, offset: float = 0.5, period: float = math.pi) -> Tensor:
    r"""Wrap an angle to $[-\text{offset} \cdot \text{period}, (1 - \text{offset}) \cdot \text{period})$."""
    return val - torch.floor(val / period + offset) * period


def _polygon_clip(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    """Sutherland-Hodgman clip of `subject` (2D, CCW) by the convex polygon `clip`."""

    def inside(p: np.ndarray) -> bool:
        return (cp2[0] - cp1[0]) * (p[1] - cp1[1]) > (cp2[1] - cp1[1]) * (p[0] - cp1[0])

    def intersection() -> np.ndarray:
        dc = cp1 - cp2
        dp = s - e
        n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
        n2 = s[0] * e[1] - s[1] * e[0]
        denom = dc[0] * dp[1] - dc[1] * dp[0]
        return np.array([(n1 * dp[0] - n2 * dc[0]) / denom, (n1 * dp[1] - n2 * dc[1]) / denom])

    output = list(subject)
    cp1 = clip[-1]
    for cp2 in clip:
        if not output:
            break
        ring, output = output, []
        s = ring[-1]
        for e in ring:
            if inside(e):
                if not inside(s):
                    output.append(intersection())
                output.append(e)
            elif inside(s):
                output.append(intersection())
            s = e
        cp1 = cp2
    return np.asarray(output)


def _sort_ccw(poly: np.ndarray) -> np.ndarray:
    """Order a convex polygon's vertices counter-clockwise (required by `_polygon_clip`)."""
    centroid = poly.mean(axis=0)
    angles = np.arctan2(poly[:, 1] - centroid[1], poly[:, 0] - centroid[0])
    return poly[np.argsort(angles)]


def _bev_intersection_area(rect1: np.ndarray, rect2: np.ndarray) -> float:
    """Area of the intersection of two convex BEV polygons given as $(4, 2)$ corner rings."""
    inter = _polygon_clip(_sort_ccw(rect1), _sort_ccw(rect2))
    if inter.shape[0] < 3 or not np.isfinite(inter).all():
        return 0.0
    try:
        return float(ConvexHull(inter).volume)
    except Exception:
        return 0.0


def _box_volume(corners: np.ndarray) -> float:
    dx = np.linalg.norm(corners[0] - corners[1])
    dy = np.linalg.norm(corners[1] - corners[2])
    dz = np.linalg.norm(corners[0] - corners[4])
    return float(dx * dy * dz)


def box3d_overlap(boxes1: Tensor, boxes2: Tensor) -> Tuple[Tensor, Tensor]:
    r"""Pairwise 3D intersection volume and IoU of two sets of boxes given as corners.

    Signature-compatible with `pytorch3d.ops.box3d_overlap`, so the exact CUDA implementation can be
    swapped in behind this interface. This fallback intersects the bird's-eye polygons (convex-hull
    clip) and multiplies by the vertical overlap.

    Args:
        boxes1: Corners of the first set, shape $(M, 8, 3)$ (see `box_corners`).
        boxes2: Corners of the second set, shape $(N, 8, 3)$.

    Returns:
        A tuple `(intersection_vol, iou)`, each shape $(M, N)$.

    Shape:
        - boxes1: $(M, 8, 3)$
        - boxes2: $(N, 8, 3)$
        - output: $(M, N)$, $(M, N)$
    """
    corners1 = boxes1.detach().cpu().numpy()
    corners2 = boxes2.detach().cpu().numpy()
    m, n = corners1.shape[0], corners2.shape[0]
    inter = np.zeros((m, n), dtype=np.float64)
    iou = np.zeros((m, n), dtype=np.float64)

    for i in range(m):
        vol1, rect1 = _box_volume(corners1[i]), corners1[i][:4, :2]
        ztop1, zbot1 = corners1[i][0, 2], corners1[i][4, 2]
        for j in range(n):
            area = _bev_intersection_area(rect1, corners2[j][:4, :2])
            height = max(0.0, min(ztop1, corners2[j][0, 2]) - max(zbot1, corners2[j][4, 2]))
            inter_vol = area * height
            inter[i, j] = inter_vol
            iou[i, j] = inter_vol / max(vol1 + _box_volume(corners2[j]) - inter_vol, 1e-8)

    return torch.from_numpy(inter), torch.from_numpy(iou)


def _bev_corners(boxes: Tensor) -> Tensor:
    r"""Bird's-eye corners of oriented boxes, ordered counter-clockwise.

    The heading rotates counter-clockwise (angle increases $x \to y$), so a box's corners are the
    axis-aligned rectangle $[-d_x/2, d_x/2] \times [-d_y/2, d_y/2]$ rotated by $+\theta$ about the center,
    matching `box_corners` and the detection models' heading convention.

    Args:
        boxes: Boxes $(c_x, c_y, c_z, d_x, d_y, d_z, \theta)$, shape $(K, 7)$.

    Returns:
        Corner coordinates $(x, y)$, shape $(K, 4, 2)$, ordered counter-clockwise.

    Shape:
        - boxes: $(K, 7)$
        - output: $(K, 4, 2)$
    """
    cx, cy, dx, dy, theta = boxes[:, 0], boxes[:, 1], boxes[:, 3], boxes[:, 4], boxes[:, 6]
    cos, sin = torch.cos(theta)[:, None], torch.sin(theta)[:, None]
    sign_x = boxes.new_tensor([-1.0, 1.0, 1.0, -1.0])
    sign_y = boxes.new_tensor([-1.0, -1.0, 1.0, 1.0])
    lx = sign_x * (dx / 2)[:, None]
    ly = sign_y * (dy / 2)[:, None]
    x = lx * cos - ly * sin + cx[:, None]
    y = lx * sin + ly * cos + cy[:, None]
    return torch.stack([x, y], dim=-1)


def _point_in_box(
    px: Tensor, py: Tensor, cx: Tensor, cy: Tensor, dx: Tensor, dy: Tensor, cos: Tensor, sin: Tensor
) -> Tensor:
    """Broadcasted test of whether points $(px, py)$ lie inside oriented BEV boxes (with a small margin)."""
    ux, uy = px - cx, py - cy
    lx = ux * cos + uy * sin
    ly = -ux * sin + uy * cos
    margin = 1e-2
    return (lx.abs() < dx / 2 + margin) & (ly.abs() < dy / 2 + margin)


def _rotated_box_bev_overlap(boxes_a: Tensor, boxes_b: Tensor) -> Tensor:
    r"""Pairwise BEV intersection area of two oriented-box sets, $(N, 7), (M, 7) \to (N, M)$.

    Vectorized clip-free polygon intersection: the intersection of two convex quads is the convex hull of
    (edge-edge crossings) + (corners of one box inside the other). Those candidate points are collected at
    fixed capacity, sorted counter-clockwise about their centroid, and the shoelace area is taken.
    """
    n, m = boxes_a.shape[0], boxes_b.shape[0]
    ca = _bev_corners(boxes_a)
    cb = _bev_corners(boxes_b)

    def cross2(u: Tensor, v: Tensor) -> Tensor:
        return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]

    a0 = ca[:, None, :, None, :].expand(n, m, 4, 1, 2)
    a1 = ca.roll(-1, dims=1)[:, None, :, None, :].expand(n, m, 4, 1, 2)
    b0 = cb[None, :, None, :, :].expand(n, m, 1, 4, 2)
    b1 = cb.roll(-1, dims=1)[None, :, None, :, :].expand(n, m, 1, 4, 2)
    r, s, qp = a1 - a0, b1 - b0, b0 - a0
    denom = cross2(r, s)
    t = cross2(qp, s) / denom
    u = cross2(qp, r) / denom
    edge_valid = (denom.abs() > 1e-12) & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
    edge_pts = a0 + t[..., None] * r
    edge_pts = torch.where(edge_valid[..., None], edge_pts, edge_pts.new_zeros(())).reshape(n, m, 16, 2)
    edge_valid = edge_valid.reshape(n, m, 16)

    ca_exp = ca[:, None, :, :].expand(n, m, 4, 2)
    cb_exp = cb[None, :, :, :].expand(n, m, 4, 2)
    cos_b, sin_b = torch.cos(boxes_b[:, 6]), torch.sin(boxes_b[:, 6])
    cos_a, sin_a = torch.cos(boxes_a[:, 6]), torch.sin(boxes_a[:, 6])
    a_in_b = _point_in_box(
        ca_exp[..., 0],
        ca_exp[..., 1],
        boxes_b[None, :, None, 0],
        boxes_b[None, :, None, 1],
        boxes_b[None, :, None, 3],
        boxes_b[None, :, None, 4],
        cos_b[None, :, None],
        sin_b[None, :, None],
    )
    b_in_a = _point_in_box(
        cb_exp[..., 0],
        cb_exp[..., 1],
        boxes_a[:, None, None, 0],
        boxes_a[:, None, None, 1],
        boxes_a[:, None, None, 3],
        boxes_a[:, None, None, 4],
        cos_a[:, None, None],
        sin_a[:, None, None],
    )

    pts = torch.cat([edge_pts, ca_exp, cb_exp], dim=2)
    valid = torch.cat([edge_valid, a_in_b, b_in_a], dim=2)
    count = valid.sum(-1)
    weight = valid[..., None].to(pts.dtype)
    centroid = (pts * weight).sum(2) / count.clamp(min=1)[..., None]
    rel = pts - centroid[:, :, None, :]
    angle = torch.atan2(rel[..., 1], rel[..., 0])
    angle = torch.where(valid, angle, torch.full_like(angle, 1e10))
    order = angle.argsort(dim=-1)
    pts = torch.gather(pts, 2, order[..., None].expand(n, m, pts.shape[2], 2))
    valid = torch.gather(valid, 2, order)
    pts = torch.where(valid[..., None], pts, pts[:, :, 0:1, :])
    x, y = pts[..., 0], pts[..., 1]
    area = 0.5 * (x * y.roll(-1, dims=-1) - x.roll(-1, dims=-1) * y).sum(-1).abs()
    return torch.where(count >= 3, area, area.new_zeros(()))


def boxes_iou_bev(boxes_a: Tensor, boxes_b: Tensor) -> Tensor:
    r"""Pairwise bird's-eye (top-down) rotated-box IoU.

    Projects both box sets onto the ground plane (ignoring $z$) and intersects the oriented rectangles. The
    heading is counter-clockwise (angle increases $x \to y$). Runs entirely in torch, so it stays on CUDA
    tensors without a custom extension; the cost is $O(N \cdot M)$.

    Args:
        boxes_a: Boxes $(c_x, c_y, c_z, d_x, d_y, d_z, \theta)$ with full extents, shape $(N, 7)$.
        boxes_b: Boxes in the same layout, shape $(M, 7)$.

    Returns:
        Pairwise BEV IoU in $[0, 1]$, shape $(N, M)$.

    Shape:
        - boxes_a: $(N, 7)$
        - boxes_b: $(M, 7)$
        - output: $(N, M)$

    Example:
        >>> a = torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]])
        >>> b = torch.tensor([[0.5, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]])
        >>> round(float(boxes_iou_bev(a, b)), 4)
        0.3333
    """
    inter = _rotated_box_bev_overlap(boxes_a, boxes_b)
    area_a = (boxes_a[:, 3] * boxes_a[:, 4])[:, None]
    area_b = (boxes_b[:, 3] * boxes_b[:, 4])[None, :]
    return inter / (area_a + area_b - inter).clamp(min=1e-8)


def boxes_iou3d(boxes_a: Tensor, boxes_b: Tensor) -> Tensor:
    r"""Pairwise oriented 3D box IoU.

    The BEV intersection area (rotated rectangles, ignoring $z$) is multiplied by the vertical overlap of
    the height intervals $[c_z - d_z/2, c_z + d_z/2]$ to give the intersection volume, then divided by the
    union. The heading is counter-clockwise about $+z$ from $+x$; boxes are
    $(c_x, c_y, c_z, d_x, d_y, d_z, \theta)$ with full extents. Runs entirely in torch (CUDA-capable, no
    custom extension); the cost is $O(N \cdot M)$.

    Args:
        boxes_a: Boxes $(c_x, c_y, c_z, d_x, d_y, d_z, \theta)$ with full extents, shape $(N, 7)$.
        boxes_b: Boxes in the same layout, shape $(M, 7)$.

    Returns:
        Pairwise 3D IoU in $[0, 1]$, shape $(N, M)$.

    Shape:
        - boxes_a: $(N, 7)$
        - boxes_b: $(M, 7)$
        - output: $(N, M)$

    Example:
        >>> a = torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]])
        >>> b = torch.tensor([[0.5, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]])
        >>> round(float(boxes_iou3d(a, b)), 4)
        0.3333
    """
    inter_area = _rotated_box_bev_overlap(boxes_a, boxes_b)
    z_a_max = (boxes_a[:, 2] + boxes_a[:, 5] / 2)[:, None]
    z_a_min = (boxes_a[:, 2] - boxes_a[:, 5] / 2)[:, None]
    z_b_max = (boxes_b[:, 2] + boxes_b[:, 5] / 2)[None, :]
    z_b_min = (boxes_b[:, 2] - boxes_b[:, 5] / 2)[None, :]
    h_overlap = (torch.min(z_a_max, z_b_max) - torch.max(z_a_min, z_b_min)).clamp(min=0)
    inter_3d = inter_area * h_overlap
    vol_a = (boxes_a[:, 3] * boxes_a[:, 4] * boxes_a[:, 5])[:, None]
    vol_b = (boxes_b[:, 3] * boxes_b[:, 4] * boxes_b[:, 5])[None, :]
    return inter_3d / (vol_a + vol_b - inter_3d).clamp(min=1e-6)


def _nms3d_single(boxes: Tensor, scores: Tensor, labels: OptTensor, iou_threshold: float) -> Tensor:
    """Greedy axis-aligned 3D NMS within a single scene; see `nms3d`."""
    corners = box_corners(boxes)
    lo, hi = corners.amin(dim=1), corners.amax(dim=1)
    volume = (hi - lo).clamp_min(0).prod(dim=-1)
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i)
        rest = order[1:]
        inter_lo = torch.maximum(lo[i], lo[rest])
        inter_hi = torch.minimum(hi[i], hi[rest])
        inter = (inter_hi - inter_lo).clamp_min(0).prod(dim=-1)
        iou = inter / (volume[i] + volume[rest] - inter)
        suppress = iou > iou_threshold
        if labels is not None:
            suppress = suppress & (labels[rest] == labels[i])
        order = rest[~suppress]
    return torch.stack(keep) if keep else boxes.new_zeros((0,), dtype=torch.long)


def nms3d(
    boxes: Tensor,
    scores: Tensor,
    iou_threshold: float,
    *,
    labels: OptTensor = None,
    batch: OptTensor = None,
) -> Tensor:
    r"""Greedy axis-aligned 3D non-maximum suppression.

    Keeps the highest-scoring box of each overlapping cluster. Pass `labels` to restrict suppression to
    boxes of the same class, and `batch` (PyG-style per-box scene index) to run NMS independently per
    scene and return a single index tensor over the concatenated input. The heading is counter-clockwise
    about $+z$ from $+x$; boxes are $(c_x, c_y, c_z, d_x, d_y, d_z, \theta)$ with full extents.

    Args:
        boxes: Boxes $(N, 7)$ (see `box_corners`).
        scores: Per-box confidence, shape $(N,)$.
        iou_threshold: Axis-aligned IoU above which a lower-scoring box is removed.
        labels: Optional per-box class, shape $(N,)$; when given, only same-class boxes suppress each other.
        batch: Optional per-box scene index, shape $(N,)$; when given, NMS runs independently per scene.

    Returns:
        Indices of the kept boxes (into the input), highest score first within each scene, shape $(K,)$ long.

    Shape:
        - boxes: $(N, 7)$
        - output: $(K,)$
    """
    if boxes.numel() == 0:
        return boxes.new_zeros((0,), dtype=torch.long)
    if batch is None:
        return _nms3d_single(boxes, scores, labels, iou_threshold)

    keep = []
    for b in torch.unique(batch):
        scene = (batch == b).nonzero(as_tuple=False).squeeze(-1)
        scene_labels = None if labels is None else labels[scene]
        keep.append(scene[_nms3d_single(boxes[scene], scores[scene], scene_labels, iou_threshold)])
    return torch.cat(keep) if keep else boxes.new_zeros((0,), dtype=torch.long)


def count_points_in_boxes(
    pos: Tensor, boxes: Tensor, *, pos_batch: OptTensor = None, box_batch: OptTensor = None
) -> Tensor:
    r"""Count how many points fall inside each oriented box.

    Pass `pos_batch` and `box_batch` (PyG-style per-point / per-box scene indices) to restrict each box's
    count to points from its own scene, so boxes of different scenes never share points. The heading is
    counter-clockwise about $+z$ from $+x$; boxes are $(c_x, c_y, c_z, d_x, d_y, d_z, \theta)$ with full
    extents.

    Args:
        pos: Point coordinates, shape $(N, 3)$.
        boxes: Boxes $(c_x, c_y, c_z, d_x, d_y, d_z, \theta)$ with full extents, shape $(K, 7)$.
        pos_batch: Optional per-point scene index, shape $(N,)$.
        box_batch: Optional per-box scene index, shape $(K,)$.

    Returns:
        Per-box point count, shape $(K,)$ long.

    Shape:
        - pos: $(N, 3)$
        - boxes: $(K, 7)$
        - output: $(K,)$

    Example:
        >>> pos = torch.tensor([[0.0, 0.0, 0.0], [5.0, 5.0, 5.0]])
        >>> boxes = torch.tensor([[0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0]])
        >>> count_points_in_boxes(pos, boxes).tolist()
        [1]
    """
    counts = boxes.new_zeros(boxes.shape[0], dtype=torch.long)
    for k in range(boxes.shape[0]):
        scene_pos = pos if pos_batch is None or box_batch is None else pos[pos_batch == box_batch[k]]
        half_box = torch.cat([boxes[k, :3], boxes[k, 3:6] / 2, boxes[k, 6:7]])
        counts[k] = int(F.points_in_oriented_box(scene_pos, half_box).sum())
    return counts
