r"""Generic 3D oriented-box geometry shared by detection models and their evaluation.

Boxes are parameterized as $(c_x, c_y, c_z, d_x, d_y, d_z, \theta)$: gravity-aligned center, full
extents, and heading $\theta$ (radians) about the up axis $z$. Axis-aligned boxes set $\theta = 0$.
Corners are $(\ldots, 8, 3)$ with the top face (max $z$) first; IoU is frame-invariant so these work
in any right-handed frame.
"""

from typing import Tuple

import numpy as np
import torch
from scipy.spatial import ConvexHull
from torch import Tensor

_CORNER_X = torch.tensor([1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0])
_CORNER_Y = torch.tensor([1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0])
_CORNER_Z = torch.tensor([1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0])


def box_corners(boxes: Tensor) -> Tensor:
    r"""Convert parameterized boxes to their 8 corners.

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
    x = corners[..., 0] * cos + corners[..., 1] * sin
    y = -corners[..., 0] * sin + corners[..., 1] * cos
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
            iou[i, j] = inter_vol / (vol1 + _box_volume(corners2[j]) - inter_vol)

    return torch.from_numpy(inter), torch.from_numpy(iou)


def nms3d(boxes: Tensor, scores: Tensor, labels: Tensor, iou_threshold: float) -> Tensor:
    r"""Greedy axis-aligned 3D non-maximum suppression that only suppresses boxes of the same class.

    Args:
        boxes: Boxes $(N, 7)$ (see `box_corners`).
        scores: Per-box confidence, shape $(N,)$.
        labels: Per-box class, shape $(N,)$; only same-class boxes suppress each other.
        iou_threshold: Axis-aligned IoU above which a lower-scoring same-class box is removed.

    Returns:
        Indices of the kept boxes, highest score first, shape $(K,)$ long.
    """
    if boxes.numel() == 0:
        return boxes.new_zeros((0,), dtype=torch.long)

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
        suppress = (iou > iou_threshold) & (labels[rest] == labels[i])
        order = rest[~suppress]
    return torch.stack(keep) if keep else boxes.new_zeros((0,), dtype=torch.long)
