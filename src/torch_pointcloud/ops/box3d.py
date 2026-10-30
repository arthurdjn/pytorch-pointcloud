r"""Generic 3D oriented-box geometry shared by detection models and their evaluation.

Boxes are parameterized as $(c_x, c_y, c_z, d_x, d_y, d_z, \theta)$: gravity-aligned center, full
extents, and heading $\theta$ (radians) counter-clockwise about $+z$ from $+x$. Axis-aligned boxes set
$\theta = 0$.
Corners are $(\ldots, 8, 3)$ with the top face (max $z$) first; IoU is frame-invariant so these work
in any right-handed frame.
"""

import math
from typing import Callable, Literal, Optional, Tuple

import torch
from torch import Tensor
from torch_geometric.utils import scatter

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
        ```pycon
        >>> anchors = torch.tensor([[0.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0]])
        >>> decode_box_residuals(torch.zeros(1, 7), anchors)
        tensor([[0.0000, 0.0000, 0.0000, 4.0000, 2.0000, 1.5000, 0.0000]])

        ```
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


def encode_box_residuals(boxes: Tensor, anchors: Tensor, *, angle_by_sincos: bool = False) -> Tensor:
    r"""Encode ground-truth boxes into anchor-relative residuals (inverse of `decode_box_residuals`).

    The exact inverse of [`decode_box_residuals`][torch_pointcloud.ops.box3d.decode_box_residuals]: the
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
        ```pycon
        >>> anchors = torch.tensor([[0.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0]])
        >>> boxes = torch.tensor([[1.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0]])
        >>> torch.allclose(decode_box_residuals(encode_box_residuals(boxes, anchors), anchors), boxes)
        True

        ```
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


def angle_to_class(angle: Tensor, num_heading_bins: int) -> Tuple[Tensor, Tensor]:
    r"""Convert continuous heading angles to discrete bin classes and residuals.

    The range $[0, 2\pi)$ is split into `num_heading_bins` equal bins centered at
    $0, 1 \cdot (2\pi / N), \ldots, (N - 1) \cdot (2\pi / N)$. The returned class and residual satisfy
    $\text{class} \cdot (2\pi / N) + \text{residual} = \text{angle}$.

    Args:
        angle: Heading angles in radians of shape $(K,)$.
        num_heading_bins: Number of heading bins $N$.

    Returns:
        A tuple of the per-angle class indices (long, shape $(K,)$) and residual angles (shape $(K,)$).
    """
    two_pi = 2 * math.pi
    angle_per_class = two_pi / num_heading_bins
    angle = angle % two_pi
    shifted = (angle + angle_per_class / 2) % two_pi
    # The division can round up to exactly N when `shifted` sits a float ulp below 2 pi; clamp keeps the
    # class in range.
    cls = (shifted / angle_per_class).long().clamp(max=num_heading_bins - 1)
    residual = shifted - (cls.to(angle.dtype) * angle_per_class + angle_per_class / 2)
    return cls, residual


def class_to_angle(heading_class: Tensor, heading_residual: Tensor, num_heading_bins: int) -> Tensor:
    r"""Invert `angle_to_class`: recover continuous heading angles from bin classes and residuals.

    A single bin (`num_heading_bins == 1`, axis-aligned boxes) always decodes to a heading of $0$.

    Args:
        heading_class: Bin class indices (long) of shape $(K,)$.
        heading_residual: Per-angle residuals of shape $(K,)$.
        num_heading_bins: Number of heading bins $N$.

    Returns:
        The recovered heading angles of shape $(K,)$.
    """
    if num_heading_bins == 1:
        return torch.zeros_like(heading_residual)
    return heading_class.to(heading_residual.dtype) * (2 * math.pi / num_heading_bins) + heading_residual


def class_to_size(size_class: Tensor, size_residual: Tensor, mean_sizes: Tensor) -> Tensor:
    r"""Recover full box edge lengths from a size class index and residual (inverse of the size encoding).

    Args:
        size_class: Size class indices (long) of shape $(K,)$.
        size_residual: Per-axis residuals of shape $(K, 3)$.
        mean_sizes: Template sizes of shape $(C, 3)$ holding full edge lengths per class.

    Returns:
        The recovered full edge lengths of shape $(K, 3)$.
    """
    return mean_sizes.to(size_residual)[size_class.long()] + size_residual


def limit_period(val: Tensor, offset: float = 0.5, period: float = math.pi) -> Tensor:
    r"""Wrap an angle to $[-\text{offset} \cdot \text{period}, (1 - \text{offset}) \cdot \text{period})$."""
    return val - torch.floor(val / period + offset) * period


# Slack of the bounds test that prefilters rotated BEV overlaps: two boxes whose rotated bounds are this close are
# still clipped, so rounding in the bounds never skips a pair that overlaps.
_BEV_MARGIN = 1e-2


def _sort_ring_ccw(ring: Tensor) -> Tensor:
    r"""Order convex rings of shape $(\ldots, V, 2)$ counter-clockwise about their centroid."""
    rel = ring - ring.mean(dim=-2, keepdim=True)
    order = torch.atan2(rel[..., 1], rel[..., 0]).argsort(dim=-1)
    return ring.gather(-2, order[..., None].expand_as(ring))


def _ring_area(ring: Tensor) -> Tensor:
    r"""Shoelace area of counter-clockwise rings of shape $(\ldots, V, 2)$."""
    nxt = ring.roll(-1, dims=-2)
    return 0.5 * (ring[..., 0] * nxt[..., 1] - ring[..., 1] * nxt[..., 0]).sum(dim=-1).abs()


def _convex_quad_intersection_area(quads_a: Tensor, quads_b: Tensor, eps: float = 1e-6) -> Tensor:
    """Pairwise intersection area of counter-clockwise convex quads $(M, 4, 2)$ and $(N, 4, 2)$.

    The polygon arithmetic runs only on the pairs whose axis-aligned bounds meet: the other pairs have no
    intersection, so their area is zero without any tolerance.
    """
    m, n = quads_a.shape[0], quads_b.shape[0]
    area = quads_a.new_zeros(m, n)
    if m == 0 or n == 0:
        return area

    lo_a, hi_a = quads_a.amin(dim=1), quads_a.amax(dim=1)
    lo_b, hi_b = quads_b.amin(dim=1), quads_b.amax(dim=1)
    meet = (lo_a[:, None] <= hi_b[None, :]) & (lo_b[None, :] <= hi_a[:, None])
    rows, cols = meet.all(dim=-1).nonzero(as_tuple=True)
    if rows.numel() > 0:
        area[rows, cols] = _quad_rings_intersection_area(quads_a[rows], quads_b[cols], eps)

    return area


def _quad_rings_intersection_area(a: Tensor, b: Tensor, eps: float = 1e-6) -> Tensor:
    r"""Intersection area of counter-clockwise convex quads paired over broadcast leading dims, $(\ldots, 4, 2)$."""
    lead = torch.broadcast_shapes(a.shape[:-2], b.shape[:-2])
    a = a.expand(*lead, 4, 2)
    b = b.expand(*lead, 4, 2)
    edge_a, edge_b = a.roll(-1, dims=-2) - a, b.roll(-1, dims=-2) - b

    # Candidate vertices of the intersection polygon: each quad's corners inside the other quad
    # (cross-product half-plane test against every CCW edge) plus all pairwise edge crossings.
    rel_ab = a.unsqueeze(-2) - b.unsqueeze(-3)  # (..., 4 verts, 4 edges, 2)
    cross_ab = edge_b.unsqueeze(-3)[..., 0] * rel_ab[..., 1] - edge_b.unsqueeze(-3)[..., 1] * rel_ab[..., 0]
    a_in_b = (cross_ab >= -eps).all(dim=-1)  # (..., 4)
    rel_ba = b.unsqueeze(-2) - a.unsqueeze(-3)
    cross_ba = edge_a.unsqueeze(-3)[..., 0] * rel_ba[..., 1] - edge_a.unsqueeze(-3)[..., 1] * rel_ba[..., 0]
    b_in_a = (cross_ba >= -eps).all(dim=-1)  # (..., 4)

    p, r = a.unsqueeze(-2), edge_a.unsqueeze(-2)  # segments of A vs segments of B: (..., 4, 1, 2)
    q, s = b.unsqueeze(-3), edge_b.unsqueeze(-3)  # (..., 1, 4, 2)
    denom = r[..., 0] * s[..., 1] - r[..., 1] * s[..., 0]  # (..., 4, 4)
    qp = q - p
    safe = denom.where(denom.abs() > eps, torch.ones_like(denom))
    t = (qp[..., 0] * s[..., 1] - qp[..., 1] * s[..., 0]) / safe
    u = (qp[..., 0] * r[..., 1] - qp[..., 1] * r[..., 0]) / safe
    crossing = (denom.abs() > eps) & (t >= -eps) & (t <= 1 + eps) & (u >= -eps) & (u <= 1 + eps)
    crossing_points = p + t.unsqueeze(-1) * r  # (..., 4, 4, 2)

    candidates = torch.cat([a, b, crossing_points.reshape(*lead, 16, 2)], dim=-2)  # (..., 24, 2)
    valid = torch.cat([a_in_b, b_in_a, crossing.reshape(*lead, 16)], dim=-1)  # (..., 24)
    count = valid.sum(dim=-1)

    # Sort the valid candidates counter-clockwise about their centroid (invalid ones to the end), then
    # take the shoelace sum over the first `count` entries with a per-pair wrap-around.
    centroid = (candidates * valid[..., None]).sum(dim=-2) / count.clamp(min=1)[..., None]
    rel = candidates - centroid.unsqueeze(-2)
    angles = torch.atan2(rel[..., 1], rel[..., 0]).where(valid, candidates.new_tensor(math.inf))
    order = angles.argsort(dim=-1)
    ring = candidates.gather(-2, order[..., None].expand(*lead, 24, 2))
    index = torch.arange(24, device=candidates.device).expand(*lead, 24)
    wrapped = torch.where(index + 1 < count[..., None], index + 1, torch.zeros_like(index))
    nxt = ring.gather(-2, wrapped[..., None].expand(*lead, 24, 2))
    terms = (ring[..., 0] * nxt[..., 1] - ring[..., 1] * nxt[..., 0]) * (index < count[..., None])

    return torch.where(count >= 3, 0.5 * terms.sum(dim=-1).abs(), terms.new_zeros(lead))


def box3d_overlap(boxes1: Tensor, boxes2: Tensor, *, aligned: bool = False) -> Tuple[Tensor, Tensor]:
    r"""Pairwise 3D intersection volume and IoU of two sets of boxes given as corners.

    Signature-compatible with `pytorch3d.ops.box3d_overlap`, so the exact CUDA implementation can be
    swapped in behind this interface. Boxes are assumed gravity-aligned (as produced by `box_corners`):
    the bird's-eye polygons are intersected exactly (corner containment plus edge crossings) and scaled
    by the vertical overlap, entirely on the input device. The result is differentiable with respect to the
    corners, so it can drive an IoU loss.

    Args:
        boxes1: Corners of the first set, shape $(M, 8, 3)$ (see `box_corners`).
        boxes2: Corners of the second set, shape $(N, 8, 3)$.
        aligned: With `True`, `boxes1` and `boxes2` pair up row by row ($M = N$) and the results are $(M,)$.

    Returns:
        A tuple `(intersection_vol, iou)`, each shape $(M, N)$, or $(M,)$ with `aligned`.

    Shape:
        - boxes1: $(M, 8, 3)$
        - boxes2: $(N, 8, 3)$
        - output: $(M, N)$, $(M, N)$, or $(M,)$, $(M,)$ with `aligned`
    """
    if aligned and boxes1.shape[0] != boxes2.shape[0]:
        raise ValueError(f"`aligned` needs as many boxes on both sides, got {boxes1.shape[0]} and {boxes2.shape[0]}.")

    m, n = boxes1.shape[0], boxes2.shape[0]
    if m == 0 or n == 0:
        shape = (m,) if aligned else (m, n)
        return boxes1.new_zeros(shape), boxes1.new_zeros(shape)

    ring1 = _sort_ring_ccw(boxes1[:, :4, :2])
    ring2 = _sort_ring_ccw(boxes2[:, :4, :2])
    area1, area2 = _ring_area(ring1), _ring_area(ring2)
    top1, bot1 = boxes1[..., 2].amax(dim=-1), boxes1[..., 2].amin(dim=-1)
    top2, bot2 = boxes2[..., 2].amax(dim=-1), boxes2[..., 2].amin(dim=-1)
    vol1 = area1 * (top1 - bot1)
    vol2 = area2 * (top2 - bot2)

    if aligned:
        bev = _quad_rings_intersection_area(ring1, ring2)
        height = (torch.minimum(top1, top2) - torch.maximum(bot1, bot2)).clamp_min(0)
        inter = bev * height
        return inter, inter / (vol1 + vol2 - inter).clamp_min(1e-8)

    bev = _convex_quad_intersection_area(ring1, ring2)
    height = (torch.minimum(top1[:, None], top2[None, :]) - torch.maximum(bot1[:, None], bot2[None, :])).clamp_min(0)
    inter = bev * height
    iou = inter / (vol1[:, None] + vol2[None, :] - inter).clamp_min(1e-8)
    return inter, iou


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


def _bev_half_extents(boxes: Tensor) -> Tuple[Tensor, Tensor]:
    r"""Half extents of the axis-aligned bounding box of each rotated BEV box, shape $(K,)$ each."""
    cos, sin = torch.cos(boxes[:, 6]).abs(), torch.sin(boxes[:, 6]).abs()
    half_x, half_y = boxes[:, 3] / 2, boxes[:, 4] / 2
    return half_x * cos + half_y * sin, half_x * sin + half_y * cos


def _rotated_box_bev_overlap(boxes_a: Tensor, boxes_b: Tensor, *, aligned: bool = False) -> Tensor:
    r"""BEV intersection area of oriented boxes: pairwise $(N, 7), (M, 7) \to (N, M)$, or per pair with `aligned`."""
    corners_a, corners_b = _bev_corners(boxes_a), _bev_corners(boxes_b)
    # The intersection never exceeds the smaller footprint: rounding cannot push the IoU of identical boxes above 1.
    area_a, area_b = boxes_a[:, 3] * boxes_a[:, 4], boxes_b[:, 3] * boxes_b[:, 4]
    if aligned:
        if boxes_a.shape[0] != boxes_b.shape[0]:
            raise ValueError(
                f"`aligned` needs as many boxes on both sides, got {boxes_a.shape[0]} and {boxes_b.shape[0]}."
            )
        return torch.minimum(_quad_overlap(corners_a, corners_b), torch.minimum(area_a, area_b))

    # Pairwise: two boxes whose axis-aligned bounds do not meet have no intersection, so the clipping runs only on
    # the candidate pairs, written into a zero matrix.
    overlap = boxes_a.new_zeros(boxes_a.shape[0], boxes_b.shape[0])
    half_x_a, half_y_a = _bev_half_extents(boxes_a)
    half_x_b, half_y_b = _bev_half_extents(boxes_b)
    meet_x = (boxes_a[:, None, 0] - boxes_b[None, :, 0]).abs() <= half_x_a[:, None] + half_x_b[None, :] + _BEV_MARGIN
    meet_y = (boxes_a[:, None, 1] - boxes_b[None, :, 1]).abs() <= half_y_a[:, None] + half_y_b[None, :] + _BEV_MARGIN
    rows, cols = (meet_x & meet_y).nonzero(as_tuple=True)
    if rows.numel() > 0:
        inter = _quad_overlap(corners_a[rows], corners_b[cols])
        overlap[rows, cols] = torch.minimum(inter, torch.minimum(area_a[rows], area_b[cols]))

    return overlap


def _quad_overlap(corners_a: Tensor, corners_b: Tensor) -> Tensor:
    r"""Intersection area of counter-clockwise convex quads over broadcast leading dims, $(\ldots, 4, 2)$ each.

    By Green's theorem the area is half the integral of $x\,dy - y\,dx$ along the boundary of the intersection: the
    edges of each quad clipped to the other quad. An edge $p + t\,d$ kept from $t_0$ to $t_1$ contributes
    $\frac{1}{2}(t_1 - t_0)\,(p \times d)$. Both clippings read one set of decisions per pair of edges (whether they
    are parallel, which side one lies on, where they cross), so rounding cannot count a boundary of the other quad
    once instead of zero or two times.
    """
    origin = corners_a.mean(dim=-2, keepdim=True)
    quad_a, quad_b = corners_a - origin, corners_b - origin
    edge_a = quad_a.roll(-1, dims=-2) - quad_a
    edge_b = quad_b.roll(-1, dims=-2) - quad_b

    # Pairs of edge i of `quad_a` (rows) and edge k of `quad_b` (columns), (..., 4, 4).
    start_a, along_a = quad_a[..., :, None, :], edge_a[..., :, None, :]
    start_b, along_b = quad_b[..., None, :, :], edge_b[..., None, :, :]
    gap = start_a - start_b
    # Side of the start of edge i against edge k (positive inside `quad_b`) and its rate along edge i; edge k
    # crosses the line of edge i at the opposite rate.
    side = along_b[..., 0] * gap[..., 1] - along_b[..., 1] * gap[..., 0]
    gap_along_b = gap[..., 0] * along_b[..., 0] + gap[..., 1] * along_b[..., 1]
    del gap  # the per-pair vectors dominate the memory of a large batch of pairs
    rate = along_b[..., 0] * along_a[..., 1] - along_b[..., 1] * along_a[..., 0]
    dot = along_a[..., 0] * along_b[..., 0] + along_a[..., 1] * along_b[..., 1]
    length_a = torch.linalg.vector_norm(along_a, dim=-1)
    length_b = torch.linalg.vector_norm(along_b, dim=-1)
    parallel = rate.abs() <= 16 * torch.finfo(rate.dtype).eps * length_a * length_b
    same_way = dot > 0
    # Edges of zero length bound nothing: a degenerate quad keeps no edge of the other one.
    degenerate_a, degenerate_b = length_a == 0, length_b == 0

    # Where edge i crosses the line of edge k, as a fraction of edge i, then of edge k.
    crossing = -side / torch.where(parallel, torch.ones_like(rate), rate)
    squared_b = torch.where(degenerate_b, 1.0, length_b * length_b)
    crossing_b = (gap_along_b + crossing * dot) / squared_b

    enter_a = torch.where(~parallel & (rate > 0), crossing, 0.0).amax(dim=-1)
    leave_a = torch.where(~parallel & (rate < 0), crossing, 1.0).amin(dim=-1)
    outside_a = degenerate_b | (parallel & ((side < 0) | ((side == 0) & ~same_way)))
    kept_a = torch.where(outside_a.any(dim=-1), 0.0, (leave_a - enter_a).clamp_min(0))

    # A boundary the two quads share is counted once, through the edges of `quad_a`.
    enter_b = torch.where(~parallel & (rate < 0), crossing_b, 0.0).amax(dim=-2)
    leave_b = torch.where(~parallel & (rate > 0), crossing_b, 1.0).amin(dim=-2)
    outside_b = degenerate_a | (parallel & torch.where(same_way, side >= 0, side <= 0))
    kept_b = torch.where(outside_b.any(dim=-2), 0.0, (leave_b - enter_b).clamp_min(0))

    moment_a = quad_a[..., 0] * edge_a[..., 1] - quad_a[..., 1] * edge_a[..., 0]
    moment_b = quad_b[..., 0] * edge_b[..., 1] - quad_b[..., 1] * edge_b[..., 0]
    area = 0.5 * ((kept_a * moment_a).sum(dim=-1) + (kept_b * moment_b).sum(dim=-1))
    return area.clamp_min(0)


def boxes_iou_bev(boxes_a: Tensor, boxes_b: Tensor, *, aligned: bool = False) -> Tensor:
    r"""Pairwise bird's-eye (top-down) rotated-box IoU.

    Projects both box sets onto the ground plane (ignoring $z$) and intersects the oriented rectangles. The
    heading is counter-clockwise (angle increases $x \to y$). Runs entirely in torch, so it stays on CUDA
    tensors without a custom extension; the cost is $O(N \cdot M)$.

    Args:
        boxes_a: Boxes $(c_x, c_y, c_z, d_x, d_y, d_z, \theta)$ with full extents, shape $(N, 7)$.
        boxes_b: Boxes in the same layout, shape $(M, 7)$.
        aligned: With `True`, `boxes_a` and `boxes_b` pair up row by row ($N = M$) and the result is $(N,)$.

    Returns:
        Pairwise BEV IoU in $[0, 1]$, shape $(N, M)$.

    Shape:
        - boxes_a: $(N, 7)$
        - boxes_b: $(M, 7)$
        - output: $(N, M)$

    Example:
        ```pycon
        >>> a = torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]])
        >>> b = torch.tensor([[0.5, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]])
        >>> round(float(boxes_iou_bev(a, b)), 4)
        0.3333

        ```
    """
    inter = _rotated_box_bev_overlap(boxes_a, boxes_b, aligned=aligned)
    area_a = boxes_a[:, 3] * boxes_a[:, 4]
    area_b = boxes_b[:, 3] * boxes_b[:, 4]
    if not aligned:
        area_a, area_b = area_a[:, None], area_b[None, :]

    return inter / (area_a + area_b - inter).clamp(min=1e-8)


def _nearest_aligned_bev_footprint(boxes: Tensor) -> Tensor:
    r"""$(x_1, y_1, x_2, y_2)$ of each box once rotated about its center to the nearest multiple of $\pi / 2$."""
    yaw = limit_period(boxes[:, 6], offset=0.5, period=math.pi).abs()
    dims = torch.where(yaw[:, None] < math.pi / 4, boxes[:, 3:5], boxes[:, [4, 3]])
    return torch.cat([boxes[:, :2] - dims / 2, boxes[:, :2] + dims / 2], dim=1)


def boxes_iou_nearest_bev(boxes_a: Tensor, boxes_b: Tensor) -> Tensor:
    r"""Pairwise bird's-eye IoU of boxes snapped to their nearest axis-aligned orientation.

    Each box keeps its center and takes the footprint of its heading rounded to the nearest multiple of
    $\pi / 2$, so $d_x$ and $d_y$ swap when the heading is closer to $\pm \pi / 2$ than to $0$; the IoU is then
    the plain axis-aligned rectangle IoU. The anchor-based detectors assign their anchors by this measure
    rather than by the exact rotated overlap of [`boxes_iou_bev`][torch_pointcloud.ops.box3d.boxes_iou_bev].

    Args:
        boxes_a: Boxes $(c_x, c_y, c_z, d_x, d_y, d_z, \theta)$ with full extents, shape $(N, 7)$.
        boxes_b: Boxes in the same layout, shape $(M, 7)$.

    Returns:
        Pairwise IoU in $[0, 1]$, shape $(N, M)$.

    Shape:
        - boxes_a: $(N, 7)$
        - boxes_b: $(M, 7)$
        - output: $(N, M)$

    Example:
        ```pycon
        >>> a = torch.tensor([[0.0, 0.0, 0.0, 2.0, 1.0, 1.0, 0.0]])
        >>> b = torch.tensor([[0.0, 0.0, 0.0, 1.0, 2.0, 1.0, math.pi / 2]])
        >>> round(float(boxes_iou_nearest_bev(a, b)), 4)
        1.0

        ```
    """
    footprint_a = _nearest_aligned_bev_footprint(boxes_a)
    footprint_b = _nearest_aligned_bev_footprint(boxes_b)
    lo = torch.maximum(footprint_a[:, None, :2], footprint_b[None, :, :2])
    hi = torch.minimum(footprint_a[:, None, 2:], footprint_b[None, :, 2:])
    inter = (hi - lo).clamp_min(0).prod(dim=-1)
    area_a = (footprint_a[:, 2] - footprint_a[:, 0]) * (footprint_a[:, 3] - footprint_a[:, 1])
    area_b = (footprint_b[:, 2] - footprint_b[:, 0]) * (footprint_b[:, 3] - footprint_b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter).clamp(min=1e-8)


def boxes_iou3d(boxes_a: Tensor, boxes_b: Tensor, *, aligned: bool = False) -> Tensor:
    r"""Pairwise oriented 3D box IoU.

    The BEV intersection area (rotated rectangles, ignoring $z$) is multiplied by the vertical overlap of
    the height intervals $[c_z - d_z/2, c_z + d_z/2]$ to give the intersection volume, then divided by the
    union. The heading is counter-clockwise about $+z$ from $+x$; boxes are
    $(c_x, c_y, c_z, d_x, d_y, d_z, \theta)$ with full extents. Runs entirely in torch (CUDA-capable, no
    custom extension); the cost is $O(N \cdot M)$.

    Args:
        boxes_a: Boxes $(c_x, c_y, c_z, d_x, d_y, d_z, \theta)$ with full extents, shape $(N, 7)$.
        boxes_b: Boxes in the same layout, shape $(M, 7)$.
        aligned: With `True`, `boxes_a` and `boxes_b` pair up row by row ($N = M$) and the result is $(N,)$.

    Returns:
        Pairwise 3D IoU in $[0, 1]$, shape $(N, M)$.

    Shape:
        - boxes_a: $(N, 7)$
        - boxes_b: $(M, 7)$
        - output: $(N, M)$

    Example:
        ```pycon
        >>> a = torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]])
        >>> b = torch.tensor([[0.5, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]])
        >>> round(float(boxes_iou3d(a, b)), 4)
        0.3333

        ```
    """
    inter_area = _rotated_box_bev_overlap(boxes_a, boxes_b, aligned=aligned)
    z_a_max, z_a_min = boxes_a[:, 2] + boxes_a[:, 5] / 2, boxes_a[:, 2] - boxes_a[:, 5] / 2
    z_b_max, z_b_min = boxes_b[:, 2] + boxes_b[:, 5] / 2, boxes_b[:, 2] - boxes_b[:, 5] / 2
    vol_a = boxes_a[:, 3] * boxes_a[:, 4] * boxes_a[:, 5]
    vol_b = boxes_b[:, 3] * boxes_b[:, 4] * boxes_b[:, 5]
    if not aligned:
        z_a_max, z_a_min, vol_a = z_a_max[:, None], z_a_min[:, None], vol_a[:, None]
        z_b_max, z_b_min, vol_b = z_b_max[None, :], z_b_min[None, :], vol_b[None, :]

    h_overlap = (torch.min(z_a_max, z_b_max) - torch.max(z_a_min, z_b_min)).clamp(min=0)
    inter_3d = inter_area * h_overlap
    return inter_3d / (vol_a + vol_b - inter_3d).clamp(min=1e-6)


def _overlapping_pairs(lo: Tensor, hi: Tensor, group: Tensor) -> Tuple[Tensor, Tensor]:
    r"""Pairs $(i, j)$, $i < j$, of intervals $[lo, hi]$ of the same group that overlap or touch, by sort and sweep.

    The starts and ends of the intervals are sorted together by group then value, starts first on ties, and each
    interval pairs with the intervals that start between its own start and end.
    """
    n = len(lo)
    values = torch.cat([lo, hi])
    events = values.argsort(stable=True)
    event_groups = group.repeat(2)[events]
    events = events[event_groups.argsort(stable=True)]
    slots = torch.empty_like(events)
    slots[events] = torch.arange(2 * n, device=lo.device)
    starts = (events < n).cumsum(0)
    first, last = starts[slots[:n]], starts[slots[n:]]
    count = (last - first).clamp_min(0)
    total = int(count.sum())
    # The pair lists hold box indices and pair positions: 32 bits are exact below 2**31 and halve their memory.
    index = torch.int32 if max(n, total) < 2**31 else torch.int64
    boxes = torch.arange(n, dtype=index, device=lo.device)
    i = torch.repeat_interleave(boxes, count, output_size=total)
    run_start = (count.cumsum(0) - count).to(index)
    offset = torch.arange(total, dtype=index, device=lo.device) - run_start[i]
    by_start = torch.empty_like(boxes)
    by_start[first - 1] = boxes
    j = by_start[first.to(index)[i] + offset]
    return torch.minimum(i, j), torch.maximum(i, j)


def _greedy_suppression(
    first: Tensor,
    second: Tensor,
    group: Tensor,
    sizes: Tensor,
    suppresses: Optional[Callable[[Tensor, Tensor], Tensor]],
    max_keep: Optional[int],
) -> Tensor:
    r"""Mask of the boxes that greedy suppression keeps, from the pairs of boxes that can suppress each other.

    The boxes run group by group (`group`, `sizes` boxes each), each group by decreasing score, and box `first[k]` ranks
    above box `second[k]` in their group. A round keeps every undecided box that no undecided box of its pairs ranks
    above, and removes the boxes they suppress: all their pairs, or those where `suppresses(first, second)` holds. Each
    round decides at least the first undecided box of every group. A pair is evaluated once its first box is kept, or
    earlier when it is the last pair that can suppress its second box.
    With `max_keep`, a group stops once `max_keep` of its boxes are kept before its first undecided box.
    """
    n = len(group)
    positions = torch.arange(n, device=group.device)
    end = sizes.cumsum(0)
    start = end - sizes
    undecided = torch.ones(n, dtype=torch.bool, device=group.device)
    kept = torch.zeros(n, dtype=torch.bool, device=group.device)
    while True:
        blocked = torch.zeros_like(undecided)
        blocked[second] = True
        new = undecided & ~blocked
        kept |= new
        undecided &= ~new

        # Besides the pairs of the kept boxes, the pair of a box that a single undecided box can still suppress is
        # evaluated early: if it does not suppress, the box is free next round instead of waiting on that box.
        from_kept = new[first]
        parents = torch.zeros(n, dtype=second.dtype, device=group.device).index_add_(0, second, torch.ones_like(second))
        evaluated = (from_kept | (parents[second] == 1)).nonzero().squeeze(-1)
        source, target, from_kept = first[evaluated], second[evaluated], from_kept[evaluated]
        suppressed = torch.ones_like(target, dtype=torch.bool) if suppresses is None else suppresses(source, target)
        suppressors = torch.zeros(n, dtype=torch.long, device=group.device).index_put_(
            (target,),
            (suppressed & from_kept).long(),
            accumulate=True,
        )
        undecided &= suppressors == 0

        if max_keep is not None:
            first_undecided = torch.where(undecided, positions, n)
            prefix_end = end.scatter_reduce(0, group, first_undecided, "amin")
            kept_before = torch.cat([kept.new_zeros(1, dtype=torch.long), kept.cumsum(0)])
            finished = kept_before[prefix_end] - kept_before[start] >= max_keep
            undecided &= ~finished[group]

        # The pairs of kept boxes are done, and so are the early pairs that do not suppress.
        pending = undecided[first] & undecided[second]
        pending.index_fill_(0, evaluated[from_kept | ~suppressed], False)
        remaining = pending.nonzero().squeeze(-1)
        first, second = first[remaining], second[remaining]
        if first.numel() == 0:
            return kept | undecided


def nms3d(
    boxes: Tensor,
    scores: Tensor,
    iou_threshold: float,
    *,
    labels: OptTensor = None,
    batch: OptTensor = None,
    rotated: bool = False,
    max_keep: Optional[int] = None,
    max_candidates: Optional[int] = None,
) -> Tensor:
    r"""Greedy 3D non-maximum suppression.

    Keeps the highest-scoring box of each overlapping cluster. By default the suppression criterion is the
    3D IoU of the boxes' axis-aligned bounding boxes (cheap corner min / max); with `rotated=True` it is
    the exact rotated bird's-eye IoU of `boxes_iou_bev` (the KITTI outdoor protocol, where the axis-aligned
    surrogate over-suppresses angled neighbors at low thresholds). Pass `labels` to restrict suppression to
    boxes of the same class, and `batch` (PyG-style per-box scene index) to run NMS independently per
    scene and return a single index tensor over the concatenated input. The heading is counter-clockwise
    about $+z$ from $+x$; boxes are $(c_x, c_y, c_z, d_x, d_y, d_z, \theta)$ with full extents.

    Only boxes whose bounds meet can suppress each other: a sort and sweep finds these pairs for all the scenes and
    classes at once, so memory grows with the number of overlapping boxes. A rotated IoU is computed only for a pair
    that can still decide whether a box is kept.

    Args:
        boxes: Boxes $(N, 7)$ (see `box_corners`).
        scores: Per-box confidence, shape $(N,)$.
        iou_threshold: Non-negative IoU above which a lower-scoring box is removed.
        labels: Optional per-box class, shape $(N,)$; when given, only same-class boxes suppress each other.
        batch: Optional per-box scene index, shape $(N,)$; when given, NMS runs independently per scene.
        rotated: Suppress on the exact rotated BEV IoU (`boxes_iou_bev`) instead of the axis-aligned 3D IoU.
        max_keep: Optional cap on the boxes kept per scene; suppression stops once it is reached, so the kept set
            equals the first `max_keep` entries of the uncapped result.
        max_candidates: Optional cap on the boxes considered per scene: only the `max_candidates` highest-scoring
            boxes enter the suppression, the others are dropped.

    Returns:
        Indices of the kept boxes (into the input), highest score first within each scene, shape $(K,)$ long.

    Shape:
        - boxes: $(N, 7)$
        - output: $(K,)$

    Example:
        ```pycon
        >>> boxes = torch.tensor([[0.0, 0, 0, 2, 2, 2, 0], [0.1, 0, 0, 2, 2, 2, 0], [5.0, 0, 0, 2, 2, 2, 0]])
        >>> scores = torch.tensor([0.9, 0.8, 0.7])
        >>> nms3d(boxes, scores, 0.5).tolist()
        [0, 2]

        ```
    """
    if iou_threshold < 0:
        raise ValueError(f"`iou_threshold` must be non-negative, got {iou_threshold}.")
    if (max_keep is not None and max_keep < 0) or (max_candidates is not None and max_candidates < 0):
        raise ValueError(f"`max_keep` and `max_candidates` must be non-negative, got {max_keep} and {max_candidates}.")

    if boxes.numel() == 0:
        return boxes.new_zeros((0,), dtype=torch.long)

    # Each scene sorts its own scores, so tied boxes rank as they would in a call on that scene alone.
    if batch is None:
        scene = boxes.new_zeros(len(boxes), dtype=torch.long)
        orders = [scores.argsort(descending=True)[:max_candidates]]
    else:
        _, scene, counts = torch.unique(batch, return_inverse=True, return_counts=True)
        members = scene.argsort(stable=True).split(counts.tolist())
        orders = [index[scores[index].argsort(descending=True)][:max_candidates] for index in members]
    order = torch.cat(orders)
    if order.numel() == 0:
        return order

    key = scene[order]
    if labels is not None:
        classes, label = torch.unique(labels[order], return_inverse=True)
        key = key * len(classes) + label
    _, group, sizes = torch.unique(key, return_inverse=True, return_counts=True)
    ranks = group.argsort(stable=True)
    group = group[ranks]
    # Only indices come out: detached boxes keep the IoU of every compared pair out of the autograd graph.
    ranked = boxes.detach()[order[ranks]]

    if rotated:
        # Rotated footprints of angled neighbors overlap far less than their AABBs at low thresholds.
        # Floor the BEV extents so coincident zero-area duplicates reach IoU 1 and suppress; the floored
        # area (1e-4) stays above the union clamp of `boxes_iou_bev`, and real boxes are far larger.
        ranked[:, 3:5] = ranked[:, 3:5].clamp_min(1e-2)
        half_x, half_y = _bev_half_extents(ranked)
        # The sweep bounds carry the margin of the meeting test on both sides, so rounding cannot drop a pair.
        reach = half_x + _BEV_MARGIN
        first, second = _overlapping_pairs(ranked[:, 0] - reach, ranked[:, 0] + reach, group)
        # As in `boxes_iou_bev`, only the pairs whose rotated bounds meet can overlap.
        meet_x = (ranked[first, 0] - ranked[second, 0]).abs() <= half_x[first] + half_x[second] + _BEV_MARGIN
        meet_y = (ranked[first, 1] - ranked[second, 1]).abs() <= half_y[first] + half_y[second] + _BEV_MARGIN
        candidates = meet_x & meet_y

        def suppresses(source: Tensor, target: Tensor) -> Tensor:
            return boxes_iou_bev(ranked[source], ranked[target], aligned=True) > iou_threshold

        suppression: Optional[Callable[[Tensor, Tensor], Tensor]] = suppresses
    else:
        corners = box_corners(ranked)
        lo, hi = corners.amin(dim=1), corners.amax(dim=1)
        # Floor each AABB extent so zero-size boxes keep a positive volume above the union clamp below, so a
        # coincident duplicate reaches IoU 1 instead of ~1e-12.
        hi = torch.maximum(hi, lo + 1e-2)
        volume = (hi - lo).prod(dim=-1)
        first, second = _overlapping_pairs(lo[:, 0], hi[:, 0], group)
        inter = (torch.minimum(hi[first], hi[second]) - torch.maximum(lo[first], lo[second])).clamp_min(0).prod(dim=-1)
        iou = inter / (volume[first] + volume[second] - inter).clamp_min(1e-6)
        # The axis-aligned IoU is cheap enough to evaluate on every overlapping pair up front.
        candidates = iou > iou_threshold
        suppression = None
    candidate = candidates.nonzero().squeeze(-1)
    kept = _greedy_suppression(first[candidate], second[candidate], group, sizes, suppression, max_keep)

    # A scene keeps its first `max_keep` boxes by rank: the ranks run scene by scene, by decreasing score.
    kept_ranks = ranks[kept].sort().values
    if max_keep is not None:
        kept_scene = scene[order[kept_ranks]]
        scene_start = torch.searchsorted(kept_scene, kept_scene)
        rank_in_scene = torch.arange(len(kept_ranks), device=kept_ranks.device) - scene_start
        kept_ranks = kept_ranks[rank_in_scene < max_keep]
    return order[kept_ranks]


def points_in_boxes(pos: Tensor, boxes: Tensor) -> Tensor:
    r"""Test every point against every oriented box.

    The point offsets to each box center are rotated into the box frame by $-\theta$ about $+z$ and
    compared with the half extents. The heading is counter-clockwise about $+z$ from $+x$; boxes are
    $(c_x, c_y, c_z, d_x, d_y, d_z, \theta)$ with full extents. Pairs are materialized at once, so
    restrict large scenes to their own boxes (or use `count_points_in_boxes`, which chunks).

    Args:
        pos: Point coordinates, shape $(N, 3)$.
        boxes: Boxes $(c_x, c_y, c_z, d_x, d_y, d_z, \theta)$ with full extents, shape $(K, 7)$.

    Returns:
        Boolean containment mask, shape $(N, K)$.

    Shape:
        - pos: $(N, 3)$
        - boxes: $(K, 7)$
        - output: $(N, K)$

    Example:
        ```pycon
        >>> pos = torch.tensor([[0.0, 0.0, 0.0], [5.0, 5.0, 5.0]])
        >>> boxes = torch.tensor([[0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0], [5.0, 5.0, 5.0, 1.0, 1.0, 1.0, 0.3]])
        >>> points_in_boxes(pos, boxes).tolist()
        [[True, False], [False, True]]

        ```
    """
    offset = pos[:, None, :] - boxes[None, :, 0:3]  # (N, K, 3)
    half = boxes[:, 3:6] / 2.0
    cos, sin = torch.cos(boxes[:, 6]), torch.sin(boxes[:, 6])
    local_x = offset[..., 0] * cos[None, :] + offset[..., 1] * sin[None, :]
    local_y = offset[..., 1] * cos[None, :] - offset[..., 0] * sin[None, :]
    inside_x = local_x.abs() <= half[None, :, 0]
    inside_y = local_y.abs() <= half[None, :, 1]
    inside_z = offset[..., 2].abs() <= half[None, :, 2]
    return inside_x & inside_y & inside_z


def count_points_in_boxes(
    pos: Tensor,
    boxes: Tensor,
    *,
    pos_batch: OptTensor = None,
    box_batch: OptTensor = None,
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
        ```pycon
        >>> pos = torch.tensor([[0.0, 0.0, 0.0], [5.0, 5.0, 5.0]])
        >>> boxes = torch.tensor([[0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0]])
        >>> count_points_in_boxes(pos, boxes).tolist()
        [1]

        ```
    """
    if (pos_batch is None) != (box_batch is None):
        raise ValueError("`pos_batch` and `box_batch` must be given together; got exactly one of them.")

    if pos_batch is None or box_batch is None:
        pos_batch = pos.new_zeros(pos.shape[0], dtype=torch.long)
        box_batch = boxes.new_zeros(boxes.shape[0], dtype=torch.long)

    counts = boxes.new_zeros(boxes.shape[0], dtype=torch.long)
    for scene in box_batch.unique().tolist():
        scene_pos = pos[pos_batch == scene]
        scene_boxes = (box_batch == scene).nonzero(as_tuple=False).squeeze(-1)
        # Boxes go through in chunks so the (N, K) point-in-box test stays within 2^24 pairs.
        chunk_size = max(1, (1 << 24) // max(scene_pos.shape[0], 1))
        for index in scene_boxes.split(chunk_size):
            counts[index] = points_in_boxes(scene_pos, boxes[index]).sum(dim=0)
    return counts


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


def projected_ignore_mask(
    boxes: Tensor,
    calib: Tensor,
    image_shape: Tensor,
    *,
    min_height: float = 25.0,
) -> Tensor:
    r"""Flag boxes whose image projection is shorter than `min_height` pixels (the KITTI difficulty rule).

    Each box's 8 corners are projected through the $(3, 4)$ homogeneous LiDAR-to-image matrix (rows $0$
    and $1$ divided by the perspective depth of row $2$), the vertical pixel coordinates are clipped to
    the image rows $[0, \text{height} - 1]$, and a box is flagged when its clipped pixel height is
    strictly below `min_height`. Only the vertical extent is used; the width entry of `image_shape` keeps
    the dataset's $(\text{height}, \text{width})$ contract. The KITTI protocol excludes such predictions
    from scoring (the prediction-side `ignore_mask` of `box_matches`), with `min_height` at
    $40$ / $25$ / $25$ px for the easy / moderate / hard difficulties. For KITTI, compose the calib as
    $P_2 \cdot [R_0 T_\text{velo}; 0\ 0\ 0\ 1]$ with the third row taken from $R_0 T_\text{velo}$, so the
    perspective divide is by the rectified depth.

    `calib` and `image_shape` broadcast on a leading box dimension: pass a single $(3, 4)$ / $(2,)$
    frame for all boxes, or per-box $(N, 3, 4)$ / $(N, 2)$ rows (e.g. a stacked per-frame calib indexed
    by the boxes' scene index) to score a multi-frame batch in one call.

    Args:
        boxes: Boxes $(c_x, c_y, c_z, d_x, d_y, d_z, \theta)$ with full extents, shape $(N, 7)$.
        calib: Homogeneous projection from LiDAR coordinates to image pixels, shape $(3, 4)$ or per-box
            $(N, 3, 4)$.
        image_shape: Image $(\text{height}, \text{width})$ in pixels, shape $(2,)$ or per-box $(N, 2)$.
        min_height: Pixel height below which a box is flagged.

    Returns:
        Boolean ignore mask, shape $(N,)$.

    Shape:
        - boxes: $(N, 7)$
        - calib: $(3, 4)$ or $(N, 3, 4)$
        - image_shape: $(2,)$ or $(N, 2)$
        - output: $(N,)$

    Example:
        ```pycon
        >>> calib = torch.tensor([[50.0, -100.0, 0.0, 0.0], [50.0, 0.0, -100.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
        >>> boxes = torch.tensor([[10.0, 0.0, 0.0, 2.0, 2.0, 1.0, 0.0]])
        >>> projected_ignore_mask(boxes, calib, torch.tensor([100, 200]))
        tensor([True])
        >>> boxes = torch.tensor([[10.0, 0.0, 0.0, 2.0, 2.0, 1.0, 0.0], [2.0, 0.0, 0.0, 2.0, 2.0, 1.0, 0.0]])
        >>> projected_ignore_mask(boxes, calib.expand(2, 3, 4), torch.tensor([[100, 200], [100, 200]]))
        tensor([ True, False])

        ```
    """
    corners = box_corners(boxes)
    hom = torch.cat([corners, corners.new_ones(corners.shape[:-1] + (1,))], dim=-1)
    projected = hom @ calib.transpose(-1, -2)
    y = projected[..., 1] / projected[..., 2]
    max_row = (image_shape[..., 0].to(y.dtype) - 1)[..., None]
    y = torch.minimum(y.clamp(min=0.0), max_row)
    return y.amax(dim=-1) - y.amin(dim=-1) < min_height
