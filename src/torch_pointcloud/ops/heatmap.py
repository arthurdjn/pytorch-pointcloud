r"""Gaussian center-heatmap targets for center-based 3D detection heads.

Center-based detectors (VoxelNeXt, VoxelMamba, LION) supervise a per-class BEV heatmap whose peaks
mark object centers, alongside per-object regression targets read back at those peak cells. A ground
truth box is splatted as a 2D Gaussian whose radius is chosen so that any box overlapping the true
box by at least `min_overlap` still lands inside the positive region.

References:
    :arxiv: [Objects as Points](https://arxiv.org/abs/1904.07850) (the min-overlap Gaussian radius),
    :arxiv: [Center-based 3D Object Detection and Tracking](https://arxiv.org/abs/2006.11275) (Yin et al., 2021) (the BEV
    center-heatmap formulation used by the 3D heads).
"""

from typing import Sequence, Tuple, Union

import torch
from torch import Tensor


def gaussian_radius(height: Tensor, width: Tensor, min_overlap: float = 0.5) -> Tensor:
    r"""Per-box Gaussian splat radius from the standard three min-overlap cases.

    Approximates the largest radius $r$ such that a box overlapping the ground truth by at least
    `min_overlap` (IoU) still has its center inside the positive Gaussian region, as the minimum over
    the inscribed, enclosing, and shifted-box cases. The quadratic roots keep the un-normalized
    $r_2, r_3$ of the original Objects as Points formulation (no $1 / (2a)$ factor), so the radii match
    the published detectors' training targets rather than the exact closed-form solutions. The formula
    is symmetric in `height` and `width`.

    Args:
        height: Box heights ($y$ extent) in feature-map cells, shape $(N,)$.
        width: Box widths ($x$ extent) in feature-map cells, shape $(N,)$.
        min_overlap: Minimum IoU a candidate box must keep with the ground truth.

    Returns:
        Per-box radius in feature-map cells, shape $(N,)$ (float; caller rounds and clamps).

    Shape:
        - height: $(N,)$
        - width: $(N,)$
        - output: $(N,)$

    Example:
        ```pycon
        >>> import torch
        >>> r = gaussian_radius(torch.tensor([4.0, 8.0]), torch.tensor([2.0, 4.0]))
        >>> bool(r[1] > r[0])
        True

        ```
    """
    a1 = 1.0
    b1 = height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = (b1**2 - 4 * a1 * c1).sqrt()
    r1 = (b1 + sq1) / 2

    a2 = 4.0
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = (b2**2 - 4 * a2 * c2).sqrt()
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = (b3**2 - 4 * a3 * c3).sqrt()
    r3 = (b3 + sq3) / 2

    return torch.min(torch.min(r1, r2), r3)


def _gaussian_2d(diameter: int, sigma: float, device: torch.device) -> Tensor:
    r"""Isotropic 2D Gaussian on a $(\text{diameter}, \text{diameter})$ grid, peak $1$ at the center."""
    radius = (diameter - 1) // 2
    offsets = torch.arange(-radius, radius + 1, dtype=torch.float64, device=device)
    y = offsets[:, None]
    x = offsets[None, :]
    h = torch.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < torch.finfo(torch.float64).eps * h.max()] = 0
    return h


def draw_gaussian_to_heatmap(
    heatmap: Tensor,
    center: Tensor,
    radius: Union[int, Tensor],
    k: float = 1.0,
) -> Tensor:
    r"""Splat a 2D Gaussian at an integer center into a heatmap, max-combining in place.

    The Gaussian (peak $k$ at `center`) is clipped to the heatmap bounds and combined with an
    element-wise maximum, so overlapping objects keep the stronger response. `heatmap` is modified
    in place and also returned. Pass a single channel slice (`heatmap[class_id]`) to target one class.

    Args:
        heatmap: Target map to draw into, shape $(H, W)$. Modified in place.
        center: Center cell $(x, y)$ in feature-map coordinates, shape $(2,)$; truncated to int.
        radius: Gaussian radius in cells (scalar); the splat spans $2 \cdot \text{radius} + 1$ cells.
        k: Peak value at the center.

    Returns:
        The same `heatmap` tensor, modified in place.

    Shape:
        - heatmap: $(H, W)$
        - center: $(2,)$
        - output: $(H, W)$

    Example:
        ```pycon
        >>> import torch
        >>> hm = torch.zeros(10, 10)
        >>> _ = draw_gaussian_to_heatmap(hm, torch.tensor([5.0, 4.0]), radius=2)
        >>> float(hm[4, 5])
        1.0

        ```
    """
    radius = int(radius)
    diameter = 2 * radius + 1
    sigma = diameter / 6
    gaussian = _gaussian_2d(diameter, sigma, heatmap.device)

    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape[0], heatmap.shape[1]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top : y + bottom, x - left : x + right]
    masked_gaussian = gaussian[radius - top : radius + bottom, radius - left : radius + right].to(heatmap)

    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        torch.max(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def draw_heatmap_targets(
    boxes: Tensor,
    labels: Tensor,
    num_classes: int,
    feature_map_size: Tuple[int, int],
    voxel_size: Sequence[float],
    point_cloud_range: Sequence[float],
    feature_map_stride: int,
    *,
    num_max_objs: int = 500,
    gaussian_overlap: float = 0.1,
    min_radius: int = 2,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    r"""Assign center-heatmap and per-object regression targets for one scene.

    Projects each ground truth box center to the BEV feature map, splats a per-class Gaussian, and
    records the regression target at that peak cell. The regression code is the sub-cell center
    offset, absolute $z$, log extents, and $(\cos\theta, \sin\theta)$, followed by any extra box
    columns (e.g. velocity): $8 + (D - 7)$ channels for a $(M, D)$ box tensor. Extents are clamped
    to $10^{-5}$ before the log so a degenerate box does not produce a non-finite target.

    Args:
        boxes: Ground truth boxes $(c_x, c_y, c_z, d_x, d_y, d_z, \theta, \ldots)$, shape $(M, D)$, $D \ge 7$.
        labels: Zero-based class ids, shape $(M,)$.
        num_classes: Number of heatmap channels.
        feature_map_size: BEV feature-map size as $(W, H)$ (x then y).
        voxel_size: Voxel size $(v_x, v_y, v_z)$ in metric units.
        point_cloud_range: Range $(x_\min, y_\min, z_\min, x_\max, y_\max, z_\max)$.
        feature_map_stride: Stride from voxel grid to feature map.
        num_max_objs: Capacity of the per-object target buffers.
        gaussian_overlap: Min-overlap passed to `gaussian_radius`.
        min_radius: Lower clamp on the integer splat radius.

    Returns:
        Tuple `(heatmap, reg_targets, inds, mask)`:

        - `heatmap`: per-class Gaussian map, shape $(\text{num\_classes}, H, W)$.
        - `reg_targets`: regression targets, shape $(\text{num\_max\_objs}, 8 + (D - 7))$.
        - `inds`: flat peak-cell index $y \cdot W + x$ per object, shape $(\text{num\_max\_objs},)$ (long).
        - `mask`: $1$ for assigned objects, shape $(\text{num\_max\_objs},)$ (long).

    Shape:
        - boxes: $(M, D)$
        - labels: $(M,)$
        - heatmap: $(\text{num\_classes}, H, W)$

    Example:
        ```pycon
        >>> import torch
        >>> boxes = torch.tensor([[0.0, 0.0, -1.0, 4.0, 2.0, 1.5, 0.3]])
        >>> labels = torch.tensor([0])
        >>> hm, reg, inds, mask = draw_heatmap_targets(
        ...     boxes, labels, num_classes=1, feature_map_size=(16, 16),
        ...     voxel_size=[0.5, 0.5, 0.5], point_cloud_range=[-4.0, -4.0, -2.0, 4.0, 4.0, 2.0],
        ...     feature_map_stride=1,
        ... )
        >>> float(hm.max()), int(mask.sum())
        (1.0, 1)

        ```
    """
    width, height = feature_map_size
    code_size = boxes.shape[-1] + 1
    heatmap = boxes.new_zeros(num_classes, height, width)
    reg_targets = boxes.new_zeros(num_max_objs, code_size)
    inds = boxes.new_zeros(num_max_objs, dtype=torch.long)
    mask = boxes.new_zeros(num_max_objs, dtype=torch.long)

    boxes, labels = boxes[:num_max_objs], labels[:num_max_objs]
    num_objs = boxes.shape[0]
    if num_objs == 0:
        return heatmap, reg_targets, inds, mask

    # Project the centers to the feature map and solve the splat radius of every box at once.
    x, y, z = boxes[:, 0], boxes[:, 1], boxes[:, 2]
    pos_x = (x - point_cloud_range[0]) / voxel_size[0] / feature_map_stride
    pos_y = (y - point_cloud_range[1]) / voxel_size[1] / feature_map_stride
    pos_x = torch.clamp(pos_x, min=0, max=width - 0.5)
    pos_y = torch.clamp(pos_y, min=0, max=height - 0.5)
    center = torch.stack([pos_x, pos_y], dim=-1)
    center_int = center.int()
    center_int_float = center_int.float()

    dx = boxes[:, 3] / voxel_size[0] / feature_map_stride
    dy = boxes[:, 4] / voxel_size[1] / feature_map_stride
    radius = gaussian_radius(dy, dx, min_overlap=gaussian_overlap)
    radius = torch.clamp_min(radius.int(), min_radius)

    # Degenerate boxes and centers outside the map (impossible after the clamp, kept for parity) get no target.
    in_map_x = (center_int[:, 0] >= 0) & (center_int[:, 0] <= width)
    in_map_y = (center_int[:, 1] >= 0) & (center_int[:, 1] <= height)
    valid = (dx > 0) & (dy > 0) & in_map_x & in_map_y

    # Per-object targets: the peak cell, its flat index and the regression code.
    mask[:num_objs] = valid.long()
    inds[:num_objs] = torch.where(valid, (center_int[:, 1] * width + center_int[:, 0]).long(), inds[:num_objs])
    code = torch.cat(
        [
            center - center_int_float,
            z[:, None],
            boxes[:, 3:6].clamp_min(1e-5).log(),
            torch.cos(boxes[:, 6:7]),
            torch.sin(boxes[:, 6:7]),
            boxes[:, 7:],
        ],
        dim=1,
    )
    reg_targets[:num_objs] = torch.where(valid[:, None], code, reg_targets[:num_objs])

    # Splat every Gaussian at once: a (2 r_max + 1)^2 window per object, max-combined into the class map.
    _splat_gaussians(heatmap, labels[valid], center_int[valid], radius[valid])

    return heatmap, reg_targets, inds, mask


def _splat_gaussians(heatmap: Tensor, labels: Tensor, center_int: Tensor, radius: Tensor) -> None:
    r"""Max-combine one Gaussian per object into the class maps of `heatmap` $(C, H, W)$, in place.

    Every object gets the same $(2 r_\max + 1)^2$ window; the cells beyond its own radius (which
    `draw_gaussian_to_heatmap` would not draw) and the cells outside the map are left out, so the
    result equals drawing the objects one by one.
    """
    if labels.numel() == 0:
        return

    num_classes, height, width = heatmap.shape
    device = heatmap.device
    max_radius = int(radius.max())

    # The Gaussian of `draw_gaussian_to_heatmap`: float64 values on the window, tiny ones zeroed.
    offsets = torch.arange(-max_radius, max_radius + 1, dtype=torch.float64, device=device)
    oy, ox = offsets[:, None], offsets[None, :]
    diameter = 2 * radius.to(torch.float64) + 1
    sigma = diameter / 6
    gaussian = torch.exp(-(ox * ox + oy * oy) / (2 * sigma * sigma)[:, None, None])  # (K, D, D)
    gaussian[gaussian < torch.finfo(torch.float64).eps] = 0

    # Keep the cells of each object's own window that fall inside the map.
    in_window = (ox.abs() <= radius[:, None, None]) & (oy.abs() <= radius[:, None, None])
    xs = center_int[:, 0, None, None].long() + ox.long()
    ys = center_int[:, 1, None, None].long() + oy.long()
    in_map = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    inside = in_window & in_map

    flat = (labels[:, None, None] * height + ys) * width + xs
    heatmap.view(-1).scatter_reduce_(0, flat[inside], gaussian[inside].to(heatmap.dtype), reduce="amax")


def transpose_gather(feat: Tensor, ind: Tensor) -> Tensor:
    r"""Gather per-object channel vectors from a dense map at flat cell indices.

    Reads the $C$-dim vector at each flat cell index $y \cdot W + x$ (the `inds` produced by
    `draw_heatmap_targets`) out of a dense $(B, C, H, W)$ prediction map, e.g. to compare head outputs
    against per-object regression targets at the Gaussian peak cells.

    Args:
        feat: Dense prediction map, shape $(B, C, H, W)$.
        ind: Flat per-object cell indices, shape $(B, M)$ (long).

    Returns:
        Gathered per-object vectors, shape $(B, M, C)$.

    Shape:
        - feat: $(B, C, H, W)$
        - ind: $(B, M)$
        - output: $(B, M, C)$

    Example:
        ```pycon
        >>> import torch
        >>> feat = torch.arange(16.0).reshape(1, 1, 4, 4)
        >>> transpose_gather(feat, torch.tensor([[5, 10]]))
        tensor([[[ 5.],
                 [10.]]])

        ```
    """
    b, c = feat.shape[0], feat.shape[1]
    feat = feat.permute(0, 2, 3, 1).reshape(b, -1, c)
    return feat.gather(1, ind.unsqueeze(2).expand(-1, -1, c))
