"""Octree construction, interpolation, and upsampling helpers built on `ocnn`."""

# Imports annotations to support string based types (Octree, Points, etc.) in case ocnn is not installed.
from __future__ import annotations

from typing import TYPE_CHECKING, Literal, overload

import torch
from torch import Tensor

from torch_pointcloud.utils.imports import _OCNN_GITHUB_URL, optional_import
from torch_pointcloud.utils.types import OptTensor

if TYPE_CHECKING:
    from ocnn.nn.octree_interp import (
        octree_linear_pts,
        octree_nearest_pts,
        octree_nearest_upsample,
    )
    from ocnn.octree import Octree, Points


octree_linear_pts, _ = optional_import("ocnn.nn.octree_interp", "octree_linear_pts", url=_OCNN_GITHUB_URL)
octree_linear_upsample, _ = optional_import("ocnn.nn.octree_interp", "octree_linear_upsample", url=_OCNN_GITHUB_URL)
octree_nearest_pts, _ = optional_import("ocnn.nn.octree_interp", "octree_nearest_pts", url=_OCNN_GITHUB_URL)
octree_nearest_upsample, _ = optional_import("ocnn.nn.octree_interp", "octree_nearest_upsample", url=_OCNN_GITHUB_URL)
Octree, _ = optional_import("ocnn.octree", "Octree", url=_OCNN_GITHUB_URL)
Points, _ = optional_import("ocnn.octree", "Points", url=_OCNN_GITHUB_URL)


@overload
def build_octree(
    pos: Tensor,
    normal: OptTensor = None,
    features: OptTensor = None,
    batch: OptTensor = None,
    labels: OptTensor = None,
    depth: int = 11,
    full_depth: int = 2,
    batch_size: int = 1,
    *,
    return_points: Literal[False] = False,
) -> Octree: ...


@overload
def build_octree(
    pos: Tensor,
    normal: OptTensor = None,
    features: OptTensor = None,
    batch: OptTensor = None,
    labels: OptTensor = None,
    depth: int = 11,
    full_depth: int = 2,
    batch_size: int = 1,
    *,
    return_points: Literal[True],
) -> tuple[Octree, Points]: ...


def build_octree(
    pos: Tensor,
    normal: OptTensor = None,
    features: OptTensor = None,
    batch: OptTensor = None,
    labels: OptTensor = None,
    depth: int = 11,
    full_depth: int = 2,
    batch_size: int = 1,
    *,
    return_points: bool = False,
) -> Octree | tuple[Octree, Points]:
    r"""Build an `ocnn.octree.Octree` from a packed point cloud.

    Wraps `ocnn.octree.Points` construction and `Octree.build_octree`. Coordinates are expected in
    the `ocnn` convention, i.e. normalized to $[-1, 1]$.

    Args:
        pos: Point coordinates of shape $(N, 3)$, normalized to $[-1, 1]$.
        normal: Optional per-point normals of shape $(N, 3)$.
        features: Optional per-point features of shape $(N, C)$.
        batch: Optional per-point batch indices of shape $(N,)$; `None` for a single sample.
        labels: Optional per-point labels of shape $(N,)$.
        depth: Depth of the octree.
        full_depth: Depth up to which all octree nodes are kept, empty or not.
        batch_size: Number of samples in the batch.
        return_points: If `True`, also return the intermediate `ocnn.octree.Points` object.

    Returns:
        The octree, or the tuple `(octree, points)` when `return_points` is `True`.

    Example:
        ```pycon
        >>> import torch
        >>> from torch_pointcloud.utils.octree import build_octree
        >>> pos = torch.rand(100, 3) * 2 - 1
        >>> octree = build_octree(pos, depth=5, full_depth=2)  # doctest: +SKIP

        ```
    """
    points = Points(
        points=pos,
        normals=normal,
        features=features,
        labels=labels,
        batch_id=batch,
        batch_size=batch_size,
    )

    octree = Octree(
        depth=depth,
        full_depth=full_depth,
        batch_size=batch_size,
    )
    octree.build_octree(points)

    if return_points:
        return octree, points
    return octree


def octree_interpolate(
    x: Tensor,
    octree: Octree,
    depth: int,
    pts: Tensor,
    method: Literal["linear", "nearest"] = "linear",
    nempty: bool = False,
    bound_check: bool = False,
    rescale_pts: bool = True,
) -> Tensor:
    r"""Functional interface for `ocnn.nn.OctreeInterp`, for ease of use.
    This function will interpolate the points with an octree feature.

    Note:
        In comparison to the original `ocnn.nn.OctreeInterp`,
        this function signature expects (x, pos, octree, depth) instead of (x, octree, depth, pos),
        to be consistent with other functions and the design philosophy of this library.

    Args:
        x: The octree features to interpolate, of shape $(M, C)$ with $M$ the number of octree
            nodes at `depth`.
        octree: The octree structure.
        depth: The depth of the octree.
        pts: The points to interpolate at, of shape $(N, 4)$ in the format $(x, y, z, \text{batch})$.
            The input is not modified.
        method: The method to use for interpolation, `"linear"` or `"nearest"`.
        nempty: Whether the features `x` only cover non-empty octree nodes.
        bound_check: Whether to check if the points are within the bounds of the octree.
        rescale_pts: Whether to rescale the point coordinates from $[-1, 1]$ to $[0, 2^\text{depth}]$.

    Returns:
        The interpolated features of shape $(N, C)$.
    """
    if method not in ["linear", "nearest"]:
        raise ValueError(f"Invalid method. Expected `method` to be one of `linear` or `nearest`, but got {method}.")

    if rescale_pts:
        scale_factor = 2 ** (depth - 1)
        pts = pts.clone()
        pts[:, :3] = (pts[:, :3] + 1.0) * scale_factor

    fn = octree_linear_pts if method == "linear" else octree_nearest_pts
    return fn(x, octree, depth, pts, nempty=nempty, bound_check=bound_check)


def octree_upsample(
    x: Tensor,
    octree: Octree,
    src_depth: int,
    dst_depth: int,
    method: Literal["linear", "nearest"] = "linear",
    nempty: bool = False,
) -> Tensor:
    r"""Upsample octree features from `src_depth` to the finer `dst_depth`.

    Interpolates the features of the octree nodes at `src_depth` at the node centers of
    `dst_depth`. When `dst_depth == src_depth` the features are returned unchanged; a single-level
    nearest upsample uses the dedicated `ocnn` kernel.

    Args:
        x: The octree features at `src_depth`, of shape $(M_\text{src}, C)$.
        octree: The octree structure.
        src_depth: The depth the features live at.
        dst_depth: The target depth; must be greater than or equal to `src_depth`.
        method: The method to use for interpolation, `"linear"` or `"nearest"`.
        nempty: Whether the features `x` only cover non-empty octree nodes.

    Returns:
        The upsampled features of shape $(M_\text{dst}, C)$.
    """
    if method not in ["linear", "nearest"]:
        raise ValueError(f"Invalid method. Expected `method` to be one of `linear` or `nearest`, but got {method}.")

    if src_depth == dst_depth:
        return x

    if dst_depth < src_depth:
        raise ValueError(
            f"Invalid destination depth. Expected `dst_depth` to be greater than `src_depth`, "
            f"but got {dst_depth} and {src_depth} respectively."
        )

    if dst_depth == src_depth + 1 and method == "nearest":
        return octree_nearest_upsample(x, octree, src_depth, nempty)

    xyzb = octree.xyzb(dst_depth, nempty)
    pts = torch.stack(xyzb, dim=1).float()
    pts[:, :3] = (pts[:, :3] + 0.5) * (2 ** (src_depth - dst_depth))

    fn = octree_linear_pts if method == "linear" else octree_nearest_pts
    return fn(x, octree, src_depth, pts, nempty)
