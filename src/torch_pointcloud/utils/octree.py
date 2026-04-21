# Imports annotations to support string based types (Octree, Points, etc.) in case ocnn is not installed.
from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Optional, overload

import torch
from torch import Tensor

from torch_pointcloud.utils.imports import optional_import
from torch_pointcloud.utils.types import OptTensor

if TYPE_CHECKING:
    import ocnn
    from ocnn.nn.octree_interp import (
        octree_linear_pts,
        octree_nearest_pts,
        octree_nearest_upsample,
    )
    from ocnn.octree import Octree, Points


ocnn, _ = optional_import("ocnn")
octree_linear_pts, _ = optional_import("ocnn.nn.octree_interp", "octree_linear_pts")
octree_linear_upsample, _ = optional_import("ocnn.nn.octree_interp", "octree_linear_upsample")
octree_nearest_pts, _ = optional_import("ocnn.nn.octree_interp", "octree_nearest_pts")
octree_nearest_upsample, _ = optional_import("ocnn.nn.octree_interp", "octree_nearest_upsample")
Octree, _ = optional_import("ocnn.octree", "Octree")
Points, _ = optional_import("ocnn.octree", "Points")


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
        x: The octree feature to interpolate.
        octree: The octree structure.
        depth: The depth of the octree.
        pts: The points to interpolate at, in the format $(x, y, z, batch)$.
        method: The method to use for interpolation.
        nempty: Whether to allow empty points.
        bound_check: Whether to check if the points are within the bounds of the octree.
        rescale_pts: Whether to rescale the points from [-1, 1] to [0, 2^depth].

    Returns:
        The interpolated features.
    """
    if method not in ["linear", "nearest"]:
        raise ValueError(f"Invalid method. Expected `method` to be one of `linear` or `nearest`, but got {method}.")

    if rescale_pts:
        scale_factor = 2 ** (depth - 1)
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


def octree_grid(
    x: Optional[Tensor],
    pos: Tensor,
    depth: int,
    full_depth: int = 2,
    batch: Optional[Tensor] = None,
    normal: Optional[Tensor] = None,
) -> Octree:
    r"""Build an octree from a point cloud.

    Note:
        This function is a utility wrapper around the `ocnn.octree.Octree` constructor.

    Note:
        The batch tensor is optional. If not provided, it is assumed that the points are from a single batch.

    Args:
        x: The features of the points.
        pos: The positions of the points.
        batch: The batch of the points.
        normal: The normal of the points.
        depth: The depth of the octree.
        full_depth: The full depth of the octree.
        scale_factor: The scale factor of the points.

    Returns:
        The octree.

    Examples:
        >>> x = torch.randn(100, 3)
        >>> pos = torch.randn(100, 3)
        >>> batch = torch.cat([torch.zeros(50), torch.ones(50)], dim=0)
        >>> normal = torch.randn(100, 3)
        >>> depth = 12
        >>> full_depth = 2
        >>> octree = octree_grid(
        ...     x,
        ...     pos,
        ...     batch=batch,
        ...     normal=normal,
        ...     depth=depth,
        ...     full_depth=full_depth,
        ... )
    """
    device = pos.device
    if batch is None:
        batch = torch.zeros(pos.size(0), dtype=torch.long, device=device)

    batch_size = batch.max().item() + 1

    point = Points(
        points=pos,
        normal=normal,
        features=x,
        batch_id=batch.unsqueeze(-1),
        batch_size=batch_size,
    )

    octree = ocnn.octree.Octree(
        depth=depth,
        full_depth=full_depth,
        batch_size=batch_size,
        device=device,
    )
    octree.build_octree(point)
    octree.construct_all_neigh()

    return octree
