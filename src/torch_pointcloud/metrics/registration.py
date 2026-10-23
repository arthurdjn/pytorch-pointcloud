r"""Rigid registration metrics: rotation and translation errors, the 3DMatch RMSE and the correspondence inlier ratio.

Every function scores a batch of pairs at once and returns one value per pair, so a split accumulates as a list of
small tensors: the recall of a benchmark is the fraction of pairs under its thresholds, e.g.
`(registration_rmse(...) < 0.2).float().mean()` on 3DMatch or
`((rotation_error(...) < 5) & (translation_error(...) < 2)).float().mean()` on KITTI.
"""

from typing import Optional

import torch
from torch import Tensor
from torch_geometric.utils import scatter

from torch_pointcloud.ops.geometry import transform_points
from torch_pointcloud.ops.registration import rotation_angle
from torch_pointcloud.utils.types import OptTensor


def rotation_error(transform: Tensor, target: Tensor) -> Tensor:
    r"""Isotropic relative rotation error (RRE) between estimated and ground-truth transforms, in degrees.

    $\text{RRE} = \arccos\left(\frac{\operatorname{tr}(R^\top \bar{R}) - 1}{2}\right)$.

    Args:
        transform: Estimated transforms, shape $(\ldots, 4, 4)$.
        target: Ground-truth transforms, same shape.

    Returns:
        The errors in degrees, shape $(\ldots)$.

    Example:
        ```pycon
        >>> import torch
        >>> from torch_pointcloud.metrics import rotation_error
        >>> rotation_error(torch.eye(4), torch.eye(4)).item()
        0.0

        ```
    """
    return rotation_angle(transform[..., :3, :3], target[..., :3, :3])


def translation_error(transform: Tensor, target: Tensor) -> Tensor:
    r"""Relative translation error (RTE) between estimated and ground-truth transforms.

    $\text{RTE} = \lVert t - \bar{t} \rVert_2$, in the unit of the point coordinates.

    Args:
        transform: Estimated transforms, shape $(\ldots, 4, 4)$.
        target: Ground-truth transforms, same shape.

    Returns:
        The errors, shape $(\ldots)$.

    Example:
        ```pycon
        >>> import torch
        >>> from torch_pointcloud.metrics import translation_error
        >>> estimate = torch.eye(4)
        >>> estimate[0, 3] = 0.3
        >>> translation_error(estimate, torch.eye(4)).item()
        0.30000001192092896

        ```
    """
    return torch.linalg.norm(transform[..., :3, 3] - target[..., :3, 3], dim=-1)


def registration_rmse(transform: Tensor, target: Tensor, information: Tensor) -> Tensor:
    r"""Root mean square alignment error of the 3DMatch benchmark.

    The error of the residual transform $\bar{T}^{-1} T$ is read through the information matrix $\Omega$ the
    benchmark ships with every ground-truth pair: with $\xi = (t, q_x, q_y, q_z)$ its translation and the vector
    part of its quaternion, $\text{RMSE}^2 = \xi^\top \Omega \xi / \Omega_{00}$, the mean squared displacement of
    the overlapping points. A pair is registered when the RMSE is under $0.2$ m.

    Args:
        transform: Estimated transforms, shape $(\ldots, 4, 4)$.
        target: Ground-truth transforms, same shape.
        information: Information matrices of the ground-truth pairs, shape $(\ldots, 6, 6)$.

    Returns:
        The RMSE of each pair, shape $(\ldots)$.

    Example:
        ```pycon
        >>> import torch
        >>> from torch_pointcloud.metrics import registration_rmse
        >>> estimate = torch.eye(4)
        >>> estimate[0, 3] = 0.1
        >>> registration_rmse(estimate, torch.eye(4), torch.eye(6)).item()
        0.10000000149011612

        ```
    """
    residual = torch.linalg.inv(target) @ transform
    rotation = residual[..., :3, :3]
    translation = residual[..., :3, 3]
    # Vector part of the unit quaternion of the residual rotation, through its trace.
    trace = rotation[..., 0, 0] + rotation[..., 1, 1] + rotation[..., 2, 2]
    w = 0.5 * torch.sqrt((1.0 + trace).clamp(min=1e-12))
    scale = 0.25 / w
    quaternion = torch.stack(
        [
            (rotation[..., 2, 1] - rotation[..., 1, 2]) * scale,
            (rotation[..., 0, 2] - rotation[..., 2, 0]) * scale,
            (rotation[..., 1, 0] - rotation[..., 0, 1]) * scale,
        ],
        dim=-1,
    )
    error = torch.cat([translation, quaternion], dim=-1).unsqueeze(-1)  # (..., 6, 1)
    squared = (error.transpose(-1, -2) @ information @ error).squeeze(-1).squeeze(-1) / information[..., 0, 0]
    return torch.sqrt(squared.clamp(min=0.0))


def inlier_ratio(
    src_points: Tensor,
    ref_points: Tensor,
    target: Tensor,
    radius: float,
    batch: OptTensor = None,
    batch_size: Optional[int] = None,
) -> Tensor:
    r"""Fraction of correspondences whose points lie within `radius` once the source is moved by the ground truth.

    The feature matching recall (FMR) of a benchmark is the fraction of pairs whose inlier ratio exceeds a
    threshold ($0.05$ on 3DMatch).

    Args:
        src_points: Source point of each correspondence, shape $(K, 3)$.
        ref_points: Reference point of each correspondence, shape $(K, 3)$.
        target: Ground-truth transform of each pair, shape $(4, 4)$ or $(B, 4, 4)$.
        radius: Distance under which a correspondence is an inlier.
        batch: Per-correspondence pair index, shape $(K,)$; `None` scores a single pair.
        batch_size: Number of pairs $B$; `None` takes it from a batched `target`, else from `batch`.

    Returns:
        The inlier ratio of each pair, shape $(B,)$, `nan` for a pair without correspondences.

    Example:
        ```pycon
        >>> import torch
        >>> from torch_pointcloud.metrics import inlier_ratio
        >>> src = torch.tensor([[0.0, 0, 0], [1.0, 0, 0]])
        >>> ref = torch.tensor([[0.0, 0, 0], [1.5, 0, 0]])
        >>> inlier_ratio(src, ref, torch.eye(4), radius=0.1).tolist()
        [0.5]

        ```
    """
    if batch is None:
        batch = torch.zeros(src_points.size(0), dtype=torch.long, device=src_points.device)
    if batch_size is None:
        batch_size = target.size(0) if target.ndim == 3 else (int(batch.max()) + 1 if batch.numel() else 1)
    if target.ndim == 2:
        target = target.unsqueeze(0).expand(batch_size, 4, 4)

    aligned = transform_points(src_points.unsqueeze(1), target[batch]).squeeze(1)
    inliers = (torch.linalg.norm(ref_points - aligned, dim=-1) < radius).float()
    counts = torch.bincount(batch, minlength=batch_size).float()
    ratios = scatter(inliers, batch, dim=0, dim_size=batch_size, reduce="sum") / counts
    return ratios


def alignment_error(
    src_points: Tensor,
    transform: Tensor,
    target: Tensor,
    batch: OptTensor = None,
    batch_size: Optional[int] = None,
) -> Tensor:
    r"""Mean distance between the source points moved by the estimated transform and by the ground truth.

    The alignment error of the ModelNet registration benchmark (its "RMSE" column), in the unit of the coordinates.

    Args:
        src_points: Source points, shape $(N, 3)$.
        transform: Estimated transform of each pair, shape $(4, 4)$ or $(B, 4, 4)$.
        target: Ground-truth transform of each pair, same shape.
        batch: Per-point pair index, shape $(N,)$; `None` scores a single pair.
        batch_size: Number of pairs $B$; `None` takes it from a batched `target`, else from `batch`.

    Returns:
        The mean displacement of each pair, shape $(B,)$.

    Example:
        ```pycon
        >>> import torch
        >>> from torch_pointcloud.metrics import alignment_error
        >>> estimate = torch.eye(4)
        >>> estimate[0, 3] = 0.2
        >>> round(alignment_error(torch.rand(10, 3), estimate, torch.eye(4)).item(), 4)
        0.2

        ```
    """
    if batch is None:
        batch = torch.zeros(src_points.size(0), dtype=torch.long, device=src_points.device)
    if batch_size is None:
        batch_size = target.size(0) if target.ndim == 3 else (int(batch.max()) + 1 if batch.numel() else 1)
    if transform.ndim == 2:
        transform = transform.unsqueeze(0).expand(batch_size, 4, 4)
    if target.ndim == 2:
        target = target.unsqueeze(0).expand(batch_size, 4, 4)

    estimated = transform_points(src_points.unsqueeze(1), transform[batch]).squeeze(1)
    expected = transform_points(src_points.unsqueeze(1), target[batch]).squeeze(1)
    distances = torch.linalg.norm(estimated - expected, dim=-1)
    return scatter(distances, batch, dim=0, dim_size=batch_size, reduce="mean")
