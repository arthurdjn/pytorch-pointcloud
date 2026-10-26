r"""Rigid registration metrics: rotation and translation errors, the alignment errors of the source points, the
correspondence inlier ratio, and the feature matching and registration recalls built on them.

The per-pair functions score a batch of pairs at once and return one value per pair, so a split accumulates as a
list of small tensors that the recalls reduce to a `float`.
"""

import torch
from torch import Tensor
from torch_geometric.utils import scatter

from torch_pointcloud.ops.geometry import rotation_matrix_to_quaternion, transform_points
from torch_pointcloud.ops.registration import rotation_angle


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


def root_mean_square_alignment_error(transform: Tensor, target: Tensor, information: Tensor) -> Tensor:
    r"""Root mean square displacement of the overlapping points under the residual transform, read through the
    information matrix of the ground truth (the RMSE of the 3DMatch benchmark).

    The error of the residual transform $\bar{T}^{-1} T$ is read through the information matrix $\Omega$ the
    benchmark ships with every ground-truth pair: with $\xi = (t, q_x, q_y, q_z)$ its translation and the vector
    part of its quaternion, $\text{RMSE}^2 = \xi^\top \Omega \xi / \Omega_{00}$, the mean squared displacement of
    the overlapping points.

    Args:
        transform: Estimated transforms, shape $(\ldots, 4, 4)$.
        target: Ground-truth transforms, same shape.
        information: Information matrices of the ground-truth pairs, shape $(\ldots, 6, 6)$.

    Returns:
        The RMSE of each pair, shape $(\ldots)$.

    Example:
        ```pycon
        >>> import torch
        >>> from torch_pointcloud.metrics import root_mean_square_alignment_error
        >>> estimate = torch.eye(4)
        >>> estimate[0, 3] = 0.1
        >>> root_mean_square_alignment_error(estimate, torch.eye(4), torch.eye(6)).item()
        0.10000000149011612

        ```
    """
    residual = torch.linalg.inv(target) @ transform
    quaternion = rotation_matrix_to_quaternion(residual[..., :3, :3])
    error = torch.cat([residual[..., :3, 3], quaternion[..., 1:]], dim=-1).unsqueeze(-1)  # (..., 6, 1)
    squared = (error.transpose(-1, -2) @ information @ error).squeeze(-1).squeeze(-1) / information[..., 0, 0]
    return torch.sqrt(squared.clamp(min=0.0))


def inlier_ratio(
    src_points: Tensor,
    ref_points: Tensor,
    target: Tensor,
    radius: float,
    batch: Tensor,
) -> Tensor:
    r"""Fraction of correspondences whose points lie within `radius` once the source is moved by the ground truth.

    Args:
        src_points: Source point of each correspondence, shape $(K, 3)$.
        ref_points: Reference point of each correspondence, shape $(K, 3)$.
        target: Ground-truth transform of each pair, shape $(B, 4, 4)$.
        radius: Distance under which a correspondence is an inlier.
        batch: Per-correspondence pair index, shape $(K,)$.

    Returns:
        The inlier ratio of each pair, shape $(B,)$, `nan` for a pair without correspondences.

    Example:
        ```pycon
        >>> import torch
        >>> from torch_pointcloud.metrics import inlier_ratio
        >>> src = torch.tensor([[0.0, 0, 0], [1.0, 0, 0]])
        >>> ref = torch.tensor([[0.0, 0, 0], [1.5, 0, 0]])
        >>> inlier_ratio(src, ref, torch.eye(4).unsqueeze(0), radius=0.1, batch=torch.tensor([0, 0])).tolist()
        [0.5]

        ```
    """
    batch_size = target.size(0)
    aligned = transform_points(src_points.unsqueeze(1), target[batch]).squeeze(1)
    inliers = (torch.linalg.norm(ref_points - aligned, dim=-1) < radius).float()
    counts = torch.bincount(batch, minlength=batch_size).float()
    return scatter(inliers, batch, dim=0, dim_size=batch_size, reduce="sum") / counts


def alignment_error(
    src_points: Tensor,
    transform: Tensor,
    target: Tensor,
    batch: Tensor,
) -> Tensor:
    r"""Mean distance between the source points moved by the estimated transform and by the ground truth.

    The alignment error of the ModelNet registration benchmark (its "RMSE" column), in the unit of the coordinates.

    Args:
        src_points: Source points, shape $(N, 3)$.
        transform: Estimated transform of each pair, shape $(B, 4, 4)$.
        target: Ground-truth transform of each pair, shape $(B, 4, 4)$.
        batch: Per-point pair index, shape $(N,)$.

    Returns:
        The mean displacement of each pair, shape $(B,)$.

    Example:
        ```pycon
        >>> import torch
        >>> from torch_pointcloud.metrics import alignment_error
        >>> estimate = torch.eye(4)
        >>> estimate[0, 3] = 0.2
        >>> batch = torch.zeros(10, dtype=torch.long)
        >>> round(alignment_error(torch.rand(10, 3), estimate.unsqueeze(0), torch.eye(4).unsqueeze(0), batch).item(), 4)
        0.2

        ```
    """
    estimated = transform_points(src_points.unsqueeze(1), transform[batch]).squeeze(1)
    expected = transform_points(src_points.unsqueeze(1), target[batch]).squeeze(1)
    distances = torch.linalg.norm(estimated - expected, dim=-1)
    return scatter(distances, batch, dim=0, dim_size=target.size(0), reduce="mean")


def feature_matching_recall(inlier_ratios: Tensor, threshold: float) -> float:
    r"""Fraction of pairs whose inlier ratio reaches `threshold`.

    Args:
        inlier_ratios: Inlier ratio of each pair, shape $(P,)$, from `inlier_ratio`.
        threshold: Minimum inlier ratio of a matched pair.

    Returns:
        The feature matching recall (FMR), in $[0, 1]$.

    Example:
        ```pycon
        >>> import torch
        >>> from torch_pointcloud.metrics import feature_matching_recall
        >>> feature_matching_recall(torch.tensor([0.0, 0.04, 0.05, 0.7]), threshold=0.05)
        0.5

        ```
    """
    return float((inlier_ratios >= threshold).float().mean())


def registration_recall(errors: Tensor, threshold: float) -> float:
    r"""Fraction of pairs whose error is under `threshold`.

    Args:
        errors: Error of each pair, shape $(P,)$, such as `root_mean_square_alignment_error`.
        threshold: Error under which a pair is registered.

    Returns:
        The registration recall (RR), in $[0, 1]$.

    Example:
        ```pycon
        >>> import torch
        >>> from torch_pointcloud.metrics import registration_recall
        >>> registration_recall(torch.tensor([0.01, 0.1, 0.2, 0.5]), threshold=0.2)
        0.5

        ```
    """
    return float((errors < threshold).float().mean())
