r"""Rigid registration primitives: pairwise distances, the weighted Procrustes fit, and coarse-to-fine patch partitions.

A rigid transform is a $4 \times 4$ matrix $T = \begin{bmatrix} R & t \\ 0 & 1 \end{bmatrix}$ mapping the source cloud
into the reference frame, $q = R p + t$; `ops.geometry.transform_points` applies it.
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.utils import index_to_mask

from torch_pointcloud.ops.geometry import transform_points
from torch_pointcloud.utils.types import OptTensor


def pairwise_squared_distance(x: Tensor, y: Tensor, normalized: bool = False) -> Tensor:
    r"""Squared Euclidean distance between every row of `x` and every row of `y`.

    Expanded as $\lVert x \rVert^2 - 2 x \cdot y + \lVert y \rVert^2$ and clamped at $0$; for unit-norm rows
    (`normalized=True`) the squared norms are known and only the inner product is computed.

    Args:
        x: Points or features, shape $(\ldots, N, C)$.
        y: Points or features, shape $(\ldots, M, C)$.
        normalized: Whether every row of `x` and `y` has unit norm.

    Returns:
        The squared distances, shape $(\ldots, N, M)$.

    Example:
        ```pycon
        >>> import torch
        >>> from torch_pointcloud.ops.registration import pairwise_squared_distance
        >>> x = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        >>> pairwise_squared_distance(x, x)
        tensor([[0., 1.],
                [1., 0.]])

        ```
    """
    xy = torch.matmul(x, y.transpose(-1, -2))
    if normalized:
        sq_distances = 2.0 - 2.0 * xy
    else:
        x2 = torch.sum(x**2, dim=-1).unsqueeze(-1)
        y2 = torch.sum(y**2, dim=-1).unsqueeze(-2)
        sq_distances = x2 - 2 * xy + y2
    return sq_distances.clamp(min=0.0)


# inspired by: https://github.com/XuyangBai/PointDSC/blob/5ca8ce5e2a4f9d0cec4b8e4dd46ce7a1be2d8e52/models/common.py#L36
def weighted_procrustes(
    src_points: Tensor,
    ref_points: Tensor,
    weights: OptTensor = None,
    weight_threshold: float = 0.0,
    eps: float = 1e-5,
) -> Tensor:
    r"""Rigid transform aligning `src_points` onto `ref_points` by weighted SVD (the Kabsch solution).

    Weights below `weight_threshold` are zeroed, then normalized to sum to one per set. The rotation is the
    closest proper rotation to the weighted cross-covariance, with the reflection corrected through the sign of
    its determinant.

    Args:
        src_points: Source correspondences, shape $(N, 3)$ or $(B, N, 3)$.
        ref_points: Reference correspondences aligned to `src_points`, same shape.
        weights: Per-correspondence weights, shape $(N,)$ or $(B, N)$; `None` weights every pair equally.
        weight_threshold: Weights strictly below it are ignored.
        eps: Added to the weight normalizer.

    Returns:
        The transforms mapping the source onto the reference, shape $(4, 4)$ or $(B, 4, 4)$.

    Example:
        ```pycon
        >>> import torch
        >>> from torch_pointcloud.ops.registration import weighted_procrustes
        >>> src = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        >>> transform = weighted_procrustes(src, src + 1.0)
        >>> torch.allclose(transform[:3, 3], torch.ones(3))
        True

        ```
    """
    squeeze = src_points.ndim == 2
    if squeeze:
        src_points = src_points.unsqueeze(0)
        ref_points = ref_points.unsqueeze(0)
        if weights is not None:
            weights = weights.unsqueeze(0)

    if weights is None:
        weights = torch.ones_like(src_points[:, :, 0])
    weights = torch.where(weights < weight_threshold, torch.zeros_like(weights), weights)
    weights = weights / (weights.sum(dim=1, keepdim=True) + eps)
    weights = weights.unsqueeze(2)  # (B, N, 1)

    src_centroid = torch.sum(src_points * weights, dim=1, keepdim=True)  # (B, 1, 3)
    ref_centroid = torch.sum(ref_points * weights, dim=1, keepdim=True)  # (B, 1, 3)
    src_centered = src_points - src_centroid
    ref_centered = ref_points - ref_centroid

    covariance = src_centered.transpose(1, 2) @ (weights * ref_centered)  # (B, 3, 3)
    u, _, vh = torch.linalg.svd(covariance)
    v = vh.transpose(1, 2)
    ut = u.transpose(1, 2)
    sign = torch.eye(3, dtype=covariance.dtype, device=covariance.device).repeat(covariance.size(0), 1, 1)
    sign[:, 2, 2] = torch.sign(torch.det(v @ ut))
    rotation = v @ sign @ ut

    translation = ref_centroid.transpose(1, 2) - rotation @ src_centroid.transpose(1, 2)  # (B, 3, 1)
    transform = torch.eye(4, dtype=covariance.dtype, device=covariance.device).repeat(covariance.size(0), 1, 1)
    transform[:, :3, :3] = rotation
    transform[:, :3, 3] = translation.squeeze(2)
    return transform.squeeze(0) if squeeze else transform


@torch.no_grad()
def point_to_node_partition(
    points: Tensor,
    nodes: Tensor,
    point_limit: int,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    r"""Assign every point to its nearest node and gather each node's patch of at most `point_limit` points.

    A patch holds the points assigned to the node, nearest first; a node with more points than `point_limit`
    keeps the nearest ones, a node with fewer is padded with the index $N$ (a shadow row) and masked.

    Args:
        points: Fine points, shape $(N, 3)$.
        nodes: Coarse points, shape $(M, 3)$.
        point_limit: Patch size $K$.

    Returns:
        A tuple `(point_to_node, node_masks, node_knn_indices, node_knn_masks)`: the node of every point $(N,)$,
        whether each node received at least one point $(M,)$, the patch point indices $(M, K)$ padded with $N$, and
        the validity of every patch slot $(M, K)$.

    Example:
        ```pycon
        >>> import torch
        >>> from torch_pointcloud.ops.registration import point_to_node_partition
        >>> points = torch.tensor([[0.0, 0, 0], [0.1, 0, 0], [5.0, 0, 0]])
        >>> nodes = torch.tensor([[0.0, 0, 0], [5.0, 0, 0]])
        >>> point_to_node, node_masks, knn_indices, knn_masks = point_to_node_partition(points, nodes, 2)
        >>> point_to_node.tolist(), knn_indices.tolist(), knn_masks.tolist()
        ([0, 0, 1], [[0, 1], [2, 3]], [[True, True], [True, False]])

        ```
    """
    sq_dist_mat = pairwise_squared_distance(nodes, points)  # (M, N)
    point_to_node = sq_dist_mat.min(dim=0)[1]  # (N,)
    node_masks = index_to_mask(point_to_node, size=nodes.size(0))

    matching_masks = torch.zeros_like(sq_dist_mat, dtype=torch.bool)
    point_indices = torch.arange(points.size(0), device=points.device)
    matching_masks[point_to_node, point_indices] = True
    sq_dist_mat.masked_fill_(~matching_masks, 1e12)

    k = min(point_limit, points.size(0))
    node_knn_indices = sq_dist_mat.topk(k=k, dim=1, largest=False)[1]  # (M, k)
    node_knn_node_indices = point_to_node[node_knn_indices]  # (M, k)
    node_indices = torch.arange(nodes.size(0), device=nodes.device).unsqueeze(1).expand(-1, k)
    node_knn_masks = node_knn_node_indices == node_indices
    node_knn_indices.masked_fill_(~node_knn_masks, points.size(0))
    if k < point_limit:
        node_knn_indices = F.pad(node_knn_indices, (0, point_limit - k), value=points.size(0))
        node_knn_masks = F.pad(node_knn_masks, (0, point_limit - k), value=False)
    return point_to_node, node_masks, node_knn_indices, node_knn_masks


@torch.no_grad()
def node_correspondences(
    ref_nodes: Tensor,
    src_nodes: Tensor,
    ref_knn_points: Tensor,
    src_knn_points: Tensor,
    transform: Tensor,
    radius: float,
    ref_masks: OptTensor = None,
    src_masks: OptTensor = None,
    ref_knn_masks: OptTensor = None,
    src_knn_masks: OptTensor = None,
) -> Tuple[Tensor, Tensor]:
    r"""Ground-truth node (patch) correspondences and their overlap ratios under a known transform.

    Two patches correspond when, after moving the source into the reference frame, they share points closer than
    `radius`. The overlap of a pair is the mean of the fraction of reference patch points with a match and the
    fraction of source patch points with a match. Pairs whose enclosing spheres cannot intersect are skipped.

    Args:
        ref_nodes: Reference nodes, shape $(M, 3)$.
        src_nodes: Source nodes, shape $(N, 3)$.
        ref_knn_points: Reference patches, shape $(M, K, 3)$.
        src_knn_points: Source patches, shape $(N, K, 3)$.
        transform: Ground-truth transform of the source onto the reference, shape $(4, 4)$.
        radius: Distance under which two points match.
        ref_masks: Nodes of the reference with a non-empty patch, shape $(M,)$.
        src_masks: Nodes of the source with a non-empty patch, shape $(N,)$.
        ref_knn_masks: Validity of the reference patch slots, shape $(M, K)$.
        src_knn_masks: Validity of the source patch slots, shape $(N, K)$.

    Returns:
        A tuple `(corr_indices, corr_overlaps)`: the `(reference node, source node)` index pairs with a positive
        overlap, shape $(C, 2)$, and their overlap ratios, shape $(C,)$.

    Example:
        ```pycon
        >>> import torch
        >>> from torch_pointcloud.ops.registration import node_correspondences
        >>> nodes = torch.tensor([[0.0, 0, 0], [5.0, 0, 0]])
        >>> patches = nodes.unsqueeze(1) + torch.tensor([[[0.0, 0, 0], [0.1, 0, 0]]])
        >>> corr, overlaps = node_correspondences(nodes, nodes, patches, patches, torch.eye(4), radius=0.05)
        >>> corr.tolist(), overlaps.tolist()
        ([[0, 0], [1, 1]], [1.0, 1.0])

        ```
    """
    src_nodes = transform_points(src_nodes, transform)
    src_knn_points = transform_points(src_knn_points, transform)

    if ref_masks is None:
        ref_masks = torch.ones(ref_nodes.size(0), dtype=torch.bool, device=ref_nodes.device)
    if src_masks is None:
        src_masks = torch.ones(src_nodes.size(0), dtype=torch.bool, device=src_nodes.device)
    if ref_knn_masks is None:
        ref_knn_masks = torch.ones(ref_knn_points.shape[:2], dtype=torch.bool, device=ref_nodes.device)
    if src_knn_masks is None:
        src_knn_masks = torch.ones(src_knn_points.shape[:2], dtype=torch.bool, device=src_nodes.device)

    node_mask_mat = ref_masks.unsqueeze(1) & src_masks.unsqueeze(0)  # (M, N)

    ref_knn_dists = torch.linalg.norm(ref_knn_points - ref_nodes.unsqueeze(1), dim=-1)  # (M, K)
    ref_knn_dists.masked_fill_(~ref_knn_masks, 0.0)
    ref_max_dists = ref_knn_dists.max(1)[0]
    src_knn_dists = torch.linalg.norm(src_knn_points - src_nodes.unsqueeze(1), dim=-1)  # (N, K)
    src_knn_dists.masked_fill_(~src_knn_masks, 0.0)
    src_max_dists = src_knn_dists.max(1)[0]
    dist_mat = torch.sqrt(pairwise_squared_distance(ref_nodes, src_nodes))  # (M, N)
    intersect_mat = (ref_max_dists.unsqueeze(1) + src_max_dists.unsqueeze(0) + radius - dist_mat) > 0
    intersect_mat = intersect_mat & node_mask_mat
    sel_ref_indices, sel_src_indices = torch.nonzero(intersect_mat, as_tuple=True)

    ref_knn_masks = ref_knn_masks[sel_ref_indices]  # (P, K)
    src_knn_masks = src_knn_masks[sel_src_indices]  # (P, K)
    ref_knn_points = ref_knn_points[sel_ref_indices]  # (P, K, 3)
    src_knn_points = src_knn_points[sel_src_indices]  # (P, K, 3)
    point_mask_mat = ref_knn_masks.unsqueeze(2) & src_knn_masks.unsqueeze(1)  # (P, K, K)

    dist_mat = pairwise_squared_distance(ref_knn_points, src_knn_points)  # (P, K, K)
    dist_mat.masked_fill_(~point_mask_mat, 1e12)
    point_overlap_mat = dist_mat < radius**2
    ref_overlap_counts = torch.count_nonzero(point_overlap_mat.sum(-1), dim=-1).float()
    src_overlap_counts = torch.count_nonzero(point_overlap_mat.sum(-2), dim=-1).float()
    ref_overlaps = ref_overlap_counts / ref_knn_masks.sum(-1).float()
    src_overlaps = src_overlap_counts / src_knn_masks.sum(-1).float()
    overlaps = (ref_overlaps + src_overlaps) / 2

    overlap_masks = overlaps > 0
    corr_indices = torch.stack([sel_ref_indices[overlap_masks], sel_src_indices[overlap_masks]], dim=1)
    return corr_indices, overlaps[overlap_masks]


def rotation_angle(rotation: Tensor, other: Optional[Tensor] = None) -> Tensor:
    r"""Angle of a rotation, or of the relative rotation between two rotations, in degrees.

    $\theta = \arccos\left(\frac{\operatorname{tr}(R_1^\top R_2) - 1}{2}\right)$, the isotropic rotation error of
    registration benchmarks when `other` is the ground truth.

    Args:
        rotation: Rotation matrices, shape $(\ldots, 3, 3)$.
        other: Rotation matrices to compare with, same shape; `None` measures `rotation` against the identity.

    Returns:
        The angles in degrees, shape $(\ldots)$.

    Example:
        ```pycon
        >>> import math
        >>> import torch
        >>> from torch_pointcloud.ops.registration import rotation_angle
        >>> c, s = math.cos(math.pi / 6), math.sin(math.pi / 6)
        >>> rotation_angle(torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])).round().item()
        30.0

        ```
    """
    relative = rotation if other is None else rotation.transpose(-1, -2) @ other
    trace = relative[..., 0, 0] + relative[..., 1, 1] + relative[..., 2, 2]
    cosine = (0.5 * (trace - 1.0)).clamp(min=-1.0, max=1.0)
    return torch.rad2deg(torch.arccos(cosine))
