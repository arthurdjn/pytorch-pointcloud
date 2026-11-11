r"""Rigid registration primitives: pairwise distances, the weighted Procrustes fit, correspondence selection, the
local-to-global registration, and coarse-to-fine patch partitions.

A rigid transform is a $4 \times 4$ matrix $T = \begin{bmatrix} R & t \\ 0 & 1 \end{bmatrix}$ mapping the source cloud
into the reference frame, $q = R p + t$; `ops.geometry.transform_points` applies it.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.utils import cumsum, degree, index_to_mask, mask_to_index, scatter, to_dense_batch

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
    its determinant. The fit runs in at least float32, outside autocast, and the transforms come back in the dtype of
    `src_points`.

    Args:
        src_points: Source correspondences, shape $(\ldots, N, 3)$.
        ref_points: Reference correspondences aligned to `src_points`, same shape.
        weights: Per-correspondence weights, shape $(\ldots, N)$; `None` weights every pair equally.
        weight_threshold: Weights strictly below it are ignored.
        eps: Added to the weight normalizer.

    Returns:
        The transforms mapping the source onto the reference, shape $(\ldots, 4, 4)$.

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
    # SVD and determinant have no half-precision kernels, so the fit runs in at least float32 and outside autocast.
    dtype = src_points.dtype
    compute_dtype = torch.promote_types(dtype, torch.float32)
    batch_shape, num_points = src_points.shape[:-2], src_points.size(-2)
    num_sets = math.prod(batch_shape)
    with torch.autocast(src_points.device.type, enabled=False):
        src_points = src_points.reshape(num_sets, num_points, 3).to(compute_dtype)
        ref_points = ref_points.reshape(num_sets, num_points, 3).to(compute_dtype)
        if weights is None:
            weights = torch.ones_like(src_points[:, :, 0])
        weights = weights.reshape(num_sets, num_points).to(compute_dtype)
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
    return transform.reshape(*batch_shape, 4, 4).to(dtype)


def mutual_topk_correspondences(
    scores: Tensor,
    k: int,
    threshold: float,
    row_mask: OptTensor = None,
    col_mask: OptTensor = None,
) -> Tensor:
    r"""Entries of score matrices among the top-$k$ of both their row and their column, and above `threshold`.

    Args:
        scores: Score matrices, shape $(B, N, M)$.
        k: Number of entries kept per row and per column; capped at the matrix size.
        threshold: Minimum score of a kept entry, exclusive.
        row_mask: Valid rows, shape $(B, N)$; every row when `None`.
        col_mask: Valid columns, shape $(B, M)$; every column when `None`.

    Returns:
        The selected entries, a boolean tensor of shape $(B, N, M)$.

    Example:
        ```pycon
        >>> import torch
        >>> from torch_pointcloud.ops.registration import mutual_topk_correspondences
        >>> scores = torch.tensor([[[0.9, 0.1], [0.2, 0.8]]])
        >>> mutual_topk_correspondences(scores, k=1, threshold=0.05).tolist()
        [[[True, False], [False, True]]]

        ```
    """
    k = min(k, scores.size(1), scores.size(2))
    row_topk_scores, row_topk_indices = scores.topk(k=k, dim=2)
    row_selected = torch.zeros_like(scores).scatter_(2, row_topk_indices, row_topk_scores) > threshold
    col_topk_scores, col_topk_indices = scores.topk(k=k, dim=1)
    col_selected = torch.zeros_like(scores).scatter_(1, col_topk_indices, col_topk_scores) > threshold
    selected = row_selected & col_selected
    if row_mask is not None:
        selected = selected & row_mask.unsqueeze(2)
    if col_mask is not None:
        selected = selected & col_mask.unsqueeze(1)
    return selected


@torch.no_grad()
def local_to_global_registration(
    src_points: Tensor,
    ref_points: Tensor,
    scores: Tensor,
    group: Tensor,
    acceptance_radius: float,
    min_group_size: int,
    num_refinement_steps: int,
    batch: OptTensor = None,
    batch_size: Optional[int] = None,
) -> Tensor:
    r"""Rigid transform of every pair from its correspondences, grouped into local hypotheses.

    Every group of at least `min_group_size` correspondences proposes the `weighted_procrustes` transform of its
    own correspondences; the proposal with the most inliers among all the correspondences of its pair (residual
    under `acceptance_radius`) is refit on those inliers, then refit `num_refinement_steps - 1` more times on the
    inliers of the previous fit. A pair without any large enough group starts from the fit on all its
    correspondences, and a pair without correspondences gets the identity. The registration runs outside autocast.

    Args:
        src_points: Source point of every correspondence, shape $(C, 3)$.
        ref_points: Reference point of every correspondence, shape $(C, 3)$.
        scores: Weight of every correspondence, shape $(C,)$.
        group: Hypothesis every correspondence belongs to, shape $(C,)$, sorted; a group lies within one pair.
        acceptance_radius: Residual under which a correspondence is an inlier of a transform.
        min_group_size: Number of correspondences a group needs to propose a transform.
        num_refinement_steps: Number of fits on the inliers of the selected transform.
        batch: Pair every correspondence belongs to, shape $(C,)$, sorted; `None` for a single pair.
        batch_size: Number of pairs $B$; inferred from `batch` when `None`.

    Returns:
        The transform moving the source onto the reference of every pair, shape $(B, 4, 4)$.

    Example:
        ```pycon
        >>> import torch
        >>> from torch_pointcloud.ops.registration import local_to_global_registration
        >>> src = torch.rand(8, 3)
        >>> group = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        >>> transform = local_to_global_registration(src, src + 0.5, torch.ones(8), group, 0.1, 3, 5)
        >>> transform.shape, torch.allclose(transform[0, :3, 3], torch.full((3,), 0.5), atol=1e-4)
        (torch.Size([1, 4, 4]), True)

        ```
    """
    if batch is None:
        batch = group.new_zeros(group.size(0))
    if batch_size is None:
        batch_size = int(batch.max()) + 1 if batch.numel() else 1

    # Residuals in half precision would blur the inlier test, so the registration runs outside autocast.
    with torch.autocast(src_points.device.type, enabled=False):
        pair_src, pair_masks = to_dense_batch(src_points, batch, batch_size=batch_size)  # (B, C_max, 3)
        pair_ref, _ = to_dense_batch(ref_points, batch, batch_size=batch_size)
        pair_scores, _ = to_dense_batch(scores, batch, batch_size=batch_size)  # zero on the padding

        def inliers(transforms: Tensor, pairs: Tensor) -> Tensor:
            residuals = torch.linalg.norm(pair_ref[pairs] - transform_points(pair_src[pairs], transforms), dim=2)
            return (residuals < acceptance_radius) & pair_masks[pairs]

        pairs = torch.arange(batch_size, device=batch.device)
        current_scores = pair_scores * inliers(weighted_procrustes(pair_src, pair_ref, pair_scores), pairs).float()
        counts = torch.bincount(group)
        proposals = mask_to_index(counts >= min_group_size)
        if proposals.numel() > 0:
            group_batch = scatter(batch, group, dim=0, dim_size=counts.size(0), reduce="max")[proposals]
            max_group_size = int(counts[proposals].max())
            group_src, _ = to_dense_batch(src_points, group, max_num_nodes=max_group_size)
            group_ref, _ = to_dense_batch(ref_points, group, max_num_nodes=max_group_size)
            group_scores, _ = to_dense_batch(scores, group, max_num_nodes=max_group_size)
            transforms = weighted_procrustes(group_src[proposals], group_ref[proposals], group_scores[proposals])
            proposal_inliers = inliers(transforms, group_batch)  # (P, C_max)
            # The first proposal with the most inliers of every pair: proposals are sorted by pair.
            num_inliers, _ = to_dense_batch(
                proposal_inliers.sum(dim=1), group_batch, fill_value=-1, batch_size=batch_size
            )
            best = num_inliers.argmax(dim=1) + cumsum(degree(group_batch, batch_size, dtype=torch.long))[:-1]
            has_proposal = num_inliers.amax(dim=1) >= 0
            best_scores = pair_scores * proposal_inliers[best.clamp(max=proposals.numel() - 1)].float()
            current_scores = torch.where(has_proposal.unsqueeze(1), best_scores, current_scores)

        transform = weighted_procrustes(pair_src, pair_ref, current_scores)
        for _ in range(num_refinement_steps - 1):
            transform = weighted_procrustes(pair_src, pair_ref, pair_scores * inliers(transform, pairs).float())
        identity = torch.eye(4, dtype=transform.dtype, device=transform.device)
        return torch.where(pair_masks.any(dim=1).view(-1, 1, 1), transform, identity)


@torch.no_grad()
def point_to_node_partition(
    points: Tensor,
    nodes: Tensor,
    point_limit: int,
    batch: OptTensor = None,
    node_batch: OptTensor = None,
    batch_size: Optional[int] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    r"""Assign every point to its nearest node of the same cloud and gather each node's patch of at most `point_limit`
    points.

    A patch holds the points assigned to the node, nearest first; a node with more points than `point_limit`
    keeps the nearest ones, a node with fewer is padded with the index $N$ (a shadow row) and masked.

    Args:
        points: Fine points, shape $(N, 3)$.
        nodes: Coarse points, shape $(M, 3)$.
        point_limit: Patch size $K$.
        batch: Cloud of every point, shape $(N,)$, sorted; `None` for a single cloud.
        node_batch: Cloud of every node, shape $(M,)$, sorted; `None` for a single cloud.
        batch_size: Number of clouds; inferred from `node_batch` when `None`.

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
    if batch is None:
        batch = points.new_zeros(points.size(0), dtype=torch.long)
    if node_batch is None:
        node_batch = nodes.new_zeros(nodes.size(0), dtype=torch.long)
    if batch_size is None:
        batch_size = int(node_batch.max()) + 1 if node_batch.numel() else 1

    dense_points, point_masks = to_dense_batch(points, batch, batch_size=batch_size)  # (B, N_max, 3)
    dense_nodes, dense_node_masks = to_dense_batch(nodes, node_batch, batch_size=batch_size)  # (B, M_max, 3)
    point_ptr = cumsum(degree(batch, batch_size, dtype=torch.long))[:-1]
    node_ptr = cumsum(degree(node_batch, batch_size, dtype=torch.long))[:-1]

    sq_dist_mat = pairwise_squared_distance(dense_nodes, dense_points)  # (B, M_max, N_max)
    sq_dist_mat.masked_fill_(~dense_node_masks.unsqueeze(2), float("inf"))
    dense_point_to_node = sq_dist_mat.min(dim=1)[1]  # (B, N_max)
    point_to_node = (dense_point_to_node + node_ptr.unsqueeze(1))[point_masks]
    node_masks = index_to_mask(point_to_node, size=nodes.size(0))

    node_indices = torch.arange(sq_dist_mat.size(1), device=nodes.device)
    matching_masks = (dense_point_to_node.unsqueeze(1) == node_indices.view(1, -1, 1)) & point_masks.unsqueeze(1)
    sq_dist_mat.masked_fill_(~matching_masks, 1e12)

    k = min(point_limit, sq_dist_mat.size(2))
    node_knn_indices = sq_dist_mat.topk(k=k, dim=2, largest=False)[1]  # (B, M_max, k)
    node_knn_masks = torch.gather(matching_masks, 2, node_knn_indices)
    node_knn_indices = (node_knn_indices + point_ptr.view(-1, 1, 1)).masked_fill(~node_knn_masks, points.size(0))
    node_knn_indices, node_knn_masks = node_knn_indices[dense_node_masks], node_knn_masks[dense_node_masks]
    if k < point_limit:
        node_knn_indices = F.pad(node_knn_indices, (0, point_limit - k), value=points.size(0))
        node_knn_masks = F.pad(node_knn_masks, (0, point_limit - k), value=False)
    return point_to_node, node_masks, node_knn_indices, node_knn_masks


@torch.no_grad()
def node_correspondences(
    src_nodes: Tensor,
    ref_nodes: Tensor,
    src_knn_points: Tensor,
    ref_knn_points: Tensor,
    transform: Tensor,
    radius: float,
    src_masks: OptTensor = None,
    ref_masks: OptTensor = None,
    src_knn_masks: OptTensor = None,
    ref_knn_masks: OptTensor = None,
    src_batch: OptTensor = None,
    ref_batch: OptTensor = None,
) -> Tuple[Tensor, Tensor]:
    r"""Ground-truth node (patch) correspondences of every pair and their overlap ratios under a known transform.

    Two patches of a pair correspond when, after moving the source into the reference frame, they share points
    closer than `radius`. The overlap of a correspondence is the mean of the fraction of reference patch points with
    a match and the fraction of source patch points with a match. Patches whose enclosing spheres cannot intersect
    are skipped.

    Args:
        src_nodes: Source nodes, shape $(N, 3)$.
        ref_nodes: Reference nodes, shape $(M, 3)$.
        src_knn_points: Source patches, shape $(N, K, 3)$.
        ref_knn_points: Reference patches, shape $(M, K, 3)$.
        transform: Ground-truth transform of the source onto the reference of every pair, shape $(B, 4, 4)$.
        radius: Distance under which two points match.
        src_masks: Nodes of the source with a non-empty patch, shape $(N,)$.
        ref_masks: Nodes of the reference with a non-empty patch, shape $(M,)$.
        src_knn_masks: Validity of the source patch slots, shape $(N, K)$.
        ref_knn_masks: Validity of the reference patch slots, shape $(M, K)$.
        src_batch: Pair of every source node, shape $(N,)$, sorted; `None` for a single pair.
        ref_batch: Pair of every reference node, shape $(M,)$, sorted; `None` for a single pair.

    Returns:
        A tuple `(corr_indices, corr_overlaps)`: the `(source node, reference node)` index pairs with a positive
        overlap, shape $(C, 2)$, ordered by pair then reference node, and their overlap ratios, shape $(C,)$.

    Example:
        ```pycon
        >>> import torch
        >>> from torch_pointcloud.ops.registration import node_correspondences
        >>> nodes = torch.tensor([[0.0, 0, 0], [5.0, 0, 0]])
        >>> patches = nodes.unsqueeze(1) + torch.tensor([[[0.0, 0, 0], [0.1, 0, 0]]])
        >>> transform = torch.eye(4).unsqueeze(0)
        >>> corr, overlaps = node_correspondences(nodes, nodes, patches, patches, transform, radius=0.05)
        >>> corr.tolist(), overlaps.tolist()
        ([[0, 0], [1, 1]], [1.0, 1.0])

        ```
    """
    if src_batch is None:
        src_batch = src_nodes.new_zeros(src_nodes.size(0), dtype=torch.long)
    if ref_batch is None:
        ref_batch = ref_nodes.new_zeros(ref_nodes.size(0), dtype=torch.long)
    if src_masks is None:
        src_masks = torch.ones(src_nodes.size(0), dtype=torch.bool, device=src_nodes.device)
    if ref_masks is None:
        ref_masks = torch.ones(ref_nodes.size(0), dtype=torch.bool, device=ref_nodes.device)
    if src_knn_masks is None:
        src_knn_masks = torch.ones(src_knn_points.shape[:2], dtype=torch.bool, device=src_nodes.device)
    if ref_knn_masks is None:
        ref_knn_masks = torch.ones(ref_knn_points.shape[:2], dtype=torch.bool, device=ref_nodes.device)

    batch_size = transform.size(0)
    src_transform = transform[src_batch]  # (N, 4, 4)
    src_nodes = transform_points(src_nodes.unsqueeze(1), src_transform).squeeze(1)
    src_knn_points = transform_points(src_knn_points, src_transform)

    ref_knn_dists = torch.linalg.norm(ref_knn_points - ref_nodes.unsqueeze(1), dim=-1)  # (M, K)
    ref_knn_dists.masked_fill_(~ref_knn_masks, 0.0)
    ref_max_dists = ref_knn_dists.max(1)[0]
    src_knn_dists = torch.linalg.norm(src_knn_points - src_nodes.unsqueeze(1), dim=-1)  # (N, K)
    src_knn_dists.masked_fill_(~src_knn_masks, 0.0)
    src_max_dists = src_knn_dists.max(1)[0]

    dense_ref_nodes, dense_ref_masks = to_dense_batch(ref_nodes, ref_batch, batch_size=batch_size)  # (B, M_max, 3)
    dense_src_nodes, dense_src_masks = to_dense_batch(src_nodes, src_batch, batch_size=batch_size)  # (B, N_max, 3)
    dense_ref_max_dists, _ = to_dense_batch(ref_max_dists, ref_batch, batch_size=batch_size)
    dense_src_max_dists, _ = to_dense_batch(src_max_dists, src_batch, batch_size=batch_size)
    dense_ref_masks &= to_dense_batch(ref_masks, ref_batch, batch_size=batch_size)[0]
    dense_src_masks &= to_dense_batch(src_masks, src_batch, batch_size=batch_size)[0]
    node_mask_mat = dense_ref_masks.unsqueeze(2) & dense_src_masks.unsqueeze(1)  # (B, M_max, N_max)

    dist_mat = torch.sqrt(pairwise_squared_distance(dense_ref_nodes, dense_src_nodes))  # (B, M_max, N_max)
    intersect_mat = (dense_ref_max_dists.unsqueeze(2) + dense_src_max_dists.unsqueeze(1) + radius - dist_mat) > 0
    intersect_mat = intersect_mat & node_mask_mat
    sel_batch, sel_ref_indices, sel_src_indices = torch.nonzero(intersect_mat, as_tuple=True)
    sel_ref_indices = sel_ref_indices + cumsum(degree(ref_batch, batch_size, dtype=torch.long))[sel_batch]
    sel_src_indices = sel_src_indices + cumsum(degree(src_batch, batch_size, dtype=torch.long))[sel_batch]

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
    corr_indices = torch.stack([sel_src_indices[overlap_masks], sel_ref_indices[overlap_masks]], dim=1)
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
