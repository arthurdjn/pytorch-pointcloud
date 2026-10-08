"""Chamfer distance between batched point sets."""

from typing import Literal

import torch
from torch import Tensor, nn
from torch_geometric.nn import knn

from torch_pointcloud.utils.imports import _KAOLIN_GITHUB_URL, _PYG_LIB_AVAILABLE, optional_import

__all__ = ["ChamferDistance"]

sided_distance, _KAOLIN_AVAILABLE = optional_import(
    "kaolin.metrics.pointcloud",
    "sided_distance",
    url=_KAOLIN_GITHUB_URL,
)


def _nearest_index(query: Tensor, points: Tensor) -> Tensor:
    r"""Index in `points` $(B, M, D)$ of the nearest point to every `query` $(B, N, D)$, without autograd."""
    if _KAOLIN_AVAILABLE and query.is_cuda:
        # kaolin's brute-force CUDA kernel, several times faster than the kNN search below.
        with torch.no_grad():
            _, index = sided_distance(query, points)

        return index.long()

    if not _PYG_LIB_AVAILABLE:
        # Without a neighbor-search kernel: the exact dense distances, whose memory grows with $N \cdot M$.
        with torch.no_grad():
            return torch.cdist(query, points, compute_mode="donot_use_mm_for_euclid_dist").argmin(dim=2)

    batch, num_queries, dim = query.shape
    num_points = points.shape[1]
    query_batch = torch.arange(batch, device=query.device).repeat_interleave(num_queries)
    point_batch = torch.arange(batch, device=query.device).repeat_interleave(num_points)
    with torch.no_grad():
        pairs = knn(points.reshape(-1, dim), query.reshape(-1, dim), 1, point_batch, query_batch)

    # `pairs` holds (query row, point row) over the packed sets: back to per-scene point indices in query order.
    index = torch.empty(batch * num_queries, dtype=torch.long, device=query.device)
    index[pairs[0]] = pairs[1] - pairs[0].div(num_queries, rounding_mode="floor") * num_points
    return index.view(batch, num_queries)


def chamfer_distance(pred: Tensor, target: Tensor, norm: Literal["l1", "l2"] = "l2") -> Tensor:
    r"""Symmetric Chamfer distance between two batched point sets.

    Set-to-set reconstruction objective introduced for point cloud generation in
    :arxiv: [A Point Set Generation Network for 3D Object Reconstruction from a Single Image](https://arxiv.org/abs/1612.00603) (Fan et al., 2017) and standard for masked point
    modeling pretraining (the SSL pretraining models return `(pred, target)` group coordinates in
    exactly this layout). For each point the squared euclidean distance to its nearest neighbor in
    the other set is computed, then reduced over all points and batches:

    $$\text{CD}_{\ell_2} = \frac{1}{BN} \sum \min_j \lVert p_i - q_j \rVert_2^2
    + \frac{1}{BM} \sum \min_i \lVert p_i - q_j \rVert_2^2$$

    $$\text{CD}_{\ell_1} = \frac{1}{2} \Big( \frac{1}{BN} \sum \min_j \lVert p_i - q_j \rVert_2
    + \frac{1}{BM} \sum \min_i \lVert p_i - q_j \rVert_2 \Big)$$

    The `"l2"` variant sums the two directed means of squared distances (no square root, no
    halving); the `"l1"` variant averages the two directed means of euclidean distances. Both
    follow the reference pretraining convention, so losses are comparable with published values.

    The nearest neighbors come from the `sided_distance` kernel of
    [kaolin](https://github.com/NVIDIAGameWorks/kaolin) when it is installed and the points are on CUDA,
    otherwise from the kNN kernel of `torch_geometric` (`pyg-lib`); either way without autograd, and only
    the matched pairs are differentiated, so the memory grows with $N + M$ rather than $N \cdot M$ and whole
    clouds fit. Without either kernel the search falls back to the dense distance matrix.

    Args:
        pred: Predicted point sets of shape $(B, N, 3)$.
        target: Target point sets of shape $(B, M, 3)$.
        norm: Distance variant, `"l1"` (euclidean) or `"l2"` (squared euclidean).

    Returns:
        Scalar Chamfer distance averaged over all points and batches.

    Shape:
        - Input: $(B, N, 3)$ and $(B, M, 3)$.
        - Output: scalar.

    Example:
        ```python
        import torch
        from torch_pointcloud.losses import chamfer_distance

        pred = torch.randn(64, 32, 3, requires_grad=True)
        target = torch.randn(64, 32, 3)
        loss = chamfer_distance(pred, target, norm="l2")
        loss.backward()
        print(loss.shape)
        ```
    """
    if norm not in ("l1", "l2"):
        raise ValueError(f"`norm` must be 'l1' or 'l2', got {norm!r}.")

    # Nearest neighbors found without autograd, then only the matched pairs differentiated.
    nearest_target = target.gather(1, _nearest_index(pred, target)[..., None].expand(-1, -1, pred.shape[-1]))
    nearest_pred = pred.gather(1, _nearest_index(target, pred)[..., None].expand(-1, -1, pred.shape[-1]))
    dist_pred = (pred - nearest_target).pow(2).sum(-1)  # (B, N)
    dist_target = (target - nearest_pred).pow(2).sum(-1)  # (B, M)
    if norm == "l1":
        return (dist_pred.sqrt().mean() + dist_target.sqrt().mean()) / 2
    return dist_pred.mean() + dist_target.mean()


class ChamferDistance(nn.Module):
    r"""Module form of [`chamfer_distance`][torch_pointcloud.losses.chamfer.chamfer_distance].

    The reconstruction criterion of the masked point modeling models, whose pretraining `forward` returns
    the `(pred, target)` pair this module takes.

    Args:
        norm: Distance variant, `"l1"` (euclidean) or `"l2"` (squared euclidean).

    Example:
        ```python
        import torch
        criterion = ChamferDistance(norm="l1")
        criterion(torch.zeros(2, 8, 3), torch.zeros(2, 4, 3)).item()  # 0.0
        ```
    """

    def __init__(self, norm: Literal["l1", "l2"] = "l2") -> None:
        super().__init__()
        if norm not in ("l1", "l2"):
            raise ValueError(f"`norm` must be 'l1' or 'l2', got {norm!r}.")

        self.norm = norm

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        r"""Compute the Chamfer distance between `pred` $(B, N, 3)$ and `target` $(B, M, 3)$."""
        return chamfer_distance(pred, target, norm=self.norm)

    def extra_repr(self) -> str:
        return f"norm={self.norm!r}"
