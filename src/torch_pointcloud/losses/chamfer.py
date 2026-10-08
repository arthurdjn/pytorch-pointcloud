"""Chamfer distance between packed point sets."""

from typing import Callable, Literal, Optional

import torch
from torch import Tensor, nn
from torch_geometric.nn import knn
from torch_geometric.utils import to_dense_batch

from torch_pointcloud.utils.imports import _KAOLIN_GITHUB_URL, _PYG_LIB_AVAILABLE, optional_import

__all__ = ["ChamferDistance"]

sided_distance, _KAOLIN_AVAILABLE = optional_import(
    "kaolin.metrics.pointcloud",
    "sided_distance",
    url=_KAOLIN_GITHUB_URL,
)


def _kaolin_nearest(query_dense: Tensor, point_dense: Tensor) -> Tensor:
    r"""kaolin's brute-force CUDA kernel over padded sets, several times faster than the kNN search."""
    _, index = sided_distance(query_dense, point_dense)
    return index.long()


def _dense_nearest(query_dense: Tensor, point_dense: Tensor) -> Tensor:
    r"""Exact search over the dense distances of padded sets, whose memory grows with $N \cdot M$."""
    return torch.cdist(query_dense, point_dense, compute_mode="donot_use_mm_for_euclid_dist").argmin(dim=2)


def _padded_nearest_index(
    query: Tensor,
    query_batch: Tensor,
    points: Tensor,
    point_batch: Tensor,
    num_sets: int,
    search: Callable[[Tensor, Tensor], Tensor],
) -> Tensor:
    r"""Packed nearest-point index through a `search` over the padded sets $(B, N_\max, D)$ and $(B, M_\max, D)$."""
    with torch.no_grad():
        query_dense, query_mask = to_dense_batch(query, query_batch, batch_size=num_sets)
        # Padding the points with infinity keeps them out of every minimum.
        point_dense, point_mask = to_dense_batch(points, point_batch, fill_value=float("inf"), batch_size=num_sets)
        index_dense = search(query_dense, point_dense)

    # Back to packed indices: the points of a set start where the previous sets end.
    counts = point_mask.sum(dim=1)
    offsets = counts.cumsum(dim=0) - counts
    return (index_dense + offsets[:, None])[query_mask]


def _nearest_index(query: Tensor, query_batch: Tensor, points: Tensor, point_batch: Tensor, num_sets: int) -> Tensor:
    r"""Packed index in `points` $(M, D)$ of the nearest point of the same set to every `query` $(N, D)$."""
    if _KAOLIN_AVAILABLE and query.is_cuda:
        return _padded_nearest_index(query, query_batch, points, point_batch, num_sets, _kaolin_nearest)

    if not _PYG_LIB_AVAILABLE:
        return _padded_nearest_index(query, query_batch, points, point_batch, num_sets, _dense_nearest)

    with torch.no_grad():
        pairs = knn(points, query, 1, point_batch, query_batch)

    # `pairs` holds (query row, point row); one neighbor per query, in whatever order the kernel emits them.
    index = torch.empty(query.shape[0], dtype=torch.long, device=query.device)
    index[pairs[0]] = pairs[1]
    return index


def chamfer_distance(
    pred: Tensor,
    target: Tensor,
    *,
    pred_batch: Optional[Tensor] = None,
    target_batch: Optional[Tensor] = None,
    norm: Literal["l1", "l2"] = "l2",
) -> Tensor:
    r"""Symmetric Chamfer distance between packed point sets.

    Set-to-set reconstruction objective introduced for point cloud generation in
    :arxiv: [A Point Set Generation Network for 3D Object Reconstruction from a Single Image](https://arxiv.org/abs/1612.00603) (Fan et al., 2017) and standard for masked point
    modeling pretraining. Every point is matched to its nearest neighbor in the other set of the same
    index (its group, its scene), and the squared euclidean distances are averaged over all points:

    $$\text{CD}_{\ell_2} = \frac{1}{N} \sum_i \min_j \lVert p_i - q_j \rVert_2^2
    + \frac{1}{M} \sum_j \min_i \lVert p_i - q_j \rVert_2^2$$

    $$\text{CD}_{\ell_1} = \frac{1}{2} \Big( \frac{1}{N} \sum_i \min_j \lVert p_i - q_j \rVert_2
    + \frac{1}{M} \sum_j \min_i \lVert p_i - q_j \rVert_2 \Big)$$

    The `"l2"` variant sums the two directed means of squared distances (no square root, no
    halving); the `"l1"` variant averages the two directed means of euclidean distances. Both
    follow the reference pretraining convention, so losses are comparable with published values.

    The nearest neighbors come from the `sided_distance` kernel of
    [kaolin](https://github.com/NVIDIAGameWorks/kaolin) when it is installed and the points are on CUDA,
    otherwise from the kNN kernel of `torch_geometric` (`pyg-lib`); either way without autograd, and only
    the matched pairs are differentiated, so the memory grows with $N + M$ rather than $N \cdot M$ and whole
    clouds fit. Without either kernel the search falls back to the dense distance matrix.

    Args:
        pred: Predicted points of every set, packed, shape $(N, 3)$.
        target: Target points of every set, packed, shape $(M, 3)$.
        pred_batch: Set index of every predicted point (sorted), shape $(N,)$; `None` for a single set.
        target_batch: Set index of every target point (sorted), shape $(M,)$; `None` for a single set.
        norm: Distance variant, `"l1"` (euclidean) or `"l2"` (squared euclidean).

    Returns:
        Scalar Chamfer distance averaged over all points.

    Shape:
        - pred: $(N, 3)$
        - target: $(M, 3)$
        - pred_batch: $(N,)$
        - target_batch: $(M,)$
        - output: scalar

    Example:
        ```python
        import torch
        from torch_pointcloud.losses import chamfer_distance

        pred = torch.randn(64 * 32, 3, requires_grad=True)
        target = torch.randn(64 * 32, 3)
        batch = torch.arange(64).repeat_interleave(32)
        loss = chamfer_distance(pred, target, pred_batch=batch, target_batch=batch)
        loss.backward()
        print(loss.shape)
        ```
    """
    if norm not in ("l1", "l2"):
        raise ValueError(f"`norm` must be 'l1' or 'l2', got {norm!r}.")

    single_set = pred_batch is None and target_batch is None
    if pred_batch is None:
        pred_batch = pred.new_zeros(pred.shape[0], dtype=torch.long)
    if target_batch is None:
        target_batch = target.new_zeros(target.shape[0], dtype=torch.long)
    num_sets = 1 if single_set else int(torch.maximum(pred_batch.max(), target_batch.max())) + 1

    # Nearest neighbors found without autograd, then only the matched pairs differentiated.
    nearest_target = target[_nearest_index(pred, pred_batch, target, target_batch, num_sets)]
    nearest_pred = pred[_nearest_index(target, target_batch, pred, pred_batch, num_sets)]
    dist_pred = (pred - nearest_target).pow(2).sum(-1)  # (N,)
    dist_target = (target - nearest_pred).pow(2).sum(-1)  # (M,)
    if norm == "l1":
        return (dist_pred.sqrt().mean() + dist_target.sqrt().mean()) / 2
    return dist_pred.mean() + dist_target.mean()


class ChamferDistance(nn.Module):
    r"""Module form of [`chamfer_distance`][torch_pointcloud.losses.chamfer.chamfer_distance].

    The reconstruction criterion of the masked point modeling models, whose pretraining `forward` returns
    the `(pred, target, pred_batch, target_batch)` tuple this module takes.

    Args:
        norm: Distance variant, `"l1"` (euclidean) or `"l2"` (squared euclidean).

    Example:
        ```python
        import torch
        criterion = ChamferDistance(norm="l1")
        criterion(torch.zeros(8, 3), torch.zeros(4, 3)).item()  # 0.0
        ```
    """

    def __init__(self, norm: Literal["l1", "l2"] = "l2") -> None:
        super().__init__()
        if norm not in ("l1", "l2"):
            raise ValueError(f"`norm` must be 'l1' or 'l2', got {norm!r}.")

        self.norm = norm

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        pred_batch: Optional[Tensor] = None,
        target_batch: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Compute the Chamfer distance between the packed sets `pred` $(N, 3)$ and `target` $(M, 3)$."""
        return chamfer_distance(pred, target, pred_batch=pred_batch, target_batch=target_batch, norm=self.norm)

    def extra_repr(self) -> str:
        return f"norm={self.norm!r}"
