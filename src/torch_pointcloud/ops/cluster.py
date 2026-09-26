"""Neighbor search and grouping: kNN, FPS, radius queries, local grids, decimation, and kNN interpolation."""

from typing import List, Literal, Optional, Tuple, Union, overload

import torch
import torch.nn.functional as F
import torch_geometric.nn.pool as pool
from torch import Tensor
from torch_geometric.utils import scatter

from torch_pointcloud.config import FPS_RANDOM_START, KNN_DENSE_BUDGET
from torch_pointcloud.utils.conversion import ensure_option
from torch_pointcloud.utils.types import OptTensor


def _check_sorted_batch(batch: Tensor, name: str) -> None:
    if batch.numel() > 1 and bool((batch[1:] < batch[:-1]).any()):
        raise ValueError(f"`{name}` must be sorted in non-decreasing order.")


def _check_packed_2d(src: Tensor, name: str) -> None:
    if src.dim() != 2:
        raise ValueError(
            f"`{name}` must be a packed 2D tensor of shape (N, D); got shape {tuple(src.shape)}. Point clouds "
            "are never padded to (B, N, D): concatenate the clouds along the first dimension and pass a "
            "`batch` index of shape (N,) instead."
        )


def knn(
    x: Tensor,
    y: Tensor,
    k: int,
    batch_x: OptTensor = None,
    batch_y: OptTensor = None,
    cosine: bool = False,
    num_workers: int = 1,
    batch_size: Optional[int] = None,
) -> Tensor:
    r"""Find the $k$ nearest neighbors in $x$ for each point in $y$.
    This function is a wrapper around the `torch_geometric.nn.pool.knn` function, and supports the same arguments.
    However, in case the `batch_x` and `batch_y` tensors are provided, and the samples have the same number of nodes,
    this function uses a more efficient implementation that is significantly faster on GPU using `torch.cdist` + `topk`.

    Important:
        If provided, the `batch_x` and `batch_y` tensors must be sorted in non-decreasing order
        (both the dense fast path and `torch_geometric` require it); unsorted batches raise a `ValueError`.

    Note:
        When a point of $y$ coincides with a point of $x$ (e.g. `knn(pos, pos, k)`), the query point
        itself counts among the $k$ neighbors, matching `torch_geometric.nn.pool.knn`.

    Args:
        x: The source tensor to find the nearest neighbors of shape $(N, *)$.
        y: The target tensor to find the nearest neighbors of shape $(M, *)$.
        k: The number of nearest neighbors to find.
        batch_x: The batch tensor of the source tensor of shape $(N,)$.
        batch_y: The batch tensor of the target tensor of shape $(M,)$.
        cosine: Whether to use cosine distance.
        num_workers: The number of workers to use for the computation.
        batch_size: The batch size to use for the computation.

    Returns:
        The nearest neighbors of shape $(2, M \cdot k)$.
    """
    _check_packed_2d(x, "x")
    _check_packed_2d(y, "y")
    if batch_x is not None:
        _check_sorted_batch(batch_x, "batch_x")
    if batch_y is not None:
        _check_sorted_batch(batch_y, "batch_y")

    def _sparse_knn() -> Tensor:
        return pool.knn(
            x=x,
            y=y,
            k=k,
            batch_x=batch_x,
            batch_y=batch_y,
            cosine=cosine,
            num_workers=num_workers,
            batch_size=batch_size,
        )

    if batch_x is None or batch_y is None:
        return _sparse_knn()

    counts_x = batch_x.bincount()
    counts_y = batch_y.bincount()

    if counts_x.numel() == 0 or not (counts_x[0] == counts_x).all() or not (counts_y[0] == counts_y).all():
        return _sparse_knn()

    N_x = int(counts_x[0].item())
    N_y = int(counts_y[0].item())
    B = counts_x.numel()

    # cdist materialises the full $(B, N, N)$ distance matrix; fall back to the
    # streaming `torch_geometric` implementation for larger clouds.
    if B * N_x * N_y > KNN_DENSE_BUDGET or N_x < k:
        return _sparse_knn()

    x_3d = x.view(B, N_x, -1)
    y_3d = y.view(B, N_y, -1)

    if cosine:
        x_3d = F.normalize(x_3d, dim=-1)
        y_3d = F.normalize(y_3d, dim=-1)

    dist = torch.cdist(y_3d, x_3d)  # (B, N_y, N_x)
    _, idx = dist.topk(k, dim=-1, largest=False)  # (B, N_y, k)

    offsets = torch.arange(B, device=x.device).view(B, 1, 1) * N_x
    src = (idx + offsets).reshape(-1)

    offsets_y = torch.arange(B, device=x.device).view(B, 1, 1) * N_y
    dst = (torch.arange(N_y, device=x.device).view(1, N_y, 1).expand(B, N_y, k) + offsets_y).reshape(-1)

    return torch.stack([dst, src], dim=0)


def knn_graph(
    x: Tensor,
    k: int,
    batch: OptTensor = None,
    loop: bool = False,
    flow: str = "source_to_target",
    cosine: bool = False,
    num_workers: int = 1,
    batch_size: Optional[int] = None,
) -> Tensor:
    r"""Compute the kNN graph of $x$.

    This function is a drop-in for `torch_geometric.nn.pool.knn_graph`, except that when the
    `batch` tensor partitions the points into uniformly-sized samples this function
    uses a `torch.cdist` + `topk` implementation that is significantly faster on GPU
    than the underlying `torch_geometric.nn.pool.knn_graph`.

    Important:
        If provided, the `batch` tensor must be sorted in non-decreasing order (both the dense
        fast path and `torch_geometric` require it); an unsorted batch raises a `ValueError`.

    Args:
        x: The input tensor of shape $(N, *)$.
        k: The number of nearest neighbors to find. When `loop=False`, the
            self-edge is excluded from the result.
        batch: The batch tensor of shape $(N,)$.
        loop: Whether to include self-edges.
        flow: Either `"source_to_target"` (PyG default, `edge_index = (src, dst)`
            where `src` is the neighbor and `dst` is the central point) or `"target_to_source"`.
        cosine: Whether to use cosine distance.
        num_workers: Forwarded to the `torch_geometric` fallback.
        batch_size: Forwarded to the `torch_geometric` fallback.

    Returns:
        Edge index of shape $(2, k \cdot N)$.
    """
    _check_packed_2d(x, "x")
    if batch is not None:
        _check_sorted_batch(batch, "batch")

    def _sparse_knn_graph() -> Tensor:
        return pool.knn_graph(
            x=x,
            k=k,
            batch=batch,
            loop=loop,
            flow=flow,
            cosine=cosine,
            num_workers=num_workers,
            batch_size=batch_size,
        )

    if batch is None:
        return _sparse_knn_graph()

    counts = batch.bincount()
    if counts.numel() == 0 or not (counts[0] == counts).all():
        return _sparse_knn_graph()

    N = int(counts[0].item())
    B = counts.numel()
    if loop and N < k or not loop and N < k + 1:
        return _sparse_knn_graph()

    # cdist materialises the full $(B, N, N)$ distance matrix; fall back to the
    # streaming `torch_geometric` implementation for larger clouds.
    if B * N * N > KNN_DENSE_BUDGET:
        return _sparse_knn_graph()

    x_3d = x.view(B, N, -1)
    if cosine:
        x_3d = F.normalize(x_3d, dim=-1)

    # `torch.cdist` returns squared euclidean^0.5; for top-k argmin the order is the same
    # as for the squared distance, so we use the cheaper `cdist` directly.
    dist = torch.cdist(x_3d, x_3d)  # (B, N, N)
    k_query = k if loop else k + 1
    _, idx = dist.topk(k_query, dim=-1, largest=False)  # (B, N, k_query)

    if not loop:
        # Drop the self-edge. `topk` is not guaranteed to put the self-distance first if
        # there are exact-zero ties, so mask explicitly on the global node index.
        offsets = torch.arange(B, device=x.device).view(B, 1, 1) * N
        idx_global = idx + offsets
        self_idx = (torch.arange(N, device=x.device).view(1, N, 1) + offsets).expand(B, N, k_query)
        keep_mask = idx_global != self_idx  # (B, N, k_query)
        # Sort so that the self-edge (if any) drops to the end, then take the first k.
        sort_keys = (~keep_mask).long()
        order = torch.argsort(sort_keys, dim=-1, stable=True)
        idx = torch.gather(idx, -1, order)[..., :k]

    offsets = torch.arange(B, device=x.device).view(B, 1, 1) * N
    src = (idx + offsets).reshape(-1)
    dst = (torch.arange(N, device=x.device).view(1, N, 1).expand(B, N, k) + offsets).reshape(-1)

    if flow == "target_to_source":
        return torch.stack([dst, src], dim=0)
    return torch.stack([src, dst], dim=0)


def fps(
    src: Tensor,
    batch: Optional[Tensor] = None,
    ratio: Optional[Union[Tensor, float]] = None,
    num_nodes: Optional[Union[Tensor, float]] = None,
    random_start: bool = True,
    batch_size: Optional[int] = None,
    ptr: Optional[Union[Tensor, List[int]]] = None,
) -> Tensor:
    r"""A sampling algorithm from the paper :arxiv: [PointNet++: Deep Hierarchical Feature
    Learning on Point Sets in a Metric Space](https://arxiv.org/abs/1706.02413)
    by Qi et al., which iteratively samples the most distant point with regard
    to the rest points.

    This function wraps `torch_geometric.nn.pool.fps` and supports sampling a fixed number of nodes per sample
    with the `num_nodes` argument.

    Important:
        If provided, the `batch` tensor is expected to be sorted.

    Args:
        src: The source tensor to sample from of shape $(N, *)$.
        batch: The batch tensor to sample from of shape $(N,)$.
        ratio: The sampling ratio.
        num_nodes: The number of nodes to sample. When a sample holds fewer than `num_nodes` points, indices
            repeat (sampling with replacement, matching the reference CUDA FPS), so the output always holds
            `num_nodes` indices per sample and downstream shapes stay stable.
        random_start: Whether to start the sampling randomly.
        batch_size: The batch size.
        ptr: The pointer tensor to sample from.

    Returns:
        The sampled indices of shape $(M,)$.

    Examples:
        ```pycon
        >>> import torch
        >>> from torch_pointcloud.ops.cluster import fps
        >>> src = torch.randn(100, 3)
        >>> batch = torch.cat([torch.zeros(50), torch.ones(50)]).long()
        >>> idx = fps(src, batch, num_nodes=10)  # doctest: +SKIP
        >>> print(idx.shape)  # doctest: +SKIP
        torch.Size([10])

        ```
    """
    random_start = random_start if FPS_RANDOM_START is None else FPS_RANDOM_START

    if ratio is None and num_nodes is None:
        raise ValueError("Either `ratio` or `num_nodes` must be provided.")
    if ratio is not None and num_nodes is not None:
        raise ValueError("Only one of `ratio` or `num_nodes` can be provided.")
    _check_packed_2d(src, "src")
    if batch is not None and src.size(0) != batch.numel():
        raise ValueError(f"Size of `src` ({src.size(0)}) must match size of `batch` ({batch.numel()}).")

    if batch is not None and batch.numel() > 0:
        sample_counts = batch.bincount()
        if bool((sample_counts == 0).any()):
            empty = int((sample_counts == 0).nonzero()[0].item())
            raise ValueError(
                f"`batch` has no points for sample {empty}: batch ids must be contiguous integers 0..B-1 with "
                "at least one point per sample."
            )

    if ptr is not None:
        ptr = torch.as_tensor(ptr, dtype=torch.long, device=src.device)
        if batch_size is None:
            batch_size = ptr.numel() - 1
        batch = torch.repeat_interleave(torch.arange(ptr.numel() - 1, device=src.device), ptr[1:] - ptr[:-1])

    if ratio is not None and not (isinstance(ratio, Tensor) and ratio.numel() > 1):
        return pool.fps(src, batch, ratio=float(ratio), random_start=random_start, batch_size=batch_size)

    if batch is not None:
        if batch_size is None:
            batch_size = int(batch.max()) + 1
        node_counts = batch.bincount(minlength=batch_size)
        ptr = torch.cat([torch.zeros(1, dtype=torch.long, device=src.device), node_counts.cumsum(0)])
    else:
        node_counts = torch.tensor([src.size(0)], device=src.device)
        batch_size = 1
        ptr = torch.tensor([0, src.size(0)], device=src.device)

    if isinstance(ratio, Tensor):
        req_nodes = (node_counts.to(ratio.dtype) * ratio.to(src.device)).ceil().long()
    elif isinstance(num_nodes, (int, float)):
        req_nodes = torch.full((batch_size,), int(num_nodes), dtype=torch.long, device=src.device)
    else:
        req_nodes = torch.as_tensor(num_nodes, dtype=torch.long, device=src.device)
        if req_nodes.ndim == 0:
            req_nodes = req_nodes.expand(batch_size)
        elif req_nodes.size(0) != batch_size:
            raise ValueError(f"Size of `num_nodes` ({req_nodes.size(0)}) must match batch size ({batch_size}).")

    # FPS takes one ratio for the whole batch. It is greedy, so sampling every sample at the largest requested
    # ratio and keeping the first `req_nodes` indices of each gives the same indices as a per-sample ratio.
    max_ratio = (req_nodes.double() / node_counts.clamp(min=1).double()).max().item()
    idx = pool.fps(src, batch, ratio=max_ratio, random_start=random_start, batch_size=batch_size)

    sampled_batch = torch.searchsorted(ptr, idx, right=True) - 1
    sampled_counts = sampled_batch.bincount(minlength=batch_size)
    offsets = torch.cat([torch.zeros(1, dtype=torch.long, device=src.device), sampled_counts.cumsum(0)])

    local_rank = torch.arange(idx.numel(), device=src.device) - offsets[sampled_batch]
    mask = local_rank < req_nodes[sampled_batch]
    return idx[mask]


def local_grid(src: Tensor, size: float, batch: Tensor | None = None) -> Tensor:
    r"""Applies local grid quantization to the source tensor as explained in the paper
    [TorchSparse++: Efficient Point Cloud Engine](https://openaccess.thecvf.com/content/CVPR2023W/WAD/papers/Tang_TorchSparse_Efficient_Point_Cloud_Engine_CVPRW_2023_paper.pdf)
    by Tang et al., which quantizes the source tensor to a local grid.

    Note:
        If a batch tensor is provided, the quantization is applied to each sample separately, so the grid of every
        sample starts at its own minimum cell.

    Args:
        src: The source tensor to quantize of shape $(N, *)$.
        size: The quantization size.
        batch: The associated batch tensor of shape $(N,)$.

    Returns:
        The quantized source tensor of shape $(N, *)$.

    Examples:
        ```pycon
        >>> import torch
        >>> from torch_pointcloud.ops.cluster import local_grid
        >>> src = torch.randn(100, 3)
        >>> batch = torch.cat([torch.zeros(50), torch.ones(50)]).long()
        >>> src_grid = local_grid(src, size=1.0, batch=batch)  # doctest: +SKIP

        ```
    """

    src_quantized = torch.div(src, size, rounding_mode="floor").long()

    # If the batch is not provided we compute the global shift the source tensor to the origin (faster)
    # Otherwise we compute the local shifts for each graph contained in the batch (slower)
    if batch is None:
        src_min, _ = src_quantized.min(0)
    else:
        src_min = scatter(src_quantized, batch, dim=0, reduce="min")
        src_min = src_min[batch]

    return src_quantized - src_min


def radius(
    x: Tensor,
    y: Tensor,
    r: float,
    batch_x: OptTensor = None,
    batch_y: OptTensor = None,
    max_num_neighbors: int = 32,
    sort: bool = False,
) -> tuple[Tensor, Tensor]:
    r"""`torch_geometric.nn.pool.radius` wrapper with an optional sort-by-source-index tie-breaker.

    With `sort=False` (default) this just delegates to `torch_geometric.nn.pool.radius` and
    returns edges in kernel-traversal order. With `sort=True`, when more than
    `max_num_neighbors` source points lie inside a ball, the $k$ smallest source
    indices are kept (PointNet++'s reference `query_ball_point` behavior). Pretrained
    PointNet++ checkpoints from yanx27 / charlesq34 overfit to this selection rule,
    so reproducing their accuracy requires `sort=True`.

    Important:
        If provided, the `batch_x` and `batch_y` tensors must be sorted in non-decreasing order
        (`torch_geometric.nn.pool.radius` requires it); unsorted batches raise a `ValueError`.

    Args:
        x: Source positions, shape $(N_x, d)$.
        y: Query positions, shape $(N_y, d)$.
        r: Ball radius.
        batch_x: Batch index for $x$, shape $(N_x,)$. `None` for a single batch.
        batch_y: Batch index for $y$, shape $(N_y,)$. `None` for a single batch.
        max_num_neighbors: Max neighbors $k$ kept per query.
        sort: Sort in-ball source indices ascending and keep the first $k$.

    Returns:
        `(row, col)` edges. `row` is the query index, `col` is the source index.
        Centroids with no in-ball neighbors emit zero edges; pooling leaves those
        rows at the reduction identity.
    """
    _check_packed_2d(x, "x")
    _check_packed_2d(y, "y")
    if batch_x is not None:
        _check_sorted_batch(batch_x, "batch_x")
    if batch_y is not None:
        _check_sorted_batch(batch_y, "batch_y")
    if not sort:
        edge_index = pool.radius(x, y, r, batch_x, batch_y, max_num_neighbors=max_num_neighbors)
        return edge_index[0], edge_index[1]

    # Custom sort-by-source-index ball query. Asking `radius` for the
    # full `max_num_neighbors=Nx` over-allocates memory; instead we materialise the
    # squared-distance matrix per batch element (bounded by $(N_y^b \cdot N_x^b)$).
    device = x.device
    if batch_x is None:
        batch_x = torch.zeros(x.size(0), dtype=torch.long, device=device)
    if batch_y is None:
        batch_y = torch.zeros(y.size(0), dtype=torch.long, device=device)
    n_batches = int(batch_x.max().item()) + 1 if batch_x.numel() else 1
    k = int(max_num_neighbors)
    r_sq = float(r) * float(r)

    rows_out: List[Tensor] = []
    cols_out: List[Tensor] = []
    for b in range(n_batches):
        x_mask = batch_x == b
        y_mask = batch_y == b
        x_b = x[x_mask]
        y_b = y[y_mask]
        if x_b.numel() == 0 or y_b.numel() == 0:
            continue
        x_idx_global = torch.nonzero(x_mask, as_tuple=False).flatten()
        y_idx_global = torch.nonzero(y_mask, as_tuple=False).flatten()
        nx_b = x_b.size(0)
        ny_b = y_b.size(0)

        sqr = ((y_b.unsqueeze(1) - x_b.unsqueeze(0)) ** 2).sum(-1)  # (Ny_b, Nx_b)
        local = torch.arange(nx_b, device=device).unsqueeze(0).expand(ny_b, -1).clone()
        local = local.masked_fill(sqr > r_sq, nx_b)  # nx_b is the out-of-radius sentinel
        sorted_local, _ = local.sort(dim=-1)
        picked = sorted_local[:, :k]  # (Ny_b, min(k, Nx_b))
        valid_mask = picked < nx_b
        kept = picked.size(-1)

        local_cols = picked[valid_mask]
        local_rows = torch.arange(ny_b, device=device).unsqueeze(-1).expand(-1, kept)[valid_mask]
        rows_out.append(y_idx_global[local_rows])
        cols_out.append(x_idx_global[local_cols])

    if not rows_out:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty
    return torch.cat(rows_out), torch.cat(cols_out)


@overload
def group(
    pos: Tensor,
    batch: Tensor,
    num_groups: int,
    group_size: int,
    random_start: bool = ...,
    *,
    return_indices: Literal[False] = ...,
) -> Tuple[Tensor, Tensor]: ...


@overload
def group(
    pos: Tensor,
    batch: Tensor,
    num_groups: int,
    group_size: int,
    random_start: bool = ...,
    *,
    return_indices: Literal[True],
) -> Tuple[Tensor, Tensor, Tensor]: ...


def group(
    pos: Tensor,
    batch: Tensor,
    num_groups: int,
    group_size: int,
    random_start: bool = False,
    *,
    return_indices: bool = False,
) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]:
    r"""Partition a packed point cloud into local groups with FPS centers and a $k$-NN neighborhood.

    Farthest point sampling selects `num_groups` centers per sample, then $k$-NN gathers the
    `group_size` nearest neighbors of each center, and each neighborhood is recentered on its center.
    Because `num_groups` is fixed per sample, the packed result densifies to a regular $(B, G, k, 3)$
    batch without padding.

    Args:
        pos: Packed point coordinates of shape $(N, 3)$.
        batch: Per-point batch index of shape $(N,)$.
        num_groups: Number of groups (FPS centers) $G$ per sample.
        group_size: Number of neighbors $k$ per group.
        random_start: Whether to start farthest point sampling from a random point.
        return_indices: If `True`, also return the flat neighbor index into the packed input.

    Returns:
        `(neighborhood, center)`, or `(neighborhood, center, idx)` when `return_indices` is `True`.
        `neighborhood` has shape $(B, G, k, 3)$ recentered on each center, `center` has shape
        $(B, G, 3)$, and `idx` has shape $(B \cdot G \cdot k,)$ indexing the packed input.

    Shape:
        - Input: $(N, 3)$ and $(N,)$.
        - Output: $(B, G, k, 3)$ and $(B, G, 3)$ (plus $(B \cdot G \cdot k,)$ when `return_indices`).

    Example:
        ```python
        import torch
        from torch_pointcloud.ops.cluster import group

        pos = torch.randn(2048, 3)
        batch = torch.cat([torch.zeros(1024), torch.ones(1024)]).long()
        neighborhood, center = group(pos, batch, num_groups=64, group_size=32)
        print(neighborhood.shape, center.shape)
        ```
    """
    batch_size = int(batch.max().item()) + 1
    # Fewer points than `num_groups` is fine (fps repeats indices), but knn cannot return more
    # neighbors than a sample has points, which would break the dense (B, G, k, 3) reshape.
    min_points = int(batch.bincount(minlength=batch_size).min())
    if min_points < group_size:
        raise ValueError(
            f"`group` requires at least `group_size` ({group_size}) points per sample to gather `num_groups` "
            f"({num_groups}) full k-NN neighborhoods, but the smallest sample has {min_points} points."
        )

    idx_center = fps(pos, batch, num_nodes=num_groups, random_start=random_start)
    center = pos[idx_center]
    batch_center = batch[idx_center]

    _, col = knn(pos, center, group_size, batch_x=batch, batch_y=batch_center)

    neighborhood = pos[col].view(batch_size, num_groups, group_size, 3)
    center = center.view(batch_size, num_groups, 3)
    neighborhood = neighborhood - center.unsqueeze(2)

    if return_indices:
        return neighborhood, center, col.view(batch_size * num_groups, group_size).reshape(-1)
    return neighborhood, center


def knn_interpolate(
    x: Tensor,
    pos_x: Tensor,
    pos_y: Tensor,
    batch_x: OptTensor = None,
    batch_y: OptTensor = None,
    k: int = 3,
    num_workers: int = 1,
    weighting: Literal["squared", "inverse"] = "squared",
    eps: float = 1e-16,
) -> Tensor:
    r"""k-NN interpolation with inverse-distance weighting.

    From :arxiv: [PointNet++: Deep Hierarchical Feature Learning on Point Sets in a
    Metric Space](https://arxiv.org/abs/1706.02413).

    For each point $y$ with position $\mathbf{p}(y)$, its
    interpolated features $\mathbf{f}(y)$ are given by

    $$
        \mathbf{f}(y) = \frac{\sum_{i=1}^k w(x_i) \mathbf{f}(x_i)}{\sum_{i=1}^k
        w(x_i)}
    $$

    where $\{ x_1, \ldots, x_k \}$ are the $k$ nearest points to $y$ and
    the weights $w(x_i)$ depend on the chosen `weighting` scheme:

    - `"squared"` (default, `torch_geometric` convention):
      $w(x_i) = 1 / d(\mathbf{p}(y), \mathbf{p}(x_i))^2$
    - `"inverse"` (PointNet++ `three_interpolation` convention):
      $w(x_i) = 1 / d(\mathbf{p}(y), \mathbf{p}(x_i))$

    Note:
        Adapted from the `torch_geometric` package.

    Args:
        x: Node feature matrix $\mathbf{X} \in \mathbb{R}^{N \times F}$.
        pos_x: Node position matrix $\in \mathbb{R}^{N \times d}$.
        pos_y: Upsampled node position matrix $\in \mathbb{R}^{M \times d}$.
        batch_x: Batch vector $\mathbf{b_x} \in \{ 0, \ldots, B-1 \}^N$,
            assigning each node from $\mathbf{X}$ to a specific example.
        batch_y: Batch vector $\mathbf{b_y} \in \{ 0, \ldots, B-1 \}^M$,
            assigning each node from $\mathbf{Y}$ to a specific example.
        k: Number of neighbors.
        num_workers: Number of workers for computation. Has no effect when
            `batch_x` or `batch_y` is not `None`, or the input lies on GPU.
        weighting: Weighting scheme for neighbors. `"squared"` for $1/d^2$
            weights (`torch_geometric` default) or `"inverse"` for $1/d$
            weights (PointNet++ convention).
        eps: Small value to avoid division by zero.

    Returns:
        Interpolated features $\in \mathbb{R}^{M \times F}$.
    """
    weighting = ensure_option(weighting, ("squared", "inverse"), name="weighting")

    with torch.no_grad():
        assign_index = pool.knn(pos_x, pos_y, k, batch_x=batch_x, batch_y=batch_y, num_workers=num_workers)
        y_idx, x_idx = assign_index[0], assign_index[1]
        diff = pos_x[x_idx] - pos_y[y_idx]
        squared_distance = (diff * diff).sum(dim=-1, keepdim=True)

        if weighting == "squared":
            weights = 1.0 / (squared_distance + eps)
        else:
            dist = squared_distance.sqrt()
            weights = 1.0 / (dist + eps)

    y = scatter(x[x_idx] * weights, y_idx, dim=0, dim_size=pos_y.size(0), reduce="sum")
    y = y / scatter(weights, y_idx, dim=0, dim_size=pos_y.size(0), reduce="sum")
    return y


@torch.no_grad()
def decimate_indices(
    batch: Tensor, factor: float, generator: Optional[torch.Generator] = None
) -> Tuple[Tensor, Tensor]:
    """Decimate indices from a packed batch index tensor.
    This function will return the decimated indices by a given factor along with the decimated batch indices.

    Note:
        This function is similar to the `decimation_indices` function in the `torch-geometric` package,
        except that this function uses the `batch` tensor instead of the `ptr` tensor representation.

    Args:
        batch: The packed batch index tensor.
        factor: The factor to decimate the indices by.
        generator: The generator to use for the random permutation. When given, it is reseeded from each
            sample's own point count, so a sample's decimation does not depend on the rest of the batch.

    Returns:
        The decimated indices and the decimated batch indices.

    Examples:
        ```pycon
        >>> batch = torch.tensor([0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3])
        >>> decimate_indices(batch, 2)  # doctest: +SKIP
        (tensor([ 0,  4,  7,  6,  9, 10]), tensor([0, 1, 2, 2, 3, 3]))

        ```
    """
    if factor < 1:
        raise ValueError(
            f"The argument `factor` should be higher than (or equal to) 1 for downsampling, but got {factor}"
        )

    decim_indices = []
    decim_batch = []

    for i in torch.unique(batch).tolist():
        mask_i = batch == i
        size_i = int(mask_i.sum().item())

        # NOTE: Get at least one point to avoid empty decimation
        decim_size = max(1, int(size_i // factor))

        indices = torch.where(mask_i)[0]
        if generator is not None:
            # Reseed per sample: one generator drawn sequentially would make a sample's permutation
            # depend on how many draws the samples before it in the batch consumed.
            generator.manual_seed(size_i)

        perm = torch.randperm(size_i, device=batch.device, generator=generator)[:decim_size]

        # Decimate indices following the random permutation
        decim_indices.append(indices[perm])
        # Add batch indices for the decimated points
        decim_batch.append(torch.full((decim_size,), i, device=batch.device))

    return torch.cat(decim_indices), torch.cat(decim_batch)


def decimate(
    tensors: Tuple[Tensor, ...],
    batch: Tensor,
    factor: int,
    generator: Optional[torch.Generator] = None,
) -> Tuple[Tuple[Tensor, ...], Tensor]:
    """Decimates each input tensor by the given factor.
    This will return the decimated tensors along with the decimated batch indices.

    Note:
        This function is similar to the `decimate` function introduced in the `torch-geometric` RandLANet example.

    Args:
        tensors: A tuple of tensors to decimate.
        batch: The batch tensor of shape $(N,)$.
        factor: The factor to decimate the tensors by.
        generator: The generator to use for the random permutation.

    Returns:
        A tuple of decimated tensors and the decimated batch indices.

    Examples:
        ```pycon
        >>> tensors = (torch.randn(10, 3), torch.randn(10, 4))
        >>> batch = torch.tensor([0, 1, 1, 1, 2, 2, 2, 2, 3, 3])
        >>> decimate(tensors, batch, 2)  # doctest: +SKIP
        ((tensor([[-1.4570, -0.1023, -0.5992],
                [ 0.2408,  0.1325,  0.7642],
                [-0.2104, -1.4391,  0.5214],
                [ 1.6192,  1.4506,  0.2695],
                [ 0.3488,  0.9676, -0.4657]]),
        tensor([[-0.1933,  0.6526, -1.9006,  0.2286],
                [ 1.2888,  0.0523, -1.5469,  0.7567],
                [ 0.9442, -0.1849,  1.0608,  0.2083],
                [ 0.4788,  1.3537, -0.1593, -0.4249],
                [ 1.3065,  0.4598,  0.2618, -0.7599]])),
        tensor([0, 1, 2, 2, 3]))

        ```
    """
    idx_decim, batch_decim = decimate_indices(batch, factor, generator=generator)
    tensors_decim = tuple(tensor[idx_decim] for tensor in tensors)
    return tensors_decim, batch_decim
