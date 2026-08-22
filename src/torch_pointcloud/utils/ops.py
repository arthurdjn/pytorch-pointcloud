"""Tensor operations on packed batches: safe division, softmax, voxel hashing, interpolation, and decimation."""

from typing import TYPE_CHECKING, Literal, Optional, Tuple, Union, overload

import numpy as np
import torch
from torch import Tensor
from torch_geometric.nn.pool.consecutive import consecutive_cluster

from .conversion import ensure_option
from .imports import _TORCH_CLUSTER_GITHUB_URL, _TORCH_SCATTER_GITHUB_URL, optional_import
from .types import OptTensor

if TYPE_CHECKING:
    from torch_cluster import knn
    from torch_scatter import scatter

scatter, _ = optional_import("torch_scatter", name="scatter", url=_TORCH_SCATTER_GITHUB_URL)
knn, _ = optional_import("torch_cluster", name="knn", url=_TORCH_CLUSTER_GITHUB_URL)


def safe_divide(a: Tensor, b: Tensor, /, default: Union[float, Tensor] = float("nan")) -> Tensor:
    """Safely divide two tensors, returning a default value if the denominator is zero.

    !!! note
        If the inputs are not floating point numbers,
        they will be converted to floating point numbers (float32).

    Args:
        a: The numerator tensor.
        b: The denominator tensor.
        default: The default value to return if the denominator is zero.

    Returns:
        The result of the division.

    Example:
        ```pycon
        >>> safe_divide(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([1.0, 0.0, 1.0]))
        tensor([1., nan, 3.])
        >>> safe_divide(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([1.0, 0.0, 1.0]), default=0.0)
        tensor([1., 0., 3.])
        >>> safe_divide(torch.tensor([1, 2, 3]), torch.tensor([1, 0, 1]), default=torch.tensor([0, 0, 0]))
        tensor([1., 0., 3.])

        ```
    """
    if not isinstance(default, Tensor):
        default = torch.full(a.shape, default, device=a.device)

    a = a if torch.is_floating_point(a) else a.float()
    b = b if torch.is_floating_point(b) else b.float()
    default = default if torch.is_floating_point(default) else default.float()
    return torch.where(b != 0, a / b, default)


def softmax(x: Tensor, batch: Tensor, dim: int = 0) -> Tensor:
    """Apply softmax on a packed x tensor.
    The x tensor is expected to be of shape $(N, *)$,
    where $N$ is the number of nodes and $*$ is the feature size.
    The `batch` tensor must be of shape $(N,)$ and must be contiguous.

    Note:
        This function is adapted from the `torch_geometric` package,
        and requires the `torch-scatter` package.

    Args:
        x: The x tensor of shape $(N, *)$.
        batch: The batch tensor of shape $(N,)$.
        dim: The dimension along which to apply the softmax.

    Returns:
        The softmaxed tensor of shape $(N, *)$.
    """
    N = batch.max() + 1
    src_max = scatter(x.detach(), batch, dim, dim_size=N, reduce="max")
    out = x - src_max.index_select(dim, batch)
    out = out.exp()
    out_sum = scatter(out, batch, dim, dim_size=N, reduce="sum") + 1e-16
    out_sum = out_sum.index_select(dim, batch)

    return out / out_sum


@overload
def voxel_grid_fnv(
    pos: Tensor,
    size: float,
    start: Optional[Tensor] = None,
    *,
    return_inverse: Literal[False] = False,
    return_counts: Literal[False] = False,
) -> Tensor: ...


@overload
def voxel_grid_fnv(
    pos: Tensor,
    size: float,
    start: Optional[Tensor] = None,
    *,
    return_inverse: Literal[True],
    return_counts: Literal[False] = False,
) -> Tuple[Tensor, Tensor]: ...


@overload
def voxel_grid_fnv(
    pos: Tensor,
    size: float,
    start: Optional[Tensor] = None,
    *,
    return_inverse: Literal[False] = False,
    return_counts: Literal[True],
) -> Tuple[Tensor, Tensor]: ...


@overload
def voxel_grid_fnv(
    pos: Tensor,
    size: float,
    start: Optional[Tensor] = None,
    *,
    return_inverse: Literal[True],
    return_counts: Literal[True],
) -> Tuple[Tensor, Tensor, Tensor]: ...


def voxel_grid_fnv(
    pos: Tensor,
    size: float,
    start: Optional[Tensor] = None,
    *,
    return_inverse: bool = False,
    return_counts: bool = False,
) -> Union[Tensor, Tuple[Tensor, ...]]:
    r"""FNV-1a 64-bit hash of integer voxel-grid coordinates. $(N, D) \to (N,)$.

    Args:
        pos: Point positions of shape $(N, D)$.
        size: Voxel side length in the same units as `pos`.
        start: Optional voxel-grid origin. When `None`, the grid origin is implicit via the
            internal `pos_grid -= pos_grid.min(0)` shift.
        return_inverse: If `True`, also return the per-point consecutive voxel index in $[0, V)$,
            following the semantics of `torch.unique(..., return_inverse=True)`.
        return_counts: If `True`, also return the per-voxel point count of shape $(V,)$.

    Returns:
        `hashed` of shape $(N,)$ when both flags are `False`. With `return_inverse=True` adds
        `inverse` of shape $(N,)$; with `return_counts=True` adds `count` of shape $(V,)$; both
        flags enabled returns `(hashed, inverse, count)`.
    """
    if start is not None:
        pos_grid = torch.floor((pos - start) / size).int()
    else:
        pos_grid = torch.floor(pos / size).int()
    pos_grid -= pos_grid.min(0).values

    # FNV-1a 64-bit hash (numpy uint64 for correct overflow semantics)
    arr = pos_grid.cpu().numpy().astype(np.uint64)
    hashed = np.full(arr.shape[0], 14695981039346656037, dtype=np.uint64)
    for j in range(arr.shape[1]):
        hashed *= np.uint64(1099511628211)
        hashed = np.bitwise_xor(hashed, arr[:, j])

    hashed_tensor = torch.from_numpy(hashed.view(np.int64)).to(pos.device)
    if not return_inverse and not return_counts:
        return hashed_tensor

    inverse, _ = consecutive_cluster(hashed_tensor)
    if return_inverse and return_counts:
        return hashed_tensor, inverse, torch.bincount(inverse)
    if return_inverse:
        return hashed_tensor, inverse
    return hashed_tensor, torch.bincount(inverse)


def first_permutation(cluster: Tensor, num_clusters: Optional[int] = None) -> Tensor:
    r"""Index of the first occurrence of each cluster id in a consecutive cluster tensor.

    The permutation returned by `consecutive_cluster` picks a backend-dependent representative per
    cluster (the last occurrence on CPU, a nondeterministic one on CUDA). This helper always picks
    the first occurrence, so `tensor[first_permutation(cluster)]` is deterministic across devices.

    Args:
        cluster: Consecutive cluster indices of shape $(N,)$ with values in $[0, V)$.
        num_clusters: Number of clusters $V$. Inferred as `cluster.max() + 1` when `None`.

    Returns:
        Long tensor of shape $(V,)$ holding, per cluster id, the smallest index in `cluster` with that id.

    Example:
        ```pycon
        >>> cluster = torch.tensor([1, 0, 1, 2, 0])
        >>> first_permutation(cluster)
        tensor([1, 0, 3])

        ```
    """
    n = cluster.numel()
    if num_clusters is None:
        num_clusters = int(cluster.max().item()) + 1 if n > 0 else 0
    perm = torch.arange(n, device=cluster.device)
    first = torch.full((num_clusters,), n, dtype=torch.long, device=cluster.device)
    return first.scatter_reduce_(0, cluster, perm, reduce="amin")


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
        Adapted from the `torch_geometric` package. Requires `torch-cluster`.

    Args:
        x: Node feature matrix $\mathbf{X} \in \mathbb{R}^{N \times F}$.
        pos_x: Node position matrix $\in \mathbb{R}^{N \times d}$.
        pos_y: Upsampled node position matrix $\in \mathbb{R}^{M \times d}$.
        batch_x: Batch vector $\mathbf{b_x} \in \{ 0, \ldots, B-1 \}^N$,
            assigning each node from $\mathbf{X}$ to a specific example.
        batch_y: Batch vector $\mathbf{b_y} \in \{ 0, \ldots, B-1 \}^M$,
            assigning each node from $\mathbf{Y}$ to a specific example.
        k: Number of neighbours.
        num_workers: Number of workers for computation. Has no effect when
            `batch_x` or `batch_y` is not `None`, or the input lies on GPU.
        weighting: Weighting scheme for neighbours. `"squared"` for $1/d^2$
            weights (`torch_geometric` default) or `"inverse"` for $1/d$
            weights (PointNet++ convention).
        eps: Small value to avoid division by zero.

    Returns:
        Interpolated features $\in \mathbb{R}^{M \times F}$.
    """
    weighting = ensure_option(weighting, ("squared", "inverse"), name="weighting")

    with torch.no_grad():
        assign_index = knn(pos_x, pos_y, k, batch_x=batch_x, batch_y=batch_y, num_workers=num_workers)
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
        generator: The generator to use for the random permutation.

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


def pad_tail(tensor: Tensor, pad_size: int, dim: int, fill_value: float = 0) -> Tensor:
    r"""Pad the tail of a tensor with a fill value.

    Args:
        tensor: The tensor to pad.
        pad_size: The size of the padding that will be added to the tail of the tensor.
        dim: The dimension along which to pad the tensor.
        fill_value: The value to fill the padding with.

    Returns:
        The padded tensor.

    Examples:
        ```pycon
        >>> tensor = torch.tensor([1, 2, 3])
        >>> pad_tail(tensor, pad_size=2, dim=0, fill_value=0)
        tensor([1, 2, 3, 0, 0])

        >>> tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
        >>> pad_tail(tensor, pad_size=2, dim=0, fill_value=0)
        tensor([[1, 2, 3],
                [4, 5, 6],
                [0, 0, 0],
                [0, 0, 0]])

        >>> tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
        >>> pad_tail(tensor, pad_size=2, dim=1, fill_value=0)
        tensor([[1, 2, 3, 0, 0],
                [4, 5, 6, 0, 0]])

        ```
    """
    if pad_size < 0:
        raise ValueError(f"The padding size must be non-negative, but got {pad_size}.")
    elif pad_size == 0:
        return tensor

    tail_shape = list(tensor.shape)
    tail_shape[dim] = pad_size
    tail = tensor.new_full(tail_shape, fill_value)
    return torch.cat([tensor, tail], dim=dim)
