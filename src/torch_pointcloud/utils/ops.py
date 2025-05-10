from typing import TYPE_CHECKING, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor

from .conversion import ensure_tuple_size
from .imports import optional_import
from .types import OptTensor

if TYPE_CHECKING:
    from torch_cluster import grid_cluster, knn
    from torch_scatter import scatter

scatter, _ = optional_import("torch_scatter", name="scatter")
grid_cluster, _ = optional_import("torch_cluster", name="grid_cluster")
knn, _ = optional_import("torch_cluster", name="knn")


def safe_divide(a: Tensor, b: Tensor, /, default: Union[float, Tensor] = float("nan")) -> Tensor:
    """Safely divide two tensors, returning a default value if the denominator is zero.

    > [!NOTE]
    > If the inputs are not floating point numbers,
    > they will be converted to floating point numbers (float32).

    Args:
        a: The numerator tensor.
        b: The denominator tensor.
        default: The default value to return if the denominator is zero.

    Returns:
        The result of the division.

    Example:
        >>> safe_divide(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([1.0, 0.0, 1.0]))
        tensor([1.0, nan, 3.0])
        >>> safe_divide(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([1.0, 0.0, 1.0]), default=0.0)
        tensor([1.0, 0.0, 3.0])
        >>> safe_divide(torch.tensor([1, 2, 3]), torch.tensor([1, 0, 1]), default=torch.tensor([0, 0, 0]))
        tensor([1.0, 0.0, 3.0])
    """
    if not isinstance(default, Tensor):
        default = torch.full(a.shape, default, device=a.device)

    a = a if torch.is_floating_point(a) else a.float()
    b = b if torch.is_floating_point(b) else b.float()
    default = default if torch.is_floating_point(default) else default.float()
    return torch.where(b != 0, a / b, default)


def softmax(x: Tensor, batch: Tensor, dim: int = 0) -> Tensor:
    """Apply softmax on a packed x tensor.
    The x tensor is expected to be of shape `(N, *)`,
    where `N` is the number of nodes and `*` is the feature size.
    The `batch` tensor must be of shape `(N,)` and must be contiguous.

    Note:
        This function is adapted from the `torch_geometric` package,
        and requires the `torch-scatter` package.

    Args:
        x: The x tensor of shape `(N, *)`.
        batch: The batch tensor of shape `(N,)`.
        dim: The dimension along which to apply the softmax.

    Returns:
        The softmaxed tensor of shape `(N, *)`.
    """
    N = batch.max() + 1
    src_max = scatter(x.detach(), batch, dim, dim_size=N, reduce="max")
    out = x - src_max.index_select(dim, batch)
    out = out.exp()
    out_sum = scatter(out, batch, dim, dim_size=N, reduce="sum") + 1e-16
    out_sum = out_sum.index_select(dim, batch)

    return out / out_sum


def voxel_grid(
    coords: Tensor,
    size: Union[float, Sequence[float], Tensor],
    batch: Optional[Tensor] = None,
    start: Optional[Union[float, Sequence[float], Tensor]] = None,
    end: Optional[Union[float, Sequence[float], Tensor]] = None,
) -> Tensor:
    """Creates a voxel grid from 3D coordinates. This function is compatible with
    batched coordinates in a packed format.

    Note:
        This function is adapted from [`torch-geometric`](https://github.com/pyg-team/pytorch_geometric)
        and depends on the [`torch-cluster`](https://github.com/rusty1s/pytorch_cluster) package.

    Args:
        coords: The 3D coordinates of shape `(N, 3)`.
        size: The size of the voxel grid.
        batch: The batch vector of shape `(N,)`.
        start: The start of the voxel grid.
        end: The end of the voxel grid.

    Returns:
        The voxel grid of shape `(N, 3)`.
    """
    coords = coords.unsqueeze(-1) if coords.dim() == 1 else coords
    dim = coords.size(1)

    if batch is None:
        batch = coords.new_zeros(coords.size(0), dtype=torch.long)

    coords = torch.cat([coords, batch.view(-1, 1).to(coords.dtype)], dim=-1)

    size = ensure_tuple_size(size, dim)
    size = torch.as_tensor(size, dtype=coords.dtype, device=coords.device)
    size = torch.cat([size, size.new_ones(1)])

    if start is not None:
        start = ensure_tuple_size(start, dim)
        start = torch.as_tensor(start, dtype=coords.dtype, device=coords.device)
        start = torch.cat([start, start.new_zeros(1)])

    if end is not None:
        end = ensure_tuple_size(end, dim)
        end = torch.as_tensor(end, dtype=coords.dtype, device=coords.device)
        end = torch.cat([end, batch.max().unsqueeze(0)])

    return grid_cluster(coords, size, start, end)


def consecutive_cluster(cluster: Tensor, return_permutation: bool = False) -> Union[Tuple[Tensor, Tensor], Tensor]:
    """Return consecutive cluster indices (and associated permutation)
    from a tensor of cluster indices. A cluster tensor is a tensor of shape `(N,)` where each element
    represents the cluster index of the corresponding point.

    Args:
        cluster: The cluster tensor of shape `(N,)`.
        return_permutation: Whether to return the permutation. Can be used
            to select batch indices, target categories etc. belonging to the same cluster.

    Returns:
        The consecutive cluster indices and the permutation.

    Example:
        >>> cluster = torch.tensor([10, 2, 31, 10, 10, 31, 5, 6, 5])
        >>> inv, perm = consecutive_cluster(cluster, return_permutation=True)
        >>> inv
        tensor([3, 0, 4, 3, 3, 4, 1, 2, 1])
        >>> perm
        tensor([1, 8, 7, 4, 5])
    """
    unique, inv = torch.unique(cluster, sorted=True, return_inverse=True)
    if not return_permutation:
        return inv

    perm = torch.arange(inv.size(0), dtype=inv.dtype, device=inv.device)
    perm = inv.new_empty(unique.size(0)).scatter_(0, inv, perm)
    return inv, perm


def knn_interpolate(
    x: Tensor,
    coords_x: Tensor,
    coords_y: Tensor,
    batch_x: OptTensor = None,
    batch_y: OptTensor = None,
    k: int = 3,
    num_workers: int = 1,
) -> Tensor:
    r"""The k-NN interpolation from the `"PointNet++: Deep Hierarchical
    Feature Learning on Point Sets in a Metric Space"
    <https://arxiv.org/abs/1706.02413>`_ paper.

    For each point $y$ with position $\mathbf{p}(y)$, its
    interpolated features $\mathbf{f}(y)$ are given by

    $$
        \mathbf{f}(y) = \frac{\sum_{i=1}^k w(x_i) \mathbf{f}(x_i)}{\sum_{i=1}^k
        w(x_i)} \textrm{, where } w(x_i) = \frac{1}{d(\mathbf{p}(y),
        \mathbf{p}(x_i))^2}
    $$

    and $\{ x_1, \ldots, x_k \}$ denoting the $k$ nearest points
    to $y$.

    Note:
        This function is adapted from the `torch_geometric` package,
        and requires the `torch-cluster` package.

    Args:
        x: Node feature matrix $\mathbf{X} \in \mathbb{R}^{N \times F}$.
        coords_x: Node position matrix $\in \mathbb{R}^{N \times d}$.
        coords_y: Upsampled node position matrix $\in \mathbb{R}^{M \times d}$.
        batch_x: Batch vector $\mathbf{b_x} \in {\{ 0, \ldots, B-1\}}^N$, which assigns
            each node from $\mathbf{X}$ to a specific example.
        batch_y: Batch vector $\mathbf{b_y} \in {\{ 0, \ldots, B-1\}}^N$, which assigns
            each node from $\mathbf{Y}$ to a specific example.
        k: Number of neighbors.
        num_workers: Number of workers to use for computation.
            Has no effect in case `batch_x` or `batch_y` is not `None`, or the input lies on the GPU.
    """
    with torch.no_grad():
        assign_index = knn(coords_x, coords_y, k, batch_x=batch_x, batch_y=batch_y, num_workers=num_workers)
        y_idx, x_idx = assign_index[0], assign_index[1]
        diff = coords_x[x_idx] - coords_y[y_idx]
        squared_distance = (diff * diff).sum(dim=-1, keepdim=True)
        weights = 1.0 / torch.clamp(squared_distance, min=1e-16)

    y = scatter(x[x_idx] * weights, y_idx, dim=0, dim_size=coords_y.size(0), reduce="sum")
    y = y / scatter(weights, y_idx, dim=0, dim_size=coords_y.size(0), reduce="sum")
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
        batch: The packed batch index tensor of consecutive integers (starting from 0).
        factor: The factor to decimate the indices by.
        generator: The generator to use for the random permutation.

    Returns:
        The decimated indices and the decimated batch indices.

    Examples:
        >>> batch = torch.tensor([0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3])
        >>> decimate_indices(batch, 2)
        (tensor([ 0,  4,  7,  6,  9, 10]), tensor([0, 1, 2, 2, 3, 3]))
    """
    if factor < 1:
        raise ValueError(
            f"The argument `factor` should be higher than (or equal to) 1 for downsampling, but got {factor}"
        )

    batch_size = int(batch.max().item() + 1)
    decim_indices = []
    decim_batch = []

    for i in range(batch_size):
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
        batch: The batch tensor of shape `(N,)`.
        factor: The factor to decimate the tensors by.
        generator: The generator to use for the random permutation.

    Returns:
        A tuple of decimated tensors and the decimated batch indices.

    Examples:
        >>> tensors = (torch.randn(10, 3), torch.randn(10, 4))
        >>> batch = torch.tensor([0, 1, 1, 1, 2, 2, 2, 2, 3, 3])
        >>> decimate(tensors, batch, 2)
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
    """
    idx_decim, batch_decim = decimate_indices(batch, factor, generator=generator)
    tensors_decim = tuple(tensor[idx_decim] for tensor in tensors)
    return tensors_decim, batch_decim
