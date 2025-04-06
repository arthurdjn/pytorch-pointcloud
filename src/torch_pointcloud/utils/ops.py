from typing import TYPE_CHECKING, Optional, Sequence, Union

import torch
from torch import Tensor

from .conversion import ensure_tuple_size
from .imports import optional_import

if TYPE_CHECKING:
    from torch_scatter import scatter

scatter, _ = optional_import("torch_scatter", name="scatter")
grid_cluster, _ = optional_import("torch_cluster", name="grid_cluster")


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
