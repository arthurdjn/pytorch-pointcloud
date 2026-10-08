"""Voxel grid subsampling of packed point clouds."""

from typing import Literal, Tuple, overload

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn.pool import voxel_grid
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from torch_geometric.utils import scatter, segment


class VoxelGridPool(nn.Module):
    r"""Voxel grid subsampling: the points falling in the same `grid_size` cell are reduced to a single point.

    Positions are averaged within a cell and features are reduced with `reduce`. The grid origin is either aligned
    to multiples of `grid_size` (`origin="grid"`, so cell boundaries do not depend on the cloud's extent) or placed
    at each cloud's minimum corner (`origin="min"`). Passing `return_inverse=True` also returns the point-to-cell map.

    Args:
        grid_size: Cell edge length.
        reduce: Feature reduction over a cell: `"max"`, `"mean"`, `"sum"` or `"min"`.
        origin: Where the grid starts, `"grid"` or `"min"`.

    Shape:
        - Input: $(N, C)$ features, $(N, 3)$ positions and $(N,)$ batch index.
        - Output: $(M, C)$ features, $(M, 3)$ positions, $(M,)$ batch index and, with `return_inverse`, the $(N,)$
          cell index of every input point.

    Example:
        ```python
        >>> import torch
        >>> from torch_pointcloud.layers import VoxelGridPool
        >>> pool = VoxelGridPool(grid_size=0.5)
        >>> x, pos, batch = torch.randn(100, 8), torch.rand(100, 3), torch.zeros(100, dtype=torch.long)
        >>> x_pooled, pos_pooled, batch_pooled = pool(x, pos, batch)  # doctest: +SKIP
        >>> x_pooled.shape[1], pos_pooled.shape[1]  # doctest: +SKIP
        (8, 3)

        ```
    """

    def __init__(self, grid_size: float, reduce: str = "max", origin: Literal["grid", "min"] = "grid") -> None:
        super().__init__()
        if reduce not in ("sum", "mean", "min", "max"):
            raise ValueError(f"Invalid reduce operation: {reduce!r}, expected 'sum', 'mean', 'min' or 'max'.")

        if origin not in ("grid", "min"):
            raise ValueError(f"Invalid origin: {origin!r}, expected 'grid' or 'min'.")

        self.grid_size = grid_size
        self.reduce = reduce
        self.origin = origin

    @overload
    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        batch: Tensor,
        return_inverse: Literal[True],
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]: ...

    @overload
    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        batch: Tensor,
        return_inverse: Literal[False] = False,
    ) -> Tuple[Tensor, Tensor, Tensor]: ...

    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        batch: Tensor,
        return_inverse: bool = False,
    ) -> Tuple[Tensor, ...]:
        if self.origin == "grid":
            start = torch.floor(pos.min(dim=0).values / self.grid_size) * self.grid_size
            cluster = voxel_grid(pos, size=self.grid_size, batch=batch, start=start)
        else:
            ptr = torch.cat([batch.new_zeros(1), torch.cumsum(batch.bincount(), dim=0)])
            start = segment(pos, ptr, reduce="min")
            cluster = voxel_grid(pos - start[batch], size=self.grid_size, batch=batch, start=0)

        # Scatter straight from the input order: sorting the points into cells first would materialize a copy of
        # the features and cost several times the peak memory for no numerical gain.
        cluster, perm = consecutive_cluster(cluster)
        pos = scatter(pos, cluster, dim=0, reduce="mean")
        x = scatter(x, cluster, dim=0, reduce=self.reduce)
        batch = batch[perm]

        if return_inverse:
            return x, pos, batch, cluster
        return x, pos, batch

    def extra_repr(self) -> str:
        return f"grid_size={self.grid_size}, reduce={self.reduce!r}, origin={self.origin!r}"
