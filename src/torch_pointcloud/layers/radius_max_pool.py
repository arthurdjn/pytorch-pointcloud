"""Farthest point sampling with a max pooling of the features in a ball around every sampled point."""

from typing import Tuple

import torch.nn as nn
from torch import Tensor
from torch_geometric.utils import scatter

from torch_pointcloud.ops.cluster import fps, radius


class RadiusMaxPool(nn.Module):
    r"""Farthest point sampling followed by a max pooling of the features in a ball around every sampled point.

    The ball keeps the `max_num_neighbors` points of smallest index (`radius` with `sort=True`).

    Args:
        num_points: Number of sampled points per sample.
        radius: Ball radius.
        max_num_neighbors: Maximum number of points pooled per ball.

    Shape:
        - Input: features $(N, C)$, positions $(N, d)$ and batch index $(N,)$.
        - Output: $(M, C)$, $(M, d)$ and $(M,)$ with $M = B \cdot \text{num\_points}$.

    Example:
        ```python
        import torch
        from torch_pointcloud.layers import RadiusMaxPool

        pool = RadiusMaxPool(num_points=32, radius=0.5, max_num_neighbors=16)
        pos = torch.randn(256, 3)
        x = torch.randn(256, 8)
        batch = torch.cat([torch.zeros(128), torch.ones(128)]).long()
        x, pos, batch = pool(x, pos, batch)
        print(x.shape, pos.shape)
        ```
    """

    def __init__(self, num_points: int, radius: float, max_num_neighbors: int) -> None:
        super().__init__()
        self.num_points = num_points
        self.radius = radius
        self.max_num_neighbors = max_num_neighbors

    def forward(self, x: Tensor, pos: Tensor, batch: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        idx = fps(pos, batch, num_nodes=self.num_points, random_start=False)
        pos_center, batch_center = pos[idx], batch[idx]
        row, col = radius(
            pos,
            pos_center,
            self.radius,
            batch,
            batch_center,
            max_num_neighbors=self.max_num_neighbors,
            sort=True,
        )
        x = scatter(x.index_select(0, col), row, dim=0, dim_size=pos_center.size(0), reduce="max")
        return x, pos_center, batch_center
