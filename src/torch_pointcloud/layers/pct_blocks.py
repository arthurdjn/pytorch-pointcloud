"""PCT blocks: sampling and grouping with an edge convolution, offset self-attention."""

from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import MLP, EdgeConv

from torch_pointcloud.ops.cluster import fps, knn
from torch_pointcloud.utils.conversion import ensure_list

from .act import create_act
from .norms import create_norm


class PCTNeighborEmbedding(nn.Module):
    r"""Neighbor embedding block from
    :arxiv: [PCT: Point Cloud Transformer](https://arxiv.org/abs/2012.09688).

    Samples the points with farthest point sampling, groups their $k$ nearest neighbors on the coordinates and
    max-pools an `EdgeConv` MLP over $[\mathbf{x}_i, \mathbf{x}_j - \mathbf{x}_i]$ on every sampled point.

    Args:
        in_channels: Number of input channels.
        out_channels: Channels of every MLP layer.
        num_points: Number of sampled points per sample.
        num_neighbors: Number of grouped neighbors $k$.
        act: Activation function.
        act_kwargs: Extra arguments for the activation.
        norm: Normalization layer.
        norm_kwargs: Extra arguments for the normalization.
        bias: Whether the MLP layers use a bias.

    Shape:
        - Input: features $(N, C_\text{in})$, positions $(N, d)$ and batch index $(N,)$.
        - Output: features $(M, C_\text{out})$, positions $(M, d)$ and batch index $(M,)$ with
          $M = B \cdot \text{num\_points}$.

    Example:
        ```python
        import torch
        from torch_pointcloud.layers import PCTNeighborEmbedding

        block = PCTNeighborEmbedding(16, [32, 32], num_points=64, num_neighbors=16)
        pos = torch.randn(256, 3)
        x = torch.randn(256, 16)
        batch = torch.cat([torch.zeros(128), torch.ones(128)]).long()
        x, pos, batch = block(x, pos, batch)
        print(x.shape)
        ```
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: Union[int, Sequence[int]],
        num_points: int,
        num_neighbors: int,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.num_points = num_points
        self.num_neighbors = num_neighbors
        nn_module = MLP(
            [2 * in_channels] + ensure_list(out_channels),
            plain_last=False,
            act=act,
            act_kwargs=act_kwargs,
            norm=norm,
            norm_kwargs=norm_kwargs,
            bias=bias,
        )
        self.conv = EdgeConv(nn_module, aggr="max")

    def forward(self, x: Tensor, pos: Tensor, batch: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        idx = fps(pos, batch, num_nodes=self.num_points, random_start=False)
        pos_center, batch_center = pos[idx], batch[idx]
        row, col = knn(pos, pos_center, self.num_neighbors, batch_x=batch, batch_y=batch_center)
        x = self.conv((x, x.index_select(0, idx)), torch.stack([col, row]))
        return x, pos_center, batch_center


class OffsetAttention(nn.Module):
    r"""Offset self-attention over a dense token sequence, from
    :arxiv: [PCT: Point Cloud Transformer](https://arxiv.org/abs/2012.09688).

    The query and key share one projection. The attention map is a softmax over the keys followed by an $L_1$
    normalization over the queries, and the block refines the tokens with the offset between the input and the
    attended values: $y = x + \phi(x - A^\top V)$ where $\phi$ is a linear layer, normalization and activation.

    Args:
        channels: Token dimension $C$.
        qk_channels: Width of the shared query / key projection. Defaults to $C / 4$.
        act: Activation of the offset projection.
        act_kwargs: Extra arguments for the activation.
        norm: Normalization of the offset projection.
        norm_kwargs: Extra arguments for the normalization.

    Shape:
        - Input: $(B, N, C)$ tokens, the same $N$ for every sample: there is no padding mask.
        - Output: $(B, N, C)$.

    Example:
        ```python
        import torch
        from torch_pointcloud.layers import OffsetAttention

        attn = OffsetAttention(256)
        x = torch.randn(2, 64, 256)
        y = attn(x)
        print(y.shape)
        ```
    """

    def __init__(
        self,
        channels: int,
        qk_channels: Optional[int] = None,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.qk = nn.Linear(channels, qk_channels or channels // 4, bias=False)
        self.v = nn.Linear(channels, channels)
        self.proj = nn.Linear(channels, channels)
        self.norm = create_norm(norm, channels, **(norm_kwargs or {})) or nn.Identity()
        self.act = create_act(act, **(act_kwargs or {})) or nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        q = self.qk(x)  # (B, N, qk_channels)
        attention = (q @ q.transpose(1, 2)).softmax(dim=-1)  # (B, N, N)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdim=True))
        offset = x - attention.transpose(1, 2) @ self.v(x)
        offset = self.act(self.norm(self.proj(offset).flatten(0, 1))).view_as(x)
        return x + offset
