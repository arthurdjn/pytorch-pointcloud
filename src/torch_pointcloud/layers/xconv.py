import math
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn.resolver import activation_resolver, normalization_resolver
from torch_geometric.utils import add_self_loops, remove_self_loops

from torch_pointcloud.layers.view import View
from torch_pointcloud.utils.types import OptTensor, PairTensor


class XConv(nn.Module):
    r"""XConv layer as described in the paper
    :arxiv: ["PointCNN: Convolution On X-Transformed Points"](https://arxiv.org/abs/1801.07791)
    by Yangyan Li, Rui Bu, Mingchao Sun, Wei Wu, Xinhan Di, Baoquan Chen.

    This layer is inspired by the PyTorch Geometric `torch_geometric.nn.XConv` layer implementation,
    excepts that this convolution layer supports bipartite graphs and more flexibility / customization.
    This implementation provides full compatibility with the PyTorch Geometric library while following the official
    paper and original tensorflow implementation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial_dim: int,
        kernel_size: int,
        hidden_channels: Optional[int] = None,
        depth_multiplier: Optional[int] = None,
        dilation: int = 1,
        act: Union[str, Callable, None] = "elu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: bool = True,
        add_self_loops: bool = False,
    ):
        super().__init__()
        act_kwargs = act_kwargs or {}
        norm_kwargs = norm_kwargs or {}

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels or self.in_channels // 4
        self.out_channels = out_channels
        self.spatial_dim = spatial_dim
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.add_self_loops = add_self_loops

        self.mlp1 = nn.Sequential(
            nn.Linear(self.spatial_dim, self.hidden_channels),
            activation_resolver(act, **act_kwargs),
            normalization_resolver(norm, self.hidden_channels, **norm_kwargs),
            nn.Linear(self.hidden_channels, self.hidden_channels),
            activation_resolver(act, **act_kwargs),
            normalization_resolver(norm, self.hidden_channels, **norm_kwargs),
            View(-1, self.kernel_size, self.hidden_channels),
        )

        self.mlp2 = nn.Sequential(
            nn.Linear(self.spatial_dim * self.kernel_size, self.kernel_size**2),
            activation_resolver(act, **act_kwargs),
            normalization_resolver(norm, self.kernel_size**2, **norm_kwargs),
            View(-1, self.kernel_size, self.kernel_size),
            nn.Conv1d(self.kernel_size, self.kernel_size**2, self.kernel_size, groups=self.kernel_size),
            activation_resolver(act, **act_kwargs),
            normalization_resolver(norm, self.kernel_size**2, **norm_kwargs),
            View(-1, self.kernel_size, self.kernel_size),
            nn.Conv1d(self.kernel_size, self.kernel_size**2, self.kernel_size, groups=self.kernel_size),
            normalization_resolver(norm, self.kernel_size**2, **norm_kwargs),
            View(-1, self.kernel_size, self.kernel_size),
        )

        in_channels = self.in_channels + self.hidden_channels
        self.depth_multiplier = depth_multiplier or int(math.ceil(self.out_channels / in_channels))

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, in_channels * self.depth_multiplier, self.kernel_size, groups=in_channels),
            View(-1, in_channels * self.depth_multiplier),
            nn.Linear(in_channels * self.depth_multiplier, self.out_channels, bias=bias),
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.apply(self.init_weights_)

    @staticmethod
    def init_weights_(module: nn.Module) -> None:
        if not isinstance(module, (nn.Linear, nn.Conv1d)):
            return

        nn.init.xavier_normal_(module.weight)
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(
        self,
        x: Union[Tensor, Tuple[Tensor, OptTensor]],
        pos: Union[Tensor, PairTensor],
        edge_index: Tensor,
    ) -> Tensor:
        if isinstance(x, Tensor):
            x = (x, None)

        if isinstance(pos, Tensor):
            pos = (pos, pos)

        if self.add_self_loops:
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=min(pos[0].size(0), pos[1].size(0)))
            # TODO: @adu add supports for sparse tensors

        if self.dilation > 1:
            edge_index = edge_index[:, :: self.dilation]

        # This is step is usually done inside the message passing class,
        # however here we do not want to do the final aggregation (not supported by this layer)
        # so we extract source / target nodes manually.
        # IMPORTANT: the flow for the edge_index is expected to be target -> source
        row, col = edge_index
        pos_src, pos_dst = pos
        x_src, _ = x

        pos_j, pos_i = pos_src[col], pos_dst[row]
        x_j = x_src[col] if x_src is not None else None

        # Compute relative neighbor positions (message)
        msg = pos_j - pos_i

        x_star = self.mlp1(msg)

        if x_j is not None:
            x_j = x_j.view(-1, self.kernel_size, self.in_channels)
            x_star = torch.cat([x_star, x_j], dim=-1)

        x_star = x_star.transpose(1, 2).contiguous()
        transform_matrix = self.mlp2(msg.view(-1, self.kernel_size * self.spatial_dim))
        x_transformed = torch.matmul(x_star, transform_matrix)

        return self.conv(x_transformed)

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, hidden_channels={self.hidden_channels}, out_channels={self.out_channels}, "
            f"spatial_dim={self.spatial_dim}, kernel_size={self.kernel_size}, dilation={self.dilation}, "
            f"add_self_loops={self.add_self_loops!r}"
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.extra_repr()})"
