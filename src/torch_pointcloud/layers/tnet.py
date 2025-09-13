from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Sequence, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import MLP

from torch_pointcloud.utils.conversion import ensure_list
from torch_pointcloud.utils.imports import optional_import
from torch_pointcloud.utils.types import AggrType

if TYPE_CHECKING:
    from torch_scatter import scatter

scatter, _ = optional_import("torch_scatter", "scatter")


class TNet(nn.Module):
    """Transformation Network (T-Net) module as described in PointNet paper
    [PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](https://arxiv.org/pdf/1612.00593).

    T-Net predicts an affine transformation matrix that helps align input point clouds
    or feature spaces to a canonical space. This network acts as a mini-PointNet that
    takes points/features as input and outputs a transformation matrix.

    This layer will apply the following transformation to the input:

    $$
        x' = x \cdot T
    $$

    where $T$ is the transformation matrix.

    There are two variants of the T-Net in PointNet:
    1. Spatial transform network (ST-Net): Operates on point coordinates (k=3)
    2. Feature transform network (FT-Net): Operates on point features (k=64 typically)

    Note:
        The transformation matrix is initialized as an identity matrix and
        adds a residual connection to help with optimization stability.

    Args:
        local_channels: Channels of the first MLP, before pooling.
        global_channels: Channels of the second MLP, after pooling.
        k: Dimension of input features to transform.
        act: Activation function to use.
        act_kwargs: Keyword arguments for the activation function.
        act_first: Whether to apply the activation function before the normalization.
        norm: Normalization to use.
        norm_kwargs: Keyword arguments for the normalization.
        bias: Whether to use bias in the linear layers.
        dropout: Dropout rate.
        aggr: Aggregation method to use.
    """

    def __init__(
        self,
        local_channels: Union[int, Sequence[int]],
        global_channels: Union[int, Sequence[int]],
        k: int,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: bool = True,
        dropout: float = 0.0,
        aggr: AggrType = "max",
    ) -> None:
        super().__init__()
        self.k = k
        self.aggr = aggr

        local_channels = [k] + ensure_list(local_channels)
        global_channels = [local_channels[-1]] + ensure_list(global_channels)

        self.local_nn = MLP(local_channels, plain_last=False, act=act, norm=norm, bias=bias, dropout=dropout)
        self.global_nn = MLP(global_channels, plain_last=False, act=act, norm=norm, bias=bias, dropout=dropout)
        self.transform = nn.Linear(global_channels[-1], k * k)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.transform.weight)
        nn.init.eye_(self.transform.bias.view(self.k, self.k))

    def forward(self, x: Tensor, batch: Tensor) -> Tensor:
        """Forward pass of the T-Net.

        Args:
            x: Input tensor of shape $(N, k, *)$ where $N$ is the batch size, $k$ is the dimension of the input features, and $*$ means any number of additional dimensions.
            batch: Batch indices of shape $(N)$ where $N$ is the batch size.

        Returns:
            Transformation matrix of shape $(N, k, k)$ where $N$ is the batch size.
        """

        xt = self.local_nn(x)
        xt = scatter(xt, batch, dim=0, reduce=self.aggr)
        xt = self.global_nn(xt)

        xt = self.transform(xt)
        identity = torch.eye(self.k, dtype=xt.dtype, device=xt.device)
        xt = xt.view(-1, self.k, self.k) + identity

        xt = xt[batch]
        return torch.bmm(x.unsqueeze(1), xt).squeeze(1)
