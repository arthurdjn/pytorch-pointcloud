from typing import TYPE_CHECKING, Any, Callable, Dict, Literal, Optional, Sequence, Tuple, Union, overload

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn.pool import knn, voxel_grid
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from torch_geometric.nn.resolver import activation_resolver, normalization_resolver
from torch_geometric.utils import scatter

from torch_pointcloud.utils.imports import optional_import

if TYPE_CHECKING:
    import spconv.pytorch as spconv
    from spconv.pytorc import SparseConvTensor

spconv, _ = optional_import("spconv.pytorch", "spconv")
SparseConvTensor, _ = optional_import("spconv.pytorch", "SparseConvTensor")


def to_spconv_tensor(
    features: Tensor,
    grid_coords: Tensor,
    batch: Tensor,
    spatial_shape: Optional[Sequence[int]] = None,
    padding: int = 96,
) -> SparseConvTensor:
    """Convert point features and coordinates to spconv SparseConvTensor"""
    if spatial_shape is None:
        spatial_shape = torch.add(torch.max(grid_coords, dim=0).values, padding).tolist()

    return spconv.SparseConvTensor(
        features=features,
        indices=torch.cat([batch.unsqueeze(-1).int(), grid_coords.int()], dim=1).contiguous(),
        spatial_shape=spatial_shape,
        batch_size=batch[-1].item() + 1,
    )


class VoxelGridPool(nn.Module):
    def __init__(
        self,
        grid_size: float,
        reduce: str = "mean",
    ):
        super().__init__()
        self.grid_size = grid_size
        self.reduce = reduce

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

    def forward(self, x: Tensor, pos: Tensor, batch: Tensor, return_inverse: bool = False) -> Tuple[Tensor, ...]:
        start = pos.min(0)[0]

        cluster = voxel_grid(pos, batch=batch, size=self.grid_size, start=start)
        cluster, perm = consecutive_cluster(cluster)

        voxel_features = scatter(x, cluster, dim=0, reduce=self.reduce)
        voxel_pos = torch.div(pos[perm] - start, self.grid_size, rounding_mode="trunc").int()
        voxel_batch = batch[perm]

        if return_inverse:
            return voxel_features, voxel_pos, voxel_batch, cluster
        return voxel_features, voxel_pos, voxel_batch


class SparseConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = False,
        indice_key: Optional[str] = None,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        if stride == 1:
            self.conv = spconv.SubMConv3d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=bias,
                indice_key=indice_key,
            )
        else:
            self.conv = spconv.SparseConv3d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
                indice_key=indice_key,
            )

        self.act = activation_resolver(act, **act_kwargs) or nn.Identity()
        self.norm = normalization_resolver(norm, **norm_kwargs) or nn.Identity()

    def forward(self, x: SparseConvTensor) -> SparseConvTensor:
        x = self.conv(x)
        x = x.replace_feature(self.norm(x.features))
        x = x.replace_feature(self.act(x.features))
        return x


class SparseResidualBlock(nn.Module):
    """Sparse residual block with skip connection"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = False,
        indice_key: Optional[str] = None,
    ):
        super().__init__()

        self.conv1 = SparseConvBlock(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias,
            indice_key,
        )
        self.conv2 = SparseConvBlock(
            out_channels,
            out_channels,
            kernel_size,
            1,
            padding,
            bias,
            indice_key,
        )

        # Skip connection
        if in_channels != out_channels or stride != 1:
            self.skip = spconv.SparseConv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=bias)
            self.skip_bn = nn.BatchNorm1d(out_channels)
        else:
            self.skip = None
            self.skip_bn = None

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: SparseConvTensor) -> SparseConvTensor:
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.skip is not None:
            identity = self.skip(identity)
            identity = identity.replace_feature(self.skip_bn(identity.features))

        out = out.replace_feature(out.features + identity.features)
        out = out.replace_feature(self.relu(out.features))

        return out
