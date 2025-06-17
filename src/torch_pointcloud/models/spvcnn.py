from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union, overload

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import MLP
from torch_geometric.nn.resolver import activation_resolver, normalization_resolver

from torch_pointcloud.layers import PoolLike, create_cls_head, create_pool
from torch_pointcloud.utils.conversion import ensure_tuple, ensure_tuple_size, to_spconv_tensor
from torch_pointcloud.utils.imports import optional_import
from torch_pointcloud.utils.types import OptTensor
from torch_pointcloud.utils.voxelization import sparse_voxelize

if TYPE_CHECKING:
    import spconv.pytorch as spconv
    from spconv.pytorc import SparseConvTensor, SparseModule, SparseSequential


spconv, _ = optional_import("spconv.pytorch")
SparseConvTensor, _ = optional_import("spconv.pytorch", "SparseConvTensor")
SparseModule, _ = optional_import("spconv.pytorch", "SparseModule")
SparseSequential, _ = optional_import("spconv.pytorch", "SparseSequential")


class SPVConv(nn.Module):
    def __init__(
        self,
        voxel_size: float,
        voxel_nn: Union[SparseModule, SparseSequential],
        point_nn: nn.Module,
        fusion_nn: nn.Module,
    ):
        super().__init__()
        self.voxel_size = voxel_size
        self.voxel_nn = voxel_nn
        self.point_nn = point_nn
        self.fusion_nn = fusion_nn

    def forward(self, x: Tensor, pos: Tensor, batch: Tensor) -> Tensor:
        # encode the points in a dense representation
        x_points = self.point_nn(x)

        # encode the points in a sparse representation
        x_voxels, pos_voxels, batch_voxels, cluster = sparse_voxelize(
            x,
            pos,
            batch,
            voxel_size=self.voxel_size,
            reduce="mean",
            return_inverse=True,
        )

        x_sparse = to_spconv_tensor(x_voxels, pos_voxels, batch_voxels)
        x_sparse = self.voxel_nn(x_sparse)
        x_voxels = x_sparse.features[cluster]

        # fuse the two representations
        x_combined = torch.cat([x_points, x_voxels], dim=1)
        x_fused = self.fusion_nn(x_combined)
        return x_fused


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
        act_kwargs = act_kwargs or {}
        norm_kwargs = norm_kwargs or {}

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
        self.norm = normalization_resolver(norm, out_channels, **norm_kwargs) or nn.Identity()

    def forward(self, x: SparseConvTensor) -> SparseConvTensor:
        x = self.conv(x)
        x = x.replace_feature(self.norm(x.features))
        x = x.replace_feature(self.act(x.features))
        return x


class SparseConvResidualBlock(nn.Module):
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
        act_kwargs = act_kwargs or {}
        norm_kwargs = norm_kwargs or {}

        self.conv1 = SparseConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            indice_key=f"{indice_key}.conv1" if indice_key else None,
            act=act,
            act_kwargs=act_kwargs,
            norm=norm,
            norm_kwargs=norm_kwargs,
        )
        self.conv2 = SparseConvBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=bias,
            indice_key=f"{indice_key}.conv2" if indice_key else None,
            act=act,
            act_kwargs=act_kwargs,
            norm=norm,
            norm_kwargs=norm_kwargs,
        )

        self.act = activation_resolver(act, **act_kwargs) or nn.Identity()
        self.skip: Optional[spconv.SparseConv3d] = None
        self.skip_norm: Optional[nn.Module] = None
        if in_channels != out_channels or stride != 1:
            self.skip = spconv.SparseConv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=bias)
            self.skip_norm = normalization_resolver(norm, out_channels, **norm_kwargs) or nn.Identity()

    def forward(self, x: SparseConvTensor) -> SparseConvTensor:
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.skip is not None:
            identity = self.skip(identity)
        if self.skip_norm is not None:
            identity = identity.replace_feature(self.skip_norm(identity.features))

        out = out.replace_feature(out.features + identity.features)
        out = out.replace_feature(self.act(out.features))
        return out


class SPVCNNEncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        voxel_size: float,
        voxel_depth: int = 2,
        point_depth: int = 2,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        indice_key: Optional[str] = None,
        downsample: Optional[SparseConvBlock] = None,
    ):
        super().__init__()
        voxel_depth = max(voxel_depth, 1)
        point_depth = max(point_depth, 1)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.voxel_size = voxel_size
        self.voxel_depth = voxel_depth
        self.point_depth = point_depth
        self.downsample = downsample

        voxel_layers = []
        for i in range(voxel_depth):
            layer = SparseConvResidualBlock(
                in_channels if i == 0 else out_channels,
                out_channels,
                act=act,
                act_kwargs=act_kwargs,
                norm=norm,
                norm_kwargs=norm_kwargs,
                indice_key=f"{indice_key}.voxel{i}" if indice_key else None,
            )
            voxel_layers.append(layer)

        voxel_nn = nn.Sequential(*voxel_layers)

        point_nn = MLP(
            [in_channels, out_channels] + [out_channels] * (point_depth - 1),
            act=act,
            act_kwargs=act_kwargs,
            norm=norm,
            norm_kwargs=norm_kwargs,
            plain_last=False,
        )

        fusion_nn = MLP(
            [out_channels * 2, out_channels],
            act=act,
            act_kwargs=act_kwargs,
            norm=norm,
            norm_kwargs=norm_kwargs,
            plain_last=False,
        )

        self.spvconv = SPVConv(
            voxel_size=voxel_size,
            voxel_nn=voxel_nn,
            point_nn=point_nn,
            fusion_nn=fusion_nn,
        )

    def forward(self, x: Tensor, pos: Tensor, batch: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        if self.downsample is not None:
            x_sparse = to_spconv_tensor(x, pos, batch)
            x_sparse = self.downsample(x_sparse)
            x = x_sparse.features

        x = self.spvconv(x, pos, batch)
        return x, pos, batch


class SPVCNNEncoder(nn.Module):
    def __init__(
        self,
        channels: Sequence[int],
        voxel_sizes: Sequence[float],
        point_depths: Sequence[int],
        voxel_depths: Sequence[int],
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        indice_key: Optional[str] = None,
    ):
        super().__init__()
        channels = ensure_tuple(channels)
        n = len(channels) - 1
        extra_msg = (
            "Invalid encoder length for `{param}`, expected {size}. "
            "HINT: make sure the length of the encoder parameters are compatible with the number of channels."
        )
        point_depths = ensure_tuple_size(point_depths, size=n, extra_msg=extra_msg.format(param="point_depths", size=n))
        voxel_depths = ensure_tuple_size(voxel_depths, size=n, extra_msg=extra_msg.format(param="voxel_depths", size=n))
        voxel_sizes = ensure_tuple_size(voxel_sizes, size=n, extra_msg=extra_msg.format(param="voxel_sizes", size=n))

        self.blocks = nn.ModuleList()
        for i in range(n):
            downsample = SparseConvBlock(
                in_channels=channels[i],
                out_channels=channels[i],
                act=act,
                act_kwargs=act_kwargs,
                norm=norm,
                norm_kwargs=norm_kwargs,
            )

            block = SPVCNNEncoderBlock(
                in_channels=channels[i],
                out_channels=channels[i + 1],
                voxel_size=voxel_sizes[i],
                point_depth=point_depths[i],
                voxel_depth=voxel_depths[i],
                act=act,
                act_kwargs=act_kwargs,
                norm=norm,
                norm_kwargs=norm_kwargs,
                indice_key=f"{indice_key}.block{i}" if indice_key else None,
                downsample=downsample,
            )
            self.blocks.append(block)

    @overload
    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        batch: Tensor,
        return_intermediates: Literal[True],
    ) -> Tuple[Tensor, Tensor, Tensor, List[Dict[str, Tensor]]]: ...

    @overload
    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        batch: Tensor,
        return_intermediates: Literal[False] = False,
    ) -> Tuple[Tensor, Tensor, Tensor]: ...

    def forward(self, x: Tensor, pos: Tensor, batch: Tensor, return_intermediates: bool = False) -> Any:
        intermediates = []
        for block in self.blocks:
            if return_intermediates:
                intermediates.append(x)
            x, pos, batch = block(x, pos, batch)

        if return_intermediates:
            return x, pos, batch, intermediates
        return x, pos, batch


class SPVCNNClassification(nn.Module):
    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        *,
        encoder_channels: Sequence[int],
        encoder_voxel_sizes: Sequence[float],
        encoder_point_depths: Sequence[int],
        encoder_voxel_depths: Sequence[int],
        stem_channels: Optional[int] = None,
        global_pool: PoolLike = "max",
        dropout: float = 0.0,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.encoder_channels = encoder_channels
        self.encoder_voxel_sizes = encoder_voxel_sizes
        self.encoder_point_depths = encoder_point_depths
        self.encoder_voxel_depths = encoder_voxel_depths
        self.stem_channels = stem_channels

        self.stem: Optional[SparseSequential] = None
        if stem_channels is not None:
            self.stem = SparseSequential(
                spconv.SubMConv3d(in_channels=in_channels, out_channels=stem_channels, kernel_size=3),
                normalization_resolver(norm, stem_channels, **norm_kwargs) or nn.Identity(),
                activation_resolver(act, **act_kwargs) or nn.Identity(),
                spconv.SparseConv3d(in_channels=stem_channels, out_channels=stem_channels, kernel_size=3),
                normalization_resolver(norm, stem_channels, **norm_kwargs) or nn.Identity(),
                activation_resolver(act, **act_kwargs) or nn.Identity(),
            )

        in_channels = stem_channels or in_channels
        self.encoder = SPVCNNEncoder(
            channels=[in_channels, *encoder_channels],
            voxel_sizes=encoder_voxel_sizes,
            point_depths=encoder_point_depths,
            voxel_depths=encoder_voxel_depths,
            act=act,
            act_kwargs=act_kwargs,
            norm=norm,
            norm_kwargs=norm_kwargs,
            indice_key="encoder",
        )

        self.global_pool = create_pool(global_pool)
        self.dropout = dropout
        self.head = create_cls_head(num_features=self.embedding_dim, num_classes=self.num_classes)

    @property
    def embedding_dim(self) -> int:
        return self.encoder_channels[-1]

    def reset_classifier(self, num_classes: int, global_pool: PoolLike = "max", **kwargs: Any) -> None:
        self.num_classes = num_classes
        self.global_pool = create_pool(global_pool)
        self.head = create_cls_head(num_features=self.embedding_dim, num_classes=self.num_classes, **kwargs)

    def forward_encoder(
        self,
        x: OptTensor,
        pos: Tensor,
        batch: Tensor,
        return_intermediates: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        # automatically use the position if no features are provided
        x = x if x is not None else pos

        if self.stem is not None:
            x = self.stem(x)

        return self.encoder(x, pos, batch)

    def forward_head(self, x: Tensor, batch: Tensor, pre_logits: bool = False) -> Tensor:
        x = self.global_pool(x, batch)
        if self.dropout:
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x if pre_logits else self.head(x)

    def forward(self, x: OptTensor, pos: Tensor, batch: Tensor) -> Tensor:
        x, _, batch = self.forward_encoder(x, pos, batch)
        return self.forward_head(x, batch)
