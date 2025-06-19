from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union, overload

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import IntTensor, Tensor
from torch_geometric.nn.resolver import activation_resolver, normalization_resolver

from torch_pointcloud.layers import PoolLike, create_cls_head, create_pool
from torch_pointcloud.utils.conversion import (
    ensure_list,
    ensure_tuple,
    ensure_tuple_size,
    packed_to_spconv_tensor,
    spconv_tensor_to_packed,
)
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

        x_sparse = packed_to_spconv_tensor(x_voxels, pos_voxels, batch_voxels)
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


class SPVCNNUpsample(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        indice_key: Optional[str] = None,
    ):
        super().__init__()
        self.upsample = spconv.SparseInverseConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            bias=False,
            indice_key=indice_key,
        )

        self.skip_mlp = nn.Sequential(
            nn.Linear(skip_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, x: SparseConvTensor, x_skip: SparseConvTensor) -> SparseConvTensor:
        x = self.upsample(x)
        x = x.replace_feature(x.features + self.skip_mlp(x_skip.features))
        return x


class SparseEncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int = 2,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        indice_key: Optional[str] = None,
        downsample: Optional[SparseConvBlock] = None,
    ):
        super().__init__()
        depth = max(depth, 1)
        self.downsample = downsample

        self.layers = nn.ModuleList()
        for i in range(depth):
            layer = SparseConvResidualBlock(
                in_channels if i == 0 else out_channels,
                out_channels,
                act=act,
                act_kwargs=act_kwargs,
                norm=norm,
                norm_kwargs=norm_kwargs,
                indice_key=f"{indice_key}.layer{i}" if indice_key else None,
            )
            self.layers.append(layer)

    def forward(self, x: SparseConvTensor) -> SparseConvTensor:
        if self.downsample is not None:
            x = self.downsample(x)

        for layer in self.layers:
            x = layer(x)
        return x


class SPVCNNDecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int = 2,
        kernel_size: int = 3,
        padding: int = 1,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        indice_key: Optional[str] = None,
        upsample: Optional[SPVCNNUpsample] = None,
    ):
        super().__init__()
        depth = max(depth, 1)
        self.upsample = upsample

        self.layers = nn.ModuleList()
        for i in range(depth):
            layer = SparseConvResidualBlock(
                in_channels if i == 0 else out_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                act=act,
                act_kwargs=act_kwargs,
                norm=norm,
                norm_kwargs=norm_kwargs,
                indice_key=f"{indice_key}.layer{i}" if indice_key else None,
            )
            self.layers.append(layer)

    def forward(self, x: SparseConvTensor, x_skip: SparseConvTensor) -> SparseConvTensor:
        if self.upsample is not None:
            x = self.upsample(x, x_skip)

        x = x.replace_feature(torch.cat([x.features, x_skip.features], dim=1))
        for layer in self.layers:
            x = layer(x)
        return x


class SPVCNNEncoder(nn.Module):
    def __init__(
        self,
        channels: Sequence[int],
        depths: Sequence[int],
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
        depths = ensure_tuple_size(depths, size=n, extra_msg=extra_msg.format(param="depths", size=n))

        self.blocks = nn.ModuleList()
        for i in range(n):
            downsample = SparseConvBlock(
                in_channels=channels[i],
                out_channels=channels[i],
                kernel_size=2,
                stride=2,
                act=act,
                act_kwargs=act_kwargs,
                norm=norm,
                norm_kwargs=norm_kwargs,
                indice_key=f"{indice_key}.downsample{i}" if indice_key else None,
            )

            block = SparseEncoderBlock(
                in_channels=channels[i],
                out_channels=channels[i + 1],
                depth=depths[i],
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
        self, x: SparseConvTensor, return_intermediates: Literal[True]
    ) -> Tuple[SparseConvTensor, List[SparseConvTensor]]: ...

    @overload
    def forward(self, x: SparseConvTensor, return_intermediates: Literal[False] = False) -> SparseConvTensor: ...

    def forward(self, x: SparseConvTensor, return_intermediates: bool = False) -> Any:
        intermediates: List[SparseConvTensor] = []

        for block in self.blocks:
            if return_intermediates:
                intermediates.append(x)
            x = block(x)

        if return_intermediates:
            return x, intermediates
        return x


class SPVCNNDecoder(nn.Module):
    def __init__(
        self,
        depths: Sequence[int],
        channels: Sequence[int],
        skip_channels: Sequence[int],
        act: Union[str, Callable, None] = "relu",
        norm: Union[str, Callable, None] = "batch_norm",
        indice_key: Optional[str] = None,
        upsample_indice_key: Optional[str] = None,
    ):
        super().__init__()
        depths = ensure_tuple(depths)
        n = len(depths)
        channels = ensure_tuple_size(channels, size=n + 1)
        skip_channels = ensure_tuple_size(skip_channels, size=n)

        self.blocks = nn.ModuleList()
        for i in range(n):
            upsample = SPVCNNUpsample(
                in_channels=channels[i],
                out_channels=channels[i + 1],
                skip_channels=skip_channels[i],
                kernel_size=2,
                indice_key=f"{upsample_indice_key}.downsample{n - i - 1}" if upsample_indice_key else None,
                act=act,
                norm=norm,
            )
            block = SPVCNNDecoderBlock(
                in_channels=channels[i + 1] + skip_channels[i],
                out_channels=channels[i + 1],
                depth=depths[i],
                act=act,
                norm=norm,
                indice_key=f"{indice_key}.block{i}",
                upsample=upsample,
            )
            self.blocks.append(block)

    def forward(self, x: SparseConvTensor, intermediates: List[SparseConvTensor]) -> SparseConvTensor:
        for block, intermediate in zip(self.blocks, reversed(intermediates)):
            x = block(x, intermediate)
        return x


class SPVCNNClassification(nn.Module):
    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        *,
        encoder_channels: Sequence[int],
        encoder_depths: Sequence[int],
        stem_channels: Optional[int] = None,
        global_pool: PoolLike = "max",
        dropout: float = 0.0,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        act_kwargs = act_kwargs or {}
        norm_kwargs = norm_kwargs or {}

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.encoder_channels = ensure_list(encoder_channels)
        self.encoder_depths = ensure_list(encoder_depths)
        self.stem_channels = stem_channels

        self.stem: Optional[SparseSequential] = None
        if stem_channels is not None:
            self.stem = SparseSequential(
                spconv.SubMConv3d(in_channels=in_channels, out_channels=stem_channels, kernel_size=3),
                normalization_resolver(norm, stem_channels, **norm_kwargs) or nn.Identity(),
                activation_resolver(act, **act_kwargs) or nn.Identity(),
                spconv.SubMConv3d(in_channels=stem_channels, out_channels=stem_channels, kernel_size=3),
                normalization_resolver(norm, stem_channels, **norm_kwargs) or nn.Identity(),
                activation_resolver(act, **act_kwargs) or nn.Identity(),
            )

        in_channels = stem_channels or in_channels
        self.encoder = SPVCNNEncoder(
            channels=[in_channels, *encoder_channels],
            depths=encoder_depths,
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
        pos: IntTensor,
        batch: Tensor,
        return_intermediates: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        # automatically use the position if no features are provided
        x = x if x is not None else pos.float()

        x_sparse = packed_to_spconv_tensor(x, pos, batch)
        if self.stem is not None:
            x_sparse = self.stem(x_sparse)

        x_sparse = self.encoder(x_sparse)
        x, pos, batch = spconv_tensor_to_packed(x_sparse)
        return x, pos, batch

    def forward_head(self, x: Tensor, batch: Tensor, pre_logits: bool = False) -> Tensor:
        x = self.global_pool(x, batch)
        if self.dropout:
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x if pre_logits else self.head(x)

    def forward(self, x: OptTensor, pos: IntTensor, batch: Tensor) -> Tensor:
        x, _, batch = self.forward_encoder(x, pos, batch)
        return self.forward_head(x, batch)


class SPVCNNSegmentation(nn.Module):
    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        *,
        encoder_channels: Sequence[int] = (32, 64, 128, 256),
        encoder_depths: Sequence[int] = (2, 2, 2, 2),
        decoder_channels: Sequence[int] = (256, 128, 96, 96),
        decoder_depths: Sequence[int] = (2, 2, 2, 2),
        stem_channels: int = 32,
    ):
        super().__init__()
        self.stem = SparseConvBlock(in_channels, stem_channels, indice_key="stem")

        self.encoder = SPVCNNEncoder(
            channels=[stem_channels, *encoder_channels],
            depths=encoder_depths,
            act="relu",
            norm="batch_norm",
            indice_key="encoder",
        )

        self.decoder = SPVCNNDecoder(
            channels=[encoder_channels[-1]] + list(decoder_channels),
            skip_channels=list(encoder_channels[:-1])[::-1] + [stem_channels],
            depths=decoder_depths,
            act="relu",
            norm="batch_norm",
            indice_key="decoder",
            upsample_indice_key="encoder",
        )

        self.head = nn.Linear(decoder_channels[-1], num_classes)

    def forward(self, x: OptTensor, pos: IntTensor, batch: Tensor) -> Tensor:
        x = x if x is not None else pos.float()
        x_sparse = packed_to_spconv_tensor(x, pos, batch)

        x_sparse = self.stem(x_sparse)
        x_sparse, intermediates = self.encoder(x_sparse, return_intermediates=True)
        x_sparse = self.decoder(x_sparse, intermediates)
        x, _, _ = spconv_tensor_to_packed(x_sparse)
        return self.head(x)
