from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    TypedDict,
    Union,
    overload,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn.resolver import activation_resolver, normalization_resolver

from torch_pointcloud.layers import PoolLike, create_cls_head, create_pool
from torch_pointcloud.layers.dropouts import DropPath
from torch_pointcloud.utils.conversion import ensure_tuple_size
from torch_pointcloud.utils.imports import optional_import

if TYPE_CHECKING:
    import torchsparse
    import torchsparse.nn as spnn
    import torchsparse.nn.functional as spF
    from torchsparse.tensor import PointTensor, SparseTensor


torchsparse, _ = optional_import("torchsparse")
spnn, _ = optional_import("torchsparse.nn")
spF, _ = optional_import("torchsparse.nn.functional")
PointTensor, _ = optional_import("torchsparse.tensor", "PointTensor")
SparseTensor, _ = optional_import("torchsparse.tensor", "SparseTensor")


def initial_voxelize(x_points: PointTensor) -> SparseTensor:
    pc_hash = spF.sphash(torch.floor(x_points.C).int())
    sparse_hash = torch.unique(pc_hash)
    idx_query = spF.sphashquery(pc_hash, sparse_hash)
    counts = spF.spcount(idx_query.int(), len(sparse_hash))

    inserted_coords = spF.spvoxelize(torch.floor(x_points.C), idx_query, counts)
    inserted_coords = torch.round(inserted_coords).int()
    inserted_feat = spF.spvoxelize(x_points.F, idx_query, counts)

    new_tensor = SparseTensor(inserted_feat, inserted_coords, 1)
    new_tensor._caches.cmaps.setdefault(new_tensor.stride, new_tensor.coords)
    x_points.additional_features["idx_query"][1] = idx_query
    x_points.additional_features["counts"][1] = counts
    return new_tensor


def point_to_voxel(x_voxels: SparseTensor, x_points: PointTensor) -> SparseTensor:
    if (
        x_points.additional_features is None
        or x_points.additional_features.get("idx_query") is None
        or x_points.additional_features["idx_query"].get(x_voxels.s) is None
    ):
        pc_hash = spF.sphash(
            torch.cat(
                [
                    torch.floor(x_points.C[:, :3] / x_voxels.s[0]).int() * x_voxels.s[0],
                    x_points.C[:, -1].int().view(-1, 1),
                ],
                1,
            )
        )
        sparse_hash = spF.sphash(x_voxels.C)
        idx_query = spF.sphashquery(pc_hash, sparse_hash)
        counts = spF.spcount(idx_query.int(), x_voxels.C.shape[0])
        x_points.additional_features["idx_query"][x_voxels.s] = idx_query
        x_points.additional_features["counts"][x_voxels.s] = counts
    else:
        idx_query = x_points.additional_features["idx_query"][x_voxels.s]
        counts = x_points.additional_features["counts"][x_voxels.s]

    inserted_feat = spF.spvoxelize(x_points.F, idx_query, counts)
    new_tensor = SparseTensor(inserted_feat, x_voxels.C, x_voxels.s)
    new_tensor._caches.cmaps = x_voxels._caches.cmaps
    new_tensor._caches.kmaps = x_voxels._caches.kmaps

    return new_tensor


def voxel_to_point(x_voxels: SparseTensor, x_points: PointTensor) -> PointTensor:
    if (
        x_points.idx_query is None
        or x_points.weights is None
        or x_points.idx_query.get(x_voxels.s) is None
        or x_points.weights.get(x_voxels.s) is None
    ):
        off = spnn.utils.get_kernel_offsets(2, x_voxels.s, 1, device=x_points.F.device)
        old_hash = spF.sphash(
            torch.cat(
                [
                    torch.floor(x_points.C[:, :3] / x_voxels.s[0]).int() * x_voxels.s[0],
                    x_points.C[:, -1].int().view(-1, 1),
                ],
                1,
            ),
            off,
        )
        pc_hash = spF.sphash(x_voxels.C.to(x_points.F.device))
        idx_query = spF.sphashquery(old_hash, pc_hash)
        weights = spF.calc_ti_weights(x_points.C, idx_query.transpose(0, 1), scale=x_voxels.s[0]).contiguous()
        idx_query = idx_query.contiguous().transpose(0, 1)

        new_feat = spF.spdevoxelize(x_voxels.F, idx_query, weights)
        new_tensor = PointTensor(new_feat, x_points.C, idx_query=x_points.idx_query, weights=x_points.weights)
        new_tensor.additional_features = x_points.additional_features
        new_tensor.idx_query[x_voxels.s] = idx_query
        new_tensor.weights[x_voxels.s] = weights
        x_points.idx_query[x_voxels.s] = idx_query
        x_points.weights[x_voxels.s] = weights

    else:
        new_feat = spF.spdevoxelize(x_voxels.F, x_points.idx_query.get(x_voxels.s), x_points.weights.get(x_voxels.s))
        new_tensor = PointTensor(new_feat, x_points.C, idx_query=x_points.idx_query, weights=x_points.weights)
        new_tensor.additional_features = x_points.additional_features

    return new_tensor


class BasicBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        transposed: bool = False,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        act_kwargs = act_kwargs or {}
        norm_kwargs = norm_kwargs or {}

        self.conv = spnn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            stride=stride,
            transposed=transposed,
        )
        self.norm = normalization_resolver(norm, out_channels, **norm_kwargs)
        self.act = activation_resolver(act, **act_kwargs)

    def forward(self, x: PointTensor) -> PointTensor:
        x = self.conv(x)
        x.F = self.act(self.norm(x.F))
        return x


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        drop_path: float = 0.0,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        act_kwargs = act_kwargs or {}
        norm_kwargs = norm_kwargs or {}

        self.conv1 = spnn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation, stride=stride)
        self.norm1 = normalization_resolver(norm, out_channels, **norm_kwargs)
        self.conv2 = spnn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, dilation=dilation, stride=1)
        self.norm2 = normalization_resolver(norm, out_channels, **norm_kwargs)

        self.conv_skip: Optional[nn.Module] = None
        self.norm_skip: Optional[nn.Module] = None
        if in_channels != out_channels or stride != 1:
            self.conv_skip = spnn.Conv3d(in_channels, out_channels, kernel_size=1, dilation=1, stride=stride)
            self.norm_skip = normalization_resolver(norm, out_channels, **norm_kwargs)

        self.act = activation_resolver(act, **act_kwargs)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else None

    def forward(self, x: PointTensor) -> PointTensor:
        x_skip = x
        x = self.conv1(x)
        x.F = self.act(self.norm1(x.F))
        x = self.conv2(x)
        x.F = self.norm2(x.F)

        if self.conv_skip is not None:
            x_skip = self.conv_skip(x_skip)
        if self.norm_skip is not None:
            x_skip.F = self.norm_skip(x_skip.F)
        if self.drop_path is not None:
            x_skip.F = self.drop_path(x_skip.F)

        x.F = self.act(x.F + x_skip.F)
        return x


class SPVFusionBlock(nn.Module):
    def __init__(
        self,
        in_channels: Optional[int],
        out_channels: int,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        act_kwargs = act_kwargs or {}
        norm_kwargs = norm_kwargs or {}

        self.lin = nn.LazyLinear(out_channels) if not in_channels else nn.Linear(in_channels, out_channels)
        self.norm = normalization_resolver(norm, out_channels, **norm_kwargs)
        self.act = activation_resolver(act, **act_kwargs)

    def forward(self, x_voxels: SparseTensor, x_points: PointTensor) -> Tuple[SparseTensor, PointTensor]:
        # NOTE: In the original paper, the fusion is done with a simple addition
        # between the voxel and point features. However, concatenating the features
        # and passing them through a MLP achieves better performance.
        x_points_out = voxel_to_point(x_voxels, x_points)
        x_points_out.F = x_points_out.F + self.act(self.norm(self.lin(x_points.F)))
        x_voxels_out = point_to_voxel(x_voxels, x_points_out)
        return x_voxels_out, x_points_out


class SPVCNNUpsampleBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.conv = BasicBlock(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
            transposed=True,
            act=act,
            act_kwargs=act_kwargs,
            norm=norm,
            norm_kwargs=norm_kwargs,
        )
        self.residual = ResidualBlock(
            out_channels + skip_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            act=act,
            act_kwargs=act_kwargs,
            norm=norm,
            norm_kwargs=norm_kwargs,
        )

    def forward(self, x: SparseTensor, x_skip: SparseTensor) -> SparseTensor:
        x = self.conv(x)
        x = self.residual(torchsparse.cat([x, x_skip]))
        return x


class SPVCNNEncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        drop_path: Union[float, Sequence[float]] = 0.0,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        fusion: Optional[nn.Module] = None,
        downsample: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.downsample = downsample
        self.fusion = fusion
        drop_path = ensure_tuple_size(drop_path, size=depth)

        self.blocks = nn.ModuleList()
        for i in range(depth):
            in_channels = in_channels if i == 0 else out_channels
            block = ResidualBlock(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                drop_path=drop_path[i],
                act=act,
                act_kwargs=act_kwargs,
                norm=norm,
                norm_kwargs=norm_kwargs,
            )
            self.blocks.append(block)

    def forward(
        self,
        x_voxels: SparseTensor,
        x_points: Optional[PointTensor],
    ) -> Tuple[SparseTensor, Optional[PointTensor]]:
        if self.downsample is not None:
            x_voxels = self.downsample(x_voxels)

        for block in self.blocks:
            x_voxels = block(x_voxels)

        if self.fusion:
            if x_points is None:
                raise ValueError("`x_points` is required when `fusion` is not None, but got None")

            x_voxels, x_points = self.fusion(x_voxels, x_points)

        return x_voxels, x_points


class SPVCNNDecoderBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        depth: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        dropout: float = 0.0,
        fusion: Optional[nn.Module] = None,
        upsample: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.upsample = upsample
        self.fusion = fusion
        self.dropout = dropout

        self.blocks = nn.ModuleList()
        for _ in range(depth):
            block = ResidualBlock(channels, channels, kernel_size=kernel_size, stride=stride, dilation=dilation)
            self.blocks.append(block)

    @overload
    def forward(
        self,
        x_voxels: SparseTensor,
        x_points: None,
        x_voxels_skip: Optional[SparseTensor],
    ) -> Tuple[SparseTensor, None]: ...

    @overload
    def forward(
        self,
        x_voxels: SparseTensor,
        x_points: PointTensor,
        x_voxels_skip: Optional[SparseTensor],
    ) -> Tuple[SparseTensor, PointTensor]: ...

    def forward(
        self,
        x_voxels: SparseTensor,
        x_points: Optional[PointTensor],
        x_voxels_skip: Optional[SparseTensor],
    ) -> Tuple[SparseTensor, Optional[PointTensor]]:
        if self.upsample is not None:
            if x_voxels_skip is None:
                raise ValueError("`x_voxels_skip` is required when `upsample` is not None, but got None")

            x_voxels = self.upsample(x_voxels, x_voxels_skip)

        for block in self.blocks:
            x_voxels = block(x_voxels)

        if self.fusion:
            if x_points is None:
                raise ValueError("`x_points` is required when `fusion` is not None, but got None")

            x_voxels, x_points = self.fusion(x_voxels, x_points)

        if self.dropout:
            x_voxels.F = torch.nn.functional.dropout(x_voxels.F, p=self.dropout, training=self.training)
        return x_voxels, x_points


class SPVCNNIntermediateDict(TypedDict):
    x_voxels: SparseTensor
    x_points: PointTensor


class SPVCNNEncoder(nn.Module):
    block_name = "block{i}"

    def __init__(
        self,
        channels: Sequence[int],
        depths: Sequence[int],
        fusion_stages: Sequence[bool],
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        drop_path: float = 0.3,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.num_blocks = len(depths)
        assert len(depths) == len(fusion_stages), f"{len(depths) = }, {len(fusion_stages) = }"
        assert len(channels) == self.num_blocks + 1, f"{len(channels) = }, {self.num_blocks + 1 = }"
        drop_paths = torch.split(torch.linspace(0, drop_path, sum(depths)), list(depths))

        for i in range(self.num_blocks):
            downsample = BasicBlock(
                channels[i],
                channels[i],
                kernel_size=2,
                stride=2,
                dilation=1,
                act=act,
                act_kwargs=act_kwargs,
                norm=norm,
                norm_kwargs=norm_kwargs,
            )

            fusion = None
            if fusion_stages[i]:
                fusion = SPVFusionBlock(
                    in_channels=None,
                    out_channels=channels[i + 1],
                    act=act,
                    act_kwargs=act_kwargs,
                    norm=norm,
                    norm_kwargs=norm_kwargs,
                )

            block = SPVCNNEncoderBlock(
                in_channels=channels[i],
                out_channels=channels[i + 1],
                depth=depths[i],
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                downsample=downsample,
                fusion=fusion,
                drop_path=drop_paths[i].tolist(),
                act=act,
                act_kwargs=act_kwargs,
                norm=norm,
                norm_kwargs=norm_kwargs,
            )
            self.add_module(self.block_name.format(i=i), block)

    @overload
    def forward(
        self,
        x_voxels: SparseTensor,
        x_points: PointTensor,
    ) -> Tuple[SparseTensor, PointTensor]: ...

    @overload
    def forward(
        self,
        x_voxels: SparseTensor,
        x_points: PointTensor,
        return_intermediates: Literal[True],
    ) -> Tuple[SparseTensor, PointTensor, List[SPVCNNIntermediateDict]]: ...

    @overload
    def forward(
        self,
        x_voxels: SparseTensor,
        x_points: PointTensor,
        return_intermediates: Literal[False],
    ) -> Tuple[SparseTensor, PointTensor]: ...

    def forward(
        self,
        x_voxels: SparseTensor,
        x_points: PointTensor,
        return_intermediates: bool = False,
    ) -> Any:
        intermediates: List[SPVCNNIntermediateDict] = []
        for i in range(self.num_blocks):
            block = self.get_submodule(self.block_name.format(i=i))
            if return_intermediates:
                intermediates.append({"x_voxels": x_voxels, "x_points": x_points})
            x_voxels, x_points = block(x_voxels, x_points)

        if return_intermediates:
            return x_voxels, x_points, intermediates
        return x_voxels, x_points


class SPVCNNDecoder(nn.Module):
    block_name = "block{i}"

    def __init__(
        self,
        depths: Sequence[int],
        channels: Sequence[int],
        skip_channels: Sequence[int],
        fusion_stages: Sequence[bool],
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        dropout: float = 0.0,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.num_blocks = len(depths)
        assert len(depths) == len(skip_channels) == len(fusion_stages)
        assert len(channels) == self.num_blocks + 1

        for i in range(self.num_blocks):
            upsample = SPVCNNUpsampleBlock(
                in_channels=channels[i],
                out_channels=channels[i + 1],
                skip_channels=skip_channels[i],
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                act=act,
                act_kwargs=act_kwargs,
                norm=norm,
                norm_kwargs=norm_kwargs,
            )

            fusion = None
            if fusion_stages[i]:
                fusion = SPVFusionBlock(
                    in_channels=None,
                    out_channels=channels[i + 1],
                    act=act,
                    act_kwargs=act_kwargs,
                    norm=norm,
                    norm_kwargs=norm_kwargs,
                )

            block = SPVCNNDecoderBlock(
                channels=channels[i + 1],
                depth=depths[i],
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                fusion=fusion,
                upsample=upsample,
            )
            self.add_module(self.block_name.format(i=i), block)

    def forward(
        self,
        x_voxels: SparseTensor,
        x_points: PointTensor,
        intermediates: List[SPVCNNIntermediateDict],
    ) -> Tuple[SparseTensor, PointTensor]:
        for i, intermediate in enumerate(reversed(intermediates)):
            block = self.get_submodule(self.block_name.format(i=i))
            x_voxels, x_points = block(x_voxels, x_points, intermediate["x_voxels"])
        return x_voxels, x_points


class SPVCNNClassification(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        *,
        spatial_dim: int = 3,
        stem_channels: int = 32,
        encoder_channels: Sequence[int] = (32, 64, 128, 256, 256),
        encoder_depths: Sequence[int] = (2, 2, 2, 2, 2),
        encoder_fusion_stages: Sequence[bool] = (False, False, False, True),
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        drop_path: float = 0.3,
        global_pool: PoolLike = "max",
        dropout: float = 0.0,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.in_channels = in_channels or spatial_dim
        self.num_classes = num_classes
        self.embedding_dim = encoder_channels[-1]
        self.dropout = dropout

        self.stem = nn.Sequential(
            BasicBlock(
                self.in_channels,
                stem_channels,
                kernel_size=3,
                stride=1,
                dilation=1,
                act=act,
                act_kwargs=act_kwargs,
                norm=norm,
                norm_kwargs=norm_kwargs,
            ),
            BasicBlock(
                stem_channels,
                stem_channels,
                kernel_size=3,
                stride=1,
                dilation=1,
                act=act,
                act_kwargs=act_kwargs,
                norm=norm,
                norm_kwargs=norm_kwargs,
            ),
        )

        self.encoder = SPVCNNEncoder(
            channels=[stem_channels, *encoder_channels],
            depths=encoder_depths,
            fusion_stages=encoder_fusion_stages,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            drop_path=drop_path,
            act=act,
            act_kwargs=act_kwargs,
            norm=norm,
            norm_kwargs=norm_kwargs,
        )

        self.global_pool = create_pool(global_pool)
        self.head = create_cls_head(num_features=self.embedding_dim, num_classes=self.num_classes)

    def reset_classifier(self, num_classes: int, global_pool: PoolLike = "max", **kwargs: Any) -> None:
        self.num_classes = num_classes
        self.global_pool = create_pool(global_pool)
        self.head = create_cls_head(num_features=self.embedding_dim, num_classes=self.num_classes, **kwargs)

    def forward(
        self,
        x: Optional[Tensor],
        pos: Tensor,
        batch: Tensor,
    ) -> Tensor:
        x = pos.float() if x is None else x
        pos = torch.cat([pos.float(), batch.unsqueeze(-1).float()], dim=1).contiguous()
        x_points = PointTensor(x, pos)
        x_voxels = initial_voxelize(x_points)

        x_voxels = self.stem(x_voxels)
        x_points = voxel_to_point(x_voxels, x_points)

        x_voxels, x_points = self.encoder(x_voxels, x_points)

        x = self.global_pool(x_points.F, batch)
        if self.dropout:
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.head(x)


class SPVCNNSegmentation(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        *,
        spatial_dim: int = 3,
        stem_channels: int = 32,
        encoder_channels: Sequence[int],
        encoder_depths: Sequence[int],
        encoder_fusion_stages: Sequence[bool],
        decoder_channels: Sequence[int],
        decoder_depths: Sequence[int],
        decoder_fusion_stages: Sequence[bool],
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        drop_path: float = 0.3,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels or spatial_dim
        self.num_classes = num_classes
        self.stem_channels = stem_channels

        self.stem = nn.Sequential(
            BasicBlock(
                self.in_channels,
                stem_channels,
                kernel_size=3,
                stride=1,
                dilation=1,
                act=act,
                act_kwargs=act_kwargs,
                norm=norm,
                norm_kwargs=norm_kwargs,
            ),
            BasicBlock(
                stem_channels,
                stem_channels,
                kernel_size=3,
                stride=1,
                dilation=1,
                act=act,
                act_kwargs=act_kwargs,
                norm=norm,
                norm_kwargs=norm_kwargs,
            ),
        )

        self.encoder = SPVCNNEncoder(
            channels=[stem_channels, *encoder_channels],
            depths=encoder_depths,
            fusion_stages=encoder_fusion_stages,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            drop_path=drop_path,
            act=act,
            act_kwargs=act_kwargs,
            norm=norm,
            norm_kwargs=norm_kwargs,
        )

        self.decoder = SPVCNNDecoder(
            depths=decoder_depths,
            channels=[encoder_channels[-1], *decoder_channels],
            skip_channels=[*reversed(encoder_channels[:-1]), stem_channels],
            fusion_stages=decoder_fusion_stages,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            act=act,
            act_kwargs=act_kwargs,
            norm=norm,
            norm_kwargs=norm_kwargs,
        )

        self.head = create_cls_head(num_features=decoder_channels[-1], num_classes=num_classes)

    def forward(
        self,
        x: Optional[Tensor],
        pos: Tensor,
        batch: Tensor,
    ) -> Tensor:
        x = pos.float() if x is None else x
        pos = torch.cat([pos.float(), batch.unsqueeze(-1).float()], dim=1).contiguous()
        x_points = PointTensor(x, pos)
        x_voxels = initial_voxelize(x_points)

        x_voxels = self.stem(x_voxels)
        x_points = voxel_to_point(x_voxels, x_points)

        x_voxels, x_points, intermediates = self.encoder(x_voxels, x_points, return_intermediates=True)
        x_voxels, x_points = self.decoder(x_voxels, x_points, intermediates)
        return self.head(x_points.F)
