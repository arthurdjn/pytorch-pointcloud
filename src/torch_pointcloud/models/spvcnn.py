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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn.resolver import activation_resolver, normalization_resolver

import torch_pointcloud.transforms as T
from torch_pointcloud.layers import PoolLike, create_cls_head, create_pool
from torch_pointcloud.layers.dropouts import DropPath
from torch_pointcloud.models._base import ClassificationModel, SegmentationModel
from torch_pointcloud.utils.conversion import ensure_tuple_size
from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.imports import optional_import

from ._registry import register_model

if TYPE_CHECKING:
    import torch_scatter
    import torchsparse
    import torchsparse.nn as spnn
    import torchsparse.nn.functional as spF
    from torchsparse.tensor import SparseTensor


torch_scatter, _ = optional_import("torch_scatter")
torchsparse, _IS_TORCHSPARSE_AVAILABLE = optional_import("torchsparse")
spnn, _ = optional_import("torchsparse.nn")
spF, _ = optional_import("torchsparse.nn.functional")
SparseTensor, _ = optional_import("torchsparse.tensor", "SparseTensor")

if _IS_TORCHSPARSE_AVAILABLE:

    class PointTensor(SparseTensor):
        """A SparseTensor subclass that caches per-stride point↔voxel mappings.

        This mirrors `mit-han-lab/spvnas` `core/models/utils.py:PointTensor`. The
        voxelisation helpers (`initial_voxelize`, `point_to_voxel`, `voxel_to_point`)
        expect coordinates in **batch-FIRST** layout `[B, X, Y, Z]` (matching torchsparse's
        `SparseTensor.C`).
        """

        def __init__(
            self,
            feats: Tensor,
            coords: Tensor,
            stride: Union[int, Tuple[int, ...]] = 1,
        ) -> None:
            super().__init__(feats=feats, coords=coords, stride=stride)
            self._caches.idx_query = dict()
            self._caches.idx_query_devox = dict()
            self._caches.weights_devox = dict()

else:
    PointTensor = SparseTensor  # type: ignore[misc]


def _sphashquery(query: Tensor, target: Tensor, kernel_size: int = 1) -> Tensor:
    """SPVNAS-style hash lookup that maps each row of `query` to its position in
    `target` (or to `-1` when not found). Both tensors use batch-FIRST coords.
    """
    hashmap_keys = torch.zeros(2 * target.shape[0], dtype=torch.int64, device=target.device)
    hashmap_vals = torch.zeros(2 * target.shape[0], dtype=torch.int32, device=target.device)
    hashmap = torchsparse.backend.GPUHashTable(hashmap_keys, hashmap_vals)
    hashmap.insert_coords(target[:, [1, 2, 3, 0]])
    ks = ensure_tuple_size(kernel_size, 3)
    kernel_volume = int(np.prod(ks))
    ks_tensor = torch.tensor(ks, dtype=torch.int32, device=target.device)
    stride = torch.tensor((1, 1, 1), dtype=torch.int32, device=target.device)
    return (hashmap.lookup_coords(query[:, [1, 2, 3, 0]], ks_tensor, stride, kernel_volume) - 1)[: query.shape[0]]


def initial_voxelize(z: "PointTensor", init_res: float = 1.0, after_res: float = 1.0) -> "SparseTensor":
    """Aggregate a `PointTensor` into a `SparseTensor` of voxel features.

    Mirrors SPVNAS's `core/models/utils.py:initial_voxelize`. Mutates `z.C` to the
    rescaled (voxel-unit) float coordinates so subsequent `voxel_to_point` calls
    can stay in voxel space.
    """
    new_float_coord = torch.cat(
        [z.C[:, 0].view(-1, 1), (z.C[:, 1:] * init_res) / after_res],
        1,
    )
    new_int_coord = torch.floor(new_float_coord).int()
    sparse_coord = torch.unique(new_int_coord, dim=0)
    idx_query = _sphashquery(new_int_coord, sparse_coord).reshape(-1)

    sparse_feat = torch_scatter.scatter_mean(z.F, idx_query.long(), dim=0)
    new_tensor = SparseTensor(sparse_feat, sparse_coord, 1)
    z._caches.idx_query[z.s] = idx_query
    z.C = new_float_coord
    return new_tensor


def point_to_voxel(x: "SparseTensor", z: "PointTensor") -> "SparseTensor":
    """Aggregate point features (`z.F`) onto the voxel grid of `x`."""
    if z._caches.idx_query.get(x.s) is None:
        # x.C has been downsampled by stride x.s[0]; re-query against the new grid.
        new_int_coord = torch.cat(
            [
                z.C[:, 0].int().view(-1, 1),
                torch.floor(z.C[:, 1:] / x.s[0]).int(),
            ],
            1,
        )
        idx_query = _sphashquery(new_int_coord, x.C)
        z._caches.idx_query[x.s] = idx_query
    else:
        idx_query = z._caches.idx_query[x.s]

    # Points whose voxel isn't in `x` get clamped to 0 — they then receive the
    # mean-aggregated feature of voxel 0 (rare, mostly affects boundaries).
    idx_query = idx_query.clamp_(0)
    sparse_feat = torch_scatter.scatter_mean(z.F, idx_query.long(), dim=0)
    new_tensor = SparseTensor(sparse_feat, x.C, x.s)
    new_tensor._caches = x._caches
    return new_tensor


def voxel_to_point(x: "SparseTensor", z: "PointTensor", nearest: bool = False) -> "PointTensor":
    """Trilinearly interpolate voxel features (`x.F`) at point positions (`z.C`)."""
    if z._caches.idx_query_devox.get(x.s) is None or z._caches.weights_devox.get(x.s) is None:
        point_coords_float = torch.cat(
            [z.C[:, 0].int().view(-1, 1), z.C[:, 1:] / x.s[0]],
            1,
        )
        point_coords_int = torch.floor(point_coords_float).int()
        idx_query = _sphashquery(point_coords_int, x.C, kernel_size=2)
        weights = spF.calc_ti_weights(point_coords_float[:, 1:], idx_query, scale=1)

        if nearest:
            weights[:, 1:] = 0.0
            idx_query[:, 1:] = -1

        new_feat = spF.spdevoxelize(x.F, idx_query, weights)
        new_tensor = PointTensor(new_feat, z.C)
        new_tensor._caches = z._caches
        new_tensor._caches.idx_query_devox[x.s] = idx_query
        new_tensor._caches.weights_devox[x.s] = weights
        z._caches.idx_query_devox[x.s] = idx_query
        z._caches.weights_devox[x.s] = weights
    else:
        new_feat = spF.spdevoxelize(
            x.F,
            z._caches.idx_query_devox.get(x.s),
            z._caches.weights_devox.get(x.s),
        )
        new_tensor = PointTensor(new_feat, z.C)
        new_tensor._caches = z._caches

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

    def forward(self, x: "PointTensor") -> "PointTensor":
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

    def forward(self, x: "PointTensor") -> "PointTensor":
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

    def forward(self, x_voxels: "SparseTensor", x_points: "PointTensor") -> Tuple["SparseTensor", "PointTensor"]:
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

    def forward(self, x: "SparseTensor", x_skip: "SparseTensor") -> "SparseTensor":
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
        x_voxels: "SparseTensor",
        x_points: Optional["PointTensor"],
    ) -> Tuple["SparseTensor", Optional["PointTensor"]]:
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
        x_voxels: "SparseTensor",
        x_points: None,
        x_voxels_skip: Optional["SparseTensor"],
    ) -> Tuple["SparseTensor", None]: ...

    @overload
    def forward(
        self,
        x_voxels: "SparseTensor",
        x_points: "PointTensor",
        x_voxels_skip: Optional["SparseTensor"],
    ) -> Tuple["SparseTensor", "PointTensor"]: ...

    def forward(
        self,
        x_voxels: "SparseTensor",
        x_points: Optional["PointTensor"],
        x_voxels_skip: Optional["SparseTensor"],
    ) -> Tuple["SparseTensor", Optional["PointTensor"]]:
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
    x_voxels: "SparseTensor"
    x_points: "PointTensor"


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
        x_voxels: "SparseTensor",
        x_points: "PointTensor",
    ) -> Tuple["SparseTensor", "PointTensor"]: ...

    @overload
    def forward(
        self,
        x_voxels: "SparseTensor",
        x_points: "PointTensor",
        return_intermediates: Literal[True],
    ) -> Tuple["SparseTensor", "PointTensor", List[SPVCNNIntermediateDict]]: ...

    @overload
    def forward(
        self,
        x_voxels: "SparseTensor",
        x_points: "PointTensor",
        return_intermediates: Literal[False],
    ) -> Tuple["SparseTensor", "PointTensor"]: ...

    def forward(
        self,
        x_voxels: "SparseTensor",
        x_points: "PointTensor",
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
        x_voxels: "SparseTensor",
        x_points: "PointTensor",
        intermediates: List[SPVCNNIntermediateDict],
    ) -> Tuple["SparseTensor", "PointTensor"]:
        for i, intermediate in enumerate(reversed(intermediates)):
            block = self.get_submodule(self.block_name.format(i=i))
            x_voxels, x_points = block(x_voxels, x_points, intermediate["x_voxels"])
        return x_voxels, x_points


class SPVCNNClassification(ClassificationModel):
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
        super().__init__(in_channels=in_channels, num_classes=num_classes)
        self.spatial_dim = spatial_dim
        self.embedding_dim = encoder_channels[-1]
        self.dropout = dropout

        self.stem = nn.Sequential(
            BasicBlock(
                self.in_channels or self.spatial_dim,
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


class SPVCNNSegmentation(SegmentationModel):
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
        super().__init__(in_channels=in_channels, num_classes=num_classes)
        self.spatial_dim = spatial_dim
        self.stem_channels = stem_channels
        self.stem = nn.Sequential(
            BasicBlock(
                self.in_channels or self.spatial_dim,
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
        coords = torch.cat([batch.unsqueeze(-1).float(), pos.float()], dim=1).contiguous()
        x_points = PointTensor(x, coords)
        x_voxels = initial_voxelize(x_points)

        x_voxels = self.stem(x_voxels)
        x_points = voxel_to_point(x_voxels, x_points)

        x_voxels, x_points, intermediates = self.encoder(x_voxels, x_points, return_intermediates=True)
        x_voxels, x_points = self.decoder(x_voxels, x_points, intermediates)
        return self.head(x_points.F)


def _spvcnn_semantickitti_transforms() -> Callable:
    return T.Compose(
        [
            # SemanticKITTI 19-class learning_map used by the SPVNAS-trained checkpoints.
            # Each (raw_id -> contiguous_idx) entry follows the convention from
            # https://github.com/mit-han-lab/spvnas/blob/master/core/datasets/semantic_kitti.py:
            #  - the static benchmark classes get indices 0..18;
            #  - moving-* variants are merged into their static counterpart;
            #  - bus, on-rails, lane-marking, other-structure, other-object, ... -> ignore (255).
            T.Relabel(
                keys=DataKeys.SEGMENT,
                labels={
                    10: 0,  # car
                    252: 0,  # moving-car
                    11: 1,  # bicycle
                    15: 2,  # motorcycle
                    18: 3,  # truck
                    258: 3,  # moving-truck
                    20: 4,  # other-vehicle
                    259: 4,  # moving-other-vehicle
                    30: 5,  # person
                    254: 5,  # moving-person
                    31: 6,  # bicyclist
                    253: 6,  # moving-bicyclist
                    32: 7,  # motorcyclist
                    255: 7,  # moving-motorcyclist
                    40: 8,  # road
                    44: 9,  # parking
                    48: 10,  # sidewalk
                    49: 11,  # other-ground
                    50: 12,  # building
                    51: 13,  # fence
                    70: 14,  # vegetation
                    71: 15,  # trunk
                    72: 16,  # terrain
                    80: 17,  # pole
                    81: 18,  # traffic-sign
                },
                default=255,
            ),
            T.Cat(keys=[DataKeys.POS, DataKeys.INTENSITY], dst_key=DataKeys.X, dim=1),
            T.VoxelGrid(
                pos_key=DataKeys.POS,
                pos_reduce="grid",
                keys=[DataKeys.X, DataKeys.SEGMENT],
                reduce=["first", "first"],
                size=0.05,
            ),
        ]
    )


def _spvcnn_semantickitti_hparams(cr: float) -> dict:
    cs = [int(cr * x) for x in [32, 32, 64, 128, 256, 256, 128, 96, 96]]
    return dict(
        in_channels=4,
        num_classes=19,
        spatial_dim=3,
        stem_channels=cs[0],
        encoder_channels=[cs[1], cs[2], cs[3], cs[4]],
        encoder_depths=[2, 2, 2, 2],
        encoder_fusion_stages=[False, False, False, True],
        decoder_channels=[cs[5], cs[6], cs[7], cs[8]],
        decoder_depths=[1, 1, 1, 1],
        decoder_fusion_stages=[False, True, False, True],
        kernel_size=3,
        stride=1,
        dilation=1,
        drop_path=0.0,
        act="relu",
        act_kwargs=None,
        norm="batch_norm",
        norm_kwargs=None,
    )


@register_model(
    "spvcnn-30gmacs.semantickitti",
    task="segmentation",
    weights="hf://torch-pointcloud/spvcnn/spvcnn-30gmacs.semantickitti.pt",
    transforms=_spvcnn_semantickitti_transforms(),
    hparams=_spvcnn_semantickitti_hparams(cr=0.5),
)
def spvcnn_30gmacs_semantickitti_seg(**hparams: Any) -> SPVCNNSegmentation:
    return SPVCNNSegmentation(**hparams)


@register_model(
    "spvcnn-47gmacs.semantickitti",
    task="segmentation",
    weights="hf://torch-pointcloud/spvcnn/spvcnn-47gmacs.semantickitti.pt",
    transforms=_spvcnn_semantickitti_transforms(),
    hparams=_spvcnn_semantickitti_hparams(cr=0.64),
)
def spvcnn_47gmacs_semantickitti_seg(**hparams: Any) -> SPVCNNSegmentation:
    return SPVCNNSegmentation(**hparams)


@register_model(
    "spvcnn-119gmacs.semantickitti",
    task="segmentation",
    weights="hf://torch-pointcloud/spvcnn/spvcnn-119gmacs.semantickitti.pt",
    transforms=_spvcnn_semantickitti_transforms(),
    hparams=_spvcnn_semantickitti_hparams(cr=1.0),
)
def spvcnn_119gmacs_semantickitti_seg(**hparams: Any) -> SPVCNNSegmentation:
    return SPVCNNSegmentation(**hparams)
