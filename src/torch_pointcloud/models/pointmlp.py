from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
    overload,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import MLP

from torch_pointcloud.layers import FPS, ActLike, PoolLike, create_pool
from torch_pointcloud.layers.geometric_affine import GeometricAffineConv
from torch_pointcloud.models._registry import register_model
from torch_pointcloud.utils.conversion import ensure_list, ensure_tuple, ensure_tuple_size
from torch_pointcloud.utils.imports import optional_import
from torch_pointcloud.utils.ops import knn_interpolate
from torch_pointcloud.utils.types import OptTensor

from ._base import ClassificationModel, SegmentationModel

if TYPE_CHECKING:
    from torch_cluster import fps, knn, scatter_mean, scatter_std


fps, _ = optional_import("torch_cluster", "fps")
scatter_mean, _ = optional_import("torch_scatter", "scatter_mean")
scatter_std, _ = optional_import("torch_scatter", "scatter_std")
knn, _ = optional_import("torch_cluster", "knn")


class PointMLPIntermediate(NamedTuple):
    x: Tensor
    pos: Tensor
    batch: Tensor


class MLPBNReLU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.bn(self.lin(x)))


class ResPBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)
        self.fc2 = nn.Linear(channels, channels, bias=False)
        self.bn2 = nn.BatchNorm1d(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        x = self.act(self.bn1(self.fc1(x)))
        x = self.bn2(self.fc2(x))
        return self.act(x + identity)


def make_resp_stack(channels: int, num_blocks: int) -> nn.Sequential:
    return nn.Sequential(*[ResPBlock(channels) for _ in range(num_blocks)])


class PointMLPEncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        k: int,
        spatial_dim: int = 3,
        num_pre_blocks: int = 2,
        num_pos_blocks: int = 2,
        normalize: Literal["center", "anchor"] = "center",
        norm: Union[str, Callable, None] = "batch_norm",
        bias: bool = False,
        dropout: float = 0.0,
        act: ActLike = "relu",
        order: str = "lan",
        downsample: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.downsample = downsample
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = k

        pre_mlp = nn.Sequential(
            MLPBNReLU(2 * in_channels + spatial_dim, out_channels),
            make_resp_stack(out_channels, num_pre_blocks),
        )

        self.conv = GeometricAffineConv(
            local_nn=pre_mlp,
            channels=in_channels,
            spatial_dim=spatial_dim,
            normalize=normalize,
            aggr="max",
        )

        self.pos_mlp = make_resp_stack(out_channels, num_pos_blocks)

    def forward(self, x: Tensor, pos: Tensor, batch: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        x_dst, pos_dst, batch_dst = x, pos, batch
        if self.downsample is not None:
            idx = self.downsample(pos, batch)
            x_dst, pos_dst, batch_dst = x[idx], pos[idx], batch[idx]

        row, col = knn(x=pos, y=pos_dst, k=self.k, batch_x=batch, batch_y=batch_dst)
        edge_index = torch.stack([col, row], dim=0)

        x_out = self.conv(
            x=(x, x_dst),
            pos=(pos, pos_dst),
            batch=(batch, batch_dst),
            edge_index=edge_index,
        )

        x_out = self.pos_mlp(x_out)
        return x_out, pos_dst, batch_dst


class PointMLPDecoderBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: Sequence[int],
        k: int,
        act: ActLike = "relu",
        norm: Union[str, Callable, None] = "batch_norm",
        bias: bool = False,
        order: str = "lan",
    ) -> None:
        super().__init__()
        self.k = k
        self.mlp = MLP(
            channels,
            in_channels=in_channels,
            act=act,
            norm=norm,
            bias=bias,
            dropout=None,
            order=order,
        )

    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        batch: Tensor,
        x_skip: OptTensor,
        pos_skip: Tensor,
        batch_skip: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)

        x = self.mlp(x)
        return x, pos_skip, batch_skip


class PointMLPEncoder(nn.Module):
    def __init__(
        self,
        *,
        channels: Sequence[int],
        spatial_dim: int = 3,
        num_neighbors: Union[int, Sequence[int]],
        ratios: Union[float, Sequence[float]],
        num_pre_blocks: Union[int, Sequence[int]] = 2,
        num_pos_blocks: Union[int, Sequence[int]] = 2,
        normalize: Literal["center", "anchor"] = "center",
        act: ActLike = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: bool = True,
        add_self_loops: bool = False,
        aggr: str = "max",
    ):
        super().__init__()
        self.channels = ensure_tuple(channels)
        self.spatial_dim = spatial_dim
        self.act = act

        depth = len(self.channels) - 1
        msg = f"Invalid parameter for {self.__class__.__name__}. Expected `{{param}}` to have length {depth}."
        self.ratios = ensure_tuple_size(ratios, size=depth, extra_msg=msg.format(param="ratios"))
        self.num_neighbors = ensure_tuple_size(num_neighbors, size=depth, extra_msg=msg.format(param="k_neighbors"))
        self.num_pre_blocks = ensure_tuple_size(
            num_pre_blocks,
            size=depth,
            extra_msg=msg.format(param="num_pre_blocks"),
        )
        self.num_pos_blocks = ensure_tuple_size(
            num_pos_blocks,
            size=depth,
            extra_msg=msg.format(param="num_pos_blocks"),
        )

        self.blocks = nn.ModuleList()
        for i in range(depth):
            downsample: Optional[nn.Module] = None
            if self.ratios[i]:
                downsample = FPS(ratio=self.ratios[i])

            block = PointMLPEncoderBlock(
                in_channels=self.channels[i],
                out_channels=self.channels[i + 1],
                k=self.num_neighbors[i],
                num_pre_blocks=self.num_pre_blocks[i],
                num_pos_blocks=self.num_pos_blocks[i],
                spatial_dim=self.spatial_dim,
                act=act,
                # act_kwargs=act_kwargs,
                # act_first=act_first,
                # norm=norm,
                # norm_kwargs=norm_kwargs,
                # bias=bias,
                # add_self_loops=add_self_loops,
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
    ) -> Tuple[Tensor, Tensor, Tensor, List[PointMLPIntermediate]]: ...

    @overload
    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        batch: Tensor,
        return_intermediates: Literal[False] = False,
    ) -> Tuple[Tensor, Tensor, Tensor]: ...

    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        batch: Tensor,
        return_intermediates: bool = False,
    ) -> Any:
        intermediates: List[PointMLPIntermediate] = []
        for block in self.blocks:
            x, pos, batch = block(x, pos, batch)

        if return_intermediates:
            return x, pos, batch, intermediates[::-1]
        return x, pos, batch


class PointMLPDecoder(nn.Module):
    pass


class PointMLPClassification(ClassificationModel):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        *,
        spatial_dim: int = 3,
        channels: Sequence[int],
        num_neighbors: Union[int, Sequence[int]],
        ratios: Union[float, Sequence[float]],
        num_pre_blocks: Union[int, Sequence[int]] = 2,
        num_pos_blocks: Union[int, Sequence[int]] = 2,
        normalize: Literal["center", "anchor"] = "center",
        act: ActLike = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: bool = True,
        add_self_loops: bool = False,
        dropout: float = 0.0,
        global_pool: PoolLike = "max",
    ):
        super().__init__(in_channels=in_channels, num_classes=num_classes)
        self.channels = ensure_list(channels)
        self.spatial_dim = spatial_dim
        self.num_neighbors = num_neighbors
        self.ratios = ratios
        self.num_pre_blocks = num_pre_blocks
        self.num_pos_blocks = num_pos_blocks
        self.normalize = normalize
        self.act = act
        self.act_kwargs = act_kwargs
        self.act_first = act_first
        self.norm = norm
        self.norm_kwargs = norm_kwargs
        self.bias = bias
        self.add_self_loops = add_self_loops
        self.dropout = dropout

        self.stem = self.configure_stem()
        self.encoder = self.configure_encoder()
        self.global_pool = create_pool(global_pool)
        self.head = self.configure_head()

    @property
    def embedding_dim(self) -> int:
        return self.channels[-1]

    def configure_stem(self) -> nn.Module:
        return MLP(
            [self.in_channels, self.channels[0]],
            dropout=0.0,
            act=self.act,
            act_kwargs=self.act_kwargs,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
            plain_last=False,
            bias=self.bias,
        )

    def configure_encoder(self) -> PointMLPEncoder:
        return PointMLPEncoder(
            channels=self.channels,
            spatial_dim=self.spatial_dim,
            num_neighbors=self.num_neighbors,
            ratios=self.ratios,
            num_pre_blocks=self.num_pre_blocks,
            num_pos_blocks=self.num_pos_blocks,
            normalize=self.normalize,
            act=self.act,
            act_kwargs=self.act_kwargs,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
            bias=self.bias,
            add_self_loops=self.add_self_loops,
        )

    def configure_head(self) -> nn.Module:
        return nn.Linear(self.embedding_dim, self.num_classes)

    def reset_classifier(self, num_classes: int, global_pool: PoolLike = "max", **kwargs: Any) -> None:
        self.num_classes = num_classes
        self.global_pool = create_pool(global_pool)
        self.head = self.configure_head()

    @overload
    def forward_features(
        self,
        x: OptTensor,
        pos: Tensor,
        batch: Tensor,
        return_intermediates: Literal[True],
    ) -> Tuple[Tensor, Tensor, Tensor, List[PointMLPIntermediate]]: ...

    @overload
    def forward_features(
        self,
        x: OptTensor,
        pos: Tensor,
        batch: Tensor,
        return_intermediates: Literal[False] = False,
    ) -> Tuple[Tensor, Tensor, Tensor]: ...

    def forward_features(
        self,
        x: OptTensor,
        pos: Tensor,
        batch: Tensor,
        return_intermediates: bool = False,
    ) -> Any:
        x = x if x is not None else pos
        x = self.stem(x)
        return self.encoder(x, pos, batch, return_intermediates=return_intermediates)

    def forward_head(self, x: Tensor, batch: Tensor, pre_logits: bool = False) -> Tensor:
        x = self.global_pool(x, batch)
        if self.dropout:
            x = F.dropout(x, p=float(self.dropout), training=self.training)
        return x if pre_logits else self.head(x)

    def forward(self, x: OptTensor, pos: Tensor, batch: Tensor) -> Tensor:
        x, _, batch = self.forward_features(x, pos, batch)
        return self.forward_head(x, batch)


class PointMLPSegmentation(SegmentationModel):
    pass


@register_model("pointmlp-base", task="classification")
def pointmlp_base_clf(in_channels: int = 3, num_classes: int = 40, **kwargs: Any) -> PointMLPClassification:
    hparams: Dict[str, Any] = dict(
        in_channels=in_channels,
        num_classes=num_classes,
        spatial_dim=3,
        channels=(64, 128, 256, 512, 1024),
        ratios=(0.5, 0.5, 0.5, 0.5),
        num_neighbors=(24, 24, 24, 24),
        num_pre_blocks=(2, 2, 2, 2),
        num_pos_blocks=(2, 2, 2, 2),
        normalize="anchor",
        # use_pos=False,  # <--- TODO: add support for this option
        act="relu",
        act_first=True,
        norm="batch_norm",
        bias=False,
        dropout=0.0,
        global_pool="mean",
        add_self_loops=False,
    )
    hparams.update(kwargs)
    return PointMLPClassification(**hparams)


@register_model("pointmlp-elite", task="classification")
def pointmlp_elite_clf(in_channels: int = 3, num_classes: int = 40, **kwargs: Any) -> PointMLPClassification:
    hparams: Dict[str, Any] = dict(
        in_channels=in_channels,
        num_classes=num_classes,
        spatial_dim=3,
        channels=(64, 128, 256, 512, 512),
        ratios=(0.5, 0.5, 0.5, 0.5),
        num_neighbors=(24, 24, 24, 24),
        num_pre_blocks=(1, 1, 2, 1),
        num_pos_blocks=(1, 1, 2, 1),
        normalize="anchor",
        # use_pos=False,  # <--- TODO: add support for this option
        act="relu",
        act_first=True,
        norm="batch_norm",
        bias=False,
        dropout=0.0,
        global_pool="mean",
        add_self_loops=False,
    )
    hparams.update(kwargs)
    return PointMLPClassification(**hparams)
