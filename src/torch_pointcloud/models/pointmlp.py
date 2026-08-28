"""PointMLP classification and segmentation models.

{{ paper("2202.07123") }}
"""

from typing import (
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
from torch_geometric.nn.inits import reset

import torch_pointcloud.transforms as T
from torch_pointcloud.datasets.modelnet import MODELNET40_CLASSES
from torch_pointcloud.datasets.scanobjectnn import SCANOBJECTNN_CLASSES
from torch_pointcloud.layers import FPS, LinearBlock, PoolLike, create_pool
from torch_pointcloud.layers.act import create_act
from torch_pointcloud.layers.geometric_affine import GeometricAffineConv
from torch_pointcloud.layers.norms import create_norm
from torch_pointcloud.models._registry import WeightsDict, register_model
from torch_pointcloud.utils.cluster import knn
from torch_pointcloud.utils.conversion import ensure_list, ensure_list_size, ensure_tuple, ensure_tuple_size
from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.ops import knn_interpolate
from torch_pointcloud.utils.types import OptTensor

from ._base import ClassificationModel, SegmentationModel


class PointMLPIntermediate(NamedTuple):
    """Input features and point cloud of one encoder block, recorded before it downsamples."""

    x: Tensor
    pos: Tensor
    batch: Tensor


class ResidualLinearBlock(nn.Module):
    r"""A residual linear block consisting of two linear layers, normalization and activation.

    The default flow is:

    ```text
    x -> Lin1 -> Norm1 -> Act -> Lin2 -> Norm2 -> Act -> y
    |                                          ^
    +------------------------------------------+
    ```

    Args:
        channels: The number of input and output channels.
        expansion: The expansion factor for the hidden channels.
        act: The activation function to use. If `None`, no activation is applied.
        act_kwargs: Keyword arguments for the activation function.
        act_first: Whether to apply the activation function before the normalization.
        norm: The normalization function to use. If `None`, no normalization is applied.
        norm_kwargs: Keyword arguments for the normalization function.
        bias: Whether to use a bias for the linear layers.

    Examples:
        ```pycon
        >>> import torch
        >>> from torch_pointcloud.models.pointmlp import ResidualLinearBlock
        >>> block = ResidualLinearBlock(64, expansion=2, act="relu", norm="batch_norm", bias=False)
        >>> x = torch.randn(32, 64)
        >>> y = block(x)
        >>> print(y.shape)
        torch.Size([32, 64])

        ```
    """

    def __init__(
        self,
        channels: int,
        expansion: float = 1.0,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: bool = True,
    ):
        super().__init__()
        act_kwargs = act_kwargs or {}
        norm_kwargs = norm_kwargs or {}
        hidden_channels = int(channels * expansion)

        self.lin1 = nn.Linear(channels, hidden_channels, bias=bias)
        self.norm1 = create_norm(norm, hidden_channels, **norm_kwargs) or nn.Identity()
        self.lin2 = nn.Linear(hidden_channels, channels, bias=bias)
        self.norm2 = create_norm(norm, channels, **norm_kwargs) or nn.Identity()
        self.act = create_act(act, **act_kwargs) or nn.Identity()
        self.act_first = act_first

    def reset_parameters(self) -> None:
        reset(self.lin1)
        reset(self.norm1)
        reset(self.lin2)
        reset(self.norm2)
        reset(self.act)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        x = self.lin1(x)
        if self.act is not None and self.act_first:
            x = self.act(x)
        x = self.norm1(x)
        if self.act is not None and not self.act_first:
            x = self.act(x)

        x = self.lin2(x)
        if self.act is not None and self.act_first:
            x = self.act(x + identity)
        x = self.norm2(x)
        if self.act is not None and not self.act_first:
            x = self.act(x + identity)

        return x


class PointMLPEncoderBlock(nn.Module):
    r"""One encoder stage: optional FPS downsampling, then a geometric affine $k$-NN aggregation.

    A pre-aggregation MLP of `num_pre_blocks` residual blocks lifts the grouped features to
    `out_channels`, and a post-aggregation MLP of `num_pos_blocks` residual blocks refines the
    max-pooled result.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        k: int,
        spatial_dim: int = 3,
        num_pre_blocks: int = 2,
        num_pos_blocks: int = 2,
        normalize: Literal["center", "anchor"] = "center",
        std_mode: Literal["graph", "batch"] = "graph",
        res_expansion: float = 1.0,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: bool = False,
        add_self_loops: bool = False,
        use_pos: bool = True,
        downsample: Optional[nn.Module] = None,
    ):
        super().__init__()
        kwargs: Dict[str, Any] = dict(
            act=act,
            act_kwargs=act_kwargs,
            act_first=act_first,
            norm=norm,
            norm_kwargs=norm_kwargs,
            bias=bias,
        )

        self.downsample = downsample
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = k
        self.use_pos = use_pos

        pre_mlp_in = 2 * in_channels + (spatial_dim if use_pos else 0)
        pre_mlp = nn.Sequential(
            LinearBlock(pre_mlp_in, out_channels, **kwargs),
            *[ResidualLinearBlock(out_channels, expansion=res_expansion, **kwargs) for _ in range(num_pre_blocks)],
        )

        self.conv = GeometricAffineConv(
            local_nn=pre_mlp,
            channels=in_channels,
            spatial_dim=spatial_dim,
            use_pos=use_pos,
            normalize=normalize,
            std_mode=std_mode,
            add_self_loops=add_self_loops,
            aggr="max",
        )

        self.pos_mlp = nn.Sequential(
            *[ResidualLinearBlock(out_channels, expansion=res_expansion, **kwargs) for _ in range(num_pos_blocks)]
        )

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


class ResidualFeaturePropagation(torch.nn.Module):
    r"""Feature propagation block: $k$-NN interpolation to the skip resolution, concatenation with the
    skip features, then a `LinearBlock` followed by `num_layers` `ResidualLinearBlock` units.
    """

    def __init__(
        self,
        in_channels: int,
        out_channel: int,
        *,
        num_layers: int = 1,
        k: int,
        expansion: float = 1.0,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        kwargs: Dict[str, Any] = dict(
            act=act,
            act_kwargs=act_kwargs,
            act_first=act_first,
            norm=norm,
            norm_kwargs=norm_kwargs,
            bias=bias,
        )

        self.k = k
        self.mlp = nn.Sequential(
            LinearBlock(in_channels, out_channel, **kwargs),
            *[ResidualLinearBlock(out_channel, expansion=expansion, **kwargs) for _ in range(num_layers)],
        )

    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        batch: Tensor,
        x_skip: Tensor,
        pos_skip: Tensor,
        batch_skip: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        x = self.mlp(x)
        return x, pos_skip, batch_skip


class PointMLPEncoder(nn.Module):
    """Stack of `PointMLPEncoderBlock` stages that progressively decimate the cloud with FPS.

    A stage with a ratio of `0` keeps every point and only transforms features. When
    `return_intermediates=True` is passed to `forward`, the pre-downsampling features of every stage
    are returned in coarse-to-fine order, ready to be consumed as decoder skips.
    """

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
        std_mode: Literal["graph", "batch"] = "graph",
        res_expansion: float = 1.0,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: bool = True,
        add_self_loops: bool = False,
        use_pos: bool = True,
        fps_random_start: Optional[bool] = None,
        aggr: str = "max",
    ):
        super().__init__()
        self.channels = ensure_tuple(channels)
        self.spatial_dim = spatial_dim
        self.act = act
        self.act_kwargs = act_kwargs
        self.act_first = act_first
        self.norm = norm
        self.norm_kwargs = norm_kwargs
        self.bias = bias
        self.add_self_loops = add_self_loops
        self.use_pos = use_pos
        self.fps_random_start = fps_random_start
        self.normalize = normalize
        self.std_mode = std_mode
        self.res_expansion = res_expansion

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
            block = self.configure_block(i)
            self.blocks.append(block)

    def configure_block(self, index: int) -> nn.Module:
        """Build the `PointMLPEncoderBlock` for stage `index`, with an FPS sampler when its ratio is non-zero."""
        downsample: Optional[nn.Module] = None
        if self.ratios[index]:
            downsample = FPS(ratio=self.ratios[index], random_start=self.fps_random_start)

        return PointMLPEncoderBlock(
            in_channels=self.channels[index],
            out_channels=self.channels[index + 1],
            spatial_dim=self.spatial_dim,
            k=self.num_neighbors[index],
            num_pre_blocks=self.num_pre_blocks[index],
            num_pos_blocks=self.num_pos_blocks[index],
            normalize=self.normalize,
            std_mode=self.std_mode,
            res_expansion=self.res_expansion,
            act=self.act,
            act_kwargs=self.act_kwargs,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
            bias=self.bias,
            add_self_loops=self.add_self_loops,
            use_pos=self.use_pos,
            downsample=downsample,
        )

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
            if return_intermediates:
                intermediate = PointMLPIntermediate(x, pos, batch)
                intermediates.append(intermediate)

            x, pos, batch = block(x, pos, batch)

        if return_intermediates:
            return x, pos, batch, intermediates[::-1]
        return x, pos, batch


class PointMLPDecoder(nn.Module):
    """Stack of `ResidualFeaturePropagation` blocks that walk the encoder intermediates back to full resolution."""

    def __init__(
        self,
        channels: Sequence[int],
        skip_channels: Sequence[int],
        depths: Sequence[int],
        *,
        spatial_dim: int = 3,
        dropout: float = 0.0,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: bool = True,
    ):
        super().__init__()
        self.channels = ensure_list(channels)
        depth = len(self.channels) - 1
        self.skip_channels = ensure_list_size(skip_channels, depth + 1)
        self.depths = ensure_list_size(depths, depth)

        self.spatial_dim = spatial_dim
        self.act = act
        self.act_kwargs = act_kwargs
        self.act_first = act_first
        self.norm = norm
        self.norm_kwargs = norm_kwargs
        self.bias = bias
        self.dropout = dropout

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = self.configure_block(i)
            self.blocks.append(block)

    def configure_block(self, index: int) -> nn.Module:
        """Build the `ResidualFeaturePropagation` block for stage `index`."""
        in_channels = self.channels[index] + self.skip_channels[index]
        return ResidualFeaturePropagation(
            in_channels=in_channels,
            out_channel=self.channels[index + 1],
            k=1 if index == 0 else self.spatial_dim,
            num_layers=self.depths[index],
            act=self.act,
            act_kwargs=self.act_kwargs,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
            bias=self.bias,
        )

    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        batch: Tensor,
        intermediates: List[PointMLPIntermediate],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        for block, intermediate in zip(self.blocks, intermediates):
            x, pos, batch = block(x, pos, batch, *intermediate)
        return x, pos, batch


class PointMLPClassification(ClassificationModel):
    r"""PointMLP classification model from
    :arxiv: [Rethinking Network Design and Local Geometry in Point Cloud: A Simple Residual MLP Framework](https://arxiv.org/abs/2202.07123)
    by Xu Ma, Can Qin, Haoxuan You, Haoxi Ran, Yun Fu.

    A pure residual MLP network: each encoder stage samples centroids with farthest point sampling,
    normalizes each $k$-NN neighborhood with a geometric affine module, and applies residual MLP
    blocks before and after the neighborhood aggregation. Point features are pooled globally after
    the encoder for classification.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        *,
        spatial_dim: int = 3,
        encoder_channels: Sequence[int],
        num_neighbors: Union[int, Sequence[int]],
        ratios: Union[float, Sequence[float]],
        num_pre_blocks: Union[int, Sequence[int]] = 2,
        num_pos_blocks: Union[int, Sequence[int]] = 2,
        normalize: Literal["center", "anchor"] = "center",
        std_mode: Literal["graph", "batch"] = "graph",
        res_expansion: float = 1.0,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: bool = True,
        add_self_loops: bool = False,
        use_pos: bool = True,
        dropout: float = 0.0,
        head_channels: Optional[Sequence[int]] = None,
        head_dropout: float = 0.0,
        global_pool: PoolLike = "max",
    ):
        super().__init__(in_channels=in_channels, num_classes=num_classes)
        self.encoder_channels = ensure_list(encoder_channels)
        self.spatial_dim = spatial_dim
        self.num_neighbors = num_neighbors
        self.ratios = ratios
        self.num_pre_blocks = num_pre_blocks
        self.num_pos_blocks = num_pos_blocks
        self.normalize = normalize
        self.std_mode = std_mode
        self.res_expansion = res_expansion
        self.act = act
        self.act_kwargs = act_kwargs
        self.act_first = act_first
        self.norm = norm
        self.norm_kwargs = norm_kwargs
        self.bias = bias
        self.add_self_loops = add_self_loops
        self.use_pos = use_pos
        self.dropout = dropout
        self.head_channels = list(head_channels) if head_channels else []
        self.head_dropout = head_dropout

        self.stem = self.configure_stem()
        self.encoder = self.configure_encoder()
        self.global_pool = create_pool(global_pool)
        self.head = self.configure_head()

    def configure_stem(self) -> nn.Module:
        """Build the linear stem lifting the input features to the first encoder channel."""
        return MLP(
            [self.in_channels, self.encoder_channels[0]],
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
        """Build the `PointMLPEncoder` backbone."""
        return PointMLPEncoder(
            channels=self.encoder_channels,
            spatial_dim=self.spatial_dim,
            num_neighbors=self.num_neighbors,
            ratios=self.ratios,
            num_pre_blocks=self.num_pre_blocks,
            num_pos_blocks=self.num_pos_blocks,
            normalize=self.normalize,
            std_mode=self.std_mode,
            res_expansion=self.res_expansion,
            act=self.act,
            act_kwargs=self.act_kwargs,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
            bias=self.bias,
            add_self_loops=self.add_self_loops,
            use_pos=self.use_pos,
        )

    @property
    def num_features(self) -> int:
        """Feature dimension $C$ of the encoder output."""
        return self.encoder_channels[-1]

    def configure_head(self) -> nn.Module:
        if self.num_classes == 0:
            return nn.Identity()
        if not self.head_channels:
            return nn.Linear(self.num_features, self.num_classes)
        return MLP(
            [self.num_features] + list(self.head_channels) + [self.num_classes],
            act=self.act,
            act_kwargs=self.act_kwargs,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
            bias=True,
            dropout=self.head_dropout,
            plain_last=True,
        )

    def reset_classifier(self, num_classes: int, global_pool: Optional[PoolLike] = None, **kwargs: Any) -> None:
        self.num_classes = num_classes
        if global_pool is not None:
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
    r"""PointMLP segmentation model from
    :arxiv: [Rethinking Network Design and Local Geometry in Point Cloud: A Simple Residual MLP Framework](https://arxiv.org/abs/2202.07123)
    by Xu Ma, Can Qin, Haoxuan You, Haoxi Ran, Yun Fu.

    The PointMLP encoder is followed by a decoder of residual feature-propagation blocks with skip
    connections and a per-point linear head.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        *,
        spatial_dim: int = 3,
        encoder_channels: Sequence[int],
        num_neighbors: Union[int, Sequence[int]],
        ratios: Union[float, Sequence[float]],
        num_pre_blocks: Union[int, Sequence[int]] = 2,
        num_pos_blocks: Union[int, Sequence[int]] = 2,
        decoder_channels: Sequence[int],
        decoder_blocks: Sequence[nn.Module],
        normalize: Literal["center", "anchor"] = "center",
        std_mode: Literal["graph", "batch"] = "graph",
        res_expansion: float = 1.0,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: bool = True,
        add_self_loops: bool = False,
        use_pos: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__(in_channels=in_channels, num_classes=num_classes)
        self.encoder_channels = ensure_list(encoder_channels)
        self.decoder_channels = ensure_list(decoder_channels)
        self.decoder_blocks = ensure_list(decoder_blocks)
        self.spatial_dim = spatial_dim
        self.num_neighbors = num_neighbors
        self.ratios = ratios
        self.num_pre_blocks = num_pre_blocks
        self.num_pos_blocks = num_pos_blocks
        self.normalize = normalize
        self.std_mode = std_mode
        self.res_expansion = res_expansion
        self.act = act
        self.act_kwargs = act_kwargs
        self.act_first = act_first
        self.norm = norm
        self.norm_kwargs = norm_kwargs
        self.bias = bias
        self.add_self_loops = add_self_loops
        self.use_pos = use_pos
        self.dropout = dropout

        self.stem = self.configure_stem()
        self.encoder = self.configure_encoder()
        self.decoder = self.configure_decoder()
        self.head = self.configure_head()

    def configure_stem(self) -> nn.Module:
        """Build the linear stem lifting the input features to the first encoder channel."""
        return MLP(
            [self.in_channels, self.encoder_channels[0]],
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
        """Build the `PointMLPEncoder` backbone."""
        return PointMLPEncoder(
            channels=self.encoder_channels,
            spatial_dim=self.spatial_dim,
            num_neighbors=self.num_neighbors,
            ratios=self.ratios,
            num_pre_blocks=self.num_pre_blocks,
            num_pos_blocks=self.num_pos_blocks,
            normalize=self.normalize,
            std_mode=self.std_mode,
            res_expansion=self.res_expansion,
            act=self.act,
            act_kwargs=self.act_kwargs,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
            bias=self.bias,
            add_self_loops=self.add_self_loops,
            use_pos=self.use_pos,
        )

    def configure_decoder(self) -> PointMLPDecoder:
        """Build the `PointMLPDecoder`, mirroring the encoder channels in reverse."""
        return PointMLPDecoder(
            channels=[self.encoder_channels[-1]] + self.decoder_channels,
            skip_channels=self.encoder_channels[:-1][::-1] + [self.in_channels],
            spatial_dim=self.spatial_dim,
            depths=self.decoder_blocks,
            act=self.act,
            act_kwargs=self.act_kwargs,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
            bias=self.bias,
        )

    @property
    def num_features(self) -> int:
        """Feature dimension $C$ of the decoder output."""
        return self.decoder_channels[-1]

    def configure_head(self) -> nn.Module:
        if self.num_classes == 0:
            return nn.Identity()
        return nn.Linear(self.num_features, self.num_classes)

    def reset_classifier(self, num_classes: int, **kwargs: Any) -> None:
        self.num_classes = num_classes
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

    def forward_decoder(
        self,
        x: Tensor,
        pos: Tensor,
        batch: Tensor,
        intermediates: List[PointMLPIntermediate],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        return self.decoder(x, pos, batch, intermediates)

    def forward_head(self, x: Tensor, pre_logits: bool = False) -> Tensor:
        if self.dropout:
            x = F.dropout(x, p=float(self.dropout), training=self.training)
        return x if pre_logits else self.head(x)

    def forward(self, x: OptTensor, pos: Tensor, batch: Tensor) -> Tensor:
        x, pos, batch, intermediates = self.forward_features(x, pos, batch, return_intermediates=True)
        x, pos, batch = self.forward_decoder(x, pos, batch, intermediates)
        return self.forward_head(x)


def _pointmlp_base_hparams(**kwargs: Any) -> Dict[str, Any]:
    """Shared encoder/backbone hparams for pointmlp-base."""
    hparams = dict(
        in_channels=3,
        spatial_dim=3,
        encoder_channels=(64, 128, 256, 512, 1024),
        ratios=(0.5, 0.5, 0.5, 0.5),
        num_neighbors=(24, 24, 24, 24),
        num_pre_blocks=(2, 2, 2, 2),
        num_pos_blocks=(2, 2, 2, 2),
        normalize="anchor",
        res_expansion=1.0,
        act="relu",
        act_first=False,
        norm="batch_norm",
        bias=False,
        use_pos=False,
        dropout=0.0,
        add_self_loops=False,
    )
    hparams.update(kwargs)
    return hparams


def _pointmlp_elite_hparams(**kwargs: Any) -> Dict[str, Any]:
    """Shared encoder/backbone hparams for pointmlp-elite."""
    hparams = dict(
        in_channels=3,
        spatial_dim=3,
        encoder_channels=(32, 64, 128, 256, 256),
        ratios=(0.5, 0.5, 0.5, 0.5),
        num_neighbors=(24, 24, 24, 24),
        num_pre_blocks=(1, 1, 2, 1),
        num_pos_blocks=(1, 1, 2, 1),
        normalize="anchor",
        res_expansion=0.25,
        act="relu",
        act_first=False,
        norm="batch_norm",
        bias=False,
        use_pos=False,
        dropout=0.0,
        add_self_loops=False,
    )
    hparams.update(kwargs)
    return hparams


def _pointmlp_base_clf_hparams(**kwargs: Any) -> Dict[str, Any]:
    hparams = _pointmlp_base_hparams()
    hparams.update(head_channels=(512, 256), head_dropout=0.5, global_pool="max")
    hparams.update(kwargs)
    return hparams


def _pointmlp_elite_clf_hparams(**kwargs: Any) -> Dict[str, Any]:
    hparams = _pointmlp_elite_hparams()
    hparams.update(head_channels=(512, 256), head_dropout=0.5, global_pool="max")
    hparams.update(kwargs)
    return hparams


def _pointmlp_base_seg_hparams(**kwargs: Any) -> Dict[str, Any]:
    hparams = _pointmlp_base_hparams()
    hparams.update(
        ratios=(0.25, 0.25, 0.25, 0.25),
        num_neighbors=(32, 32, 32, 32),
        decoder_channels=(512, 256, 128, 128),
        decoder_blocks=(4, 4, 4, 4),
    )
    hparams.update(kwargs)
    return hparams


def _pointmlp_elite_seg_hparams(**kwargs: Any) -> Dict[str, Any]:
    hparams = _pointmlp_elite_hparams()
    hparams.update(
        ratios=(0.25, 0.25, 0.25, 0.25),
        num_neighbors=(32, 32, 32, 32),
        decoder_channels=(128, 64, 32, 32),
        decoder_blocks=(1, 1, 2, 1),
    )
    hparams.update(kwargs)
    return hparams


@register_model("pointmlp-base", task="classification", hparams=_pointmlp_base_clf_hparams())
def pointmlp_base_clf(**hparams: Any) -> PointMLPClassification:
    return PointMLPClassification(**hparams)


@register_model("pointmlp-elite", task="classification", hparams=_pointmlp_elite_clf_hparams())
def pointmlp_elite_clf(**hparams: Any) -> PointMLPClassification:
    return PointMLPClassification(**hparams)


@register_model("pointmlp-base", task="segmentation", hparams=_pointmlp_base_seg_hparams())
def pointmlp_base_seg(**hparams: Any) -> PointMLPSegmentation:
    return PointMLPSegmentation(**hparams)


@register_model("pointmlp-elite", task="segmentation", hparams=_pointmlp_elite_seg_hparams())
def pointmlp_elite_seg(**hparams: Any) -> PointMLPSegmentation:
    return PointMLPSegmentation(**hparams)


@register_model(
    "pointmlp-base.modelnet40.xu-ma",
    task="classification",
    weights=WeightsDict(
        url="hf://torch-pointcloud/pointmlp/pointmlp-base.modelnet40.xu-ma.safetensors",
        dataset="modelnet40",
        metrics={"OA": 93.88},
        classes=MODELNET40_CLASSES,
        author="xu-ma",
        license="Apache-2.0",
    ),
    transform=T.Compose(
        [
            T.CopyItems(keys=DataKeys.POS, names=DataKeys.ORIGIN_POS),
            T.FarthestPointSample(
                pos_key=DataKeys.POS,
                keys=[DataKeys.NORMAL],
                num_samples=1024,
                allow_missing_keys=True,
                dst_index_key=DataKeys.INDEX,
            ),
        ]
    ),
    hparams=_pointmlp_base_clf_hparams(num_classes=40),
)
def pointmlp_base_modelnet40_clf(**hparams: Any) -> PointMLPClassification:
    return PointMLPClassification(**hparams)


@register_model(
    "pointmlp-elite.modelnet40.xu-ma",
    task="classification",
    weights=WeightsDict(
        url="hf://torch-pointcloud/pointmlp/pointmlp-elite.modelnet40.xu-ma.safetensors",
        dataset="modelnet40",
        metrics={"OA": 92.79},
        classes=MODELNET40_CLASSES,
        author="xu-ma",
        license="Apache-2.0",
    ),
    transform=T.Compose(
        [
            T.CopyItems(keys=DataKeys.POS, names=DataKeys.ORIGIN_POS),
            T.FarthestPointSample(
                pos_key=DataKeys.POS,
                keys=[DataKeys.NORMAL],
                num_samples=1024,
                allow_missing_keys=True,
                dst_index_key=DataKeys.INDEX,
            ),
        ]
    ),
    hparams=_pointmlp_elite_clf_hparams(num_classes=40),
)
def pointmlp_elite_modelnet40_clf(**hparams: Any) -> PointMLPClassification:
    return PointMLPClassification(**hparams)


# The ScanObjectNN weights are the `model31C-demo1` checkpoints of the original release
# (https://drive.google.com/drive/folders/1Jn9HNpPsrq-1XqSmOUtw4cwPMjsIiIpz), trained before the
# reference repository's std fix, i.e. with one standard deviation over the whole batch
# (`std_mode="batch"`). That mode scores 77.8 / 75.8 OA at batch size 32 against 77.2 / 75.3 with the
# per-graph default, so the registrations keep the default and its per-sample independence. The README
# numbers (86.1 / 84.1 OA) belong to the post-fix `fixstd/scanobjectnn/*` checkpoints, whose download
# URLs are dead.
@register_model(
    "pointmlp-base.scanobjectnn.xu-ma",
    task="classification",
    weights=WeightsDict(
        url="hf://torch-pointcloud/pointmlp/pointmlp-base.scanobjectnn.xu-ma.safetensors",
        dataset="scanobjectnn",
        metrics={"OA": 77.48},
        classes=SCANOBJECTNN_CLASSES,
        author="xu-ma",
        license="Apache-2.0",
    ),
    transform=T.Compose(
        [
            T.CopyItems(keys=DataKeys.POS, names=DataKeys.ORIGIN_POS),
            T.FarthestPointSample(
                pos_key=DataKeys.POS,
                keys=[DataKeys.NORMAL],
                num_samples=1024,
                allow_missing_keys=True,
                dst_index_key=DataKeys.INDEX,
            ),
        ]
    ),
    hparams=_pointmlp_base_clf_hparams(num_classes=15),
)
def pointmlp_base_scanobjectnn_clf(**hparams: Any) -> PointMLPClassification:
    return PointMLPClassification(**hparams)


@register_model(
    "pointmlp-elite.scanobjectnn.xu-ma",
    task="classification",
    weights=WeightsDict(
        url="hf://torch-pointcloud/pointmlp/pointmlp-elite.scanobjectnn.xu-ma.safetensors",
        dataset="scanobjectnn",
        metrics={"OA": 76.72},
        classes=SCANOBJECTNN_CLASSES,
        author="xu-ma",
        license="Apache-2.0",
    ),
    transform=T.Compose(
        [
            T.CopyItems(keys=DataKeys.POS, names=DataKeys.ORIGIN_POS),
            T.FarthestPointSample(
                pos_key=DataKeys.POS,
                keys=[DataKeys.NORMAL],
                num_samples=1024,
                allow_missing_keys=True,
                dst_index_key=DataKeys.INDEX,
            ),
        ]
    ),
    hparams=_pointmlp_elite_clf_hparams(num_classes=15),
)
def pointmlp_elite_scanobjectnn_clf(**hparams: Any) -> PointMLPClassification:
    return PointMLPClassification(**hparams)
