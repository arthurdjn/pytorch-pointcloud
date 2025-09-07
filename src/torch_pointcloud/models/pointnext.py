from typing import Any, Callable, Dict, List, Literal, NamedTuple, Optional, Sequence, Tuple, Union, overload

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import MLP

from torch_pointcloud.layers import PoolLike, create_cls_head, create_pool
from torch_pointcloud.layers.pointnet2_blocks import PointNet2FeaturePropagation
from torch_pointcloud.layers.pointnext_blocks import PointNeXtResidualBlock, PointNeXtSetAbstraction
from torch_pointcloud.utils.conversion import ensure_list, ensure_tuple, ensure_tuple_size
from torch_pointcloud.utils.types import AggrType, OptTensor


class PointNeXtIntermediate(NamedTuple):
    x: Tensor
    pos: Tensor
    batch: Tensor


class PointNeXtEncoderBlock(nn.Module):
    def __init__(
        self,
        spatial_dim: int,
        channels: int,
        depth: int,
        expansion: int,
        ratio: float,
        radius: float,
        num_neighbors: int,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: Union[bool, List[bool]] = True,
        add_self_loops: bool = False,
        aggr: AggrType = "max",
        downsample: Optional[PointNeXtSetAbstraction] = None,
    ):
        super().__init__()
        act_kwargs = act_kwargs or {}
        norm_kwargs = norm_kwargs or {}

        self.downsample = downsample
        self.layers = nn.ModuleList()
        for _ in range(depth):
            layer = PointNeXtResidualBlock(
                spatial_dim=spatial_dim,
                channels=channels,
                expansion=expansion,
                ratio=ratio,
                radius=radius,
                num_neighbors=num_neighbors,
                act=act,
                act_kwargs=act_kwargs,
                act_first=act_first,
                norm=norm,
                norm_kwargs=norm_kwargs,
                bias=bias,
                add_self_loops=add_self_loops,
                aggr=aggr,
            )
            self.layers.append(layer)

    def forward(self, x: Tensor, pos: Tensor, batch: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        if self.downsample is not None:
            x, pos, batch = self.downsample(x, pos, batch)

        for layer in self.layers:
            x = layer(x, pos, batch)

        return x, pos, batch


class PointNeXtEncoder(nn.Module):
    def __init__(
        self,
        channels: Sequence[int],
        *,
        spatial_dim: int = 3,
        depths: Sequence[int],
        expansion: Union[int, Sequence[int]] = 4,
        ratios: Sequence[float],
        radiuses: Sequence[Union[float, Sequence[float]]],
        num_neighbors: Sequence[Union[int, Sequence[int]]],
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: Union[bool, List[bool]] = True,
        add_self_loops: bool = False,
        aggr: AggrType = "max",
    ) -> None:
        super().__init__()
        self.channels = ensure_tuple(channels)

        size = len(self.channels) - 1
        extra_msg = (
            f"Invalid {self.__class__.__name__} parameter: "
            f"expected `{{param}}` to have the same length as the number of channels ({size})."
        )
        self.depths = ensure_tuple_size(depths, size, extra_msg=extra_msg.format(param="depths"))
        self.ratios = ensure_tuple_size(ratios, size, extra_msg=extra_msg.format(param="ratios"))
        self.radiuses = ensure_tuple_size(radiuses, size, extra_msg=extra_msg.format(param="radiuses"))
        self.num_neighbors = ensure_tuple_size(num_neighbors, size, extra_msg=extra_msg.format(param="num_neighbors"))
        self.expansion = ensure_tuple_size(expansion, size, extra_msg=extra_msg.format(param="expansion"))

        self.blocks = nn.ModuleList()
        for i in range(size):
            downsample = PointNeXtSetAbstraction(
                spatial_dim=spatial_dim,
                in_channels=channels[i],
                channels=[channels[i + 1]],
                ratio=self.ratios[i],
                radius=self.radiuses[i],
                num_neighbors=self.num_neighbors[i],
                act=act,
                act_kwargs=act_kwargs,
                act_first=act_first,
                norm=norm,
                norm_kwargs=norm_kwargs,
                bias=bias,
                add_self_loops=add_self_loops,
                aggr=aggr,
            )
            block = PointNeXtEncoderBlock(
                spatial_dim=spatial_dim,
                channels=channels[i + 1],
                depth=self.depths[i],
                expansion=self.expansion[i],
                ratio=self.ratios[i],
                radius=self.radiuses[i],
                num_neighbors=self.num_neighbors[i],
                act=act,
                act_kwargs=act_kwargs,
                act_first=act_first,
                norm=norm,
                norm_kwargs=norm_kwargs,
                bias=bias,
                add_self_loops=add_self_loops,
                aggr=aggr,
                downsample=downsample,
            )
            self.blocks.append(block)

    @overload
    def forward(
        self,
        x: OptTensor,
        pos: Tensor,
        batch: Tensor,
        return_intermediates: Literal[True],
    ) -> Tuple[Tensor, Tensor, Tensor, List[PointNeXtIntermediate]]: ...

    @overload
    def forward(
        self,
        x: OptTensor,
        pos: Tensor,
        batch: Tensor,
        return_intermediates: Literal[False] = False,
    ) -> Tuple[Tensor, Tensor, Tensor]: ...

    def forward(
        self,
        x: OptTensor,
        pos: Tensor,
        batch: Tensor,
        return_intermediates: bool = False,
    ) -> Any:
        intermediates = []
        for block in self.blocks:
            if return_intermediates:
                intermediate = PointNeXtIntermediate(x, pos, batch)
                intermediates.append(intermediate)

            x, pos, batch = block(x, pos, batch)

        if return_intermediates:
            return x, pos, batch, reversed(intermediates)
        return x, pos, batch


class PointNeXtDecoder(nn.Module):
    r"""The PointNeXt decoder, using the Feature Propagation (FP) module from the PointNet++ architecture.

    Note:
        The number of channels should be equal to the number of decoder blocks + 1.

    Args:
        channels: List of channels for each FP block.
            The first element should correspond to the last channel of the encoder.
        skip_channels: List of channels for the skip connections.
            This is usually the same as the first $N-1$ channels of the encoder, in reverse order.
        depths: List of depths for each FP block.
        spatial_dim: Spatial dimension of the input point cloud.
        dropout: Dropout rate before the classification head.
        act: Activation function to use for the FP blocks.
        act_kwargs: Keyword arguments for the activation function.
        act_first: Whether to apply the activation function before the normalization.
        norm: Normalization function to use for the FP blocks.
        norm_kwargs: Keyword arguments for the normalization function.
        bias: Whether to use a bias for the FP blocks.
    """

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
        bias: Union[bool, List[bool]] = True,
    ):
        super().__init__()
        self.channels = ensure_tuple(channels)
        size = len(channels) - 1

        extra_msg = (
            f"Invalid {self.__class__.__name__} parameter: "
            f"expected `{{param}}` to have the same length as the number of channels ({size})."
        )
        self.depths = ensure_tuple_size(depths, size, extra_msg=extra_msg.format(param="depths"))
        self.skip_channels = ensure_tuple_size(skip_channels, size, extra_msg=extra_msg.format(param="skip_channels"))

        self.blocks = nn.ModuleList()
        for i in range(size):
            in_channels = self.channels[i] + self.skip_channels[i]
            block = PointNet2FeaturePropagation(
                channels=[in_channels] + [self.channels[i + 1]] * self.depths[i],
                k=1 if i == 0 else spatial_dim,
                act=act,
                act_kwargs=act_kwargs,
                act_first=act_first,
                norm=norm,
                norm_kwargs=norm_kwargs,
                bias=bias,
                dropout=dropout,
            )
            self.blocks.append(block)

    def forward(
        self,
        x: OptTensor,
        pos: Tensor,
        batch: Tensor,
        intermediates: List[PointNeXtIntermediate],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        for block, intermediate in zip(self.blocks, intermediates):
            x, pos, batch = block(x, pos, batch, *intermediate)
        return x, pos, batch


class PointNeXtClassification(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        *,
        stem_channels: Optional[Union[int, Sequence[int]]] = None,
        encoder_channels: Sequence[int],
        encoder_depths: Sequence[int],
        encoder_expansion: Union[int, Sequence[int]] = 4,
        ratios: Sequence[float],
        radiuses: Sequence[Union[float, Sequence[float]]],
        num_neighbors: Sequence[Union[int, Sequence[int]]],
        add_self_loops: bool = False,
        spatial_dim: int = 3,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: Union[bool, List[bool]] = True,
        dropout: float = 0.0,
        global_pool: PoolLike = "max",
    ):
        super().__init__()
        stem_channels = ensure_list(stem_channels, none_as_empty=True)
        encoder_channels = ensure_list(encoder_channels)
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.stem: Optional[nn.Module] = None
        if stem_channels:
            self.stem = MLP(
                stem_channels,
                act=act,
                act_kwargs=act_kwargs,
                act_first=act_first,
                norm=norm,
                norm_kwargs=norm_kwargs,
                bias=bias,
                plain_last=False,
            )
            in_channels = stem_channels

        self.encoder = PointNeXtEncoder(
            spatial_dim=spatial_dim,
            channels=[in_channels] + encoder_channels,
            depths=encoder_depths,
            expansion=encoder_expansion,
            ratios=ratios,
            radiuses=radiuses,
            num_neighbors=num_neighbors,
            act=act,
            act_kwargs=act_kwargs,
            act_first=act_first,
            norm=norm,
            norm_kwargs=norm_kwargs,
            bias=bias,
            add_self_loops=add_self_loops,
        )

        self.dropout = dropout
        self.global_pool = create_pool(global_pool)
        self.head = create_cls_head(num_features=self.embedding_dim, num_classes=self.num_classes)

    @property
    def embedding_dim(self) -> int:
        return self.encoder.channels[-1]

    def reset_classifier(self, num_classes: int, global_pool: PoolLike = "max", **kwargs: Any) -> None:
        self.num_classes = num_classes
        self.global_pool = create_pool(global_pool)
        self.head = create_cls_head(num_features=self.embedding_dim, num_classes=self.num_classes, **kwargs)

    @overload
    def forward_encoder(
        self,
        x: OptTensor,
        pos: Tensor,
        batch: Tensor,
        return_intermediates: Literal[True],
    ) -> Tuple[Tensor, Tensor, Tensor, List[PointNeXtIntermediate]]: ...

    @overload
    def forward_encoder(
        self,
        x: OptTensor,
        pos: Tensor,
        batch: Tensor,
        return_intermediates: Literal[False] = False,
    ) -> Tuple[Tensor, Tensor, Tensor]: ...

    def forward_encoder(
        self,
        x: OptTensor,
        pos: Tensor,
        batch: Tensor,
        return_intermediates: bool = False,
    ) -> Any:
        x = x if x is not None else pos
        if self.stem is not None:
            x = self.stem(x)
        return self.encoder(x, pos, batch, return_intermediates=return_intermediates)

    def forward_head(self, x: Tensor, batch: Tensor, pre_logits: bool = False) -> Tensor:
        x = self.global_pool(x, batch)
        if self.dropout:
            x = F.dropout(x, p=float(self.dropout), training=self.training)
        return x if pre_logits else self.head(x)

    def forward(self, x: OptTensor, pos: Tensor, batch: Tensor) -> Tensor:
        x, _, batch = self.forward_encoder(x, pos, batch)
        return self.forward_head(x, batch)


class PointNeXtSegmentation(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        *,
        stem_channels: Optional[Union[int, Sequence[int]]] = None,
        encoder_channels: Sequence[int],
        encoder_depths: Sequence[int],
        encoder_expansion: Union[int, Sequence[int]] = 4,
        decoder_channels: Sequence[int],
        decoder_depths: Sequence[int],
        ratios: Sequence[float],
        radiuses: Sequence[Union[float, Sequence[float]]],
        num_neighbors: Sequence[Union[int, Sequence[int]]],
        add_self_loops: bool = False,
        spatial_dim: int = 3,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: Union[bool, List[bool]] = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        stem_channels = ensure_list(stem_channels, none_as_empty=True)
        encoder_channels = ensure_list(encoder_channels)
        decoder_channels = ensure_list(decoder_channels)
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.stem: Optional[nn.Module] = None
        if stem_channels:
            self.stem = MLP(
                stem_channels,
                act=act,
                act_kwargs=act_kwargs,
                act_first=act_first,
                norm=norm,
                norm_kwargs=norm_kwargs,
                bias=bias,
                plain_last=False,
            )
            in_channels = stem_channels

        self.encoder = PointNeXtEncoder(
            spatial_dim=spatial_dim,
            channels=[in_channels] + encoder_channels,
            depths=encoder_depths,
            expansion=encoder_expansion,
            ratios=ratios,
            radiuses=radiuses,
            num_neighbors=num_neighbors,
            act=act,
            act_kwargs=act_kwargs,
            act_first=act_first,
            norm=norm,
            norm_kwargs=norm_kwargs,
            bias=bias,
            add_self_loops=add_self_loops,
        )

        self.decoder = PointNeXtDecoder(
            spatial_dim=spatial_dim,
            channels=[encoder_channels[-1]] + decoder_channels,
            skip_channels=encoder_channels[:-1][::-1] + [in_channels],
            depths=decoder_depths,
            act=act,
            act_kwargs=act_kwargs,
            act_first=act_first,
            norm=norm,
            norm_kwargs=norm_kwargs,
            bias=bias,
        )

        self.dropout = dropout
        self.head = create_cls_head(num_features=self.embedding_dim, num_classes=self.num_classes)

    @property
    def embedding_dim(self) -> int:
        return self.decoder.channels[-1]

    def reset_classifier(self, num_classes: int, **kwargs: Any) -> None:
        self.num_classes = num_classes
        self.head = create_cls_head(num_features=self.embedding_dim, num_classes=self.num_classes, **kwargs)

    @overload
    def forward_encoder(
        self,
        x: OptTensor,
        pos: Tensor,
        batch: Tensor,
        return_intermediates: Literal[True],
    ) -> Tuple[Tensor, Tensor, Tensor, List[PointNeXtIntermediate]]: ...

    @overload
    def forward_encoder(
        self,
        x: OptTensor,
        pos: Tensor,
        batch: Tensor,
        return_intermediates: Literal[False] = False,
    ) -> Tuple[Tensor, Tensor, Tensor]: ...

    def forward_encoder(
        self,
        x: OptTensor,
        pos: Tensor,
        batch: Tensor,
        return_intermediates: bool = False,
    ) -> Any:
        x = x if x is not None else pos
        if self.stem is not None:
            x = self.stem(x)
        return self.encoder(x, pos, batch, return_intermediates=return_intermediates)

    def forward_decoder(
        self,
        x: Tensor,
        pos: Tensor,
        batch: Tensor,
        intermediates: List[PointNeXtIntermediate],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        return self.decoder(x, pos, batch, intermediates)

    def forward_head(self, x: Tensor, pre_logits: bool = False) -> Tensor:
        if self.dropout:
            x = F.dropout(x, p=float(self.dropout), training=self.training)
        return x if pre_logits else self.head(x)

    def forward(self, x: OptTensor, pos: Tensor, batch: Tensor) -> Tensor:
        x, pos, batch, intermediates = self.forward_encoder(x, pos, batch, return_intermediates=True)
        x, pos, batch = self.forward_decoder(x, pos, batch, intermediates)
        return self.forward_head(x)
