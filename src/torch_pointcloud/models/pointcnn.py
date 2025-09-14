from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import MLP, XConv, fps
from torch_geometric.nn.resolver import activation_resolver

from torch_pointcloud.layers import PoolLike, create_pool
from torch_pointcloud.utils.conversion import ensure_list, ensure_tuple, ensure_tuple_size
from torch_pointcloud.utils.types import OptTensor

from ._base import ClassificationModel
from ._registry import register_model


class FPS(nn.Module):
    def __init__(self, ratio: float):
        super().__init__()
        self.ratio = ratio

    def forward(self, x: Tensor, pos: Tensor, batch: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        idx = fps(pos, batch, ratio=self.ratio)
        return x[idx], pos[idx], batch[idx]

    def extra_repr(self) -> str:
        return f"ratio={self.ratio}"


class PointCNNEncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial_dim: int,
        kernel_size: int,
        hidden_channels: Optional[int] = None,
        dilation: int = 1,
        bias: bool = True,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.downsample = downsample
        self.conv = XConv(
            in_channels,
            out_channels,
            dim=spatial_dim,
            kernel_size=kernel_size,
            hidden_channels=hidden_channels,
            dilation=dilation,
            bias=bias,
        )
        self.act = activation_resolver(act, **(act_kwargs or {}))

    def forward(self, x: Tensor, pos: Tensor, batch: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        if self.downsample is not None:
            x, pos, batch = self.downsample(x, pos, batch)

        x = self.conv(x, pos, batch)
        x = self.act(x)
        return x, pos, batch


class PointCNNEncoder(nn.Module):
    def __init__(
        self,
        channels: Sequence[int],
        kernel_sizes: Sequence[int],
        spatial_dim: int,
        ratios: Sequence[float],
        hidden_channels: Optional[Union[int, Sequence[int]]] = None,
        dilations: Sequence[int] = (1, 1, 1, 1),
        bias: bool = True,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.channels = ensure_tuple(channels)

        depth = len(self.channels) - 1
        msg = f"Invalid parameter for {self.__class__.__name__}. Expected `{{param}}` to have length {depth}."
        self.kernel_sizes = ensure_tuple_size(kernel_sizes, size=depth, extra_msg=msg.format(param="kernel_sizes"))
        self.dilations = ensure_tuple_size(dilations, size=depth, extra_msg=msg.format(param="dilations"))
        self.ratios = ensure_tuple_size(ratios, size=depth, extra_msg=msg.format(param="ratios"))
        self.hidden_channels = ensure_tuple_size(
            hidden_channels,
            size=depth,
            extra_msg=msg.format(param="hidden_channels"),
        )

        self.blocks = nn.ModuleList()
        for i in range(depth):
            downsample: Optional[nn.Module] = None
            if self.ratios[i]:
                downsample = FPS(self.ratios[i])

            block = PointCNNEncoderBlock(
                in_channels=self.channels[i],
                out_channels=self.channels[i + 1],
                spatial_dim=spatial_dim,
                kernel_size=self.kernel_sizes[i],
                hidden_channels=self.hidden_channels[i],
                dilation=self.dilations[i],
                bias=bias,
                act=act,
                act_kwargs=act_kwargs,
                downsample=downsample,
            )
            self.blocks.append(block)

    def forward(self, x: Tensor, pos: Tensor, batch: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        for block in self.blocks:
            x = block(x, pos, batch)
        return x, pos, batch


class PointCNNClassification(ClassificationModel):
    r"""
    Classification model as described in the paper
    ["PointCNN: Convolution On X-Transformed Points"](https://arxiv.org/abs/1801.07791)
    by Yangyan Li, Rui Bu, Mingchao Sun, Wei Wu, Xinhan Di, Baoquan Chen.

    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        *,
        spatial_dim: int = 3,
        channels: Sequence[int],
        kernel_sizes: Sequence[int],
        ratios: Sequence[float],
        hidden_channels: Optional[Union[int, Sequence[int]]] = None,
        dilations: Sequence[int] = (1, 1, 1, 1),
        bias: bool = True,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        dropout: float = 0.0,
        head_channels: Optional[Union[int, Sequence[int]]] = None,
        global_pool: PoolLike = "max",
    ):
        super().__init__(in_channels=in_channels, num_classes=num_classes)
        self.spatial_dim = spatial_dim
        self.channels = ensure_list(channels)
        self.kernel_sizes = ensure_list(kernel_sizes)
        self.ratios = ensure_list(ratios)
        self.hidden_channels = ensure_list(hidden_channels)
        self.dilations = ensure_list(dilations)
        self.head_channels = ensure_list(head_channels, none_as_empty=True)
        self.bias = bias
        self.act = act
        self.act_kwargs = act_kwargs
        self.act_first = act_first
        self.norm = norm
        self.norm_kwargs = norm_kwargs

        self.encoder = PointCNNEncoder(
            channels=[self.in_channels] + self.channels,
            kernel_sizes=kernel_sizes,
            spatial_dim=spatial_dim,
            ratios=ratios,
            hidden_channels=hidden_channels,
            dilations=dilations,
            bias=bias,
            act=act,
            act_kwargs=act_kwargs,
        )

        self.dropout = dropout
        self.global_pool = create_pool(global_pool)
        self.head = MLP(
            [self.embedding_dim] + self.head_channels + [self.num_classes],
            act=self.act,
            act_kwargs=self.act_kwargs,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
            bias=self.bias,
            dropout=[0] * len(self.head_channels) + [self.dropout],
            plain_last=True,
        )

    @property
    def embedding_dim(self) -> int:
        return self.channels[-1]

    def reset_classifier(self, num_classes: int, global_pool: PoolLike = "max", **kwargs: Any) -> None:
        self.num_classes = num_classes
        self.global_pool = create_pool(global_pool)
        self.head = MLP(
            [self.embedding_dim] + self.head_channels + [self.num_classes],
            act=self.act,
            act_kwargs=self.act_kwargs,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
            bias=self.bias,
            dropout=[0] * len(self.head_channels) + [self.dropout],
            plain_last=True,
        )

    def forward_features(self, x: OptTensor, pos: Tensor, batch: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        x, pos, batch = self.encoder(x, pos, batch)
        return x, pos, batch

    def forward_head(self, x: Tensor, batch: Tensor, pre_logits: bool = False) -> Tensor:
        x = self.global_pool(x, batch)
        if self.dropout:
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x if pre_logits else self.head(x)

    def forward(self, x: OptTensor, pos: Tensor, batch: Tensor) -> Tensor:
        x, _, batch = self.forward_features(x, pos, batch)
        return self.forward_head(x, batch)


@register_model("pointcnn-base", task="classification")
def pointcnn_base_cls(in_channels: int, num_classes: int, **kwargs: Any) -> PointCNNClassification:
    hparams = dict(
        in_channels=in_channels,
        num_classes=num_classes,
        spatial_dim=3,
        channels=[48, 96, 192, 384],
        hidden_channels=[32, 64, 128, 256],
        kernel_sizes=[8, 12, 16, 16],
        dilations=[1, 2, 2, 2],
        ratios=[0.0, 0.375, 0.334, 0.0],
        act="relu",
        act_first=False,
        norm="batch_norm",
        bias=True,
        dropout=0.5,
        head_channels=[256, 128],
        global_pool="mean",
    )
    hparams.update(kwargs)
    return PointCNNClassification(**hparams)  # type: ignore[arg-type]
