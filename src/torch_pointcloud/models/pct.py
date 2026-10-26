"""Point Cloud Transformer classification model.

{{ paper("2012.09688") }}
"""

from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import MLP

import torch_pointcloud.transforms as T
from torch_pointcloud.datasets.modelnet import MODELNET40_CLASSES
from torch_pointcloud.layers import CatPool, OffsetAttention, PCTNeighborEmbedding, PoolLike, create_pool
from torch_pointcloud.utils.conversion import ensure_list, ensure_list_size, is_iterable
from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.types import OptTensor

from ._base import ClassificationModel
from ._registry import WeightsDict, register_model


class PCTClassification(ClassificationModel):
    r"""
    Classification model as described in the paper
    :arxiv: [PCT: Point Cloud Transformer](https://arxiv.org/abs/2012.09688)
    by Meng-Hao Guo, Jun-Xiong Cai, Zheng-Ning Liu, Tai-Jiang Mu, Ralph R. Martin, Shi-Min Hu.

    A point-wise stem embeds the coordinates, two `PCTNeighborEmbedding` blocks sample and group the points, and a
    stack of `OffsetAttention` blocks refines the sampled points, whose outputs are concatenated with the embedding,
    projected and max-pooled into the global feature. The attention runs on the dense $(B, M, C)$ batch that the
    fixed number of sampled points per sample yields.

    Args:
        in_channels: Number of input feature channels, concatenated with the coordinates.
        num_classes: Number of output classes.
        spatial_dim: Spatial dimension of the input point cloud.
        stem_channels: Channels of the point-wise stem MLP.
        channels: Output channels of every neighbor embedding block.
        num_points: Sampled points per sample of every neighbor embedding block.
        num_neighbors: Neighbors grouped by every neighbor embedding block.
        attention_channels: Width of the attention input MLP and of every attention block.
        num_attention_blocks: Number of offset attention blocks.
        proj_channels: Width of the projection applied before pooling.
        head_channels: Hidden channels of the classification head.
        act: Activation of the stem, the neighbor embedding and the attention blocks.
        act_kwargs: Extra arguments for `act`.
        head_act: Activation of the projection and the head.
        head_act_kwargs: Extra arguments for `head_act`.
        norm: Normalization layer type.
        norm_kwargs: Additional arguments for the normalization layer.
        bias: Whether the linear layers before the head use a bias; the head layers always do.
        dropout: Dropout probability after every hidden layer of the head.
        global_pool: Global pooling operation for final feature aggregation.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        *,
        spatial_dim: int = 3,
        stem_channels: Sequence[int],
        channels: Sequence[int],
        num_points: Sequence[int],
        num_neighbors: Union[int, Sequence[int]],
        attention_channels: int,
        num_attention_blocks: int,
        proj_channels: int,
        head_channels: Optional[Union[int, Sequence[int]]] = None,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        head_act: Union[str, Callable, None] = "relu",
        head_act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: bool = True,
        dropout: float = 0.0,
        global_pool: PoolLike | Sequence[PoolLike] = "max",
    ):
        if len(channels) != len(num_points):
            raise ValueError(f"`channels` and `num_points` must have the same length, got {channels} and {num_points}.")

        super().__init__(in_channels=in_channels, num_classes=num_classes)
        self.spatial_dim = spatial_dim
        self.stem_channels = ensure_list(stem_channels)
        self.channels = ensure_list(channels)
        self.num_points = ensure_list(num_points)
        self.num_neighbors = ensure_list_size(num_neighbors, size=len(self.channels))
        self.attention_channels = attention_channels
        self.num_attention_blocks = num_attention_blocks
        self.proj_channels = proj_channels
        self.head_channels = ensure_list(head_channels, none_as_empty=True)
        self.act = act
        self.act_kwargs = act_kwargs
        self.head_act = head_act
        self.head_act_kwargs = head_act_kwargs
        self.norm = norm
        self.norm_kwargs = norm_kwargs
        self.bias = bias
        self.dropout = dropout

        self.stem = self.configure_stem()
        self.embed = self.configure_embed()
        self.attention_in = self.configure_attention_in()
        self.attention = self.configure_attention()
        self.proj = self.configure_proj()
        self.global_pool = CatPool(global_pool) if is_iterable(global_pool) else create_pool(global_pool)  # type: ignore[arg-type]
        self.head = self.configure_head()

    def configure_stem(self) -> MLP:
        """Build the point-wise stem MLP embedding the coordinates and features."""
        return MLP(
            [self.in_channels + self.spatial_dim] + self.stem_channels,
            plain_last=False,
            act=self.act,
            act_kwargs=self.act_kwargs,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
            bias=self.bias,
        )

    def configure_embed(self) -> nn.ModuleList:
        """Build the `PCTNeighborEmbedding` blocks."""
        blocks = nn.ModuleList()
        in_channels = self.stem_channels[-1]
        for out_channels, num_points, num_neighbors in zip(self.channels, self.num_points, self.num_neighbors):
            block = PCTNeighborEmbedding(
                in_channels=in_channels,
                out_channels=[out_channels, out_channels],
                num_points=num_points,
                num_neighbors=num_neighbors,
                act=self.act,
                act_kwargs=self.act_kwargs,
                norm=self.norm,
                norm_kwargs=self.norm_kwargs,
                bias=self.bias,
            )
            blocks.append(block)
            in_channels = out_channels
        return blocks

    def configure_attention_in(self) -> MLP:
        """Build the MLP projecting the embedding into the attention width."""
        return MLP(
            [self.channels[-1], self.attention_channels, self.attention_channels],
            plain_last=False,
            act=self.act,
            act_kwargs=self.act_kwargs,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
            bias=self.bias,
        )

    def configure_attention(self) -> nn.ModuleList:
        """Build the `OffsetAttention` blocks."""
        return nn.ModuleList(
            OffsetAttention(
                self.attention_channels,
                act=self.act,
                act_kwargs=self.act_kwargs,
                norm=self.norm,
                norm_kwargs=self.norm_kwargs,
            )
            for _ in range(self.num_attention_blocks)
        )

    def configure_proj(self) -> MLP:
        """Build the projection fusing the attention outputs and the embedding before pooling."""
        return MLP(
            [self.num_attention_blocks * self.attention_channels + self.channels[-1], self.proj_channels],
            plain_last=False,
            act=self.head_act,
            act_kwargs=self.head_act_kwargs,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
            bias=self.bias,
        )

    @property
    def num_features(self) -> int:
        """Channel count $C$ of the pooled features entering the head."""
        return (
            self.proj_channels * self.global_pool.num_pools
            if isinstance(self.global_pool, CatPool)
            else self.proj_channels
        )

    def configure_head(self) -> nn.Module:
        if self.num_classes == 0:
            return nn.Identity()

        channels_list = [self.num_features] + self.head_channels + [self.num_classes]
        dropout_list = [self.dropout] * (len(channels_list) - 1)
        dropout_list[-1] = 0.0
        return MLP(
            channels_list,
            act=self.head_act,
            act_kwargs=self.head_act_kwargs,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
            bias=True,
            dropout=dropout_list,
            plain_last=True,
        )

    def reset_classifier(
        self,
        num_classes: int,
        global_pool: PoolLike | Sequence[PoolLike] | None = None,
        **kwargs: Any,
    ) -> None:
        self.num_classes = num_classes
        if global_pool is not None:
            self.global_pool = CatPool(global_pool) if is_iterable(global_pool) else create_pool(global_pool)  # type: ignore[arg-type]
        self.head = self.configure_head()

    def forward_features(self, x: OptTensor, pos: Tensor, batch: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        x = torch.cat([x, pos], dim=1) if x is not None else pos
        x = self.stem(x)
        for block in self.embed:
            x, pos, batch = block(x, pos, batch)

        tokens = self.attention_in(x).view(-1, self.num_points[-1], self.attention_channels)  # (B, M, C)
        outputs = []
        for attention in self.attention:
            tokens = attention(tokens)
            outputs.append(tokens)

        x = torch.cat([torch.cat(outputs, dim=-1).flatten(0, 1), x], dim=1)
        return self.proj(x), pos, batch

    def forward_head(self, x: Tensor, batch: Tensor, pre_logits: bool = False) -> Tensor:
        x = self.global_pool(x, batch)
        return x if pre_logits else self.head(x)

    def forward(self, x: OptTensor, pos: Tensor, batch: Tensor) -> Tensor:
        x, _, batch = self.forward_features(x, pos, batch)
        return self.forward_head(x, batch)


@register_model(
    "pct.modelnet40.strawberry-eat-mango",
    task="classification",
    weights=WeightsDict(
        url="hf://torch-pointcloud/pct.modelnet40.strawberry-eat-mango/resolve/0000000000000000000000000000000000000000/model.safetensors",
        dataset="modelnet40",
        metrics={"OA": 93.27, "mAcc": 89.99},
        classes=MODELNET40_CLASSES,
        author="strawberry-eat-mango",
        license="MIT",
    ),
    hparams=dict(
        in_channels=0,
        num_classes=40,
        spatial_dim=3,
        stem_channels=[64, 64],
        channels=[128, 256],
        num_points=[512, 256],
        num_neighbors=32,
        attention_channels=256,
        num_attention_blocks=4,
        proj_channels=1024,
        head_channels=[512, 256],
        act="relu",
        head_act="leaky_relu",
        head_act_kwargs={"negative_slope": 0.2},
        norm="batch_norm",
        bias=False,
        dropout=0.5,
        global_pool="max",
    ),
    transform=T.Compose(
        [
            T.CopyItems(keys=DataKeys.POS, dst_keys=DataKeys.ORIGIN_POS),
            T.FarthestPointSample(
                pos_key=DataKeys.POS,
                keys=[DataKeys.NORMAL],
                num_samples=1024,
                dst_index_key=DataKeys.INDEX,
                allow_missing_keys=True,
            ),
        ]
    ),
)
def pct_strawberry_eat_mango_modelnet40_cls(**hparams: Any) -> PCTClassification:
    # from the repo: https://github.com/Strawberry-Eat-Mango/PCT_Pytorch
    return PCTClassification(**hparams)
