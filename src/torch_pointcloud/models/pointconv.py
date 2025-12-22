from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union, overload

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import MLP
from torch_geometric.typing import OptTensor

from torch_pointcloud.layers import FPS, PointConvSetAbstraction, PoolLike, create_pool
from torch_pointcloud.utils.conversion import ensure_list

from ._base import ClassificationModel


class PointConvEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: Sequence[Sequence[int]],
        num_neighbors: Sequence[int],
        bandwidths: Sequence[float],
        ratios: Sequence[float],
        density_channels: Sequence[Sequence[int]],
        weight_channels: Sequence[Sequence[int]],
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels

        # Validate input lengths
        num_layers = len(channels)
        assert len(num_neighbors) == num_layers
        assert len(bandwidths) == num_layers
        assert len(ratios) == num_layers

        # Handle default list-of-lists expansion if single lists are provided
        if not isinstance(density_channels[0], (list, tuple)):
            density_channels = [density_channels] * num_layers  # type: ignore
        if not isinstance(weight_channels[0], (list, tuple)):
            weight_channels = [weight_channels] * num_layers  # type: ignore

        self.blocks = nn.ModuleList()

        current_in_channels = in_channels

        for i in range(num_layers):
            # If ratio is 1.0 or 0.0 (global), we typically don't downsample via FPS here
            # or we handle it inside the block.
            # Here we assume ratio < 1.0 triggers downsampling.
            downsample = None
            if ratios[i] < 1.0 and ratios[i] > 0.0:
                downsample = FPS(ratios[i])

            # The original PointConv often groups ALL points in the last layer.
            # If ratio is 0.0, we treat it as grouping all (no downsampling module needed
            # if the aggregation handles it, or downsample to 1 point).
            # The provided PointConvSetAbstraction handles generic KNN.
            # To mimic "Group All" (Global SA), we typically set k=Large or handle it via Global Pooling.
            # Here, we assume the user provides appropriate k (e.g., None or total points) for global layers.

            block = PointConvSetAbstraction(
                in_channels=current_in_channels,
                num_neighbors=num_neighbors[i],
                bandwidth=bandwidths[i],
                channels=channels[i],
                density_channels=density_channels[i],
                weight_channels=weight_channels[i],
                act=act,
                act_kwargs=act_kwargs,
                act_first=act_first,
                norm=norm,
                norm_kwargs=norm_kwargs,
                bias=bias,
                downsample=downsample,
            )
            self.blocks.append(block)
            current_in_channels = channels[i][-1]

        self.out_channels = current_in_channels

    def forward(
        self, x: OptTensor, pos: Tensor, batch: Tensor, return_intermediates: bool = False
    ) -> Union[Tuple[Tensor, Tensor, Tensor], Tuple[Tensor, Tensor, Tensor, List[Any]]]:
        intermediates = []

        for block in self.blocks:
            x, pos, batch = block(x, pos, batch)
            if return_intermediates:
                intermediates.append((x, pos, batch))

        if return_intermediates:
            return x, pos, batch, intermediates
        return x, pos, batch


class PointConvClassification(ClassificationModel):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        *,
        channels: Sequence[Sequence[int]] = ([64, 64, 128], [128, 128, 256], [256, 512, 1024]),
        num_neighbors: Sequence[int] = (32, 64, 1024),
        bandwidths: Sequence[float] = (0.1, 0.2, 0.4),
        ratios: Sequence[float] = (0.5, 0.25, 0.0),  # 0.0 implies global aggregation or last layer
        density_channels: Sequence[int] = (16, 8),
        weight_channels: Sequence[int] = (8, 8),
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: bool = True,
        dropout: float = 0.5,
        global_pool: PoolLike = "max",
        classifier_channels: Sequence[int] = (512, 256),
    ):
        super().__init__(in_channels=in_channels, num_classes=num_classes)
        self.channels = channels
        self.num_neighbors = ensure_list(num_neighbors)
        self.bandwidths = ensure_list(bandwidths)
        self.ratios = ensure_list(ratios)
        self.density_channels = ensure_list(density_channels)
        self.weight_channels = ensure_list(weight_channels)

        self.act = act
        self.act_kwargs = act_kwargs
        self.act_first = act_first
        self.norm = norm
        self.norm_kwargs = norm_kwargs
        self.bias = bias
        self.dropout = dropout
        self.classifier_channels = classifier_channels

        self.encoder = self.configure_encoder()
        self.global_pool = create_pool(global_pool)
        self.head = self.configure_head()

    @property
    def embedding_dim(self) -> int:
        return self.encoder.out_channels

    def configure_encoder(self) -> PointConvEncoder:
        return PointConvEncoder(
            in_channels=self.in_channels,
            channels=self.channels,
            num_neighbors=self.num_neighbors,
            bandwidths=self.bandwidths,
            ratios=self.ratios,
            density_channels=self.density_channels,
            weight_channels=self.weight_channels,
            act=self.act,
            act_kwargs=self.act_kwargs,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
            bias=self.bias,
        )

    def configure_head(self) -> nn.Module:
        layers = []
        in_dim = self.embedding_dim

        for out_dim in self.classifier_channels:
            layers.append(nn.Linear(in_dim, out_dim))
            if self.norm is not None:
                layers.append(nn.BatchNorm1d(out_dim))
            if self.act is not None:
                layers.append(nn.ReLU(inplace=True))  # Simpler access to ReLU, or use factory
            if self.dropout > 0:
                layers.append(nn.Dropout(p=self.dropout))
            in_dim = out_dim

        layers.append(nn.Linear(in_dim, self.num_classes))
        return nn.Sequential(*layers)

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
    ) -> Tuple[Tensor, Tensor, Tensor, List[Any]]: ...

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
        return self.encoder(x, pos, batch, return_intermediates=return_intermediates)

    def forward_head(self, x: Tensor, batch: Tensor, pre_logits: bool = False) -> Tensor:
        # If the last encoder layer reduced to 1 point per batch (global SA),
        # x is (B, C). If not, x is (N_out, C), and we need to pool.

        # Check if x is already pooled (B, C) or still dense (B*N, C)
        if x.dim() == 2 and x.size(0) == batch.max().item() + 1:
            # Already one point per batch (likely from a global SA layer)
            pass
        else:
            x = self.global_pool(x, batch)

        return x if pre_logits else self.head(x)

    def forward(self, x: OptTensor, pos: Tensor, batch: Tensor) -> Tensor:
        x, _, batch = self.forward_features(x, pos, batch)
        return self.forward_head(x, batch)
