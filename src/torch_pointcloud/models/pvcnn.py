from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union, overload

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import MLP

from torch_pointcloud.layers import PoolLike, create_cls_head, create_pool
from torch_pointcloud.layers.pvcnn_blocks import PVConv
from torch_pointcloud.utils.conversion import ensure_tuple, ensure_tuple_size
from torch_pointcloud.utils.types import OptTensor


class PVConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int,
        kernel_size: int,
        resolution: int,
        with_se: bool = False,
        normalize: bool = True,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        kwargs = dict(
            act=act,
            act_kwargs=act_kwargs,
            act_first=act_first,
            norm=norm,
            norm_kwargs=norm_kwargs,
        )

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            if not resolution:
                # In case resolution is 0 or None, use a linear block
                layer = MLP([in_channels, out_channels], plain_last=False, **kwargs)
            else:
                layer = PVConv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    resolution=resolution,
                    with_se=with_se,
                    normalize=normalize,
                    **kwargs,  # type: ignore[arg-type]
                )

            self.layers.append(layer)
            in_channels = out_channels

    @overload
    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        batch: Tensor,
        return_intermediates: Literal[True],
    ) -> Tuple[Tensor, List[Tensor]]: ...

    @overload
    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        batch: Tensor,
        return_intermediates: Literal[False] = False,
    ) -> Tensor: ...

    def forward(self, x: Tensor, pos: Tensor, batch: Tensor, return_intermediates: bool = False) -> Any:
        intermediates = []
        for layer in self.layers:
            x = layer(x) if isinstance(layer, MLP) else layer(x, pos, batch)
            if return_intermediates:
                intermediates.append(x)

        if return_intermediates:
            return x, intermediates
        return x


class PVCNNClassification(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        *,
        channels: Sequence[int],
        global_channels: Optional[Sequence[int]] = None,
        depths: Sequence[int],
        kernel_sizes: Sequence[int],
        resolutions: Sequence[int],
        with_se: bool = False,
        normalize: bool = True,
        dropout: float = 0.0,
        global_pool: PoolLike = "max",
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.depths = ensure_tuple(depths)
        self.channels = ensure_tuple_size([in_channels] + list(channels), size=len(self.depths) + 1)
        self.global_channels = ensure_tuple(global_channels, none_as_empty=True)
        self.kernel_sizes = ensure_tuple_size(kernel_sizes, size=len(self.depths))
        self.resolutions = ensure_tuple_size(resolutions, size=len(self.depths))
        self.with_se = with_se
        self.normalize = normalize
        self.dropout = dropout
        self.act = act
        self.act_kwargs = act_kwargs
        self.act_first = act_first
        self.norm = norm
        self.norm_kwargs = norm_kwargs

        self.blocks = self.configure_blocks()
        self.global_mlp = self.configure_global_mlp()
        self.global_pool = create_pool(global_pool)
        self.head = create_cls_head(self.embedding_dim, self.num_classes)

    @property
    def embedding_dim(self) -> int:
        return self.channels[-1]

    def configure_blocks(self) -> nn.ModuleList:
        blocks = nn.ModuleList()
        for i in range(len(self.depths)):
            in_channels = self.channels[i]
            out_channels = self.channels[i + 1]
            block = PVConvBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                depth=self.depths[i],
                kernel_size=self.kernel_sizes[i],
                resolution=self.resolutions[i],
                with_se=self.with_se,
                normalize=self.normalize,
                act=self.act,
                act_kwargs=self.act_kwargs,
                act_first=self.act_first,
                norm=self.norm,
                norm_kwargs=self.norm_kwargs,
            )
            blocks.append(block)
        return blocks

    def configure_global_mlp(self) -> Optional[MLP]:
        if not self.global_channels:
            return None

        return MLP(
            [self.channels[-1]] + list(self.global_channels),
            act=self.act,
            act_kwargs=self.act_kwargs,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
            plain_last=False,
        )

    @overload
    def forward_features(
        self,
        x: OptTensor,
        pos: Tensor,
        batch: Tensor,
        return_intermediates: Literal[True],
    ) -> Tuple[Tensor, List[Tensor]]: ...

    @overload
    def forward_features(
        self,
        x: OptTensor,
        pos: Tensor,
        batch: Tensor,
        return_intermediates: Literal[False] = False,
    ) -> Tensor: ...

    def forward_features(self, x: OptTensor, pos: Tensor, batch: Tensor, return_intermediates: bool = False) -> Any:
        x = x if x is not None else pos
        intermediates = []
        for block in self.blocks:
            x = block(x, pos, batch, return_intermediates=return_intermediates)
            if return_intermediates:
                x, x_inters = x
                intermediates.extend(x_inters)

        if return_intermediates:
            return x, intermediates
        return x

    def forward_head(self, x: Tensor, batch: Tensor, pre_logits: bool = False) -> Tensor:
        x = self.global_pool(x, batch)
        if self.dropout:
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x if pre_logits else self.head(x)

    def forward(self, x: OptTensor, pos: Tensor, batch: Tensor) -> Tensor:
        x = self.forward_features(x, pos, batch)
        x = self.forward_head(x, batch)
        return x


class PVCNNSegmentation(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        *,
        channels: Sequence[int],
        global_channels: Optional[Sequence[int]] = None,
        depths: Sequence[int],
        kernel_sizes: Sequence[int],
        resolutions: Sequence[int],
        spatial_dim: int = 3,
        with_se: bool = False,
        normalize: bool = True,
        dropout: float = 0.0,
        global_pool: PoolLike = "max",
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.in_channels = max(in_channels, spatial_dim)
        self.num_classes = num_classes

        self.depths = ensure_tuple(depths)
        self.channels = ensure_tuple_size([self.in_channels] + list(channels), size=len(self.depths) + 1)
        self.global_channels = ensure_tuple(global_channels, none_as_empty=True)
        self.kernel_sizes = ensure_tuple_size(kernel_sizes, size=len(self.depths))
        self.resolutions = ensure_tuple_size(resolutions, size=len(self.depths))
        self.with_se = with_se
        self.normalize = normalize
        self.dropout = dropout
        self.act = act
        self.act_kwargs = act_kwargs
        self.act_first = act_first
        self.norm = norm
        self.norm_kwargs = norm_kwargs

        self.blocks = self.configure_blocks()
        self.global_mlp = self.configure_global_mlp()
        self.global_pool = create_pool(global_pool)
        self.head = create_cls_head(self.embedding_dim, self.num_classes)

    @property
    def embedding_dim(self) -> int:
        embedding_dim = sum(channels * depth for channels, depth in zip(self.channels[1:], self.depths))
        if self.global_channels:
            return embedding_dim + self.global_channels[-1]
        return embedding_dim + self.channels[-1]

    def configure_blocks(self) -> nn.ModuleList:
        blocks = nn.ModuleList()
        for i in range(len(self.depths)):
            in_channels = self.channels[i]
            out_channels = self.channels[i + 1]
            block = PVConvBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                depth=self.depths[i],
                kernel_size=self.kernel_sizes[i],
                resolution=self.resolutions[i],
                with_se=self.with_se,
                normalize=self.normalize,
                act=self.act,
                act_kwargs=self.act_kwargs,
                act_first=self.act_first,
                norm=self.norm,
                norm_kwargs=self.norm_kwargs,
            )
            blocks.append(block)
        return blocks

    def configure_global_mlp(self) -> Optional[MLP]:
        if not self.global_channels:
            return None

        return MLP(
            [self.channels[-1]] + list(self.global_channels),
            act=self.act,
            act_kwargs=self.act_kwargs,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
            plain_last=False,
        )

    @overload
    def forward_features(
        self,
        x: OptTensor,
        pos: Tensor,
        batch: Tensor,
        return_intermediates: Literal[True],
    ) -> Tuple[Tensor, List[Tensor]]: ...

    @overload
    def forward_features(
        self,
        x: OptTensor,
        pos: Tensor,
        batch: Tensor,
        return_intermediates: Literal[False] = False,
    ) -> Tensor: ...

    def forward_features(self, x: OptTensor, pos: Tensor, batch: Tensor, return_intermediates: bool = False) -> Any:
        x = x if x is not None else pos
        intermediates = []
        for block in self.blocks:
            x = block(x, pos, batch, return_intermediates=return_intermediates)
            if return_intermediates:
                x, x_inters = x
                intermediates.extend(x_inters)

        if return_intermediates:
            return x, intermediates
        return x

    # NOTE: Does it make sense to have this forward_decoder?
    # NOTE: Maybe we should raise an error/warning specifying that this method
    # NOTE: is irrelevant for this model. ...And include the global_mlp in the head instead?
    def forward_decoder(self, x: Tensor, batch: Tensor, intermediates: List[Tensor]) -> Tensor:
        x_global = self.global_pool(x, batch)
        if self.global_mlp:
            x_global = self.global_mlp(x_global)

        intermediates.append(x_global[batch])
        return torch.cat(intermediates, dim=1)

    def forward_head(self, x: Tensor, pre_logits: bool = False) -> Tensor:
        if self.dropout:
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x if pre_logits else self.head(x)

    def forward(self, x: OptTensor, pos: Tensor, batch: Tensor) -> Tensor:
        x, intermediates = self.forward_features(x, pos, batch, return_intermediates=True)
        x = self.forward_decoder(x, batch, intermediates)
        x = self.forward_head(x)
        return x
