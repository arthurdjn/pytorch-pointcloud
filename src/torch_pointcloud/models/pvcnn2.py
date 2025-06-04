from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union, overload

import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import MLP

from torch_pointcloud.utils.conversion import ensure_tuple, ensure_tuple_size

from .pointnet2 import SAModule
from .pvcnn import PVConv


class PVCNN2EncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int,
        resolution: int,
        kernel_size: int,
        with_se: bool = False,
        normalize: bool = True,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        # SA module parameters (?)
        # sa_channels: Optional[Sequence[Sequence[int]]] = None,  # For MSG: [[32, 64], [64, 128]]
        # sa_ratio: Optional[float] = None,
        # sa_radii: Optional[Union[float, Sequence[float]]] = None,
        # sa_num_neighbors: Optional[Union[int, Sequence[int]]] = None,
        sa_module: Optional[SAModule] = None,
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

        self.sa_module = sa_module
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

    def forward(self, x: Tensor, pos: Tensor, batch: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        print(f"Block: {self.in_channels} -> {self.out_channels}")
        if self.sa_module is not None:
            print(f"SA module: {x.shape=} | {pos.shape=} | {batch.shape=}")
            x, pos, batch = self.sa_module(x, pos, batch)

        for layer in self.layers:
            print(f"Layer: {x.shape=} | {pos.shape=} | {batch.shape=}")
            x = layer(x) if isinstance(layer, MLP) else layer(x, pos, batch)
        return x, pos, batch


class PVCNN2Encoder(nn.Module):
    def __init__(
        self,
        *,
        channels: Sequence[int],
        depths: Sequence[int],
        resolutions: Sequence[Optional[int]],
        kernel_sizes: Sequence[int],
        with_se: bool = False,
        normalize: bool = True,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        # SA parameters for each block
        sa_channels: Sequence[Sequence[Sequence[int]]],
        ratios: Sequence[float],
        radii: Sequence[Union[float, Sequence[float]]],
        num_neighbors: Sequence[Union[int, Sequence[int]]],
    ):
        super().__init__()
        self.depths = ensure_tuple(depths)
        num_blocks = len(self.depths)

        self.channels = ensure_tuple_size(channels, size=num_blocks + 1)
        self.resolutions = ensure_tuple_size(resolutions, size=num_blocks)
        self.kernel_sizes = ensure_tuple_size(kernel_sizes, size=num_blocks)
        self.sa_channels = ensure_tuple_size(sa_channels, size=num_blocks)
        self.ratios = ensure_tuple_size(ratios, size=num_blocks)
        self.radii = ensure_tuple_size(radii, size=num_blocks)
        self.num_neighbors = ensure_tuple_size(num_neighbors, size=num_blocks)

        self.blocks = nn.ModuleList([])
        for i in range(num_blocks):
            sa_block: Optional[SAModule] = None
            if i > 0:
                sa_block = SAModule(
                    in_channels=self.channels[i],
                    channels=self.sa_channels[i - 1],
                    ratio=self.ratios[i - 1],
                    radii=self.radii[i - 1],
                    num_neighbors=self.num_neighbors[i - 1],
                    # TODO: add act, norm, etc.
                )

            block = PVCNN2EncoderBlock(
                in_channels=self.channels[i] if i == 0 else self.channels[i + 1],
                out_channels=self.channels[i + 1],
                depth=self.depths[i],
                resolution=self.resolutions[i],
                kernel_size=self.kernel_sizes[i],
                with_se=with_se,
                normalize=normalize,
                act=act,
                act_kwargs=act_kwargs,
                act_first=act_first,
                norm=norm,
                norm_kwargs=norm_kwargs,
                sa_module=sa_block,
            )

            self.blocks.append(block)

    @overload
    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        batch: Tensor,
        return_intermediates: Literal[True],
    ) -> Tuple[Tensor, Tensor, Tensor, List[Dict[str, Tensor]]]: ...

    @overload
    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        batch: Tensor,
        return_intermediates: Literal[False] = False,
    ) -> Tuple[Tensor, Tensor, Tensor]: ...

    def forward(self, x: Tensor, pos: Tensor, batch: Tensor, return_intermediates: bool = False) -> Any:
        intermediates = []

        for block in self.blocks:
            if return_intermediates:
                intermediates.append({"features": x, "pos": pos, "batch": batch})
            x, pos, batch = block(x, pos, batch)

        if return_intermediates:
            return x, pos, batch, intermediates
        return x, pos, batch
