from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import MLP

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
        # So a single SA block, with X PVConv
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
        if self.sa_module is not None:
            x, pos, batch = self.sa_module(x, pos, batch)

        for layer in self.layers:
            x = layer(x) if isinstance(layer, MLP) else layer(x, pos, batch)
        return x, pos, batch
