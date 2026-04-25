from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Union

import torch.nn as nn
from torch import Tensor
from torch_geometric.nn.resolver import activation_resolver, normalization_resolver

from torch_pointcloud.utils.conversion import convert_to_spconv_tensor
from torch_pointcloud.utils.imports import optional_import

if TYPE_CHECKING:
    import spconv.pytorch as spconv


spconv, _ = optional_import("spconv.pytorch")


class SubMConv3dBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int,
        norm: Union[str, Callable, None] = None,
        act: Union[str, Callable, None] = None,
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: bool = True,
        stem_indice_key: Optional[str] = None,
    ):
        super().__init__()
        norm_kwargs = norm_kwargs or {}
        act_kwargs = act_kwargs or {}

        self.stem = spconv.SubMConv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
            indice_key=stem_indice_key,
        )
        self.norm = normalization_resolver(norm, out_channels, **norm_kwargs) if norm is not None else None
        self.act = activation_resolver(act, **act_kwargs) if act is not None else None

    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        batch: Tensor,
    ) -> Tensor:
        x_spconv = convert_to_spconv_tensor(x, pos, batch)
        x_spconv = self.stem(x_spconv)

        x = x_spconv.features
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)

        return x
