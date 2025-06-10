from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union, overload

import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import MLP
from torch_geometric.typing import OptTensor

from torch_pointcloud.utils.conversion import ensure_tuple, ensure_tuple_size

from .pointnet2 import FPModule, SAModule, ensure_msg_list
from .pvcnn import PVConv


def ensure_msg_list_size(items: Sequence[Any], size: int, extra_msg: str = "") -> Sequence[Any]:
    if len(items) != size:
        raise ValueError(f"The number of `{{param}}` must match the number of blocks ({size})." + extra_msg)
    return items


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
        for i in range(depth):
            in_channels = self.in_channels if i == 0 else self.out_channels
            if not resolution:
                # In case resolution is 0 or None, use a linear block
                layer = MLP([in_channels, self.out_channels], plain_last=False, **kwargs)
            else:
                layer = PVConv(
                    in_channels=in_channels,
                    out_channels=self.out_channels,
                    kernel_size=kernel_size,
                    resolution=resolution,
                    with_se=with_se,
                    normalize=normalize,
                    **kwargs,  # type: ignore[arg-type]
                )

            self.layers.append(layer)

    def forward(self, x: Tensor, pos: Tensor, batch: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        if self.sa_module is not None:
            x, pos, batch = self.sa_module(x, pos, batch)

        for layer in self.layers:
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
        sa_channels: Sequence[Sequence[Sequence[int]]],
        ratios: Sequence[float],
        radii: Sequence[Union[float, Sequence[float]]],
        num_neighbors: Sequence[Union[int, Sequence[int]]],
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.depths = ensure_tuple(depths)
        num_blocks = len(self.depths)

        self.channels = ensure_tuple_size(channels, size=num_blocks + 1)
        self.resolutions = ensure_tuple_size(resolutions, size=num_blocks)
        self.kernel_sizes = ensure_tuple_size(kernel_sizes, size=num_blocks)
        self.sa_channels = ensure_msg_list_size(sa_channels, size=num_blocks)
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


class PVCNN2DecoderBlock(nn.Module):
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
        fp_module: Optional[FPModule] = None,
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

        self.fp_module = fp_module
        self.layers = nn.ModuleList([])
        for i in range(depth):
            in_channels = self.in_channels if i == 0 else self.out_channels
            if not resolution:
                # In case resolution is 0 or None, use a linear block
                layer = MLP([in_channels, self.out_channels], plain_last=False, **kwargs)
            else:
                layer = PVConv(
                    in_channels=in_channels,
                    out_channels=self.out_channels,
                    kernel_size=kernel_size,
                    resolution=resolution,
                    with_se=with_se,
                    normalize=normalize,
                    **kwargs,  # type: ignore[arg-type]
                )

            self.layers.append(layer)

    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        batch: Tensor,
        x_skip: OptTensor = None,
        pos_skip: OptTensor = None,
        batch_skip: OptTensor = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        if self.fp_module is not None:
            x, pos, batch = self.fp_module(x, pos, batch, x_skip, pos_skip, batch_skip)

        for layer in self.layers:
            x = layer(x) if isinstance(layer, MLP) else layer(x, pos, batch)
        return x, pos, batch


class PVCNN2Decoder(nn.Module):
    def __init__(
        self,
        depths: Sequence[int],
        channels: Sequence[int],
        skip_channels: Sequence[int],
        fp_channels: Sequence[Sequence[Sequence[int]]],
        resolutions: Sequence[Optional[int]],
        kernel_sizes: Sequence[int],
        with_se: bool = False,
        normalize: bool = True,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.depths = ensure_tuple(depths)
        n = len(self.depths)

        extra_msg = f"The number of `{{param}}` must match the number of blocks ({n})."
        self.channels = ensure_tuple_size(channels, size=n, extra_msg=extra_msg.format(param="channels"))
        self.skip_channels = ensure_tuple_size(skip_channels, size=n, extra_msg=extra_msg.format(param="skip_channels"))
        self.fp_channels = ensure_tuple_size(fp_channels, size=n, extra_msg=extra_msg.format(param="fp_channels"))
        self.resolutions = ensure_tuple_size(resolutions, size=n, extra_msg=extra_msg.format(param="resolutions"))
        self.kernel_sizes = ensure_tuple_size(kernel_sizes, size=n, extra_msg=extra_msg.format(param="kernel_sizes"))

        self.blocks = nn.ModuleList([])
        for i in range(n):
            fp_module = FPModule(
                in_channels=self.fp_channels[i][0] + self.skip_channels[i],  # TODO: handle MSG
                channels=self.fp_channels[i],
                k=1 if i == 0 else 3,  # TODO: replace with spatial_dim
                # TODO: add act, norm, etc.
            )

            block = PVCNN2DecoderBlock(
                in_channels=self.channels[i],
                out_channels=self.channels[i],
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
                fp_module=fp_module,
            )
            self.blocks.append(block)

    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        batch: Tensor,
        intermediates: List[Dict[str, Tensor]],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        for block, intermediate in zip(self.blocks, reversed(intermediates)):
            x_skip, pos_skip, batch_skip = intermediate["features"], intermediate["pos"], intermediate["batch"]
            x, pos, batch = block(x, pos, batch, x_skip, pos_skip, batch_skip)
        return x, pos, batch
