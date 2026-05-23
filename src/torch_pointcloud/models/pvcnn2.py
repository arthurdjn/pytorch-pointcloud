from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union, overload

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import MLP
from torch_geometric.typing import OptTensor

from torch_pointcloud.layers import PoolLike, create_cls_head, create_pool
from torch_pointcloud.layers.pointnet2_blocks import FPModule, SAModule, ensure_msg_list
from torch_pointcloud.layers.pvcnn_blocks import PVConv
from torch_pointcloud.utils.conversion import ensure_tuple, ensure_tuple_size


def ensure_msg_list_size(value: Sequence[Any], size: int, extra_msg: str = "") -> Sequence[Any]:
    if len(value) != size:
        raise ValueError(f"Expected a list of size {size}, got {len(value)}. {extra_msg}")
    return ensure_msg_list(value, extra_msg=extra_msg)


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
        fp_channels: Sequence[Sequence[int]],
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


class PVCNN2Classification(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        *,
        ratios: Sequence[float],
        radii: Sequence[Union[float, Sequence[float]]],
        num_neighbors: Sequence[Union[int, Sequence[int]]],
        sa_channels: Sequence[Sequence[Union[int, Sequence[int]]]],
        encoder_channels: Sequence[int],
        encoder_depths: Sequence[int],
        encoder_resolutions: Sequence[Optional[int]],
        encoder_kernel_sizes: Sequence[int],
        with_se: bool = False,
        normalize: bool = True,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        dropout: float = 0.0,
        global_pool: PoolLike = "max",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.embedding_dim = encoder_channels[-1]
        sa_channels = ensure_msg_list(sa_channels)

        self.encoder = PVCNN2Encoder(
            channels=encoder_channels,
            depths=encoder_depths,
            resolutions=encoder_resolutions,
            kernel_sizes=encoder_kernel_sizes,
            sa_channels=sa_channels,
            ratios=ratios,
            radii=radii,
            num_neighbors=num_neighbors,
            with_se=with_se,
            normalize=normalize,
            act=act,
            act_kwargs=act_kwargs,
            act_first=act_first,
            norm=norm,
            norm_kwargs=norm_kwargs,
        )

        self.dropout = dropout
        self.global_pool = create_pool(global_pool)
        self.head = create_cls_head(self.embedding_dim, self.num_classes)

    def reset_head(self, num_classes: int, **kwargs: Any) -> None:
        self.num_classes = num_classes
        self.head = create_cls_head(num_features=self.embedding_dim, num_classes=self.num_classes, **kwargs)

    @overload
    def forward_features(
        self,
        x: OptTensor,
        pos: Tensor,
        batch: Tensor,
        return_intermediates: Literal[True],
    ) -> Tuple[Tensor, Tensor, Tensor, List[Dict[str, Tensor]]]: ...

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
        return self.encoder(x, pos, batch, return_intermediates=return_intermediates)

    def forward_head(self, x: Tensor, batch: Tensor, pre_logits: bool = False) -> Tensor:
        x = self.global_pool(x, batch)
        if self.dropout:
            x = F.dropout(x, p=float(self.dropout), training=self.training)
        return x if pre_logits else self.head(x)

    def forward(self, x: OptTensor, pos: Tensor, batch: Tensor) -> Tensor:
        x, pos, batch, intermediates = self.forward_features(x, pos, batch, return_intermediates=True)
        return self.forward_head(x, batch)


class PVCNN2Segmentation(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        *,
        ratios: Sequence[float],
        radii: Sequence[Union[float, Sequence[float]]],
        num_neighbors: Sequence[Union[int, Sequence[int]]],
        sa_channels: Sequence[Sequence[Union[int, Sequence[int]]]],
        encoder_channels: Sequence[int],
        encoder_depths: Sequence[int],
        encoder_resolutions: Sequence[Optional[int]],
        encoder_kernel_sizes: Sequence[int],
        fp_channels: Sequence[Sequence[int]],
        decoder_channels: Sequence[int],
        decoder_depths: Sequence[int],
        decoder_resolutions: Sequence[Optional[int]],
        decoder_kernel_sizes: Sequence[int],
        with_se: bool = False,
        normalize: bool = True,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels or 3
        self.num_classes = num_classes
        sa_channels = ensure_msg_list(sa_channels)

        self.encoder = PVCNN2Encoder(
            channels=encoder_channels,
            depths=encoder_depths,
            resolutions=encoder_resolutions,
            kernel_sizes=encoder_kernel_sizes,
            sa_channels=sa_channels,
            ratios=ratios,
            radii=radii,
            num_neighbors=num_neighbors,
            with_se=with_se,
            normalize=normalize,
            act=act,
            act_kwargs=act_kwargs,
            act_first=act_first,
            norm=norm,
            norm_kwargs=norm_kwargs,
        )

        skip_channels = [self.in_channels]
        for i in range(len(sa_channels) - 1):
            skip_channels.append(sum([c[0] for c in sa_channels[i]]))

        self.decoder = PVCNN2Decoder(
            depths=decoder_depths,
            channels=decoder_channels,
            skip_channels=skip_channels[::-1],
            fp_channels=fp_channels,
            resolutions=decoder_resolutions,
            kernel_sizes=decoder_kernel_sizes,
            with_se=with_se,
            normalize=normalize,
            act=act,
            act_kwargs=act_kwargs,
        )

        self.dropout = dropout
        self.head = create_cls_head(num_features=decoder_channels[-1], num_classes=self.num_classes)

    @property
    def embedding_dim(self) -> int:
        return self.decoder.blocks[-1].layers[-1].out_features  # type: ignore[index, union-attr]

    def reset_head(self, num_classes: int, **kwargs: Any) -> None:
        self.num_classes = num_classes
        self.head = create_cls_head(num_features=self.embedding_dim, num_classes=self.num_classes, **kwargs)

    @overload
    def forward_features(
        self,
        x: OptTensor,
        pos: Tensor,
        batch: Tensor,
        return_intermediates: Literal[True],
    ) -> Tuple[Tensor, Tensor, Tensor, List[Dict[str, Tensor]]]: ...

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
        return self.encoder(x, pos, batch, return_intermediates=return_intermediates)

    def forward_decoder(
        self,
        x: Tensor,
        pos: Tensor,
        batch: Tensor,
        intermediates: List[Dict[str, Tensor]],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        return self.decoder(x, pos, batch, intermediates)

    def forward_head(self, x: Tensor, pre_logits: bool = False) -> Tensor:
        if self.dropout:
            x = F.dropout(x, p=float(self.dropout), training=self.training)
        return x if pre_logits else self.head(x)

    def forward(self, x: OptTensor, pos: Tensor, batch: Tensor) -> Tensor:
        x, pos, batch, intermediates = self.forward_features(x, pos, batch, return_intermediates=True)
        x, _, _ = self.forward_decoder(x, pos, batch, intermediates)
        return self.forward_head(x)
