from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple, Union

import torch.nn as nn
from torch import Tensor

from torch_pointcloud.utils.conversion import convert_to_spconv_tensor
from torch_pointcloud.utils.imports import _SPCONV_AVAILABLE, optional_import

from .act import create_act
from .norms import create_norm

if TYPE_CHECKING:
    import spconv.pytorch as spconv
    from spconv.pytorch import SparseSequential
else:
    SparseSequential, _ = optional_import("spconv.pytorch", "SparseSequential")


spconv, _ = optional_import("spconv.pytorch")

SparseModule: Any = spconv.SparseModule if _SPCONV_AVAILABLE else nn.Module


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
        self.norm = create_norm(norm, out_channels, **norm_kwargs)
        self.act = create_act(act, **act_kwargs)

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


class SparseConvBlock(SparseSequential):
    r"""Sparse 3D convolution followed by normalization and activation (the OpenPCDet `post_act_block`).

    A `subm` (submanifold), `spconv` (regular, optionally strided), or `inverseconv` convolution over a
    `SparseConvTensor`, followed by normalization and activation on the features. As a `SparseSequential`
    subclass it drops directly into the sparse voxel backbones' convolution stacks.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Convolution kernel size.
        stride: Convolution stride (used by `spconv`).
        padding: Convolution padding (used by `spconv`).
        indice_key: spconv index key for the convolution.
        conv_type: One of `subm`, `spconv`, or `inverseconv`.
        norm: Normalization passed to `create_norm`.
        norm_kwargs: Extra keyword arguments for the normalization (e.g. `eps`, `momentum`).
        act: Activation passed to `create_act`.
        act_kwargs: Extra keyword arguments for the activation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, ...]],
        *,
        stride: Union[int, Tuple[int, ...]] = 1,
        padding: Union[int, Tuple[int, ...]] = 0,
        indice_key: str,
        conv_type: str = "subm",
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        if conv_type == "subm":
            conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
        elif conv_type == "spconv":
            conv = spconv.SparseConv3d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
                indice_key=indice_key,
            )
        elif conv_type == "inverseconv":
            conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
        else:
            raise ValueError(f"Unknown conv_type {conv_type!r}. Expected 'subm', 'spconv', or 'inverseconv'.")

        super().__init__(
            conv,
            create_norm(norm, out_channels, dim=1, **(norm_kwargs or {})),
            create_act(act, **(act_kwargs or {})),
        )


class SparseResidualBlock(SparseModule):
    r"""Pre-activation sparse residual block shared by the SPFormer and SphereFormer SpConv U-Nets.

    The convolutional branch applies $\text{norm} \to \text{act} \to \text{conv} \to \text{norm} \to \text{act} \to \text{conv}$
    over two $3\times3\times3$ submanifold convolutions; the identity branch is `nn.Identity` when the channels match,
    else a $1\times1\times1$ `SubMConv3d` projection.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        indice_key: spconv index key shared by the two submanifold convolutions.
        act: Activation passed to `create_act`.
        act_kwargs: Extra keyword arguments for the activation.
        norm: Normalization passed to `create_norm`.
        norm_kwargs: Extra keyword arguments for the normalization (e.g. `eps`, `momentum`).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        indice_key: Optional[str] = None,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        act_kwargs = act_kwargs or {}
        norm_kwargs = norm_kwargs or {}

        if in_channels == out_channels:
            self.i_branch: nn.Module = nn.Identity()
        else:
            self.i_branch = spconv.SubMConv3d(in_channels, out_channels, kernel_size=1, bias=False)

        self.conv_branch = spconv.SparseSequential(
            create_norm(norm, in_channels, **norm_kwargs) or nn.Identity(),
            create_act(act, **act_kwargs) or nn.Identity(),
            spconv.SubMConv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key),
            create_norm(norm, out_channels, **norm_kwargs) or nn.Identity(),
            create_act(act, **act_kwargs) or nn.Identity(),
            spconv.SubMConv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key),
        )

    def forward(self, x: "spconv.SparseConvTensor") -> "spconv.SparseConvTensor":
        identity = spconv.SparseConvTensor(x.features, x.indices, x.spatial_shape, x.batch_size)
        out = self.conv_branch(x)
        return out.replace_feature(out.features + self.i_branch(identity).features)
