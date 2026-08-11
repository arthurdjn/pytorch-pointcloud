from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Sequence, Union

import torch.nn as nn
from torch import Tensor

from torch_pointcloud.utils.conversion import ensure_list
from torch_pointcloud.utils.imports import _OCNN_GITHUB_URL, optional_import

from .act import create_act
from .norms import create_norm

if TYPE_CHECKING:
    import ocnn
    from ocnn.octree import Octree

ocnn, _ = optional_import("ocnn", url=_OCNN_GITHUB_URL)
Octree, _ = optional_import("ocnn.octree", "Octree", url=_OCNN_GITHUB_URL)

MAX_BUFFER = int(2e8)


def _disable_triton() -> None:
    # OCNN 2.3.x defaults octree convs to a GPU-only Triton implicit-GEMM kernel whose split-K reductions
    # drift from the deterministic GEMM the pretrained weights were exported with (and cannot run on CPU).
    # Force the deterministic path so convs reproduce the reference outputs and run on CPU. The
    # implementation is chosen when the conv module is constructed, so this must run before building any
    # `ocnn` conv (including convs that take no `method`, e.g. OctFormer's CPE `OctreeGroupConv`).
    ocnn.nn.octree_conv.DISABLE_TRITON = True


class OctreeConvBlock(nn.Module):
    r"""Octree convolution followed by normalization and activation.

    Wraps `ocnn.nn.OctreeConv` with a norm / act pair built by `create_norm` / `create_act`.
    With `act_first=True` the activation runs before the normalization instead of after.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Convolution kernel size (an `int` is broadcast to all axes).
        stride: Convolution stride; `2` downsamples the octree by one depth level.
        nempty: Whether the features only cover non-empty octree nodes.
        act: Activation passed to `create_act`.
        act_kwargs: Extra keyword arguments for the activation.
        act_first: Whether to apply the activation before the normalization.
        norm: Normalization passed to `create_norm`.
        norm_kwargs: Extra keyword arguments for the normalization.
        bias: Whether the convolution uses a bias.
        method: `ocnn` convolution implementation (e.g. `"explicit_gemm"`).
        max_buffer: Maximum buffer size (in elements) used by the `ocnn` convolution.

    Shape:
        - Input: $(M_\text{in}, C_\text{in})$ octree features at `depth`.
        - Output: $(M_\text{out}, C_\text{out})$ octree features (at `depth - 1` when `stride=2`).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Sequence[int]],
        stride: int = 1,
        nempty: bool = False,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: bool = True,
        method: str = "explicit_gemm",
        max_buffer: int = MAX_BUFFER,
    ):
        super().__init__()
        # NOTE: OCNN expects the kernel size to be a list, otherwise assertion error will be raised.
        kernel_size = ensure_list(kernel_size)
        act_kwargs = act_kwargs or {}
        norm_kwargs = norm_kwargs or {}

        _disable_triton()
        self.conv = ocnn.nn.OctreeConv(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            nempty=nempty,
            use_bias=bias,
            method=method,
            max_buffer=max_buffer,
        )
        self.norm = create_norm(norm, out_channels, **norm_kwargs) or nn.Identity()
        self.act = create_act(act, **act_kwargs) or nn.Identity()
        self.act_first = act_first

    def forward(self, x: Tensor, octree: Octree, depth: int) -> Tensor:
        x = self.conv(x, octree, depth)
        if self.act is not None and self.act_first:
            x = self.act(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None and not self.act_first:
            x = self.act(x)
        return x


class OctreeDeconvBlock(nn.Module):
    r"""Octree transposed convolution followed by normalization and activation.

    Wraps `ocnn.nn.OctreeDeconv` with a norm / act pair built by `create_norm` / `create_act`.
    With `act_first=True` the activation runs before the normalization instead of after.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Convolution kernel size (an `int` is broadcast to all axes).
        stride: Convolution stride; `2` upsamples the octree by one depth level.
        nempty: Whether the features only cover non-empty octree nodes.
        act: Activation passed to `create_act`.
        act_kwargs: Extra keyword arguments for the activation.
        act_first: Whether to apply the activation before the normalization.
        norm: Normalization passed to `create_norm`.
        norm_kwargs: Extra keyword arguments for the normalization.
        bias: Whether the convolution uses a bias.
        method: `ocnn` convolution implementation (e.g. `"explicit_gemm"`).
        max_buffer: Maximum buffer size (in elements) used by the `ocnn` convolution.

    Shape:
        - Input: $(M_\text{in}, C_\text{in})$ octree features at `depth`.
        - Output: $(M_\text{out}, C_\text{out})$ octree features (at `depth + 1` when `stride=2`).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Sequence[int]],
        stride: int = 1,
        nempty: bool = False,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: bool = True,
        method: str = "explicit_gemm",
        max_buffer: int = MAX_BUFFER,
    ):
        super().__init__()
        # NOTE: OCNN expects the kernel size to be a list, otherwise assertion error will be raised.
        kernel_size = ensure_list(kernel_size)
        act_kwargs = act_kwargs or {}
        norm_kwargs = norm_kwargs or {}

        _disable_triton()
        self.deconv = ocnn.nn.OctreeDeconv(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            nempty=nempty,
            use_bias=bias,
            method=method,
            max_buffer=max_buffer,
        )
        self.norm = create_norm(norm, out_channels, **norm_kwargs) or nn.Identity()
        self.act = create_act(act, **act_kwargs) or nn.Identity()
        self.act_first = act_first

    def forward(self, x: Tensor, octree: Octree, depth: int) -> Tensor:
        x = self.deconv(x, octree, depth)
        if self.act is not None and self.act_first:
            x = self.act(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None and not self.act_first:
            x = self.act(x)
        return x
