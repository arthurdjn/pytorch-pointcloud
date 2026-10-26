r"""Cylinder3D sparse blocks: the two-branch residual block, the upsample block and the dimension-decomposition gate.

Every block takes one kernel size per convolution. Convolutions of a block that share a kernel size share their spconv
index table, keyed by the block's `indice_key` and the kernel size.

Adapted from :github: [xinge008/Cylinder3D](https://github.com/xinge008/Cylinder3D).
"""

from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple, Union

import torch.nn as nn
from torch import Tensor

from torch_pointcloud.utils.imports import _SPCONV_AVAILABLE, _SPCONV_GITHUB_URL, optional_import

from .act import create_act
from .norms import create_norm

if TYPE_CHECKING:
    import spconv.pytorch as spconv
    from spconv.pytorch import SparseConvTensor


spconv, _ = optional_import("spconv.pytorch", url=_SPCONV_GITHUB_URL)

SparseModule: Any = spconv.SparseModule if _SPCONV_AVAILABLE else nn.Module

KernelSize = Tuple[int, int, int]


def _submconv3d(in_channels: int, out_channels: int, kernel_size: KernelSize, indice_key: str) -> nn.Module:
    """A bias-free `SubMConv3d` centered on each voxel, whose index table is keyed by `indice_key` and the kernel."""
    return spconv.SubMConv3d(
        in_channels,
        out_channels,
        kernel_size,
        padding=tuple(k // 2 for k in kernel_size),
        bias=False,
        indice_key=f"{indice_key}_{'x'.join(str(k) for k in kernel_size)}",
    )


class AsymmetricResidualBlock(SparseModule):
    r"""Two-branch residual block of asymmetric submanifold convolutions.

    The first branch applies `SubMConv3d` with `kernel_sizes[0]` then `kernel_sizes[1]`, the second branch the same
    kernels in the reverse order, and the block returns the sum of the branches. Each convolution is followed by the
    activation and then the normalization. Both branches project `in_channels` to `out_channels`, so there is no
    identity path.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_sizes: The two kernel sizes $(k_0, k_1, k_2)$, odd along each axis, e.g. `((1, 3, 3), (3, 1, 3))`.
        indice_key: spconv index key prefix of the block.
        act: Activation passed to `create_act`.
        act_kwargs: Extra keyword arguments for the activation.
        norm: Normalization passed to `create_norm`.
        norm_kwargs: Extra keyword arguments for the normalization.

    Example:
        ```python
        import spconv.pytorch as spconv
        import torch
        from torch_pointcloud.layers import AsymmetricResidualBlock

        block = AsymmetricResidualBlock(16, 32, ((1, 3, 3), (3, 1, 3)), indice_key="block").cuda()
        indices = torch.cat([torch.zeros(500, 1), torch.randint(0, 32, (500, 3))], dim=1).int().unique(dim=0).cuda()
        x = spconv.SparseConvTensor(torch.randn(len(indices), 16).cuda(), indices, [32, 32, 32], 1)
        block(x).features.shape  # (V, 32)
        ```
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: Tuple[KernelSize, KernelSize],
        *,
        indice_key: str,
        act: Union[str, Callable, None] = "leaky_relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        act_kwargs = act_kwargs or {}
        norm_kwargs = norm_kwargs or {}
        first, second = kernel_sizes

        self.conv1 = _submconv3d(in_channels, out_channels, first, indice_key)
        self.norm1 = create_norm(norm, out_channels, **norm_kwargs) or nn.Identity()
        self.conv2 = _submconv3d(out_channels, out_channels, second, indice_key)
        self.norm2 = create_norm(norm, out_channels, **norm_kwargs) or nn.Identity()
        self.conv3 = _submconv3d(in_channels, out_channels, second, indice_key)
        self.norm3 = create_norm(norm, out_channels, **norm_kwargs) or nn.Identity()
        self.conv4 = _submconv3d(out_channels, out_channels, first, indice_key)
        self.norm4 = create_norm(norm, out_channels, **norm_kwargs) or nn.Identity()
        self.act = create_act(act, **act_kwargs) or nn.Identity()

    def forward(self, x: "SparseConvTensor") -> "SparseConvTensor":
        branch1 = self.conv1(x)
        branch1 = branch1.replace_feature(self.norm1(self.act(branch1.features)))
        branch1 = self.conv2(branch1)
        branch1 = branch1.replace_feature(self.norm2(self.act(branch1.features)))

        branch2 = self.conv3(x)
        branch2 = branch2.replace_feature(self.norm3(self.act(branch2.features)))
        branch2 = self.conv4(branch2)
        branch2 = branch2.replace_feature(self.norm4(self.act(branch2.features)))
        return branch2.replace_feature(branch1.features + branch2.features)


class AsymmetricUpsampleBlock(SparseModule):
    r"""Upsample block: a cubic projection, an inverse convolution, a skip sum and three asymmetric convolutions.

    The input is projected to `out_channels` by a $3 \times 3 \times 3$ `SubMConv3d`, upsampled by a
    `SparseInverseConv3d` that undoes the strided convolution registered under `upsample_indice_key`, summed with the
    skip features of that resolution (one row per voxel of the strided convolution's input), then refined by three `SubMConv3d` of `kernel_sizes`. Every convolution is followed
    by the activation and then the normalization.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels, the width of the skip features.
        kernel_sizes: Kernel sizes $(k_0, k_1, k_2)$ of the three convolutions after the skip sum, odd along each
            axis, e.g. `((1, 3, 3), (3, 1, 3), (3, 3, 3))`.
        indice_key: spconv index key prefix of the block.
        upsample_indice_key: spconv index key of the strided convolution the inverse convolution undoes.
        act: Activation passed to `create_act`.
        act_kwargs: Extra keyword arguments for the activation.
        norm: Normalization passed to `create_norm`.
        norm_kwargs: Extra keyword arguments for the normalization.

    Example:
        ```python
        import spconv.pytorch as spconv
        import torch
        from torch_pointcloud.layers import AsymmetricUpsampleBlock

        downsample = spconv.SparseConv3d(8, 8, 3, stride=2, padding=1, bias=False, indice_key="down").cuda()
        block = AsymmetricUpsampleBlock(
            8,
            8,
            ((1, 3, 3), (3, 1, 3), (3, 3, 3)),
            indice_key="up",
            upsample_indice_key="down",
        ).cuda()
        indices = torch.cat([torch.zeros(500, 1), torch.randint(0, 32, (500, 3))], dim=1).int().unique(dim=0).cuda()
        skip = spconv.SparseConvTensor(torch.randn(len(indices), 8).cuda(), indices, [32, 32, 32], 1)
        block(downsample(skip), skip.features).features.shape  # (V, 8)
        ```
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: Tuple[KernelSize, KernelSize, KernelSize],
        *,
        indice_key: str,
        upsample_indice_key: str,
        act: Union[str, Callable, None] = "leaky_relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        act_kwargs = act_kwargs or {}
        norm_kwargs = norm_kwargs or {}

        self.conv1 = _submconv3d(in_channels, out_channels, (3, 3, 3), indice_key + "_proj")
        self.norm1 = create_norm(norm, out_channels, **norm_kwargs) or nn.Identity()
        self.upsample = spconv.SparseInverseConv3d(
            out_channels,
            out_channels,
            3,
            bias=False,
            indice_key=upsample_indice_key,
        )
        self.conv2 = _submconv3d(out_channels, out_channels, kernel_sizes[0], indice_key)
        self.norm2 = create_norm(norm, out_channels, **norm_kwargs) or nn.Identity()
        self.conv3 = _submconv3d(out_channels, out_channels, kernel_sizes[1], indice_key)
        self.norm3 = create_norm(norm, out_channels, **norm_kwargs) or nn.Identity()
        self.conv4 = _submconv3d(out_channels, out_channels, kernel_sizes[2], indice_key)
        self.norm4 = create_norm(norm, out_channels, **norm_kwargs) or nn.Identity()
        self.act = create_act(act, **act_kwargs) or nn.Identity()

    def forward(self, x: "SparseConvTensor", skip: Tensor) -> "SparseConvTensor":
        x = self.conv1(x)
        x = x.replace_feature(self.norm1(self.act(x.features)))
        x = self.upsample(x)
        x = x.replace_feature(x.features + skip)

        x = self.conv2(x)
        x = x.replace_feature(self.norm2(self.act(x.features)))
        x = self.conv3(x)
        x = x.replace_feature(self.norm3(self.act(x.features)))
        x = self.conv4(x)
        x = x.replace_feature(self.norm4(self.act(x.features)))
        return x


class DimensionDecompositionBlock(SparseModule):
    r"""Dimension-decomposition context gate: three parallel gated convolutions scale the input features.

    Three `SubMConv3d` of `kernel_sizes` run on the input in parallel, each followed by the normalization and the
    gating activation; their sum multiplies the input features elementwise, so the block keeps the channel count.

    Args:
        channels: Number of input and output channels.
        kernel_sizes: Kernel sizes $(k_0, k_1, k_2)$ of the three convolutions, odd along each axis, e.g.
            `((3, 1, 1), (1, 3, 1), (1, 1, 3))`.
        indice_key: spconv index key prefix of the block.
        act: Gating activation passed to `create_act`.
        act_kwargs: Extra keyword arguments for the activation.
        norm: Normalization passed to `create_norm`.
        norm_kwargs: Extra keyword arguments for the normalization.

    Example:
        ```python
        import spconv.pytorch as spconv
        import torch
        from torch_pointcloud.layers import DimensionDecompositionBlock

        block = DimensionDecompositionBlock(16, ((3, 1, 1), (1, 3, 1), (1, 1, 3)), indice_key="gate").cuda()
        indices = torch.cat([torch.zeros(500, 1), torch.randint(0, 32, (500, 3))], dim=1).int().unique(dim=0).cuda()
        x = spconv.SparseConvTensor(torch.randn(len(indices), 16).cuda(), indices, [32, 32, 32], 1)
        block(x).features.shape  # (V, 16)
        ```
    """

    def __init__(
        self,
        channels: int,
        kernel_sizes: Tuple[KernelSize, KernelSize, KernelSize],
        *,
        indice_key: str,
        act: Union[str, Callable, None] = "sigmoid",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        act_kwargs = act_kwargs or {}
        norm_kwargs = norm_kwargs or {}

        self.conv1 = _submconv3d(channels, channels, kernel_sizes[0], indice_key)
        self.norm1 = create_norm(norm, channels, **norm_kwargs) or nn.Identity()
        self.conv2 = _submconv3d(channels, channels, kernel_sizes[1], indice_key)
        self.norm2 = create_norm(norm, channels, **norm_kwargs) or nn.Identity()
        self.conv3 = _submconv3d(channels, channels, kernel_sizes[2], indice_key)
        self.norm3 = create_norm(norm, channels, **norm_kwargs) or nn.Identity()
        self.act = create_act(act, **act_kwargs) or nn.Identity()

    def forward(self, x: "SparseConvTensor") -> "SparseConvTensor":
        gate1 = self.act(self.norm1(self.conv1(x).features))
        gate2 = self.act(self.norm2(self.conv2(x).features))
        gate3 = self.act(self.norm3(self.conv3(x).features))
        return x.replace_feature((gate1 + gate2 + gate3) * x.features)
