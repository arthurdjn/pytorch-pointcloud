"""torchsparse convolution blocks: a conv + norm + act block and a residual block over `SparseTensor`s."""

from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Union

import torch.nn as nn

from torch_pointcloud.layers.act import create_act
from torch_pointcloud.layers.dropouts import DropPath
from torch_pointcloud.layers.norms import create_norm
from torch_pointcloud.utils.imports import _TORCHSPARSE_GITHUB_URL, optional_import

if TYPE_CHECKING:
    import torchsparse.nn as spnn
    from torchsparse.tensor import SparseTensor

spnn, _ = optional_import("torchsparse.nn", url=_TORCHSPARSE_GITHUB_URL)


class TorchSparseConvBlock(nn.Module):
    """Sparse 3D convolution followed by normalization and activation.

    Set `transposed=True` for an upsampling (inverse) convolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        transposed: bool = False,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        act_kwargs = act_kwargs or {}
        norm_kwargs = norm_kwargs or {}

        self.conv = spnn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            stride=stride,
            transposed=transposed,
        )
        self.norm = create_norm(norm, out_channels, **norm_kwargs) or nn.Identity()
        self.act = create_act(act, **act_kwargs) or nn.Identity()

    def forward(self, x: "SparseTensor") -> "SparseTensor":
        x = self.conv(x)
        x.F = self.act(self.norm(x.F))
        return x


class TorchSparseResidualBlock(nn.Module):
    """Residual block of two sparse 3D convolutions.

    A pointwise convolution projects the skip connection when the channel count or stride changes.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        drop_path: float = 0.0,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        act_kwargs = act_kwargs or {}
        norm_kwargs = norm_kwargs or {}

        self.conv1 = spnn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation, stride=stride)
        self.norm1 = create_norm(norm, out_channels, **norm_kwargs) or nn.Identity()
        self.conv2 = spnn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, dilation=dilation, stride=1)
        self.norm2 = create_norm(norm, out_channels, **norm_kwargs) or nn.Identity()

        self.conv_skip: Optional[nn.Module] = None
        self.norm_skip: Optional[nn.Module] = None
        if in_channels != out_channels or stride != 1:
            self.conv_skip = spnn.Conv3d(in_channels, out_channels, kernel_size=1, dilation=1, stride=stride)
            self.norm_skip = create_norm(norm, out_channels, **norm_kwargs) or nn.Identity()

        self.act = create_act(act, **act_kwargs) or nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else None

    def forward(self, x: "SparseTensor") -> "SparseTensor":
        x_skip = x
        x = self.conv1(x)
        x.F = self.act(self.norm1(x.F))
        x = self.conv2(x)
        x.F = self.norm2(x.F)

        if self.conv_skip is not None:
            x_skip = self.conv_skip(x_skip)
        if self.norm_skip is not None:
            x_skip.F = self.norm_skip(x_skip.F)
        if self.drop_path is not None:
            x_skip.F = self.drop_path(x_skip.F)

        x.F = self.act(x.F + x_skip.F)
        return x
