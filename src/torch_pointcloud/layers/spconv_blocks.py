"""Sparse convolution blocks: submanifold, strided, and residual variants."""

from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple, Union

import torch.nn as nn
from torch import Tensor

from torch_pointcloud.utils.conversion import convert_to_spconv_tensor
from torch_pointcloud.utils.imports import _SPCONV_AVAILABLE, _SPCONV_GITHUB_URL, optional_import

from .act import create_act
from .norms import create_norm

if TYPE_CHECKING:
    import spconv.pytorch as spconv
    from spconv.pytorch import SparseSequential
else:
    SparseSequential, _ = optional_import("spconv.pytorch", "SparseSequential", url=_SPCONV_GITHUB_URL)


spconv, _ = optional_import("spconv.pytorch", url=_SPCONV_GITHUB_URL)

SparseModule: Any = spconv.SparseModule if _SPCONV_AVAILABLE else nn.Module


class SubMConv3dBlock(nn.Module):
    r"""Submanifold sparse 3D convolution followed by normalization and activation, on packed point features.

    Unlike `SparseConvBlock`, this block takes and returns packed $(N, C)$ features: it builds the
    `SparseConvTensor` from `pos` and `batch` itself, so it drops into a point-based backbone as a
    conditional position embedding.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Convolution kernel size.
        padding: Convolution padding.
        norm: Normalization passed to `create_norm`.
        act: Activation passed to `create_act`.
        act_kwargs: Extra keyword arguments for the activation.
        norm_kwargs: Extra keyword arguments for the normalization.
        bias: Whether the convolution has a bias term.
        stem_indice_key: spconv index key for the convolution.
    """

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
        self.norm: Optional[nn.Module] = create_norm(norm, out_channels, **norm_kwargs)
        self.act = create_act(act, **act_kwargs)

    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        batch: Tensor,
        condition: Optional[str] = None,
    ) -> Tensor:
        norm_kwargs = {} if condition is None else {"condition": condition}
        x_spconv = convert_to_spconv_tensor(x, pos, batch)
        x_spconv = self.stem(x_spconv)

        x = x_spconv.features
        if self.norm is not None:
            x = self.norm(x, **norm_kwargs)
        if self.act is not None:
            x = self.act(x)

        return x


class SparseConvBlock(SparseSequential):
    r"""Sparse 3D convolution followed by normalization and activation.

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
            create_norm(norm, out_channels, dim=1, **(norm_kwargs or {})) or nn.Identity(),
            create_act(act, **(act_kwargs or {})) or nn.Identity(),
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


class SubMConv3dResidualBlock(SparseModule):
    r"""Residual block with a single $3\times3\times3$ submanifold convolution: conv, norm, add the input, act.

    Args:
        channels: Input and output channels.
        indice_key: Shared submanifold indice key.
        act: Activation type or callable.
        act_kwargs: Extra activation arguments.
        norm: Normalization type or callable.
        norm_kwargs: Extra normalization arguments.
    """

    def __init__(
        self,
        channels: int,
        *,
        indice_key: str,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.conv1 = spconv.SubMConv3d(channels, channels, 3, padding=1, bias=True, indice_key=indice_key)
        self.bn1 = create_norm(norm, channels, dim=1, **(norm_kwargs or {}))
        self.act = create_act(act, **(act_kwargs or {}))

    def forward(self, x: "spconv.SparseConvTensor") -> "spconv.SparseConvTensor":
        out = self.conv1(x)
        feat = out.features
        if self.bn1 is not None:
            feat = self.bn1(feat)
        feat = feat + x.features
        if self.act is not None:
            feat = self.act(feat)
        return out.replace_feature(feat)
