"""Sparse convolution blocks: submanifold, strided, transposed, and residual variants."""

from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from torch_pointcloud.ops.voxelization import lookup_coordinates
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
SparseConv3d: Any = spconv.SparseConv3d if _SPCONV_AVAILABLE else nn.Module


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


class StridedSparseConv3d(SparseConv3d):
    r"""Strided sparse 3D convolution with one output voxel per occupied parent cell.

    The output voxels are the parents $\lfloor c / s \rfloor$ of the input voxels $c$, in sorted order, and nothing
    else: `spconv.SparseConv3d` also emits every output whose window merely reaches an input voxel. The window of
    output $o$ is centred on input $s o$ for an odd kernel and starts there for an even one; an output whose window
    holds no input voxel (a $1$-voxel kernel over a parent with only odd children) is zero.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Convolution kernel size.
        stride: Convolution stride $s$.
        bias: Whether the convolution has a bias term.

    Example:
        ```python
        import spconv.pytorch as spconv
        import torch
        from torch_pointcloud.layers import StridedSparseConv3d

        indices = torch.tensor([[0, 0, 0, 0], [0, 1, 0, 0], [0, 3, 3, 3]], dtype=torch.int32, device="cuda")
        x = spconv.SparseConvTensor(torch.randn(3, 4, device="cuda"), indices, [4, 4, 4], 1)
        out = StridedSparseConv3d(4, 8, kernel_size=3, stride=2).cuda()(x)
        out.indices  # [[0, 0, 0, 0], [0, 1, 1, 1]]
        ```
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        bias: bool = False,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=kernel_size // 2 if kernel_size % 2 else 0,
            bias=bias,
        )

    def forward(
        self,
        input: "spconv.SparseConvTensor",
        add_input: Optional["spconv.SparseConvTensor"] = None,
    ) -> "spconv.SparseConvTensor":
        indices = input.indices
        parents = torch.cat([indices[:, :1], torch.div(indices[:, 1:], self.stride[0], rounding_mode="floor")], dim=1)
        parents = torch.unique(parents, dim=0)

        out = super().forward(input, add_input)
        index = lookup_coordinates(out.indices, parents)
        features = out.features[index.clamp(min=0)] * (index >= 0).unsqueeze(1).to(out.features.dtype)
        return spconv.SparseConvTensor(features, parents.int().contiguous(), out.spatial_shape, out.batch_size)


class SparseConvBlock(SparseSequential):
    r"""Sparse 3D convolution followed by normalization and activation.

    A `subm` (submanifold), `spconv` (regular, optionally strided), `inverseconv` or `transposeconv` (transposed,
    generating the output voxels of every input voxel's window) convolution over a `SparseConvTensor`, followed by
    normalization and activation on the features. As a `SparseSequential` subclass it drops directly into the
    sparse voxel backbones' convolution stacks.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Convolution kernel size.
        stride: Convolution stride (used by `spconv` and `transposeconv`).
        padding: Convolution padding (used by `spconv` and `transposeconv`).
        indice_key: spconv index key for the convolution.
        conv_type: One of `subm`, `spconv`, `inverseconv`, or `transposeconv`.
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
        indice_key: Optional[str],
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
        elif conv_type == "transposeconv":
            conv = spconv.SparseConvTranspose3d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
                indice_key=indice_key,
            )
        else:
            raise ValueError(
                f"Unknown conv_type {conv_type!r}. Expected 'subm', 'spconv', 'inverseconv', or 'transposeconv'."
            )

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


class SparseBasicBlock(SparseModule):
    r"""Residual block of two sparse convolutions: conv, norm, act, conv, norm, add the input, act.

    Both convolutions are submanifold, except a strided first one (`stride` above $1$), a `StridedSparseConv3d` onto
    the parent voxels. A pointwise convolution and a normalization project the skip connection when the channel
    count or the resolution changes (strided like the first convolution).

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Kernel size of the two convolutions.
        bias: Whether the two convolutions carry a bias.
        indice_key: spconv index key shared by the submanifold convolutions.
        stride: Stride of the first convolution and the skip projection.
        act: Activation passed to `create_act`.
        act_kwargs: Extra keyword arguments for the activation.
        norm: Normalization passed to `create_norm`.
        norm_kwargs: Extra keyword arguments for the normalization.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        bias: bool = False,
        indice_key: Optional[str] = None,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        stride: int = 1,
    ) -> None:
        super().__init__()
        act_kwargs = act_kwargs or {}
        norm_kwargs = norm_kwargs or {}
        padding = kernel_size // 2

        if in_channels == out_channels and stride == 1:
            self.proj: nn.Module = spconv.SparseSequential(nn.Identity())
        elif stride == 1:
            self.proj = spconv.SparseSequential(
                spconv.SubMConv3d(in_channels, out_channels, kernel_size=1, bias=False),
                create_norm(norm, out_channels, **norm_kwargs) or nn.Identity(),
            )
        else:
            self.proj = spconv.SparseSequential(
                StridedSparseConv3d(in_channels, out_channels, kernel_size=1, stride=stride),
                create_norm(norm, out_channels, **norm_kwargs) or nn.Identity(),
            )

        if stride == 1:
            self.conv1: nn.Module = spconv.SubMConv3d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=bias,
                indice_key=indice_key,
            )
        else:
            self.conv1 = StridedSparseConv3d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                bias=bias,
            )
        self.norm1 = create_norm(norm, out_channels, **norm_kwargs) or nn.Identity()
        self.conv2 = spconv.SubMConv3d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
            indice_key=indice_key,
        )
        self.norm2 = create_norm(norm, out_channels, **norm_kwargs) or nn.Identity()
        self.act = create_act(act, **act_kwargs) or nn.Identity()

    def forward(self, x: "spconv.SparseConvTensor") -> "spconv.SparseConvTensor":
        residual = x
        out = self.conv1(x)
        out = out.replace_feature(self.act(self.norm1(out.features)))
        out = self.conv2(out)
        out = out.replace_feature(self.norm2(out.features))
        out = out.replace_feature(self.act(out.features + self.proj(residual).features))
        return out


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
        self.norm1 = create_norm(norm, channels, dim=1, **(norm_kwargs or {}))
        self.act = create_act(act, **(act_kwargs or {}))

    def forward(self, x: "spconv.SparseConvTensor") -> "spconv.SparseConvTensor":
        out = self.conv1(x)
        feat = out.features
        if self.norm1 is not None:
            feat = self.norm1(feat)
        feat = feat + x.features
        if self.act is not None:
            feat = self.act(feat)
        return out.replace_feature(feat)
