import math
import random
import warnings
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Sequence, Tuple, Union, overload

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torch_pointcloud.layers import (
    MLP,
    ActLike,
    NormLike,
    PoolLike,
    create_act,
    create_cls_head,
    create_norm,
    create_pool,
)
from torch_pointcloud.layers.blocks import linear_block
from torch_pointcloud.utils.config import CACHE_DIR
from torch_pointcloud.utils.conversion import ensure_tuple, ensure_tuple_size
from torch_pointcloud.utils.geometry import rodrigues_rotation_matrix, spherical_points_gradient, spherical_points_lloyd
from torch_pointcloud.utils.imports import optional_import
from torch_pointcloud.utils.ops import consecutive_cluster, voxel_grid
from torch_pointcloud.utils.types import OptTensor

from ._base import ClassificationModel, SegmentationModel
from ._registry import register_model
from .pointnet2 import create_fp_blocks

if TYPE_CHECKING:
    from torch_cluster import radius, radius_graph
    from torch_scatter import scatter

radius, _ = optional_import("torch_cluster", name="radius")
radius_graph, _ = optional_import("torch_cluster", name="radius_graph")
scatter, _ = optional_import("torch_scatter", name="scatter")


def create_kernel_points(
    radius: float,
    num_points: int,
    fixed_position: Literal["none", "center", "vertical"] = "center",
    method: Literal["lloyd", "gradient"] = "lloyd",
) -> torch.Tensor:
    if method not in ["lloyd", "gradient"]:
        raise ValueError(f"Unknown method: {method!r}, expected 'lloyd' or 'gradient'.")
    if num_points > 30 and method != "lloyd":
        warnings.warn("Too many points, consider using Lloyds algorithm with `method='lloyd'`.")

    # Check if kernel is already computed
    kernel_path = Path(CACHE_DIR, "kernels", f"k_{num_points}_{fixed_position}_{method}.pt")
    if kernel_path.exists():
        kernel_points = torch.load(kernel_path, weights_only=True)
    else:
        if method == "lloyd":
            kernel_points = spherical_points_lloyd(radius=1.0, num_points=num_points, fixed_position=fixed_position)
        else:
            kernel_points, _ = spherical_points_gradient(
                radius=1.0,
                num_points=num_points,
                fixed_position=fixed_position,
            )

        kernel_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(kernel_points, kernel_path)

    # Random rotations for the kernel
    R = torch.eye(3)
    theta = torch.rand(1) * 2 * math.pi

    if fixed_position != "vertical":
        c, s = torch.cos(theta), torch.sin(theta)
        R = torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=torch.float32)
    else:
        phi = (torch.rand(1) - 0.5) * math.pi
        # Create the first vector in cartesian coordinates
        u = torch.tensor([torch.cos(theta) * torch.cos(phi), torch.sin(theta) * torch.cos(phi), torch.sin(phi)])
        # Choose a random rotation angle
        alpha = random.random() * 2 * math.pi
        R = rodrigues_rotation_matrix(u, theta=alpha)

    # Add a small noise, scale and rotate
    kernel_points += torch.normal(mean=0, std=0.01, size=kernel_points.shape)
    kernel_points *= radius
    kernel_points = torch.matmul(kernel_points, R)

    return kernel_points


class KPConv(nn.Module):
    def __init__(
        self,
        spatial_dim: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        kp_radius: float,
        kp_sigma: float,
        kp_influence: str = "linear",
        fixed_position: Literal["none", "center", "vertical"] = "center",
        aggregation_mode: str = "sum",
        deformable: bool = False,
        modulated: bool = False,
        bias: bool = False,
        track_running_stats: bool = True,
    ) -> None:
        super().__init__()
        if aggregation_mode not in ["sum", "closest"]:
            raise ValueError(f"Unknown aggregation mode: {aggregation_mode!r}, expected 'sum' or 'closest'.")

        self.spatial_dim = spatial_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.kp_radius = kp_radius
        self.kp_sigma = kp_sigma
        self.fixed_position = fixed_position
        self.kp_influence = kp_influence
        self.aggregation_mode = aggregation_mode
        self.modulated = modulated

        self.weight = nn.Parameter(torch.zeros(kernel_size, in_channels, out_channels), requires_grad=True)
        self.register_buffer("kernel", self.configure_kernel())
        self.register_parameter("bias", nn.Parameter(torch.zeros(size=(out_channels,))) if bias else None)
        self.reset_parameters()

        self.offset_conv: Optional[nn.Module]
        self.offset_conv, offset_bias = self.configure_offsets() if deformable else (None, None)
        self.register_parameter("offset_bias", offset_bias)

        # Track running statistics (mostly for regularization)
        self.track_running_stats = track_running_stats
        self.register_buffer("running_min_d2", None)
        self.register_buffer("running_deformed_kernel", None)
        self.register_buffer("running_offset_features", None)

    @property
    def deformable(self) -> bool:
        return self.offset_conv is not None

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(k), 1/sqrt(k)), where k = weight.size(1) * prod(*kernel_size)
        # For more details see: https://github.com/pytorch/pytorch/issues/15314#issuecomment-477448573
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)  # type: ignore[arg-type]

    def configure_offsets(self) -> Tuple[nn.Module, nn.Parameter]:
        offset_dim = self.spatial_dim * self.kernel_size
        if self.modulated:
            offset_dim += self.kernel_size

        offset_conv = KPConv(
            spatial_dim=self.spatial_dim,
            in_channels=self.in_channels,
            out_channels=offset_dim,
            kernel_size=self.kernel_size,
            kp_radius=self.kp_radius,
            kp_sigma=self.kp_sigma,
            kp_influence=self.kp_influence,
            fixed_position=self.fixed_position,
            aggregation_mode=self.aggregation_mode,
            deformable=False,
            modulated=False,
            bias=False,
        )
        offset_bias = nn.Parameter(torch.zeros(offset_dim))
        return offset_conv, offset_bias

    def configure_kernel(self) -> Tensor:
        return create_kernel_points(self.kp_radius, self.kernel_size, fixed_position=self.fixed_position)

    def _compute_weights(self, sq_distances: Tensor) -> Tensor:
        if self.kp_influence == "constant":
            return torch.ones_like(sq_distances)
        elif self.kp_influence == "linear":
            return torch.clamp(1 - torch.sqrt(sq_distances) / self.kp_sigma, min=0.0)
        elif self.kp_influence == "gaussian":
            # TODO: kp_sigma should already be provided
            # sigma = self.kp_extent * 0.3
            return torch.exp(-sq_distances / (2 * self.kp_sigma**2 + 1e-6))
        else:
            raise ValueError(f"Unknown influence type: {self.kp_influence}")

    def _compute_offsets(
        self,
        features: Tensor,
        coords_query: Tensor,
        coords_support: Tensor,
        edge_index: Tensor,
    ) -> Tuple[OptTensor, OptTensor]:
        if self.offset_conv is None:
            return None, None

        # Use the offset convolution to get offset features
        offset_features = self.offset_conv(features, coords_query, coords_support, edge_index)
        if self.offset_bias is not None:
            offset_features = offset_features + self.offset_bias

        if self.track_running_stats:
            self.running_offset_features = offset_features

        if self.modulated:
            # Split into offsets and modulations
            unscaled_offsets = offset_features[:, : self.spatial_dim * self.kernel_size]
            unscaled_offsets = unscaled_offsets.view(-1, self.kernel_size, self.spatial_dim)

            # Get modulations (sigmoid to keep between 0 and 2)
            modulations = 2 * torch.sigmoid(offset_features[:, self.spatial_dim * self.kernel_size :])
            modulations = modulations.view(-1, self.kernel_size)
        else:
            # Just offsets, no modulations
            unscaled_offsets = offset_features.view(-1, self.kernel_size, self.spatial_dim)
            modulations = None

        # Scale offsets by kp_sigma (equivalent to KP_extent in original)
        offsets = unscaled_offsets * self.kp_sigma

        return offsets, modulations

    def forward(
        self,
        features: Tensor,
        coords_query: Tensor,
        coords_support: Tensor,
        edge_index: Tensor,
    ) -> Tensor:
        row, col = edge_index
        coords_rel = coords_support[col] - coords_query[row]

        # For deformable KPConv, compute offsets and modulations
        offsets, modulations = self._compute_offsets(features, coords_query, coords_support, edge_index)

        # Get kernel points at each point
        if self.deformable and offsets is not None:
            kernel_points = self.kernel.unsqueeze(0) + offsets[row]  # type: ignore[operator]
            if self.track_running_stats:
                self.running_deformed_kernel = kernel_points
        else:
            kernel_points = self.kernel.unsqueeze(0).expand(coords_rel.size(0), -1, -1)  # type: ignore[operator]

        coords_rel = coords_rel.unsqueeze(1)  # [E, 1, p_dim]
        sq_distances = torch.sum((coords_rel - kernel_points) ** 2, dim=-1)  # [E, K]

        if self.track_running_stats and self.deformable:
            self.running_min_d2, _ = torch.min(sq_distances, dim=1)
            # Optional: Optimization by ignoring points outside a deformed KP range

        weights = self._compute_weights(sq_distances)

        if self.aggregation_mode == "closest":
            neighbors_1nn = torch.argmin(sq_distances, dim=1)  # [E]
            one_hot = torch.zeros_like(weights).scatter_(1, neighbors_1nn.unsqueeze(1), 1)
            weights = weights * one_hot  # [E, K]

        # Apply modulations if deformable and modulated
        if self.deformable and self.modulated and modulations is not None:
            weights = weights * modulations[row]

        source_features = features[col]  # [E, in_channels]
        output = torch.zeros(coords_query.size(0), self.out_channels, device=features.device, dtype=features.dtype)
        for k in range(self.kernel_size):
            weights_k = weights[:, k].unsqueeze(1)  # [E, 1]
            weighted_features = weights_k * source_features  # [E, in_channels]
            transformed_features = torch.matmul(weighted_features, self.weight[k].to(features.dtype))
            scatter(transformed_features, row, dim=0, out=output, reduce="sum")

        return output

    def extra_repr(self) -> str:
        return (
            f"spatial_dim={self.spatial_dim}, "
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"kp_radius={self.kp_radius}, "
            f"kp_sigma={self.kp_sigma}, "
            f"kp_influence={self.kp_influence!r}, "
            f"fixed_position={self.fixed_position!r}, "
            f"aggregation_mode={self.aggregation_mode!r}, "
            f"deformable={self.deformable}, "
            f"modulated={self.modulated}"
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.extra_repr()})"


class KPConvBlock(nn.Module):
    def __init__(
        self,
        spatial_dim: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        kp_radius: float,
        kp_sigma: float,
        kp_influence: str = "linear",
        fixed_position: Literal["none", "center", "vertical"] = "center",
        aggregation_mode: str = "sum",
        deformable: bool = False,
        modulated: bool = False,
        norm: NormLike = "layer_norm",
        act: ActLike = "leaky_relu",
        bias: bool = False,
    ):
        super().__init__()
        self.conv = KPConv(
            spatial_dim=spatial_dim,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            kp_sigma=kp_sigma,
            kp_radius=kp_radius,
            fixed_position=fixed_position,
            kp_influence=kp_influence,
            aggregation_mode=aggregation_mode,
            deformable=deformable,
            modulated=modulated,
            bias=bias,
        )
        self.norm = create_norm(norm, out_channels) if norm is not None else None
        self.act = create_act(act) if act is not None else None

    def forward(self, features: Tensor, coords_query: Tensor, coords_support: Tensor, edge_index: Tensor) -> Tensor:
        features = self.conv(features, coords_query, coords_support, edge_index)
        if self.norm is not None:
            features = self.norm(features)
        if self.act is not None:
            features = self.act(features)
        return features


class KPResidualBlock(nn.Module):
    def __init__(
        self,
        spatial_dim: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        kp_radius: float,
        kp_sigma: float,
        kp_influence: str = "linear",
        fixed_position: Literal["none", "center", "vertical"] = "center",
        aggregation_mode: str = "sum",
        deformable: bool = False,
        modulated: bool = False,
        strided: bool = False,
        norm: NormLike = "layer_norm",
        act: ActLike = "leaky_relu",
        bias: bool = False,
    ):
        super().__init__()
        # Avoid bottleneck if mid_channels is too small
        mid_channels = max(out_channels // 4, 8)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.strided = strided

        self.unary1 = MLP(in_channels=in_channels, out_channels=mid_channels, norm=norm, act=act, dropout=None)
        self.conv = KPConvBlock(
            spatial_dim=spatial_dim,
            in_channels=mid_channels,
            out_channels=mid_channels,
            kernel_size=kernel_size,
            kp_radius=kp_radius,
            kp_sigma=kp_sigma,
            kp_influence=kp_influence,
            fixed_position=fixed_position,
            aggregation_mode=aggregation_mode,
            deformable=deformable,
            modulated=modulated,
            norm=norm,
            act=act,
            bias=bias,
        )
        self.unary2 = MLP(in_channels=mid_channels, out_channels=out_channels, norm=norm, act=None, dropout=None)
        self.shortcut = (
            MLP(in_channels=in_channels, out_channels=out_channels, norm=norm, act=None, dropout=None)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.act = create_act(act) if act is not None else None

    def forward(self, features: Tensor, coords_query: Tensor, coords_support: Tensor, edge_index: Tensor) -> Tensor:
        shortcut = features
        if self.strided:
            row, col = edge_index
            shortcut = scatter(features[col], row, dim=0, reduce="max")

        shortcut = self.shortcut(shortcut)
        features = self.unary1(features)
        features = self.conv(features, coords_query, coords_support, edge_index)
        features = self.unary2(features)

        features = features + shortcut
        if self.act is not None:
            features = self.act(features)

        return features


class GridPool(nn.Module):
    def __init__(self, grid_size: float, reduce: str = "max"):
        super().__init__()
        self.grid_size = grid_size
        self.reduce = reduce

    @overload
    def forward(
        self,
        features: Tensor,
        coords: Tensor,
        batch: Tensor,
        return_inverse: Literal[True] = True,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]: ...

    @overload
    def forward(
        self,
        features: Tensor,
        coords: Tensor,
        batch: Tensor,
        return_inverse: Literal[False] = False,
    ) -> Tuple[Tensor, Tensor, Tensor]: ...

    def forward(
        self,
        features: Tensor,
        coords: Tensor,
        batch: Tensor,
        return_inverse: bool = False,
    ) -> Tuple[Tensor, ...]:
        cluster = voxel_grid(coords, size=self.grid_size, batch=batch)
        cluster, perm = consecutive_cluster(cluster, return_permutation=True)
        coords = scatter(coords, cluster, dim=0, reduce="mean")
        features = scatter(features, cluster, dim=0, reduce=self.reduce)
        batch = batch[perm]

        if return_inverse:
            return features, coords, batch, cluster
        return features, coords, batch

    def extra_repr(self) -> str:
        return f"grid_size={self.grid_size}, reduce={self.reduce!r}"


class EncoderBlock(nn.Module):
    def __init__(
        self,
        *,
        depth: int,
        radius: float,
        max_num_neighbors: int,
        spatial_dim: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        kp_radius: Union[float, Sequence[float]],
        kp_sigma: Union[float, Sequence[float]],
        kp_influence: str = "linear",
        fixed_position: Literal["none", "center", "vertical"] = "center",
        aggregation_mode: str = "sum",
        deformable: bool = False,
        modulated: bool = False,
        bias: bool = False,
        norm: NormLike = "batch_norm1d",
        act: ActLike = "leaky_relu",
        downsample: Optional[GridPool] = None,
    ):
        super().__init__()
        self.max_num_neighbors = max_num_neighbors
        self.radius = radius
        self.downsample = downsample
        extra_msg = "Expected encoder `{param_name}` to be of length `depth`."
        kp_radius = ensure_tuple_size(kp_radius, size=depth, extra_msg=extra_msg.format(param_name="kp_radius"))
        kp_sigma = ensure_tuple_size(kp_sigma, size=depth, extra_msg=extra_msg.format(param_name="kp_sigma"))

        self.blocks = nn.ModuleList()
        for i in range(depth):
            strided = downsample is not None and i == 0
            block = KPResidualBlock(
                spatial_dim=spatial_dim,
                in_channels=in_channels if i == 0 or (downsample is not None and i == 1) else out_channels,
                out_channels=in_channels if strided else out_channels,
                kernel_size=kernel_size,
                kp_radius=kp_radius[i],
                kp_sigma=kp_sigma[i],
                kp_influence=kp_influence,
                fixed_position=fixed_position,
                aggregation_mode=aggregation_mode,
                deformable=deformable,
                modulated=modulated,
                strided=strided,
                norm=norm,
                act=act,
                bias=bias,
            )
            self.blocks.append(block)

    @overload
    def forward(
        self,
        features: Tensor,
        coords: Tensor,
        batch: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]: ...

    @overload
    def forward(
        self,
        features: Tensor,
        coords: Tensor,
        batch: Tensor,
        return_inverse: Literal[True] = True,
    ) -> Tuple[Tensor, Tensor, Tensor, OptTensor]: ...

    def forward(
        self,
        features: Tensor,
        coords: Tensor,
        batch: Tensor,
        return_inverse: bool = False,
    ) -> Any:
        inv = None
        coords_down, batch_down = coords, batch
        if self.downsample is not None:
            _, coords_down, batch_down, inv = self.downsample(features, coords, batch, return_inverse=True)

        # Pre-computed neighbors edge indices, for both strided and non-strided blocks.
        # For strided block (first block after downsampling),
        # compute edge indices between downsampled coords and original coords.
        # For non-strided blocks, then downsampled coords is the same as original coords, and corresponds to the
        # `radius_graph(coords)` output.
        edge_index = radius(
            coords,
            coords_down,
            r=self.radius,
            batch_x=batch,
            batch_y=batch_down,
            max_num_neighbors=self.max_num_neighbors,
        )

        for i, block in enumerate(self.blocks):
            if i == 1 and self.downsample is not None:
                # Edge indices between downsampled coords and original coords are computed
                # only once for the first block after downsampling.
                # After that, we compute neighboring edges between the downsampled coords just like
                # in the non-strided blocks.
                edge_index = radius_graph(
                    coords_down,
                    r=self.radius,
                    batch=batch_down,
                    max_num_neighbors=self.max_num_neighbors,
                    flow="target_to_source",
                )

            features = block(features, coords_down, coords, edge_index)

        if return_inverse:
            return features, coords_down, batch_down, inv
        return features, coords_down, batch_down


def create_encoder_blocks(
    in_channels: int,
    *,
    depths: Sequence[int],
    grid_sizes: Sequence[float],
    radii: Sequence[float],
    channels: Sequence[int],
    max_num_neighbors: Sequence[int],
    kernel_size: int,
    kp_sigma: Union[float, Sequence[float]],
    kp_radius: Union[float, Sequence[float]],
    kp_influence: str = "linear",
    fixed_position: Literal["none", "center", "vertical"] = "center",
    aggregation_mode: str = "sum",
    deformable: bool = False,
    modulated: bool = False,
    norm: NormLike = "batch_norm1d",
    act: ActLike = "leaky_relu",
    bias: bool = False,
    spatial_dim: int = 3,
) -> nn.ModuleList:
    depths = ensure_tuple(depths)
    n = len(depths)
    extra_msg = "Expected `{param_name}` to be of length `depths`."
    channels = ensure_tuple_size(channels, size=n, extra_msg=extra_msg.format(param_name="channels"))
    max_num_neighbors = ensure_tuple_size(
        max_num_neighbors,
        size=n,
        extra_msg=extra_msg.format(param_name="max_num_neighbors"),
    )
    grid_sizes = ensure_tuple_size(grid_sizes, size=n - 1, extra_msg="Encoder length `grid_sizes` != `depths` - 1.")
    kp_radius = ensure_tuple_size(kp_radius, size=n, extra_msg=extra_msg.format(param_name="kp_radius"))
    kp_sigma = ensure_tuple_size(kp_sigma, size=n, extra_msg=extra_msg.format(param_name="kp_sigma"))

    blocks = nn.ModuleList()
    for i in range(n):
        downsample: Optional[GridPool] = None
        if i > 0:
            downsample = GridPool(grid_size=grid_sizes[i - 1], reduce="max")

        block = EncoderBlock(
            downsample=downsample,
            radius=radii[i],
            max_num_neighbors=max_num_neighbors[i],
            spatial_dim=spatial_dim,
            depth=depths[i],
            in_channels=in_channels,
            out_channels=channels[i],
            kernel_size=kernel_size,
            kp_radius=([kp_radius[i - 1]] + [kp_radius[i]] * (depths[i] - 1)) if i > 0 else kp_radius[i],
            kp_sigma=([kp_sigma[i - 1]] + [kp_sigma[i]] * (depths[i] - 1)) if i > 0 else kp_sigma[i],
            kp_influence=kp_influence,
            fixed_position=fixed_position,
            aggregation_mode=aggregation_mode,
            deformable=deformable,
            modulated=modulated,
            norm=norm,
            act=act,
            bias=bias,
        )
        blocks.append(block)
        in_channels = channels[i]
    return blocks


# TODO: Make KPConv stem
class KPConvNetClassification(ClassificationModel):
    """KPConv Network for classification tasks as described in the paper
    [KPConv: Flexible and Efficient Convolution for Point Clouds](https://arxiv.org/abs/1904.08889)
    by Hugues Thomas, Charles R. Qi, Jean-Emmanuel Deschaud, Beatriz Marcotegui, François Goulette, Leonidas J. Guibas.

    KPConv introduces a novel point convolution operator that uses kernel points to define the spatial extent and weights
    of the convolution. The kernel points are arranged in space to define the convolution pattern, with weights determined
    by their spatial correlation with input points. This allows for flexible and efficient convolution on irregular point
    clouds while maintaining permutation invariance and translation invariance. The network uses a hierarchical architecture
    with strided convolutions for spatial pooling and feature aggregation.

    Note:
        The implementation is based on the original paper and the authors' code
        [KPConv-PyTorch](https://github.com/HuguesTHOMAS/KPConv-PyTorch).

    Important:
        This implementation was completely rewritten to be compatible with
        [`torch-geometric`](https://github.com/pyg-team/pytorch_geometric) library.

    Args:
        num_classes: Number of output classes.
        in_channels: Number of input channels.
        spatial_dim: Spatial dimension of the input point cloud.
        stem_channels: Number of channels in the stem layer.
        stem_type: Type of stem layer to use.
        encoder_depths: List of depths for each encoder block,
            i.e. corresponds to the number of residual blocks at each level.
        encoder_channels: List of channels for each encoder block.
        encoder_num_neighbors: List of maximum number of neighbors for each encoder block.
        grid_sizes: List of grid sizes for each downsampling block.
        radii: Search radius for each downsampling block.
        kernel_size: Size of the kernel for each KPConv block.
        kp_radius: List of kernel radius for KPConv blocks, at each level.
        kp_sigma: List of kernel extent for KPConv blocks, at each level.
        kp_influence: Influence function to use for KPConv blocks. Options are "constant", "linear", "gaussian".
        fixed_position: Whether to fix the position of the kernel points in KPConv blocks. Options are "none", "center", "vertical".
        aggregation_mode: Aggregation mode to use for the KPConv blocks. Options are "sum", "mean", "max".
        deformable: Whether to use a deformable kernel for the KPConv blocks.
        modulated: Whether to use a modulated kernel in KPConv operation.
        norm: Normalization to use for the KPConv blocks.
        act: Activation function to use for the KPConv blocks.
        bias: Whether to use a bias for the KPConv blocks.
        dropout: Dropout rate before the classification head.
        global_pool: Global pooling method to use before the classification head. Options are "max", "mean".

    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        *,
        spatial_dim: int = 3,
        stem_channels: Optional[int] = None,
        stem_type: Literal["linear", "kpconv"] = "kpconv",
        encoder_depths: Sequence[int],
        encoder_channels: Sequence[int],
        encoder_num_neighbors: Sequence[int],
        grid_sizes: Sequence[float],
        radii: Sequence[float],
        kernel_size: int,
        kp_radius: Union[float, Sequence[float]],
        kp_sigma: Union[float, Sequence[float]],
        kp_influence: str = "linear",
        fixed_position: Literal["none", "center", "vertical"] = "center",
        aggregation_mode: str = "sum",
        deformable: bool = False,
        modulated: bool = False,
        norm: NormLike = "batch_norm1d",
        act: ActLike = "leaky_relu",
        bias: bool = False,
        dropout: float = 0.0,
        global_pool: PoolLike = "max",
    ):
        super().__init__(in_channels, num_classes)
        kp_radius = ensure_tuple_size(kp_radius, size=len(encoder_depths))
        kp_sigma = ensure_tuple_size(kp_sigma, size=len(encoder_depths))

        self.stem_type = stem_type
        self.stem: Optional[nn.Module] = None
        if stem_channels is not None:
            self.stem = linear_block(
                in_features=in_channels,
                out_features=stem_channels,
                norm=norm,
                act=act,
            )
            in_channels = stem_channels

        self.encoder_blocks = create_encoder_blocks(
            in_channels=in_channels if stem_channels is None else stem_channels,
            depths=encoder_depths,
            channels=encoder_channels,
            grid_sizes=grid_sizes,
            radii=radii,
            max_num_neighbors=encoder_num_neighbors,
            spatial_dim=spatial_dim,
            kernel_size=kernel_size,
            kp_radius=kp_radius,
            kp_sigma=kp_sigma,
            kp_influence=kp_influence,
            fixed_position=fixed_position,
            aggregation_mode=aggregation_mode,
            deformable=deformable,
            modulated=modulated,
            norm=norm,
            act=act,
            bias=bias,
        )

        self.embedding_dim = encoder_channels[-1]
        self.dropout = dropout
        self.global_pool = create_pool(global_pool)
        self.head = create_cls_head(num_features=self.embedding_dim, num_classes=self.num_classes)

    def reset_classifier(self, num_classes: int, global_pool: PoolLike = "max", **kwargs: Any) -> None:
        """Resets the classification head with new parameters.

        Note:
            To set an empty classification head, use `num_classes=0`.

        Args:
            num_classes: Number of output classes.
            global_pool: Pooling method to aggregate point features ("max" or "mean").
            **kwargs: Additional keyword arguments to pass to the classification head.
        """
        self.num_classes = num_classes
        self.global_pool = create_pool(global_pool)
        self.head = create_cls_head(num_features=self.embedding_dim, num_classes=self.num_classes, **kwargs)

    @overload
    def forward_features(
        self,
        features: OptTensor,
        coords: Tensor,
        batch: Tensor,
        return_intermediates: Literal[True],
    ) -> Tuple[Tensor, Tensor, Tensor, List[Dict[str, Tensor]]]: ...

    @overload
    def forward_features(
        self,
        features: OptTensor,
        coords: Tensor,
        batch: Tensor,
        return_intermediates: Literal[False] = False,
    ) -> Tuple[Tensor, Tensor, Tensor]: ...

    def forward_features(
        self,
        features: OptTensor,
        coords: Tensor,
        batch: Tensor,
        return_intermediates: bool = False,
    ) -> Any:
        """Forward pass of the encoder, returning pre-pooling features.

        Args:
            features: Additional point features of shape $(N, features_dim)$.
            coords: Point coordinates of shape $(N, coords_dim)$.
            batch: Batch indices for each point of shape $(N,)$.

        Returns:
            Pre-pooling features of shape $(N, mlp2_dims[-1])$ where $N$ is the batch size.
        """
        features = features if features is not None else coords

        if self.stem is not None:
            features = self.stem(features)

        intermediates = []
        for i, block in enumerate(self.encoder_blocks):
            intermediate = {"features": features, "coords": coords, "batch": batch}
            features, coords, batch, inv = block(features, coords, batch, return_inverse=True)

            if i > 0:
                intermediate["pooling_inverse"] = inv
                intermediates.append(intermediate)

        if return_intermediates:
            return features, coords, batch, intermediates
        return features, coords, batch

    def forward_head(self, features: Tensor, batch: Tensor, pre_logits: bool = False) -> Tensor:
        """Forward pass of the classification head from pre-pooling features.

        Args:
            features: Pre-pooling features of shape $(N, mlp2_dims[-1])$ where $N$ is the batch size.
            batch: Batch indices for each point of shape $(N,)$.
            pre_logits: Whether to return pre-logits. Defaults to False.

        Returns:
            Classification logits of shape $(B, num_classes)$.
        """
        features = self.global_pool(features, batch)
        if self.dropout:
            features = F.dropout(features, p=float(self.dropout), training=self.training)
        return features if pre_logits else self.head(features)

    def forward(self, features: OptTensor, coords: Tensor, batch: Tensor) -> Tensor:
        """Forward pass of the classification model.

        Args:
            features: Additional point features of shape $(N, features_dim)$.
            coords: Point coordinates of shape $(N, coords_dim)$.
            batch: Batch indices for each point of shape $(N,)$.

        Returns:
            Classification logits of shape $(B, num_classes)$.
        """
        features, _, batch = self.forward_features(features, coords, batch)
        return self.forward_head(features, batch, pre_logits=False)


# TODO: Do not use FP channels, but use a simple decoder_channels
# TODO: param that will be used to create the FP blocks,
# TODO: i.e. from fp_channels=[[256], [128], [64], [32]] -> decoder_channels=[256, 128, 64, 32]
class KPConvNetSegmentation(SegmentationModel):
    """KPConv Network for segmentation tasks as described in the paper
    [KPConv: Flexible and Efficient Convolution for Point Clouds](https://arxiv.org/abs/1904.08889)
    by Hugues Thomas, Charles R. Qi, Jean-Emmanuel Deschaud, Beatriz Marcotegui, François Goulette, Leonidas J. Guibas.

    KPConv introduces a novel point convolution operator that uses kernel points to define the spatial extent and weights
    of the convolution. The kernel points are arranged in space to define the convolution pattern, with weights determined
    by their spatial correlation with input points. This allows for flexible and efficient convolution on irregular point
    clouds while maintaining permutation invariance and translation invariance. The network uses a hierarchical architecture
    with strided convolutions for spatial pooling and feature aggregation.

    Note:
        The implementation is based on the original paper and the authors' code
        [KPConv-PyTorch](https://github.com/HuguesTHOMAS/KPConv-PyTorch).

    Important:
        This implementation was completely rewritten to be compatible with
        [`torch-geometric`](https://github.com/pyg-team/pytorch_geometric) library.

    Args:
        num_classes: Number of output classes.
        in_channels: Number of input channels.
        spatial_dim: Spatial dimension of the input point cloud.
        stem_channels: Number of channels in the stem layer.
        stem_type: Type of stem layer to use.
        encoder_depths: List of depths for each encoder block,
            i.e. corresponds to the number of residual blocks at each level.
        encoder_channels: List of channels for each encoder block.
        encoder_num_neighbors: List of maximum number of neighbors for each encoder block.
        fp_channels: List of channels for each feature propagation block.
        grid_sizes: List of grid sizes for each downsampling block.
        radii: Search radius for each downsampling block.
        kernel_size: Size of the kernel for each KPConv block.
        kp_radius: List of kernel radius for KPConv blocks, at each level.
        kp_sigma: List of kernel extent for KPConv blocks, at each level.
        kp_influence: Influence function to use for KPConv blocks. Options are "constant", "linear", "gaussian".
        fixed_position: Whether to fix the position of the kernel points in KPConv blocks. Options are "none", "center", "vertical".
        aggregation_mode: Aggregation mode to use for the KPConv blocks. Options are "sum", "mean", "max".
        deformable: Whether to use a deformable kernel for the KPConv blocks.
        modulated: Whether to use a modulated kernel in KPConv operation.
        norm: Normalization to use for the KPConv blocks.
        act: Activation function to use for the KPConv blocks.
        bias: Whether to use a bias for the KPConv blocks.
        dropout: Dropout rate before the classification head.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        *,
        spatial_dim: int = 3,
        stem_channels: Optional[int] = None,
        stem_type: Literal["linear", "kpconv"] = "kpconv",
        encoder_depths: Sequence[int],
        encoder_channels: Sequence[int],
        encoder_num_neighbors: Sequence[int],
        fp_channels: Sequence[Sequence[int]],
        grid_sizes: Sequence[float],
        radii: Sequence[float],
        kernel_size: int,
        kp_radius: Union[float, Sequence[float]],
        kp_sigma: Union[float, Sequence[float]],
        kp_influence: str = "linear",
        fixed_position: Literal["none", "center", "vertical"] = "center",
        aggregation_mode: str = "sum",
        deformable: bool = False,
        modulated: bool = False,
        norm: NormLike = "batch_norm1d",
        act: ActLike = "leaky_relu",
        bias: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__(in_channels, num_classes)
        kp_radius = ensure_tuple_size(kp_radius, size=len(encoder_depths))
        kp_sigma = ensure_tuple_size(kp_sigma, size=len(encoder_depths))

        self.stem_type = stem_type
        self.stem: Optional[nn.Module] = None
        if stem_channels is not None:
            self.stem = linear_block(
                in_features=in_channels,
                out_features=stem_channels,
                norm=norm,
                act=act,
            )
            in_channels = stem_channels

        self.encoder_blocks = create_encoder_blocks(
            in_channels=in_channels if stem_channels is None else stem_channels,
            depths=encoder_depths,
            channels=encoder_channels,
            grid_sizes=grid_sizes,
            radii=radii,
            max_num_neighbors=encoder_num_neighbors,
            spatial_dim=spatial_dim,
            kernel_size=kernel_size,
            kp_radius=kp_radius,
            kp_sigma=kp_sigma,
            kp_influence=kp_influence,
            fixed_position=fixed_position,
            aggregation_mode=aggregation_mode,
            deformable=deformable,
            modulated=modulated,
            norm=norm,
            act=act,
            bias=bias,
        )

        skip_channels = [in_channels] + list(encoder_channels[:-1])
        self.fp_blocks = create_fp_blocks(
            in_channels=encoder_channels[-1],
            skip_channels=skip_channels[::-1],
            fp_channels=fp_channels,
            bias=bias,
            act=act,
            norm=norm,
        )

        self.embedding_dim = fp_channels[-1][-1]
        self.dropout = dropout
        self.head = create_cls_head(num_features=self.embedding_dim, num_classes=self.num_classes)

    def reset_classifier(self, num_classes: int, **kwargs: Any) -> None:
        self.num_classes = num_classes
        self.head = create_cls_head(num_features=self.embedding_dim, num_classes=self.num_classes, **kwargs)

    @overload
    def forward_features(
        self,
        features: OptTensor,
        coords: Tensor,
        batch: Tensor,
        return_intermediates: Literal[True],
    ) -> Tuple[Tensor, Tensor, Tensor, List[Dict[str, Tensor]]]: ...

    @overload
    def forward_features(
        self,
        features: OptTensor,
        coords: Tensor,
        batch: Tensor,
        return_intermediates: Literal[False] = False,
    ) -> Tuple[Tensor, Tensor, Tensor]: ...

    def forward_features(
        self,
        features: OptTensor,
        coords: Tensor,
        batch: Tensor,
        return_intermediates: bool = False,
    ) -> Any:
        features = features if features is not None else coords

        if self.stem is not None:
            features = self.stem(features)

        intermediates = [{"features": features, "coords": coords, "batch": batch}] if return_intermediates else []
        for i, block in enumerate(self.encoder_blocks):
            features, coords, batch, inv = block(features, coords, batch, return_inverse=True)
            if return_intermediates and i < len(self.encoder_blocks) - 1:
                # NOTE: Do not store the last result, as it will be the returned output.
                intermediates.append({"features": features, "coords": coords, "batch": batch})

        if return_intermediates:
            return features, coords, batch, intermediates
        return features, coords, batch

    def forward_decoder(
        self,
        features: Tensor,
        coords: Tensor,
        batch: Tensor,
        intermediates: List[Dict[str, Tensor]],
    ) -> Tensor:
        for block, intermediate in zip(self.fp_blocks, reversed(intermediates)):
            features_skip = intermediate["features"]
            coords_skip = intermediate["coords"]
            batch_skip = intermediate["batch"]

            features, coords, batch = block(features, coords, batch, features_skip, coords_skip, batch_skip)
        return features

    def forward_head(self, features: Tensor, pre_logits: bool = False) -> Tensor:
        if self.dropout:
            features = F.dropout(features, p=float(self.dropout), training=self.training)
        return features if pre_logits else self.head(features)

    def forward(self, features: OptTensor, coords: Tensor, batch: Tensor) -> Tensor:
        features, coords, batch, intermediates = self.forward_features(
            features, coords, batch, return_intermediates=True
        )
        features = self.forward_decoder(features, coords, batch, intermediates)
        return self.forward_head(features)


def _kpconvnet_small_clf(in_channels: int, num_classes: int, **kwargs: Any) -> KPConvNetClassification:
    hparams: Dict[str, Any] = dict(
        in_channels=in_channels,
        num_classes=num_classes,
        stem_channels=32,
        stem_type="kpconv",
        encoder_depths=[1, 3, 3, 3],
        encoder_channels=[64, 128, 256, 512],
        encoder_num_neighbors=[20, 35, 40, 40],
        grid_sizes=[0.08, 0.16, 0.32],
        radii=[0.1, 0.2, 0.4, 0.8],
        kernel_size=15,
        kp_radius=[0.1, 0.2, 0.4, 0.8],
        kp_sigma=[0.05, 0.1, 0.2, 0.4],
        act="leaky_relu",
        norm=partial(torch.nn.BatchNorm1d, momentum=0.05),
    )
    hparams.update(kwargs)

    return KPConvNetClassification(**hparams)


@register_model("kpconv-original.modelnet40", task="classification")
def kpconvnet_original_clf(in_channels: int = 6, num_classes: int = 40, **kwargs: Any) -> KPConvNetClassification:
    return _kpconvnet_small_clf(in_channels=in_channels, num_classes=num_classes, **kwargs)


@register_model("kpconv-sm.modelnet40", task="classification")
def kpconvnet_small_clf(in_channels: int = 6, num_classes: int = 40, **kwargs: Any) -> KPConvNetClassification:
    return _kpconvnet_small_clf(in_channels=in_channels, num_classes=num_classes, **kwargs)
