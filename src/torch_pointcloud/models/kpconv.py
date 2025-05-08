import math
import random
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Sequence, Tuple, Union, overload

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter

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

if TYPE_CHECKING:
    from torch_cluster import radius, radius_graph
    from torch_scatter import scatter, scatter_add

radius, _ = optional_import("torch_cluster", name="radius")
radius_graph, _ = optional_import("torch_cluster", name="radius_graph")
scatter, _ = optional_import("torch_scatter", name="scatter")
scatter_add, _ = optional_import("torch_scatter", name="scatter_add")


def create_kernel_points(
    radius: float,
    num_points: int,
    fixed_position: Literal["none", "center", "vertical"] = "center",
    method: Literal["lloyd", "gradient"] = "lloyd",
) -> torch.Tensor:
    if num_points > 30 and method != "lloyd":
        warnings.warn("Too many points, consider using Lloyds algorithm `method='lloyd'`.")

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
        fixed_kernel_points: Literal["none", "center", "vertical"] = "center",
        kp_influence: str = "linear",
        aggregation_mode: str = "sum",
        deformable: bool = False,
        modulated: bool = False,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.spatial_dim = spatial_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.kp_radius = kp_radius
        self.kp_sigma = kp_sigma
        self.fixed_kernel_points = fixed_kernel_points
        self.kp_influence = kp_influence
        self.aggregation_mode = aggregation_mode
        self.modulated = modulated

        # Initialize parameters
        self.weights = Parameter(torch.zeros(kernel_size, in_channels, out_channels), requires_grad=True)
        self.init_weights_()

        self.register_buffer("kernel", self.configure_kernel())

    def init_weights_(self) -> None:
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        # if self.bias is not None:
        #     fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
        #     bound = 1 / math.sqrt(fan_in)
        #     nn.init.uniform_(self.bias, -bound, bound)

    def configure_offsets(self) -> Tuple[nn.Module, Tensor]:
        offset_dim = self.spatial_dim * self.kernel_size
        if self.modulated:
            offset_dim += self.spatial_dim

        offset_conv = KPConv(
            spatial_dim=self.spatial_dim,
            in_channels=self.in_channels,
            out_channels=offset_dim,
            kernel_size=self.kernel_size,
            kp_radius=self.kp_radius,
            kp_sigma=self.kp_sigma,
            fixed_kernel_points=self.fixed_kernel_points,
            kp_influence=self.kp_influence,
            aggregation_mode=self.aggregation_mode,
        )
        offset_bias = Parameter(torch.zeros(offset_dim), requires_grad=True)
        return offset_conv, offset_bias

    def configure_kernel(self) -> Tensor:
        return create_kernel_points(self.kp_radius, self.kernel_size, fixed_position=self.fixed_kernel_points)

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

    def forward(
        self,
        features: Tensor,
        coords_query: Tensor,
        coords_support: Tensor,
        edge_index: Tensor,
    ) -> Tensor:
        row, col = edge_index
        rel_coords = coords_support[col] - coords_query[row]

        kernel_points = self.kernel.unsqueeze(0).expand(rel_coords.size(0), -1, -1)  # type: ignore[operator]

        rel_coords = rel_coords.unsqueeze(1)  # [E, 1, p_dim]
        sq_distances = torch.sum((rel_coords - kernel_points) ** 2, dim=-1)  # [E, K]
        weights = self._compute_weights(sq_distances)

        if self.aggregation_mode == "closest":
            neighbors_1nn = torch.argmin(sq_distances, dim=1)  # [E]
            one_hot = torch.zeros_like(weights).scatter_(1, neighbors_1nn.unsqueeze(1), 1)
            weights = weights * one_hot  # [E, K]

        source_features = features[col]  # [E, in_channels]
        output = torch.zeros(coords_query.size(0), self.out_channels, device=features.device, dtype=features.dtype)
        for k in range(self.kernel_size):
            weights_k = weights[:, k].unsqueeze(1)  # [E, 1]
            weighted_features = weights_k * source_features  # [E, in_channels]
            transformed_features = torch.matmul(weighted_features, self.weights[k].to(features.dtype))
            scatter_add(transformed_features, row, dim=0, out=output)

        return output

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"kp_radius={self.kp_radius}, "
            f"kp_sigma={self.kp_sigma}, "
            f"kp_influence={self.kp_influence!r}, "
            f"fixed_kernel_points={self.fixed_kernel_points!r}, "
            f"aggregation_mode={self.aggregation_mode!r}, "
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
            fixed_kernel_points="center",
            kp_influence="linear",
            aggregation_mode="sum",
            deformable=False,
            modulated=False,
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


# class UnaryBlock(nn.Module):
#     def __init__(
#         self,
#         in_channels: int,
#         out_channels: int,
#         bias: bool = True,
#         norm: NormLike = "layer_norm",
#         act: ActLike = "leaky_relu",
#     ):
#         super().__init__()
#         self.mlp = nn.Linear(in_channels, out_channels, bias=bias)
#         self.norm = create_norm(norm, out_channels) if norm is not None else None
#         self.act = create_act(act) if act is not None else None

#     def forward(self, x: Tensor) -> Tensor:
#         x = self.mlp(x)
#         if self.norm is not None:
#             x = self.norm(x)
#         if self.act is not None:
#             x = self.act(x)
#         return x


class KPResidualBlock(nn.Module):
    def __init__(
        self,
        spatial_dim: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        kp_radius: float,
        kp_sigma: float,
        strided: bool = False,
        norm: NormLike = "layer_norm",
        act: ActLike = "leaky_relu",
        bias: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.strided = strided

        mid_channels = out_channels // 4

        self.unary1 = MLP(in_channels=in_channels, out_channels=mid_channels, norm=norm, act=act, dropout=None)
        self.conv = KPConvBlock(
            spatial_dim=spatial_dim,
            in_channels=mid_channels,
            out_channels=mid_channels,
            kernel_size=kernel_size,
            kp_radius=kp_radius,
            kp_sigma=kp_sigma,
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
        spatial_dim: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        kp_radius: Union[float, Sequence[float]],
        kp_sigma: Union[float, Sequence[float]],
        radius: float,
        max_num_neighbors: int,
        norm: NormLike = "batch_norm1d",
        act: ActLike = "relu",
        bias: bool = False,
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
            block = KPResidualBlock(
                spatial_dim=spatial_dim,
                in_channels=in_channels if i == 0 else out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                kp_radius=kp_radius[i],
                kp_sigma=kp_sigma[i],
                strided=downsample is not None and i == 0,
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
    norm: NormLike = "batch_norm1d",
    act: ActLike = "relu",
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
            spatial_dim=spatial_dim,
            depth=depths[i],
            in_channels=in_channels,
            out_channels=channels[i],
            kernel_size=kernel_size,
            kp_radius=([kp_radius[i - 1]] + [kp_radius[i]] * (depths[i] - 1)) if i > 0 else kp_radius[i],
            kp_sigma=([kp_sigma[i - 1]] + [kp_sigma[i]] * (depths[i] - 1)) if i > 0 else kp_sigma[i],
            radius=radii[i],
            max_num_neighbors=max_num_neighbors[i],
            norm=norm,
            act=act,
            downsample=downsample,
            bias=bias,
        )
        blocks.append(block)
        in_channels = channels[i]
    return blocks


class KPConvNetClassification(nn.Module):
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
        norm: NormLike = "batch_norm1d",
        act: ActLike = "relu",
        dropout: float = 0.0,
        global_pool: PoolLike = "max",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.stem: Optional[nn.Module] = None
        if stem_channels is not None:
            self.stem = linear_block(
                in_features=in_channels,
                out_features=stem_channels,
                norm=norm,
                act=act,
            )
            in_channels = stem_channels

        self.encoder = create_encoder_blocks(
            in_channels=in_channels if stem_channels is None else stem_channels,
            depths=encoder_depths,
            channels=encoder_channels,
            grid_sizes=grid_sizes,
            radii=radii,
            max_num_neighbors=encoder_num_neighbors,
            kernel_size=kernel_size,
            kp_radius=kp_radius,
            kp_sigma=kp_sigma,
            norm=norm,
            act=act,
        )

        self.embedding_dim = encoder_channels[-1]
        self.dropout = dropout
        self.global_pool = create_pool(global_pool)
        self.head = create_cls_head(num_features=self.embedding_dim, num_classes=self.num_classes)

    def reset_classifier(self, num_classes: int, global_pool: PoolLike = "max", **kwargs: Any) -> None:
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
        features = features if features is not None else coords

        if self.stem is not None:
            features = self.stem(features)

        intermediates = []
        for i, block in enumerate(self.encoder):
            intermediate = {"features": features, "coords": coords, "batch": batch}
            features, coords, batch, inv = block(features, coords, batch, return_inverse=True)

            if i > 0:
                intermediate["pooling_inverse"] = inv
                intermediates.append(intermediate)

        if return_intermediates:
            return features, coords, batch, intermediates
        return features, coords, batch

    def forward_head(self, features: Tensor, batch: Tensor, pre_logits: bool = False) -> Tensor:
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
