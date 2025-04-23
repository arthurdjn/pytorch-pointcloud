import math
import random
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Sequence, Tuple, overload

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.init import kaiming_uniform_
from torch.nn.parameter import Parameter

from torch_pointcloud.layers import MLP, ActLike, NormLike, create_act, create_cls_head, create_norm, create_pool
from torch_pointcloud.layers.blocks import linear_block
from torch_pointcloud.utils.config import CACHE_DIR
from torch_pointcloud.utils.conversion import ensure_tuple, ensure_tuple_size
from torch_pointcloud.utils.geometry import rodrigues_rotation_matrix, spherical_points_gradient, spherical_points_lloyd
from torch_pointcloud.utils.imports import optional_import
from torch_pointcloud.utils.ops import consecutive_cluster, softmax, voxel_grid
from torch_pointcloud.utils.types import OptTensor, ValueCollection

if TYPE_CHECKING:
    from torch_cluster import knn_graph
    from torch_scatter import scatter, scatter_add, scatter_max, scatter_sum, segment_csr

knn_graph, _ = optional_import("torch_cluster", name="knn_graph")
scatter, _ = optional_import("torch_scatter", name="scatter")
scatter_add, _ = optional_import("torch_scatter", name="scatter_add")
scatter_sum, _ = optional_import("torch_scatter", name="scatter_sum")
scatter_max, _ = optional_import("torch_scatter", name="scatter_max")
segment_csr, _ = optional_import("torch_scatter", name="segment_csr")


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
        kernel_size: int,
        p_dim: int,
        in_channels: int,
        out_channels: int,
        kp_extent: float,
        radius: float,
        fixed_kernel_points: Literal["none", "center", "vertical"] = "center",
        kp_influence: str = "linear",
        aggregation_mode: str = "sum",
        deformable: bool = False,
        modulated: bool = False,
    ) -> None:
        super().__init__()

        self.kernel_size = kernel_size
        self.p_dim = p_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kp_extent = kp_extent
        self.radius = radius
        self.fixed_kernel_points = fixed_kernel_points
        self.kp_influence = kp_influence
        self.aggregation_mode = aggregation_mode
        self.modulated = modulated

        # Initialize parameters
        self.weights = Parameter(torch.zeros(kernel_size, in_channels, out_channels), requires_grad=True)
        self.init_weights_()

        kernel = create_kernel_points(self.radius, self.kernel_size, fixed_position=self.fixed_kernel_points)
        self.register_buffer("kernel", kernel)

    def init_weights_(self) -> None:
        # kaiming_uniform_(self.weights, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        # if self.bias is not None:
        #     fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
        #     bound = 1 / math.sqrt(fan_in)
        #     nn.init.uniform_(self.bias, -bound, bound)

    def _init_offsets(self) -> Tuple[nn.Module, Tensor]:
        offset_dim = (self.p_dim + 1) * self.kernel_size if self.modulated else self.p_dim * self.kernel_size
        offset_conv = KPConv(
            kernel_size=self.kernel_size,
            p_dim=self.p_dim,
            in_channels=self.in_channels,
            out_channels=offset_dim,
            kp_extent=self.kp_extent,
            radius=self.radius,
            fixed_kernel_points=self.fixed_kernel_points,
            kp_influence=self.kp_influence,
            aggregation_mode=self.aggregation_mode,
        )
        offset_bias = Parameter(torch.zeros(offset_dim), requires_grad=True)
        return offset_conv, offset_bias

    def _init_kernel(self) -> Tensor:
        kernel_points = create_kernel_points(self.radius, self.kernel_size, fixed_position=self.fixed_kernel_points)
        return Parameter(kernel_points, requires_grad=False)

    def _compute_weights(self, sq_distances: Tensor) -> Tensor:
        if self.kp_influence == "constant":
            return torch.ones_like(sq_distances)
        elif self.kp_influence == "linear":
            return torch.clamp(1 - torch.sqrt(sq_distances) / self.kp_extent, min=0.0)
        elif self.kp_influence == "gaussian":
            # sigma = self.kp_extent * 0.3
            return torch.exp(-sq_distances / (2 * self.kp_extent**2 + 1e-6))
        else:
            raise ValueError(f"Unknown influence type: {self.kp_influence}")

    def forward(
        self,
        features: Tensor,
        query_coords: Tensor,
        support_coords: Tensor,
        edge_index: Tensor,
    ) -> Tensor:
        row, col = edge_index
        rel_coords = support_coords[col] - query_coords[row]
        print(f"rel_coords: {rel_coords.shape}")

        kernel_points = self.kernel.unsqueeze(0).expand(rel_coords.size(0), -1, -1)
        print(f"kernel_points: {kernel_points.shape}")

        rel_coords = rel_coords.unsqueeze(1)  # [E, 1, p_dim]
        sq_distances = torch.sum((rel_coords - kernel_points) ** 2, dim=-1)  # [E, K]
        weights = self._compute_weights(sq_distances)
        print(f"weights: {weights.shape}")

        if self.aggregation_mode == "closest":
            neighbors_1nn = torch.argmin(sq_distances, dim=1)  # [E]
            one_hot = torch.zeros_like(weights).scatter_(1, neighbors_1nn.unsqueeze(1), 1)
            weights = weights * one_hot  # [E, K]

        source_features = features[col]  # [E, in_channels]
        output = torch.zeros(query_coords.size(0), self.out_channels, device=features.device, dtype=features.dtype)
        print(f"output: {output.shape}, {source_features.shape}, {weights.shape}")
        for k in range(self.kernel_size):
            weights_k = weights[:, k].unsqueeze(1)  # [E, 1]
            weighted_features = weights_k * source_features  # [E, in_channels]
            transformed_features = torch.matmul(weighted_features, self.weights[k].to(features.dtype))
            scatter_add(transformed_features, row, dim=0, out=output)

        return output

    def extra_repr(self) -> str:
        return (
            f"radius={self.radius}, in_channels={self.in_channels}, out_channels={self.out_channels} "
            f"kp_extent={self.kp_extent}, fixed_kernel_points={self.fixed_kernel_points!r}, "
            f"kp_influence={self.kp_influence!r}, aggregation_mode={self.aggregation_mode!r}, "
            # f"deformable={self.deformable}, modulated={self.modulated}"
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.extra_repr()})"


class KPConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        radius: float,
        sigma: float,
        groups: int = 1,
        dimension: int = 3,
        norm: NormLike = "layer_norm",
        act: ActLike = "leaky_relu",
        bias: bool = True,
    ):
        super().__init__()

        self.conv = KPConv(
            kernel_size=kernel_size,
            p_dim=3,
            in_channels=in_channels,
            out_channels=out_channels,
            kp_extent=sigma,
            radius=radius,
            fixed_kernel_points="center",
            kp_influence="linear",
            aggregation_mode="sum",
            deformable=False,
            modulated=False,
        )
        self.norm = create_norm(norm, out_channels) if norm is not None else None
        self.act = create_act(act) if act is not None else None

    def forward(self, features: Tensor, coords: Tensor, edge_index: Tensor) -> Tensor:
        features = self.conv(features, coords, edge_index)
        if self.norm is not None:
            features = self.norm(features)
        if self.act is not None:
            features = self.act(features)
        return features


class UnaryBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        norm: NormLike = "layer_norm",
        act: ActLike = "leaky_relu",
    ):
        super().__init__()
        self.mlp = nn.Linear(in_channels, out_channels, bias=bias)
        self.norm = create_norm(norm, out_channels) if norm is not None else None
        self.act = create_act(act) if act is not None else None

    def forward(self, x: Tensor) -> Tensor:
        x = self.mlp(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        return x


class KPResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        radius: float,
        sigma: float,
        groups: int = 1,
        dimension: int = 3,
        strided: bool = False,
        norm: NormLike = "layer_norm",
        act: ActLike = "leaky_relu",
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.strided = strided

        mid_channels = out_channels // 4

        self.unary1 = UnaryBlock(in_channels, mid_channels, norm=norm, act=act)
        self.conv = KPConvBlock(
            mid_channels,
            mid_channels,
            kernel_size,
            radius,
            sigma,
            groups=groups,
            dimension=dimension,
            norm=norm,
            act=act,
        )
        self.unary2 = UnaryBlock(mid_channels, out_channels, norm=norm, act=None)

        if in_channels != out_channels:
            self.shortcut = UnaryBlock(in_channels, out_channels, norm=norm, act=None)
        else:
            self.shortcut = nn.Identity()  # type: ignore

        self.act = create_act(act) if act is not None else None

    def forward(self, features: Tensor, coords: Tensor, edge_index: Tensor) -> Tensor:
        shortcut = features
        if self.strided:
            row, col = edge_index
            shortcut = scatter_max(features[row], col, dim=0)[0]

        shortcut = self.shortcut(shortcut)
        features = self.unary1(features)
        features = self.conv(features, coords, edge_index)
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
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        radius: float,
        sigma: float,
        num_neighbors: int,
        groups: int = 1,
        norm: NormLike = "batch_norm1d",
        act: ActLike = "relu",
        attn_drop: ValueCollection[float] = 0.0,
        drop_path: ValueCollection[float] = 0.0,
        downsample: Optional[GridPool] = None,
    ):
        super().__init__()
        attn_drop = ensure_tuple_size(attn_drop, depth)
        drop_path = ensure_tuple_size(drop_path, depth)

        self.num_neighbors = num_neighbors
        self.downsample = downsample

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = KPResidualBlock(
                in_channels=in_channels if i == 0 else out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                radius=radius,
                sigma=sigma,
                groups=groups,
                dimension=3,
                strided=False,
                norm=norm,
                act=act,
            )
            self.blocks.append(block)

    def forward(
        self,
        features: Tensor,
        coords: Tensor,
        batch: Tensor,
        return_inverse: bool = False,
    ) -> Tuple[Tensor, ...]:
        inv = None
        if self.downsample is not None:
            features, coords, batch, inv = self.downsample(features, coords, batch, return_inverse=True)

        neighbors = knn_graph(coords, k=self.num_neighbors, batch=batch, loop=False, flow="source_to_target")
        for block in self.blocks:
            features = block(features, coords, neighbors)

        if return_inverse:
            return features, coords, batch, inv
        return features, coords, batch


def create_encoder_blocks(
    depths: Sequence[int],
    channels: Sequence[int],
    num_groups: Sequence[int],
    num_neighbors: Sequence[int],
    grid_sizes: Sequence[float],
    kernel_size: int,
    radii: Sequence[float],
    sigmas: Sequence[float],
    norm: NormLike = "batch_norm1d",
    act: ActLike = "relu",
) -> nn.ModuleList:
    depths = ensure_tuple(depths)
    n = len(depths)
    channels = ensure_tuple_size(channels, size=n, extra_msg="Encoder length `channels` != `depths` + 1.")
    num_groups = ensure_tuple_size(num_groups, size=n, extra_msg="Encoder length `num_groups` != `depths`.")
    num_neighbors = ensure_tuple_size(num_neighbors, size=n, extra_msg="Encoder length `num_neighbors` != `depths`.")
    grid_sizes = ensure_tuple_size(grid_sizes, size=n - 1, extra_msg="Encoder length `grid_sizes` != `depths` - 1.")

    blocks = nn.ModuleList()
    for i in range(n):
        downsample: Optional[GridPool] = None
        if i > 0:
            downsample = GridPool(grid_size=grid_sizes[i - 1], reduce="max")

        block = EncoderBlock(
            depth=depths[i],
            in_channels=channels[i - 1] if i > 0 else channels[i],
            out_channels=channels[i],
            kernel_size=kernel_size,
            radius=radii[i],
            sigma=sigmas[i],
            num_neighbors=num_neighbors[i],
            groups=num_groups[i],
            norm=norm,
            act=act,
            downsample=downsample,
        )
        blocks.append(block)
    return blocks


class KPConvClassification(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        *,
        encoder_depths: Sequence[int],
        encoder_channels: Sequence[int],
        encoder_num_groups: Sequence[int],
        encoder_num_neighbors: Sequence[int],
        grid_sizes: Sequence[float],
        kernel_size: int,
        radii: Sequence[float],
        sigmas: Sequence[float],
        norm: NormLike = "batch_norm1d",
        act: ActLike = "relu",
        dropout: float = 0.0,
        global_pool: str = "max",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.stem = nn.Sequential(
            nn.Linear(in_channels, encoder_channels[0]),
            create_norm(norm, encoder_channels[0]),
            create_act(act),
        )

        self.encoder = create_encoder_blocks(
            depths=encoder_depths,
            channels=encoder_channels,
            num_groups=encoder_num_groups,
            num_neighbors=encoder_num_neighbors,
            grid_sizes=grid_sizes,
            kernel_size=kernel_size,
            radii=radii,
            sigmas=sigmas,
            norm=norm,
            act=act,
        )

        self.embedding_dim = encoder_channels[-1]
        self.dropout = dropout
        self.global_pool = create_pool(global_pool)
        self.head = create_cls_head(num_features=self.embedding_dim, num_classes=self.num_classes)

    def reset_classifier(self, num_classes: int, global_pool: str = "max", **kwargs: Any) -> None:
        self.num_classes = num_classes
        self.global_pool = create_pool(global_pool) if isinstance(global_pool, str) else global_pool
        self.head = create_cls_head(num_features=self.embedding_dim, num_classes=self.num_classes, **kwargs)

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

        intermediates: List[Dict[str, Tensor]] = []
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
        features, _, batch = self.forward_features(features, coords, batch)
        return self.forward_head(features, batch)
