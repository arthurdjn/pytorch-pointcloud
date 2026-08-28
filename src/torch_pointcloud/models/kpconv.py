"""KPConv classification and segmentation models.

{{ paper("1904.08889") }}
"""

import math
import random
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union, overload

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import MLP
from torch_geometric.nn.pool import voxel_grid
from torch_geometric.nn.pool.consecutive import consecutive_cluster

import torch_pointcloud.transforms as T
from torch_pointcloud.config import CACHE_DIR
from torch_pointcloud.datasets.s3dis import S3DIS_CLASSES
from torch_pointcloud.layers import PoolLike, create_pool
from torch_pointcloud.layers.act import create_act
from torch_pointcloud.layers.norms import create_norm
from torch_pointcloud.utils.conversion import ensure_tuple, ensure_tuple_size
from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.geometry import rodrigues_rotation_matrix, spherical_points_gradient, spherical_points_lloyd
from torch_pointcloud.utils.imports import _TORCH_CLUSTER_GITHUB_URL, _TORCH_SCATTER_GITHUB_URL, optional_import
from torch_pointcloud.utils.types import OptTensor

from ._base import ClassificationModel, SegmentationModel
from ._registry import WeightsDict, register_model
from .pointnet2 import PointNet2Decoder

if TYPE_CHECKING:
    from torch_cluster import radius, radius_graph
    from torch_scatter import scatter

radius, _ = optional_import("torch_cluster", name="radius", url=_TORCH_CLUSTER_GITHUB_URL)
radius_graph, _ = optional_import("torch_cluster", name="radius_graph", url=_TORCH_CLUSTER_GITHUB_URL)
scatter, _ = optional_import("torch_scatter", name="scatter", url=_TORCH_SCATTER_GITHUB_URL)


def create_kernel_points(
    radius: float,
    num_points: int,
    fixed_position: Literal["none", "center", "vertical"] = "center",
    method: Literal["lloyd", "gradient"] = "lloyd",
) -> torch.Tensor:
    r"""Builds the kernel point positions of a KPConv kernel, randomly rotated and jittered.

    Positions are optimized on the unit sphere, cached under `CACHE_DIR` and reused across calls.

    Args:
        radius: Radius of the sphere the kernel points are scaled to.
        num_points: Number of kernel points $K$.
        fixed_position: Which kernel point is pinned: `"none"`, `"center"`, or `"vertical"`.
        method: Optimization used to spread the points, either `"lloyd"` or `"gradient"`.

    Returns:
        Kernel point positions of shape $(K, 3)$.
    """
    if method not in ["lloyd", "gradient"]:
        raise ValueError(f"Unknown method: {method!r}, expected 'lloyd' or 'gradient'.")
    if num_points > 30 and method != "lloyd":
        warnings.warn("Too many points, consider using Lloyds algorithm with `method='lloyd'`.", stacklevel=2)

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
                return_grad_norms=True,
            )

        kernel_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(kernel_points, kernel_path)

    # Random rotations for the kernel
    R = torch.eye(3)
    theta = torch.rand(1).item() * 2 * math.pi

    if fixed_position != "vertical":
        c, s = math.cos(theta), math.sin(theta)
        R = torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=torch.float32)
    else:
        phi = (torch.rand(1).item() - 0.5) * math.pi
        # Create the first vector in cartesian coordinates
        u = torch.tensor([math.cos(theta) * math.cos(phi), math.sin(theta) * math.cos(phi), math.sin(phi)])
        # Choose a random rotation angle
        alpha = random.random() * 2 * math.pi
        R = rodrigues_rotation_matrix(u, theta=alpha)

    # Add a small noise, scale and rotate
    kernel_points += torch.normal(mean=0, std=0.01, size=kernel_points.shape)
    kernel_points *= radius
    kernel_points = torch.matmul(kernel_points, R)

    return kernel_points


class KPConv(nn.Module):
    r"""Kernel point convolution over a neighborhood graph.

    Each of the $K$ kernel points carries its own weight matrix, and a neighbor contributes to a kernel point
    with a weight given by their distance through the influence function. When `deformable` is set, a nested
    `KPConv` predicts a per-point offset for every kernel point.

    Args:
        spatial_dim: Spatial dimension of the input point cloud.
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Number of kernel points $K$.
        kp_radius: Radius of the sphere the kernel points are placed on.
        kp_sigma: Kernel extent, the distance over which a kernel point still influences a neighbor.
        kp_influence: Influence function: `"constant"`, `"linear"`, or `"gaussian"`.
        fixed_position: Which kernel point is pinned: `"none"`, `"center"`, or `"vertical"`.
        aggregation_mode: `"sum"` over all kernel points, or `"closest"` to keep only the nearest one.
        deformable: Whether to predict per-point kernel offsets.
        modulated: Whether the deformable branch also predicts a per-kernel-point modulation.
        bias: Whether to add a bias to the output.
        track_running_stats: Whether to keep the deformable activations of the last forward pass, for regularization.
    """

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

        # Track running statistics (mostly for regularization).
        # Plain attributes, not buffers: these hold per-forward activations, so registering them
        # would bloat checkpoints, retain autograd graphs and break DDP buffer broadcasts.
        self.track_running_stats = track_running_stats
        self.running_min_d2: OptTensor = None
        self.running_deformed_kernel: OptTensor = None
        self.running_offset_features: OptTensor = None

    @property
    def deformable(self) -> bool:
        """Whether the kernel points are shifted by predicted per-point offsets."""
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
        """Builds the rigid `KPConv` and the bias predicting the deformable offsets (and modulations)."""
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
        """Builds the fixed kernel point positions registered as the `kernel` buffer."""
        return create_kernel_points(self.kp_radius, self.kernel_size, fixed_position=self.fixed_position)

    def _compute_weights(self, sq_distances: Tensor) -> Tensor:
        if self.kp_influence == "constant":
            return torch.ones_like(sq_distances)
        elif self.kp_influence == "linear":
            return torch.clamp(1 - torch.sqrt(sq_distances) / self.kp_sigma, min=0.0)
        elif self.kp_influence == "gaussian":
            return torch.exp(-sq_distances / (2 * self.kp_sigma**2 + 1e-6))
        else:
            raise ValueError(f"Unknown influence type: {self.kp_influence}")

    def _compute_offsets(
        self,
        x: Tensor,
        pos_query: Tensor,
        pos_support: Tensor,
        edge_index: Tensor,
    ) -> Tuple[OptTensor, OptTensor]:
        if self.offset_conv is None:
            return None, None

        offset_x = self.offset_conv(x, pos_query, pos_support, edge_index)
        if self.offset_bias is not None:
            offset_x = offset_x + self.offset_bias

        if self.track_running_stats:
            self.running_offset_features = offset_x

        if self.modulated:
            # Split into offsets and modulations
            unscaled_offsets = offset_x[:, : self.spatial_dim * self.kernel_size]
            unscaled_offsets = unscaled_offsets.view(-1, self.kernel_size, self.spatial_dim)

            # Get modulations (sigmoid to keep between 0 and 2)
            modulations = 2 * torch.sigmoid(offset_x[:, self.spatial_dim * self.kernel_size :])
            modulations = modulations.view(-1, self.kernel_size)
        else:
            # Just offsets, no modulations
            unscaled_offsets = offset_x.view(-1, self.kernel_size, self.spatial_dim)
            modulations = None

        # Scale offsets by kp_sigma (equivalent to KP_extent in original)
        offsets = unscaled_offsets * self.kp_sigma

        return offsets, modulations

    def forward(
        self,
        x: Tensor,
        pos_query: Tensor,
        pos_support: Tensor,
        edge_index: Tensor,
    ) -> Tensor:
        row, col = edge_index
        pos_rel = pos_support[col] - pos_query[row]

        # For deformable KPConv, compute offsets and modulations
        offsets, modulations = self._compute_offsets(x, pos_query, pos_support, edge_index)

        # Get kernel points at each point
        if self.deformable and offsets is not None:
            kernel_points = self.kernel.unsqueeze(0) + offsets[row]  # type: ignore[operator]
            if self.track_running_stats:
                self.running_deformed_kernel = kernel_points
        else:
            kernel_points = self.kernel.unsqueeze(0).expand(pos_rel.size(0), -1, -1)  # type: ignore[operator]

        pos_rel = pos_rel.unsqueeze(1)  # [E, 1, p_dim]
        sq_distances = torch.sum((pos_rel - kernel_points) ** 2, dim=-1)  # [E, K]

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

        source_x = x[col]  # [E, in_channels]
        output = torch.zeros(pos_query.size(0), self.out_channels, device=x.device, dtype=x.dtype)
        for k in range(self.kernel_size):
            weights_k = weights[:, k].unsqueeze(1)  # [E, 1]
            weighted_x = weights_k * source_x  # [E, in_channels]
            transformed_x = torch.matmul(weighted_x, self.weight[k].to(x.dtype))
            # Autocast makes the matmul low precision, but scatter's out= buffer requires a matching
            # dtype, so cast back to output (fp32 under AMP) and accumulate the reduction there.
            scatter(transformed_x.to(output.dtype), row, dim=0, out=output, reduce="sum")

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
    """Kernel point convolution followed by normalization and activation."""

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
        act: Union[str, Callable, None] = "leaky_relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
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
        self.norm = create_norm(norm, out_channels, **(norm_kwargs or {})) or nn.Identity()
        self.act = create_act(act, **(act_kwargs or {})) or nn.Identity()

    def forward(self, x: Tensor, pos_query: Tensor, pos_support: Tensor, edge_index: Tensor) -> Tensor:
        x = self.conv(x, pos_query, pos_support, edge_index)
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        return x


class KPResidualBlock(nn.Module):
    """Bottleneck residual block: a channel-reducing MLP, a `KPConvBlock`, and a channel-restoring MLP.

    Set `strided=True` when the block maps a support cloud onto a coarser query cloud, so that the skip
    connection is max-pooled over the same neighborhoods.
    """

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
        act: Union[str, Callable, None] = "leaky_relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: bool = False,
    ):
        super().__init__()
        mid_channels = max(out_channels // 4, 8)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.strided = strided

        mlp_kwargs: Dict[str, Any] = dict(
            act_first=act_first,
            norm=norm,
            norm_kwargs=norm_kwargs,
            bias=bias,
            plain_last=False,
        )
        self.unary1 = MLP([in_channels, mid_channels], act=act, act_kwargs=act_kwargs, **mlp_kwargs)
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
            act=act,
            act_kwargs=act_kwargs,
            norm=norm,
            norm_kwargs=norm_kwargs,
            bias=bias,
        )
        self.unary2 = MLP([mid_channels, out_channels], act=None, **mlp_kwargs)
        self.shortcut = (
            MLP([in_channels, out_channels], act=None, **mlp_kwargs) if in_channels != out_channels else nn.Identity()
        )
        self.act = create_act(act, **(act_kwargs or {})) or nn.Identity()

    def forward(self, x: Tensor, pos_query: Tensor, pos_support: Tensor, edge_index: Tensor) -> Tensor:
        shortcut = x
        if self.strided:
            row, col = edge_index
            shortcut = scatter(x[col], row, dim=0, dim_size=pos_query.size(0), reduce="max").clamp(min=0)

        shortcut = self.shortcut(shortcut)
        x = self.unary1(x)
        x = self.conv(x, pos_query, pos_support, edge_index)
        x = self.unary2(x)

        x = x + shortcut
        if self.act is not None:
            x = self.act(x)

        return x


class KPFCNNGridPool(nn.Module):
    """Voxel grid subsampling: points falling in the same cell are reduced to a single point.

    Positions are averaged within a cell while features use the `reduce` operation.
    """

    def __init__(self, grid_size: float, reduce: str = "max"):
        super().__init__()
        self.grid_size = grid_size
        self.reduce = reduce

    @overload
    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        batch: Tensor,
        return_inverse: Literal[True] = True,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]: ...

    @overload
    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        batch: Tensor,
        return_inverse: Literal[False] = False,
    ) -> Tuple[Tensor, Tensor, Tensor]: ...

    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        batch: Tensor,
        return_inverse: bool = False,
    ) -> Tuple[Tensor, ...]:
        start = torch.floor(pos.min(dim=0).values / self.grid_size) * self.grid_size
        cluster = voxel_grid(pos, size=self.grid_size, batch=batch, start=start)
        cluster, perm = consecutive_cluster(cluster)
        pos = scatter(pos, cluster, dim=0, reduce="mean")
        x = scatter(x, cluster, dim=0, reduce=self.reduce)
        batch = batch[perm]

        if return_inverse:
            return x, pos, batch, cluster
        return x, pos, batch

    def extra_repr(self) -> str:
        return f"grid_size={self.grid_size}, reduce={self.reduce!r}"


class KPFCNNEncoderBlock(nn.Module):
    """One encoder stage: an optional grid subsampling followed by `depth` `KPResidualBlock` blocks.

    Neighborhoods are recomputed once at the stage entry, using `pool_radius` for the strided first block
    and `radius` for the remaining ones.

    Args:
        downsample: Grid pooling applied before the first block. Makes that block strided.
    """

    def __init__(
        self,
        *,
        depth: int,
        radius: float,
        pool_radius: Optional[float] = None,
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
        deformable: Union[bool, Sequence[bool]] = False,
        modulated: Union[bool, Sequence[bool]] = False,
        bias: bool = False,
        act: Union[str, Callable, None] = "leaky_relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        downsample: Optional[KPFCNNGridPool] = None,
    ):
        super().__init__()
        self.max_num_neighbors = max_num_neighbors
        self.radius = radius
        self.pool_radius = pool_radius if pool_radius is not None else radius
        self.downsample = downsample
        extra_msg = "Expected encoder `{param_name}` to be of length `depth`."
        kp_radius = ensure_tuple_size(kp_radius, size=depth, extra_msg=extra_msg.format(param_name="kp_radius"))
        kp_sigma = ensure_tuple_size(kp_sigma, size=depth, extra_msg=extra_msg.format(param_name="kp_sigma"))
        deformable = ensure_tuple_size(deformable, size=depth, extra_msg=extra_msg.format(param_name="deformable"))
        modulated = ensure_tuple_size(modulated, size=depth, extra_msg=extra_msg.format(param_name="modulated"))

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
                deformable=deformable[i],
                modulated=modulated[i],
                strided=strided,
                act=act,
                act_kwargs=act_kwargs,
                act_first=act_first,
                norm=norm,
                norm_kwargs=norm_kwargs,
                bias=bias,
            )
            self.blocks.append(block)

    @overload
    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        batch: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]: ...

    @overload
    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        batch: Tensor,
        return_inverse: Literal[True] = True,
    ) -> Tuple[Tensor, Tensor, Tensor, OptTensor]: ...

    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        batch: Tensor,
        return_inverse: bool = False,
    ) -> Any:
        inv = None
        pos_down, batch_down = pos, batch
        if self.downsample is not None:
            _, pos_down, batch_down, inv = self.downsample(x, pos, batch, return_inverse=True)

        # Pre-computed neighbors edge indices, for both strided and non-strided blocks.
        # For strided block (first block after downsampling),
        # compute edge indices between downsampled coords and original coords.
        # For non-strided blocks, then downsampled coords is the same as original coords, and corresponds to the
        # `radius_graph(pos)` output.
        edge_index = radius(
            pos,
            pos_down,
            r=self.pool_radius if self.downsample is not None else self.radius,
            batch_x=batch,
            batch_y=batch_down,
            max_num_neighbors=self.max_num_neighbors,
        )

        for i, block in enumerate(self.blocks):
            if i == 1 and self.downsample is not None:
                edge_index = radius_graph(
                    pos_down,
                    r=self.radius,
                    batch=batch_down,
                    max_num_neighbors=self.max_num_neighbors,
                    flow="target_to_source",
                    loop=True,
                )
                pos = pos_down

            x = block(x, pos_down, pos, edge_index)

        if return_inverse:
            return x, pos_down, batch_down, inv
        return x, pos_down, batch_down


class KPFCNNEncoder(nn.Module):
    r"""KP-FCNN encoder: `KPFCNNEncoderBlock` stages from finest to coarsest, every stage but the first preceded by a
    `KPFCNNGridPool` subsampling.

    Args:
        in_channels: Number of channels entering the first stage.
        depths: Number of residual blocks in each stage.
        grid_sizes: Grid size of each downsampling step, one per stage transition.
        radii: Neighborhood search radius of each stage.
        channels: Output channels of each stage.
        max_num_neighbors: Maximum number of neighbors queried at each stage.
        kernel_size: Number of kernel points of every KPConv.
        kp_sigma: Kernel extent of each stage.
        kp_radius: Kernel point radius of each stage.
        kp_influence: Influence function: `"constant"`, `"linear"`, or `"gaussian"`.
        fixed_position: Which kernel point is pinned: `"none"`, `"center"`, or `"vertical"`.
        aggregation_mode: `"sum"` over all kernel points, or `"closest"` to keep only the nearest one.
        deformable: Whether each stage uses deformable kernels.
        modulated: Whether each stage uses modulated deformable kernels.
        act: Activation function.
        act_kwargs: Optional keyword arguments for the activation factory.
        act_first: Whether the activation comes before the normalization in the MLPs.
        norm: Normalization layer.
        norm_kwargs: Optional keyword arguments for the normalization factory.
        bias: Whether the convolutions and MLPs use a bias.
        spatial_dim: Spatial dimension of the input point cloud.

    Inputs:
        x: Point features of shape $(N, \text{in\_channels})$.
        pos: Point coordinates of shape $(N, D)$.
        batch: Batch indices of shape $(N,)$.

    Outputs:
        Features, coordinates and batch indices at the coarsest stage. With `return_intermediates=True`, also one
        skip per downsampled stage: the features, coordinates, batch indices and pooling inverse entering that stage.
    """

    def __init__(
        self,
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
        deformable: Union[bool, Sequence] = False,
        modulated: Union[bool, Sequence] = False,
        act: Union[str, Callable, None] = "leaky_relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: bool = False,
        spatial_dim: int = 3,
    ):
        super().__init__()
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
        deformable = ensure_tuple_size(deformable, size=n, extra_msg=extra_msg.format(param_name="deformable"))
        modulated = ensure_tuple_size(modulated, size=n, extra_msg=extra_msg.format(param_name="modulated"))

        self.blocks = nn.ModuleList()
        for i in range(n):
            downsample: Optional[KPFCNNGridPool] = None
            if i > 0:
                downsample = KPFCNNGridPool(grid_size=grid_sizes[i - 1], reduce="max")

            block = KPFCNNEncoderBlock(
                downsample=downsample,
                radius=radii[i],
                pool_radius=radii[i - 1] if i > 0 else None,
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
                deformable=deformable[i],
                modulated=modulated[i],
                act=act,
                act_kwargs=act_kwargs,
                act_first=act_first,
                norm=norm,
                norm_kwargs=norm_kwargs,
                bias=bias,
            )
            self.blocks.append(block)
            in_channels = channels[i]

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

    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        batch: Tensor,
        return_intermediates: bool = False,
    ) -> Any:
        intermediates = []
        for i, block in enumerate(self.blocks):
            intermediate = {"x": x, "pos": pos, "batch": batch}
            x, pos, batch, inv = block(x, pos, batch, return_inverse=True)

            if i > 0:
                intermediate["pooling_inverse"] = inv
                intermediates.append(intermediate)

        if return_intermediates:
            return x, pos, batch, intermediates
        return x, pos, batch


class KPFCNNClassification(ClassificationModel):
    """KPConv Network for classification tasks as described in the paper
    :arxiv: [KPConv: Flexible and Efficient Convolution for Point Clouds](https://arxiv.org/abs/1904.08889)
    by Hugues Thomas, Charles R. Qi, Jean-Emmanuel Deschaud, Beatriz Marcotegui, François Goulette, Leonidas J. Guibas.

    KPConv introduces a novel point convolution operator that uses kernel points to define the spatial extent and weights
    of the convolution. The kernel points are arranged in space to define the convolution pattern, with weights determined
    by their spatial correlation with input points. This allows for flexible and efficient convolution on irregular point
    clouds while maintaining permutation invariance and translation invariance. The network uses a hierarchical architecture
    with strided convolutions for spatial pooling and feature aggregation.

    Note:
        The implementation is based on the original paper and the authors' code
        :github: [KPConv-PyTorch](https://github.com/HuguesTHOMAS/KPConv-PyTorch).

    Important:
        This implementation was completely rewritten to be compatible with
        :github: [`torch-geometric`](https://github.com/pyg-team/pytorch_geometric) library.

    Args:
        in_channels: Number of input channels.
        num_classes: Number of output classes.
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
        deformable: Union[bool, Sequence] = False,
        modulated: Union[bool, Sequence] = False,
        act: Union[str, Callable, None] = "leaky_relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: bool = False,
        dropout: float = 0.0,
        global_pool: PoolLike = "max",
    ):
        super().__init__(in_channels, num_classes)
        self.spatial_dim = spatial_dim
        self.stem_channels = stem_channels
        self.stem_type = stem_type
        self.encoder_depths = encoder_depths
        self.encoder_channels = encoder_channels
        self.encoder_num_neighbors = encoder_num_neighbors
        self.grid_sizes = grid_sizes
        self.radii = radii
        self.kernel_size = kernel_size
        self.kp_radius = ensure_tuple_size(kp_radius, size=len(encoder_depths))
        self.kp_sigma = ensure_tuple_size(kp_sigma, size=len(encoder_depths))
        self.kp_influence = kp_influence
        self.fixed_position = fixed_position
        self.aggregation_mode = aggregation_mode
        self.deformable = deformable
        self.modulated = modulated
        self.act = act
        self.act_kwargs = act_kwargs
        self.act_first = act_first
        self.norm = norm
        self.norm_kwargs = norm_kwargs
        self.bias = bias
        self.dropout = dropout

        self.stem = self.configure_stem()
        self.encoder = self.configure_encoder()
        self.global_pool = create_pool(global_pool)
        self.head = self.configure_head()

    def configure_stem(self) -> Optional[nn.Module]:
        """Build the stem lifting the input features to `stem_channels`, or `None` when `stem_channels` is unset."""
        if self.stem_channels is None:
            return None
        if self.stem_type == "kpconv":
            return KPConvBlock(
                spatial_dim=self.spatial_dim,
                in_channels=self.in_channels,
                out_channels=self.stem_channels,
                kernel_size=self.kernel_size,
                kp_radius=self.kp_radius[0],
                kp_sigma=self.kp_sigma[0],
                kp_influence=self.kp_influence,
                fixed_position=self.fixed_position,
                aggregation_mode=self.aggregation_mode,
                act=self.act,
                act_kwargs=self.act_kwargs,
                norm=self.norm,
                norm_kwargs=self.norm_kwargs,
                bias=self.bias,
            )
        act = create_act(self.act, **(self.act_kwargs or {})) or nn.Identity()
        norm = create_norm(self.norm, self.stem_channels, **(self.norm_kwargs or {})) or nn.Identity()
        return nn.Sequential(nn.Linear(self.in_channels, self.stem_channels), norm, act)

    def configure_encoder(self) -> KPFCNNEncoder:
        """Build the `KPFCNNEncoder` backbone."""
        return KPFCNNEncoder(
            in_channels=self.in_channels if self.stem_channels is None else self.stem_channels,
            depths=self.encoder_depths,
            channels=self.encoder_channels,
            grid_sizes=self.grid_sizes,
            radii=self.radii,
            max_num_neighbors=self.encoder_num_neighbors,
            spatial_dim=self.spatial_dim,
            kernel_size=self.kernel_size,
            kp_radius=self.kp_radius,
            kp_sigma=self.kp_sigma,
            kp_influence=self.kp_influence,
            fixed_position=self.fixed_position,
            aggregation_mode=self.aggregation_mode,
            deformable=self.deformable,
            modulated=self.modulated,
            act=self.act,
            act_kwargs=self.act_kwargs,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
            bias=self.bias,
        )

    @property
    def num_features(self) -> int:
        """Feature dimension $C$ of the encoder output."""
        return self.encoder_channels[-1]

    def configure_head(self) -> nn.Module:
        if self.num_classes == 0:
            return nn.Identity()
        return nn.Linear(self.num_features, self.num_classes)

    def reset_classifier(self, num_classes: int, global_pool: Optional[PoolLike] = None, **kwargs: Any) -> None:
        """Resets the classification head with new parameters.

        Note:
            To set an empty classification head, use `num_classes=0`.

        Args:
            num_classes: Number of output classes.
            global_pool: Pooling method to aggregate point features ("max" or "mean"). If `None`, keeps the current
                pooling.
            **kwargs: Additional keyword arguments to pass to the classification head.
        """
        self.num_classes = num_classes
        if global_pool is not None:
            self.global_pool = create_pool(global_pool)
        self.head = self.configure_head()

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
        r"""Forward pass of the encoder, returning pre-pooling features.

        Args:
            x: Point features of shape $(N, C)$. If `None`, `pos` is used as features, which requires
                `in_channels` to match the dimension of `pos`.
            pos: Point coordinates of shape $(N, D)$.
            batch: Batch indices for each point of shape $(N,)$.
            return_intermediates: Whether to also return the per-stage intermediates.

        Returns:
            A tuple `(x, pos, batch)` at the coarsest level, where `x` has shape
                $(N', \text{encoder\_channels}[-1])$ and $N'$ is the number of downsampled points.
                If `return_intermediates=True`, the per-stage intermediates are appended to the tuple.
        """
        if x is None:
            if self.in_channels != pos.size(1):
                raise ValueError(
                    f"Got `x=None` but the model expects in_channels={self.in_channels} features while `pos` has "
                    f"{pos.size(1)} channels; pass `x` of shape (N, {self.in_channels})."
                )
            x = pos

        if self.stem is not None:
            if self.stem_type == "kpconv":
                edge_index = radius_graph(
                    pos,
                    r=self.radii[0],
                    batch=batch,
                    max_num_neighbors=self.encoder_num_neighbors[0],
                    flow="target_to_source",
                    loop=True,
                )
                x = self.stem(x, pos, pos, edge_index)
            else:
                x = self.stem(x)

        return self.encoder(x, pos, batch, return_intermediates=return_intermediates)

    def forward_head(self, x: Tensor, batch: Tensor, pre_logits: bool = False) -> Tensor:
        r"""Forward pass of the classification head from pre-pooling features.

        Args:
            x: Pre-pooling features of shape $(N', \text{encoder\_channels}[-1])$ where $N'$ is the
                number of downsampled points.
            batch: Batch indices for each downsampled point of shape $(N',)$.
            pre_logits: Whether to return pre-logits. Defaults to False.

        Returns:
            Classification logits of shape $(B, \text{num\_classes})$.
        """
        x = self.global_pool(x, batch)
        if self.dropout:
            x = F.dropout(x, p=float(self.dropout), training=self.training)
        return x if pre_logits else self.head(x)

    def forward(self, x: OptTensor, pos: Tensor, batch: Tensor) -> Tensor:
        r"""Forward pass of the classification model.

        Args:
            x: Point features of shape $(N, C)$. If `None`, `pos` is used as features, which requires
                `in_channels` to match the dimension of `pos`.
            pos: Point coordinates of shape $(N, D)$.
            batch: Batch indices for each point of shape $(N,)$.

        Returns:
            Classification logits of shape $(B, \text{num\_classes})$.
        """
        x, _, batch = self.forward_features(x, pos, batch)
        return self.forward_head(x, batch, pre_logits=False)


class KPFCNNSegmentation(SegmentationModel):
    """KPConv Network for segmentation tasks as described in the paper
    :arxiv: [KPConv: Flexible and Efficient Convolution for Point Clouds](https://arxiv.org/abs/1904.08889)
    by Hugues Thomas, Charles R. Qi, Jean-Emmanuel Deschaud, Beatriz Marcotegui, François Goulette, Leonidas J. Guibas.

    KPConv introduces a novel point convolution operator that uses kernel points to define the spatial extent and weights
    of the convolution. The kernel points are arranged in space to define the convolution pattern, with weights determined
    by their spatial correlation with input points. This allows for flexible and efficient convolution on irregular point
    clouds while maintaining permutation invariance and translation invariance. The network uses a hierarchical architecture
    with strided convolutions for spatial pooling and feature aggregation.

    Note:
        The implementation is based on the original paper and the authors' code
        :github: [KPConv-PyTorch](https://github.com/HuguesTHOMAS/KPConv-PyTorch).

    Important:
        This implementation was completely rewritten to be compatible with
        :github: [`torch-geometric`](https://github.com/pyg-team/pytorch_geometric) library.

    Args:
        in_channels: Number of input channels.
        num_classes: Number of output classes.
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
        in_channels: int,
        num_classes: int,
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
        deformable: Union[bool, Sequence] = False,
        modulated: Union[bool, Sequence] = False,
        act: Union[str, Callable, None] = "leaky_relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: bool = False,
        dropout: float = 0.0,
        head_channels: Optional[Sequence[int]] = None,
    ):
        super().__init__(in_channels, num_classes)
        self.spatial_dim = spatial_dim
        self.stem_channels = stem_channels
        self.stem_type = stem_type
        self.encoder_depths = encoder_depths
        self.encoder_channels = encoder_channels
        self.encoder_num_neighbors = encoder_num_neighbors
        self.fp_channels = fp_channels
        self.head_channels = list(head_channels) if head_channels else []
        self.grid_sizes = grid_sizes
        self.radii = radii
        self.kernel_size = kernel_size
        self.kp_radius = ensure_tuple_size(kp_radius, size=len(encoder_depths))
        self.kp_sigma = ensure_tuple_size(kp_sigma, size=len(encoder_depths))
        self.kp_influence = kp_influence
        self.fixed_position = fixed_position
        self.aggregation_mode = aggregation_mode
        self.deformable = deformable
        self.modulated = modulated
        self.act = act
        self.act_kwargs = act_kwargs
        self.act_first = act_first
        self.norm = norm
        self.norm_kwargs = norm_kwargs
        self.bias = bias
        self.dropout = dropout

        self.stem = self.configure_stem()
        self.encoder = self.configure_encoder()
        self.decoder = self.configure_decoder()
        self.head = self.configure_head()

    def configure_stem(self) -> Optional[nn.Module]:
        """Build the stem lifting the input features to `stem_channels`, or `None` when `stem_channels` is unset."""
        if self.stem_channels is None:
            return None
        if self.stem_type == "kpconv":
            return KPConvBlock(
                spatial_dim=self.spatial_dim,
                in_channels=self.in_channels,
                out_channels=self.stem_channels,
                kernel_size=self.kernel_size,
                kp_radius=self.kp_radius[0],
                kp_sigma=self.kp_sigma[0],
                kp_influence=self.kp_influence,
                fixed_position=self.fixed_position,
                aggregation_mode=self.aggregation_mode,
                act=self.act,
                act_kwargs=self.act_kwargs,
                norm=self.norm,
                norm_kwargs=self.norm_kwargs,
                bias=self.bias,
            )
        act = create_act(self.act, **(self.act_kwargs or {})) or nn.Identity()
        norm = create_norm(self.norm, self.stem_channels, **(self.norm_kwargs or {})) or nn.Identity()
        return nn.Sequential(nn.Linear(self.in_channels, self.stem_channels), norm, act)

    def configure_encoder(self) -> KPFCNNEncoder:
        """Build the `KPFCNNEncoder` backbone."""
        return KPFCNNEncoder(
            in_channels=self.in_channels if self.stem_channels is None else self.stem_channels,
            depths=self.encoder_depths,
            channels=self.encoder_channels,
            grid_sizes=self.grid_sizes,
            radii=self.radii,
            max_num_neighbors=self.encoder_num_neighbors,
            spatial_dim=self.spatial_dim,
            kernel_size=self.kernel_size,
            kp_radius=self.kp_radius,
            kp_sigma=self.kp_sigma,
            kp_influence=self.kp_influence,
            fixed_position=self.fixed_position,
            aggregation_mode=self.aggregation_mode,
            deformable=self.deformable,
            modulated=self.modulated,
            act=self.act,
            act_kwargs=self.act_kwargs,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
            bias=self.bias,
        )

    def configure_decoder(self) -> PointNet2Decoder:
        """Build the `PointNet2Decoder` upsampling the coarsest features back through the encoder skips."""
        stem_channels = self.in_channels if self.stem_channels is None else self.stem_channels
        all_skip_channels = [stem_channels, *self.encoder_channels[:-1]]
        skip_channels = all_skip_channels[-len(self.fp_channels) :][::-1]
        return PointNet2Decoder(
            in_channels=self.encoder_channels[-1],
            skip_channels=skip_channels,
            fp_channels=self.fp_channels,
            bias=self.bias,
            act=self.act,
            act_kwargs=self.act_kwargs,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
            k=1,
        )

    @property
    def num_features(self) -> int:
        """Feature dimension $C$ of the decoder output."""
        return self.fp_channels[-1][-1]

    def configure_head(self) -> nn.Module:
        if self.num_classes == 0:
            return nn.Identity()
        if not self.head_channels:
            return nn.Linear(self.num_features, self.num_classes)
        head_act = create_act(self.act, **(self.act_kwargs or {})) or nn.Identity()
        layers: List[nn.Module] = []
        ch_in = self.num_features
        for ch in self.head_channels:
            layers.append(nn.Sequential(nn.Linear(ch_in, ch, bias=True), head_act))
            ch_in = ch
        layers.append(nn.Linear(ch_in, self.num_classes))
        return nn.Sequential(*layers)

    def reset_classifier(self, num_classes: int, **kwargs: Any) -> None:
        self.num_classes = num_classes
        self.head = self.configure_head()

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
        if x is None:
            if self.in_channels != pos.size(1):
                raise ValueError(
                    f"Got `x=None` but the model expects in_channels={self.in_channels} features while `pos` has "
                    f"{pos.size(1)} channels; pass `x` of shape (N, {self.in_channels})."
                )
            x = pos

        if self.stem is not None:
            if self.stem_type == "kpconv":
                edge_index = radius_graph(
                    pos,
                    r=self.radii[0],
                    batch=batch,
                    max_num_neighbors=self.encoder_num_neighbors[0],
                    flow="target_to_source",
                    loop=True,
                )
                x = self.stem(x, pos, pos, edge_index)
            else:
                x = self.stem(x)

        intermediates = [{"x": x, "pos": pos, "batch": batch}]
        x, pos, batch, encoder_intermediates = self.encoder(x, pos, batch, return_intermediates=True)
        intermediates.extend(encoder_intermediates)

        if return_intermediates:
            return x, pos, batch, intermediates
        return x, pos, batch

    def forward_decoder(
        self,
        x: Tensor,
        pos: Tensor,
        batch: Tensor,
        intermediates: List[Dict[str, Tensor]],
    ) -> Tensor:
        x, _, _ = self.decoder(x, pos, batch, intermediates)
        return x

    def forward_head(self, x: Tensor, pre_logits: bool = False) -> Tensor:
        if self.dropout:
            x = F.dropout(x, p=float(self.dropout), training=self.training)
        return x if pre_logits else self.head(x)

    def forward(self, x: OptTensor, pos: Tensor, batch: Tensor) -> Tensor:
        x, pos, batch, intermediates = self.forward_features(x, pos, batch, return_intermediates=True)
        x = self.forward_decoder(x, pos, batch, intermediates)
        return self.forward_head(x)


@register_model(
    "kpfcnn.modelnet40",
    task="classification",
    hparams=dict(
        in_channels=6,
        num_classes=40,
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
        norm="batch_norm",
        norm_kwargs={"momentum": 0.05},
    ),
)
def kpfcnn_modelnet40_clf(**hparams: Any) -> KPFCNNClassification:
    return KPFCNNClassification(**hparams)


_BASE_S3DIS_TRANSFORMS = T.Compose(
    [
        # Original implementation of KPConv uses custom C++ code for grid subsampling,
        # which behaves differently from the PyG implementation, but is close enough.
        # The main difference is that labels are reduced using the most frequent value per voxel.
        # NOTE: tensors are automatically converted to float before reduction (if other than "first")
        T.CopyItems(
            keys=[DataKeys.POS, DataKeys.SEGMENT],
            names=[DataKeys.ORIGIN_POS, DataKeys.ORIGIN_SEGMENT],
            allow_missing_keys=True,
        ),
        T.Voxelize(
            pos_key=DataKeys.POS,
            pos_reduce="mean",
            keys=[DataKeys.COLOR, DataKeys.SEGMENT],
            reduce=["mean", "first"],
            size=0.03,
            method="pyg",
            dst_inverse_key=DataKeys.INVERSE,
        ),
        T.Scale(keys=DataKeys.COLOR, scale=1.0 / 255),
        T.AxisMinOffset(keys=DataKeys.POS, dst_keys="height", axis=2),
        T.OnesLike(keys="height", dst_keys="ones"),
        T.Cat(keys=["ones", DataKeys.COLOR, "height"], dst_key=DataKeys.X),
        T.RenameItems(keys=[DataKeys.SEGMENT], names=[DataKeys.LABEL]),
        T.KeepItems(
            keys=[
                DataKeys.X,
                DataKeys.POS,
                DataKeys.LABEL,
                DataKeys.ORIGIN_POS,
                DataKeys.ORIGIN_SEGMENT,
                DataKeys.INVERSE,
            ]
        ),
    ]
)


@register_model(
    "kpfcnn-base-sm.s3dis.hugues-thomas",
    task="segmentation",
    transform=_BASE_S3DIS_TRANSFORMS,
    weights=WeightsDict(
        url="hf://torch-pointcloud/kpfcnn-base-sm.s3dis.hugues-thomas/resolve/main/model.safetensors",
        dataset="s3dis",
        metrics={"mIoU": 63.92},
        classes=S3DIS_CLASSES,
        author="hugues-thomas",
        license="MIT",
    ),
    hparams=dict(
        in_channels=5,
        num_classes=13,
        stem_channels=64,
        stem_type="kpconv",
        encoder_depths=[1, 2, 2, 3, 3],
        encoder_channels=[128, 256, 512, 1024, 2048],
        encoder_num_neighbors=[128, 128, 128, 128, 128],
        fp_channels=[[1024], [512], [256], [128]],
        head_channels=[128],
        grid_sizes=[0.06, 0.12, 0.24, 0.48],
        radii=[0.075, 0.15, 0.3, 0.6, 1.2],
        kernel_size=15,
        kp_radius=[0.075, 0.15, 0.3, 0.6, 1.2],
        kp_sigma=[0.036, 0.072, 0.144, 0.288, 0.576],
        kp_influence="linear",
        fixed_position="center",
        aggregation_mode="sum",
        deformable=False,
        modulated=False,
        bias=False,
        act="leaky_relu",
        act_kwargs={"negative_slope": 0.1},
        norm="batch_norm",
        norm_kwargs={"momentum": 0.02},
    ),
)
def kpfcnn_base_sm_seg(**hparams: Any) -> KPFCNNSegmentation:
    return KPFCNNSegmentation(**hparams)


@register_model(
    "kpfcnn-base.s3dis.hugues-thomas",
    task="segmentation",
    transform=_BASE_S3DIS_TRANSFORMS,
    weights=WeightsDict(
        url="hf://torch-pointcloud/kpfcnn-base.s3dis.hugues-thomas/resolve/main/model.safetensors",
        dataset="s3dis",
        metrics={"mIoU": 65.64},
        classes=S3DIS_CLASSES,
        author="hugues-thomas",
        license="MIT",
    ),
    hparams=dict(
        in_channels=5,
        num_classes=13,
        stem_channels=64,
        stem_type="kpconv",
        encoder_depths=[1, 3, 3, 3, 3],
        encoder_channels=[128, 256, 512, 1024, 2048],
        encoder_num_neighbors=[128, 128, 128, 128, 128],
        fp_channels=[[1024], [512], [256], [128]],
        head_channels=[128],
        grid_sizes=[0.06, 0.12, 0.24, 0.48],
        radii=[0.075, 0.15, 0.3, 0.6, 1.2],
        kernel_size=15,
        kp_radius=[0.075, 0.15, 0.3, 0.6, 1.2],
        kp_sigma=[0.036, 0.072, 0.144, 0.288, 0.576],
        kp_influence="linear",
        fixed_position="center",
        aggregation_mode="sum",
        deformable=False,
        modulated=False,
        bias=False,
        act="leaky_relu",
        act_kwargs={"negative_slope": 0.1},
        norm="batch_norm",
        norm_kwargs={"momentum": 0.02},
    ),
)
def kpfcnn_base_seg(**hparams: Any) -> KPFCNNSegmentation:
    return KPFCNNSegmentation(**hparams)


@register_model(
    "kpfcnn-base-deform.s3dis.hugues-thomas",
    task="segmentation",
    transform=_BASE_S3DIS_TRANSFORMS,
    weights=WeightsDict(
        url="hf://torch-pointcloud/kpfcnn-base-deform.s3dis.hugues-thomas/resolve/main/model.safetensors",
        dataset="s3dis",
        metrics={"mIoU": 65.66},
        classes=S3DIS_CLASSES,
        author="hugues-thomas",
        license="MIT",
    ),
    hparams=dict(
        in_channels=5,
        num_classes=13,
        stem_channels=64,
        stem_type="kpconv",
        encoder_depths=[1, 3, 3, 3, 3],
        encoder_channels=[128, 256, 512, 1024, 2048],
        encoder_num_neighbors=[128, 128, 1024, 1024, 1024],
        fp_channels=[[1024], [512], [256], [128]],
        head_channels=[128],
        grid_sizes=[0.06, 0.12, 0.24, 0.48],
        radii=[0.075, 0.15, 0.72, 1.44, 2.88],
        kernel_size=15,
        kp_radius=[0.075, 0.15, 0.3, 0.6, 1.2],
        kp_sigma=[0.036, 0.072, 0.144, 0.288, 0.576],
        kp_influence="linear",
        fixed_position="center",
        aggregation_mode="sum",
        deformable=[False, False, [False, True, True], True, True],
        modulated=False,
        bias=False,
        act="leaky_relu",
        act_kwargs={"negative_slope": 0.1},
        norm="batch_norm",
        norm_kwargs={"momentum": 0.02},
    ),
)
def kpfcnn_base_deform_seg(**hparams: Any) -> KPFCNNSegmentation:
    return KPFCNNSegmentation(**hparams)


@register_model(
    "kpfcnn-base-sm-deform.s3dis.hugues-thomas",
    task="segmentation",
    transform=_BASE_S3DIS_TRANSFORMS,
    weights=WeightsDict(
        url="hf://torch-pointcloud/kpfcnn-base-sm-deform.s3dis.hugues-thomas/resolve/main/model.safetensors",
        dataset="s3dis",
        metrics={"mIoU": 64.47},
        classes=S3DIS_CLASSES,
        author="hugues-thomas",
        license="MIT",
    ),
    hparams=dict(
        in_channels=5,
        num_classes=13,
        stem_channels=64,
        stem_type="kpconv",
        encoder_depths=[1, 3, 3, 3, 3],
        encoder_channels=[128, 256, 512, 1024, 2048],
        encoder_num_neighbors=[128, 128, 128, 1024, 1024],
        fp_channels=[[1024], [512], [256], [128]],
        head_channels=[128],
        grid_sizes=[0.06, 0.12, 0.24, 0.48],
        radii=[0.075, 0.15, 0.3, 1.2, 2.4],
        kernel_size=15,
        kp_radius=[0.075, 0.15, 0.3, 0.6, 1.2],
        kp_sigma=[0.036, 0.072, 0.144, 0.288, 0.576],
        kp_influence="linear",
        fixed_position="center",
        aggregation_mode="sum",
        deformable=[False, False, False, [False, True, True], True],
        modulated=False,
        bias=False,
        act="leaky_relu",
        act_kwargs={"negative_slope": 0.1},
        norm="batch_norm",
        norm_kwargs={"momentum": 0.02},
    ),
)
def kpfcnn_base_sm_deform_seg(**hparams: Any) -> KPFCNNSegmentation:
    return KPFCNNSegmentation(**hparams)
