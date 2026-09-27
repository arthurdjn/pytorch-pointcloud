"""Kernel point convolution, its kernel point placement, and the normalized and residual blocks built on it."""

import math
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import MLP, MessagePassing
from torch_geometric.typing import Adj, PairTensor
from torch_geometric.utils import scatter

from torch_pointcloud.config import CACHE_DIR
from torch_pointcloud.layers.act import create_act
from torch_pointcloud.layers.norms import create_norm
from torch_pointcloud.ops.cluster import dense_neighbors, gather_neighbors
from torch_pointcloud.ops.geometry import rodrigues_rotation_matrix, spherical_points_gradient, spherical_points_lloyd
from torch_pointcloud.utils.types import OptTensor


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
        alpha = torch.rand(1).item() * 2 * math.pi
        R = rodrigues_rotation_matrix(u, theta=alpha)

    # Add a small noise, scale and rotate
    kernel_points += torch.normal(mean=0, std=0.01, size=kernel_points.shape)
    kernel_points *= radius
    kernel_points = torch.matmul(kernel_points, R)

    return kernel_points


class KPConv(MessagePassing):
    r"""Kernel point convolution over a neighborhood graph.

    Each of the $K$ kernel points carries its own weight matrix, and a neighbor contributes to a kernel point
    with a weight given by their distance through the influence function. When `deformable` is set, a nested
    `KPConv` predicts a per-point offset for every kernel point.

    `edge_index` lists `[source, target]` pairs: messages flow from the support (source) points to the query
    (target) points, and a bipartite graph is given as a `(pos_source, pos_target)` pair of positions.

    The message is the $(E, K)$ influence of every kernel point on every edge; the aggregation pools the
    neighbor features per kernel point and only then applies that kernel point's weight matrix, so the $K$
    matrix products run over the $N_t$ target points rather than over the $E$ edges: the neighborhoods are laid
    out as a padded $(N_t, k)$ table and the pooling is one batched matrix product over it.

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

    Shape:
        Input: $(N_s, C_\text{in})$ features, $(N, D)$ positions or a $((N_s, D), (N_t, D))$ pair, $(2, E)$ edge index
        Output: $(N_t, C_\text{out})$ features
    """

    kernel: Tensor

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
        super().__init__(aggr=None)
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
                nn.init.uniform_(self.bias, -bound, bound)

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

    def _compute_offsets(self, x: Tensor, pos: PairTensor, edge_index: Adj) -> Tuple[OptTensor, OptTensor]:
        if self.offset_conv is None:
            return None, None

        offset_x = self.offset_conv(x, pos, edge_index)
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
        x: Union[Tensor, PairTensor],
        pos: Union[Tensor, PairTensor],
        edge_index: Adj,
    ) -> Tensor:
        x_source = x[0] if isinstance(x, tuple) else x
        if not isinstance(pos, tuple):
            pos = (pos, pos)

        offsets, modulations = self._compute_offsets(x_source, pos, edge_index)
        if offsets is not None:
            # `MessagePassing` lifts node tensors along `node_dim=-2`, so the per-node offsets stay two-dimensional.
            offsets = offsets.flatten(1)
        # propagate_type: (x_source: Tensor, pos: PairTensor, offsets: OptTensor, modulations: OptTensor)
        out = self.propagate(edge_index, x_source=x_source, pos=pos, offsets=offsets, modulations=modulations)
        if self.bias is not None:
            out = out + self.bias
        return out

    def message(self, pos_i: Tensor, pos_j: Tensor, offsets_i: OptTensor, modulations_i: OptTensor) -> Tensor:
        pos_rel = pos_j - pos_i  # (E, spatial_dim)

        if offsets_i is not None:
            offsets_i = offsets_i.view(-1, self.kernel_size, self.spatial_dim)
            kernel_points = self.kernel.unsqueeze(0) + offsets_i  # (E, K, spatial_dim)
            if self.track_running_stats:
                self.running_deformed_kernel = kernel_points
        else:
            kernel_points = self.kernel.unsqueeze(0).expand(pos_rel.size(0), -1, -1)

        sq_distances = torch.sum((pos_rel.unsqueeze(1) - kernel_points) ** 2, dim=-1)  # (E, K)
        if self.track_running_stats and self.deformable:
            self.running_min_d2, _ = torch.min(sq_distances, dim=1)

        weights = self._compute_weights(sq_distances)
        if self.aggregation_mode == "closest":
            neighbors_1nn = torch.argmin(sq_distances, dim=1)  # (E,)
            one_hot = torch.zeros_like(weights).scatter_(1, neighbors_1nn.unsqueeze(1), 1)
            weights = weights * one_hot
        if modulations_i is not None:
            weights = weights * modulations_i
        return weights

    def aggregate(
        self,
        inputs: Tensor,
        index: Tensor,
        edge_index_j: Tensor,
        x_source: Tensor,
        dim_size: Optional[int] = None,
    ) -> Tensor:
        num_targets = dim_size if dim_size is not None else int(index.max()) + 1
        table, slot = dense_neighbors(torch.stack([edge_index_j, index]), x_source.size(0), num_targets)
        x_dense = gather_neighbors(x_source, table)  # (N_t, k, in_channels)
        # Filled through a permuted view so the (N_t, K, k) operand is contiguous: `bmm` on the transposed view
        # of a (N_t, k, K) table runs 3-4x slower at these shapes.
        weights = inputs.new_zeros(num_targets, self.kernel_size, table.size(1))
        weights.permute(0, 2, 1)[index, slot] = inputs
        pooled = torch.bmm(weights, x_dense)  # (N_t, K, in_channels)
        weight = self.weight.reshape(self.kernel_size * self.in_channels, self.out_channels)
        return torch.matmul(pooled.reshape(num_targets, -1), weight.to(pooled.dtype))

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

    def forward(self, x: Tensor, pos: Union[Tensor, PairTensor], edge_index: Tensor) -> Tensor:
        x = self.conv(x, pos, edge_index)
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

    def forward(self, x: Tensor, pos: Union[Tensor, PairTensor], edge_index: Tensor) -> Tensor:
        shortcut = x
        if self.strided:
            source, target = edge_index
            num_targets = pos[1].size(0) if isinstance(pos, tuple) else pos.size(0)
            shortcut = scatter(x[source], target, dim=0, dim_size=num_targets, reduce="max").clamp(min=0)

        shortcut = self.shortcut(shortcut)
        x = self.unary1(x)
        x = self.conv(x, pos, edge_index)
        x = self.unary2(x)

        x = x + shortcut
        if self.act is not None:
            x = self.act(x)

        return x
