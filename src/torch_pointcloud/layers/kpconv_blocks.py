"""Kernel point convolution, its kernel point placement, and the normalized and residual blocks built on it."""

import inspect
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
from torch_pointcloud.utils.types import OptTensor, PairOptTensor


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
        normalize_by_positive_neighbors: Whether to divide the output of every query point by the number of its
            neighbors whose features sum to a positive value (at least one).
        track_running_stats: Whether to keep the deformable statistics of the last forward pass (`running_min_d2`,
            `running_deformed_kernel`, `running_offset_features`) for `losses.kpconv_deform_regularizer`.

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
        normalize_by_positive_neighbors: bool = False,
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
        self.normalize_by_positive_neighbors = normalize_by_positive_neighbors

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
            if self.track_running_stats:
                self.running_deformed_kernel = self.kernel + offsets  # (N_t, K, spatial_dim)
            # `MessagePassing` lifts node tensors along `node_dim=-2`, so the per-node offsets stay two-dimensional.
            offsets = offsets.flatten(1)
        # propagate_type: (x_source: Tensor, pos: PairTensor, offsets: OptTensor, modulations: OptTensor)
        out = self.propagate(edge_index, x_source=x_source, pos=pos, offsets=offsets, modulations=modulations)
        if self.bias is not None:
            out = out + self.bias
        return out

    def message(
        self,
        pos_i: Tensor,
        pos_j: Tensor,
        offsets_i: OptTensor,
        modulations_i: OptTensor,
        index: Tensor,
        size_i: Optional[int],
    ) -> Tensor:
        pos_rel = pos_j - pos_i  # (E, spatial_dim)

        if offsets_i is not None:
            offsets_i = offsets_i.view(-1, self.kernel_size, self.spatial_dim)
            kernel_points = self.kernel.unsqueeze(0) + offsets_i  # (E, K, spatial_dim)
        else:
            kernel_points = self.kernel.unsqueeze(0).expand(pos_rel.size(0), -1, -1)

        sq_distances = torch.sum((pos_rel.unsqueeze(1) - kernel_points) ** 2, dim=-1)  # (E, K)
        if self.track_running_stats and self.deformable:
            # Squared distance from every deformed kernel point to its closest neighbor, per target: (N_t, K).
            self.running_min_d2 = scatter(sq_distances, index, dim=0, dim_size=size_i, reduce="min")

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
        out = torch.matmul(pooled.reshape(num_targets, -1), weight.to(pooled.dtype))
        if self.normalize_by_positive_neighbors:
            num_neighbors = (x_dense.sum(dim=-1) > 0).sum(dim=1).clamp(min=1)
            out = out / num_neighbors.unsqueeze(1).to(out.dtype)
        return out

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
            f"modulated={self.modulated}, "
            f"normalize_by_positive_neighbors={self.normalize_by_positive_neighbors}"
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.extra_repr()})"


class KPConvBlock(nn.Module):
    r"""Kernel point convolution followed by normalization and activation.

    Given a batch index, a normalization whose forward takes one (`GroupNorm`, the PyG instance and layer
    normalizations) is computed per batch element, as in PyG's `MLP`.

    Shape:
        - Input: $(N_s, C_\text{in})$ features, $(N, D)$ positions or a $((N_s, D), (N_t, D))$ pair, a $(2, E)$
          `[source, target]` edge index, an optional $(N_t,)$ batch index and the optional number of batch elements.
        - Output: $(N_t, C_\text{out})$.
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
        act: Union[str, Callable, None] = "leaky_relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: bool = False,
        normalize_by_positive_neighbors: bool = False,
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
            normalize_by_positive_neighbors=normalize_by_positive_neighbors,
        )
        self.norm = create_norm(norm, out_channels, **(norm_kwargs or {})) or nn.Identity()
        self.norm_takes_batch = "batch" in inspect.signature(self.norm.forward).parameters
        self.act = create_act(act, **(act_kwargs or {})) or nn.Identity()

    def forward(
        self,
        x: Tensor,
        pos: Union[Tensor, PairTensor],
        edge_index: Tensor,
        batch: Optional[Tensor] = None,
        batch_size: Optional[int] = None,
    ) -> Tensor:
        x = self.conv(x, pos, edge_index)
        x = self.norm(x, batch, batch_size) if self.norm_takes_batch else self.norm(x)
        return self.act(x)


class KPResidualBlock(nn.Module):
    r"""Bottleneck residual block: a channel-reducing MLP, a `KPConvBlock`, and a channel-restoring MLP.

    The bottleneck width is a quarter of `out_channels`, at least 8. Set `strided=True` when the block maps a support
    cloud onto a coarser query cloud: the skip connection is then the max over each query's neighbors and zero or,
    with `max_num_neighbors`, the max over the neighbor table padded with zeros to that many slots, so zero only
    enters the max of a query with fewer neighbors. Given a batch index, the normalizations are computed per batch
    element, as in `KPConvBlock`.

    Args:
        spatial_dim: Spatial dimension of the input point cloud.
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Number of kernel points.
        kp_radius: Radius of the sphere the kernel points are placed on.
        kp_sigma: Kernel extent, the distance over which a kernel point still influences a neighbor.
        kp_influence: Influence function: `"constant"`, `"linear"`, or `"gaussian"`.
        fixed_position: Which kernel point is pinned: `"none"`, `"center"`, or `"vertical"`.
        aggregation_mode: `"sum"` over all kernel points, or `"closest"` to keep only the nearest one.
        deformable: Whether to predict per-point kernel offsets.
        modulated: Whether the deformable branch also predicts a per-kernel-point modulation.
        strided: Whether the block pools a support cloud onto a coarser query cloud.
        act: Activation of the bottleneck layers and of the output.
        act_kwargs: Extra arguments of the activation.
        act_first: Whether the MLPs apply the activation before the normalization.
        norm: Normalization of every layer, resolved by `create_norm`.
        norm_kwargs: Extra arguments of the normalization.
        bias: Whether the linear layers and the convolution add a bias.
        normalize_by_positive_neighbors: Whether the convolution divides its output by the number of neighbors with
            positive features.
        max_num_neighbors: Width of the zero-padded neighbor table of a strided block, the cap of the search that
            produced the edges.

    Shape:
        - Input: $(N_s, C_\text{in})$ features, $(N, D)$ positions or a $((N_s, D), (N_t, D))$ pair, a $(2, E)$
          `[source, target]` edge index, an optional batch index of the normalizations, $(N,)$ or a
          $((N_s,), (N_t,))$ pair for a strided block, and the optional number of batch elements.
        - Output: $(N_t, C_\text{out})$.
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
        normalize_by_positive_neighbors: bool = False,
        max_num_neighbors: Optional[int] = None,
    ):
        super().__init__()
        mid_channels = max(out_channels // 4, 8)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.strided = strided
        self.max_num_neighbors = max_num_neighbors

        # PyG's `MLP` resolves a norm name with its own resolver, which knows no "group_norm", so the norms are built
        # here; a single-layer `MLP` holds the one instance and passes it the batch index when its forward takes one.
        def unary(in_channels: int, out_channels: int, act: Union[str, Callable, None]) -> MLP:
            return MLP(
                [in_channels, out_channels],
                act=act,
                act_kwargs=act_kwargs,
                act_first=act_first,
                norm=create_norm(norm, out_channels, **(norm_kwargs or {})),
                bias=bias,
                plain_last=False,
            )

        self.unary1 = unary(in_channels, mid_channels, act)
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
            normalize_by_positive_neighbors=normalize_by_positive_neighbors,
        )
        self.unary2 = unary(mid_channels, out_channels, None)
        self.shortcut: Optional[MLP] = unary(in_channels, out_channels, None) if in_channels != out_channels else None
        self.act = create_act(act, **(act_kwargs or {})) or nn.Identity()

    def forward(
        self,
        x: Tensor,
        pos: Union[Tensor, PairTensor],
        edge_index: Tensor,
        batch: Union[Tensor, PairOptTensor, None] = None,
        batch_size: Optional[int] = None,
    ) -> Tensor:
        batch_source, batch_target = batch if isinstance(batch, tuple) else (batch, batch)
        shortcut = x
        if self.strided:
            num_targets = pos[1].size(0) if isinstance(pos, tuple) else pos.size(0)
            if self.max_num_neighbors is None:
                source, target = edge_index
                shortcut = scatter(x[source], target, dim=0, dim_size=num_targets, reduce="max").clamp(min=0)
            else:
                table, _ = dense_neighbors(edge_index, x.size(0), num_targets, max_num_neighbors=self.max_num_neighbors)
                shortcut = gather_neighbors(x, table).max(dim=1).values
        if self.shortcut is not None:
            shortcut = self.shortcut(shortcut, batch_target, batch_size)

        x = self.unary1(x, batch_source, batch_size)
        x = self.conv(x, pos, edge_index, batch_target, batch_size)
        x = self.unary2(x, batch_target, batch_size)
        return self.act(x + shortcut)
