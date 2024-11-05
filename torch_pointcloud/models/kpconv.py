import math
import random
import warnings
from pathlib import Path
from typing import Any, Literal, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.init import kaiming_uniform_
from torch.nn.parameter import Parameter

from torch_pointcloud.utils import CACHE_DIR
from torch_pointcloud.utils.geometry import rodrigues_rotation_matrix, spherical_points_gradient, spherical_points_lloyd


def create_kernel_points(
    radius: float,
    num_points: int,
    fixed_position: Literal["none", "center", "vertical"] = "center",
    method: Literal["lloyd", "gradient"] = "lloyd",
) -> torch.Tensor:
    if num_points > 30:
        warnings.warn("Too many points, consider using Lloyds algorithm (algorithm='lloyd')")

    # Check if kernel is already computed
    kernel_path = Path(CACHE_DIR, "kernels", f"k_{num_points}_{fixed_position}_{method}.pt")
    if kernel_path.exists():
        kernel_points = torch.load(kernel_path)
    else:
        if method == "lloyd":
            kernel_points = spherical_points_lloyd(radius=1.0, num_points=num_points, fixed_position=fixed_position)
        else:
            kernel_points, _ = spherical_points_gradient(
                radius=1.0, num_points=num_points, fixed_position=fixed_position
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

        # Create the rotation matrix with this vector and angle
        R = rodrigues_rotation_matrix(u, theta_degrees=math.degrees(alpha))

    # Add a small noise, scale and rotate
    kernel_points += torch.normal(mean=0, std=0.01, size=kernel_points.shape)
    kernel_points *= radius
    kernel_points = torch.matmul(kernel_points, R)

    return kernel_points


def gather(x: Tensor, idx: Tensor) -> Tensor:
    # Expand x to match idx along its additional dimensions
    for i, ni in enumerate(idx.size()[1:]):
        x = x.unsqueeze(i + 1)
        new_s = list(x.size())
        new_s[i + 1] = ni
        x = x.expand(new_s)

    # Expand idx to match x in dimensions after the first n dimensions
    n = len(idx.size())
    for i, di in enumerate(x.size()[n:]):
        idx = idx.unsqueeze(i + n)
        new_s = list(idx.size())
        new_s[i + n] = di
        idx = idx.expand(new_s)

    # Perform the gather operation along dimension 0
    return x.gather(0, idx)


class KPConv(nn.Module):
    def __init__(
        self,
        kernel_size: int,
        p_dim: int,
        in_channels: int,
        out_channels: int,
        KP_extent: float,
        radius: float,
        fixed_kernel_points: Literal["none", "center", "vertical"] = "center",
        KP_influence: str = "linear",
        aggregation_mode: str = "sum",
        deformable: bool = False,
        modulated: bool = False,
    ) -> None:
        super().__init__()

        self.kernel_size = kernel_size
        self.p_dim = p_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.KP_extent = KP_extent
        self.radius = radius
        self.fixed_kernel_points = fixed_kernel_points
        self.KP_influence = KP_influence
        self.aggregation_mode = aggregation_mode
        self.modulated = modulated

        # Initialize parameters
        self.weights = Parameter(torch.zeros(kernel_size, in_channels, out_channels), requires_grad=True)
        self.kernel = self._init_kernel()
        self.offset_conv, self.offset_bias = self._init_offsets() if deformable else (None, None)

        # Running variable containing deformed KP distance to input points (used in regularization loss)
        self.min_d2: Union[None, torch.Tensor] = None
        self.deformed_kernel: Union[None, torch.Tensor] = None
        self.offset_features: Union[None, torch.Tensor] = None

        self.init_parameters()

    @property
    def deformable(self) -> bool:
        return self.offset_conv is not None and self.offset_bias is not None

    def init_parameters(self) -> None:
        kaiming_uniform_(self.weights, a=math.sqrt(5))
        if self.offset_bias is not None:
            nn.init.zeros_(self.offset_bias)

    def _init_offsets(self) -> Tuple[nn.Module, Tensor]:
        offset_dim = (self.p_dim + 1) * self.kernel_size if self.modulated else self.p_dim * self.kernel_size
        offset_conv = KPConv(
            kernel_size=self.kernel_size,
            p_dim=self.p_dim,
            in_channels=self.in_channels,
            out_channels=offset_dim,
            KP_extent=self.KP_extent,
            radius=self.radius,
            fixed_kernel_points=self.fixed_kernel_points,
            KP_influence=self.KP_influence,
            aggregation_mode=self.aggregation_mode,
        )
        offset_bias = Parameter(torch.zeros(offset_dim), requires_grad=True)
        return offset_conv, offset_bias

    def _init_kernel(self) -> Tensor:
        kernel_points = create_kernel_points(self.radius, self.kernel_size, fixed_position=self.fixed_kernel_points)
        return Parameter(kernel_points, requires_grad=False)

    def _compute_weights(self, sq_distances: Tensor) -> Tensor:
        if self.KP_influence == "constant":
            return torch.ones_like(sq_distances).transpose(1, 2)
        elif self.KP_influence == "linear":
            return torch.clamp(1 - torch.sqrt(sq_distances) / self.KP_extent, min=0.0).transpose(1, 2)
        elif self.KP_influence == "gaussian":
            sigma = self.KP_extent * 0.3
            return torch.exp(-sq_distances / (2 * sigma**2 + 1e-6)).transpose(1, 2)
        else:
            raise ValueError(f"Unknown influence type: {self.KP_influence}")

    # TODO: make support points optional
    def forward(self, q_pts: Tensor, s_pts: Tensor, neighbor_idxs: Tensor, x: Tensor) -> Tensor:
        # 1. Compute the offsets if the layer is deformable
        if self.deformable and self.offset_conv is not None:
            offset_features = self.offset_conv(q_pts, s_pts, neighbor_idxs, x) + self.offset_bias
            self.offset_features = offset_features

            unscaled_offsets = offset_features[:, : self.p_dim * self.kernel_size]
            unscaled_offsets = unscaled_offsets.view(-1, self.kernel_size, self.p_dim)

            offsets = unscaled_offsets * self.KP_extent
            offset_features = offset_features[:, self.p_dim * self.kernel_size :]
            modulations = 2 * torch.sigmoid(offset_features) if self.modulated else None

        else:
            offsets, modulations = None, None

        # 2. Apply the main convolution
        s_pts = torch.cat([s_pts, torch.zeros_like(s_pts[:1, :]) + 1e6], 0)
        neighbors = s_pts[neighbor_idxs, :] - q_pts.unsqueeze(1)

        # Apply offsets to kernel points
        if offsets is not None:
            deformed_kernel = self.kernel + offsets
            self.deformed_kernel = deformed_kernel
            kernel_points = deformed_kernel.unsqueeze(1)
        else:
            kernel_points = self.kernel

        sq_distances = torch.sum((neighbors.unsqueeze(2) - kernel_points) ** 2, dim=-1)

        # Save minimum distances for regularization loss
        if self.deformable:
            self.min_d2, _ = torch.min(sq_distances, dim=1)

            # Filter neighbors in range
            in_range = torch.any(sq_distances < self.KP_extent**2, dim=2).type(torch.int32)
            new_max_neighb = torch.max(torch.sum(in_range, dim=1))

            # For each point, get the indices of neighbors in range
            neighb_row_bool, neighb_row_inds = torch.topk(in_range, k=int(new_max_neighb.item()), dim=1)

            # Gather new neighbor indices and distances
            new_neighb_inds = neighbor_idxs.gather(1, neighb_row_inds)
            neighb_row_inds = neighb_row_inds.unsqueeze(2).expand(-1, -1, self.kernel_size)
            sq_distances = sq_distances.gather(1, neighb_row_inds)

            # Shadow neighbors point to the last shadow point
            new_neighb_inds *= neighb_row_bool
            new_neighb_inds -= (neighb_row_bool.type(torch.int64) - 1) * int(s_pts.shape[0] - 1)
        else:
            new_neighb_inds = neighbor_idxs

        weights = self._compute_weights(sq_distances)

        if self.aggregation_mode == "closest":
            neighbors_1nn = torch.argmin(sq_distances, dim=2)
            weights = weights * torch.transpose(nn.functional.one_hot(neighbors_1nn, self.kernel_size), 1, 2)

        # Add a zero feature for shadow neighbors
        x = torch.cat((x, torch.zeros_like(x[:1, :])), 0)

        neighbor_x = gather(x, new_neighb_inds)
        weighted_features = torch.matmul(weights, neighbor_x)

        if self.deformable and modulations is not None:
            weighted_features *= modulations.unsqueeze(2)

        kernel_outputs = torch.matmul(weighted_features.permute(1, 0, 2), self.weights)

        return torch.sum(kernel_outputs, dim=0)

    def extra_repr(self) -> str:
        return (
            f"radius={self.radius}, in_channels={self.in_channels}, out_channels={self.out_channels} "
            f"KP_extent={self.KP_extent}, fixed_kernel_points={self.fixed_kernel_points}, "
            f"KP_influence={self.KP_influence}, aggregation_mode={self.aggregation_mode}, "
            f"deformable={self.deformable}, modulated={self.modulated}"
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.extra_repr()})"


class ResNetBottleneckBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        radius: float,
        layer_ind: int,
        KP_extent: float,
        conv_radius: float,
        use_batch_norm: bool,
        batch_norm_momentum: float,
        num_kernel_points: int,
        in_points_dim: int,
        KP_influence: str,
        aggregation_mode: str,
        fixed_kernel_points: Literal["none", "center", "vertical"],
        deformable: bool,
        modulated: bool,
        strided: bool,
    ) -> None:
        super().__init__()

        # Calculate KP extent from current radius
        current_extent = radius * KP_extent / conv_radius

        # Save configuration options
        self.use_bn = use_batch_norm
        self.bn_momentum = batch_norm_momentum
        self.layer_ind = layer_ind
        self.strided = strided

        # Downscaling block
        self.unary1 = (
            UnaryBlock(in_dim, out_dim // 4, self.use_bn, self.bn_momentum) if in_dim != out_dim // 4 else nn.Identity()
        )

        # KPConv block with batch normalization
        self.KPConv = KPConv(
            num_kernel_points,
            in_points_dim,
            out_dim // 4,
            out_dim // 4,
            current_extent,
            radius,
            fixed_kernel_points=fixed_kernel_points,
            KP_influence=KP_influence,
            aggregation_mode=aggregation_mode,
            deformable=deformable,
            modulated=modulated,
        )
        self.batch_norm_conv = BatchNormBlock(out_dim // 4, self.use_bn, self.bn_momentum)

        # Upscaling block
        self.unary2 = UnaryBlock(out_dim // 4, out_dim, self.use_bn, self.bn_momentum, no_relu=True)

        # Shortcut block
        self.unary_shortcut = (
            UnaryBlock(in_dim, out_dim, self.use_bn, self.bn_momentum, no_relu=True)
            if in_dim != out_dim
            else nn.Identity()
        )

        # Activation function
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, points: torch.Tensor, features: Tensor, neighb_inds: torch.Tensor) -> Tensor:
        if self.strided:
            q_pts = points[self.layer_ind + 1]
            s_pts = points[self.layer_ind]
            # neighb_inds = pools[self.layer_ind]
        else:
            q_pts = points[self.layer_ind]
            s_pts = points[self.layer_ind]
            # neighb_inds = neighbors[self.layer_ind]

        # Apply first downscaling MLP
        x = self.unary1(features)

        # Apply KPConv
        x = self.KPConv(q_pts, s_pts, neighb_inds, x)
        x = self.leaky_relu(self.batch_norm_conv(x))

        # Apply second upscaling MLP
        x = self.unary2(x)

        # Apply shortcut and combine
        shortcut = self._get_shortcut(features, neighb_inds)
        return self.leaky_relu(x + shortcut)

    def _get_points_and_neighbors(self, batch: Any) -> Tuple[Tensor, Tensor, Tensor]:
        if self.strided:
            q_pts = batch.points[self.layer_ind + 1]
            s_pts = batch.points[self.layer_ind]
            neighb_inds = batch.pools[self.layer_ind]
        else:
            q_pts = batch.points[self.layer_ind]
            s_pts = batch.points[self.layer_ind]
            neighb_inds = batch.neighbors[self.layer_ind]
        return q_pts, s_pts, neighb_inds

    def _get_shortcut(self, features: Tensor, neighb_inds: Tensor) -> Tensor:
        if self.strided:
            return self.unary_shortcut(max_pool(features, neighb_inds))
        else:
            return self.unary_shortcut(features)


class KPConvClassification(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        #####################
        # Network opperations
        #####################

        # Current radius of convolution and feature dimension
        layer = 0
        r = config.first_subsampling_dl * config.conv_radius
        in_dim = config.in_features_dim
        out_dim = config.first_features_dim
        self.K = config.num_kernel_points

        # Save all block operations in a list of modules
        self.block_ops = nn.ModuleList()

        # Loop over consecutive blocks
        block_in_layer = 0
        for block_i, block in enumerate(config.architecture):
            # Check equivariance
            if ("equivariant" in block) and (not out_dim % 3 == 0):
                raise ValueError("Equivariant block but features dimension is not a factor of 3")

            # Detect upsampling block to stop
            if "upsample" in block:
                break

            # Apply the good block function defining tf ops
            self.block_ops.append(block_decider(block, r, in_dim, out_dim, layer, config))

            # Index of block in this layer
            block_in_layer += 1

            # Update dimension of input from output
            if "simple" in block:
                in_dim = out_dim // 2
            else:
                in_dim = out_dim

            # Detect change to a subsampled layer
            if "pool" in block or "strided" in block:
                # Update radius and feature dimension for next layer
                layer += 1
                r *= 2
                out_dim *= 2
                block_in_layer = 0

        self.head_mlp = UnaryBlock(out_dim, 1024, False, 0)
        self.head_softmax = UnaryBlock(1024, config.num_classes, False, 0, no_relu=True)

        ################
        # Network Losses
        ################

        self.criterion = torch.nn.CrossEntropyLoss()
        self.deform_fitting_mode = config.deform_fitting_mode
        self.deform_fitting_power = config.deform_fitting_power
        self.deform_lr_factor = config.deform_lr_factor
        self.repulse_extent = config.repulse_extent
        self.output_loss = 0
        self.reg_loss = 0
        self.l1 = nn.L1Loss()

        return

    def forward(self, batch: Any, config: Any) -> torch.Tensor:
        # Save all block operations in a list of modules
        x = batch.features.clone().detach()

        # Loop over consecutive blocks
        for block_op in self.block_ops:
            x = block_op(x, batch)

        # Head of network
        x = self.head_mlp(x, batch)
        x = self.head_softmax(x, batch)

        return x

    def loss(self, outputs: Any, labels: Any) -> torch.Tensor:
        """
        Runs the loss on outputs of the model
        :param outputs: logits
        :param labels: labels
        :return: loss
        """

        # Cross entropy loss
        self.output_loss = self.criterion(outputs, labels)

        # Regularization of deformable offsets
        if self.deform_fitting_mode == "point2point":
            self.reg_loss = p2p_fitting_regularizer(self)
        elif self.deform_fitting_mode == "point2plane":
            raise ValueError("point2plane fitting mode not implemented yet.")
        else:
            raise ValueError("Unknown fitting mode: " + self.deform_fitting_mode)

        # Combined loss
        return self.output_loss + self.reg_loss

    @staticmethod
    def accuracy(outputs: Any, labels: Any) -> float:
        predicted = torch.argmax(outputs.data, dim=1)
        total = labels.size(0)
        correct = (predicted == labels).sum().item()

        return correct / total
