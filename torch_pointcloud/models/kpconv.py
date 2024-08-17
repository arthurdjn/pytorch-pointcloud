import math

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.init import kaiming_uniform_
from torch.nn.parameter import Parameter


def kernel_points(method: str = "lloyd") -> None:
    pass


class KPConv(nn.Module):
    def __init__(
        self,
        kernel_size: int,
        p_dim: int,
        in_channels: int,
        out_channels: int,
        KP_extent: float,
        radius: float,
        fixed_kernel_points: str = "center",
        KP_influence: str = "linear",
        aggregation_mode: str = "sum",
        deformable: bool = False,
        modulated: bool = False,
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.p_dim = p_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.radius = radius
        self.KP_extent = KP_extent
        self.fixed_kernel_points = fixed_kernel_points
        self.KP_influence = KP_influence
        self.aggregation_mode = aggregation_mode
        self.deformable = deformable
        self.modulated = modulated

        # Running variable containing deformed KP distance to input points. (used in regularization loss)
        self.min_d2 = None
        self.deformed_KP = None
        self.offset_features = None

        # Initialize weights
        self.weights = Parameter(
            torch.zeros((kernel_size, in_channels, out_channels), dtype=torch.float32), requires_grad=True
        )

        # Initiate weights for offsets
        if deformable:
            if modulated:
                self.offset_dim = (self.p_dim + 1) * self.K
            else:
                self.offset_dim = self.p_dim * self.K
            self.offset_conv = KPConv(
                self.K,
                self.p_dim,
                self.in_channels,
                self.offset_dim,
                KP_extent,
                radius,
                fixed_kernel_points=fixed_kernel_points,
                KP_influence=KP_influence,
                aggregation_mode=aggregation_mode,
            )
            self.offset_bias = Parameter(torch.zeros(self.offset_dim, dtype=torch.float32), requires_grad=True)

        else:
            self.offset_dim = None
            self.offset_conv = None
            self.offset_bias = None

        # Reset parameters
        self.reset_parameters()

        # Initialize kernel points
        self.kernel_points = self.load_kernel()

    def reset_parameters(self) -> None:
        kaiming_uniform_(self.weights, a=math.sqrt(5))
        if self.deformable:
            nn.init.zeros_(self.offset_bias)

    def load_kernel(self) -> Tensor:
        # Create one kernel disposition (as numpy array). Choose the KP distance to center thanks to the KP extent
        K_points_numpy = load_kernels(self.radius, self.K, dimension=self.p_dim, fixed=self.fixed_kernel_points)
        return Parameter(torch.tensor(K_points_numpy, dtype=torch.float32), requires_grad=False)

    def forward(self, q_pts: Tensor, s_pts: Tensor, neighb_inds: Tensor, x: Tensor) -> Tensor:
        if self.deformable:
            # Get offsets with a KPConv that only takes part of the features
            self.offset_features = self.offset_conv(q_pts, s_pts, neighb_inds, x) + self.offset_bias

            if self.modulated:

                # Get offset (in normalized scale) from features
                unscaled_offsets = self.offset_features[:, : self.p_dim * self.K]
                unscaled_offsets = unscaled_offsets.view(-1, self.K, self.p_dim)

                # Get modulations
                modulations = 2 * torch.sigmoid(self.offset_features[:, self.p_dim * self.K :])

            else:

                # Get offset (in normalized scale) from features
                unscaled_offsets = self.offset_features.view(-1, self.K, self.p_dim)

                # No modulations
                modulations = None

            # Rescale offset for this layer
            offsets = unscaled_offsets * self.KP_extent

        else:
            offsets = None
            modulations = None

        ######################
        # Deformed convolution
        ######################

        # Add a fake point in the last row for shadow neighbors
        s_pts = torch.cat((s_pts, torch.zeros_like(s_pts[:1, :]) + 1e6), 0)

        # Get neighbor points [n_points, n_neighbors, dim]
        neighbors = s_pts[neighb_inds, :]

        # Center every neighborhood
        neighbors = neighbors - q_pts.unsqueeze(1)

        # Apply offsets to kernel points [n_points, n_kpoints, dim]
        if self.deformable:
            self.deformed_KP = offsets + self.kernel_points
            deformed_K_points = self.deformed_KP.unsqueeze(1)
        else:
            deformed_K_points = self.kernel_points

        # Get all difference matrices [n_points, n_neighbors, n_kpoints, dim]
        neighbors.unsqueeze_(2)
        differences = neighbors - deformed_K_points

        # Get the square distances [n_points, n_neighbors, n_kpoints]
        sq_distances = torch.sum(differences**2, dim=3)

        # Optimization by ignoring points outside a deformed KP range
        if self.deformable:

            # Save distances for loss
            self.min_d2, _ = torch.min(sq_distances, dim=1)

            # Boolean of the neighbors in range of a kernel point [n_points, n_neighbors]
            in_range = torch.any(sq_distances < self.KP_extent**2, dim=2).type(torch.int32)

            # New value of max neighbors
            new_max_neighb = torch.max(torch.sum(in_range, dim=1))

            # For each row of neighbors, indices of the ones that are in range [n_points, new_max_neighb]
            neighb_row_bool, neighb_row_inds = torch.topk(in_range, new_max_neighb.item(), dim=1)

            # Gather new neighbor indices [n_points, new_max_neighb]
            new_neighb_inds = neighb_inds.gather(1, neighb_row_inds, sparse_grad=False)

            # Gather new distances to KP [n_points, new_max_neighb, n_kpoints]
            neighb_row_inds.unsqueeze_(2)
            neighb_row_inds = neighb_row_inds.expand(-1, -1, self.K)
            sq_distances = sq_distances.gather(1, neighb_row_inds, sparse_grad=False)

            # New shadow neighbors have to point to the last shadow point
            new_neighb_inds *= neighb_row_bool
            new_neighb_inds -= (neighb_row_bool.type(torch.int64) - 1) * int(s_pts.shape[0] - 1)
        else:
            new_neighb_inds = neighb_inds

        # Get Kernel point influences [n_points, n_kpoints, n_neighbors]
        if self.KP_influence == "constant":
            # Every point get an influence of 1.
            all_weights = torch.ones_like(sq_distances)
            all_weights = torch.transpose(all_weights, 1, 2)

        elif self.KP_influence == "linear":
            # Influence decrease linearly with the distance, and get to zero when d = KP_extent.
            all_weights = torch.clamp(1 - torch.sqrt(sq_distances) / self.KP_extent, min=0.0)
            all_weights = torch.transpose(all_weights, 1, 2)

        elif self.KP_influence == "gaussian":
            # Influence in gaussian of the distance.
            sigma = self.KP_extent * 0.3
            all_weights = radius_gaussian(sq_distances, sigma)
            all_weights = torch.transpose(all_weights, 1, 2)
        else:
            raise ValueError("Unknown influence function type (config.KP_influence)")

        # In case of closest mode, only the closest KP can influence each point
        if self.aggregation_mode == "closest":
            neighbors_1nn = torch.argmin(sq_distances, dim=2)
            all_weights *= torch.transpose(nn.functional.one_hot(neighbors_1nn, self.K), 1, 2)

        elif self.aggregation_mode != "sum":
            raise ValueError("Unknown convolution mode. Should be 'closest' or 'sum'")

        # Add a zero feature for shadow neighbors
        x = torch.cat((x, torch.zeros_like(x[:1, :])), 0)

        # Get the features of each neighborhood [n_points, n_neighbors, in_fdim]
        neighb_x = gather(x, new_neighb_inds)

        # Apply distance weights [n_points, n_kpoints, in_fdim]
        weighted_features = torch.matmul(all_weights, neighb_x)

        # Apply modulations
        if self.deformable and self.modulated:
            weighted_features *= modulations.unsqueeze(2)

        # Apply network weights [n_kpoints, n_points, out_fdim]
        weighted_features = weighted_features.permute((1, 0, 2))
        kernel_outputs = torch.matmul(weighted_features, self.weights)

        # Convolution sum [n_points, out_fdim]
        return torch.sum(kernel_outputs, dim=0)

    def extra_repr(self) -> str:
        return f"radius={self.radius}, in_channels={self.in_channels}, out_channels={self.out_channels}"
