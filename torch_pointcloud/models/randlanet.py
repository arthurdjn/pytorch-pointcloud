from typing import Any, Optional, Tuple

import torch
import torch.nn as nn

from torch_pointcloud.ops import knn, knn_interpolate


class SharedMLP(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        transpose: bool = False,
        bn: bool = True,
        activation_fn: Any = None,
    ) -> None:
        super(SharedMLP, self).__init__()
        self.conv: nn.Module

        if transpose:
            self.conv = nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=(kernel_size - 1) // 2,
            )
        else:
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=(kernel_size - 1) // 2,
            )

        self.batch_norm = nn.BatchNorm2d(out_channels, eps=1e-6, momentum=0.01) if bn else None
        self.activation_fn = activation_fn

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.conv(input)
        if self.batch_norm:
            x = self.batch_norm(x)
        if self.activation_fn:
            x = self.activation_fn(x)
        return x


class LocalSpatialEncoding(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_neighbors: int, encode_coords: bool = False) -> None:
        super().__init__()
        self.encode_coords = encode_coords
        self.num_neighbors = num_neighbors
        # self.mlp = shared_mlp2d([in_features, out_features], act=nn.LeakyReLU(0.2), bn=True, plain_last=False)
        self.mlp = SharedMLP(in_channels, out_channels, bn=True, activation_fn=nn.LeakyReLU(0.2))

    @staticmethod
    def gather_neighbors(coords: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """Gather features based on neighbor indices.

        Args:
            coords: torch.Tensor of shape (B, N, d)
            idxs: torch.Tensor of shape (B, N, K)

        Returns:
            gathered neighbors of shape (B, dim, N, K)
        """
        B, N, K = indices.size()
        d = coords.size(-1)
        extended_indices = indices.unsqueeze(1).expand(B, d, N, K)
        extended_coords = coords.transpose(-2, -1).unsqueeze(-1).expand(B, d, N, K)
        return torch.gather(extended_coords, 2, extended_indices)

    def forward(
        self,
        coords: torch.Tensor,
        features: torch.Tensor,
        neighbor_indices: torch.Tensor,
        relative_features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the Module.

        Args:
            coords: Coordinates of the pointcloud, of shape (B, N, d).
            features: Features of the pointcloud, of shape (B, C, N).
            neighbor_indices: Indices of k neighbors, of shape (B, N, K).
            relative_features: Relative neighbor features calculated on first pass.

        Returns:
            torch.Tensor of shape (B, 2*d, N, K)
        """
        B, N, K = neighbor_indices.size()
        d = coords.size(-1)

        if self.encode_coords:
            neighbor_coords = self.gather_neighbors(coords, neighbor_indices)
            extended_coords = coords.transpose(-2, -1).unsqueeze(-1).expand(B, d, N, K)
            relative_coords = extended_coords - neighbor_coords
            relative_dists = torch.sqrt(torch.sum(torch.square(relative_coords), dim=1, keepdim=True))
            relative_features = torch.cat([relative_dists, relative_coords, extended_coords, neighbor_coords], dim=1)

        if relative_features is None:
            raise ValueError(
                "LocalSpatialEncoding requires `relative_features`. Either pass it or set `encode_coords=True`."
            )

        relative_features: torch.Tensor = self.mlp(relative_features)  # type: ignore[no-redef]
        neighbor_features = self.gather_neighbors(features.transpose(1, 2).squeeze(3), neighbor_indices)

        return torch.cat([neighbor_features, relative_features], dim=1), relative_features


class AttentivePooling(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.attn = nn.Sequential(nn.Linear(in_channels, in_channels), nn.Softmax(dim=-2))
        # self.mlp = shared_mlp2d([in_channels, out_channels], act=nn.LeakyReLU(0.2), bn=True, plain_last=False)
        self.mlp = SharedMLP(in_channels, out_channels, bn=True, activation_fn=nn.LeakyReLU(0.2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.attn(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = torch.sum(x * attn, dim=-1, keepdim=True)  # (B, d_in, N, 1)
        return self.mlp(x)


class LocalFeatureAggregation(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_neighbors: int,
        act: nn.Module = nn.LeakyReLU(),  # TODO: fix activation function with get_activation
    ) -> None:
        super().__init__()

        self.num_neighbors = num_neighbors
        self.act = act

        self.mlp1 = SharedMLP(in_channels, out_channels // 2, bn=True, activation_fn=nn.LeakyReLU(0.2))
        self.mlp2 = SharedMLP(out_channels, 2 * out_channels, bn=True, activation_fn=None)
        self.mlp_skip = SharedMLP(in_channels, 2 * out_channels, bn=True, activation_fn=None)

        self.lse1 = LocalSpatialEncoding(10, out_channels // 2, num_neighbors=num_neighbors, encode_coords=True)
        self.lse2 = LocalSpatialEncoding(out_channels // 2, out_channels // 2, num_neighbors=num_neighbors)

        self.pool1 = AttentivePooling(out_channels, out_channels // 2)
        self.pool2 = AttentivePooling(out_channels, out_channels)

    def forward(self, coords: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        _, neighbor_idxs = knn(coords, coords, k=self.num_neighbors)
        x = features.unsqueeze(-1)

        skip_x = self.mlp_skip(x)
        x = self.mlp1(x)
        x, relative_features = self.lse1(coords, x, neighbor_idxs)
        x = self.pool1(x)

        x, _ = self.lse2(coords, x, neighbor_idxs, relative_features=relative_features)
        x = self.pool2(x)

        x = self.act(self.mlp2(x) + skip_x)
        return x.squeeze(-1)


class FPModule(nn.Module):
    def __init__(self, nn: nn.Module, k: int = 1) -> None:
        super().__init__()
        self.nn = nn
        self.k = k

    def forward(
        self,
        xyz: torch.Tensor,
        features: torch.Tensor,
        xyz_skip: torch.Tensor,
        features_skip: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        features_t = features.transpose(1, 2)  # (B, C, M)
        new_features_t = knn_interpolate(features_t, xyz, xyz_skip, k=self.k)  # (B, N, C)
        new_features = new_features_t.transpose(1, 2)  # (B, C, N)
        new_features = torch.cat([new_features, features_skip], dim=1)  # (B, C2 + C1, N)
        new_features = self.nn(new_features.unsqueeze(-1)).squeeze(-1)
        return xyz_skip, new_features


def decimate(
    coords: torch.Tensor,
    features: torch.Tensor,
    factor: float,
    lengths: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Randomly samples the point cloud data (coordinates and features) by a specified factor.

    Args:
        coords: The xyz coordinates of the point cloud, shape (B, N, 3).
        features: The features associated with each point, shape (B, C, N).
        factor: The factor by which to decimate the point cloud.
        lengths: A tensor of lengths specifying the number of points in each batch for padded tensors, shape (B,).
            If `None`, all points are considered.

    Returns:
        The decimated coordinates, features, and the new lengths.
    """
    B, C, _ = features.shape
    _, N, D = coords.shape

    if factor < 1:
        raise ValueError(f"The argument `factor` should be higher than (or equal to) 1. Got {factor}.")

    if lengths is None:
        lengths = torch.full((B,), N, dtype=torch.long, device=coords.device)

    # Ensure decimation factor does not exceed the number of points
    lengths = torch.clamp(lengths, min=1)
    out_lengths = torch.div(lengths, factor, rounding_mode="floor").clamp(min=1)
    max_length = int(out_lengths.max().item())

    # Initialize tensors for decimated outputs
    out_coords = torch.zeros((B, max_length, D), dtype=coords.dtype, device=coords.device)
    out_features = torch.zeros((B, C, max_length), dtype=features.dtype, device=features.device)

    # Populate decimated tensors
    for b in range(B):
        n = int(lengths[b])
        indices = torch.randperm(n, device=coords.device)[: out_lengths[b]]
        out_coords[b, : out_lengths[b]] = coords[b, indices]
        out_features[b, :, : out_lengths[b]] = features[b, :, indices]

    return out_coords, out_features, out_lengths, indices


class RandLANetClassification(nn.Module):
    def __init__(self, num_features: int, num_classes: int, num_neighbors: int = 16, decimation: int = 4) -> None:
        super().__init__()
        self.num_neighbors = num_neighbors
        self.decimation = decimation

        # NOTE: Authors use 8, which is a bottleneck
        # for the final MLP, and also when `num_classes > 8` or `num_features > 8`.
        # num_features_bottleneck = max(32, num_classes, num_features)

        # encoder
        self.fc0 = nn.Linear(num_features, 8)
        self.bn0 = nn.BatchNorm2d(8)

        # 2 DilatedResidualBlock converges better than 4 on ModelNet.
        self.block1 = LocalFeatureAggregation(8, 16, num_neighbors)
        self.block2 = LocalFeatureAggregation(32, 64, num_neighbors)
        self.mlp = SharedMLP(128, 128, bn=True, activation_fn=nn.ReLU())
        self.head = nn.Sequential(
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, num_classes),
        )

    # TODO refactor some blocks to return both xyz and features to be easier to used (we expect to have both)
    def forward(
        self,
        xyz: torch.Tensor,
        features: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, _ = xyz.size()
        feat = features if features is not None else xyz.transpose(1, 2)  # (B, C, N)

        feat = self.fc0(feat.transpose(1, 2)).transpose(1, 2)
        feat = self.bn0(feat.unsqueeze(-1)).squeeze(-1)

        lfa1_feat = self.block1(xyz, feat)
        lfa1_xyz, lfa1_feat, lfa1_lengths, _ = decimate(xyz, lfa1_feat, self.decimation, lengths=lengths)

        b2_feat = self.block2(lfa1_xyz, lfa1_feat)
        b2_xyz, b2_feat, b2_lengths, _ = decimate(lfa1_xyz, b2_feat, self.decimation, lengths=lfa1_lengths)

        x = self.mlp(b2_feat.unsqueeze(-1)).squeeze(-1)
        # Max pooling
        x = torch.max(x, dim=-1)[0]
        return self.head(x)


class RandLANetSegmentation(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        features_dim: int = 8,
        num_neighbors: int = 16,
        decimation: int = 4,
    ):
        super().__init__()
        self.num_neighbors = num_neighbors
        self.decimation = decimation

        # NOTE: Authors use 8, which is a bottleneck
        # for the final MLP, and also when `num_classes > 8` or `num_features > 8`.
        # num_features_bottleneck = max(32, num_classes, num_features)

        self.fc0 = nn.Linear(in_channels, features_dim)
        self.bn0 = nn.BatchNorm2d(features_dim)
        # encoder
        self.lfa1 = LocalFeatureAggregation(features_dim, 16, num_neighbors)
        self.lfa2 = LocalFeatureAggregation(32, 64, num_neighbors)
        self.lfa3 = LocalFeatureAggregation(128, 128, num_neighbors)
        self.lfa4 = LocalFeatureAggregation(256, 256, num_neighbors)
        self.mlp_summit = SharedMLP(512, 512, activation_fn=nn.LeakyReLU(0.2))
        # decoder
        # self.fp4 = FPModule(shared_mlp2d([512 + 256, 256]), k=1)
        # self.fp3 = FPModule(shared_mlp2d([256 + 128, 128]), k=1)
        # self.fp2 = FPModule(shared_mlp2d([128 + 32, 32]), k=1)
        # self.fp1 = FPModule(shared_mlp2d([32 + 32, features_dim]), k=1)
        self.fp4 = FPModule(SharedMLP(512 + 256, 256, activation_fn=None), k=1)
        self.fp3 = FPModule(SharedMLP(256 + 128, 128, activation_fn=None), k=1)
        self.fp2 = FPModule(SharedMLP(128 + 32, 32, activation_fn=None), k=1)
        self.fp1 = FPModule(SharedMLP(32 + 32, features_dim, activation_fn=None), k=1)

        # self.fp4 = SharedMLP(512 + 256, 256, transpose=True, activation_fn=nn.LeakyReLU(0.2))
        # self.fp3 = SharedMLP(256 + 128, 128, transpose=True, activation_fn=nn.LeakyReLU(0.2))
        # self.fp2 = SharedMLP(128 + 32, 32, transpose=True, activation_fn=nn.LeakyReLU(0.2))
        # self.fp1 = SharedMLP(32 + 32, 32, transpose=True, activation_fn=nn.LeakyReLU(0.2))

        # head
        self.head = nn.Sequential(
            SharedMLP(features_dim, 64, bn=True, activation_fn=nn.LeakyReLU(0.2)),
            SharedMLP(64, 32, activation_fn=nn.LeakyReLU(0.2)),
            nn.Dropout(0.5),
            SharedMLP(32, num_classes, bn=False),
        )

    def forward(
        self,
        coords: torch.Tensor,
        features: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = features if features is not None else coords.transpose(1, 2)  # (B, C, N)

        x = self.fc0(x.transpose(1, 2)).transpose(1, 2)
        x = self.bn0(x.unsqueeze(-1)).squeeze(-1)

        lfa1_feat = self.lfa1(coords, x)
        lfa1_feat_before = lfa1_feat.clone()
        lfa1_coords, lfa1_feat, lfa1_lengths, _ = decimate(coords, lfa1_feat, self.decimation, lengths=lengths)

        b2_feat = self.lfa2(lfa1_coords, lfa1_feat)
        b2_coords, b2_feat, b2_lengths, _ = decimate(lfa1_coords, b2_feat, self.decimation, lengths=lfa1_lengths)

        b3_feat = self.lfa3(b2_coords, b2_feat)
        b3_coords, b3_feat, b3_lengths, _ = decimate(b2_coords, b3_feat, self.decimation, lengths=b2_lengths)

        b4_feat = self.lfa4(b3_coords, b3_feat)
        b4_coords, b4_feat, *_ = decimate(b3_coords, b4_feat, self.decimation, lengths=b3_lengths)

        feat = self.mlp_summit(b4_feat.unsqueeze(-1)).squeeze(-1)

        fp4_coords, fp4_feat = self.fp4(b4_coords, feat, b3_coords, b3_feat)
        fp3_coords, fp3_feat = self.fp3(fp4_coords, fp4_feat, b2_coords, b2_feat)
        fp2_coords, fp2_feat = self.fp2(fp3_coords, fp3_feat, lfa1_coords, lfa1_feat)
        _, fp1_feat = self.fp1(fp2_coords, fp2_feat, coords, lfa1_feat_before)

        # ? Use a return_logits flag ?
        return self.head(fp1_feat.unsqueeze(-1)).squeeze(-1)
