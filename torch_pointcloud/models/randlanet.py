from typing import Optional, Tuple

import torch
import torch.nn as nn

from torch_pointcloud.layers.mlp import SharedMLP, shared_mlp2d
from torch_pointcloud.ops import knn, knn_interpolate


class LocalSpatialEncoding(nn.Module):
    def __init__(self, num_channels: int, num_neighbors: int) -> None:
        super().__init__()
        self.num_neighbors = num_neighbors
        self.mlp = shared_mlp2d([10, num_channels], act="relu", bn=True)

    def forward(
        self,
        xyz: torch.Tensor,
        features: torch.Tensor,
        dists: torch.Tensor,
        idxs: torch.Tensor,
    ) -> torch.Tensor:
        B, N, K = idxs.size()
        # idx(B, N, K), coords(B, N, 3)
        extended_idx = idxs.unsqueeze(1).expand(B, 3, N, K)  # (B, 3, N, K)
        xyz = xyz.transpose(-2, -1).unsqueeze(-1).expand(B, 3, N, K)  # (B, 3, N, K)
        xyz_neighbors = torch.gather(xyz, 2, extended_idx)  # (B, 3, N, K)

        # relative point position encoding
        concat = torch.cat((xyz, xyz_neighbors, xyz - xyz_neighbors, dists.unsqueeze(-3)), dim=-3).to(xyz.device)
        out_features = self.mlp(concat)
        return torch.cat((out_features, features.expand(B, -1, N, K)), dim=-3)


class AttentivePooling(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.attn = nn.Sequential(nn.Linear(in_channels, in_channels, bias=False), nn.Softmax(dim=-2))
        self.mlp = shared_mlp2d([in_channels, out_channels], bn=True, act="relu")

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
        act: nn.Module = nn.LeakyReLU(),
        # TODO: fix activation function
    ) -> None:
        super().__init__()

        self.num_neighbors = num_neighbors
        self.act = act

        self.mlp1 = shared_mlp2d([in_channels, out_channels // 2], act=nn.LeakyReLU(0.2), plain_last=False)
        self.mlp2 = shared_mlp2d([out_channels, 2 * out_channels])
        self.mlp_skip = shared_mlp2d([in_channels, 2 * out_channels])

        self.lse1 = LocalSpatialEncoding(out_channels // 2, num_neighbors)
        self.lse2 = LocalSpatialEncoding(out_channels // 2, num_neighbors)

        self.pool1 = AttentivePooling(out_channels, out_channels // 2)
        self.pool2 = AttentivePooling(out_channels, out_channels)

    def forward(self, xyz: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        dists, idxs = knn(xyz, xyz, k=self.num_neighbors)

        skip_features = self.mlp_skip(features)
        features = self.mlp1(features)

        features = self.lse1(xyz, features, dists, idxs)
        features = self.pool1(features)

        features = self.lse2(xyz, features, dists, idxs)
        features = self.pool2(features)

        return self.act(self.mlp2(features) + skip_features)


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
        new_features_t = knn_interpolate(features_t, xyz, xyz_skip, k=3)  # (B, N, C)
        new_features = new_features_t.transpose(1, 2)  # (B, C, N)
        new_features = torch.cat([new_features, features_skip], dim=1)  # (B, C2 + C1, N)
        new_features = self.nn(new_features.unsqueeze(-1)).squeeze(-1)
        return xyz_skip, new_features


class RandLANet(nn.Module):
    def __init__(
        self,
        num_features: int,
        num_classes: int,
        num_neighbors: int = 16,
        decimation: int = 4,
    ):
        super(RandLANet, self).__init__()
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_neighbors = num_neighbors
        self.decimation = decimation

        # Authors use 8, which is a bottleneck
        # for the final MLP, and also when `num_classes > 8` or `num_features > 8`.
        num_features_bottleneck = max(32, num_classes, num_features)

        # encoder
        self.mlp0 = nn.Sequential(
            nn.Linear(num_features, 8),
            nn.BatchNorm2d(8, eps=1e-6, momentum=0.99),
            nn.LeakyReLU(0.2),
        )
        self.fc0 = nn.Linear(num_features, 8)  # d_bottleneck
        self.bn0 = nn.BatchNorm2d(8)

        self.block1 = LocalFeatureAggregation(8, 16, num_neighbors)  # (num_neighbors, d_bottleneck, 32)
        self.block2 = LocalFeatureAggregation(32, 64, num_neighbors)  # (num_neighbors, 32, 128)
        self.block3 = LocalFeatureAggregation(128, 128, num_neighbors)  # (num_neighbors, 128, 256)
        self.block4 = LocalFeatureAggregation(256, 256, num_neighbors)  # (num_neighbors, 256, 512)
        self.mlp_summit = shared_mlp2d([512, 512], act="relu", bn=True)
        # decoder
        self.fp4 = FPModule(shared_mlp2d([512 + 256, 256]), k=4)
        self.fp3 = FPModule(shared_mlp2d([256 + 128, 128]), k=2)
        self.fp2 = FPModule(shared_mlp2d([128 + 32, 32]), k=1)
        self.fp1 = FPModule(shared_mlp2d([32 + 32, 8]), k=1)

        # final semantic prediction
        self.head = nn.Sequential(
            nn.Conv2d(8, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv2d(32, num_classes, kernel_size=1),
        )

    def decimate(
        self, xyz: torch.Tensor, features: torch.Tensor, factor: float, lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decimates the point cloud data (xyz coordinates and features) by a specified factor.

        Args:
            xyz: The xyz coordinates of the point cloud, shape (B, N, 3).
            features: The features associated with each point, shape (B, C, N).
            factor: The factor by which to decimate the point cloud.
            lengths: A tensor of lengths specifying the number of points in each batch for padded tensors, shape (B,).
                If `None`, all points are considered.

        Returns:
            The decimated xyz coordinates, features, and the new lengths.
        """
        B, C, *_ = features.shape

        if factor < 1:
            raise ValueError(f"The argument `factor` should be higher than (or equal to) 1. Got {factor}.")

        if lengths is None:
            lengths = torch.full((B,), xyz.size(1), dtype=torch.long, device=xyz.device)

        # Ensure decimation factor does not exceed the number of points
        lengths = torch.clamp(lengths, min=1)
        out_lengths = torch.div(lengths, factor, rounding_mode="floor").clamp(min=1)
        max_length = int(out_lengths.max().item())

        # Initialize tensors for decimated outputs
        out_xyz = torch.zeros((B, max_length, 3), dtype=xyz.dtype, device=xyz.device)
        out_features = torch.zeros((B, C, max_length), dtype=features.dtype, device=features.device)

        # Populate decimated tensors
        for b in range(B):
            n = int(lengths[b])
            indices = torch.randperm(n, device=xyz.device)[: out_lengths[b]]
            out_xyz[b, : out_lengths[b]] = xyz[b, indices]
            out_features[b, :, : out_lengths[b]] = features[b, :, indices]

        return out_xyz, out_features, out_lengths, indices

    # TODO refactor some blocks to return both xyz and features to be easier to used (we expect to have both)
    def forward(
        self,
        xyz: torch.Tensor,
        features: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, _ = xyz.size()
        feat = features if features is not None else xyz.transpose(1, 2)  # (B, C, N)

        feat = self.fc0(feat.transpose(1, 2)).transpose(1, 2).unsqueeze(-1)  # (B, C, N, 1)
        feat = self.bn0(feat)  # (B, C, N, 1)

        b1_feat = self.block1(xyz, feat)  # (B, C, N, 1)
        b1_feat_before = b1_feat.clone().squeeze(-1)
        b1_xyz, b1_feat, b1_lengths, _ = self.decimate(
            xyz, b1_feat.squeeze(-1), factor=self.decimation, lengths=lengths
        )

        b2_feat = self.block2(b1_xyz, b1_feat.unsqueeze(-1))  # (B, C, N, 1)
        b2_xyz, b2_feat, b2_lengths, _ = self.decimate(
            b1_xyz, b2_feat.squeeze(-1), factor=self.decimation, lengths=b1_lengths
        )

        b3_feat = self.block3(b2_xyz, b2_feat.unsqueeze(-1))  # (B, C, N, 1)
        b3_xyz, b3_feat, b3_lengths, _ = self.decimate(
            b2_xyz, b3_feat.squeeze(-1), factor=self.decimation, lengths=b2_lengths
        )

        b4_feat = self.block4(b3_xyz, b3_feat.unsqueeze(-1))  # (B, C, N, 1)
        b4_xyz, b4_feat, b4_lengths, _ = self.decimate(
            b3_xyz, b4_feat.squeeze(-1), factor=self.decimation, lengths=b3_lengths
        )

        feat = self.mlp_summit(b4_feat.unsqueeze(-1)).squeeze(-1)  # (B, C, N, 1)

        fp4_xyz, fp4_feat = self.fp4(b4_xyz, feat, b3_xyz, b3_feat)
        fp3_xyz, fp3_feat = self.fp3(fp4_xyz, fp4_feat, b2_xyz, b2_feat)
        fp2_xyz, fp2_feat = self.fp2(fp3_xyz, fp3_feat, b1_xyz, b1_feat)
        _, fp1_feat = self.fp1(fp2_xyz, fp2_feat, xyz, b1_feat_before)

        # TODO: Use a return_logits flag ?
        # logits = self.fc_classif(x)
        # if self.return_logits:
        #     return logits
        # probas = logits.log_softmax(dim=-1)
        # return probas

        return self.head(fp1_feat.unsqueeze(-1)).squeeze(-1)
