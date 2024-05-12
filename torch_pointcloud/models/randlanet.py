from numbers import Number, Real
from typing import Any, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import FloatTensor, LongTensor, Tensor

from torch_pointcloud.layers.mlp import SharedMLP, shared_mlp2d
from torch_pointcloud.ops import knn, knn_interpolate


class LocalSpatialEncoding(nn.Module):
    r"""
    Parameters
    ----------
    coords: torch.Tensor, shape (B, N, 3)
        coordinates of the point cloud
    features: torch.Tensor, shape (B, d, N, 1)
        features of the point cloud
    neighbors: tuple

    Returns
    -------
    torch.Tensor, shape (B, 2*d, N, K)
    """

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
    r"""
    Forward pass

    Parameters
    ----------
    x: torch.Tensor, shape (B, in_channels, N, K)

    Returns
    -------
    torch.Tensor, shape (B, out_channels, N, 1)
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.attn = nn.Sequential(nn.Linear(in_channels, in_channels, bias=False), nn.Softmax(dim=-2))
        self.mlp = shared_mlp2d([in_channels, out_channels], bn=True, act="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.attn(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = torch.sum(x * attn, dim=-1, keepdim=True)  # (B, d_in, N, 1)
        return self.mlp(x)


class LocalFeatureAggregation(nn.Module):
    r"""
    Forward pass

    Parameters
    ----------
    coords: torch.Tensor, shape (B, N, 3)
        coordinates of the point cloud
    features: torch.Tensor, shape (B, d_in, N, 1)
        features of the point cloud

    Returns
    -------
    torch.Tensor, shape (B, 2*d_out, N, 1)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_neighbors: int,
        act: nn.Module = nn.LeakyReLU(),
    ) -> None:
        super().__init__()

        self.num_neighbors = num_neighbors
        self.act = act

        # TODO: fix activation function
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
        # for the final MLP, and also when num_classes>8
        # or num_features>8.
        num_features_bottleneck = max(32, num_classes, num_features)

        # encoder
        self.mlp0 = nn.Sequential(
            nn.Linear(num_features, 8),
            nn.BatchNorm2d(8, eps=1e-6, momentum=0.99),
            nn.LeakyReLU(0.2),
        )
        self.fc0 = nn.Linear(num_features, 8)  # d_bottleneck
        self.bc0 = nn.BatchNorm1d(8)

        self.block1 = LocalFeatureAggregation(8, 16, num_neighbors)  # (num_neighbors, d_bottleneck, 32)
        self.block2 = LocalFeatureAggregation(32, 64, num_neighbors)  # (num_neighbors, 32, 128)
        self.block3 = LocalFeatureAggregation(128, 128, num_neighbors)  # (num_neighbors, 128, 256)
        self.block4 = LocalFeatureAggregation(256, 256, num_neighbors)  # (num_neighbors, 256, 512)
        self.mlp_summit = shared_mlp2d([512, 512], act="relu", bn=True)
        # decoder
        decoder_kwargs = dict(transpose=True, bn=True, activation_fn=nn.ReLU())
        self.fp4 = shared_mlp2d([1024, 256], act="relu", bn=True, plain_last=True)  # [512 + 256, 256]
        self.fp3 = shared_mlp2d([512, 128], act="relu", bn=True, plain_last=True)  # [256 + 128, 128]
        self.fp2 = shared_mlp2d([256, 32], act="relu", bn=True, plain_last=True)  # [128 + 32, 32]
        self.fp1 = shared_mlp2d([64, 8], act="relu", bn=True, plain_last=True)  # [32 + 32, d_bottleneck]
        # head
        self.mlp_classif = SharedMLP([8, 64, 32], dropout=[0.0, 0.5])
        self.fc_classif = nn.Linear(32, num_classes)

        # final semantic prediction
        # self.head = nn.Sequential(
        #     shared_mlp2d([8, 64], bn=True, act=nn.ReLU()),
        #     shared_mlp2d([64, 32], bn=True, act=nn.ReLU()),
        #     nn.Dropout(),
        #     shared_mlp2d([32, num_classes], bn=False, act=None),
        # )

    def decimate(self, x: torch.Tensor, ptr: torch.Tensor, d: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decimate the input tensor, i.e. reduce the number of points by a factor `d`.

        Args:
            x: _description_
            ptr: _description_
            d: _description_

        Returns:
            _description_
        """
        B, N, C = x.size()
        idx = torch.arange(0, N, d, device=x.device)
        x = x[:, idx]
        ptr = ptr[:, idx]
        return x, ptr

    def forward(self, xyz: torch.Tensor, features: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, _ = xyz.size()

        feat = features if features is not None else xyz
        feat = self.fc0(feat).transpose(-2, -1).unsqueeze(-1)
        feat = self.bn0(feat)  # shape (B, d, N, 1)

        # <<<<<<<<<< ENCODER
        x_stack = []

        permutation = torch.randperm(N)
        coords = xyz[:, permutation]
        x = feat[:, :, permutation]

        d = self.decimation
        decimation_ratio = 1

        for lfa in self.encoder:
            # at iteration i, x.shape = (B, N//(d**i), d_in)
            x = lfa(coords[:, : N // decimation_ratio], x)
            x_stack.append(x.clone())
            decimation_ratio *= d
            x = x[:, :, : N // decimation_ratio]

        # # >>>>>>>>>> ENCODER

        x = self.mlp(x)

        # <<<<<<<<<< DECODER
        for mlp in self.decoder:
            neighbors, _ = knn(coords[:, : N // decimation_ratio], coords[:, : d * N // decimation_ratio], 1)
            # neighbors: shape (B, N, 1)
            neighbors = neighbors.to(x.device)
            extended_neighbors = neighbors.unsqueeze(1).expand(-1, x.size(1), -1, 1)
            x_neighbors = torch.gather(x, -2, extended_neighbors)
            x = torch.cat((x_neighbors, x_stack.pop()), dim=1)
            x = mlp(x)
            decimation_ratio //= d

        # >>>>>>>>>> DECODER
        # inverse permutation
        x = x[:, :, torch.argsort(permutation)]

        scores = self.fc_end(x)

        return scores.squeeze(-1)


class FPModule(nn.Module):
    """Upsampling with a skip connection."""

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
        new_features = self.nn(new_features)
        return xyz_skip, new_features
