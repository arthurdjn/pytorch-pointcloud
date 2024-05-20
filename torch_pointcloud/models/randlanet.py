from numbers import Number, Real
from typing import Any, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import FloatTensor, LongTensor, Tensor

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
        new_features = self.nn(new_features)
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
        self.fp2 = FPModule(shared_mlp2d([128 + 64, 32]), k=1)
        self.fp1 = FPModule(shared_mlp2d([32 + 8, 8]), k=1)

        # decoder_kwargs = dict(transpose=True, bn=True, activation_fn=nn.ReLU())
        # self.fp4 = shared_mlp2d([1024, 256], act="relu", bn=True, transpose=True, plain_last=True)  # [512 + 256, 256]
        # self.fp3 = shared_mlp2d([512, 128], act="relu", bn=True, transpose=True, plain_last=True)  # [256 + 128, 128]
        # self.fp2 = shared_mlp2d([256, 32], act="relu", bn=True, transpose=True, plain_last=True)  # [128 + 32, 32]
        # self.fp1 = shared_mlp2d(
        #     [64, 8], act="relu", bn=True, transpose=True, plain_last=True
        # )  # [32 + 32, d_bottleneck]
        # head
        # self.mlp_classif = SharedMLP([8, 64, 32], dropout=[0.0, 0.5])
        # self.fc_classif = nn.Linear(32, num_classes)

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

    def forward(self, xyz: torch.Tensor, features: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, _ = xyz.size()
        feat = features if features is not None else xyz.transpose(1, 2)
        feat = self.fc0(feat.transpose(1, 2)).transpose(1, 2).unsqueeze(-1)
        feat = self.bn0(feat)  # shape (B, d, N, 1)
        print(f"{feat.shape = }")

        b1_out = self.block1(xyz, feat)
        print(b1_out.shape)
        lengths = torch.full((B,), N, dtype=torch.long, device=xyz.device)
        b1_out_decimated, idxs1 = decimate(xyz, lengths, self.decimation)

        return b1_out_decimated

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


def decimation_indices(
    lengths: torch.Tensor,
    decimation_factor: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if decimation_factor < 1:
        raise ValueError(
            f"The argument `decimation_factor` should be higher than (or "
            f"equal to) 1 for downsampling. (got {decimation_factor})"
        )

    batch_size = int(lengths.size(0))
    lengths.clamp_(min=1)
    decim_lengths = torch.div(lengths, decimation_factor, rounding_mode="floor").clamp(min=1)
    max_decim_length = int(decim_lengths.max().item())
    decim_indices = torch.full((batch_size, max_decim_length), -1, dtype=torch.long, device=lengths.device)
    for i in range(batch_size):
        l = int(lengths[i])
        sampled_indices = torch.randperm(l, device=lengths.device)[: decim_lengths[i]]
        decim_indices[i, : sampled_indices.size(0)] = sampled_indices

    return decim_indices, decim_lengths


def decimate(
    tensors: Tuple[torch.Tensor, ...],
    lengths: torch.Tensor,
    decimation_factor: float,
) -> Tuple[Tuple[torch.Tensor, ...], torch.Tensor]:
    idx_decim, ptr_decim = decimation_indices(lengths, decimation_factor)
    tensors_decim = tuple(tensor[idx_decim] for tensor in tensors)
    return tensors_decim, ptr_decim
