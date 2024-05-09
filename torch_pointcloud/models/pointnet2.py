from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import BatchNorm1d, Conv1d, Dropout, Module, ModuleList, ReLU, Sequential

from torch_pointcloud.layers.mlp import MLP, shared_mlp2d
from torch_pointcloud.ops import ball_grouping, ball_query, fps, knn_interpolate


class PointNetSA(Module):
    def __init__(
        self,
        num_points: int,
        radius_list: List[float],
        samples_list: List[int],
        channels: List[List[int]],
        use_pos: bool = True,
        normalize_pos: bool = True,
    ) -> None:
        super().__init__()
        if not (len(radius_list) == len(samples_list) == len(channels)):
            raise ValueError(
                f"Invalid arguments. Expected len(radiuses) == len(samples) == len(channels), "
                f"but got {len(radius_list)=}, {len(samples_list)=}, {len(channels)=}"
            )

        self.num_points = num_points
        self.radius_list = radius_list
        self.samples_list = samples_list
        self.use_pos = use_pos
        self.normalize_pos = normalize_pos
        self.mlps = ModuleList([shared_mlp2d(c, bias=False) for c in channels])

    def forward(
        self,
        pos: Tensor,
        features: Optional[Tensor] = None,
        indices: Optional[Tensor] = None,
        # lengths: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        B, N, D = pos.shape
        # TODO: add check that there are enough points in the neighborhood
        # i.e. handle cases for padded tensors
        indices = indices if indices is not None else fps(pos, num_samples=self.num_points)  # (B, P)
        indices = indices.unsqueeze(-1).repeat(1, 1, D).long()  # (B, P, D)
        out_pos = pos.gather(1, indices)  # (B, P, D)

        out_layers = []
        for radius, k, mlp in zip(self.radius_list, self.samples_list, self.mlps):
            lengths = torch.tensor([N] * B, dtype=torch.int64)
            _, ball_idxs = ball_query(
                out_pos,
                pos,
                radius=radius,
                max_neighbors=k,
                lengths1=lengths,
                lengths2=lengths,
            )  # (B, P, K)

            grouped_pos = ball_grouping(pos.transpose(1, 2), ball_idxs)
            grouped_pos -= out_pos.transpose(1, 2).unsqueeze(-1)  # (B, D, P, K)

            if self.normalize_pos:
                grouped_pos /= radius

            if features is not None:
                grouped_features = ball_grouping(features, ball_idxs)  # (B, C, P, K)
                out_features = torch.cat([grouped_pos, grouped_features], dim=1) if self.use_pos else grouped_features
            else:
                assert self.use_pos, "Cannot have not features and not use xyz as a feature!"
                out_features = grouped_pos

            # out_features: (B, D + C, P, K)
            out_features = mlp(out_features)  # (B, C', P, K)
            out_features = F.max_pool2d(out_features, kernel_size=[1, out_features.size(3)])  # (B, C', P, 1)
            out_features = out_features.squeeze(-1)  # (B, C', P)
            out_layers.append(out_features)

        out_features = torch.cat(out_layers, 1)
        return out_pos, out_features, indices


class PointNetFP(Module):
    def __init__(self, channels: List[int], bn: bool = True, bias: bool = False) -> None:
        super().__init__()
        self.mlp = shared_mlp2d(channels, bias=bias, bn=bn)

    def forward(
        self,
        pos: Optional[torch.Tensor],
        features: Tensor,
        pos_skip: Tensor,
        features_skip: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        assert pos_skip.shape[2] == 3
        B, C, _ = features.size()
        N = pos_skip.size(1)

        if pos is not None:
            features_t = features.transpose(1, 2)  # (B, C, M)
            new_features_t = knn_interpolate(features_t, pos, pos_skip, k=3)  # (B, N, C)
            new_features = new_features_t.transpose(1, 2)  # (B, C, N)
        else:
            new_features = features.expand((B, C, N))

        if features_skip is not None:
            new_features = torch.cat([new_features, features_skip], dim=1)  # (B, C2 + C1, N)

        new_features = new_features.unsqueeze(-1)
        new_features = self.mlp(new_features)
        return pos_skip, new_features.squeeze(-1)


class PointNetGlobalSA(Module):
    def __init__(self, channels: List[int], mode: str = "max", bn: bool = True) -> None:
        super().__init__()
        if mode not in ["mean", "max"]:
            raise ValueError(f"Unrecognized mode {mode!r} for the PointNetGlobalSA. Must be 'mean' or 'max'.")

        self.mode = mode
        self.mlp = shared_mlp2d(channels, bias=False, bn=bn)

    def forward(self, pos: Tensor, features: Tensor) -> Tuple[Tensor, Tensor]:
        pos_t = pos.transpose(1, 2).contiguous()
        features = self.mlp(torch.cat([features, pos_t], dim=1).unsqueeze(-1))

        # pooling
        features = features.squeeze(-1).max(-1)[0] if self.mode == "max" else features.squeeze(-1).mean(-1)

        pos = pos.new_zeros((pos.size(0), pos.size(2)))
        return pos, features


class PointNetClassification(Module):
    def __init__(self, num_dim: int = 3, num_channels: int = 0, num_classes: int = 10) -> None:
        super().__init__()
        self.num_dim = num_dim
        self.num_channels = num_channels
        self.num_classes = num_classes

        self.sa1 = PointNetSA(
            num_points=512,
            radius_list=[0.1, 0.2, 0.4],
            samples_list=[32, 64, 128],
            channels=[
                [num_dim + num_channels, 32, 32, 64],
                [num_dim + num_channels, 64, 64, 128],
                [num_dim + num_channels, 64, 96, 128],
            ],
        )
        self.sa2 = PointNetSA(
            num_points=128,
            radius_list=[0.4, 0.8],
            samples_list=[64, 128],
            channels=[
                [64 + 128 + 128 + num_dim, 128, 128, 256],
                [64 + 128 + 128 + num_dim, 128, 196, 256],
            ],
        )
        self.sag = PointNetGlobalSA([256 + 256 + num_dim, 256, 512, 1024])
        self.mlp = MLP([1024, 512, 256, num_classes])  # TODO: add dropout=0.5, norm=None

    def forward(self, pos: Tensor, feats: Optional[Tensor] = None) -> Tensor:
        pos1, feats1, _ = self.sa1(pos, feats)
        pos2, feats2, _ = self.sa2(pos1, feats1)
        _, feats3 = self.sag(pos2, feats2)
        return self.mlp(feats3).log_softmax(dim=-1)


class PointNetSegmentation(Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()

        self.sa1 = PointNetSA(
            num_points=1024,
            radius_list=[0.05, 0.1],
            samples_list=[16, 32],
            channels=[[3 + 3, 16, 16, 32], [3 + 3, 32, 32, 64]],
        )
        self.sa2 = PointNetSA(
            num_points=256,
            radius_list=[0.1, 0.2],
            samples_list=[16, 32],
            channels=[[32 + 64 + 3, 64, 64, 128], [32 + 64 + 3, 64, 96, 128]],
        )
        self.sa3 = PointNetSA(
            num_points=64,
            radius_list=[0.2, 0.4],
            samples_list=[16, 32],
            channels=[[128 + 128 + 3, 128, 196, 256], [128 + 128 + 3, 128, 196, 256]],
        )
        self.sa4 = PointNetSA(
            num_points=16,
            radius_list=[0.4, 0.8],
            samples_list=[16, 32],
            channels=[[256 + 256 + 3, 256, 256, 512], [256 + 256 + 3, 256, 384, 512]],
        )

        self.fp4 = PointNetFP([512 + 512 + 256 + 256, 256, 256])
        self.fp3 = PointNetFP([128 + 128 + 256, 256, 256])
        self.fp2 = PointNetFP([32 + 64 + 256, 256, 128])
        self.fp1 = PointNetFP([128, 128, 128])

        self.head = Sequential(
            Conv1d(128, 128, 1),
            BatchNorm1d(128),
            ReLU(),
            Dropout(0.5),
            Conv1d(128, num_classes, 1),
        )

    def forward(self, pos: Tensor, feats: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        pos1, feats1, _ = self.sa1(pos, feats)
        pos2, feats2, _ = self.sa2(pos1, feats1)
        pos3, feats3, _ = self.sa3(pos2, feats2)
        pos4, feats4, _ = self.sa4(pos3, feats3)

        _, feats3 = self.fp4(pos4, feats4, pos3, feats3)
        _, feats2 = self.fp3(pos3, feats3, pos2, feats2)
        _, feats1 = self.fp2(pos2, feats2, pos1, feats1)
        _, feats = self.fp1(pos1, feats1, pos, None)

        pred = self.head(feats)
        pred = F.log_softmax(pred, dim=1)
        return pred, feats4
