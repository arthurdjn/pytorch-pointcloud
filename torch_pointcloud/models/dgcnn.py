from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torch_pointcloud.ops import knn


class Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, k: int = 20) -> None:
        super().__init__()
        self.k = k
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x: Tensor, idxs: Optional[Tensor] = None) -> Tensor:
        batch_size = x.size(0)
        num_points = x.size(2)
        x = x.view(batch_size, -1, num_points)

        if idxs is None:
            _, idxs = knn(x, x, k=self.k)

        idxs_base = torch.arange(0, batch_size).view(-1, 1, 1) * num_points
        idxs = idxs + idxs_base.to(idxs.device)
        idxs = idxs.view(-1)
        _, num_dims, _ = x.size()

        x = x.transpose(2, 1).contiguous()
        features = x.view(batch_size * num_points, -1)[idxs, :]
        features = features.view(batch_size, num_points, self.k, num_dims)
        x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, self.k, 1)

        features = torch.cat((features - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

        return self.act(self.bn(self.conv(features)))


class DGCNN(nn.Module):
    def __init__(self, num_classes: int, features_dim: int = 1024, top_k: int = 20, dropout: float = 0.5) -> None:
        super().__init__()

        self.conv1 = Block(6, 64, k=top_k)
        self.conv2 = Block(64 * 2, 64, k=top_k)
        self.conv3 = Block(64 * 2, 128, k=top_k)
        self.conv4 = Block(128 * 2, 256, k=top_k)
        self.conv5 = nn.Sequential(
            nn.Conv1d(512, features_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(features_dim),
            nn.LeakyReLU(negative_slope=0.2),
        )

        # TODO: use MLP instead of sequential etc.
        self.head = nn.Sequential(
            nn.Linear(features_dim * 2, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(p=dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )

    def forward_features(self, x: Tensor) -> Tensor:
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)

        x1 = x1.max(dim=-1, keepdim=False)[0]
        x2 = x2.max(dim=-1, keepdim=False)[0]
        x3 = x3.max(dim=-1, keepdim=False)[0]
        x4 = x4.max(dim=-1, keepdim=False)[0]
        x = torch.cat((x1, x2, x3, x4), dim=1)

        return self.conv5(x)

    def forward_head(self, x: Tensor) -> Tensor:
        x1 = F.adaptive_max_pool1d(x, 1).view(x.size(0), -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(x.size(0), -1)
        x = torch.cat((x1, x2), 1)

        return self.head(x)

    def forward(self, x: Tensor) -> Tensor:
        x = self.forward_features(x)
        return self.forward_head(x)
