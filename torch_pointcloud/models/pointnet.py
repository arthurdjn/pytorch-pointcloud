import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torch_pointcloud.layers.activations import get_act


class Block(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        bias: bool = False,
        act: str = "relu",
    ) -> None:
        super(Block, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, bias=bias)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = get_act(act)

    def forward(self, x: Tensor) -> Tensor:
        return F.relu(self.bn(self.conv(x)))


class PointNet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        in_channels: int = 3,
        features_dim: int = 1024,
        dropout: float = 0.5,
        act: str = "relu",
    ) -> None:
        super().__init__()

        self.backbone = nn.Sequential(
            Block(in_channels, 64, act=act),
            Block(64, 64, act=act),
            Block(64, 64, act=act),
            Block(64, 128, act=act),
            Block(128, features_dim, act=act),
        )

        self.global_pool = nn.AdaptiveMaxPool1d(1)

        self.head = nn.Sequential(
            nn.Linear(features_dim, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes),
        )

    def forward_features(self, x: Tensor) -> Tensor:
        return self.backbone(x)

    def forward_head(self, x: Tensor) -> Tensor:
        x = self.global_pool(x)
        return self.head(x)

    def forward(self, x: Tensor) -> Tensor:
        x = self.forward_features(x)
        return self.forward_head(x)
