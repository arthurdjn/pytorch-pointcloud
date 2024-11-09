from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_max, scatter_mean

from torch_pointcloud.layers.convs import Conv1dBlock, LinearBlock


class MLPBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        use_bn: bool = True,
    ):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm1d(out_channels) if use_bn else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)))


class TNet(nn.Module):
    def __init__(
        self,
        k: int = 3,
        mlp_channels: Sequence[int] = (64, 128, 1024),
        fc_channels: Sequence[int] = (512, 256),
        use_bn: bool = True,
    ) -> None:
        super().__init__()

        self.k = k
        self.mlp_channels = mlp_channels
        self.fc_channels = fc_channels

        # Create feature extraction MLPs
        self.mlps = nn.ModuleList()
        in_channels = k
        for channels in mlp_channels:
            self.mlps.append(Conv1dBlock(in_channels, channels))
            in_channels = channels

        # Create fully connected layers
        self.fcs = nn.ModuleList()
        in_channels = mlp_channels[-1]
        for channels in fc_channels:
            self.fcs.append(LinearBlock(in_channels, channels, dropout=None))
            in_channels = channels

        # Final transformation layer
        self.transform = nn.Linear(fc_channels[-1], k * k)
        nn.init.zeros_(self.transform.weight)
        nn.init.eye_(self.transform.bias.view(k, k))

        # self.conv1 = nn.Conv1d(k, 64, 1)
        # self.conv2 = nn.Conv1d(64, 128, 1)
        # self.conv3 = nn.Conv1d(128, 1024, 1)

        # self.fc1 = nn.Linear(1024, 512)
        # self.fc2 = nn.Linear(512, 256)
        # self.fc3 = nn.Linear(256, k * k)

        # self.bn1 = nn.BatchNorm1d(64)
        # self.bn2 = nn.BatchNorm1d(128)
        # self.bn3 = nn.BatchNorm1d(1024)
        # self.bn4 = nn.BatchNorm1d(512)
        # self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x: torch.Tensor, batch_idxs: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1)

        # Feature extraction
        for mlp in self.mlps:
            x = mlp(x)

        # Global pooling
        x = x.squeeze(-1)
        x = scatter_max(x, batch_idxs, dim=0)[0]

        # FC layers
        for i in range(0, len(self.fcs), 2):
            x = F.relu(self.fcs[i + 1](self.fcs[i](x)))

        x = self.transform(x)
        iden = torch.eye(self.k, dtype=x.dtype, device=x.device)
        x = x.view(-1, self.k, self.k) + iden

        return x[batch_idxs]

    # def forward(self, x: torch.Tensor, batch_idxs: torch.Tensor) -> torch.Tensor:
    #     x = x.unsqueeze(-1)  # (N, k, 1)

    #     # Point feature extraction
    #     x = F.relu(self.bn1(self.conv1(x)))
    #     x = F.relu(self.bn2(self.conv2(x)))
    #     x = F.relu(self.bn3(self.conv3(x)))

    #     # Global feature extraction using scatter operations
    #     x = x.squeeze(-1)
    #     x = scatter_max(x, batch_idxs, dim=0)[0]

    #     # MLP for transformation matrix
    #     x = F.relu(self.bn4(self.fc1(x)))
    #     x = F.relu(self.bn5(self.fc2(x)))
    #     x = self.fc3(x)

    #     # Initialize as identity
    #     iden = torch.eye(self.k, dtype=x.dtype, device=x.device)
    #     x = x.view(-1, self.k, self.k) + iden

    #     # Broadcast transformation back to each point
    #     return x[batch_idxs]


class TNet_Original(nn.Module):
    def __init__(self, k: int = 3) -> None:
        super().__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x: torch.Tensor, batch_idxs: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1)  # (N, k, 1)

        # Point feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Global feature extraction using scatter operations
        x = x.squeeze(-1)
        x = scatter_max(x, batch_idxs, dim=0)[0]

        # MLP for transformation matrix
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        # Initialize as identity
        iden = torch.eye(self.k, dtype=x.dtype, device=x.device)
        x = x.view(-1, self.k, self.k) + iden

        # Broadcast transformation back to each point
        return x[batch_idxs]


class PointNet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        input_channels: int = 3,
        use_input_tnet: bool = True,
        use_feature_tnet: bool = True,
    ):
        super().__init__()
        self.use_input_tnet = use_input_tnet
        self.use_feature_tnet = use_feature_tnet

        if use_input_tnet:
            self.input_tnet = TNet(k=input_channels)
        if use_feature_tnet:
            self.feature_tnet = TNet(k=64)

        # Point feature extraction
        self.conv1 = nn.Conv1d(input_channels, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, 1024, 1)

        # Batch normalization layers
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)

        # Classification head
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

        self.bn6 = nn.BatchNorm1d(512)
        self.bn7 = nn.BatchNorm1d(256)

        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x: torch.Tensor, batch_idxs: torch.Tensor) -> torch.Tensor:
        # Input transformation
        if self.use_input_tnet:
            trans = self.input_tnet(x, batch_idxs)
            x = torch.bmm(x.unsqueeze(1), trans).squeeze(1)

        x = x.unsqueeze(-1)  # (N, C, 1)

        # First MLP
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Feature transformation
        if self.use_feature_tnet:
            x = x.squeeze(-1)
            trans = self.feature_tnet(x, batch_idxs)
            x = torch.bmm(x.unsqueeze(1), trans).squeeze(1).unsqueeze(-1)

        # Second MLP
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))

        # Global feature extraction
        x = x.squeeze(-1)
        x = scatter_max(x, batch_idxs, dim=0)[0]

        # Classification MLP
        x = F.relu(self.bn6(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn7(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)

        return x
