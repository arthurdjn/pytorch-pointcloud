from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter

from torch_pointcloud.layers.blocks import conv1d_block, linear_block


class TNet(nn.Module):
    def __init__(
        self,
        k: int = 3,
        mlp_channels: Sequence[int] = (64, 128, 1024),
        fc_channels: Sequence[int] = (512, 256),
        global_pool: str = "max",
    ) -> None:
        super().__init__()

        self.k = k
        self.mlp_channels = mlp_channels
        self.fc_channels = fc_channels
        self.global_pool = global_pool

        # Create feature extraction MLPs
        self.mlps = nn.ModuleList()
        in_channels = k
        for channels in mlp_channels:
            self.mlps.append(conv1d_block(in_channels, channels, kernel_size=1, dropout=None))
            in_channels = channels

        # Create fully connected layers
        self.fcs = nn.ModuleList()
        in_channels = mlp_channels[-1]
        for channels in fc_channels:
            self.fcs.append(linear_block(in_channels, channels, dropout=None))
            in_channels = channels

        # Final transformation layer
        self.transform = nn.Linear(fc_channels[-1], k * k)
        nn.init.zeros_(self.transform.weight)
        nn.init.eye_(self.transform.bias.view(k, k))

    def forward(self, x: torch.Tensor, batch_idxs: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1)

        # Feature extraction
        for mlp in self.mlps:
            x = mlp(x)

        # Global pooling
        x = x.squeeze(-1)
        x = scatter(x, batch_idxs, dim=0, reduce=self.global_pool)[0]

        # FC layers
        for i in range(0, len(self.fcs), 2):
            x = F.relu(self.fcs[i + 1](self.fcs[i](x)))

        x = self.transform(x)
        iden = torch.eye(self.k, dtype=x.dtype, device=x.device)
        x = x.view(-1, self.k, self.k) + iden

        return x[batch_idxs]
