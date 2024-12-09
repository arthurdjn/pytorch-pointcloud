import itertools
from typing import Sequence

import torch
import torch.nn as nn
from torch_scatter import scatter

from torch_pointcloud.layers.blocks import linear_block


class TNet(nn.Module):
    def __init__(
        self,
        k: int = 3,
        mlp1_dims: Sequence[int] = (64, 128, 1024),
        mlp2_dims: Sequence[int] = (512, 256),
        global_pool: str = "max",
    ) -> None:
        super().__init__()
        self.k = k
        self.global_pool = global_pool

        mlp1_dims = list(mlp1_dims)
        mlp2_dims = list(mlp2_dims)

        blocks = []
        for in_features, out_features in itertools.pairwise([k] + mlp1_dims):
            blocks.append(linear_block(in_features, out_features, dropout=None, order="lan"))
        self.mlp1 = nn.Sequential(*blocks)

        blocks = []
        for in_features, out_features in itertools.pairwise([mlp1_dims[-1]] + mlp2_dims):
            blocks.append(linear_block(in_features, out_features, dropout=None, order="lan"))
        self.mlp2 = nn.Sequential(*blocks)

        self.transform = nn.Linear(mlp2_dims[-1], k * k)
        nn.init.zeros_(self.transform.weight)
        nn.init.eye_(self.transform.bias.view(k, k))

    def forward(self, x: torch.Tensor, batch_idxs: torch.Tensor) -> torch.Tensor:
        x = self.mlp1(x)
        x = scatter(x, batch_idxs, dim=0, reduce=self.global_pool)
        x = self.mlp2(x)

        x = self.transform(x)
        iden = torch.eye(self.k, dtype=x.dtype, device=x.device)
        x = x.view(-1, self.k, self.k) + iden

        return x[batch_idxs]
