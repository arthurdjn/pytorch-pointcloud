from typing import Literal

import torch
from torch import nn

from ._modules import ModuleLike, get_module

# For classification task


class ClsHead(nn.Module):
    def __init__(self, num_features: int, num_classes: int) -> None:
        super().__init__()


class FCHead(ClsHead):
    def __init__(self, num_features: int, num_classes: int) -> None:
        super().__init__(num_features, num_classes)
        self.fc = nn.Linear(num_features, num_classes)


class MLPHead(ClsHead):
    def __init__(self, num_features: int, num_classes: int, hidden_dims: tuple[int, ...]) -> None:
        super().__init__(num_features, num_classes)
        self.mlp = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Linear(in_dim, out_dim),
                    nn.ReLU(),
                )
                for in_dim, out_dim in zip(hidden_dims[:-1], hidden_dims[1:])
            ],
            nn.Linear(hidden_dims[-1], num_classes),
        )
