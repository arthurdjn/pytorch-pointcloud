import math
import warnings
from pathlib import Path
from typing import Any, List, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.init import kaiming_uniform_
from torch.nn.parameter import Parameter

from torch_pointcloud.utils.config import CACHE_DIR


class KPCNNClassification(nn.Module):
    def __init__(
        self,
        blocks: Any,
        in_features: int,
        out_features: int,
        first_subsampling_dl: float = 0.02,
        conv_radius: float = 2.5,
        num_kernel_points: int = 15,
    ) -> None:
        super().__init__()
        layer = 0
        r = first_subsampling_dl * conv_radius
        in_dim = in_features
        out_dim = out_features
        self.K = num_kernel_points

        # Save all block operations in a list of modules
        self.block_ops = nn.ModuleList()

        # Loop over consecutive blocks
        block_in_layer = 0
        for block_i, block in enumerate(architecture):
            # Check equivariance
            if ("equivariant" in block) and (not out_dim % 3 == 0):
                raise ValueError("Equivariant block but features dimension is not a factor of 3")

            # Detect upsampling block to stop
            if "upsample" in block:
                break

            # Apply the good block function defining tf ops
            self.block_ops.append(block_decider(block, r, in_dim, out_dim, layer, config))

            # Index of block in this layer
            block_in_layer += 1

            # Update dimension of input from output
            if "simple" in block:
                in_dim = out_dim // 2
            else:
                in_dim = out_dim

            # Detect change to a subsampled layer
            if "pool" in block or "strided" in block:
                # Update radius and feature dimension for next layer
                layer += 1
                r *= 2
                out_dim *= 2
                block_in_layer = 0

        # Put average here

        self.head_mlp = UnaryBlock(out_dim, 1024, False, 0)
        self.head_softmax = UnaryBlock(1024, config.num_classes, False, 0, no_relu=True)

    def forward(self, xyz: Tensor, features: Optional[Tensor] = None) -> torch.Tensor:
        x = features if features is not None else xyz.clone().detach()

        # Loop over consecutive blocks
        for block_op in self.block_ops:
            x = block_op(x, batch)

        # Head of network
        x = self.head_mlp(x, batch)
        x = self.head_softmax(x, batch)
        return x
