"""Farthest point sampling (FPS) as a module returning sampled indices."""

from typing import List, Optional, Union

import torch.nn as nn
from torch import Tensor

from torch_pointcloud.utils.cluster import fps


class FPS(nn.Module):
    r"""Farthest point sampling as a module, returning the sampled indices.

    Args:
        ratio: Fraction of points to keep per sample.
        random_start: Whether farthest point sampling starts from a random point. With the default
            `None`, the start is random in training mode and pinned in eval mode, so eval predictions
            are reproducible across runs.

    Shape:
        - Input: positions of shape $(N, D)$ with an optional batch vector of shape $(N,)$.
        - Output: sampled indices of shape $(M,)$ with $M \approx N \cdot \text{ratio}$.
    """

    def __init__(self, ratio: float, random_start: Optional[bool] = None):
        super().__init__()
        self.ratio = ratio
        self.random_start = random_start

    def forward(
        self,
        x: Tensor,
        batch: Optional[Tensor] = None,
        batch_size: Optional[int] = None,
        ptr: Optional[Union[Tensor, List[int]]] = None,
    ) -> Tensor:
        random_start = self.random_start if self.random_start is not None else self.training
        return fps(x, batch, ratio=self.ratio, random_start=random_start, batch_size=batch_size, ptr=ptr)

    def extra_repr(self) -> str:
        return f"ratio={self.ratio}"
