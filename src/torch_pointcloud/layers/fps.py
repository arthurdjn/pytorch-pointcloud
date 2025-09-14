from typing import TYPE_CHECKING, List, Optional, Union

import torch.nn as nn
from torch import Tensor

from torch_pointcloud.utils.imports import optional_import

if TYPE_CHECKING:
    from torch_cluster import fps

fps, _ = optional_import("torch_cluster", name="fps")


class FPS(nn.Module):
    def __init__(self, ratio: float, random_start: bool = True):
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
        return fps(x, batch, ratio=self.ratio, random_start=self.random_start, batch_size=batch_size, ptr=ptr)

    def extra_repr(self) -> str:
        return f"ratio={self.ratio}"
