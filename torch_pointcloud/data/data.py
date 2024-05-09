from dataclasses import dataclass

from torch import Tensor


@dataclass
class PointCloudData:
    pos: Tensor
    features: Tensor
