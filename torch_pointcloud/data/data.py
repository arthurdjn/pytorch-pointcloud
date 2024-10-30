from dataclasses import dataclass

from torch import Tensor


@dataclass
class PointCloudData:
    pos: Tensor  # TODO: rename to coords
    features: Tensor
