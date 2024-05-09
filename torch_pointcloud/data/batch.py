from dataclasses import dataclass
from typing import List

import torch
from torch import Tensor

from .data import PointCloudData


@dataclass
class PointCloudBatch:
    pos: Tensor
    feats: Tensor


def collate(data_list: List[PointCloudData]) -> PointCloudBatch:
    pos = torch.stack([data.pos for data in data_list])
    feats = torch.stack([data.features for data in data_list])
    return PointCloudBatch(pos, feats)
