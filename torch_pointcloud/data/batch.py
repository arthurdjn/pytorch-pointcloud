from dataclasses import dataclass
from typing import List

import torch
from torch import Tensor

from .data import PointCloudData


@dataclass
class PointCloudBatch:
    pos: Tensor
    feats: Tensor



# TODO: Return same type as in the data_list, e.g. can be named tuple, dict, dataclass etc.
def collate(data_list: List[PointCloudData]) -> PointCloudBatch:
    pos = torch.stack([data.pos for data in data_list])
    feats = torch.stack([data.features for data in data_list])
    return PointCloudBatch(pos, feats)
