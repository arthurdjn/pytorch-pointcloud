from typing import TYPE_CHECKING, Any, Dict, List

import torch
from torch import Tensor

from torch_pointcloud.utils.imports import _OCNN_AVAILABLE, optional_import
from torch_pointcloud.utils.types import StrEnum

if TYPE_CHECKING:
    import ocnn
    from ocnn.octree import Octree, Points


ocnn, _ = optional_import("ocnn")
Octree, _ = optional_import("ocnn.octree", "Octree")
Points, _ = optional_import("ocnn.octree", "Points")


class DataKeys(StrEnum):
    # General keys (PyG convention)
    X = "x"
    POS = "pos"
    POS_GRID = "pos_grid"
    COLOR = "color"
    NORMAL = "normal"
    FACE = "face"
    SEGMENT = "segment"
    SEMANTIC = "semantic"
    INSTANCE = "instance"
    INTENSITY = "intensity"
    CATEGORY = "category"
    LABEL = "label"
    BATCH = "batch"
    CLUSTER = "cluster"
    # Octree-based keys (OCNN convention)
    OCTREE = "octree"
    POINTS = "points"
    INBOX_MASK = "inbox_mask"


def _tails_equal(tensors: List[Tensor]) -> bool:
    if tensors[0].ndim < 1:
        return False

    tail = tensors[0].shape[1:]
    return all(t.ndim >= 1 and t.shape[1:] == tail for t in tensors)


def _collate_value(values: List[Any]) -> Any:
    first = values[0]

    if isinstance(first, Tensor):
        if first.ndim == 0:
            return torch.stack(values)
        if _tails_equal(values):
            return torch.cat(values, dim=0)
        return list(values)

    if isinstance(first, (bool, int, float)):
        return torch.tensor(values)

    if _OCNN_AVAILABLE and isinstance(first, Points):
        return ocnn.octree.merge_points(values)

    if _OCNN_AVAILABLE and isinstance(first, Octree):
        octree = ocnn.octree.merge_octrees(values)
        octree.construct_all_neigh()
        return octree

    return list(values)


def collate(data_list: List[Dict[str, Any]], batch_from: str = "pos", batch_key: str = "batch") -> Dict[str, Any]:
    if not data_list:
        return {}

    out = {k: _collate_value([d[k] for d in data_list]) for k in data_list[0].keys()}

    if batch_from in data_list[0]:
        lengths = []
        for d in data_list:
            v = d[batch_from]
            if isinstance(v, Tensor):
                lengths.append(v.shape[0] if v.ndim >= 1 else 1)
            elif hasattr(v, "__len__"):
                lengths.append(len(v))
            else:
                lengths.append(1)

        out[batch_key] = torch.cat([torch.full((n,), i, dtype=torch.long) for i, n in enumerate(lengths)])

    return out
