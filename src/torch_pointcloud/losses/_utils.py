r"""Glue between the packed ground-truth batch and the per-scene tensors the detection losses consume."""

from typing import Any, Dict, List, Tuple

import torch
from torch import Tensor

from torch_pointcloud.utils.data import DataKeys


def unbatch_boxes(data: Dict[str, Any], batch_size: int, device: torch.device) -> Tuple[List[Tensor], List[Tensor]]:
    r"""Split the packed `box` $(K, D)$ / zero-based `label` $(K,)$ into per-scene lists by `batch_box`.

    Args:
        data: Ground-truth dict carrying `DataKeys.BOX`, `DataKeys.LABEL` and `DataKeys.BATCH_BOX`.
        batch_size: Number of scenes $B$ (a scene without boxes gets empty tensors).
        device: Device the per-scene tensors are moved to.

    Returns:
        `(boxes_per_scene, labels_per_scene)`, each a length-$B$ list.
    """
    box: Tensor = data[DataKeys.BOX]
    label: Tensor = data[DataKeys.LABEL].long()
    box_batch: Tensor = data[DataKeys.BATCH_BOX]
    boxes_per_scene: List[Tensor] = []
    labels_per_scene: List[Tensor] = []
    for b in range(batch_size):
        mask = box_batch == b
        boxes_per_scene.append(box[mask].to(device))
        labels_per_scene.append(label[mask].to(device))
    return boxes_per_scene, labels_per_scene


def scene_splits(batch: Tensor, batch_size: int) -> Tuple[Tensor, List[int]]:
    r"""Order that groups a packed tensor by scene, with the size of every group.

    Indexing a packed tensor with the returned `order` and splitting it by `counts` gives one block per
    scene, each in the packed order of its rows, after a single synchronization (the counts); a per-scene
    boolean mask would synchronize once per scene.

    Args:
        batch: Per-row scene index, shape $(N,)$.
        batch_size: Number of scenes $B$ (a scene without rows gets an empty block).

    Returns:
        `(order, counts)`: the row permutation $(N,)$ and the length-$B$ list of rows per scene.
    """
    order = torch.argsort(batch, stable=True)
    counts = torch.bincount(batch, minlength=batch_size).tolist()
    return order, counts
