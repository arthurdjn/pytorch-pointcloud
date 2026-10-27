r"""Superpoint operations on packed point clouds: feature pooling and instance mask targets.

A superpoint is a group of points of one scene (typically a geometric over-segmentation of its mesh). Per-point
superpoint ids are given within each scene, so two scenes of a batch may reuse the same ids; the operations here key
superpoints by (scene, id) and number them consecutively across the batch, scene by scene.
"""

from typing import Tuple

import torch
from torch import Tensor
from torch_geometric.utils import scatter


def superpoint_pool(x: Tensor, superpoint: Tensor, batch: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    r"""Average point features over every superpoint of a packed batch.

    Args:
        x: Point features, shape $(N, C)$.
        superpoint: Superpoint id of every point within its scene, shape $(N,)$; any integer ids.
        batch: Scene index of every point, shape $(N,)$.

    Returns:
        A tuple `(features, index, superpoint_batch)`: the mean feature of every superpoint $(S, C)$, the packed
        superpoint index of every point $(N,)$ in $[0, S)$, and the scene index of every superpoint $(S,)$, sorted.

    Shape:
        - x: $(N, C)$
        - superpoint: $(N,)$
        - batch: $(N,)$
        - output: $(S, C)$, $(N,)$, $(S,)$

    Example:
        ```pycon
        >>> x = torch.tensor([[1.0], [3.0], [5.0], [7.0]])
        >>> features, index, superpoint_batch = superpoint_pool(x, torch.tensor([4, 4, 0, 0]), torch.tensor([0, 0, 0, 1]))
        >>> features.flatten().tolist(), index.tolist(), superpoint_batch.tolist()
        ([5.0, 2.0, 7.0], [1, 1, 0, 2], [0, 0, 1])

        ```
    """
    # One integer key per (scene, id rank): a 1-D unique sorts far faster than the row-wise `unique(dim=1)`.
    key = batch.long() * superpoint.numel() + torch.unique(superpoint, return_inverse=True)[1]
    keys, index = torch.unique(key, return_inverse=True)
    features = scatter(x, index, dim=0, dim_size=keys.numel(), reduce="mean")
    return features, index, keys // superpoint.numel()


def superpoint_instance_masks(
    instance: Tensor,
    label: Tensor,
    superpoint: Tensor,
    superpoint_batch: Tensor,
    *,
    ignore_index: int = -1,
    threshold: float = 0.5,
) -> Tuple[Tensor, Tensor, Tensor]:
    r"""Instance masks over the superpoints of a packed batch, from per-point instance ids.

    Ground-truth instances are the (scene, `instance`) pairs among the points with a non-negative id and a label
    other than `ignore_index`, numbered scene by scene in ascending id order. A superpoint belongs to an instance when
    more than `threshold` of its points do; the fraction counts every point of the superpoint, ignored ones included.
    Any packed point clustering works as `superpoint`, e.g. the voxels of a voxelized scene (its `inverse` map).

    Args:
        instance: Instance id of every point within its scene, shape $(N,)$; negative marks no instance.
        label: Class of every point, shape $(N,)$; each instance must carry a single class.
        superpoint: Packed superpoint index of every point, shape $(N,)$, in $[0, S)$ (see `superpoint_pool`).
        superpoint_batch: Scene index of every superpoint, shape $(S,)$.
        ignore_index: Label of the points that belong to no instance class.
        threshold: Fraction of a superpoint's points an instance must exceed to cover it.

    Returns:
        A tuple `(masks, labels, batch)`: the boolean mask of every instance over the superpoints $(I, S)$, its class
        $(I,)$ and its scene index $(I,)$.

    Shape:
        - instance: $(N,)$
        - label: $(N,)$
        - superpoint: $(N,)$
        - superpoint_batch: $(S,)$
        - output: $(I, S)$, $(I,)$, $(I,)$

    Example:
        ```pycon
        >>> instance = torch.tensor([0, 0, 0, -1, 2, 2])
        >>> label = torch.tensor([3, 3, 3, -1, 1, 1])
        >>> masks, labels, batch = superpoint_instance_masks(
        ...     instance, label, torch.tensor([0, 0, 1, 1, 2, 2]), torch.tensor([0, 0, 1])
        ... )
        >>> masks.tolist(), labels.tolist(), batch.tolist()
        ([[True, False, False], [False, False, True]], [3, 1], [0, 1])

        ```
    """
    valid = (instance >= 0) & (label != ignore_index)
    ids = instance[valid]
    key = superpoint_batch[superpoint[valid]] * ids.numel() + torch.unique(ids, return_inverse=True)[1]
    keys, index = torch.unique(key, return_inverse=True)
    num_instances, num_superpoints = keys.numel(), superpoint_batch.shape[0]
    labels = scatter(label[valid], index, dim=0, dim_size=num_instances, reduce="max")
    counts = torch.zeros(num_instances * num_superpoints, device=instance.device)
    counts.index_add_(0, index * num_superpoints + superpoint[valid], torch.ones_like(index, dtype=counts.dtype))
    sizes = torch.bincount(superpoint, minlength=num_superpoints)
    masks = counts.view(num_instances, num_superpoints) / sizes.clamp(min=1) > threshold
    return masks, labels, keys // ids.numel()
