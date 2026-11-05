r"""Operations on sparse instance masks: pairwise IoU, non-maximum suppression and per-row instance ids.

Masks are given by the (mask, row) pairs they cover, `mask_index` $(2, M)$ with each pair once, over the rows of a
scene or of a packed batch (points or voxels), as a model's `decode` returns them in `InstanceSegmentation3D`. Masks
of different scenes of a batch never share a row, so they never overlap.
"""

import torch
from torch import Tensor


def mask_iou(mask_index: Tensor, num_masks: int) -> Tensor:
    r"""Intersection over union of every pair of sparse masks.

    The intersections are counted from the pairs of masks covering each row: a row covered by $c$ masks gives $c^2$
    pairs, so the cost is $\sum_r c_r^2$. That is a few times $M$ for masks that rarely overlap, such as the
    proposals of disjoint objects, and up to $K^2 N$ when most masks cover most rows, where the dense product of the
    $(K, N)$ masks is faster.

    Args:
        mask_index: Mask and row index of every covered row, shape $(2, M)$, each pair once.
        num_masks: Number of masks $K$.

    Returns:
        The pairwise IoU, shape $(K, K)$ float; $0$ between two empty masks.

    Shape:
        - mask_index: $(2, M)$
        - output: $(K, K)$

    Example:
        ```pycon
        >>> import torch
        >>> from torch_pointcloud.ops.mask import mask_iou
        >>> mask_index = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 2]])
        >>> mask_iou(mask_index, num_masks=2).tolist()
        [[1.0, 0.3333333432674408], [0.3333333432674408, 1.0]]

        ```
    """
    order = torch.argsort(mask_index[1], stable=True)
    mask, row = mask_index[0][order], mask_index[1][order]
    _, counts = torch.unique_consecutive(row, return_counts=True)
    # Every (mask, row) entry pairs with the entries of its row: the c entries of a row give c^2 pairs.
    repeats = counts.repeat_interleave(counts)
    starts = (torch.cumsum(counts, dim=0) - counts).repeat_interleave(counts)
    left = torch.repeat_interleave(torch.arange(mask.numel(), device=mask.device), repeats)
    offset = torch.arange(left.numel(), device=mask.device) - torch.repeat_interleave(
        torch.cumsum(repeats, dim=0) - repeats,
        repeats,
    )
    right = starts[left] + offset
    pairs = torch.bincount(mask[left] * num_masks + mask[right], minlength=num_masks * num_masks)
    intersection = pairs.view(num_masks, num_masks).float()
    sizes = intersection.diagonal()
    union = sizes.unsqueeze(1) + sizes.unsqueeze(0) - intersection
    return intersection / union.clamp(min=1)


def mask_nms(mask_index: Tensor, scores: Tensor, iou_threshold: float) -> Tensor:
    r"""Greedy non-maximum suppression of sparse masks.

    Masks are visited by decreasing score; each kept mask removes the remaining ones whose IoU with it exceeds
    `iou_threshold`. Masks of different scenes of a packed batch never overlap, so one call suppresses every scene
    independently. The pairwise IoU comes from `mask_iou`, cheap when few masks share a row.

    Args:
        mask_index: Mask and row index of every covered row, shape $(2, M)$, each pair once.
        scores: Confidence of every mask, shape $(K,)$.
        iou_threshold: IoU above which the lower-scoring mask is removed.

    Returns:
        The indices of the kept masks, highest score first, shape $(L,)$ long.

    Shape:
        - mask_index: $(2, M)$
        - scores: $(K,)$
        - output: $(L,)$

    Example:
        ```pycon
        >>> import torch
        >>> from torch_pointcloud.ops.mask import mask_nms
        >>> mask_index = torch.tensor([[0, 0, 0, 1, 1, 2], [0, 1, 2, 0, 1, 3]])
        >>> mask_nms(mask_index, torch.tensor([0.5, 0.9, 0.7]), iou_threshold=0.5).tolist()
        [1, 2]

        ```
    """
    if scores.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=scores.device)

    order = torch.argsort(scores, descending=True).cpu()
    overlap = (mask_iou(mask_index, scores.numel()) > iou_threshold).cpu()
    suppressed = torch.zeros(scores.numel(), dtype=torch.bool)
    keep = []
    for i in order.tolist():
        if not suppressed[i]:
            keep.append(i)
            suppressed |= overlap[i]
    return torch.tensor(keep, dtype=torch.long, device=scores.device)


def mask_to_instance(mask_index: Tensor, scores: Tensor, num_rows: int) -> Tensor:
    r"""Instance id of every row: the index of the highest-scoring mask covering it.

    Overlapping masks share their rows this way: a row covered by several masks goes to the one with the highest
    score, ties to the larger index.

    Args:
        mask_index: Mask and row index of every covered row, shape $(2, M)$, each pair once.
        scores: Confidence of every mask, shape $(K,)$.
        num_rows: Number of rows $N$.

    Returns:
        The mask index of every row, $-1$ for rows no mask covers, shape $(N,)$ long.

    Shape:
        - mask_index: $(2, M)$
        - scores: $(K,)$
        - output: $(N,)$

    Example:
        ```pycon
        >>> import torch
        >>> from torch_pointcloud.ops.mask import mask_to_instance
        >>> mask_index = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 2]])
        >>> mask_to_instance(mask_index, torch.tensor([0.9, 0.4]), num_rows=4).tolist()
        [0, 0, 1, -1]

        ```
    """
    covering = torch.full((num_rows,), -1, dtype=torch.long, device=mask_index.device)
    if scores.numel() == 0:
        return covering

    order = torch.argsort(scores, stable=True)
    rank = torch.empty_like(order)
    rank[order] = torch.arange(order.numel(), device=order.device)
    covering = covering.scatter_reduce(0, mask_index[1], rank[mask_index[0]], reduce="amax")
    return torch.where(covering >= 0, order[covering.clamp(min=0)], -1)
