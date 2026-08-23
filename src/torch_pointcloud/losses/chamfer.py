"""Chamfer distance between batched point sets."""

from typing import Literal

from torch import Tensor


def chamfer_distance(pred: Tensor, target: Tensor, norm: Literal["l1", "l2"] = "l2") -> Tensor:
    r"""Symmetric Chamfer distance between two batched point sets.

    Set-to-set reconstruction objective introduced for point cloud generation in
    :arxiv: [Fan et al., 2017](https://arxiv.org/abs/1612.00603) and standard for masked point
    modeling pretraining (the SSL pretraining models return `(pred, target)` group coordinates in
    exactly this layout). For each point the squared euclidean distance to its nearest neighbor in
    the other set is computed, then reduced over all points and batches:

    $$\text{CD}_{\ell_2} = \frac{1}{BN} \sum \min_j \lVert p_i - q_j \rVert_2^2
    + \frac{1}{BM} \sum \min_i \lVert p_i - q_j \rVert_2^2$$

    $$\text{CD}_{\ell_1} = \frac{1}{2} \Big( \frac{1}{BN} \sum \min_j \lVert p_i - q_j \rVert_2
    + \frac{1}{BM} \sum \min_i \lVert p_i - q_j \rVert_2 \Big)$$

    The `"l2"` variant sums the two directed means of squared distances (no square root, no
    halving); the `"l1"` variant averages the two directed means of euclidean distances. Both
    follow the reference pretraining convention, so losses are comparable with published values.

    Args:
        pred: Predicted point sets of shape $(B, N, 3)$.
        target: Target point sets of shape $(B, M, 3)$.
        norm: Distance variant, `"l1"` (euclidean) or `"l2"` (squared euclidean).

    Returns:
        Scalar Chamfer distance averaged over all points and batches.

    Shape:
        - Input: $(B, N, 3)$ and $(B, M, 3)$.
        - Output: scalar.

    Example:
        ```python
        import torch
        from torch_pointcloud.losses import chamfer_distance

        pred = torch.randn(64, 32, 3, requires_grad=True)
        target = torch.randn(64, 32, 3)
        loss = chamfer_distance(pred, target, norm="l2")
        loss.backward()
        print(loss.shape)
        ```
    """
    if norm not in ("l1", "l2"):
        raise ValueError(f"`norm` must be 'l1' or 'l2', got {norm!r}.")
    sq_dist = (pred.unsqueeze(2) - target.unsqueeze(1)).pow(2).sum(-1)  # (B, N, M)
    dist_pred = sq_dist.min(2).values  # (B, N)
    dist_target = sq_dist.min(1).values  # (B, M)
    if norm == "l1":
        return (dist_pred.sqrt().mean() + dist_target.sqrt().mean()) / 2
    return dist_pred.mean() + dist_target.mean()
