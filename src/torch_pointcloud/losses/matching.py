r"""Set-prediction matching between predictions and ground-truth objects."""

from typing import Sequence, Tuple

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor


@torch.no_grad()
def hungarian_match(cost: Tensor) -> Tuple[Tensor, Tensor]:
    r"""One-to-one assignment of predictions to targets minimizing the total cost (Hungarian algorithm).

    The matcher of the set-prediction detectors (3DETR, TransFusion): every target is matched to exactly
    one prediction and the sum of the matched costs is minimal. The cost matrix is built by the caller
    (e.g. a weighted sum of a classification, a center-distance and an IoU term) and must be finite.

    Args:
        cost: Pairwise cost of every prediction / target pair, shape $(Q, M)$.

    Returns:
        A tuple `(pred_indices, target_indices)` of long tensors on `cost.device`, each of length
        $\min(Q, M)$, such that `pred_indices[i]` is matched to `target_indices[i]`; both are empty when
        either side is.

    Shape:
        - cost: $(Q, M)$
        - pred_indices: $(\min(Q, M),)$
        - target_indices: $(\min(Q, M),)$

    Example:
        ```pycon
        >>> cost = torch.tensor([[0.9, 0.1], [0.2, 0.8], [0.5, 0.5]])
        >>> pred, target = hungarian_match(cost)
        >>> pred.tolist(), target.tolist()
        ([0, 1], [1, 0])

        ```
    """
    if cost.ndim != 2:
        raise ValueError(f"`cost` must be a (Q, M) matrix, got shape {tuple(cost.shape)}.")

    if cost.numel() == 0:
        empty = torch.zeros((0,), dtype=torch.long, device=cost.device)
        return empty, empty

    if not torch.isfinite(cost).all():
        raise ValueError("`cost` must be finite; clamp or mask the degenerate pairs before matching.")

    row, col = linear_sum_assignment(cost.detach().cpu().numpy())
    return (
        torch.as_tensor(row, dtype=torch.long, device=cost.device),
        torch.as_tensor(col, dtype=torch.long, device=cost.device),
    )


@torch.no_grad()
def hungarian_match_batched(cost: Tensor, num_targets: Sequence[int]) -> Tensor:
    r"""One-to-one assignment of every scene's predictions to its targets, with a single device transfer.

    The batched form of [`hungarian_match`][torch_pointcloud.losses.matching.hungarian_match] for a padded
    cost tensor: scene $i$ uses the first `num_targets[i]` target columns of `cost[i]`. The whole tensor is
    moved to the CPU once and every scene is solved there, so a loop over scenes (and decoder layers, when
    they are stacked along the batch dimension) costs one synchronization instead of one per scene.

    Args:
        cost: Pairwise costs, shape $(N, Q, M_\max)$; the columns beyond a scene's target count are ignored
            and may hold anything finite or not.
        num_targets: Number of valid target columns of each scene, length $N$.

    Returns:
        The matched target index of every prediction, shape $(N, Q)$ long on `cost.device`, $-1$ where the
        prediction is unmatched.

    Shape:
        - cost: $(N, Q, M_\max)$
        - output: $(N, Q)$

    Example:
        ```pycon
        >>> cost = torch.tensor([[[0.9, 0.1], [0.2, 0.8], [0.5, 0.5]]])
        >>> hungarian_match_batched(cost, [2]).tolist()
        [[1, 0, -1]]

        ```
    """
    if cost.ndim != 3:
        raise ValueError(f"`cost` must be a (N, Q, M) tensor, got shape {tuple(cost.shape)}.")
    if len(num_targets) != cost.shape[0]:
        raise ValueError(f"`num_targets` lists {len(num_targets)} scenes for a cost of {cost.shape[0]}.")

    # Solved on the CPU in numpy: the matrices are small, so the Python loop and the transfers dominate.
    num_queries = cost.shape[1]
    cost_np = cost.detach().cpu().numpy()
    assignment = np.full((cost.shape[0], num_queries), -1, dtype=np.int64)
    for i, n in enumerate(num_targets):
        if n == 0 or num_queries == 0:
            continue

        scene = cost_np[i, :, :n]
        if not np.isfinite(scene).all():
            raise ValueError("`cost` must be finite; clamp or mask the degenerate pairs before matching.")

        row, col = linear_sum_assignment(scene)
        assignment[i, row] = col

    return torch.from_numpy(assignment).to(cost.device)
