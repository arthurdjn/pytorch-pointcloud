r"""Negative log-likelihood of partial assignments between two point sets, with a dustbin row and column for the
points without a match."""

from typing import Optional

import torch
from torch import Tensor

from torch_pointcloud.ops.registration import pairwise_squared_distance


def assignment_negative_log_likelihood(
    log_assignment: Tensor,
    row_points: Tensor,
    col_points: Tensor,
    radius: float,
    row_mask: Optional[Tensor] = None,
    col_mask: Optional[Tensor] = None,
) -> Tensor:
    r"""Negative mean log-likelihood of the ground-truth matches in assignments with a dustbin row and column.

    A row point and a column point match when they are closer than `radius`; a valid point without any match is
    assigned to the dustbin. The points are given in a common frame, the column points moved by the ground truth.

    Args:
        log_assignment: Log assignments, the last row and column being the dustbins, shape $(B, N + 1, M + 1)$.
        row_points: Points of the rows, shape $(B, N, 3)$.
        col_points: Points of the columns, shape $(B, M, 3)$.
        radius: Distance under which two points match.
        row_mask: Valid rows, shape $(B, N)$; every row when `None`.
        col_mask: Valid columns, shape $(B, M)$; every column when `None`.

    Returns:
        The scalar loss, zero when no entry is labeled.

    Example:
        ```pycon
        >>> import torch
        >>> from torch_pointcloud.losses import assignment_negative_log_likelihood
        >>> points = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
        >>> log_assignment = torch.log(torch.full((1, 3, 3), 0.5))
        >>> round(assignment_negative_log_likelihood(log_assignment, points, points, radius=0.1).item(), 4)
        0.6931

        ```
    """
    if row_mask is None:
        row_mask = row_points.new_ones(row_points.shape[:2], dtype=torch.bool)
    if col_mask is None:
        col_mask = col_points.new_ones(col_points.shape[:2], dtype=torch.bool)

    distances = pairwise_squared_distance(row_points, col_points)  # (B, N, M)
    matches = (distances < radius**2) & row_mask.unsqueeze(2) & col_mask.unsqueeze(1)
    labels = torch.zeros_like(log_assignment, dtype=torch.bool)
    labels[:, :-1, :-1] = matches
    labels[:, :-1, -1] = (matches.sum(2) == 0) & row_mask
    labels[:, -1, :-1] = (matches.sum(1) == 0) & col_mask
    if not bool(labels.any()):
        return log_assignment.sum() * 0.0
    return -log_assignment[labels].mean()
