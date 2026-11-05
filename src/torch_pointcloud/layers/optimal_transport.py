"""Sinkhorn optimal transport with a dustbin row and column, for partial assignments between two sets."""

import torch
import torch.nn as nn
from torch import Tensor

from torch_pointcloud.utils.types import OptTensor

INF = 1e12


class LogOptimalTransport(nn.Module):
    r"""Sinkhorn optimal transport in log space with a learnable dustbin score, as described in the paper
    :arxiv: [SuperGlue: Learning Feature Matching with Graph Neural Networks](https://arxiv.org/abs/1911.11763).

    The score matrix is padded with a dustbin row and column holding the learnable `alpha`, masked rows and columns
    are sent to $-\infty$, and `num_iterations` Sinkhorn normalizations spread a uniform mass over the valid rows and
    columns. The iterations run in at least float32 and the assignment comes back in the dtype of the scores.

    Args:
        num_iterations: Number of Sinkhorn iterations.

    Shape:
        - Input: $(B, M, N)$ scores and optional $(B, M)$ row and $(B, N)$ column masks (`True` where valid).
        - Output: $(B, M + 1, N + 1)$ log assignment.

    Example:
        ```pycon
        >>> import torch
        >>> from torch_pointcloud.layers import LogOptimalTransport
        >>> LogOptimalTransport(num_iterations=10)(torch.randn(2, 5, 6)).shape
        torch.Size([2, 6, 7])

        ```
    """

    def __init__(self, num_iterations: int) -> None:
        super().__init__()
        self.num_iterations = num_iterations
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def log_sinkhorn_normalization(self, scores: Tensor, log_mu: Tensor, log_nu: Tensor) -> Tensor:
        u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
        for _ in range(self.num_iterations):
            u = log_mu - torch.logsumexp(scores + v.unsqueeze(1), dim=2)
            v = log_nu - torch.logsumexp(scores + u.unsqueeze(2), dim=1)
        return scores + u.unsqueeze(2) + v.unsqueeze(1)

    def forward(self, scores: Tensor, row_masks: OptTensor = None, col_masks: OptTensor = None) -> Tensor:
        dtype = scores.dtype
        scores = scores.to(torch.promote_types(dtype, torch.float32))
        batch_size, num_row, num_col = scores.shape
        device = scores.device
        if row_masks is None:
            row_masks = torch.ones(batch_size, num_row, dtype=torch.bool, device=device)
        if col_masks is None:
            col_masks = torch.ones(batch_size, num_col, dtype=torch.bool, device=device)

        padded_row_masks = torch.zeros(batch_size, num_row + 1, dtype=torch.bool, device=device)
        padded_row_masks[:, :num_row] = ~row_masks
        padded_col_masks = torch.zeros(batch_size, num_col + 1, dtype=torch.bool, device=device)
        padded_col_masks[:, :num_col] = ~col_masks
        padded_score_masks = padded_row_masks.unsqueeze(2) | padded_col_masks.unsqueeze(1)

        alpha = self.alpha.to(scores.dtype)
        padded_col = alpha.expand(batch_size, num_row, 1)
        padded_row = alpha.expand(batch_size, 1, num_col + 1)
        padded_scores = torch.cat([torch.cat([scores, padded_col], dim=-1), padded_row], dim=1)
        padded_scores = padded_scores.masked_fill(padded_score_masks, -INF)

        num_valid_row = row_masks.float().sum(1)
        num_valid_col = col_masks.float().sum(1)
        norm = -torch.log(num_valid_row + num_valid_col)  # (B,)

        log_mu = torch.empty(batch_size, num_row + 1, dtype=scores.dtype, device=device)
        log_mu[:, :num_row] = norm.unsqueeze(1)
        log_mu[:, num_row] = torch.log(num_valid_col) + norm
        log_mu[padded_row_masks] = -INF

        log_nu = torch.empty(batch_size, num_col + 1, dtype=scores.dtype, device=device)
        log_nu[:, :num_col] = norm.unsqueeze(1)
        log_nu[:, num_col] = torch.log(num_valid_row) + norm
        log_nu[padded_col_masks] = -INF

        outputs = self.log_sinkhorn_normalization(padded_scores, log_mu, log_nu)
        return (outputs - norm.unsqueeze(1).unsqueeze(2)).to(dtype)

    def extra_repr(self) -> str:
        return f"num_iterations={self.num_iterations}"
