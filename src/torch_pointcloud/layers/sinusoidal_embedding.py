"""Sinusoidal embedding of continuous scalar indices."""

import math

import torch
from torch import Tensor


def sinusoidal_embedding(indices: Tensor, dim: int) -> Tensor:
    r"""Sinusoidal embedding of continuous indices, sine and cosine interleaved per frequency.

    Args:
        indices: Values to embed, any shape $(\ldots)$.
        dim: Embedding width $D$, even.

    Returns:
        The embeddings, shape $(\ldots, D)$.

    Example:
        ```pycon
        >>> import torch
        >>> from torch_pointcloud.layers import sinusoidal_embedding
        >>> sinusoidal_embedding(torch.zeros(2, 3), dim=4).shape
        torch.Size([2, 3, 4])

        ```
    """
    if dim % 2 != 0:
        raise ValueError(f"`dim` must be even, got {dim}.")

    div_term = torch.exp(torch.arange(0, dim, 2, device=indices.device).float() * (-math.log(10000.0) / dim))
    omegas = indices.unsqueeze(-1).unsqueeze(-1) * div_term.view(1, -1, 1)  # (..., D / 2, 1)
    embeddings = torch.cat([torch.sin(omegas), torch.cos(omegas)], dim=-1)  # (..., D / 2, 2)
    return embeddings.view(*indices.shape, dim)
