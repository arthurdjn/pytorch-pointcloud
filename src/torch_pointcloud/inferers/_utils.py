from typing import Any, Dict, List, Optional, Union

import torch
from torch import Tensor


def index_select_dict(data: Dict[str, Any], idx: Tensor, n_points: int) -> Dict[str, Any]:
    """Pick `idx` rows from every tensor in `data` whose first dim equals `n_points`.

    Non-tensor entries and tensors with a different leading dimension (e.g. the
    `batch` index after re-keying, scalar metadata) are passed through untouched.
    """
    out: Dict[str, Any] = {}
    for key, value in data.items():
        if torch.is_tensor(value) and value.size(0) == n_points:
            out[key] = value[idx]
        else:
            out[key] = value
    return out


def split_chunks(n: int, max_size: Optional[int], rng: torch.Generator) -> List[Tensor]:
    r"""Partition `range(n)` into index chunks of at most `max_size` points each.

    When `max_size` is `None` or `n` is within `max_size`, returns a single in-order
    chunk holding every index. Otherwise the indices are permuted with `rng` and
    sliced into $\lceil n / \text{max\_size} \rceil$ chunks. Chunk sizes sum to `n`;
    the last chunk may be smaller than `max_size`.

    Args:
        n: Number of indices to partition.
        max_size: Maximum points per chunk, or `None` for a single unsplit chunk.
        rng: Generator for the permutation. Only consumed when a split is needed.

    Returns:
        List of 1-D `long` index tensors on `rng`'s device. The chunks partition
        `range(n)` exactly: every index appears in one and only one chunk.
    """
    device = rng.device
    if max_size is None:
        return [torch.arange(n, device=device)]
    if n > max_size:
        perm = torch.randperm(n, generator=rng, device=device)
        return [perm[i : i + max_size] for i in range(0, n, max_size)]
    return [torch.arange(n, device=device)]


def gaussian_weights(distances: Tensor, sigma: Union[float, Tensor], eps: float = 1e-12) -> Tensor:
    r"""Gaussian falloff weights $w_i = \exp(-d_i^2 / 2\sigma^2)$.

    Args:
        distances: Per-point distances $d_i$ to a window or block center.
        sigma: Gaussian standard deviation. A scalar applies a single radius to every
            point; a `Tensor` broadcastable against `distances` (one radius per window
            row) applies per-row radii.
        eps: Lower bound on `sigma`, guarding against division by zero.

    Returns:
        Weights with the same shape as `distances`, each in $(0, 1]$.
    """
    safe_sigma = sigma.clamp_min(eps) if isinstance(sigma, Tensor) else max(sigma, eps)
    return torch.exp(-0.5 * (distances / safe_sigma) ** 2)
