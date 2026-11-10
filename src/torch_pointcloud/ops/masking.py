"""Random token masks for masked modelling."""

from typing import Optional, Union

import torch
from torch import Tensor


def random_mask(
    batch_size: int,
    num_tokens: int,
    ratio: float,
    device: Optional[Union[str, torch.device]] = None,
) -> Tensor:
    r"""Draw a boolean mask hiding the same number of random tokens in every row.

    Each row masks $\lfloor \text{ratio} \cdot T \rfloor$ tokens, drawn independently per row by sorting uniform noise.

    Args:
        batch_size: Number of rows $B$.
        num_tokens: Number of tokens $T$ per row.
        ratio: Fraction of the tokens to mask, in $[0, 1]$.
        device: Device of the mask.

    Returns:
        The mask, `True` where a token is masked.

    Shape:
        - Output: $(B, T)$.

    Example:
        ```python
        from torch_pointcloud.ops.masking import random_mask

        mask = random_mask(2, 64, 0.6)
        print(mask.shape, mask.sum(dim=1))
        ```
    """
    if not 0.0 <= ratio <= 1.0:
        raise ValueError(f"`ratio` must be in [0, 1], got {ratio}.")

    num_mask = int(ratio * num_tokens)
    noise = torch.rand(batch_size, num_tokens, device=device)
    keep = torch.argsort(noise, dim=1)[:, : num_tokens - num_mask]
    mask = torch.ones(batch_size, num_tokens, dtype=torch.bool, device=device)
    mask.scatter_(1, keep, False)
    return mask
