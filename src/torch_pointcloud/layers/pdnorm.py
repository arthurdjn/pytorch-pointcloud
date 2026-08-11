from typing import Any, Callable, Optional, Sequence, Union

import torch.nn as nn
from torch import Tensor

from .norms import create_norm


class PDNorm(nn.Module):
    r"""Prompt-driven normalization that routes each batch through a per-condition norm.

    Multi-dataset joint training feeds batches drawn from a single dataset at a time,
    identified by a string `condition`. When `decouple` is `True`, `PDNorm` holds one
    independent inner norm per condition (an `nn.ModuleList` indexed by
    `conditions.index(condition)`), so each dataset keeps its own running statistics and
    affine parameters. When `decouple` is `False`, a single shared norm is applied
    regardless of the condition.

    Inner norms are built with `create_norm`, so any name / class / instance accepted
    there is valid. The decoupled layout stores children under `norm.{i}` keys, matching
    the order of `conditions`.

    Introduced in :arxiv: [Towards Large-scale 3D Representation Learning
    with Multi-dataset Point Prompt Training](https://arxiv.org/abs/2308.09718)

    Args:
        channels: Number of feature channels.
        conditions: Ordered condition names; index $i$ selects `norm.{i}` when decoupled.
        norm: Inner norm passed to `create_norm` (name, class, instance, or `None`).
        decouple: If `True`, use one norm per condition; if `False`, share a single norm.
        dim: Spatial dimensionality hint forwarded to each inner `create_norm` (see its `dim` argument).
        **norm_kwargs: Extra keyword arguments forwarded to each inner norm constructor.

    Shape:
        - Input: $(N, C)$ packed features with $C =$ `channels`.
        - Output: $(N, C)$, same shape as the input.

    Example:
        ```python
        import torch
        from torch_pointcloud.layers import PDNorm

        norm = PDNorm(64, conditions=["ScanNet", "S3DIS"], norm="batch_norm")
        x = torch.randn(32, 64)
        y = norm(x, condition="S3DIS")
        print(y.shape)
        ```
    """

    def __init__(
        self,
        channels: int,
        conditions: Sequence[str],
        norm: Union[str, Callable, None] = "batch_norm",
        decouple: bool = True,
        *,
        dim: int = 1,
        **norm_kwargs: Any,
    ) -> None:
        super().__init__()
        self.conditions = tuple(conditions)
        self.decouple = decouple
        if decouple:
            self.norm: nn.Module = nn.ModuleList(
                [create_norm(norm, channels, dim=dim, **norm_kwargs) or nn.Identity() for _ in self.conditions]
            )
        else:
            self.norm = create_norm(norm, channels, dim=dim, **norm_kwargs) or nn.Identity()

    def forward(self, x: Tensor, condition: Optional[str] = None) -> Tensor:
        if isinstance(self.norm, nn.ModuleList):
            if condition is None:
                options = ", ".join(repr(c) for c in self.conditions)
                raise ValueError(f"PDNorm requires a condition when decoupled. Valid conditions are: {options}.")
            if condition not in self.conditions:
                options = ", ".join(repr(c) for c in self.conditions)
                raise ValueError(f"Unknown condition {condition!r}. Valid conditions are: {options}.")
            return self.norm[self.conditions.index(condition)](x)
        return self.norm(x)
