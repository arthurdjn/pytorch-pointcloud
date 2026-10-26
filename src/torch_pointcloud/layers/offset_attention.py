"""Offset self-attention block of the Point Cloud Transformer."""

from typing import Any, Callable, Dict, Optional, Union

import torch.nn as nn
from torch import Tensor

from .act import create_act
from .norms import create_norm


class OffsetAttention(nn.Module):
    r"""Offset self-attention over a dense token sequence, from
    :arxiv: [PCT: Point Cloud Transformer](https://arxiv.org/abs/2012.09688).

    The query and key share one projection. The attention map is a softmax over the keys followed by an $L_1$
    normalization over the queries, and the block refines the tokens with the offset between the input and the
    attended values: $y = x + \phi(x - A^\top V)$ where $\phi$ is a linear layer, normalization and activation.

    Args:
        channels: Token dimension $C$.
        qk_channels: Width of the shared query / key projection. Defaults to $C / 4$.
        act: Activation of the offset projection.
        act_kwargs: Extra arguments for the activation.
        norm: Normalization of the offset projection.
        norm_kwargs: Extra arguments for the normalization.

    Shape:
        - Input: $(B, N, C)$ tokens.
        - Output: $(B, N, C)$.

    Example:
        ```python
        import torch
        from torch_pointcloud.layers import OffsetAttention

        attn = OffsetAttention(256)
        x = torch.randn(2, 64, 256)
        y = attn(x)
        print(y.shape)
        ```
    """

    def __init__(
        self,
        channels: int,
        qk_channels: Optional[int] = None,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.qk = nn.Linear(channels, qk_channels or channels // 4, bias=False)
        self.v = nn.Linear(channels, channels)
        self.proj = nn.Linear(channels, channels)
        self.norm = create_norm(norm, channels, **(norm_kwargs or {})) or nn.Identity()
        self.act = create_act(act, **(act_kwargs or {})) or nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        q = self.qk(x)  # (B, N, qk_channels)
        attention = (q @ q.transpose(1, 2)).softmax(dim=-1)  # (B, N, N)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdim=True))
        offset = x - attention.transpose(1, 2) @ self.v(x)
        offset = self.act(self.norm(self.proj(offset).flatten(0, 1))).view_as(x)
        return x + offset
