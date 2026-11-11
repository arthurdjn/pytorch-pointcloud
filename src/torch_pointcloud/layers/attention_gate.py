"""Additive attention gate scoring skip features against a gating signal."""

from typing import Any, Callable, Dict, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import MLP

from .act import create_act


class AttentionGate(nn.Module):
    r"""Additive attention gate of
    :arxiv: [Attention U-Net: Learning Where to Look for the Pancreas](https://arxiv.org/abs/1804.03999).

    Scores every skip feature from the sum of its projection and the projection of the gating signal, and returns
    the sigmoid gate to multiply the skip features with.

    Args:
        in_channels: Number of channels of the gating signal.
        skip_channels: Number of channels of the skip features.
        channels: Width of the joint projection.
        act: Activation of the joint projection.
        act_kwargs: Extra arguments for the activation.
        norm: Normalization layer.
        norm_kwargs: Extra arguments for the normalization.

    Shape:
        - Input: gating signal $(N, C_\text{in})$ and skip features $(N, C_\text{skip})$.
        - Output: $(N, 1)$ gate in $[0, 1]$.

    Example:
        ```python
        import torch
        from torch_pointcloud.layers import AttentionGate

        gate = AttentionGate(64, 32, 16)
        g = torch.randn(128, 64)
        x = torch.randn(128, 32)
        print(gate(g, x).shape)
        ```
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        channels: int,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.gate = MLP([in_channels, channels], plain_last=False, act=None, norm=norm, norm_kwargs=norm_kwargs)
        self.skip = MLP([skip_channels, channels], plain_last=False, act=None, norm=norm, norm_kwargs=norm_kwargs)
        self.act = create_act(act, **(act_kwargs or {})) or nn.Identity()
        self.psi = MLP([channels, 1], plain_last=False, act=None, norm=norm, norm_kwargs=norm_kwargs)

    def forward(self, g: Tensor, x: Tensor) -> Tensor:
        return torch.sigmoid(self.psi(self.act(self.gate(g) + self.skip(x))))
