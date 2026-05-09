"""3D rotary position embedding for point-cloud attention.

Introduced in [Utonia: Toward One Encoder for All Point Clouds](https://arxiv.org/abs/2603.03283).
Splits each attention head's channel dimension into three equal chunks (one per
spatial axis) and rotates the query/key vectors by a per-axis sinusoidal phase
indexed by the real-valued coordinate. This injects continuous 3D position
into attention without any learnable parameters.
"""

from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor


class Point3DRoPE(nn.Module):
    r"""3D Rotary Position Embedding for point cloud attention.

    Args:
        head_dim: Channel dimension of a single attention head. Must be divisible by 3.
        base: RoPE frequency base ($\theta$). Smaller values encode finer spatial detail.

    Inputs:
        q: Query tensor of shape $(N, H, D)$ where $D$ is `head_dim`.
        k: Key tensor of the same shape as `q`.
        pos: Real-valued 3D positions of shape $(N, 3)$ corresponding to each token.

    Outputs:
        Tuple `(q_rot, k_rot)` of tensors with the same shapes as `q` and `k`.
    """

    inv_freq: Tensor

    def __init__(self, head_dim: int, base: float = 10.0) -> None:
        if head_dim % 3 != 0:
            raise ValueError(f"head_dim must be divisible by 3 for 3D RoPE, got {head_dim}.")
        super().__init__()
        self.head_dim = head_dim
        self.chunk_dim = head_dim // 3
        self.base = base

        inv_freq = 1.0 / (base ** (torch.arange(0, self.chunk_dim, 2).float() / self.chunk_dim))
        self.register_buffer("inv_freq", inv_freq)

    def _cos_sin(self, pos: Tensor) -> Tuple[Tensor, Tensor]:
        freqs = self.inv_freq.unsqueeze(0)
        emb_x = (pos[:, 0:1] * freqs).repeat(1, 2)
        emb_y = (pos[:, 1:2] * freqs).repeat(1, 2)
        emb_z = (pos[:, 2:3] * freqs).repeat(1, 2)
        emb = torch.cat([emb_x, emb_y, emb_z], dim=-1)
        return emb.cos().unsqueeze(1), emb.sin().unsqueeze(1)

    @staticmethod
    def _rotate_half(x: Tensor) -> Tensor:
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, q: Tensor, k: Tensor, pos: Tensor) -> Tuple[Tensor, Tensor]:
        cos, sin = self._cos_sin(pos)
        q_chunks = torch.split(q, self.chunk_dim, dim=-1)
        k_chunks = torch.split(k, self.chunk_dim, dim=-1)
        cos_chunks = torch.split(cos, self.chunk_dim, dim=-1)
        sin_chunks = torch.split(sin, self.chunk_dim, dim=-1)

        q_out = []
        k_out = []
        for qi, ki, ci, si in zip(q_chunks, k_chunks, cos_chunks, sin_chunks):
            q_out.append(qi * ci + self._rotate_half(qi) * si)
            k_out.append(ki * ci + self._rotate_half(ki) * si)
        return torch.cat(q_out, dim=-1), torch.cat(k_out, dim=-1)
