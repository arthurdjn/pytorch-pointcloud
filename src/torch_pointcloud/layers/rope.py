"""3D rotary position embeddings for point-cloud attention.

They rotate the query and key channels of each attention head by phases that grow with the token position along each
spatial axis, so the attention logits depend on relative positions, without any learnable parameter.
"""

from typing import Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor


class Point3DRoPE(nn.Module):
    r"""3D Rotary Position Embedding for point cloud attention.

    Introduced in :arxiv: [Utonia: Toward One Encoder for All Point Clouds](https://arxiv.org/abs/2603.03283). Splits
    each attention head's channel dimension into three equal chunks (one per spatial axis) and rotates the query/key
    vectors by a per-axis sinusoidal phase indexed by the real-valued coordinate.

    Args:
        head_dim: Channel dimension of a single attention head. Must be divisible by 6: the head splits
            into three axis chunks, each of which needs an even size for the half-rotation.
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
        if head_dim % 6 != 0:
            raise ValueError(
                f"head_dim must be divisible by 6 for 3D RoPE (three axis chunks, each even-sized so channels "
                f"rotate in pairs), got {head_dim}."
            )

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


class AxialRoPE(nn.Module):
    r"""Axial 3D rotary position embedding with one frequency band per axis.

    The head channels form adjacent pairs $(2i, 2i + 1)$; the first $F_x$ pairs rotate with the x coordinate, the next
    $F_y$ with y and the last $F_z$ with z, so the head dimension is $2 (F_x + F_y + F_z)$. Pair $j$ of a band of $F$
    pairs turns by the angle $p \, \theta^{-j / (F - 1)}$ for the coordinate $p$: frequencies are log-spaced from $1$
    down to $1 / \theta$. Unlike `Point3DRoPE`, the bands can differ in size.

    Args:
        num_frequencies: Number of channel pairs $(F_x, F_y, F_z)$ rotated by each axis.
        base: Frequency base $\theta$.

    Shape:
        - Input: queries and keys of shape $(N, H, D)$ and positions of shape $(N, 3)$.
        - Output: rotated queries and keys of shape $(N, H, D)$, in the input dtype.

    Example:
        ```python
        import torch
        from torch_pointcloud.layers import AxialRoPE

        rope = AxialRoPE(num_frequencies=(12, 12, 8), base=100.0)
        q, k = torch.randn(2, 128, 6, 64).unbind(0)
        q, k = rope(q, k, torch.randint(0, 64, (128, 3)))
        ```
    """

    inv_freq: Tensor
    axis: Tensor

    def __init__(self, num_frequencies: Sequence[int], base: float) -> None:
        if len(num_frequencies) != 3 or min(num_frequencies) < 1:
            raise ValueError(f"`num_frequencies` must hold three positive counts, got {tuple(num_frequencies)}.")

        super().__init__()
        self.num_frequencies = tuple(num_frequencies)
        self.base = base
        self.head_dim = 2 * sum(self.num_frequencies)

        inv_freq = torch.cat([1.0 / base ** torch.linspace(0, 1, n) for n in self.num_frequencies])
        axis = torch.repeat_interleave(torch.arange(3), torch.tensor(self.num_frequencies))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("axis", axis, persistent=False)

    @staticmethod
    def _rotate(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        pairs = x.float().unflatten(-1, (-1, 2))
        even, odd = pairs[..., 0], pairs[..., 1]
        rotated = torch.stack([even * cos - odd * sin, even * sin + odd * cos], dim=-1)
        return rotated.flatten(-2).type_as(x)

    def forward(self, q: Tensor, k: Tensor, pos: Tensor) -> Tuple[Tensor, Tensor]:
        angle = pos.to(self.inv_freq.dtype)[:, self.axis] * self.inv_freq
        cos, sin = angle.cos().unsqueeze(1), angle.sin().unsqueeze(1)
        return self._rotate(q, cos, sin), self._rotate(k, cos, sin)

    def extra_repr(self) -> str:
        return f"num_frequencies={self.num_frequencies}, base={self.base}"
