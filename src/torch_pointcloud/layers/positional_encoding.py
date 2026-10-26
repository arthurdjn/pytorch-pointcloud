"""Fixed positional embeddings of continuous coordinates."""

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class PositionEmbeddingSine(nn.Module):
    r"""Parameter-free sinusoidal positional embedding of continuous coordinates.

    Each of the $n$ coordinate axes is multiplied by `scale` and expanded into $\lfloor d / 2n \rfloor$ frequencies
    $\omega_i = \text{temperature}^{-2i \cdot 2n / d}$, as interleaved $\sin(\omega_i p), \cos(\omega_i p)$ pairs; the
    axes are concatenated and the channels left over when $d$ is not a multiple of $2n$ are zero-padded.

    Args:
        embed_dim: The output embedding dimension $d$.
        spatial_dim: The number of input coordinate dimensions $n$.
        temperature: The frequency base of the geometric progression.
        scale: The factor multiplying the coordinates before the embedding.

    Shape:
        - Input: $(*, n)$.
        - Output: $(*, d)$.

    Example:
        ```python
        import torch
        from torch_pointcloud.layers import PositionEmbeddingSine

        pe = PositionEmbeddingSine(384)
        center = torch.randn(2, 64, 3)
        emb = pe(center)
        print(emb.shape)
        ```
    """

    def __init__(
        self,
        embed_dim: int,
        spatial_dim: int = 3,
        temperature: float = 10000.0,
        scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.spatial_dim = spatial_dim
        self.embed_dim = embed_dim
        self.num_pos_feats = embed_dim // spatial_dim // 2 * 2
        self.temperature = temperature
        self.padding = embed_dim - self.num_pos_feats * spatial_dim
        self.scale = scale

    def forward(self, pos: Tensor) -> Tensor:
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=pos.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="trunc") / self.num_pos_feats)

        pos = pos * self.scale
        pos_divided = pos.unsqueeze(-1) / dim_t
        pos_sin = pos_divided[..., 0::2].sin()
        pos_cos = pos_divided[..., 1::2].cos()
        emb = torch.stack([pos_sin, pos_cos], dim=-1).reshape(*pos.shape[:-1], -1)
        return F.pad(emb, (0, self.padding))
