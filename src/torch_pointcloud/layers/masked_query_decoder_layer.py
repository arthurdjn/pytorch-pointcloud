"""Query decoder layer with per-scene masked cross-attention, for mask-predicting set decoders."""

from typing import Callable, Sequence, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import MLP


class MaskedQueryDecoderLayer(nn.Module):
    r"""Query decoder layer: masked cross-attention, self-attention and a feed-forward block, each a residual.

    The queries of each scene attend to that scene's sources (e.g. superpoint features), restricted by an attention
    mask to the sources their current mask prediction covers. Self-attention among the queries of a scene and the
    feed-forward block follow. Every residual is post-normalized, except the cross-attention one when
    `cross_attn_norm=False`.

    Args:
        embed_dim: Query and source embedding dimension.
        num_heads: Number of attention heads.
        mlp_dim: Hidden width of the feed-forward block.
        dropout: Dropout probability.
        act: Activation of the feed-forward block, passed to PyG `MLP`.
        cross_attn_norm: Normalize the cross-attention residual.

    Shape:
        - Input: queries $(B, Q, D)$, the sources of each scene $(S_b, D)$ and the attention masks $(Q, S_b)$
          (`True` blocks a source).
        - Output: queries $(B, Q, D)$.

    Example:
        ```pycon
        >>> import torch
        >>> layer = MaskedQueryDecoderLayer(16, num_heads=2, mlp_dim=32)
        >>> sources = [torch.randn(7, 16), torch.randn(4, 16)]
        >>> attn_masks = [torch.zeros(5, 7, dtype=torch.bool), torch.zeros(5, 4, dtype=torch.bool)]
        >>> layer(torch.randn(2, 5, 16), sources, attn_masks).shape
        torch.Size([2, 5, 16])

        ```
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float = 0.0,
        act: Union[str, Callable, None] = "gelu",
        cross_attn_norm: bool = True,
    ) -> None:
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm0 = nn.LayerNorm(embed_dim) if cross_attn_norm else nn.Identity()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.mlp = MLP([embed_dim, mlp_dim, embed_dim], act=act, norm=None, dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        for param in self.cross_attn.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def forward(self, queries: Tensor, sources: Sequence[Tensor], attn_masks: Sequence[Tensor]) -> Tensor:
        attended = []
        for query, source, attn_mask in zip(queries, sources, attn_masks):
            out = self.cross_attn(query[None], source[None], source[None], attn_mask=attn_mask)[0]
            attended.append(self.norm0(out[0] + query))
        queries = torch.stack(attended)
        queries = self.norm1(self.dropout(self.self_attn(queries, queries, queries)[0]) + queries)
        return self.norm2(self.dropout(self.mlp(queries)) + queries)
