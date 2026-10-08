r"""Transformer building blocks for point-cloud backbones and detectors.

`Attention` and `TransformerBlock` are the plain ViT-style multi-head self-attention and pre-norm residual block
over dense token sequences of shape $(B, N, C)$. `TransformerEncoderLayer` and `TransformerDecoderLayer` are the
DETR-style layers with positional embeddings added to the attention inputs, over sequence-first $(S, B, C)$ tokens.
"""

from typing import Any, Callable, Dict, Optional, Union

from torch import Tensor, nn
from torch_geometric.nn import MLP

from torch_pointcloud.layers.act import create_act
from torch_pointcloud.layers.dropouts import DropPath
from torch_pointcloud.layers.norms import create_norm
from torch_pointcloud.utils.types import OptTensor


class Attention(nn.Module):
    r"""Multi-head self-attention over a dense token sequence.

    Computes scaled dot-product attention with a single fused `qkv` projection and an
    output `proj`. An optional additive `mask` is added to the pre-softmax attention
    logits, which supports local / windowed attention (masked positions get a large
    negative bias).

    Args:
        dim: Token dimension $C$. Must be divisible by `num_heads`.
        num_heads: Number of attention heads $h$.
        qkv_bias: Whether the fused query/key/value projection uses a bias.
        qk_scale: Override for the $1/\sqrt{d_\text{head}}$ logit scale. Defaults to
            $d_\text{head}^{-1/2}$ when `None`.
        attn_dropout: Dropout applied to the attention weights.
        proj_dropout: Dropout applied to the output projection.

    Shape:
        - Input: $(B, N, C)$ tokens and an optional `mask` broadcastable to
            $(B, h, N, N)$.
        - Output: $(B, N, C)$.

    Example:
        ```python
        import torch
        from torch_pointcloud.layers import Attention

        attn = Attention(384, num_heads=6)
        x = torch.randn(2, 64, 384)
        y = attn(x)
        print(y.shape)
        ```
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_dropout)

    def forward(self, x: Tensor, mask: OptTensor = None) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn + mask

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerBlock(nn.Module):
    r"""Pre-norm transformer block: residual multi-head attention then a residual MLP.

    Applies $x \leftarrow x + \text{DropPath}(\text{Attn}(\text{Norm}(x)))$ followed by
    $x \leftarrow x + \text{DropPath}(\text{MLP}(\text{Norm}(x)))$. The feed-forward is a
    plain-last `torch_geometric.nn.MLP` of hidden size $\lfloor C \cdot \text{mlp\_ratio}
    \rfloor$, so activation and dropout are configurable through the resolver API.

    Args:
        dim: Token dimension $C$.
        num_heads: Number of attention heads.
        mlp_ratio: Hidden-to-input ratio of the feed-forward MLP.
        qkv_bias: Whether the attention `qkv` projection uses a bias.
        qk_scale: Override for the attention logit scale.
        dropout: Dropout used in the MLP and the attention output projection.
        attn_dropout: Dropout applied to the attention weights.
        drop_path: Stochastic-depth rate for the two residual branches.
        act: Activation for the feed-forward MLP.
        act_kwargs: Extra arguments for the activation.
        norm: Normalization applied before attention and before the MLP.
        norm_kwargs: Extra arguments for the normalization.

    Shape:
        - Input: $(B, N, C)$ tokens and an optional `mask` broadcastable to
            $(B, h, N, N)$.
        - Output: $(B, N, C)$.

    Example:
        ```python
        import torch
        from torch_pointcloud.layers import TransformerBlock

        block = TransformerBlock(384, num_heads=6, drop_path=0.1)
        x = torch.randn(2, 64, 384)
        y = block(x)
        print(y.shape)
        ```
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        drop_path: float = 0.0,
        act: Union[str, Callable, None] = "gelu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = nn.LayerNorm,
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()

        def make_norm() -> nn.Module:
            module = create_norm(norm, dim, **(norm_kwargs or {}))
            if module is None:
                raise ValueError("TransformerBlock requires a normalization layer, got norm=None.")

            return module

        self.norm1 = make_norm()
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_dropout=attn_dropout,
            proj_dropout=dropout,
        )
        self.drop_path = DropPath(drop_path)
        self.norm2 = make_norm()
        self.mlp = MLP(
            [dim, int(dim * mlp_ratio), dim],
            act=act,
            act_kwargs=act_kwargs,
            norm=None,
            dropout=dropout,
            plain_last=True,
        )

    def forward(self, x: Tensor, mask: OptTensor = None) -> Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransformerEncoderLayer(nn.Module):
    r"""DETR-style transformer encoder layer: self-attention with positional embeddings added to the query and key,
    then a feed-forward block, each wrapped in a residual.

    Tokens are sequence-first, $(S, B, C)$, as `nn.MultiheadAttention` expects. With `norm_first` the layer norms
    are applied to the input of each sub-block (pre-norm); otherwise after each residual sum (post-norm).

    Args:
        embed_dim: Token embedding dimension.
        num_heads: Number of attention heads.
        mlp_dim: Hidden width of the feed-forward block.
        dropout: Dropout probability.
        act: Activation type or callable for the feed-forward block.
        act_kwargs: Extra activation arguments.
        norm_first: Whether to normalize the input of each sub-block rather than its residual sum.

    Shape:
        Input: $(S, B, C)$ tokens, an optional $(S, S)$ attention mask, $(B, S)$ key padding mask and $(S, B, C)$ positions
        Output: $(S, B, C)$ tokens
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float,
        *,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm_first: bool = True,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.norm_first = norm_first
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.linear1 = nn.Linear(embed_dim, mlp_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(mlp_dim, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = create_act(act, **(act_kwargs or {})) or nn.ReLU()

    def _self_attention(self, x: Tensor, pos: OptTensor, attn_mask: OptTensor, key_padding_mask: OptTensor) -> Tensor:
        q = k = x if pos is None else x + pos
        return self.self_attn(q, k, value=x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)[0]

    def _feed_forward(self, x: Tensor) -> Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))

    def forward(
        self,
        src: Tensor,
        src_mask: OptTensor = None,
        src_key_padding_mask: OptTensor = None,
        pos: OptTensor = None,
    ) -> Tensor:
        if self.norm_first:
            src = src + self.dropout1(self._self_attention(self.norm1(src), pos, src_mask, src_key_padding_mask))
            return src + self.dropout2(self._feed_forward(self.norm2(src)))

        src = self.norm1(src + self.dropout1(self._self_attention(src, pos, src_mask, src_key_padding_mask)))
        return self.norm2(src + self.dropout2(self._feed_forward(src)))


class TransformerDecoderLayer(nn.Module):
    r"""DETR-style transformer decoder layer: self-attention over the queries, cross-attention onto the memory and a
    feed-forward block, each wrapped in a residual.

    Query positions are added to the self-attention query and key and to the cross-attention query; memory
    positions are added to the cross-attention key. With `pos_in_value` both are added to the attention values as
    well. Tokens are sequence-first, $(S, B, C)$. With `norm_first` the layer norms are applied to the input of each
    sub-block (pre-norm); otherwise after each residual sum (post-norm).

    Args:
        embed_dim: Token embedding dimension.
        num_heads: Number of attention heads.
        mlp_dim: Hidden width of the feed-forward block.
        dropout: Dropout probability.
        act: Activation type or callable for the feed-forward block.
        act_kwargs: Extra activation arguments.
        norm_first: Whether to normalize the input of each sub-block rather than its residual sum.
        pos_in_value: Whether the positional embeddings are also added to the attention values.

    Shape:
        Input: $(T, B, C)$ queries, $(S, B, C)$ memory, $(S, B, C)$ memory positions, $(T, B, C)$ query positions and
            an optional $(B, S)$ memory key padding mask
        Output: $(T, B, C)$ queries
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float,
        *,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm_first: bool = True,
        pos_in_value: bool = False,
    ) -> None:
        super().__init__()
        self.norm_first = norm_first
        self.pos_in_value = pos_in_value
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.linear1 = nn.Linear(embed_dim, mlp_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(mlp_dim, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = create_act(act, **(act_kwargs or {})) or nn.ReLU()

    def _self_attention(self, x: Tensor, query_pos: OptTensor) -> Tensor:
        q = k = x if query_pos is None else x + query_pos
        value = q if self.pos_in_value else x
        return self.self_attn(q, k, value=value)[0]

    def _cross_attention(
        self,
        x: Tensor,
        memory: Tensor,
        pos: OptTensor,
        query_pos: OptTensor,
        memory_key_padding_mask: OptTensor,
    ) -> Tensor:
        query = x if query_pos is None else x + query_pos
        key = memory if pos is None else memory + pos
        value = key if self.pos_in_value else memory
        return self.multihead_attn(query=query, key=key, value=value, key_padding_mask=memory_key_padding_mask)[0]

    def _feed_forward(self, x: Tensor) -> Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        *,
        pos: OptTensor = None,
        query_pos: OptTensor = None,
        memory_key_padding_mask: OptTensor = None,
    ) -> Tensor:
        if self.norm_first:
            tgt = tgt + self.dropout1(self._self_attention(self.norm1(tgt), query_pos))
            tgt = tgt + self.dropout2(
                self._cross_attention(self.norm2(tgt), memory, pos, query_pos, memory_key_padding_mask)
            )
            return tgt + self.dropout3(self._feed_forward(self.norm3(tgt)))

        tgt = self.norm1(tgt + self.dropout1(self._self_attention(tgt, query_pos)))
        tgt = self.norm2(
            tgt + self.dropout2(self._cross_attention(tgt, memory, pos, query_pos, memory_key_padding_mask))
        )
        return self.norm3(tgt + self.dropout3(self._feed_forward(tgt)))
