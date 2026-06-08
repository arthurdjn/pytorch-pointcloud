r"""Standard pre-norm transformer building blocks for point-cloud backbones.

The `Attention` and `TransformerBlock` here are the plain ViT-style multi-head
self-attention and residual block. They operate on dense token sequences of shape $(B, N, C)$.
"""

from typing import Any, Callable, Dict, Optional, Union

from torch import Tensor, nn
from torch_geometric.nn import MLP

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
