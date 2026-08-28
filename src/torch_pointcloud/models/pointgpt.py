"""PointGPT classification and generative pretraining models.

{{ paper("2305.11487") }}
"""

from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.nn import MLP

import torch_pointcloud.transforms as T
from torch_pointcloud.datasets.modelnet import MODELNET40_CLASSES
from torch_pointcloud.datasets.scanobjectnn import SCANOBJECTNN_CLASSES
from torch_pointcloud.layers import AdaptivePoolLike, PointPatchEmbed, create_adaptive_pool
from torch_pointcloud.utils.cluster import group
from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.types import OptTensor

from ._base import BaseModel, ClassificationModel
from ._registry import WeightsDict, register_model


def morton_sort(center: Tensor) -> Tensor:
    r"""Greedy nearest-neighbor ("simplified Morton") ordering of patch centers.

    Reproduces the `simplied_morton_sorting` of :arxiv: [PointGPT: Auto-regressively Generative
    Pre-training from Point Clouds](https://arxiv.org/abs/2305.11487), adapted from
    :github: [CGuangyan-BIT/PointGPT](https://github.com/CGuangyan-BIT/PointGPT). Starting from the
    first center, the next center is repeatedly chosen as the nearest not-yet-visited center to the
    last one, giving a space-filling traversal that approximates a Z-order (Morton) curve.

    Args:
        center: Patch centers of shape $(B, G, 3)$.

    Returns:
        A permutation of shape $(B, G)$ indexing the $G$ centers of each sample.

    Shape:
        - Input: $(B, G, 3)$.
        - Output: $(B, G)$.

    Example:
        ```python
        import torch
        from torch_pointcloud.models.pointgpt import morton_sort

        center = torch.randn(2, 64, 3)
        order = morton_sort(center)
        print(order.shape)
        ```
    """
    B, G, _ = center.shape
    dist = torch.cdist(center, center)
    eye = torch.eye(G, dtype=torch.bool, device=center.device)
    dist.masked_fill_(eye.unsqueeze(0), float("inf"))

    order = torch.zeros(B, G, dtype=torch.long, device=center.device)
    visited = torch.zeros(B, G, dtype=torch.bool, device=center.device)
    visited[:, 0] = True
    batch_idx = torch.arange(B, device=center.device)
    last = order[:, 0]
    for i in range(1, G):
        row = dist[batch_idx, last].masked_fill(visited, float("inf"))
        last = row.argmin(dim=-1)
        order[:, i] = last
        visited[batch_idx, last] = True
    return order


class PositionEmbeddingSine(nn.Module):
    r"""Parameter-free sinusoidal positional embedding of continuous coordinates.

    Implements the `PositionEmbeddingCoordsSine` of :arxiv: [PointGPT: Auto-regressively Generative
    Pre-training from Point Clouds](https://arxiv.org/abs/2305.11487), adapted from
    :github: [CGuangyan-BIT/PointGPT](https://github.com/CGuangyan-BIT/PointGPT). Each input
    dimension is scaled by $2 \pi$ and expanded into interleaved sine / cosine features over a
    geometric range of frequencies; unused channels are zero-padded.

    Args:
        spatial_dim: The number of input coordinate dimensions $n$.
        embed_dim: The output embedding dimension $d$.
        temperature: The frequency base of the geometric progression.
        scale: The coordinate scale applied before $2 \pi$ (defaults to $1$).

    Shape:
        - Input: $(*, n)$.
        - Output: $(*, d)$.

    Example:
        ```python
        import torch
        from torch_pointcloud.models.pointgpt import PositionEmbeddingSine

        pe = PositionEmbeddingSine(spatial_dim=3, embed_dim=384)
        center = torch.randn(2, 64, 3)
        emb = pe(center)
        print(emb.shape)
        ```
    """

    def __init__(
        self,
        spatial_dim: int = 3,
        embed_dim: int = 384,
        temperature: float = 10000.0,
        scale: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.spatial_dim = spatial_dim
        self.embed_dim = embed_dim
        self.num_pos_feats = embed_dim // spatial_dim // 2 * 2
        self.temperature = temperature
        self.padding = embed_dim - self.num_pos_feats * spatial_dim
        self.scale = (1.0 if scale is None else scale) * 2 * torch.pi

    def forward(self, pos: Tensor) -> Tensor:
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=pos.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="trunc") / self.num_pos_feats)

        pos = pos * self.scale
        pos_divided = pos.unsqueeze(-1) / dim_t
        pos_sin = pos_divided[..., 0::2].sin()
        pos_cos = pos_divided[..., 1::2].cos()
        emb = torch.stack([pos_sin, pos_cos], dim=-1).reshape(*pos.shape[:-1], -1)
        return F.pad(emb, (0, self.padding))


class PointGPTBlock(nn.Module):
    r"""GPT transformer block of PointGPT (masked multi-head attention then a residual MLP).

    Implements the `Block` of :arxiv: [PointGPT: Auto-regressively Generative Pre-training from Point
    Clouds](https://arxiv.org/abs/2305.11487), adapted from
    :github: [CGuangyan-BIT/PointGPT](https://github.com/CGuangyan-BIT/PointGPT). Unlike the plain
    pre-norm ViT block, the attention residual adds the raw attention output to the normalized input
    ($x \leftarrow \text{Norm}_1(x) + \text{Attn}(\text{Norm}_1(x)))$, then a standard residual MLP
    follows. Self-attention uses `torch.nn.MultiheadAttention` with an additive causal / masking
    `attn_mask`, so the weights are the fused `in_proj` / `out_proj` of that module.

    Args:
        embed_dim: The token dimension $C$.
        num_heads: The number of attention heads.
        mlp_ratio: The hidden-to-input ratio of the feed-forward MLP.
        act: The activation of the feed-forward MLP.
        act_kwargs: Keyword arguments for the activation.

    Shape:
        - Input: $(L, B, C)$ tokens and an additive / boolean `attn_mask` of shape $(L, L)$.
        - Output: $(L, B, C)$.

    Example:
        ```python
        import torch
        from torch_pointcloud.models.pointgpt import PointGPTBlock

        block = PointGPTBlock(embed_dim=384, num_heads=6)
        x = torch.randn(66, 2, 384)
        mask = torch.triu(torch.ones(66, 66, dtype=torch.bool), diagonal=1)
        y = block(x, mask)
        print(y.shape)
        ```
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        act: Union[str, Callable, None] = "gelu",
        act_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.ln_2 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.mlp = MLP(
            [embed_dim, int(embed_dim * mlp_ratio), embed_dim],
            act=act,
            act_kwargs=act_kwargs,
            norm=None,
            plain_last=True,
        )

    def forward(self, x: Tensor, attn_mask: OptTensor = None) -> Tensor:
        x = self.ln_1(x)
        attended, _ = self.attn(x, x, x, attn_mask=attn_mask, need_weights=False)
        x = x + attended
        x = x + self.mlp(self.ln_2(x))
        return x


class PointGPTExtractor(nn.Module):
    r"""Auto-regressive GPT extractor (encoder) of PointGPT.

    Implements the `GPT_extractor` of :arxiv: [PointGPT: Auto-regressively Generative Pre-training from
    Point Clouds](https://arxiv.org/abs/2305.11487), adapted from
    :github: [CGuangyan-BIT/PointGPT](https://github.com/CGuangyan-BIT/PointGPT). A start-of-sequence
    token is prepended to the patch-token sequence, a stack of causally-masked `PointGPTBlock` layers
    consumes the tokens plus their positional embedding, and a final layer normalization produces the
    encoded points. This is the backbone reused for downstream classification.

    Args:
        embed_dim: The token dimension $C$.
        num_heads: The number of attention heads.
        depth: The number of transformer blocks.
        act: The activation of the feed-forward MLPs.
        act_kwargs: Keyword arguments for the activation.

    Shape:
        - Input: $(B, L, C)$ tokens, $(B, L + 1, C)$ positions, and an $(L + 1, L + 1)$ mask.
        - Output: $(B, L + 1, C)$ encoded points.

    Example:
        ```python
        import torch
        from torch_pointcloud.models.pointgpt import PointGPTExtractor

        extractor = PointGPTExtractor(embed_dim=384, num_heads=6, depth=12)
        tokens = torch.randn(2, 65, 384)
        pos = torch.randn(2, 66, 384)
        mask = torch.triu(torch.ones(66, 66, dtype=torch.bool), diagonal=1)
        out = extractor(tokens, pos, mask)
        print(out.shape)
        ```
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        depth: int,
        act: Union[str, Callable, None] = "gelu",
        act_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.sos = nn.Parameter(torch.zeros(embed_dim))
        self.layers = nn.ModuleList(
            [PointGPTBlock(embed_dim, num_heads, act=act, act_kwargs=act_kwargs) for _ in range(depth)]
        )
        self.ln_f = nn.LayerNorm(embed_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.sos)

    def forward(self, x: Tensor, pos: Tensor, attn_mask: Tensor, shift: bool = False) -> Tensor:
        batch = x.size(0)
        x = x.transpose(0, 1)
        pos = pos.transpose(0, 1)

        sos = x.new_ones(1, batch, self.embed_dim) * self.sos
        if shift:
            x = torch.cat([sos, x[:-1]], dim=0)
        else:
            x = torch.cat([sos, x], dim=0)

        for layer in self.layers:
            x = layer(x + pos, attn_mask)
        x = self.ln_f(x)
        return x.transpose(0, 1)


class PointGPTGenerator(nn.Module):
    r"""Auto-regressive GPT generator (decoder) of PointGPT.

    Implements the `GPT_generator` of :arxiv: [PointGPT: Auto-regressively Generative Pre-training from
    Point Clouds](https://arxiv.org/abs/2305.11487), adapted from
    :github: [CGuangyan-BIT/PointGPT](https://github.com/CGuangyan-BIT/PointGPT). A stack of
    causally-masked `PointGPTBlock` layers consumes the encoded points plus a relative positional embedding
    and a per-token head reconstructs the next patch's $M$ centered coordinates.

    Args:
        embed_dim: The token dimension $C$.
        num_heads: The number of attention heads.
        depth: The number of transformer blocks.
        group_size: The number of points $M$ reconstructed per patch.
        act: The activation of the feed-forward MLPs.
        act_kwargs: Keyword arguments for the activation.

    Shape:
        - Input: $(B, L, C)$ encoded points, $(B, L, C)$ positions, and an $(L, L)$ mask.
        - Output: $(B \cdot L, M, 3)$ reconstructed patch coordinates.

    Example:
        ```python
        import torch
        from torch_pointcloud.models.pointgpt import PointGPTGenerator

        generator = PointGPTGenerator(embed_dim=384, num_heads=6, depth=4, group_size=32)
        x = torch.randn(2, 64, 384)
        pos = torch.randn(2, 64, 384)
        mask = torch.triu(torch.ones(64, 64, dtype=torch.bool), diagonal=1)
        out = generator(x, pos, mask)
        print(out.shape)
        ```
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        depth: int,
        group_size: int,
        act: Union[str, Callable, None] = "gelu",
        act_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.group_size = group_size
        self.sos = nn.Parameter(torch.zeros(embed_dim))
        self.layers = nn.ModuleList(
            [PointGPTBlock(embed_dim, num_heads, act=act, act_kwargs=act_kwargs) for _ in range(depth)]
        )
        self.ln_f = nn.LayerNorm(embed_dim)
        self.increase_dim = MLP([embed_dim, 3 * group_size], act=None, norm=None, bias=True, plain_last=True)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.sos)

    def forward(self, x: Tensor, pos: Tensor, attn_mask: Tensor) -> Tensor:
        batch, length, _ = x.shape
        x = x.transpose(0, 1)
        pos = pos.transpose(0, 1)
        for layer in self.layers:
            x = layer(x + pos, attn_mask)

        x = self.ln_f(x)
        rebuilt = self.increase_dim(x).transpose(0, 1).reshape(batch * length, -1, 3)
        return rebuilt


class PointGPTClassification(ClassificationModel):
    r"""PointGPT classification model.

    Implements the finetuning model (`PointTransformer`) of :arxiv: [PointGPT: Auto-regressively
    Generative Pre-training from Point Clouds](https://arxiv.org/abs/2305.11487), adapted from
    :github: [CGuangyan-BIT/PointGPT](https://github.com/CGuangyan-BIT/PointGPT). Patches are
    tokenized with a mini-PointNet, ordered by a greedy nearest-neighbor ("simplified Morton")
    traversal of the centers, and consumed by a causally-masked GPT extractor that prepends a
    start-of-sequence and a class token. The global feature concatenates the class-token output with
    the pooled patch outputs (`global_pool`, max-pool by default), so the head input dimension is $2 \cdot \text{embed\_dim}$.

    Args:
        in_channels: The number of input channels (PointGPT uses coordinates only, so $0$).
        num_classes: The number of output classes.
        embed_dim: The transformer / token-embedding dimension.
        depth: The number of extractor blocks.
        num_heads: The number of attention heads.
        num_group: The number of patches $G$.
        group_size: The number of points $M$ per patch.
        decoder_depth: The number of generator blocks (kept for weight compatibility).
        encoder_local_channels: Hidden widths of the patch embedder's per-point MLP.
        encoder_global_channels: Hidden widths of the patch embedder's per-patch MLP.
        dropout: The dropout rate of the classification head.
        act: The activation of the transformer MLPs.
        act_kwargs: Keyword arguments for the activation.
        head_act: The activation of the classification head.
        global_pool: The pooling over the patch tokens for the global feature ("max" or "mean").
        spatial_dim: The spatial dimension of the input point cloud.

    Shape:
        - Input: $(N, 3)$ and $(N,)$.
        - Output: $(B, \text{num\_classes})$.

    Example:
        ```python
        import torch
        from torch_pointcloud.models.pointgpt import PointGPTClassification

        model = PointGPTClassification(in_channels=0, num_classes=40)
        pos = torch.randn(2048, 3)
        batch = torch.cat([torch.zeros(1024), torch.ones(1024)]).long()
        logits = model(None, pos, batch)
        print(logits.shape)
        ```
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        *,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        num_group: int = 64,
        group_size: int = 32,
        decoder_depth: int = 4,
        encoder_local_channels: Sequence[int] = (128, 256),
        encoder_global_channels: Sequence[int] = (512,),
        dropout: float = 0.5,
        act: Union[str, Callable, None] = "gelu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        head_act: Union[str, Callable, None] = "relu",
        global_pool: AdaptivePoolLike = "max",
        spatial_dim: int = 3,
    ) -> None:
        super().__init__(in_channels=in_channels, num_classes=num_classes)
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.num_group = num_group
        self.group_size = group_size
        self.decoder_depth = decoder_depth
        self.encoder_local_channels = encoder_local_channels
        self.encoder_global_channels = encoder_global_channels
        self.dropout = dropout
        self.act = act
        self.act_kwargs = act_kwargs
        self.head_act = head_act
        self.global_pool = create_adaptive_pool(global_pool)
        self.spatial_dim = spatial_dim

        self.encoder = self.configure_encoder()
        self.pos_embed = self.configure_pos_embed()

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.sos_pos = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.blocks = self.configure_blocks()
        self.generator_blocks = self.configure_generator_blocks()
        self.norm = nn.LayerNorm(embed_dim)  # unused in forward; kept for weight compatibility
        self.cls_norm = nn.LayerNorm(embed_dim)
        self.head = self.configure_head()
        self.reset_parameters()

    def configure_encoder(self) -> PointPatchEmbed:
        """Build the mini-PointNet patch embedder tokenizing each patch."""
        return PointPatchEmbed(
            embed_dim=self.embed_dim,
            in_channels=self.spatial_dim + self.in_channels,
            local_channels=self.encoder_local_channels,
            global_channels=self.encoder_global_channels,
        )

    def configure_pos_embed(self) -> PositionEmbeddingSine:
        """Build the sinusoidal positional embedding of the patch centers."""
        return PositionEmbeddingSine(self.spatial_dim, self.embed_dim, temperature=1.0)

    def configure_blocks(self) -> PointGPTExtractor:
        """Build the causally-masked GPT extractor."""
        return PointGPTExtractor(self.embed_dim, self.num_heads, self.depth, act=self.act, act_kwargs=self.act_kwargs)

    def configure_generator_blocks(self) -> PointGPTGenerator:
        """Build the GPT generator predicting the next patch (unused at finetuning, kept for weight compatibility)."""
        return PointGPTGenerator(
            self.embed_dim,
            self.num_heads,
            self.decoder_depth,
            self.group_size,
            act=self.act,
            act_kwargs=self.act_kwargs,
        )

    @property
    def num_features(self) -> int:
        """Channel count $C$ of the pooled features entering the head."""
        return self.embed_dim * 2

    def configure_head(self) -> nn.Module:
        if self.num_classes == 0:
            return nn.Identity()
        return MLP(
            [self.num_features, 256, 256, self.num_classes],
            act=self.head_act,
            norm="batch_norm",
            dropout=[self.dropout, self.dropout, 0.0],
            bias=True,
            plain_last=True,
        )

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.cls_pos, std=0.02)

    def reset_classifier(self, num_classes: int, global_pool: Optional[AdaptivePoolLike] = None, **kwargs: Any) -> None:
        self.num_classes = num_classes
        if global_pool is not None:
            self.global_pool = create_adaptive_pool(global_pool)
        self.head = self.configure_head()

    def forward_features(self, x: OptTensor, pos: Tensor, batch: Tensor) -> Tensor:
        if x is None:
            neighborhood, center = group(
                pos,
                batch,
                num_group=self.num_group,
                group_size=self.group_size,
                random_start=self.training,
            )
        else:
            neighborhood, center, neighbor_idx = group(
                pos,
                batch,
                num_group=self.num_group,
                group_size=self.group_size,
                random_start=self.training,
                return_indices=True,
            )
            neighborhood = torch.cat([neighborhood, x[neighbor_idx].reshape(*neighborhood.shape[:3], -1)], dim=-1)

        order = morton_sort(center)
        center = torch.gather(center, 1, order.unsqueeze(-1).expand(-1, -1, center.size(-1)))
        neighborhood = torch.gather(
            neighborhood,
            1,
            order.view(*order.shape, 1, 1).expand(-1, -1, neighborhood.size(2), neighborhood.size(3)),
        )

        tokens = self.encoder(neighborhood)
        B, L, _ = tokens.shape

        cls_tokens = self.cls_token.expand(B, -1, -1)
        cls_pos = self.cls_pos.expand(B, -1, -1)
        sos_pos = self.sos_pos.expand(B, -1, -1)
        pos_seq = torch.cat([sos_pos, self.pos_embed(center)], dim=1)

        x_seq = torch.cat([cls_tokens, tokens], dim=1)
        pos_seq = torch.cat([cls_pos, pos_seq], dim=1)

        attn_mask = torch.triu(torch.ones(L + 2, L + 2, dtype=torch.bool, device=tokens.device), diagonal=1)
        encoded = self.blocks(x_seq, pos_seq, attn_mask)
        return self.cls_norm(encoded)

    def forward_head(self, x: Tensor, pre_logits: bool = False) -> Tensor:
        pooled = self.global_pool(x[:, 2:].transpose(1, 2)).squeeze(-1)
        global_feat = torch.cat([x[:, 1], pooled], dim=-1)
        return global_feat if pre_logits else self.head(global_feat)

    def forward(self, x: OptTensor, pos: Tensor, batch: Tensor) -> Tensor:
        x = self.forward_features(x, pos, batch)
        return self.forward_head(x)


class PointGPTGenerativePretraining(BaseModel):
    r"""PointGPT auto-regressive generative pretraining model.

    Implements the pretraining model (`PointGPT` / `GPT_Transformer`) of :arxiv: [PointGPT:
    Auto-regressively Generative Pre-training from Point Clouds](https://arxiv.org/abs/2305.11487),
    adapted from :github: [CGuangyan-BIT/PointGPT](https://github.com/CGuangyan-BIT/PointGPT).

    Patches are tokenized, ordered by a greedy nearest-neighbor ("simplified Morton") traversal, and
    fed to a causally-masked GPT extractor with an additional column mask that randomly hides patches
    beyond the first `keep_attend` tokens (the dual-masking strategy). The generator then predicts the
    next patch from the extractor features and a relative positional embedding. `forward` returns the
    predicted and target patch coordinates for a set-to-set reconstruction objective such as
    `chamfer_distance` from `torch_pointcloud.losses`.

    Args:
        in_channels: The number of input channels ($0$, coordinates only).
        embed_dim: The transformer / token-embedding dimension.
        depth: The number of extractor blocks.
        decoder_depth: The number of generator blocks.
        num_heads: The number of attention heads (shared by the extractor and the generator).
        num_group: The number of patches $G$.
        group_size: The number of points $M$ per patch.
        encoder_local_channels: Hidden widths of the patch embedder's per-point MLP.
        encoder_global_channels: Hidden widths of the patch embedder's per-patch MLP.
        mask_ratio: The fraction of maskable patches hidden by the column mask.
        keep_attend: The number of leading patches never hidden by the column mask.
        act: The activation of the transformer MLPs.
        act_kwargs: Keyword arguments for the activation.
        spatial_dim: The spatial dimension of the input point cloud.

    Shape:
        - Input: $(N, 3)$ and $(N,)$.
        - Output: predicted and target patches, each of shape $(B \cdot G, M, 3)$.

    Example:
        ```python
        import torch
        from torch_pointcloud.models.pointgpt import PointGPTGenerativePretraining

        model = PointGPTGenerativePretraining(in_channels=0)
        pos = torch.randn(2048, 3)
        batch = torch.cat([torch.zeros(1024), torch.ones(1024)]).long()
        pred, target = model(None, pos, batch)
        print(pred.shape, target.shape)
        ```
    """

    keep_attend: int = 10

    def __init__(
        self,
        in_channels: int,
        *,
        embed_dim: int = 384,
        depth: int = 12,
        decoder_depth: int = 4,
        num_heads: int = 6,
        num_group: int = 64,
        group_size: int = 32,
        encoder_local_channels: Sequence[int] = (128, 256),
        encoder_global_channels: Sequence[int] = (512,),
        mask_ratio: float = 0.7,
        keep_attend: int = 10,
        act: Union[str, Callable, None] = "gelu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        spatial_dim: int = 3,
    ) -> None:
        super().__init__(in_channels=in_channels)
        self.embed_dim = embed_dim
        self.depth = depth
        self.decoder_depth = decoder_depth
        self.num_heads = num_heads
        self.num_group = num_group
        self.group_size = group_size
        self.mask_ratio = mask_ratio
        self.keep_attend = keep_attend
        self.act = act
        self.act_kwargs = act_kwargs
        self.spatial_dim = spatial_dim
        self.encoder_local_channels = encoder_local_channels
        self.encoder_global_channels = encoder_global_channels
        self.num_mask = int((num_group - keep_attend) * mask_ratio)

        self.encoder = self.configure_encoder()
        self.pos_embed = self.configure_pos_embed()

        self.sos_pos = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.blocks = self.configure_blocks()
        self.generator_blocks = self.configure_generator_blocks()
        self.norm = nn.LayerNorm(embed_dim)  # unused in forward; kept for weight compatibility

    def configure_encoder(self) -> PointPatchEmbed:
        """Build the mini-PointNet patch embedder tokenizing each patch."""
        return PointPatchEmbed(
            embed_dim=self.embed_dim,
            in_channels=self.spatial_dim + self.in_channels,
            local_channels=self.encoder_local_channels,
            global_channels=self.encoder_global_channels,
        )

    def configure_pos_embed(self) -> PositionEmbeddingSine:
        """Build the sinusoidal positional embedding of the patch centers and relative directions."""
        return PositionEmbeddingSine(self.spatial_dim, self.embed_dim, temperature=1.0)

    def configure_blocks(self) -> PointGPTExtractor:
        """Build the causally-masked GPT extractor."""
        return PointGPTExtractor(self.embed_dim, self.num_heads, self.depth, act=self.act, act_kwargs=self.act_kwargs)

    def configure_generator_blocks(self) -> PointGPTGenerator:
        """Build the GPT generator predicting the next patch from the extractor features."""
        return PointGPTGenerator(
            self.embed_dim,
            self.num_heads,
            self.decoder_depth,
            self.group_size,
            act=self.act,
            act_kwargs=self.act_kwargs,
        )

    def _column_mask(self, device: torch.device) -> Tensor:
        maskable = torch.cat(
            [
                torch.zeros(self.num_group - self.keep_attend - self.num_mask, dtype=torch.bool, device=device),
                torch.ones(self.num_mask, dtype=torch.bool, device=device),
            ]
        )
        maskable = maskable[torch.randperm(maskable.size(0), device=device)]
        return torch.cat([torch.zeros(self.keep_attend, dtype=torch.bool, device=device), maskable])

    def forward(self, x: OptTensor, pos: Tensor, batch: Tensor) -> Tuple[Tensor, Tensor]:
        if x is None:
            neighborhood, center = group(pos, batch, self.num_group, self.group_size, random_start=self.training)
        else:
            neighborhood, center, neighbor_idx = group(
                pos, batch, self.num_group, self.group_size, random_start=self.training, return_indices=True
            )
            neighborhood = torch.cat([neighborhood, x[neighbor_idx].reshape(*neighborhood.shape[:3], -1)], dim=-1)
        order = morton_sort(center)
        center = torch.gather(center, 1, order.unsqueeze(-1).expand(-1, -1, center.size(-1)))
        neighborhood = torch.gather(
            neighborhood, 1, order.view(*order.shape, 1, 1).expand(-1, -1, neighborhood.size(2), neighborhood.size(3))
        )

        tokens = self.encoder(neighborhood)
        B, G, _ = tokens.shape

        relative = center[:, 1:, :] - center[:, :-1, :]
        # Duplicate centers give a zero norm; clamp to the smallest normal number so the
        # degenerate direction becomes 0 instead of NaN while normal inputs are untouched.
        relative = relative / relative.norm(dim=-1, keepdim=True).clamp_min(torch.finfo(relative.dtype).tiny)
        position = torch.cat([center[:, :1, :], relative], dim=1)

        pos_relative = self.pos_embed(position)
        sos_pos = self.sos_pos.expand(B, -1, -1)
        pos_absolute = torch.cat([sos_pos, self.pos_embed(center[:, :-1, :])], dim=1)

        causal = torch.triu(torch.ones(G, G, dtype=torch.bool, device=tokens.device), diagonal=1)
        column = self._column_mask(tokens.device)
        eye = torch.eye(G, dtype=torch.bool, device=tokens.device)
        attn_mask = causal | (column.unsqueeze(0) & ~eye)

        encoded = self.blocks(tokens, pos_absolute, attn_mask, shift=True)
        pred = self.generator_blocks(encoded, pos_relative, attn_mask)
        target = neighborhood[..., : self.spatial_dim].reshape(B * G, self.group_size, self.spatial_dim)
        return pred, target


def _modelnet_transforms(num_samples: int) -> Callable:
    return T.Compose(
        [
            T.Rescale(keys=DataKeys.POS, method="centroid"),
            T.CopyItems(keys=DataKeys.POS, names=DataKeys.ORIGIN_POS),
            T.FarthestPointSample(
                pos_key=DataKeys.POS,
                keys=[DataKeys.NORMAL],
                num_samples=num_samples,
                random_start=False,
                dst_index_key=DataKeys.INDEX,
            ),
        ]
    )


def _scanobjectnn_transforms() -> Callable:
    return T.Compose(
        [
            T.CopyItems(keys=DataKeys.POS, names=DataKeys.ORIGIN_POS),
            T.FarthestPointSample(
                pos_key=DataKeys.POS,
                num_samples=2048,
                random_start=False,
                dst_index_key=DataKeys.INDEX,
            ),
        ]
    )


_SIZE_HPARAMS: Dict[str, Dict[str, Any]] = {
    "s": dict(
        embed_dim=384,
        depth=12,
        num_heads=6,
        decoder_depth=4,
        encoder_local_channels=(128, 256),
        encoder_global_channels=(512,),
    ),
    "b": dict(
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_depth=4,
        encoder_local_channels=(256, 512, 1024),
        encoder_global_channels=(2048,),
    ),
    "l": dict(
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_depth=4,
        encoder_local_channels=(256, 512, 1024),
        encoder_global_channels=(2048,),
    ),
}


def _cls_hparams(size: str, num_classes: int, num_group: int = 64) -> Dict[str, Any]:
    return dict(
        in_channels=0,
        num_classes=num_classes,
        num_group=num_group,
        group_size=32,
        dropout=0.5,
        act="gelu",
        head_act="relu",
        spatial_dim=3,
        **_SIZE_HPARAMS[size],
    )


def _pretrain_hparams(size: str) -> Dict[str, Any]:
    return dict(
        in_channels=0,
        num_group=64,
        group_size=32,
        mask_ratio=0.7,
        keep_attend=10,
        act="gelu",
        spatial_dim=3,
        **_SIZE_HPARAMS[size],
    )


_OA: Dict[str, Dict[str, float]] = {
    "modelnet40": {"s": 93.31, "b": 94.37, "l": 93.88},
    "modelnet40-8k": {"s": 93.76, "b": 94.25, "l": 93.92},
    "scanobjectnn-hardest": {"s": 86.95, "b": 91.92, "l": 93.75},
    "scanobjectnn-objbg": {"s": 91.57, "b": 97.07, "l": 98.45},
    "scanobjectnn-objonly": {"s": 90.71, "b": 95.18, "l": 96.90},
}


def _weights(size: str, checkpoint: str, dataset: str, classes: Optional[Sequence[str]] = None) -> WeightsDict:
    weights = WeightsDict(
        url=f"hf://torch-pointcloud/pointgpt/pointgpt-{size}.{checkpoint}.guangyan-chen.safetensors",
        dataset=dataset,
        author="guangyan-chen",
        license="MIT",
    )
    score = _OA.get(checkpoint, {}).get(size)
    if score is not None:
        weights["metrics"] = {"OA": score}
    if classes is not None:
        weights["classes"] = classes
    return weights


for _size in ("s", "b", "l"):

    @register_model(
        f"pointgpt-{_size}.modelnet40.guangyan-chen",
        task="classification",
        weights=_weights(_size, "modelnet40", dataset="modelnet40", classes=MODELNET40_CLASSES),
        transform=_modelnet_transforms(1024),
        hparams=_cls_hparams(_size, 40, num_group=64),
    )
    def _pointgpt_modelnet40(_size: str = _size, **kwargs: Any) -> PointGPTClassification:
        return PointGPTClassification(**kwargs)

    @register_model(
        f"pointgpt-{_size}.modelnet40-8k.guangyan-chen",
        task="classification",
        weights=_weights(_size, "modelnet40-8k", dataset="modelnet40", classes=MODELNET40_CLASSES),
        transform=_modelnet_transforms(8192),
        hparams=_cls_hparams(_size, 40, num_group=512),
    )
    def _pointgpt_modelnet40_8k(_size: str = _size, **kwargs: Any) -> PointGPTClassification:
        return PointGPTClassification(**kwargs)

    @register_model(
        f"pointgpt-{_size}.scanobjectnn-hardest.guangyan-chen",
        task="classification",
        weights=_weights(_size, "scanobjectnn-hardest", dataset="scanobjectnn-hardest", classes=SCANOBJECTNN_CLASSES),
        transform=_scanobjectnn_transforms(),
        hparams=_cls_hparams(_size, 15, num_group=128),
    )
    def _pointgpt_scanobjectnn_hardest(_size: str = _size, **kwargs: Any) -> PointGPTClassification:
        return PointGPTClassification(**kwargs)

    @register_model(
        f"pointgpt-{_size}.scanobjectnn-objbg.guangyan-chen",
        task="classification",
        weights=_weights(_size, "scanobjectnn-objbg", dataset="scanobjectnn-objbg", classes=SCANOBJECTNN_CLASSES),
        transform=_scanobjectnn_transforms(),
        hparams=_cls_hparams(_size, 15, num_group=128),
    )
    def _pointgpt_scanobjectnn_objbg(_size: str = _size, **kwargs: Any) -> PointGPTClassification:
        return PointGPTClassification(**kwargs)

    @register_model(
        f"pointgpt-{_size}.scanobjectnn-objonly.guangyan-chen",
        task="classification",
        weights=_weights(_size, "scanobjectnn-objonly", dataset="scanobjectnn-objonly", classes=SCANOBJECTNN_CLASSES),
        transform=_scanobjectnn_transforms(),
        hparams=_cls_hparams(_size, 15, num_group=128),
    )
    def _pointgpt_scanobjectnn_objonly(_size: str = _size, **kwargs: Any) -> PointGPTClassification:
        return PointGPTClassification(**kwargs)

    @register_model(
        f"pointgpt-{_size}.pretrain.guangyan-chen",
        task="base",
        weights=_weights(_size, "pretrain", dataset="shapenet55"),
        hparams=_pretrain_hparams(_size),
    )
    def _pointgpt_pretrain(_size: str = _size, **kwargs: Any) -> PointGPTGenerativePretraining:
        return PointGPTGenerativePretraining(**kwargs)
