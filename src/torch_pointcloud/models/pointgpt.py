"""PointGPT classification, part-segmentation and generative pretraining models.

{{ paper("2305.11487") }}
"""

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union, overload

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.nn import MLP

import torch_pointcloud.transforms as T
from torch_pointcloud.datasets.modelnet import MODELNET40_CLASSES
from torch_pointcloud.datasets.scanobjectnn import SCANOBJECTNN_CLASSES
from torch_pointcloud.layers import AdaptivePoolLike, PointNet2FeaturePropagation, PointPatchEmbed, create_adaptive_pool
from torch_pointcloud.ops.cluster import group
from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.types import OptTensor

from ._base import ClassificationModel, PartSegmentationModel, PretrainingModel
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
    encoded points. This is the backbone reused for downstream classification and part segmentation;
    `fetch_idx` returns the hidden states after the given blocks instead, before the final normalization.

    Args:
        embed_dim: The token dimension $C$.
        num_heads: The number of attention heads.
        depth: The number of transformer blocks.
        act: The activation of the feed-forward MLPs.
        act_kwargs: Keyword arguments for the activation.

    Shape:
        - Input: $(B, L, C)$ tokens, $(B, L + 1, C)$ positions, and an $(L + 1, L + 1)$ mask.
        - Output: $(B, L + 1, C)$ encoded points, or one $(B, L + 1, C)$ hidden state per index of `fetch_idx`.

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

    @overload
    def forward(
        self, x: Tensor, pos: Tensor, attn_mask: Tensor, shift: bool = ..., *, fetch_idx: Sequence[int]
    ) -> List[Tensor]: ...

    @overload
    def forward(
        self, x: Tensor, pos: Tensor, attn_mask: Tensor, shift: bool = ..., *, fetch_idx: None = None
    ) -> Tensor: ...

    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        attn_mask: Tensor,
        shift: bool = False,
        *,
        fetch_idx: Optional[Sequence[int]] = None,
    ) -> Any:
        batch = x.size(0)
        x = x.transpose(0, 1)
        pos = pos.transpose(0, 1)

        sos = x.new_ones(1, batch, self.embed_dim) * self.sos
        if shift:
            x = torch.cat([sos, x[:-1]], dim=0)
        else:
            x = torch.cat([sos, x], dim=0)

        features: List[Tensor] = []
        for i, layer in enumerate(self.layers):
            x = layer(x + pos, attn_mask)
            if fetch_idx is not None and i in fetch_idx:
                features.append(x.transpose(0, 1))
        if fetch_idx is not None:
            return features
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
        num_groups: The number of patches $G$.
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
        num_groups: int = 64,
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
        self.num_groups = num_groups
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
                num_groups=self.num_groups,
                group_size=self.group_size,
                random_start=self.training,
            )
        else:
            neighborhood, center, neighbor_idx = group(
                pos,
                batch,
                num_groups=self.num_groups,
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


class PointGPTPartSegmentation(PartSegmentationModel):
    r"""PointGPT part-segmentation model.

    Implements the segmentation model (`segmentation/models/pt.py`) of :arxiv: [PointGPT: Auto-regressively
    Generative Pre-training from Point Clouds](https://arxiv.org/abs/2305.11487), adapted from
    :github: [CGuangyan-BIT/PointGPT](https://github.com/CGuangyan-BIT/PointGPT). Patches are tokenized
    with a mini-PointNet, ordered by a greedy nearest-neighbor ("simplified Morton") traversal of the
    centers, and consumed by the causally-masked GPT extractor with a start-of-sequence and a class token.
    The normalized hidden states of three extractor blocks are concatenated per patch and propagated back to
    every point with a PointNet++-style feature propagation (`PointNet2FeaturePropagation`); a
    category-conditioned global branch (max- and mean-pooled patch features plus the embedded category
    one-hot) is fused before the per-point classifier. Every sample in the packed batch must contain the
    same number of points; a ragged batch raises `ValueError`.

    Args:
        in_channels: The number of per-point feature channels concatenated to the coordinates ($0$ for
            coordinates only).
        num_classes: The number of output part classes (across all categories).
        num_categories: The number of object categories for the category one-hot branch.
        embed_dim: The transformer / token-embedding dimension.
        depth: The number of extractor blocks.
        num_heads: The number of attention heads.
        num_groups: The number of patches $G$.
        group_size: The number of points $M$ per patch.
        decoder_depth: The number of generator blocks (kept for weight compatibility).
        encoder_local_channels: Hidden widths of the patch embedder's per-point MLP.
        encoder_global_channels: Hidden widths of the patch embedder's per-patch MLP.
        propagation_channels: Widths of the feature-propagation MLP, the last one being its output width.
        head_channels: Hidden widths of the per-point classifier.
        dropout: The dropout rate after the first layer of the per-point classifier.
        act: The activation of the transformer MLPs.
        act_kwargs: Keyword arguments for the activation.
        spatial_dim: The spatial dimension of the input point cloud.

    Shape:
        - Input: $(N, 3)$, $(N,)$, and a category one-hot $(B, \text{num\_categories})$.
        - Output: $(N, \text{num\_classes})$.

    Example:
        ```python
        import torch
        from torch_pointcloud.models.pointgpt import PointGPTPartSegmentation

        model = PointGPTPartSegmentation(in_channels=0, num_classes=50, num_categories=16)
        pos = torch.randn(4096, 3)
        batch = torch.cat([torch.zeros(2048), torch.ones(2048)]).long()
        category = torch.nn.functional.one_hot(torch.tensor([3, 7]), 16).float()
        logits = model(None, pos, batch, category)
        print(logits.shape)
        ```
    """

    fetch_idx: Tuple[int, int, int] = (3, 7, 11)

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        *,
        num_categories: int = 16,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        num_groups: int = 128,
        group_size: int = 32,
        decoder_depth: int = 4,
        encoder_local_channels: Sequence[int] = (128, 256),
        encoder_global_channels: Sequence[int] = (512,),
        propagation_channels: Sequence[int] = (1536, 1024),
        head_channels: Sequence[int] = (512, 256),
        dropout: float = 0.5,
        act: Union[str, Callable, None] = "gelu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        spatial_dim: int = 3,
    ) -> None:
        super().__init__(in_channels=in_channels, num_classes=num_classes, num_categories=num_categories)
        if max(self.fetch_idx) >= depth:
            raise ValueError(
                f"`fetch_idx` {self.fetch_idx} requires at least {max(self.fetch_idx) + 1} blocks; got depth={depth}."
            )

        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.group_size = group_size
        self.decoder_depth = decoder_depth
        self.encoder_local_channels = encoder_local_channels
        self.encoder_global_channels = encoder_global_channels
        self.propagation_channels = propagation_channels
        self.head_channels = head_channels
        self.dropout = dropout
        self.act = act
        self.act_kwargs = act_kwargs
        self.spatial_dim = spatial_dim

        self.encoder = self.configure_encoder()
        self.pos_embed = self.configure_pos_embed()

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.sos_pos = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.blocks = self.configure_blocks()
        self.generator_blocks = self.configure_generator_blocks()
        self.norm = nn.LayerNorm(embed_dim)

        self.label_conv = self.configure_label_conv()
        self.propagation_0 = self.configure_propagation_0()
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

    def configure_label_conv(self) -> MLP:
        """Build the MLP embedding the category one-hot for the global branch."""
        return MLP(
            [self.num_categories, 64],
            act="leaky_relu",
            act_kwargs=dict(negative_slope=0.2),
            norm="batch_norm",
            bias=False,
            plain_last=False,
        )

    def configure_propagation_0(self) -> PointNet2FeaturePropagation:
        """Build the feature-propagation module interpolating patch features back to every point."""
        return PointNet2FeaturePropagation(
            channels=[3 * self.embed_dim + self.spatial_dim, *self.propagation_channels],
            k=3,
            act="relu",
            norm="batch_norm",
            bias=True,
            plain_last=False,
            weighting="squared",
            eps=1e-8,
        )

    @property
    def num_features(self) -> int:
        """Channel count $C$ of the per-point features entering the head."""
        return self.propagation_channels[-1] + 3 * self.embed_dim * 2 + 64

    def configure_head(self) -> nn.Module:
        if self.num_classes == 0:
            return nn.Identity()
        return MLP(
            [self.num_features, *self.head_channels, self.num_classes],
            act="relu",
            norm="batch_norm",
            dropout=[self.dropout, *(0.0 for _ in self.head_channels)],
            bias=True,
            plain_last=True,
        )

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.cls_pos, std=0.02)

    def reset_classifier(self, num_classes: int, **kwargs: Any) -> None:
        self.num_classes = num_classes
        self.head = self.configure_head()

    def forward_features(self, x: OptTensor, pos: Tensor, batch: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        if x is None:
            neighborhood, center = group(
                pos,
                batch,
                num_groups=self.num_groups,
                group_size=self.group_size,
                random_start=self.training,
            )
        else:
            neighborhood, center, neighbor_idx = group(
                pos,
                batch,
                num_groups=self.num_groups,
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
        feature_list = self.blocks(x_seq, pos_seq, attn_mask, fetch_idx=self.fetch_idx)
        feature_list = [self.norm(feat[:, 2:]).transpose(-1, -2).contiguous() for feat in feature_list]
        x_feat = torch.cat([feature_list[0], feature_list[1], feature_list[2]], dim=1)
        return x_feat, center, batch

    def forward_decoder(self, x_feat: Tensor, pos: Tensor, batch: Tensor, center: Tensor, category: Tensor) -> Tensor:
        B = x_feat.size(0)
        counts = batch.bincount()
        if bool((counts != counts[0]).any()):
            raise ValueError(
                f"{self.__class__.__name__} requires the same number of points per sample, got per-sample "
                f"counts from {int(counts.min())} to {int(counts.max())}."
            )

        N = pos.size(0) // B

        x_max = torch.max(x_feat, 2)[0]
        x_avg = torch.mean(x_feat, 2)
        x_max_feature = x_max.view(B, -1).unsqueeze(-1).repeat(1, 1, N)
        x_avg_feature = x_avg.view(B, -1).unsqueeze(-1).repeat(1, 1, N)
        cls_label_feature = self.label_conv(category).unsqueeze(-1).repeat(1, 1, N)
        x_global_feature = torch.cat([x_max_feature, x_avg_feature, cls_label_feature], 1)

        group_batch = torch.arange(B, device=pos.device).repeat_interleave(self.num_groups)
        x_groups = x_feat.transpose(1, 2).reshape(B * self.num_groups, -1)
        center_pos = center.reshape(B * self.num_groups, self.spatial_dim)
        f_level_0, _, _ = self.propagation_0(x_groups, center_pos, group_batch, pos, pos, batch)
        f_level_0 = f_level_0.reshape(B, N, -1).transpose(1, 2)
        return torch.cat([f_level_0, x_global_feature], 1)

    def forward_head(self, x: Tensor, pre_logits: bool = False) -> Tensor:
        B, _, N = x.shape
        x = x.transpose(1, 2).reshape(B * N, -1)
        return x if pre_logits else self.head(x)

    def forward(self, x: OptTensor, pos: Tensor, batch: Tensor, category: Tensor) -> Tensor:
        x_feat, center, batch = self.forward_features(x, pos, batch)
        x = self.forward_decoder(x_feat, pos, batch, center, category)
        return self.forward_head(x)


class PointGPTPretraining(PretrainingModel):
    r"""PointGPT auto-regressive generative pretraining model.

    Implements the pretraining model (`PointGPT` / `GPT_Transformer`) of :arxiv: [PointGPT:
    Auto-regressively Generative Pre-training from Point Clouds](https://arxiv.org/abs/2305.11487),
    adapted from :github: [CGuangyan-BIT/PointGPT](https://github.com/CGuangyan-BIT/PointGPT).

    Patches are tokenized, ordered by a greedy nearest-neighbor ("simplified Morton") traversal, and
    fed to a causally-masked GPT extractor with an additional column mask that randomly hides patches
    beyond the first `keep_attend` tokens (the dual-masking strategy). The generator then predicts the
    next patch from the extractor features and a relative positional embedding. `forward` returns the
    predicted and target patch coordinates, each packed with the patch index of its points, for a
    set-to-set reconstruction objective such as
    `chamfer_distance` from `torch_pointcloud.losses`.

    Args:
        in_channels: The number of input channels ($0$, coordinates only).
        embed_dim: The transformer / token-embedding dimension.
        depth: The number of extractor blocks.
        decoder_depth: The number of generator blocks.
        num_heads: The number of attention heads (shared by the extractor and the generator).
        num_groups: The number of patches $G$.
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
        - Output: $(B \cdot G \cdot M, 3)$ predicted and target coordinates, each with its $(B \cdot G \cdot M,)$ patch
          index.

    Example:
        ```python
        import torch
        from torch_pointcloud.models.pointgpt import PointGPTPretraining

        model = PointGPTPretraining(in_channels=0)
        pos = torch.randn(2048, 3)
        batch = torch.cat([torch.zeros(1024), torch.ones(1024)]).long()
        pred, target, pred_batch, target_batch = model(None, pos, batch)
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
        num_groups: int = 64,
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
        self.num_groups = num_groups
        self.group_size = group_size
        self.mask_ratio = mask_ratio
        self.keep_attend = keep_attend
        self.act = act
        self.act_kwargs = act_kwargs
        self.spatial_dim = spatial_dim
        self.encoder_local_channels = encoder_local_channels
        self.encoder_global_channels = encoder_global_channels
        self.num_mask = int((num_groups - keep_attend) * mask_ratio)

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
                torch.zeros(self.num_groups - self.keep_attend - self.num_mask, dtype=torch.bool, device=device),
                torch.ones(self.num_mask, dtype=torch.bool, device=device),
            ]
        )
        maskable = maskable[torch.randperm(maskable.size(0), device=device)]
        return torch.cat([torch.zeros(self.keep_attend, dtype=torch.bool, device=device), maskable])

    def forward(self, x: OptTensor, pos: Tensor, batch: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        if x is None:
            neighborhood, center = group(pos, batch, self.num_groups, self.group_size, random_start=self.training)
        else:
            neighborhood, center, neighbor_idx = group(
                pos,
                batch,
                self.num_groups,
                self.group_size,
                random_start=self.training,
                return_indices=True,
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
        pred = self.generator_blocks(encoded, pos_relative, attn_mask).reshape(-1, self.spatial_dim)
        target = neighborhood[..., : self.spatial_dim].reshape(-1, self.spatial_dim)
        batch = torch.arange(B * G, device=pred.device).repeat_interleave(self.group_size)
        return pred, target, batch, batch


def _modelnet_transforms(num_samples: int) -> T.Compose:
    return T.Compose(
        [
            T.Rescale(keys=DataKeys.POS, method="centroid"),
            T.CopyItems(keys=DataKeys.POS, dst_keys=DataKeys.ORIGIN_POS),
            T.FarthestPointSample(
                pos_key=DataKeys.POS,
                keys=[DataKeys.NORMAL],
                num_samples=num_samples,
                random_start=False,
                dst_index_key=DataKeys.INDEX,
            ),
        ]
    )


def _scanobjectnn_transforms() -> T.Compose:
    return T.Compose(
        [
            T.CopyItems(keys=DataKeys.POS, dst_keys=DataKeys.ORIGIN_POS),
            T.FarthestPointSample(
                pos_key=DataKeys.POS,
                num_samples=2048,
                random_start=False,
                dst_index_key=DataKeys.INDEX,
            ),
        ]
    )


def _shapenetpart_transforms() -> T.Compose:
    return T.Compose(
        [
            T.Rescale(keys=DataKeys.POS, method="centroid"),
            T.CopyItems(keys=[DataKeys.POS, DataKeys.SEGMENT], dst_keys=[DataKeys.ORIGIN_POS, DataKeys.ORIGIN_SEGMENT]),
            T.FarthestPointSample(
                pos_key=DataKeys.POS,
                keys=[DataKeys.NORMAL, DataKeys.SEGMENT],
                num_samples=2048,
                random_start=False,
                dst_index_key=DataKeys.INDEX,
            ),
            T.OneHot(keys=DataKeys.CATEGORY, num_classes=16),
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


def _cls_hparams(size: str, num_classes: int, num_groups: int = 64) -> Dict[str, Any]:
    return dict(
        in_channels=0,
        num_classes=num_classes,
        num_groups=num_groups,
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
        num_groups=64,
        group_size=32,
        mask_ratio=0.7,
        keep_attend=10,
        act="gelu",
        spatial_dim=3,
        **_SIZE_HPARAMS[size],
    )


_OA: Dict[str, Dict[str, float]] = {
    "modelnet40": {"s": 93.31, "b": 94.37, "l": 93.88},
    "modelnet40-8k": {"s": 93.72, "b": 94.25, "l": 93.92},
    "scanobjectnn-hardest": {"s": 86.95, "b": 91.92, "l": 93.75},
    "scanobjectnn-objbg": {"s": 91.57, "b": 97.07, "l": 98.45},
    "scanobjectnn-objonly": {"s": 90.71, "b": 95.18, "l": 96.90},
}
_MACC: Dict[str, Dict[str, float]] = {
    "modelnet40": {"s": 90.82, "b": 91.65, "l": 92.05},
    "modelnet40-8k": {"s": 90.38, "b": 90.99, "l": 92.06},
    "scanobjectnn-hardest": {"s": 85.51, "b": 90.80, "l": 93.66},
    "scanobjectnn-objbg": {"s": 89.98, "b": 96.19, "l": 98.19},
    "scanobjectnn-objonly": {"s": 89.57, "b": 94.91, "l": 96.56},
}


_REVISIONS = {
    "pointgpt-b.modelnet40-8k.guangyan-chen": "c6733a50218d08eea8cbb6762e341fdffd9de15c",
    "pointgpt-b.modelnet40.guangyan-chen": "a2d43e7f4908c4e283093b9e915eb96c38e5c7b9",
    "pointgpt-b.pretrain.guangyan-chen": "7390bc297925d3c241653bddaa9143edf20cc2d8",
    "pointgpt-b.scanobjectnn-hardest.guangyan-chen": "ed7090c0e874c4ce1b5eca3ba6f355fea16a4352",
    "pointgpt-b.scanobjectnn-objbg.guangyan-chen": "a7b3e7aef3ba9811c77c47ff55665c89d72478c1",
    "pointgpt-b.scanobjectnn-objonly.guangyan-chen": "5b25ad63177fdd529f6b3973b93d39791e769213",
    "pointgpt-l.modelnet40-8k.guangyan-chen": "7c3dbde3ff917c1d5e1a6c438bb5796d34712a83",
    "pointgpt-l.modelnet40.guangyan-chen": "9e399ab55593cad42e459c15077b51e1b0538e6b",
    "pointgpt-l.pretrain.guangyan-chen": "0e83640c82e6b6e742ee9be4bf657cb064f49efa",
    "pointgpt-l.scanobjectnn-hardest.guangyan-chen": "529649f41803c2814905395f03ad0e4ac3968545",
    "pointgpt-l.scanobjectnn-objbg.guangyan-chen": "6e121e2b249f836723177963c14d374923d56a4c",
    "pointgpt-l.scanobjectnn-objonly.guangyan-chen": "87946ac32f65aa89ca7075844d74c51f3f60a76a",
    "pointgpt-s.modelnet40-8k.guangyan-chen": "55852773fae4adb4e1ddb2fa507c67dd5ced97e9",
    "pointgpt-s.modelnet40.guangyan-chen": "d5d3b80eb846df24051dc753cd9187d2b369f7f7",
    "pointgpt-s.pretrain.guangyan-chen": "60f2989ead639ac920878c2d4cd467f36fbe6584",
    "pointgpt-s.scanobjectnn-hardest.guangyan-chen": "27fb50b626547e92252f9caf61fca7fa39a5a7a3",
    "pointgpt-s.scanobjectnn-objbg.guangyan-chen": "391f078316a4fb24d54d805df280fb86eb6945b6",
    "pointgpt-s.scanobjectnn-objonly.guangyan-chen": "ccfdb72ef5c8bc365cba1b857e2e30551cc56fb3",
}


def _weights(size: str, checkpoint: str, dataset: str, classes: Optional[Sequence[str]] = None) -> WeightsDict:
    name = f"pointgpt-{size}.{checkpoint}.guangyan-chen"
    weights = WeightsDict(
        url=f"hf://torch-pointcloud/{name}/resolve/{_REVISIONS[name]}/model.safetensors",
        dataset=dataset,
        author="guangyan-chen",
        license="MIT",
    )
    score = _OA.get(checkpoint, {}).get(size)
    if score is not None:
        weights["metrics"] = {"OA": score, "mAcc": _MACC[checkpoint][size]}
    if classes is not None:
        weights["classes"] = classes
    return weights


for _size in ("s", "b", "l"):

    @register_model(
        f"pointgpt-{_size}.modelnet40.guangyan-chen",
        task="classification",
        weights=_weights(_size, "modelnet40", dataset="modelnet40", classes=MODELNET40_CLASSES),
        transform=_modelnet_transforms(1024),
        hparams=_cls_hparams(_size, 40, num_groups=64),
    )
    def _pointgpt_modelnet40(_size: str = _size, **kwargs: Any) -> PointGPTClassification:
        return PointGPTClassification(**kwargs)

    @register_model(
        f"pointgpt-{_size}.modelnet40-8k.guangyan-chen",
        task="classification",
        weights=_weights(_size, "modelnet40-8k", dataset="modelnet40", classes=MODELNET40_CLASSES),
        transform=_modelnet_transforms(8192),
        hparams=_cls_hparams(_size, 40, num_groups=512),
    )
    def _pointgpt_modelnet40_8k(_size: str = _size, **kwargs: Any) -> PointGPTClassification:
        return PointGPTClassification(**kwargs)

    @register_model(
        f"pointgpt-{_size}.scanobjectnn-hardest.guangyan-chen",
        task="classification",
        weights=_weights(_size, "scanobjectnn-hardest", dataset="scanobjectnn-hardest", classes=SCANOBJECTNN_CLASSES),
        transform=_scanobjectnn_transforms(),
        hparams=_cls_hparams(_size, 15, num_groups=128),
    )
    def _pointgpt_scanobjectnn_hardest(_size: str = _size, **kwargs: Any) -> PointGPTClassification:
        return PointGPTClassification(**kwargs)

    @register_model(
        f"pointgpt-{_size}.scanobjectnn-objbg.guangyan-chen",
        task="classification",
        weights=_weights(_size, "scanobjectnn-objbg", dataset="scanobjectnn-objbg", classes=SCANOBJECTNN_CLASSES),
        transform=_scanobjectnn_transforms(),
        hparams=_cls_hparams(_size, 15, num_groups=128),
    )
    def _pointgpt_scanobjectnn_objbg(_size: str = _size, **kwargs: Any) -> PointGPTClassification:
        return PointGPTClassification(**kwargs)

    @register_model(
        f"pointgpt-{_size}.scanobjectnn-objonly.guangyan-chen",
        task="classification",
        weights=_weights(_size, "scanobjectnn-objonly", dataset="scanobjectnn-objonly", classes=SCANOBJECTNN_CLASSES),
        transform=_scanobjectnn_transforms(),
        hparams=_cls_hparams(_size, 15, num_groups=128),
    )
    def _pointgpt_scanobjectnn_objonly(_size: str = _size, **kwargs: Any) -> PointGPTClassification:
        return PointGPTClassification(**kwargs)

    @register_model(
        f"pointgpt-{_size}.pretrain.guangyan-chen",
        task="pretraining",
        weights=_weights(_size, "pretrain", dataset="shapenet55"),
        hparams=_pretrain_hparams(_size),
    )
    def _pointgpt_pretrain(_size: str = _size, **kwargs: Any) -> PointGPTPretraining:
        return PointGPTPretraining(**kwargs)


@register_model(
    "pointgpt-s.shapenetpart.guangyan-chen",
    task="part-segmentation",
    input_keys=("x", "pos", "batch", "category"),
    weights=WeightsDict(
        url="hf://torch-pointcloud/pointgpt-s.shapenetpart.guangyan-chen/resolve/0000000000000000000000000000000000000000/model.safetensors",
        dataset="shapenetpart",
        author="guangyan-chen",
        license="MIT",
    ),
    transform=_shapenetpart_transforms(),
    hparams=dict(
        in_channels=0,
        num_classes=50,
        num_categories=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        num_groups=128,
        group_size=32,
        decoder_depth=4,
        encoder_local_channels=(128, 256),
        encoder_global_channels=(512,),
        propagation_channels=(1536, 1024),
        head_channels=(512, 256),
        dropout=0.5,
        act="gelu",
        spatial_dim=3,
    ),
)
def pointgpt_s_shapenetpart_seg(**kwargs: Any) -> PointGPTPartSegmentation:
    return PointGPTPartSegmentation(**kwargs)


@register_model(
    "pointgpt-b.shapenetpart.guangyan-chen",
    task="part-segmentation",
    input_keys=("x", "pos", "batch", "category"),
    weights=WeightsDict(
        url="hf://torch-pointcloud/pointgpt-b.shapenetpart.guangyan-chen/resolve/0000000000000000000000000000000000000000/model.safetensors",
        dataset="shapenetpart",
        author="guangyan-chen",
        license="MIT",
    ),
    transform=_shapenetpart_transforms(),
    hparams=dict(
        in_channels=0,
        num_classes=50,
        num_categories=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        num_groups=128,
        group_size=32,
        decoder_depth=4,
        encoder_local_channels=(256, 512, 1024),
        encoder_global_channels=(2048,),
        propagation_channels=(3072, 2048),
        head_channels=(1024, 512),
        dropout=0.5,
        act="gelu",
        spatial_dim=3,
    ),
)
def pointgpt_b_shapenetpart_seg(**kwargs: Any) -> PointGPTPartSegmentation:
    return PointGPTPartSegmentation(**kwargs)


@register_model(
    "pointgpt-l.shapenetpart.guangyan-chen",
    task="part-segmentation",
    input_keys=("x", "pos", "batch", "category"),
    weights=WeightsDict(
        url="hf://torch-pointcloud/pointgpt-l.shapenetpart.guangyan-chen/resolve/0000000000000000000000000000000000000000/model.safetensors",
        dataset="shapenetpart",
        author="guangyan-chen",
        license="MIT",
    ),
    transform=_shapenetpart_transforms(),
    hparams=dict(
        in_channels=0,
        num_classes=50,
        num_categories=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        num_groups=128,
        group_size=32,
        decoder_depth=4,
        encoder_local_channels=(256, 512, 1024),
        encoder_global_channels=(2048,),
        propagation_channels=(4096, 2048),
        head_channels=(1024, 512),
        dropout=0.5,
        act="gelu",
        spatial_dim=3,
    ),
)
def pointgpt_l_shapenetpart_seg(**kwargs: Any) -> PointGPTPartSegmentation:
    return PointGPTPartSegmentation(**kwargs)


@register_model(
    "pointgpt-b.post-pretrain.guangyan-chen",
    task="classification",
    weights=WeightsDict(
        url="hf://torch-pointcloud/pointgpt-b.post-pretrain.guangyan-chen/resolve/0000000000000000000000000000000000000000/model.safetensors",
        dataset="labeled-hybrid",
        author="guangyan-chen",
        license="MIT",
    ),
    hparams=dict(
        in_channels=0,
        num_classes=87,
        embed_dim=768,
        depth=12,
        num_heads=12,
        num_groups=64,
        group_size=32,
        decoder_depth=4,
        encoder_local_channels=(256, 512, 1024),
        encoder_global_channels=(2048,),
        dropout=0.5,
        act="gelu",
        head_act="relu",
        spatial_dim=3,
    ),
)
def pointgpt_b_post_pretrain(**kwargs: Any) -> PointGPTClassification:
    return PointGPTClassification(**kwargs)


@register_model(
    "pointgpt-l.post-pretrain.guangyan-chen",
    task="classification",
    weights=WeightsDict(
        url="hf://torch-pointcloud/pointgpt-l.post-pretrain.guangyan-chen/resolve/0000000000000000000000000000000000000000/model.safetensors",
        dataset="labeled-hybrid",
        author="guangyan-chen",
        license="MIT",
    ),
    hparams=dict(
        in_channels=0,
        num_classes=87,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        num_groups=64,
        group_size=32,
        decoder_depth=4,
        encoder_local_channels=(256, 512, 1024),
        encoder_global_channels=(2048,),
        dropout=0.5,
        act="gelu",
        head_act="relu",
        spatial_dim=3,
    ),
)
def pointgpt_l_post_pretrain(**kwargs: Any) -> PointGPTClassification:
    return PointGPTClassification(**kwargs)
