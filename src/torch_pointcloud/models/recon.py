"""ReCon classification, part-segmentation and pretraining models.

{{ paper("2302.02318") }}
"""

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union, overload

import torch
from torch import Tensor, nn
from torch_geometric.nn import MLP

import torch_pointcloud.transforms as T
from torch_pointcloud.datasets.modelnet import MODELNET40_CLASSES
from torch_pointcloud.datasets.scanobjectnn import SCANOBJECTNN_CLASSES
from torch_pointcloud.layers import Attention, CrossAttention, DropPath, PartSegmentationDecoder, PointPatchEmbed
from torch_pointcloud.layers.norms import create_norm
from torch_pointcloud.ops.cluster import group
from torch_pointcloud.ops.masking import random_mask
from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.types import OptTensor

from ._base import ClassificationModel, PartSegmentationModel, PretrainingModel
from ._registry import WeightsDict, register_model
from .point_mae import TransformerDecoder


class ReConEncoderBlock(nn.Module):
    r"""Transformer block with a parallel global-query branch, as in :arxiv: [Contrast with Reconstruct: Contrastive 3D
    Representation Learning Guided by Generative Pretraining](https://arxiv.org/abs/2302.02318), adapted from
    :github: [qizekun/ReCon](https://github.com/qizekun/ReCon).

    The first `num_queries` tokens of the sequence are global queries. They cross-attend to the local tokens through
    their own normalization, attention and MLP, while the local tokens run a plain self-attention block and never see
    the queries. Both branches normalize before each sub-block and carry the normalized activations in their residual
    stream. With `detach_tokens`, the local tokens enter the query branch without gradient.

    Args:
        dim: Token dimension $C$.
        num_heads: Number of attention heads.
        mlp_ratio: Hidden-to-input ratio of the feed-forward MLPs.
        qkv_bias: Whether the attention projections use a bias.
        qk_scale: Override for the attention logit scale.
        dropout: Dropout used in the MLPs and the attention output projections.
        attn_dropout: Dropout applied to the attention weights.
        drop_path: Stochastic-depth rate of the residual branches.
        num_queries: Number of global query tokens leading the sequence.
        detach_tokens: Whether the local tokens are detached before the query cross-attention.
        act: Activation of the feed-forward MLPs.
        act_kwargs: Extra arguments for the activation.
        norm: Normalization applied before each sub-block.
        norm_kwargs: Extra arguments for the normalization.

    Shape:
        - Input: $(B, Q + G, C)$ tokens, the $Q$ queries first.
        - Output: $(B, Q + G, C)$.

    Example:
        ```python
        import torch
        from torch_pointcloud.models.recon import ReConEncoderBlock

        block = ReConEncoderBlock(384, num_heads=6, num_queries=3)
        x = torch.randn(2, 3 + 64, 384)
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
        num_queries: int = 3,
        detach_tokens: bool = False,
        act: Union[str, Callable, None] = "gelu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = nn.LayerNorm,
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.num_queries = num_queries
        self.detach_tokens = detach_tokens

        def make_norm() -> nn.Module:
            module = create_norm(norm, dim, **(norm_kwargs or {}))
            if module is None:
                raise ValueError("ReConEncoderBlock requires a normalization layer, got norm=None.")

            return module

        def make_mlp() -> MLP:
            return MLP(
                [dim, int(dim * mlp_ratio), dim],
                act=act,
                act_kwargs=act_kwargs,
                norm=None,
                dropout=dropout,
                plain_last=True,
            )

        attn_kwargs: Dict[str, Any] = dict(
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_dropout=attn_dropout, proj_dropout=dropout
        )
        self.norm1 = make_norm()
        self.attn = Attention(dim, **attn_kwargs)
        self.norm2 = make_norm()
        self.mlp = make_mlp()
        self.query_norm1 = make_norm()
        self.query_attn = CrossAttention(dim, **attn_kwargs)
        self.query_norm2 = make_norm()
        self.query_mlp = make_mlp()
        self.drop_path = DropPath(drop_path)

    def forward(self, x: Tensor) -> Tensor:
        queries, tokens = x[:, : self.num_queries], x[:, self.num_queries :]

        queries = self.query_norm1(queries)
        tokens = self.norm1(tokens)
        keys = tokens.detach() if self.detach_tokens else tokens
        queries = queries + self.drop_path(self.query_attn(queries, keys))
        tokens = tokens + self.drop_path(self.attn(tokens))

        queries = self.query_norm2(queries)
        tokens = self.norm2(tokens)
        queries = queries + self.drop_path(self.query_mlp(queries))
        tokens = tokens + self.drop_path(self.mlp(tokens))
        return torch.cat([queries, tokens], dim=1)


class ReConEncoder(nn.Module):
    r"""Plain transformer encoder of `ReConEncoderBlock`s, as in :arxiv: [Contrast with Reconstruct: Contrastive 3D
    Representation Learning Guided by Generative Pretraining](https://arxiv.org/abs/2302.02318), adapted from
    :github: [qizekun/ReCon](https://github.com/qizekun/ReCon).

    The positional embedding is added to the tokens before every block. The encoder can optionally return the
    hidden states at a fixed set of block indices together with the final sequence (used by the segmentation
    decoder).

    Args:
        embed_dim: Token channels.
        depth: Number of blocks.
        num_heads: Number of attention heads.
        mlp_ratio: Hidden-channel expansion ratio of the MLPs.
        qkv_bias: Whether to use a bias term in the attention projections.
        dropout: Dropout rate inside the blocks.
        attn_dropout: Dropout rate applied to the attention matrix.
        drop_path: Stochastic-depth drop-path rate(s). A float applies to every block.
        num_queries: Number of global query tokens leading the sequence.
        detach_tokens: Whether the local tokens are detached before the query cross-attention.
        act: Activation function of the MLPs.
        act_kwargs: Extra arguments for the activation.
        norm: Normalization applied before each sub-block.
        norm_kwargs: Extra arguments for the normalization.

    Shape:
        - Input: $(B, Q + G, C)$ tokens and positional embeddings.
        - Output: $(B, Q + G, C)$, or the hidden states at `fetch_idx` and the final sequence.
    """

    def __init__(
        self,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        drop_path: Union[float, List[float]] = 0.0,
        num_queries: int = 3,
        detach_tokens: bool = False,
        act: Union[str, Callable, None] = "gelu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = nn.LayerNorm,
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                ReConEncoderBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    num_queries=num_queries,
                    detach_tokens=detach_tokens,
                    act=act,
                    act_kwargs=act_kwargs,
                    norm=norm,
                    norm_kwargs=norm_kwargs,
                )
                for i in range(depth)
            ]
        )

    @overload
    def forward(self, x: Tensor, pos: Tensor, fetch_idx: Sequence[int]) -> Tuple[List[Tensor], Tensor]: ...

    @overload
    def forward(self, x: Tensor, pos: Tensor, fetch_idx: None = None) -> Tensor: ...

    def forward(self, x: Tensor, pos: Tensor, fetch_idx: Optional[Sequence[int]] = None) -> Any:
        features: List[Tensor] = []
        for i, block in enumerate(self.blocks):
            x = block(x + pos)
            if fetch_idx is not None and i in fetch_idx:
                features.append(x)
        if fetch_idx is not None:
            return features, x
        return x


class ReConClassification(ClassificationModel):
    r"""ReCon classification model, as in :arxiv: [Contrast with Reconstruct: Contrastive 3D Representation Learning
    Guided by Generative Pretraining](https://arxiv.org/abs/2302.02318), adapted from
    :github: [qizekun/ReCon](https://github.com/qizekun/ReCon).

    Tokenizes local groups with a mini-PointNet encoder, prepends three learnable global queries (class, image and
    text), processes the sequence with a `ReConEncoder` whose queries cross-attend to the local tokens, and classifies
    the concatenation of the queries and the max-pooled local tokens.

    Args:
        in_channels: Number of per-point feature channels concatenated to the coordinates ($0$ for coordinates only).
        num_classes: Number of output classes.
        embed_dim: Token channels.
        depth: Number of transformer blocks.
        num_heads: Number of attention heads.
        num_groups: Number of groups (FPS centers) per sample.
        group_size: Number of neighbors per group.
        encoder_local_channels: Hidden widths of the patch embedder's per-point MLP.
        encoder_global_channels: Hidden widths of the patch embedder's per-group MLP.
        pos_embed_channels: Hidden widths of the positional-embedding MLP.
        drop_path: Stochastic-depth drop-path rate.
        dropout: Dropout rate in the classification head.
        act: Activation function.
        act_kwargs: Extra arguments for the activation.
        norm: Normalization function.
        norm_kwargs: Extra arguments for the normalization.
        spatial_dim: Spatial dimension of the input point cloud.

    Shape:
        - Input: $(N, 3)$ and $(N,)$.
        - Output: $(B, C)$ where $B$ is the batch size and $C$ is `num_classes`.
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
        encoder_local_channels: Sequence[int] = (128, 256),
        encoder_global_channels: Sequence[int] = (512,),
        pos_embed_channels: Sequence[int] = (128,),
        drop_path: float = 0.1,
        dropout: float = 0.5,
        act: Union[str, Callable, None] = "gelu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = nn.LayerNorm,
        norm_kwargs: Optional[Dict[str, Any]] = None,
        spatial_dim: int = 3,
    ) -> None:
        super().__init__(in_channels=in_channels, num_classes=num_classes)
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.group_size = group_size
        self.encoder_local_channels = encoder_local_channels
        self.encoder_global_channels = encoder_global_channels
        self.pos_embed_channels = pos_embed_channels
        self.drop_path = drop_path
        self.dropout = dropout
        self.act = act
        self.act_kwargs = act_kwargs
        self.norm = norm
        self.norm_kwargs = norm_kwargs
        self.spatial_dim = spatial_dim

        self.encoder = self.configure_encoder()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.image_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.text_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_pos = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.image_pos = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.text_pos = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = self.configure_pos_embed()
        self.blocks = self.configure_blocks()
        self.norm_f = nn.LayerNorm(embed_dim)
        self.head = self.configure_head()
        self.reset_parameters()

    def configure_encoder(self) -> PointPatchEmbed:
        """Build the mini-PointNet patch embedder tokenizing each local group."""
        return PointPatchEmbed(
            embed_dim=self.embed_dim,
            in_channels=self.spatial_dim + self.in_channels,
            local_channels=self.encoder_local_channels,
            global_channels=self.encoder_global_channels,
        )

    def configure_pos_embed(self) -> MLP:
        """Build the positional-embedding MLP mapping group centers to token channels."""
        return MLP(
            [self.spatial_dim, *self.pos_embed_channels, self.embed_dim],
            act=self.act,
            act_kwargs=self.act_kwargs,
            norm=None,
            plain_last=True,
        )

    def configure_blocks(self) -> ReConEncoder:
        """Build the encoder with a linearly scaled stochastic-depth schedule."""
        dpr = [x.item() for x in torch.linspace(0, self.drop_path, self.depth)]
        return ReConEncoder(
            embed_dim=self.embed_dim,
            depth=self.depth,
            drop_path=dpr,
            num_heads=self.num_heads,
            num_queries=3,
            act=self.act,
            act_kwargs=self.act_kwargs,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
        )

    @property
    def num_features(self) -> int:
        """Channel count $C$ of the pooled features entering the head."""
        return self.embed_dim * 4

    def configure_head(self) -> nn.Module:
        if self.num_classes == 0:
            return nn.Identity()
        return MLP(
            [self.num_features, 256, 256, self.num_classes],
            act="relu",
            norm="batch_norm",
            dropout=[self.dropout, self.dropout, 0.0],
            bias=True,
            plain_last=True,
        )

    def reset_parameters(self) -> None:
        for param in (self.image_token, self.text_token, self.cls_pos, self.image_pos, self.text_pos):
            nn.init.trunc_normal_(param, std=0.02)

    def reset_classifier(self, num_classes: int, global_pool: Any = None, **kwargs: Any) -> None:
        if global_pool is not None:
            raise ValueError(
                f"{self.__class__.__name__} pools with a fixed query-token + max-pool concatenation; "
                "`global_pool` is not configurable."
            )

        self.num_classes = num_classes
        self.head = self.configure_head()

    def forward_features(self, x: OptTensor, pos: Tensor, batch: Tensor) -> Tensor:
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
        group_input_tokens = self.encoder(neighborhood)

        B = group_input_tokens.size(0)
        queries = torch.cat([self.cls_token, self.image_token, self.text_token], dim=1).expand(B, -1, -1)
        query_pos = torch.cat([self.cls_pos, self.image_pos, self.text_pos], dim=1).expand(B, -1, -1)
        pos_embed = self.pos_embed(center)

        x = torch.cat([queries, group_input_tokens], dim=1)
        pos_embed = torch.cat([query_pos, pos_embed], dim=1)

        x = self.blocks(x, pos_embed)
        x = self.norm_f(x)
        return x

    def forward_head(self, x: Tensor, pre_logits: bool = False) -> Tensor:
        concat_f = torch.cat([x[:, :3].flatten(1), x[:, 3:].max(1)[0]], dim=-1)
        return concat_f if pre_logits else self.head(concat_f)

    def forward(self, x: OptTensor, pos: Tensor, batch: Tensor) -> Tensor:
        x = self.forward_features(x, pos, batch)
        return self.forward_head(x)


class ReConPartSegmentation(PartSegmentationModel):
    r"""ReCon part-segmentation model, as in :arxiv: [Contrast with Reconstruct: Contrastive 3D Representation
    Learning Guided by Generative Pretraining](https://arxiv.org/abs/2302.02318), adapted from
    :github: [qizekun/ReCon](https://github.com/qizekun/ReCon).

    Tokenizes local groups with a mini-PointNet encoder, prepends the class, image and text queries, processes the
    sequence with a `ReConEncoder` and concatenates the normalized local hidden states at three block depths. A
    `PartSegmentationDecoder` propagates the group features back to every point and appends their max- and
    mean-pooled descriptor and the embedded category one-hot; the max-pooled final queries are appended last, before
    the per-point classifier.

    Args:
        in_channels: Number of per-point feature channels concatenated to the coordinates ($0$ for coordinates only).
        num_classes: Number of output part classes (across all categories).
        num_categories: Number of object categories for the category one-hot branch.
        embed_dim: Token channels.
        depth: Number of transformer blocks.
        num_heads: Number of attention heads.
        num_groups: Number of groups (FPS centers) per sample.
        group_size: Number of neighbors per group.
        encoder_local_channels: Hidden widths of the patch embedder's per-point MLP.
        encoder_global_channels: Hidden widths of the patch embedder's per-group MLP.
        pos_embed_channels: Hidden widths of the positional-embedding MLP.
        drop_path: Stochastic-depth drop-path rate.
        dropout: Dropout rate in the per-point head.
        act: Activation function.
        act_kwargs: Extra arguments for the activation.
        norm: Normalization function.
        norm_kwargs: Extra arguments for the normalization.
        spatial_dim: Spatial dimension of the input point cloud.

    Shape:
        - Input: $(N, 3)$, $(N,)$, and a category one-hot $(B, \text{num\_categories})$.
        - Output: $(N, C)$ logits where $C$ is `num_classes`.
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
        encoder_local_channels: Sequence[int] = (128, 256),
        encoder_global_channels: Sequence[int] = (512,),
        pos_embed_channels: Sequence[int] = (128,),
        drop_path: float = 0.1,
        dropout: float = 0.5,
        act: Union[str, Callable, None] = "gelu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = nn.LayerNorm,
        norm_kwargs: Optional[Dict[str, Any]] = None,
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
        self.encoder_local_channels = encoder_local_channels
        self.encoder_global_channels = encoder_global_channels
        self.pos_embed_channels = pos_embed_channels
        self.drop_path = drop_path
        self.dropout = dropout
        self.act = act
        self.act_kwargs = act_kwargs
        self.norm = norm
        self.norm_kwargs = norm_kwargs
        self.spatial_dim = spatial_dim

        self.encoder = self.configure_encoder()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.image_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.text_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.image_pos = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.text_pos = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = self.configure_pos_embed()
        self.blocks = self.configure_blocks()
        self.norm_f = nn.LayerNorm(embed_dim)

        self.decoder = self.configure_decoder()
        self.head = self.configure_head()

    def configure_encoder(self) -> PointPatchEmbed:
        """Build the mini-PointNet patch embedder tokenizing each local group."""
        return PointPatchEmbed(
            embed_dim=self.embed_dim,
            in_channels=self.spatial_dim + self.in_channels,
            local_channels=self.encoder_local_channels,
            global_channels=self.encoder_global_channels,
        )

    def configure_pos_embed(self) -> MLP:
        """Build the positional-embedding MLP mapping group centers to token channels."""
        return MLP(
            [self.spatial_dim, *self.pos_embed_channels, self.embed_dim],
            act=self.act,
            act_kwargs=self.act_kwargs,
            norm=None,
            plain_last=True,
        )

    def configure_blocks(self) -> ReConEncoder:
        """Build the encoder with a linearly scaled stochastic-depth schedule."""
        dpr = [x.item() for x in torch.linspace(0, self.drop_path, self.depth)]
        return ReConEncoder(
            embed_dim=self.embed_dim,
            depth=self.depth,
            drop_path=dpr,
            num_heads=self.num_heads,
            num_queries=3,
            act=self.act,
            act_kwargs=self.act_kwargs,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
        )

    def configure_decoder(self) -> PartSegmentationDecoder:
        """Build the decoder interpolating the group features back to every point with the global descriptor."""
        return PartSegmentationDecoder(
            in_channels=3 * self.embed_dim,
            num_categories=self.num_categories,
            propagation_channels=(4 * self.embed_dim, 1024),
            spatial_dim=self.spatial_dim,
        )

    @property
    def num_features(self) -> int:
        """Channel count $C$ of the per-point features entering the head."""
        return self.decoder.out_channels + self.embed_dim

    def configure_head(self) -> nn.Module:
        if self.num_classes == 0:
            return nn.Identity()
        return MLP(
            [self.num_features, 512, 256, self.num_classes],
            act="relu",
            norm="batch_norm",
            dropout=[self.dropout, 0.0, 0.0],
            bias=True,
            plain_last=True,
        )

    def reset_classifier(self, num_classes: int, **kwargs: Any) -> None:
        self.num_classes = num_classes
        self.head = self.configure_head()

    def forward_features(self, x: OptTensor, pos: Tensor, batch: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
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
        group_input_tokens = self.encoder(neighborhood)

        B = group_input_tokens.size(0)
        queries = torch.cat([self.cls_token, self.image_token, self.text_token], dim=1).expand(B, -1, -1)
        query_pos = torch.cat([self.cls_pos, self.image_pos, self.text_pos], dim=1).expand(B, -1, -1)
        pos_embed = self.pos_embed(center)

        x = torch.cat([queries, group_input_tokens], dim=1)
        pos_embed = torch.cat([query_pos, pos_embed], dim=1)

        feature_list, x = self.blocks(x, pos_embed, fetch_idx=self.fetch_idx)
        x_feat = torch.cat([self.norm_f(feat[:, 3:]) for feat in feature_list], dim=-1)
        x_query = x[:, :3].max(1)[0]
        return x_feat, x_query, center

    def forward_decoder(
        self,
        x_feat: Tensor,
        x_query: Tensor,
        pos: Tensor,
        batch: Tensor,
        center: Tensor,
        category: Tensor,
    ) -> Tensor:
        x = self.decoder(x_feat, center, pos, batch, category)
        return torch.cat([x, x_query[batch]], dim=-1)

    def forward_head(self, x: Tensor, pre_logits: bool = False) -> Tensor:
        return x if pre_logits else self.head(x)

    def forward(self, x: OptTensor, pos: Tensor, batch: Tensor, category: Tensor) -> Tensor:
        x_feat, x_query, center = self.forward_features(x, pos, batch)
        x = self.forward_decoder(x_feat, x_query, pos, batch, center, category)
        return self.forward_head(x)


class ReConPretraining(PretrainingModel):
    r"""ReCon contrastive and generative pretraining model, as in :arxiv: [Contrast with Reconstruct: Contrastive 3D
    Representation Learning Guided by Generative Pretraining](https://arxiv.org/abs/2302.02318), adapted from
    :github: [qizekun/ReCon](https://github.com/qizekun/ReCon).

    Masks a fraction of the group tokens and encodes the visible ones with a `ReConEncoder` whose three global queries
    (class, image and text) cross-attend to the detached local tokens. A transformer decoder reconstructs the
    centered coordinates of the masked groups from learnable mask tokens, as in Point-MAE, and the image and text
    queries are projected to the widths of the frozen image and text encoders whose features they regress. `forward`
    returns the predicted and target coordinates of the masked groups, each packed with the group index of its
    points, followed by the projected image and text queries.

    Args:
        in_channels: Number of input channels (unused beyond coordinates; kept for the registry contract).
        embed_dim: Token channels.
        encoder_depth: Number of encoder blocks.
        decoder_depth: Number of decoder transformer blocks.
        num_heads: Number of encoder attention heads.
        decoder_num_heads: Number of decoder attention heads.
        num_groups: Number of groups (FPS centers) per sample.
        group_size: Number of neighbors per group.
        image_dim: Width of the image features the image query regresses.
        text_dim: Width of the text features the text query regresses.
        encoder_local_channels: Hidden widths of the patch embedder's per-point MLP.
        encoder_global_channels: Hidden widths of the patch embedder's per-group MLP.
        pos_embed_channels: Hidden widths of the encoder and decoder positional-embedding MLPs.
        mask_ratio: Fraction of tokens to mask.
        drop_path: Stochastic-depth drop-path rate.
        act: Activation function.
        act_kwargs: Extra arguments for the activation.
        norm: Normalization function.
        norm_kwargs: Extra arguments for the normalization.
        spatial_dim: Spatial dimension of the input point cloud.

    Shape:
        - Input: $(N, 3)$ and $(N,)$.
        - Output: predicted and target points $(B \cdot G_\text{mask} \cdot M, 3)$ with their group indices, then the
          image query $(B, \text{image\_dim})$ and the text query $(B, \text{text\_dim})$.
    """

    def __init__(
        self,
        in_channels: int,
        *,
        image_dim: int,
        text_dim: int,
        embed_dim: int = 384,
        encoder_depth: int = 12,
        decoder_depth: int = 4,
        num_heads: int = 6,
        decoder_num_heads: int = 6,
        num_groups: int = 64,
        group_size: int = 32,
        encoder_local_channels: Sequence[int] = (128, 256),
        encoder_global_channels: Sequence[int] = (512,),
        pos_embed_channels: Sequence[int] = (128,),
        mask_ratio: float = 0.6,
        drop_path: float = 0.1,
        act: Union[str, Callable, None] = "gelu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = nn.LayerNorm,
        norm_kwargs: Optional[Dict[str, Any]] = None,
        spatial_dim: int = 3,
    ) -> None:
        super().__init__(in_channels=in_channels)
        if not 0.0 < mask_ratio < 1.0:
            raise ValueError(f"`mask_ratio` must be in (0, 1), got {mask_ratio}.")

        self.embed_dim = embed_dim
        self.encoder_depth = encoder_depth
        self.decoder_depth = decoder_depth
        self.num_heads = num_heads
        self.decoder_num_heads = decoder_num_heads
        self.num_groups = num_groups
        self.group_size = group_size
        self.image_dim = image_dim
        self.text_dim = text_dim
        self.encoder_local_channels = encoder_local_channels
        self.encoder_global_channels = encoder_global_channels
        self.pos_embed_channels = pos_embed_channels
        self.mask_ratio = mask_ratio
        self.drop_path = drop_path
        self.act = act
        self.act_kwargs = act_kwargs
        self.norm = norm
        self.norm_kwargs = norm_kwargs
        self.spatial_dim = spatial_dim

        self.encoder = PointPatchEmbed(
            embed_dim=embed_dim,
            local_channels=encoder_local_channels,
            global_channels=encoder_global_channels,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.image_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.text_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_pos = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.image_pos = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.text_pos = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = self.configure_pos_embed()
        self.blocks = self.configure_blocks()
        self.norm_f = nn.LayerNorm(embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.decoder_pos_embed = self.configure_pos_embed()
        self.decoder = self.configure_decoder()
        self.increase_dim = MLP([embed_dim, 3 * group_size], act=None, norm=None, bias=True, plain_last=True)
        self.image_proj = nn.Linear(embed_dim, image_dim)
        self.text_proj = nn.Linear(embed_dim, text_dim)
        self.reset_parameters()

    def configure_pos_embed(self) -> MLP:
        """Build a positional-embedding MLP mapping group centers to token channels."""
        return MLP(
            [self.spatial_dim, *self.pos_embed_channels, self.embed_dim],
            act=self.act,
            act_kwargs=self.act_kwargs,
            norm=None,
            plain_last=True,
        )

    def configure_blocks(self) -> ReConEncoder:
        """Build the encoder, whose queries see the local tokens without gradient."""
        dpr = [x.item() for x in torch.linspace(0, self.drop_path, self.encoder_depth)]
        return ReConEncoder(
            embed_dim=self.embed_dim,
            depth=self.encoder_depth,
            drop_path=dpr,
            num_heads=self.num_heads,
            num_queries=3,
            detach_tokens=True,
            act=self.act,
            act_kwargs=self.act_kwargs,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
        )

    def configure_decoder(self) -> TransformerDecoder:
        """Build the transformer decoder reconstructing the masked tokens."""
        dpr = [x.item() for x in torch.linspace(0, self.drop_path, self.decoder_depth)]
        return TransformerDecoder(
            embed_dim=self.embed_dim,
            depth=self.decoder_depth,
            drop_path=dpr,
            num_heads=self.decoder_num_heads,
            act=self.act,
            act_kwargs=self.act_kwargs,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
        )

    def reset_parameters(self) -> None:
        for param in (self.image_token, self.text_token, self.cls_pos, self.image_pos, self.text_pos, self.mask_token):
            nn.init.trunc_normal_(param, std=0.02)

    def forward(
        self, x: OptTensor, pos: Tensor, batch: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        neighborhood, center = group(pos, batch, self.num_groups, self.group_size, random_start=self.training)
        B, G, _ = center.shape
        mask = random_mask(B, G, self.mask_ratio, device=center.device)

        tokens = self.encoder(neighborhood)
        C = tokens.shape[-1]
        x_vis = tokens[~mask].reshape(B, -1, C)
        pos_vis = self.pos_embed(center[~mask].reshape(B, -1, self.spatial_dim))
        queries = torch.cat([self.cls_token, self.image_token, self.text_token], dim=1).expand(B, -1, -1)
        query_pos = torch.cat([self.cls_pos, self.image_pos, self.text_pos], dim=1).expand(B, -1, -1)
        x = self.norm_f(self.blocks(torch.cat([queries, x_vis], dim=1), torch.cat([query_pos, pos_vis], dim=1)))
        queries, x_vis = x[:, :3], x[:, 3:]

        decoder_pos_vis = self.decoder_pos_embed(center[~mask].reshape(B, -1, self.spatial_dim))
        decoder_pos_mask = self.decoder_pos_embed(center[mask].reshape(B, -1, self.spatial_dim))
        M = decoder_pos_mask.shape[1]
        x_full = torch.cat([x_vis, self.mask_token.expand(B, M, -1)], dim=1)
        pos_full = torch.cat([decoder_pos_vis, decoder_pos_mask], dim=1)
        x_rec = self.decoder(x_full, pos_full, M)

        pred = self.increase_dim(x_rec).reshape(-1, 3)
        target = neighborhood[mask].reshape(-1, 3)
        point_batch = torch.arange(B * M, device=pred.device).repeat_interleave(self.group_size)
        return pred, target, point_batch, point_batch, self.image_proj(queries[:, 1]), self.text_proj(queries[:, 2])


_MODELNET_TRANSFORM = T.Compose(
    [
        T.CopyItems(keys=DataKeys.POS, dst_keys=DataKeys.ORIGIN_POS),
        T.FarthestPointSample(
            pos_key=DataKeys.POS,
            keys=[DataKeys.NORMAL],
            num_samples=1024,
            random_start=False,
            dst_index_key=DataKeys.INDEX,
            allow_missing_keys=True,
        ),
        T.Rescale(keys=DataKeys.POS, method="centroid"),
    ]
)


@register_model(
    "recon.modelnet40.zekun-qi",
    task="classification",
    weights=WeightsDict(
        url="hf://torch-pointcloud/recon.modelnet40.zekun-qi/resolve/0000000000000000000000000000000000000000/model.safetensors",
        dataset="modelnet40",
        metrics={"OA": 93.68, "mAcc": 90.73},
        classes=MODELNET40_CLASSES,
        author="zekun-qi",
        license="MIT",
    ),
    transform=_MODELNET_TRANSFORM,
    hparams=dict(
        in_channels=0,
        num_classes=40,
        embed_dim=384,
        depth=12,
        num_heads=6,
        num_groups=64,
        group_size=32,
        encoder_local_channels=(128, 256),
        encoder_global_channels=(512,),
        pos_embed_channels=(128,),
        drop_path=0.2,
        dropout=0.5,
        act="gelu",
        spatial_dim=3,
    ),
)
def recon_modelnet40_clf(**kwargs: Any) -> ReConClassification:
    return ReConClassification(**kwargs)


@register_model(
    "recon.modelnet40-8k.zekun-qi",
    task="classification",
    weights=WeightsDict(
        url="hf://torch-pointcloud/recon.modelnet40-8k.zekun-qi/resolve/0000000000000000000000000000000000000000/model.safetensors",
        dataset="modelnet40",
        metrics={"OA": 94.25, "mAcc": 91.35},
        classes=MODELNET40_CLASSES,
        author="zekun-qi",
        license="MIT",
    ),
    transform=T.Compose(
        [
            T.CopyItems(keys=DataKeys.POS, dst_keys=DataKeys.ORIGIN_POS),
            T.FarthestPointSample(
                pos_key=DataKeys.POS,
                keys=[DataKeys.NORMAL],
                num_samples=8192,
                random_start=False,
                dst_index_key=DataKeys.INDEX,
                allow_missing_keys=True,
            ),
            T.Rescale(keys=DataKeys.POS, method="centroid"),
        ]
    ),
    hparams=dict(
        in_channels=0,
        num_classes=40,
        embed_dim=384,
        depth=12,
        num_heads=6,
        num_groups=512,
        group_size=32,
        encoder_local_channels=(128, 256),
        encoder_global_channels=(512,),
        pos_embed_channels=(128,),
        drop_path=0.2,
        dropout=0.5,
        act="gelu",
        spatial_dim=3,
    ),
)
def recon_modelnet40_8k_clf(**kwargs: Any) -> ReConClassification:
    return ReConClassification(**kwargs)


@register_model(
    "recon.scanobjectnn-objbg.zekun-qi",
    task="classification",
    weights=WeightsDict(
        url="hf://torch-pointcloud/recon.scanobjectnn-objbg.zekun-qi/resolve/0000000000000000000000000000000000000000/model.safetensors",
        dataset="scanobjectnn-objbg",
        metrics={"OA": 95.18, "mAcc": 94.48},
        classes=SCANOBJECTNN_CLASSES,
        author="zekun-qi",
        license="MIT",
    ),
    # The reference feeds the raw 2048-point ScanObjectNN sample, without sampling or normalization.
    transform=None,
    hparams=dict(
        in_channels=0,
        num_classes=15,
        embed_dim=384,
        depth=12,
        num_heads=6,
        num_groups=128,
        group_size=32,
        encoder_local_channels=(128, 256),
        encoder_global_channels=(512,),
        pos_embed_channels=(128,),
        drop_path=0.2,
        dropout=0.5,
        act="gelu",
        spatial_dim=3,
    ),
)
def recon_scanobjectnn_objbg_clf(**kwargs: Any) -> ReConClassification:
    return ReConClassification(**kwargs)


@register_model(
    "recon.scanobjectnn-objonly.zekun-qi",
    task="classification",
    weights=WeightsDict(
        url="hf://torch-pointcloud/recon.scanobjectnn-objonly.zekun-qi/resolve/0000000000000000000000000000000000000000/model.safetensors",
        dataset="scanobjectnn-objonly",
        metrics={"OA": 93.63, "mAcc": 93.74},
        classes=SCANOBJECTNN_CLASSES,
        author="zekun-qi",
        license="MIT",
    ),
    # The reference feeds the raw 2048-point ScanObjectNN sample, without sampling or normalization.
    transform=None,
    hparams=dict(
        in_channels=0,
        num_classes=15,
        embed_dim=384,
        depth=12,
        num_heads=6,
        num_groups=128,
        group_size=32,
        encoder_local_channels=(128, 256),
        encoder_global_channels=(512,),
        pos_embed_channels=(128,),
        drop_path=0.2,
        dropout=0.5,
        act="gelu",
        spatial_dim=3,
    ),
)
def recon_scanobjectnn_objonly_clf(**kwargs: Any) -> ReConClassification:
    return ReConClassification(**kwargs)


@register_model(
    "recon.scanobjectnn-hardest.zekun-qi",
    task="classification",
    weights=WeightsDict(
        url="hf://torch-pointcloud/recon.scanobjectnn-hardest.zekun-qi/resolve/0000000000000000000000000000000000000000/model.safetensors",
        dataset="scanobjectnn-hardest",
        metrics={"OA": 90.56, "mAcc": 89.39},
        classes=SCANOBJECTNN_CLASSES,
        author="zekun-qi",
        license="MIT",
    ),
    # The reference feeds the raw 2048-point ScanObjectNN sample, without sampling or normalization.
    transform=None,
    hparams=dict(
        in_channels=0,
        num_classes=15,
        embed_dim=384,
        depth=12,
        num_heads=6,
        num_groups=128,
        group_size=32,
        encoder_local_channels=(128, 256),
        encoder_global_channels=(512,),
        pos_embed_channels=(128,),
        drop_path=0.2,
        dropout=0.5,
        act="gelu",
        spatial_dim=3,
    ),
)
def recon_scanobjectnn_hardest_clf(**kwargs: Any) -> ReConClassification:
    return ReConClassification(**kwargs)


@register_model(
    "recon.shapenetpart.zekun-qi",
    task="part-segmentation",
    input_keys=("x", "pos", "batch", "category"),
    weights=WeightsDict(
        url="hf://torch-pointcloud/recon.shapenetpart.zekun-qi/resolve/0000000000000000000000000000000000000000/model.safetensors",
        dataset="shapenetpart",
        metrics={"ins_mIoU": 86.15, "cls_mIoU": 84.40},
        author="zekun-qi",
        license="MIT",
    ),
    transform=T.Compose(
        [
            T.Rescale(keys=DataKeys.POS, method="centroid"),
            T.CopyItems(
                keys=[DataKeys.POS, DataKeys.SEGMENT],
                dst_keys=[DataKeys.ORIGIN_POS, DataKeys.ORIGIN_SEGMENT],
                allow_missing_keys=True,
            ),
            T.FarthestPointSample(
                pos_key=DataKeys.POS,
                keys=[DataKeys.NORMAL, DataKeys.SEGMENT],
                num_samples=2048,
                random_start=False,
                dst_index_key=DataKeys.INDEX,
                allow_missing_keys=True,
            ),
            T.OneHot(keys=DataKeys.CATEGORY, num_classes=16),
        ]
    ),
    hparams=dict(
        in_channels=0,
        num_classes=50,
        num_categories=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        num_groups=128,
        group_size=32,
        encoder_local_channels=(128, 256),
        encoder_global_channels=(512,),
        pos_embed_channels=(128,),
        drop_path=0.1,
        dropout=0.5,
        act="gelu",
        spatial_dim=3,
    ),
)
def recon_shapenetpart_seg(**kwargs: Any) -> ReConPartSegmentation:
    return ReConPartSegmentation(**kwargs)


@register_model(
    "recon.pretrain.zekun-qi",
    task="pretraining",
    weights=WeightsDict(
        url="hf://torch-pointcloud/recon.pretrain.zekun-qi/resolve/0000000000000000000000000000000000000000/model.safetensors",
        dataset="shapenet55",
        author="zekun-qi",
        license="MIT",
    ),
    hparams=dict(
        in_channels=0,
        image_dim=768,
        text_dim=512,
        embed_dim=384,
        encoder_depth=12,
        decoder_depth=4,
        num_heads=6,
        decoder_num_heads=6,
        num_groups=64,
        group_size=32,
        encoder_local_channels=(128, 256),
        encoder_global_channels=(512,),
        pos_embed_channels=(128,),
        mask_ratio=0.6,
        drop_path=0.1,
        act="gelu",
        spatial_dim=3,
    ),
)
def recon_pretrain(**kwargs: Any) -> ReConPretraining:
    return ReConPretraining(**kwargs)
