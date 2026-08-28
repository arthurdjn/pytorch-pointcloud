"""Point-MAE classification, segmentation, and masked autoencoder pretraining models.

{{ paper("2203.06604") }}
"""

from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    overload,
)

import torch
from torch import Tensor, nn
from torch_geometric.nn import MLP

import torch_pointcloud.transforms as T
from torch_pointcloud.datasets.modelnet import MODELNET40_CLASSES
from torch_pointcloud.datasets.scanobjectnn import SCANOBJECTNN_CLASSES
from torch_pointcloud.layers import FPModule, PointPatchEmbed, TransformerBlock
from torch_pointcloud.utils.cluster import group
from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.types import OptTensor

from ._base import BaseModel, ClassificationModel, SegmentationModel
from ._registry import WeightsDict, register_model


class TransformerEncoder(nn.Module):
    r"""Plain (non-hierarchical) transformer encoder, as in :arxiv: [Masked Autoencoders for Point Cloud
    Self-supervised Learning](https://arxiv.org/abs/2203.06604), adapted from
    :github: [Pang-Yatian/Point-MAE](https://github.com/Pang-Yatian/Point-MAE).

    A stack of pre-norm transformer blocks. The positional embedding is added to the tokens
    before every block. The encoder can optionally return the hidden states at a fixed set of
    block indices (used by the segmentation decoder).

    Args:
        embed_dim: Token channels.
        depth: Number of transformer blocks.
        num_heads: Number of attention heads.
        mlp_ratio: Hidden-channel expansion ratio of the MLP.
        qkv_bias: Whether to use a bias term in the query / key / value projection.
        dropout: Dropout rate inside the blocks.
        attn_dropout: Dropout rate applied to the attention matrix.
        drop_path: Stochastic-depth drop-path rate(s). A float applies to every block.
        act: Activation function of the MLP.
        act_kwargs: Extra arguments for the activation.
        norm: Normalization applied before attention and before the MLP.
        norm_kwargs: Extra arguments for the normalization.

    Shape:
        - Input: $(B, N, C)$ where $B$ is the batch size, $N$ is the sequence length, and $C$ is `embed_dim`.
        - Output: $(B, N, C)$.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        depth: int = 4,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        drop_path: Union[float, List[float]] = 0.0,
        act: Union[str, Callable, None] = "gelu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = nn.LayerNorm,
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    act=act,
                    act_kwargs=act_kwargs,
                    norm=norm,
                    norm_kwargs=norm_kwargs,
                )
                for i in range(depth)
            ]
        )

    @overload
    def forward(self, x: Tensor, pos: Tensor, fetch_idx: Sequence[int]) -> List[Tensor]: ...

    @overload
    def forward(self, x: Tensor, pos: Tensor, fetch_idx: None = None) -> Tensor: ...

    def forward(self, x: Tensor, pos: Tensor, fetch_idx: Optional[Sequence[int]] = None) -> Any:
        features: List[Tensor] = []
        for i, block in enumerate(self.blocks):
            x = block(x + pos)
            if fetch_idx is not None and i in fetch_idx:
                features.append(x)
        if fetch_idx is not None:
            return features
        return x


class TransformerDecoder(nn.Module):
    r"""Transformer decoder, as in :arxiv: [Masked Autoencoders for Point Cloud Self-supervised
    Learning](https://arxiv.org/abs/2203.06604), adapted from
    :github: [Pang-Yatian/Point-MAE](https://github.com/Pang-Yatian/Point-MAE).

    A stack of pre-norm transformer blocks followed by a final layer normalization. Only the
    hidden states of the last `return_token_num` tokens are returned (the masked tokens).

    Args:
        embed_dim: Token channels.
        depth: Number of transformer blocks.
        num_heads: Number of attention heads.
        mlp_ratio: Hidden-channel expansion ratio of the MLP.
        qkv_bias: Whether to use a bias term in the query / key / value projection.
        dropout: Dropout rate inside the blocks.
        attn_dropout: Dropout rate applied to the attention matrix.
        drop_path: Stochastic-depth drop-path rate(s). A float applies to every block.
        act: Activation function of the MLP.
        act_kwargs: Extra arguments for the activation.
        norm: Normalization applied before attention and before the MLP.
        norm_kwargs: Extra arguments for the normalization.

    Shape:
        - Input: $(B, N, C)$ where $B$ is the batch size, $N$ is the sequence length, and $C$ is `embed_dim`.
        - Output: $(B, R, C)$ where $R$ is `return_token_num`.
    """

    def __init__(
        self,
        embed_dim: int = 384,
        depth: int = 4,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        drop_path: Union[float, List[float]] = 0.1,
        act: Union[str, Callable, None] = "gelu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = nn.LayerNorm,
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    act=act,
                    act_kwargs=act_kwargs,
                    norm=norm,
                    norm_kwargs=norm_kwargs,
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Identity()

    def forward(self, x: Tensor, pos: Tensor, return_token_num: int) -> Tensor:
        for block in self.blocks:
            x = block(x + pos)
        out: Tensor = self.head(self.norm(x[:, -return_token_num:]))
        return out


class MaskTransformer(nn.Module):
    r"""Masked patch-embedding transformer encoder, as in :arxiv: [Masked Autoencoders for Point Cloud
    Self-supervised Learning](https://arxiv.org/abs/2203.06604), adapted from
    :github: [Pang-Yatian/Point-MAE](https://github.com/Pang-Yatian/Point-MAE).

    Tokenizes local groups with the mini-PointNet encoder, randomly masks a fraction of the
    tokens, and processes only the visible tokens with a plain transformer encoder. This is the
    encoder of the Point-MAE pretraining model.

    Args:
        embed_dim: Token channels.
        depth: Number of transformer blocks.
        num_heads: Number of attention heads.
        mask_ratio: Fraction of tokens to mask.
        drop_path: Stochastic-depth drop-path rate.
        encoder_local_channels: Hidden widths of the patch embedder's per-point MLP.
        encoder_global_channels: Hidden widths of the patch embedder's per-group MLP.
        pos_embed_channels: Hidden widths of the positional-embedding MLP.
        spatial_dim: Spatial dimension of the input point cloud.
        act: Activation function of the transformer MLPs and the positional embedding.
        act_kwargs: Extra arguments for the activation.
        norm: Normalization applied inside the transformer blocks.
        norm_kwargs: Extra arguments for the normalization.

    Shape:
        - Input: $(B, G, M, 3)$ and $(B, G, 3)$.
        - Output: visible tokens $(B, V, C)$ and a boolean mask $(B, G)$.
    """

    def __init__(
        self,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mask_ratio: float = 0.6,
        drop_path: float = 0.1,
        encoder_local_channels: Sequence[int] = (128, 256),
        encoder_global_channels: Sequence[int] = (512,),
        pos_embed_channels: Sequence[int] = (128,),
        spatial_dim: int = 3,
        act: Union[str, Callable, None] = "gelu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = nn.LayerNorm,
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        if not 0.0 < mask_ratio < 1.0:
            raise ValueError(f"`mask_ratio` must be in (0, 1), got {mask_ratio}.")
        self.mask_ratio = mask_ratio
        self.embed_dim = embed_dim
        self.encoder = PointPatchEmbed(
            embed_dim=embed_dim,
            local_channels=encoder_local_channels,
            global_channels=encoder_global_channels,
        )
        self.pos_embed = MLP(
            [spatial_dim, *pos_embed_channels, embed_dim], act=act, act_kwargs=act_kwargs, norm=None, plain_last=True
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        self.blocks = TransformerEncoder(
            embed_dim=embed_dim,
            depth=depth,
            drop_path=dpr,
            num_heads=num_heads,
            act=act,
            act_kwargs=act_kwargs,
            norm=norm,
            norm_kwargs=norm_kwargs,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def mask_center_rand(self, center: Tensor) -> Tensor:
        """Draw a random mask hiding `mask_ratio` of the groups, independently for every sample.

        Args:
            center: Group centers of shape $(B, G, 3)$.

        Returns:
            A boolean mask of shape $(B, G)$, `True` where the group is masked out.
        """
        B, G, _ = center.shape
        num_mask = int(self.mask_ratio * G)
        noise = torch.rand(B, G, device=center.device)
        keep = torch.argsort(noise, dim=1)[:, : G - num_mask]
        mask = torch.ones(B, G, dtype=torch.bool, device=center.device)
        mask.scatter_(1, keep, False)
        return mask

    def forward(self, neighborhood: Tensor, center: Tensor) -> Tuple[Tensor, Tensor]:
        bool_masked_pos = self.mask_center_rand(center)

        group_input_tokens = self.encoder(neighborhood)
        batch_size, _, C = group_input_tokens.size()

        x_vis = group_input_tokens[~bool_masked_pos].reshape(batch_size, -1, C)
        masked_center = center[~bool_masked_pos].reshape(batch_size, -1, 3)
        pos = self.pos_embed(masked_center)

        x_vis = self.blocks(x_vis, pos)
        x_vis = self.norm(x_vis)
        return x_vis, bool_masked_pos


class PointMAEClassification(ClassificationModel):
    r"""Point-MAE classification model, as in :arxiv: [Masked Autoencoders for Point Cloud Self-supervised
    Learning](https://arxiv.org/abs/2203.06604), adapted from
    :github: [Pang-Yatian/Point-MAE](https://github.com/Pang-Yatian/Point-MAE).

    Tokenizes local groups with a mini-PointNet encoder, prepends a class token, processes the
    sequence with a plain transformer encoder, and classifies the concatenation of the class
    token and the max-pooled patch tokens.

    Args:
        in_channels: Number of per-point feature channels concatenated to the coordinates ($0$ for coordinates only).
        num_classes: Number of output classes.
        embed_dim: Token channels.
        depth: Number of transformer blocks.
        num_heads: Number of attention heads.
        num_group: Number of groups (FPS centers) per sample.
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
        num_group: int = 64,
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
        self.num_group = num_group
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
        self.cls_pos = nn.Parameter(torch.randn(1, 1, embed_dim))
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

    def configure_blocks(self) -> TransformerEncoder:
        """Build the transformer encoder with a linearly scaled stochastic-depth schedule."""
        dpr = [x.item() for x in torch.linspace(0, self.drop_path, self.depth)]
        return TransformerEncoder(
            embed_dim=self.embed_dim,
            depth=self.depth,
            drop_path=dpr,
            num_heads=self.num_heads,
            act=self.act,
            act_kwargs=self.act_kwargs,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
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
            act="relu",
            norm="batch_norm",
            dropout=[self.dropout, self.dropout, 0.0],
            bias=True,
            plain_last=True,
        )

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.cls_pos, std=0.02)

    def reset_classifier(self, num_classes: int, global_pool: Any = None, **kwargs: Any) -> None:
        if global_pool is not None:
            raise ValueError(
                f"{self.__class__.__name__} pools with a fixed cls-token + max-pool concatenation; "
                "`global_pool` is not configurable."
            )
        self.num_classes = num_classes
        self.head = self.configure_head()

    def forward_features(self, x: OptTensor, pos: Tensor, batch: Tensor) -> Tensor:
        if x is None:
            neighborhood, center = group(pos, batch, self.num_group, self.group_size, random_start=self.training)
        else:
            neighborhood, center, neighbor_idx = group(
                pos, batch, self.num_group, self.group_size, random_start=self.training, return_indices=True
            )
            neighborhood = torch.cat([neighborhood, x[neighbor_idx].reshape(*neighborhood.shape[:3], -1)], dim=-1)
        group_input_tokens = self.encoder(neighborhood)

        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)
        pos_embed = self.pos_embed(center)

        x = torch.cat([cls_tokens, group_input_tokens], dim=1)
        pos_embed = torch.cat([cls_pos, pos_embed], dim=1)

        x = self.blocks(x, pos_embed)
        x = self.norm_f(x)
        return x

    def forward_head(self, x: Tensor, pre_logits: bool = False) -> Tensor:
        concat_f = torch.cat([x[:, 0], x[:, 1:].max(1)[0]], dim=-1)
        return concat_f if pre_logits else self.head(concat_f)

    def forward(self, x: OptTensor, pos: Tensor, batch: Tensor) -> Tensor:
        x = self.forward_features(x, pos, batch)
        return self.forward_head(x)


class PointMAESegmentation(SegmentationModel):
    r"""Point-MAE part-segmentation model, as in :arxiv: [Masked Autoencoders for Point Cloud Self-supervised
    Learning](https://arxiv.org/abs/2203.06604), adapted from
    :github: [Pang-Yatian/Point-MAE](https://github.com/Pang-Yatian/Point-MAE).

    Tokenizes local groups with a mini-PointNet encoder, processes them with a plain transformer
    encoder, concatenates the normalized hidden states at three block depths, and propagates the
    group features back to every point with a PointNet++-style feature propagation (`FPModule`). A
    category-conditioned global branch is fused before the per-point classifier. Every sample in
    the packed batch must contain the same number of points; a ragged batch raises `ValueError`.

    Args:
        in_channels: Number of per-point feature channels concatenated to the coordinates ($0$ for coordinates only).
        num_classes: Number of output part classes (across all categories).
        num_categories: Number of object categories for the category one-hot branch.
        embed_dim: Token channels.
        depth: Number of transformer blocks.
        num_heads: Number of attention heads.
        num_group: Number of groups (FPS centers) per sample.
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
        num_group: int = 128,
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
        if max(self.fetch_idx) >= depth:
            raise ValueError(
                f"`fetch_idx` {self.fetch_idx} requires at least {max(self.fetch_idx) + 1} blocks; got depth={depth}."
            )
        self.num_categories = num_categories
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.num_group = num_group
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
        self.pos_embed = self.configure_pos_embed()
        self.blocks = self.configure_blocks()
        self.norm_f = nn.LayerNorm(embed_dim)

        self.label_conv = self.configure_label_conv()
        self.propagation_0 = self.configure_propagation_0()
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

    def configure_blocks(self) -> TransformerEncoder:
        """Build the transformer encoder with a linearly scaled stochastic-depth schedule."""
        dpr = [x.item() for x in torch.linspace(0, self.drop_path, self.depth)]
        return TransformerEncoder(
            embed_dim=self.embed_dim,
            depth=self.depth,
            drop_path=dpr,
            num_heads=self.num_heads,
            act=self.act,
            act_kwargs=self.act_kwargs,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
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

    def configure_propagation_0(self) -> FPModule:
        """Build the feature-propagation module interpolating group features back to every point."""
        return FPModule(
            in_channels=3 * self.embed_dim + self.spatial_dim,
            channels=[self.embed_dim * 4, 1024],
            k=3,
            act="relu",
            norm="batch_norm",
            bias=True,
            weighting="squared",
            eps=1e-8,
        )

    @property
    def num_features(self) -> int:
        """Channel count $C$ of the per-point features entering the head."""
        return 1024 + 3 * self.embed_dim * 2 + 64

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
            neighborhood, center = group(pos, batch, self.num_group, self.group_size, random_start=self.training)
        else:
            neighborhood, center, neighbor_idx = group(
                pos, batch, self.num_group, self.group_size, random_start=self.training, return_indices=True
            )
            neighborhood = torch.cat([neighborhood, x[neighbor_idx].reshape(*neighborhood.shape[:3], -1)], dim=-1)
        group_input_tokens = self.encoder(neighborhood)
        pos_embed = self.pos_embed(center)

        feature_list = self.blocks(group_input_tokens, pos_embed, fetch_idx=self.fetch_idx)
        feature_list = [self.norm_f(feat).transpose(-1, -2).contiguous() for feat in feature_list]
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

        group_batch = torch.arange(B, device=pos.device).repeat_interleave(self.num_group)
        x_groups = x_feat.transpose(1, 2).reshape(B * self.num_group, -1)
        center_pos = center.reshape(B * self.num_group, self.spatial_dim)
        f_level_0, _, _ = self.propagation_0(x_groups, center_pos, group_batch, pos, pos, batch)
        f_level_0 = f_level_0.reshape(B, N, -1).transpose(1, 2)
        return torch.cat([f_level_0, x_global_feature], 1)

    def forward_head(self, x: Tensor, pre_logits: bool = False) -> Tensor:
        B, _, N = x.shape
        x = x.transpose(1, 2).reshape(B * N, -1)
        if pre_logits:
            return x
        x = self.head(x)
        return x.reshape(B, N, -1).transpose(1, 2)

    def forward(self, x: OptTensor, pos: Tensor, batch: Tensor, category: Tensor) -> Tensor:
        x_feat, center, batch = self.forward_features(x, pos, batch)
        x = self.forward_decoder(x_feat, pos, batch, center, category)
        logits = self.forward_head(x)
        B, C, N = logits.shape
        return logits.permute(0, 2, 1).reshape(B * N, C)


class PointMAEMaskedAutoEncoder(BaseModel):
    r"""Point-MAE masked-autoencoder pretraining model, as in :arxiv: [Masked Autoencoders for Point Cloud
    Self-supervised Learning](https://arxiv.org/abs/2203.06604), adapted from
    :github: [Pang-Yatian/Point-MAE](https://github.com/Pang-Yatian/Point-MAE).

    Masks a fraction of the group tokens, encodes the visible tokens, then reconstructs the
    masked groups' centered coordinates from learnable mask tokens with a transformer decoder
    and a per-token coordinate head. `forward` returns the predicted and target group coordinates
    for a set-to-set reconstruction objective such as `chamfer_distance` from
    `torch_pointcloud.losses`.

    Args:
        in_channels: Number of input channels (unused beyond coordinates; kept for the registry contract).
        embed_dim: Token channels.
        encoder_depth: Number of encoder transformer blocks.
        decoder_depth: Number of decoder transformer blocks.
        num_heads: Number of encoder attention heads.
        decoder_num_heads: Number of decoder attention heads.
        num_group: Number of groups (FPS centers) per sample.
        group_size: Number of neighbors per group.
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
        - Output: predicted and target groups, each of shape $(B \cdot M_\text{mask}, M, 3)$.
    """

    def __init__(
        self,
        in_channels: int,
        *,
        embed_dim: int = 384,
        encoder_depth: int = 12,
        decoder_depth: int = 4,
        num_heads: int = 6,
        decoder_num_heads: int = 6,
        num_group: int = 64,
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
        self.embed_dim = embed_dim
        self.encoder_depth = encoder_depth
        self.decoder_depth = decoder_depth
        self.num_heads = num_heads
        self.decoder_num_heads = decoder_num_heads
        self.num_group = num_group
        self.group_size = group_size
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

        self.MAE_encoder = self.configure_MAE_encoder()
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.decoder_pos_embed = self.configure_decoder_pos_embed()
        self.MAE_decoder = self.configure_MAE_decoder()
        self.increase_dim = self.configure_increase_dim()
        self.reset_parameters()

    def configure_MAE_encoder(self) -> MaskTransformer:
        """Build the masked transformer encoding the visible group tokens."""
        return MaskTransformer(
            embed_dim=self.embed_dim,
            depth=self.encoder_depth,
            num_heads=self.num_heads,
            mask_ratio=self.mask_ratio,
            drop_path=self.drop_path,
            encoder_local_channels=self.encoder_local_channels,
            encoder_global_channels=self.encoder_global_channels,
            pos_embed_channels=self.pos_embed_channels,
            spatial_dim=self.spatial_dim,
            act=self.act,
            act_kwargs=self.act_kwargs,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
        )

    def configure_decoder_pos_embed(self) -> MLP:
        """Build the decoder positional-embedding MLP mapping group centers to token channels."""
        return MLP(
            [self.spatial_dim, *self.pos_embed_channels, self.embed_dim],
            act=self.act,
            act_kwargs=self.act_kwargs,
            norm=None,
            plain_last=True,
        )

    def configure_MAE_decoder(self) -> TransformerDecoder:
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

    def configure_increase_dim(self) -> MLP:
        """Build the per-token linear head predicting the coordinates of each masked group."""
        return MLP([self.embed_dim, 3 * self.group_size], act=None, norm=None, bias=True, plain_last=True)

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(self, x: OptTensor, pos: Tensor, batch: Tensor) -> Tuple[Tensor, Tensor]:
        neighborhood, center = group(pos, batch, self.num_group, self.group_size, random_start=self.training)

        x_vis, mask = self.MAE_encoder(neighborhood, center)
        B, _, C = x_vis.shape

        pos_emd_vis = self.decoder_pos_embed(center[~mask].reshape(B, -1, 3))
        pos_emd_mask = self.decoder_pos_embed(center[mask].reshape(B, -1, 3))

        _, M, _ = pos_emd_mask.shape
        mask_token = self.mask_token.expand(B, M, -1)
        x_full = torch.cat([x_vis, mask_token], dim=1)
        pos_full = torch.cat([pos_emd_vis, pos_emd_mask], dim=1)

        x_rec = self.MAE_decoder(x_full, pos_full, M)

        B, R, C = x_rec.shape
        pred = self.increase_dim(x_rec).reshape(B * R, -1, 3)
        target = neighborhood[mask].reshape(B * R, -1, 3)
        return pred, target


_MODELNET_TRANSFORM = T.Compose(
    [
        T.CopyItems(keys=DataKeys.POS, names=DataKeys.ORIGIN_POS),
        T.FarthestPointSample(
            pos_key=DataKeys.POS,
            keys=[DataKeys.NORMAL],
            num_samples=1024,
            random_start=False,
            dst_index_key=DataKeys.INDEX,
        ),
        T.Rescale(keys=DataKeys.POS, method="centroid"),
    ]
)


@register_model(
    "point-mae-base.modelnet40.yatian-pang",
    task="classification",
    weights=WeightsDict(
        url="hf://torch-pointcloud/point-mae/point-mae-base.modelnet40.yatian-pang.safetensors",
        dataset="modelnet40",
        metrics={"OA": 92.87},
        classes=MODELNET40_CLASSES,
        author="yatian-pang",
        license="MIT",
    ),
    transform=_MODELNET_TRANSFORM,
    hparams=dict(
        in_channels=0,
        num_classes=40,
        embed_dim=384,
        depth=12,
        num_heads=6,
        num_group=64,
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
def point_mae_base_modelnet40_clf(**kwargs: Any) -> PointMAEClassification:
    return PointMAEClassification(**kwargs)


@register_model(
    "point-mae-base.modelnet40-8k.yatian-pang",
    task="classification",
    weights=WeightsDict(
        url="hf://torch-pointcloud/point-mae/point-mae-base.modelnet40-8k.yatian-pang.safetensors",
        dataset="modelnet40",
        metrics={"OA": 93.35},
        classes=MODELNET40_CLASSES,
        author="yatian-pang",
        license="MIT",
    ),
    transform=T.Compose(
        [
            T.CopyItems(keys=DataKeys.POS, names=DataKeys.ORIGIN_POS),
            T.FarthestPointSample(
                pos_key=DataKeys.POS,
                keys=[DataKeys.NORMAL],
                num_samples=8192,
                random_start=False,
                dst_index_key=DataKeys.INDEX,
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
        num_group=512,
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
def point_mae_base_modelnet40_8k_clf(**kwargs: Any) -> PointMAEClassification:
    return PointMAEClassification(**kwargs)


@register_model(
    "point-mae-base.scanobjectnn-objbg.yatian-pang",
    task="classification",
    weights=WeightsDict(
        url="hf://torch-pointcloud/point-mae/point-mae-base.scanobjectnn-objbg.yatian-pang.safetensors",
        dataset="scanobjectnn-objbg",
        metrics={"OA": 90.19},
        classes=SCANOBJECTNN_CLASSES,
        author="yatian-pang",
        license="MIT",
    ),
    transform=None,
    hparams=dict(
        in_channels=0,
        num_classes=15,
        embed_dim=384,
        depth=12,
        num_heads=6,
        num_group=128,
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
def point_mae_base_scanobjectnn_objbg_clf(**kwargs: Any) -> PointMAEClassification:
    return PointMAEClassification(**kwargs)


@register_model(
    "point-mae-base.scanobjectnn-objonly.yatian-pang",
    task="classification",
    weights=WeightsDict(
        url="hf://torch-pointcloud/point-mae/point-mae-base.scanobjectnn-objonly.yatian-pang.safetensors",
        dataset="scanobjectnn-objonly",
        metrics={"OA": 88.30},
        classes=SCANOBJECTNN_CLASSES,
        author="yatian-pang",
        license="MIT",
    ),
    transform=None,
    hparams=dict(
        in_channels=0,
        num_classes=15,
        embed_dim=384,
        depth=12,
        num_heads=6,
        num_group=128,
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
def point_mae_base_scanobjectnn_objonly_clf(**kwargs: Any) -> PointMAEClassification:
    return PointMAEClassification(**kwargs)


@register_model(
    "point-mae-base.scanobjectnn-hardest.yatian-pang",
    task="classification",
    weights=WeightsDict(
        url="hf://torch-pointcloud/point-mae/point-mae-base.scanobjectnn-hardest.yatian-pang.safetensors",
        dataset="scanobjectnn-hardest",
        metrics={"OA": 85.05},
        classes=SCANOBJECTNN_CLASSES,
        author="yatian-pang",
        license="MIT",
    ),
    transform=None,
    hparams=dict(
        in_channels=0,
        num_classes=15,
        embed_dim=384,
        depth=12,
        num_heads=6,
        num_group=128,
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
def point_mae_base_scanobjectnn_hardest_clf(**kwargs: Any) -> PointMAEClassification:
    return PointMAEClassification(**kwargs)


@register_model(
    "point-mae-base.shapenetpart.yatian-pang",
    task="segmentation",
    weights=WeightsDict(
        url="hf://torch-pointcloud/point-mae/point-mae-base.shapenetpart.yatian-pang.safetensors",
        dataset="shapenetpart",
        metrics={"ins_mIoU": 86.05, "cls_mIoU": 84.12},
        author="yatian-pang",
        license="MIT",
    ),
    transform=T.Compose(
        [
            T.Rescale(keys=DataKeys.POS, method="centroid"),
            T.CopyItems(keys=[DataKeys.POS, DataKeys.SEGMENT], names=[DataKeys.ORIGIN_POS, DataKeys.ORIGIN_SEGMENT]),
            T.FarthestPointSample(
                pos_key=DataKeys.POS,
                keys=[DataKeys.NORMAL, DataKeys.SEGMENT],
                num_samples=2048,
                random_start=False,
                dst_index_key=DataKeys.INDEX,
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
        num_group=128,
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
def point_mae_base_shapenetpart_seg(**kwargs: Any) -> PointMAESegmentation:
    return PointMAESegmentation(**kwargs)


@register_model(
    "point-mae-base.pretrain.yatian-pang",
    task="base",
    weights=WeightsDict(
        url="hf://torch-pointcloud/point-mae/point-mae-base.pretrain.yatian-pang.safetensors",
        dataset="shapenet55",
        author="yatian-pang",
        license="MIT",
    ),
    hparams=dict(
        in_channels=0,
        embed_dim=384,
        encoder_depth=12,
        decoder_depth=4,
        num_heads=6,
        decoder_num_heads=6,
        num_group=64,
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
def point_mae_base_pretrain(**kwargs: Any) -> PointMAEMaskedAutoEncoder:
    return PointMAEMaskedAutoEncoder(**kwargs)
