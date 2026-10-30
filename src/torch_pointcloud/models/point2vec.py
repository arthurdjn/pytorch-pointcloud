"""point2vec classification, part-segmentation and pretraining models.

{{ paper("2303.16570") }}
"""

from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.nn import MLP

import torch_pointcloud.transforms as T
from torch_pointcloud.datasets.modelnet import MODELNET40_CLASSES
from torch_pointcloud.datasets.scanobjectnn import SCANOBJECTNN_CLASSES
from torch_pointcloud.layers import PointNet2FeaturePropagation, PointPatchEmbed
from torch_pointcloud.ops.cluster import group
from torch_pointcloud.ops.masking import random_mask
from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.state_dict import transform_state_dict
from torch_pointcloud.utils.types import OptTensor

from ._base import ClassificationModel, PartSegmentationModel, PretrainingModel
from ._registry import WeightsDict, register_model
from .point_mae import TransformerEncoder


class Point2VecClassification(ClassificationModel):
    r"""point2vec classification model, as in :arxiv: [Point2Vec for Self-Supervised Representation Learning on Point
    Clouds](https://arxiv.org/abs/2303.16570), adapted from
    :github: [karimknaebel/point2vec](https://github.com/karimknaebel/point2vec).

    Tokenizes local groups with a mini-PointNet encoder, processes the tokens with a plain transformer encoder whose
    attention projections carry a bias, and classifies the concatenation of the max- and mean-pooled tokens. The
    layers followed by a batch normalization (in the patch embedder and the head) have no bias.

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
        drop_path: float = 0.2,
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
        self.pos_embed = self.configure_pos_embed()
        self.blocks = self.configure_blocks()
        self.norm_f = nn.LayerNorm(embed_dim)
        self.head = self.configure_head()

    def configure_encoder(self) -> PointPatchEmbed:
        """Build the mini-PointNet patch embedder tokenizing each local group."""
        return PointPatchEmbed(
            embed_dim=self.embed_dim,
            in_channels=self.spatial_dim + self.in_channels,
            local_channels=self.encoder_local_channels,
            global_channels=self.encoder_global_channels,
            bias=False,
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
            num_heads=self.num_heads,
            qkv_bias=True,
            drop_path=dpr,
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
            bias=[False, False, True],
            plain_last=True,
        )

    def reset_classifier(self, num_classes: int, global_pool: Any = None, **kwargs: Any) -> None:
        if global_pool is not None:
            raise ValueError(
                f"{self.__class__.__name__} pools with a fixed max-pool + mean-pool concatenation; "
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
        pos_embed = self.pos_embed(center)

        x = self.blocks(group_input_tokens, pos_embed)
        x = self.norm_f(x)
        return x

    def forward_head(self, x: Tensor, pre_logits: bool = False) -> Tensor:
        concat_f = torch.cat([x.max(1)[0], x.mean(1)], dim=-1)
        return concat_f if pre_logits else self.head(concat_f)

    def forward(self, x: OptTensor, pos: Tensor, batch: Tensor) -> Tensor:
        x = self.forward_features(x, pos, batch)
        return self.forward_head(x)


class Point2VecPartSegmentation(PartSegmentationModel):
    r"""point2vec part-segmentation model, as in :arxiv: [Point2Vec for Self-Supervised Representation Learning on
    Point Clouds](https://arxiv.org/abs/2303.16570), adapted from
    :github: [karimknaebel/point2vec](https://github.com/karimknaebel/point2vec).

    Tokenizes local groups with a mini-PointNet encoder, processes them with a plain transformer encoder, averages
    the layer-normalized hidden states at three block depths, and propagates the group features back to every point
    with a PointNet++-style feature propagation (`PointNet2FeaturePropagation`) over the point coordinates. A global
    branch made of the max- and mean-pooled group features and the embedded category one-hot is fused before the
    per-point classifier.

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
        propagation_channels: Hidden widths of the feature-propagation MLP.
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
        propagation_channels: Sequence[int] = (384, 384),
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
        self.propagation_channels = propagation_channels
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
            bias=False,
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
            num_heads=self.num_heads,
            qkv_bias=True,
            drop_path=dpr,
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

    def configure_propagation_0(self) -> PointNet2FeaturePropagation:
        """Build the feature-propagation module interpolating group features back to every point."""
        return PointNet2FeaturePropagation(
            channels=[self.embed_dim + self.spatial_dim, *self.propagation_channels],
            k=3,
            act="relu",
            norm="batch_norm",
            bias=False,
            plain_last=False,
            weighting="squared",
            eps=torch.finfo(torch.float32).eps,
        )

    @property
    def num_features(self) -> int:
        """Channel count $C$ of the per-point features entering the head."""
        return self.propagation_channels[-1] + 2 * self.embed_dim + 64

    def configure_head(self) -> nn.Module:
        if self.num_classes == 0:
            return nn.Identity()
        return MLP(
            [self.num_features, 512, 256, self.num_classes],
            act="relu",
            norm="batch_norm",
            dropout=[self.dropout, 0.0, 0.0],
            bias=[False, False, True],
            plain_last=True,
        )

    def reset_classifier(self, num_classes: int, **kwargs: Any) -> None:
        self.num_classes = num_classes
        self.head = self.configure_head()

    def forward_features(self, x: OptTensor, pos: Tensor, batch: Tensor) -> Tuple[Tensor, Tensor]:
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
        pos_embed = self.pos_embed(center)

        feature_list = self.blocks(group_input_tokens, pos_embed, fetch_idx=self.fetch_idx)
        x_feat = torch.stack([F.layer_norm(feat, feat.shape[-1:]) for feat in feature_list]).mean(0)
        return x_feat, center

    def forward_decoder(self, x_feat: Tensor, pos: Tensor, batch: Tensor, center: Tensor, category: Tensor) -> Tensor:
        B, G, C = x_feat.shape
        x_global = torch.cat([x_feat.max(1)[0], x_feat.mean(1), self.label_conv(category)], dim=-1)

        group_batch = torch.arange(B, device=pos.device).repeat_interleave(G)
        x_groups = x_feat.reshape(B * G, C)
        center_pos = center.reshape(B * G, self.spatial_dim)
        x, _, _ = self.propagation_0(x_groups, center_pos, group_batch, pos, pos, batch)
        return torch.cat([x, x_global[batch]], dim=-1)

    def forward_head(self, x: Tensor, pre_logits: bool = False) -> Tensor:
        return x if pre_logits else self.head(x)

    def forward(self, x: OptTensor, pos: Tensor, batch: Tensor, category: Tensor) -> Tensor:
        x_feat, center = self.forward_features(x, pos, batch)
        x = self.forward_decoder(x_feat, pos, batch, center, category)
        return self.forward_head(x)


class Point2VecPretraining(PretrainingModel):
    r"""point2vec self-distillation pretraining model, as in :arxiv: [Point2Vec for Self-Supervised Representation
    Learning on Point Clouds](https://arxiv.org/abs/2303.16570), adapted from
    :github: [karimknaebel/point2vec](https://github.com/karimknaebel/point2vec).

    A student encodes the visible group tokens, and a shallow decoder fills learnable mask tokens at the positions of
    the masked groups before a linear regressor predicts their teacher representation. The teacher, an exponential
    moving average of the student blocks kept in `teacher` and moved by `update_teacher` after each optimizer step,
    encodes every token without gradient; its target
    is the layer-normalized mean of its layer-normalized hidden states at `target_layers`. `forward` returns the
    predictions and targets of the masked tokens.

    Args:
        in_channels: Number of input channels (unused beyond coordinates; kept for the registry contract).
        embed_dim: Token channels.
        encoder_depth: Number of student and teacher transformer blocks.
        decoder_depth: Number of decoder transformer blocks.
        num_heads: Number of attention heads.
        num_groups: Number of groups (FPS centers) per sample.
        group_size: Number of neighbors per group.
        target_layers: Teacher blocks whose hidden states are averaged into the targets.
        encoder_local_channels: Hidden widths of the patch embedder's per-point MLP.
        encoder_global_channels: Hidden widths of the patch embedder's per-group MLP.
        pos_embed_channels: Hidden widths of the positional-embedding MLP.
        mask_ratio: Fraction of tokens to mask.
        drop_path: Stochastic-depth drop-path rate of the student and the decoder.
        attn_dropout: Dropout applied to the attention weights of the student and the decoder.
        act: Activation function.
        act_kwargs: Extra arguments for the activation.
        norm: Normalization function.
        norm_kwargs: Extra arguments for the normalization.
        spatial_dim: Spatial dimension of the input point cloud.

    Shape:
        - Input: $(N, 3)$ and $(N,)$.
        - Output: predictions and targets, each of shape $(B \cdot G_\text{mask}, C)$.
    """

    def __init__(
        self,
        in_channels: int,
        *,
        target_layers: Sequence[int],
        embed_dim: int = 384,
        encoder_depth: int = 12,
        decoder_depth: int = 4,
        num_heads: int = 6,
        num_groups: int = 64,
        group_size: int = 32,
        encoder_local_channels: Sequence[int] = (128, 256),
        encoder_global_channels: Sequence[int] = (512,),
        pos_embed_channels: Sequence[int] = (128,),
        mask_ratio: float = 0.65,
        drop_path: float = 0.25,
        attn_dropout: float = 0.05,
        act: Union[str, Callable, None] = "gelu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = nn.LayerNorm,
        norm_kwargs: Optional[Dict[str, Any]] = None,
        spatial_dim: int = 3,
    ) -> None:
        super().__init__(in_channels=in_channels)
        if not 0.0 < mask_ratio < 1.0:
            raise ValueError(f"`mask_ratio` must be in (0, 1), got {mask_ratio}.")
        if not target_layers or max(target_layers) >= encoder_depth:
            raise ValueError(f"`target_layers` must index blocks below {encoder_depth}, got {list(target_layers)}.")

        self.embed_dim = embed_dim
        self.encoder_depth = encoder_depth
        self.decoder_depth = decoder_depth
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.group_size = group_size
        self.target_layers = tuple(target_layers)
        self.encoder_local_channels = encoder_local_channels
        self.encoder_global_channels = encoder_global_channels
        self.pos_embed_channels = pos_embed_channels
        self.mask_ratio = mask_ratio
        self.drop_path = drop_path
        self.attn_dropout = attn_dropout
        self.act = act
        self.act_kwargs = act_kwargs
        self.norm = norm
        self.norm_kwargs = norm_kwargs
        self.spatial_dim = spatial_dim

        self.encoder = PointPatchEmbed(
            embed_dim=embed_dim,
            local_channels=encoder_local_channels,
            global_channels=encoder_global_channels,
            bias=False,
        )
        self.pos_embed = MLP(
            [spatial_dim, *pos_embed_channels, embed_dim], act=act, act_kwargs=act_kwargs, norm=None, plain_last=True
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.blocks = self.configure_blocks(encoder_depth, drop_path, attn_dropout)
        self.norm_f = nn.LayerNorm(embed_dim)
        self.teacher = self.configure_blocks(encoder_depth, 0.0, 0.0)
        self.teacher.requires_grad_(False)
        self.decoder = self.configure_blocks(decoder_depth, drop_path, attn_dropout)
        self.decoder_norm = nn.LayerNorm(embed_dim)
        self.regressor = nn.Linear(embed_dim, embed_dim)
        self.reset_parameters()

    def configure_blocks(self, depth: int, drop_path: float, attn_dropout: float) -> TransformerEncoder:
        """Build a transformer encoder with a linearly scaled stochastic-depth schedule."""
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        return TransformerEncoder(
            embed_dim=self.embed_dim,
            depth=depth,
            num_heads=self.num_heads,
            qkv_bias=True,
            attn_dropout=attn_dropout,
            drop_path=dpr,
            act=self.act,
            act_kwargs=self.act_kwargs,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
        )

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.mask_token, std=0.02, a=-0.02, b=0.02)
        self.teacher.load_state_dict(self.blocks.state_dict())

    def forward(self, x: OptTensor, pos: Tensor, batch: Tensor) -> Tuple[Tensor, Tensor]:
        neighborhood, center = group(pos, batch, self.num_groups, self.group_size, random_start=self.training)
        B, G, _ = center.shape
        mask = random_mask(B, G, self.mask_ratio, device=center.device)

        tokens = self.encoder(neighborhood)
        C = tokens.shape[-1]
        pos_embed = self.pos_embed(center)
        pos_vis = pos_embed[~mask].reshape(B, -1, C)
        pos_mask = pos_embed[mask].reshape(B, -1, C)
        M = pos_mask.shape[1]

        x_vis = self.norm_f(self.blocks(tokens[~mask].reshape(B, -1, C), pos_vis))
        x_full = torch.cat([x_vis, self.mask_token.expand(B, M, -1)], dim=1)
        x_rec = self.decoder_norm(self.decoder(x_full, torch.cat([pos_vis, pos_mask], dim=1)))
        predictions = self.regressor(x_rec[:, -M:]).reshape(-1, C)

        with torch.no_grad():
            hidden = self.teacher(tokens, pos_embed, fetch_idx=self.target_layers)
            targets = torch.stack([F.layer_norm(h, h.shape[-1:]) for h in hidden]).mean(0)
            targets = F.layer_norm(targets, targets.shape[-1:])[mask]
        return predictions, targets

    @torch.no_grad()
    def update_teacher(self, decay: float) -> None:
        r"""Move the teacher one exponential-moving-average step toward the student blocks.

        Each teacher parameter becomes $\tau \theta_\text{teacher} + (1 - \tau) \theta_\text{student}$. The teacher
        tracks `blocks` only, not the patch embedder, the positional embedding or `norm_f`. Call it after each
        optimizer step.

        Args:
            decay: The decay $\tau$ in $[0, 1]$; $0$ copies the student blocks, $1$ keeps the teacher.

        Example:
            ```python
            from torch_pointcloud.models.point2vec import Point2VecPretraining

            model = Point2VecPretraining(in_channels=0, target_layers=range(6, 12))
            model.update_teacher(decay=0.9998)
            ```
        """
        for teacher, student in zip(self.teacher.parameters(), self.blocks.parameters()):
            teacher.lerp_(student, 1.0 - decay)


_CHECKPOINT_MAPPING = {
    "tokenizer.embedding.first_conv.0.{param}": "encoder.local_mlp.lins.0.{param}",
    "tokenizer.embedding.first_conv.1.{param}": "encoder.local_mlp.norms.0.module.{param}",
    "tokenizer.embedding.first_conv.3.{param}": "encoder.local_mlp.lins.1.{param}",
    "tokenizer.embedding.second_conv.0.{param}": "encoder.global_mlp.lins.0.{param}",
    "tokenizer.embedding.second_conv.1.{param}": "encoder.global_mlp.norms.0.module.{param}",
    "tokenizer.embedding.second_conv.3.{param}": "encoder.global_mlp.lins.1.{param}",
    "positional_encoding.0.{param}": "pos_embed.lins.0.{param}",
    "positional_encoding.2.{param}": "pos_embed.lins.1.{param}",
    "encoder.blocks.{i}.mlp.fc1.{param}": "blocks.blocks.{i}.mlp.lins.0.{param}",
    "encoder.blocks.{i}.mlp.fc2.{param}": "blocks.blocks.{i}.mlp.lins.1.{param}",
    "encoder.blocks.{i}.attn.{layer}.{param}": "blocks.blocks.{i}.attn.{layer}.{param}",
    "encoder.blocks.{i}.{layer}.{param}": "blocks.blocks.{i}.{layer}.{param}",
    "student.blocks.{i}.mlp.fc1.{param}": "blocks.blocks.{i}.mlp.lins.0.{param}",
    "student.blocks.{i}.mlp.fc2.{param}": "blocks.blocks.{i}.mlp.lins.1.{param}",
    "student.blocks.{i}.attn.{layer}.{param}": "blocks.blocks.{i}.attn.{layer}.{param}",
    "student.blocks.{i}.{layer}.{param}": "blocks.blocks.{i}.{layer}.{param}",
    "teacher.ema_model.blocks.{i}.mlp.fc1.{param}": "teacher.blocks.{i}.mlp.lins.0.{param}",
    "teacher.ema_model.blocks.{i}.mlp.fc2.{param}": "teacher.blocks.{i}.mlp.lins.1.{param}",
    "teacher.ema_model.blocks.{i}.attn.{layer}.{param}": "teacher.blocks.{i}.attn.{layer}.{param}",
    "teacher.ema_model.blocks.{i}.{layer}.{param}": "teacher.blocks.{i}.{layer}.{param}",
    "decoder.blocks.{i}.mlp.fc1.{param}": "decoder.blocks.{i}.mlp.lins.0.{param}",
    "decoder.blocks.{i}.mlp.fc2.{param}": "decoder.blocks.{i}.mlp.lins.1.{param}",
    "decoder.blocks.{i}.attn.{layer}.{param}": "decoder.blocks.{i}.attn.{layer}.{param}",
    "decoder.blocks.{i}.{layer}.{param}": "decoder.blocks.{i}.{layer}.{param}",
    "encoder.norm.{param}": "norm_f.{param}",
    "student.norm.{param}": "norm_f.{param}",
    "decoder.norm.{param}": "decoder_norm.{param}",
    "cls_head.0.{param}": "head.lins.0.{param}",
    "cls_head.1.{param}": "head.norms.0.module.{param}",
    "cls_head.4.{param}": "head.lins.1.{param}",
    "cls_head.5.{param}": "head.norms.1.module.{param}",
    "cls_head.8.{param}": "head.lins.2.{param}",
    "label_embedding.0.{param}": "label_conv.lins.0.{param}",
    "label_embedding.1.{param}": "label_conv.norms.0.module.{param}",
    "upsampling.mlp_convs.{i}.{param}": "propagation_0.mlp.lins.{i}.{param}",
    "upsampling.mlp_bns.{i}.{param}": "propagation_0.mlp.norms.{i}.module.{param}",
    "seg_head.0.{param}": "head.lins.0.{param}",
    "seg_head.1.{param}": "head.norms.0.module.{param}",
    "seg_head.4.{param}": "head.lins.1.{param}",
    "seg_head.5.{param}": "head.norms.1.module.{param}",
    "seg_head.7.{param}": "head.lins.2.{param}",
}
# The copy of the student and the moving-average counters held by the teacher of a pretraining checkpoint.
_DROPPED_KEYS = ("teacher.online_model.", "teacher.ema_model.norm.", "teacher.initted", "teacher.step")


def checkpoint_filter_fn(state_dict: Dict[str, Any], model: nn.Module) -> Dict[str, Any]:
    r"""Rename the keys of a released point2vec Lightning checkpoint to the `Point2VecClassification`,
    `Point2VecPartSegmentation` or `Point2VecPretraining` ones, as the `pretrained_filter_fn` of a `WeightsDict`.

    The $1 \times 1$ convolution weights become linear weights, the first feature-propagation layer swaps its input
    columns from `[coordinates, interpolated]` to the `[interpolated, coordinates]` order of
    `PointNet2FeaturePropagation`, and the encoder's final normalization is dropped when `model` has no `norm_f`
    (the part-segmentation model normalizes its hidden states without parameters). In a pretraining checkpoint the
    student becomes the encoder, and the copy of the student and the moving-average counters held by the teacher are
    dropped.

    Args:
        state_dict: State dict keyed as in the released checkpoint file.
        model: Model the state dict is loaded into.

    Returns:
        The state dict keyed as `model` expects.
    """
    dropped = _DROPPED_KEYS if hasattr(model, "norm_f") else (*_DROPPED_KEYS, "encoder.norm.")
    state_dict = {key: value for key, value in state_dict.items() if not key.startswith(dropped)}
    state_dict = transform_state_dict(
        state_dict,
        _CHECKPOINT_MAPPING,
        value_transform=lambda value: value.squeeze(-1) if value.dim() == 3 and value.shape[-1] == 1 else value,
    )
    if "mask_token" in state_dict:
        state_dict["mask_token"] = state_dict["mask_token"].reshape(1, 1, -1)
    if "propagation_0.mlp.lins.0.weight" in state_dict:
        weight = state_dict["propagation_0.mlp.lins.0.weight"]
        spatial_dim = getattr(model, "spatial_dim", 3)
        state_dict["propagation_0.mlp.lins.0.weight"] = torch.cat([weight[:, spatial_dim:], weight[:, :spatial_dim]], 1)
    return state_dict


_POINT2VEC_REVISION = "35ae2318f870390b954588b6018729a8d7ddb2f3"


@register_model(
    "point2vec.modelnet40.karim-knaebel",
    task="classification",
    weights=WeightsDict(
        url=f"hf://rwthvision/point2vec/resolve/{_POINT2VEC_REVISION}/fine_modelnet40-epoch.125-step.38682-val_acc.0.9465.ckpt",
        pretrained_filter_fn=checkpoint_filter_fn,
        dataset="modelnet40",
        metrics={"OA": 94.33, "mAcc": 92.14},
        classes=MODELNET40_CLASSES,
        author="karim-knaebel",
        license="MIT",
    ),
    transform=T.Compose(
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
    ),
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
def point2vec_modelnet40_clf(**kwargs: Any) -> Point2VecClassification:
    return Point2VecClassification(**kwargs)


@register_model(
    "point2vec.scanobjectnn-hardest.karim-knaebel",
    task="classification",
    weights=WeightsDict(
        url=f"hf://rwthvision/point2vec/resolve/{_POINT2VEC_REVISION}/fine_scanobjectnn-epoch.146-step.52332-val_acc.0.8747.ckpt",
        pretrained_filter_fn=checkpoint_filter_fn,
        dataset="scanobjectnn-hardest",
        metrics={"OA": 86.92, "mAcc": 85.23},
        classes=SCANOBJECTNN_CLASSES,
        author="karim-knaebel",
        license="MIT",
    ),
    transform=T.Rescale(keys=DataKeys.POS, method="centroid"),
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
def point2vec_scanobjectnn_hardest_clf(**kwargs: Any) -> Point2VecClassification:
    return Point2VecClassification(**kwargs)


@register_model(
    "point2vec.shapenetpart.karim-knaebel",
    task="part-segmentation",
    input_keys=("x", "pos", "batch", "category"),
    weights=WeightsDict(
        url=f"hf://rwthvision/point2vec/resolve/{_POINT2VEC_REVISION}/fine_shapenetpart-epoch.288-step.252586-val_cat_miou.0.8459.ckpt",
        pretrained_filter_fn=checkpoint_filter_fn,
        dataset="shapenetpart",
        metrics={"ins_mIoU": 85.97, "cls_mIoU": 84.38},
        author="karim-knaebel",
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
        propagation_channels=(384, 384),
        drop_path=0.1,
        dropout=0.5,
        act="gelu",
        spatial_dim=3,
    ),
)
def point2vec_shapenetpart_seg(**kwargs: Any) -> Point2VecPartSegmentation:
    return Point2VecPartSegmentation(**kwargs)


@register_model(
    "point2vec.pretrain.karim-knaebel",
    task="pretraining",
    weights=WeightsDict(
        url=f"hf://rwthvision/point2vec/resolve/{_POINT2VEC_REVISION}/pre_point2vec-epoch.799-step.64800.ckpt",
        pretrained_filter_fn=checkpoint_filter_fn,
        dataset="shapenet55",
        author="karim-knaebel",
        license="MIT",
    ),
    hparams=dict(
        in_channels=0,
        target_layers=(6, 7, 8, 9, 10, 11),
        embed_dim=384,
        encoder_depth=12,
        decoder_depth=4,
        num_heads=6,
        num_groups=64,
        group_size=32,
        encoder_local_channels=(128, 256),
        encoder_global_channels=(512,),
        pos_embed_channels=(128,),
        mask_ratio=0.65,
        drop_path=0.25,
        attn_dropout=0.05,
        act="gelu",
        spatial_dim=3,
    ),
)
def point2vec_pretrain(**kwargs: Any) -> Point2VecPretraining:
    return Point2VecPretraining(**kwargs)
