"""PCP-MAE classification and pretraining models.

{{ paper("2408.08753") }}
"""

from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn
from torch_geometric.nn import MLP

import torch_pointcloud.transforms as T
from torch_pointcloud.datasets.modelnet import MODELNET40_CLASSES
from torch_pointcloud.datasets.scanobjectnn import SCANOBJECTNN_CLASSES
from torch_pointcloud.layers import PointPatchEmbed, PositionEmbeddingSine
from torch_pointcloud.ops.cluster import group
from torch_pointcloud.ops.masking import random_mask
from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.types import OptTensor

from ._base import ClassificationModel, PretrainingModel
from ._registry import WeightsDict, register_model
from .point_mae import TransformerDecoder, TransformerEncoder


class PCPMAEClassification(ClassificationModel):
    r"""PCP-MAE classification model, as in :arxiv: [PCP-MAE: Learning to Predict Centers for Point Masked
    Autoencoders](https://arxiv.org/abs/2408.08753), adapted from
    :github: [aHapBean/PCP-MAE](https://github.com/aHapBean/PCP-MAE).

    Tokenizes local groups with a mini-PointNet encoder, prepends a class token, processes the sequence with a plain
    transformer encoder, and classifies the concatenation of the class token and the max-pooled patch tokens. The
    patch centers are embedded by a fixed sinusoidal embedding (`PositionEmbeddingSine`) refined by an MLP, the
    positional embedding of the pretraining stage; the center-prediction branch of that stage is not part of the
    classifier.

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
        pos_embed_channels: Hidden widths of the MLP refining the sinusoidal positional embedding.
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
        pos_embed_channels: Sequence[int] = (384,),
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

    def configure_pos_embed(self) -> nn.Sequential:
        """Build the sinusoidal positional embedding of the group centers and the MLP refining it."""
        return nn.Sequential(
            PositionEmbeddingSine(self.embed_dim, self.spatial_dim),
            MLP(
                [self.embed_dim, *self.pos_embed_channels, self.embed_dim],
                act=self.act,
                act_kwargs=self.act_kwargs,
                norm=None,
                plain_last=True,
            ),
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


class PCPMAEPretraining(PretrainingModel):
    r"""PCP-MAE masked-autoencoder pretraining model, as in :arxiv: [PCP-MAE: Learning to Predict Centers for Point
    Masked Autoencoders](https://arxiv.org/abs/2408.08753), adapted from
    :github: [aHapBean/PCP-MAE](https://github.com/aHapBean/PCP-MAE).

    Masks a fraction of the group tokens. The encoder runs the visible tokens with their positional embedding next to
    the masked tokens with a shared learnable position token instead: the masked tokens attend to every token, the
    visible ones only to each other, and both share the transformer weights. A projection of the encoded masked
    tokens predicts the sinusoidal embedding of their centers (`PositionEmbeddingSine`), which, refined by the
    positional MLP and without gradient, positions the mask tokens of a Point-MAE decoder reconstructing the centered
    coordinates of the masked groups. `forward` returns the predicted and target coordinates of the masked groups,
    each packed with the group index of its points, followed by the predicted and target center embeddings.

    Args:
        in_channels: Number of input channels (unused beyond coordinates; kept for the registry contract).
        embed_dim: Token channels.
        encoder_depth: Number of encoder transformer blocks.
        decoder_depth: Number of decoder transformer blocks.
        num_heads: Number of encoder attention heads.
        decoder_num_heads: Number of decoder attention heads.
        num_groups: Number of groups (FPS centers) per sample.
        group_size: Number of neighbors per group.
        encoder_local_channels: Hidden widths of the patch embedder's per-point MLP.
        encoder_global_channels: Hidden widths of the patch embedder's per-group MLP.
        pos_embed_channels: Hidden widths of the MLP refining the sinusoidal positional embedding.
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
          predicted and target center embeddings $(B \cdot G_\text{mask}, C)$.
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
        num_groups: int = 64,
        group_size: int = 32,
        encoder_local_channels: Sequence[int] = (128, 256),
        encoder_global_channels: Sequence[int] = (512,),
        pos_embed_channels: Sequence[int] = (384,),
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
        self.pos_embed = nn.Sequential(
            PositionEmbeddingSine(embed_dim, spatial_dim),
            MLP(
                [embed_dim, *pos_embed_channels, embed_dim], act=act, act_kwargs=act_kwargs, norm=None, plain_last=True
            ),
        )
        self.mask_pos_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.blocks = self.configure_blocks()
        self.norm_f = nn.LayerNorm(embed_dim)
        self.center_proj = MLP(
            [embed_dim, embed_dim, embed_dim],
            act="relu",
            norm="layer_norm",
            norm_kwargs=dict(mode="node"),
            plain_last=True,
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.decoder = self.configure_decoder()
        self.increase_dim = MLP([embed_dim, 3 * group_size], act=None, norm=None, bias=True, plain_last=True)
        self.reset_parameters()

    def configure_blocks(self) -> TransformerEncoder:
        """Build the transformer encoder with a linearly scaled stochastic-depth schedule."""
        dpr = [x.item() for x in torch.linspace(0, self.drop_path, self.encoder_depth)]
        return TransformerEncoder(
            embed_dim=self.embed_dim,
            depth=self.encoder_depth,
            drop_path=dpr,
            num_heads=self.num_heads,
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
        nn.init.trunc_normal_(self.mask_pos_token, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(
        self, x: OptTensor, pos: Tensor, batch: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        neighborhood, center = group(pos, batch, self.num_groups, self.group_size, random_start=self.training)
        B, G, _ = center.shape
        mask = random_mask(B, G, self.mask_ratio, device=center.device)

        tokens = self.encoder(neighborhood)
        C = tokens.shape[-1]
        x_vis = tokens[~mask].reshape(B, -1, C)
        x_mask = tokens[mask].reshape(B, -1, C)
        V, M = x_vis.shape[1], x_mask.shape[1]
        pos_vis = self.pos_embed(center[~mask].reshape(B, V, self.spatial_dim))

        x = torch.cat([x_vis, x_mask], dim=1)
        pos_embed = torch.cat([pos_vis, self.mask_pos_token.expand(B, M, -1)], dim=1)
        attn_mask = torch.zeros(V + M, V + M, device=x.device)
        attn_mask[:V, V:] = float("-inf")
        for block in self.blocks.blocks:
            x = block(x + pos_embed, attn_mask)
        x = self.norm_f(x)
        x_vis, x_mask = x[:, :V], x[:, V:]

        center_pred = self.center_proj(x_mask)
        center_target = self.pos_embed[0](center[mask].reshape(B, M, self.spatial_dim))
        pos_mask = self.pos_embed[1](center_pred.detach())

        x_full = torch.cat([x_vis, self.mask_token.expand(B, M, -1)], dim=1)
        x_rec = self.decoder(x_full, torch.cat([pos_vis, pos_mask], dim=1), M)

        pred = self.increase_dim(x_rec).reshape(-1, 3)
        target = neighborhood[mask].reshape(-1, 3)
        point_batch = torch.arange(B * M, device=pred.device).repeat_interleave(self.group_size)
        return pred, target, point_batch, point_batch, center_pred.reshape(-1, C), center_target.reshape(-1, C)


@register_model(
    "pcp-mae.modelnet40.xiangdong-zhang",
    task="classification",
    weights=WeightsDict(
        url="hf://torch-pointcloud/pcp-mae.modelnet40.xiangdong-zhang/resolve/0000000000000000000000000000000000000000/model.safetensors",
        dataset="modelnet40",
        metrics={"OA": 93.64, "mAcc": 90.60},
        classes=MODELNET40_CLASSES,
        author="xiangdong-zhang",
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
        pos_embed_channels=(384,),
        drop_path=0.2,
        dropout=0.5,
        act="gelu",
        spatial_dim=3,
    ),
)
def pcp_mae_modelnet40_clf(**kwargs: Any) -> PCPMAEClassification:
    return PCPMAEClassification(**kwargs)


@register_model(
    "pcp-mae.scanobjectnn-objbg.xiangdong-zhang",
    task="classification",
    weights=WeightsDict(
        url="hf://torch-pointcloud/pcp-mae.scanobjectnn-objbg.xiangdong-zhang/resolve/0000000000000000000000000000000000000000/model.safetensors",
        dataset="scanobjectnn-objbg",
        metrics={"OA": 95.52, "mAcc": 94.82},
        classes=SCANOBJECTNN_CLASSES,
        author="xiangdong-zhang",
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
        pos_embed_channels=(384,),
        drop_path=0.1,
        dropout=0.5,
        act="gelu",
        spatial_dim=3,
    ),
)
def pcp_mae_scanobjectnn_objbg_clf(**kwargs: Any) -> PCPMAEClassification:
    return PCPMAEClassification(**kwargs)


@register_model(
    "pcp-mae.scanobjectnn-objonly.xiangdong-zhang",
    task="classification",
    weights=WeightsDict(
        url="hf://torch-pointcloud/pcp-mae.scanobjectnn-objonly.xiangdong-zhang/resolve/0000000000000000000000000000000000000000/model.safetensors",
        dataset="scanobjectnn-objonly",
        metrics={"OA": 94.32, "mAcc": 94.06},
        classes=SCANOBJECTNN_CLASSES,
        author="xiangdong-zhang",
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
        pos_embed_channels=(384,),
        drop_path=0.1,
        dropout=0.5,
        act="gelu",
        spatial_dim=3,
    ),
)
def pcp_mae_scanobjectnn_objonly_clf(**kwargs: Any) -> PCPMAEClassification:
    return PCPMAEClassification(**kwargs)


@register_model(
    "pcp-mae.scanobjectnn-hardest.xiangdong-zhang",
    task="classification",
    weights=WeightsDict(
        url="hf://torch-pointcloud/pcp-mae.scanobjectnn-hardest.xiangdong-zhang/resolve/0000000000000000000000000000000000000000/model.safetensors",
        dataset="scanobjectnn-hardest",
        metrics={"OA": 90.28, "mAcc": 89.42},
        classes=SCANOBJECTNN_CLASSES,
        author="xiangdong-zhang",
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
        pos_embed_channels=(384,),
        drop_path=0.1,
        dropout=0.5,
        act="gelu",
        spatial_dim=3,
    ),
)
def pcp_mae_scanobjectnn_hardest_clf(**kwargs: Any) -> PCPMAEClassification:
    return PCPMAEClassification(**kwargs)


@register_model(
    "pcp-mae.pretrain.xiangdong-zhang",
    task="pretraining",
    weights=WeightsDict(
        url="hf://torch-pointcloud/pcp-mae.pretrain.xiangdong-zhang/resolve/0000000000000000000000000000000000000000/model.safetensors",
        dataset="shapenet55",
        author="xiangdong-zhang",
        license="MIT",
    ),
    hparams=dict(
        in_channels=0,
        embed_dim=384,
        encoder_depth=12,
        decoder_depth=4,
        num_heads=6,
        decoder_num_heads=6,
        num_groups=64,
        group_size=32,
        encoder_local_channels=(128, 256),
        encoder_global_channels=(512,),
        pos_embed_channels=(384,),
        mask_ratio=0.6,
        drop_path=0.1,
        act="gelu",
        spatial_dim=3,
    ),
)
def pcp_mae_pretrain(**kwargs: Any) -> PCPMAEPretraining:
    return PCPMAEPretraining(**kwargs)
