"""SPFormer superpoint transformer for instance segmentation.

{{ paper("2211.15766") }}
"""

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple, TypedDict, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import MLP

import torch_pointcloud.transforms as T
from torch_pointcloud.datasets.scannet import SCANNET20_CLASSES
from torch_pointcloud.layers import MaskedQueryDecoderLayer
from torch_pointcloud.models._base import InstanceSegmentationModel
from torch_pointcloud.models._registry import WeightsDict, register_model
from torch_pointcloud.models.spformer_unet import SPFormerUNetDecoder, SPFormerUNetEncoder
from torch_pointcloud.ops.superpoint import superpoint_pool
from torch_pointcloud.utils.conversion import ensure_tuple, ensure_tuple_size
from torch_pointcloud.utils.data import DataKeys, offset_index
from torch_pointcloud.utils.types import InstanceSegmentation3D

if TYPE_CHECKING:
    from spconv.pytorch import SparseConvTensor


class SPFormerPrediction(TypedDict):
    r"""Predictions of one SPFormer decoder layer for a batch of $B$ scenes, $Q$ queries and $S$ superpoints.

    Attributes:
        class_logits: Per-query class logits with a trailing no-object slot, shape $(B, Q, C + 1)$.
        scores: Per-query mask quality score, shape $(B, Q)$.
        mask_logits: Per-query mask logits over the superpoints of its scene, shape $(Q, S)$: column $s$ holds the
            logits of the $Q$ queries of the scene `superpoint_batch[s]`.
    """

    class_logits: Tensor
    scores: Tensor
    mask_logits: Tensor


class SPFormerOutput(SPFormerPrediction):
    r"""Raw SPFormer output: the last decoder layer's predictions, the earlier layers' and the superpoint layout.

    Attributes:
        superpoint: Packed superpoint index of every input point, shape $(N,)$, in $[0, S)$.
        superpoint_batch: Scene index of every superpoint, shape $(S,)$, sorted.
        aux_outputs: Predictions of the query embeddings and of every decoder layer but the last, in order.
    """

    superpoint: Tensor
    superpoint_batch: Tensor
    aux_outputs: List[SPFormerPrediction]


class SPFormerQueryDecoder(nn.Module):
    r"""SPFormer query decoder: learned queries predicting a class, a score and a superpoint mask each.

    Superpoint features are projected once into attention sources and once into mask embeddings. The learned
    queries are refined by `num_layers` `MaskedQueryDecoderLayer` blocks, without normalizing the cross-attention
    residual (`cross_attn_norm=False`); the queries before the first layer and after every layer predict class
    logits, a mask quality score and mask logits (the dot product of the normalized query with the mask embeddings).
    Each layer's cross-attention is restricted to the superpoints the previous prediction's mask covers (sigmoid
    $\geq 0.5$), or to all of them when it covers none.

    Args:
        in_channels: Width of the superpoint features.
        num_classes: Number of instance classes; the class head adds a no-object slot.
        num_queries: Number of learned queries, i.e. of instance candidates per scene.
        embed_dim: Query embedding dimension.
        num_layers: Number of decoder layers.
        num_heads: Number of attention heads.
        mlp_dim: Hidden width of the feed-forward blocks.
        dropout: Dropout probability.
        act: Activation of the feed-forward blocks.

    Shape:
        - Input: superpoint features $(S, \text{in\_channels})$ and their scene index $(S,)$, sorted.
        - Output: the predictions after every layer, see `SPFormerPrediction`.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        *,
        num_queries: int,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float = 0.0,
        act: Union[str, Callable, None] = "gelu",
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.embed_dim = embed_dim
        self.input_proj = nn.Sequential(nn.Linear(in_channels, embed_dim), nn.LayerNorm(embed_dim), nn.ReLU())
        self.mask_proj = MLP([in_channels, embed_dim, embed_dim], act="relu", norm=None)
        self.query_embed = nn.Embedding(num_queries, embed_dim)
        self.layers = nn.ModuleList(
            [
                MaskedQueryDecoderLayer(embed_dim, num_heads, mlp_dim, dropout=dropout, act=act, cross_attn_norm=False)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.class_head = self.configure_class_head()
        self.score_head = MLP([embed_dim, embed_dim, 1], act="relu", norm=None)

    def configure_class_head(self) -> nn.Module:
        """Build the class head for the current `num_classes`, with the trailing no-object slot."""
        return MLP([self.embed_dim, self.embed_dim, self.num_classes + 1], act="relu", norm=None)

    def predict(self, queries: Tensor, mask_features: Sequence[Tensor]) -> Tuple[SPFormerPrediction, List[Tensor]]:
        r"""Predict classes, scores and masks from the queries, and the attention masks of the next layer.

        Args:
            queries: Query embeddings, shape $(B, Q, D)$.
            mask_features: Mask embeddings of each scene's superpoints, shapes $(S_b, D)$.

        Returns:
            The `SPFormerPrediction` and the per-scene attention masks $(Q, S_b)$.
        """
        queries = self.norm(queries)
        mask_logits = [torch.einsum("nd,md->nm", query, features) for query, features in zip(queries, mask_features)]
        attn_masks = []
        for logits in mask_logits:
            attn_mask = logits.detach().sigmoid() < 0.5
            attn_mask[attn_mask.all(dim=-1)] = False
            attn_masks.append(attn_mask)
        prediction: SPFormerPrediction = {
            "class_logits": self.class_head(queries),
            "scores": self.score_head(queries).squeeze(-1),
            "mask_logits": torch.cat(mask_logits, dim=1),
        }
        return prediction, attn_masks

    def forward(self, x: Tensor, batch: Tensor) -> List[SPFormerPrediction]:
        sizes = torch.bincount(batch).tolist()
        sources = self.input_proj(x).split(sizes)
        mask_features = self.mask_proj(x).split(sizes)
        queries = self.query_embed.weight.repeat(len(sizes), 1, 1)

        prediction, attn_masks = self.predict(queries, mask_features)
        predictions = [prediction]
        for layer in self.layers:
            queries = layer(queries, sources, attn_masks)
            prediction, attn_masks = self.predict(queries, mask_features)
            predictions.append(prediction)
        return predictions


class SPFormerInstanceSegmentation(InstanceSegmentationModel):
    r"""SPFormer: superpoint-based transformer for 3D instance segmentation.

    Reference: :github: [sunjiahao1999/SPFormer](https://github.com/sunjiahao1999/SPFormer).
    The SPFormer sparse U-Net (`SPFormerUNetEncoder` and `SPFormerUNetDecoder`) encodes the voxels; the voxel
    features are gathered back to the points and averaged over each superpoint. A [`SPFormerQueryDecoder`](#)
    predicts `num_queries` instance candidates per scene, each a class, a mask quality score and a mask over the
    superpoints. `decode` keeps the `num_instances` best (query, class) pairs of every scene and expands their masks
    to the points.

    `forward` takes the voxelized scenes (`x`, `pos_grid` and the voxel `batch`) and three per-point tensors:
    `superpoint`, the superpoint id of every point within its scene (any integer ids), `inverse`, the voxel row of
    every point within its scene, and `batch_inverse`, the scene of every point. Collating with
    `cat_keys=[DataKeys.INVERSE]` produces `batch_inverse`.

    Args:
        in_channels: Number of input voxel feature channels.
        num_classes: Number of instance classes.
        channels: Per-level U-Net channel widths, deepest level last.
        depths: Residual blocks per U-Net level; an `int` is broadcast to every level.
        stem_kernel_size: Kernel size of the submanifold stem convolution.
        spatial_padding: Padding (in voxels) added to the inferred spatial shape.
        act: U-Net activation passed to `create_act`.
        act_kwargs: Extra keyword arguments for the U-Net activation.
        norm: U-Net normalization passed to `create_norm`.
        norm_kwargs: Extra keyword arguments for the U-Net normalization (e.g. `eps`, `momentum`).
        num_queries: Number of learned queries.
        embed_dim: Query embedding dimension.
        num_layers: Number of query decoder layers.
        num_heads: Number of attention heads.
        mlp_dim: Hidden width of the query decoder feed-forward blocks.
        dropout: Dropout probability of the query decoder.
        head_act: Activation of the query decoder feed-forward blocks.
        num_instances: Number of (query, class) pairs `decode` keeps per scene.

    Shape:
        - Input: voxel features $(V, \text{in\_channels})$, voxel grid coordinates $(V, 3)$, voxel batch $(V,)$,
          point superpoint ids $(N,)$, point-to-voxel `inverse` $(N,)$ and point batch $(N,)$.
        - Output: `SPFormerOutput`, decoded by `decode` into $(K, N)$ instance masks.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        *,
        channels: Sequence[int] = (32, 64, 96, 128, 160),
        depths: Union[int, Sequence[int]] = 2,
        stem_kernel_size: int = 3,
        spatial_padding: int = 96,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        num_queries: int = 400,
        embed_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        mlp_dim: int = 1024,
        dropout: float = 0.0,
        head_act: Union[str, Callable, None] = "gelu",
        num_instances: int = 100,
    ) -> None:
        super().__init__(in_channels=in_channels, num_classes=num_classes)
        self.channels = ensure_tuple(channels)
        self.depths = ensure_tuple_size(depths, size=len(self.channels), extra_msg="`depths` must match `channels`.")
        self.stem_kernel_size = stem_kernel_size
        self.spatial_padding = spatial_padding
        self.act = act
        self.act_kwargs = act_kwargs
        self.norm = norm
        self.norm_kwargs = norm_kwargs
        self.num_queries = num_queries
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.dropout = dropout
        self.head_act = head_act
        self.num_instances = num_instances

        self.encoder = self.configure_encoder()
        self.decoder = self.configure_decoder()
        self.head = self.configure_head()

    def configure_encoder(self) -> SPFormerUNetEncoder:
        """Build the sparse encoder producing the bottleneck features and the per-stage skips."""
        return SPFormerUNetEncoder(
            self.in_channels,
            self.channels,
            self.depths,
            stem_kernel_size=self.stem_kernel_size,
            spatial_padding=self.spatial_padding,
            act=self.act,
            act_kwargs=self.act_kwargs,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
        )

    def configure_decoder(self) -> SPFormerUNetDecoder:
        """Build the sparse decoder upsampling the bottleneck back to the input voxels."""
        return SPFormerUNetDecoder(
            self.channels,
            self.depths,
            act=self.act,
            act_kwargs=self.act_kwargs,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
        )

    def configure_head(self) -> SPFormerQueryDecoder:
        """Build the query decoder predicting the instances from the superpoint features."""
        return SPFormerQueryDecoder(
            self.num_features,
            self.num_classes,
            num_queries=self.num_queries,
            embed_dim=self.embed_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            mlp_dim=self.mlp_dim,
            dropout=self.dropout,
            act=self.head_act,
        )

    @property
    def num_features(self) -> int:
        """Channel count $C$ of the superpoint features entering the query decoder."""
        return self.channels[0]

    def reset_classifier(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.head.num_classes = num_classes
        self.head.class_head = self.head.configure_class_head()

    def forward_features(
        self,
        x: Tensor,
        pos_grid: Tensor,
        batch: Tensor,
        superpoint: Tensor,
        inverse: Tensor,
        batch_inverse: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        r"""Encode the voxels and average their features over each superpoint.

        Args:
            x: Voxel features, shape $(V, C_\text{in})$.
            pos_grid: Voxel grid coordinates, shape $(V, 3)$.
            batch: Voxel batch index, shape $(V,)$.
            superpoint: Superpoint id of every point within its scene, shape $(N,)$.
            inverse: Voxel row of every point within its scene, shape $(N,)$.
            batch_inverse: Scene index of every point, shape $(N,)$.

        Returns:
            The superpoint features $(S, C)$, the packed superpoint index of every point $(N,)$ and the scene index
            of every superpoint $(S,)$, sorted.
        """
        bottleneck, skips = self.encoder(x, pos_grid, batch, return_intermediates=True)
        sparse_x: "SparseConvTensor" = self.decoder(bottleneck, skips)
        inverse = offset_index(inverse, batch_inverse, batch)
        return superpoint_pool(sparse_x.features[inverse], superpoint, batch_inverse)

    def forward_head(self, x: Tensor, superpoint: Tensor, superpoint_batch: Tensor) -> SPFormerOutput:
        r"""Predict the instance candidates of every scene from its superpoint features.

        Args:
            x: Superpoint features, shape $(S, C)$.
            superpoint: Packed superpoint index of every point, shape $(N,)$.
            superpoint_batch: Scene index of every superpoint, shape $(S,)$, sorted.

        Returns:
            The `SPFormerOutput` of the batch.
        """
        *aux_outputs, last = self.head(x, superpoint_batch)
        return {
            "class_logits": last["class_logits"],
            "scores": last["scores"],
            "mask_logits": last["mask_logits"],
            "superpoint": superpoint,
            "superpoint_batch": superpoint_batch,
            "aux_outputs": aux_outputs,
        }

    def forward(
        self,
        x: Tensor,
        pos_grid: Tensor,
        batch: Tensor,
        superpoint: Tensor,
        inverse: Tensor,
        batch_inverse: Tensor,
    ) -> SPFormerOutput:
        features, superpoint, superpoint_batch = self.forward_features(
            x,
            pos_grid,
            batch,
            superpoint,
            inverse,
            batch_inverse,
        )
        return self.forward_head(features, superpoint, superpoint_batch)

    @torch.no_grad()
    def decode(self, output: SPFormerOutput) -> InstanceSegmentation3D:
        r"""Decode the raw output into the `num_instances` best instance masks of every scene.

        Every (query, class) pair is scored by the class probability times the query's mask quality score; the
        `num_instances` best pairs of a scene become instances. Their masks keep the superpoints with a positive logit,
        and their score is further multiplied by the mean mask probability over those superpoints.

        Args:
            output: The `forward` output.

        Returns:
            The packed instances: `masks` $(K, N)$ over the points of the batch (false outside the instance's scene),
            `labels`, `scores` and `batch` $(K,)$.
        """
        class_scores = output["class_logits"].softmax(dim=-1)[..., :-1] * output["scores"][..., None]
        superpoint_batch = output["superpoint_batch"]
        sizes = torch.bincount(superpoint_batch).tolist()
        masks, labels, scores, batch = [], [], [], []
        start = 0
        for index, (logits, size) in enumerate(zip(output["mask_logits"].split(sizes, dim=1), sizes)):
            score, pair = class_scores[index].flatten().topk(min(self.num_instances, class_scores[index].numel()))
            mask_logits = logits[pair // self.num_classes]
            mask = mask_logits > 0
            mask_score = (mask_logits.sigmoid() * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-6)
            packed = mask.new_zeros(mask.shape[0], superpoint_batch.shape[0])
            packed[:, start : start + size] = mask
            masks.append(packed)
            labels.append(pair % self.num_classes)
            scores.append(score * mask_score)
            batch.append(torch.full_like(pair, index))
            start += size
        return {
            "masks": torch.cat(masks)[:, output["superpoint"]],
            "labels": torch.cat(labels),
            "scores": torch.cat(scores),
            "batch": torch.cat(batch),
        }


@register_model(
    "spformer.scannet.jiahao-sun",
    task="instance-segmentation",
    input_keys=("x", "pos_grid", "batch", "superpoint", "inverse", "batch_inverse"),
    weights=WeightsDict(
        url="hf://torch-pointcloud/spformer.scannet.jiahao-sun/resolve/0000000000000000000000000000000000000000/model.safetensors",
        dataset="scannet",
        metrics={"mAP": 56.47, "mAP@0.5": 73.93, "mAP@0.25": 82.55},
        classes=SCANNET20_CLASSES[2:],
        author="jiahao-sun",
        license="MIT",
    ),
    transform=T.Compose(
        [
            T.Relabel(keys=DataKeys.SEGMENT, labels=range(3, 21), default=-1, allow_missing_keys=True),
            T.CopyItems(
                keys=[DataKeys.POS, DataKeys.SEGMENT, DataKeys.INSTANCE],
                dst_keys=[DataKeys.ORIGIN_POS, DataKeys.ORIGIN_SEGMENT, DataKeys.ORIGIN_INSTANCE],
                allow_missing_keys=True,
            ),
            T.Normalize(keys=DataKeys.COLOR, mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5]),
            T.Shift(keys=DataKeys.POS, method="centroid"),
            T.Cat(keys=[DataKeys.COLOR, DataKeys.POS], dst_key=DataKeys.X, dim=1),
            # 2 cm voxels: the grid is the scaled positions, floored from their minimum.
            T.Scale(keys=DataKeys.POS, scale=50.0),
            T.Shift(keys=DataKeys.POS, method="min"),
            T.Voxelize(
                pos_key=DataKeys.POS,
                pos_reduce="grid",
                dst_pos_grid_key=DataKeys.POS_GRID,
                keys=[DataKeys.X, DataKeys.SEGMENT, DataKeys.INSTANCE, DataKeys.COLOR, DataKeys.NORMAL],
                size=1.0,
                allow_missing_keys=True,
                dst_inverse_key=DataKeys.INVERSE,
            ),
        ]
    ),
    hparams=dict(
        in_channels=6,
        num_classes=18,
        channels=[32, 64, 96, 128, 160],
        depths=2,
        norm_kwargs=dict(eps=1e-4, momentum=0.1),
        num_queries=400,
        num_instances=100,
    ),
)
def spformer_scannet(**hparams: Any) -> SPFormerInstanceSegmentation:
    return SPFormerInstanceSegmentation(**hparams)
