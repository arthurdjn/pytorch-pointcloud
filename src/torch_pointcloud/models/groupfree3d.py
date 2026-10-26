r"""Group-Free 3D: transformer object detection over sampled object candidates.

{{ paper("2104.00678") }}

Reference: :arxiv: [Group-Free 3D Object Detection via Transformers](https://arxiv.org/abs/2104.00678) (Liu et al., 2021).
Reference implementation: :github: [zeliu98/Group-Free-3D](https://github.com/zeliu98/Group-Free-3D).
"""

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TypedDict, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import MLP
from torch_geometric.utils import to_dense_batch

import torch_pointcloud.transforms as T
from torch_pointcloud.datasets.scannet import SCANNET_DETECTION_CLASSES, SCANNET_DETECTION_MEAN_SIZES
from torch_pointcloud.datasets.sunrgbd import SUNRGBD_CLASSES
from torch_pointcloud.layers.transformer import TransformerDecoderLayer
from torch_pointcloud.ops.box3d import class_to_angle
from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.types import Detection3D, OptTensor

from ._base import DetectionModel
from ._registry import WeightsDict, register_model
from .votenet import VoteNetBackbone


class _GroupFree3DOutput(TypedDict):
    objectness_scores: Tensor
    center: Tensor
    heading_scores: Tensor
    heading_residuals_normalized: Tensor
    heading_residuals: Tensor
    size: Tensor
    sem_cls_scores: Tensor
    query_pos: Tensor
    query_indices: Tensor
    seed_objectness_logits: Tensor
    pos_seed: Tensor
    batch_seed: Tensor
    seed_indices: Tensor


class GroupFree3DOutput(_GroupFree3DOutput, total=False):
    r"""Predictions of every stage (the proposal head, then each decoder layer) for $B$ scenes with $Q$ candidates.

    The stage tensors stack the $L + 1$ stages along the first axis: `objectness_scores` $(L + 1, B, Q, 1)$, `center`
    and `size` $(L + 1, B, Q, 3)$, `heading_scores`, `heading_residuals_normalized` and `heading_residuals`
    $(L + 1, B, Q, H)$, `sem_cls_scores` $(L + 1, B, Q, C)$; a class-aware size head adds `size_scores`
    $(L + 1, B, Q, S)$, `size_residuals_normalized` and `size_residuals` $(L + 1, B, Q, S, 3)$. `query_pos` $(B, Q, 3)$
    is the candidate positions and `query_indices` $(B, Q)$ their indices into the packed seeds, whose objectness
    logits `seed_objectness_logits` $(N_s,)$, positions `pos_seed` $(N_s, 3)$, scene indices `batch_seed` $(N_s,)$ and
    input indices `seed_indices` $(N_s,)$ complete the output.
    """

    size_scores: Tensor
    size_residuals_normalized: Tensor
    size_residuals: Tensor


class GroupFree3DPredictionHead(nn.Module):
    r"""Box prediction head applied to every object candidate.

    A two-layer MLP followed by one linear layer per quantity: objectness, center offset from the candidate position,
    heading bin scores and normalized residuals, size and semantic class. The size is either class-aware (bin scores
    plus per-bin normalized residuals, decoded with the argmax bin) or a direct $(d_x, d_y, d_z)$ regression.

    Args:
        in_channels: Candidate feature channels.
        num_classes: Number of semantic classes.
        num_heading_bins: Number of heading-angle bins.
        num_size_clusters: Number of size templates of the class-aware size head (ignored when agnostic).
        size_class_agnostic: Regress the size directly instead of through size templates.
        act: Activation type or callable.
        act_kwargs: Extra activation arguments.
        norm: Normalization type or callable.
        norm_kwargs: Extra normalization arguments.

    Shape:
        - Input: $(B, Q, C)$ candidate features, $(B, Q, 3)$ candidate positions, $(S, 3)$ size templates
        - Output: a dict of $(B, Q, \ldots)$ predictions
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        num_heading_bins: int,
        num_size_clusters: int,
        *,
        size_class_agnostic: bool,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_heading_bins = num_heading_bins
        self.num_size_clusters = num_size_clusters
        self.size_class_agnostic = size_class_agnostic

        self.mlp = MLP(
            [in_channels, in_channels, in_channels],
            act=act,
            act_kwargs=act_kwargs,
            norm=norm,
            norm_kwargs=norm_kwargs,
            plain_last=False,
        )
        self.objectness_head = nn.Linear(in_channels, 1)
        self.center_head = nn.Linear(in_channels, 3)
        self.heading_class_head = nn.Linear(in_channels, num_heading_bins)
        self.heading_residual_head = nn.Linear(in_channels, num_heading_bins)
        if size_class_agnostic:
            self.size_head = nn.Linear(in_channels, 3)
        else:
            self.size_class_head = nn.Linear(in_channels, num_size_clusters)
            self.size_residual_head = nn.Linear(in_channels, num_size_clusters * 3)
        self.sem_cls_head = nn.Linear(in_channels, num_classes)

    def forward(self, x: Tensor, pos: Tensor, mean_sizes: OptTensor = None) -> Dict[str, Tensor]:
        batch_size, num_queries, _ = x.shape
        x = self.mlp(x.reshape(batch_size * num_queries, -1)).view(batch_size, num_queries, -1)

        heading_residuals_normalized = self.heading_residual_head(x)
        out = {
            "objectness_scores": self.objectness_head(x),
            "center": pos + self.center_head(x),
            "heading_scores": self.heading_class_head(x),
            "heading_residuals_normalized": heading_residuals_normalized,
            "heading_residuals": heading_residuals_normalized * (math.pi / self.num_heading_bins),
            "sem_cls_scores": self.sem_cls_head(x),
        }
        if self.size_class_agnostic:
            out["size"] = self.size_head(x)
            return out

        assert mean_sizes is not None, "A class-aware size head requires the size templates `mean_sizes`."
        ns = self.num_size_clusters
        size_scores = self.size_class_head(x)
        size_residuals_normalized = self.size_residual_head(x).view(batch_size, num_queries, ns, 3)
        size_residuals = size_residuals_normalized * mean_sizes.view(1, 1, ns, 3)
        size_class = size_scores.argmax(dim=-1).view(batch_size, num_queries, 1, 1).expand(-1, -1, 1, 3)
        out["size_scores"] = size_scores
        out["size_residuals_normalized"] = size_residuals_normalized
        out["size_residuals"] = size_residuals
        out["size"] = (size_residuals + mean_sizes.view(1, 1, ns, 3)).gather(2, size_class).squeeze(2)
        return out


class GroupFree3DDetection(DetectionModel):
    r"""Group-Free 3D object detector (packed point format).

    Reference: :arxiv: [Group-Free 3D Object Detection via Transformers](https://arxiv.org/abs/2104.00678) (Liu et al., 2021).
    Reference implementation: :github: [zeliu98/Group-Free-3D](https://github.com/zeliu98/Group-Free-3D).

    A PointNet++ backbone ([`VoteNetBackbone`][torch_pointcloud.models.votenet.VoteNetBackbone]) extracts seed
    points. A per-seed objectness MLP scores the seeds and the `num_proposals` highest become the object candidates
    (k-closest points sampling), each first decoded by a proposal head. A stack of post-norm transformer decoder
    layers then refines the candidate features, attending over the candidates and over every seed, and a prediction
    head after each layer re-decodes the box. The learned positional embeddings are rebuilt at every layer from the
    previous stage's detached center and size (candidates) and from the seed positions (keys).

    Args:
        in_channels: Input feature channels per point excluding xyz ($0$ for the released checkpoints).
        num_classes: Number of semantic classes.
        num_heading_bins: Number of heading-angle bins ($1$ for axis-aligned ScanNet boxes, $12$ for oriented
            SUN RGB-D boxes).
        num_size_clusters: Number of size templates of the class-aware size head ($0$ when class agnostic).
        mean_sizes: Per-template mean box size, shape $(\text{num\_size\_clusters}, 3)$, or `None` when the size
            head is class agnostic.
        size_class_agnostic: Regress the box size directly instead of through size templates.
        num_proposals: Number of object candidates per scene.
        sa_channels: Per-SA-block MLP channel lists for the backbone.
        sa_num_points: Per-SA-block farthest-point-sample counts.
        sa_radii: Per-SA-block ball-query radii.
        sa_num_neighbors: Per-SA-block neighbor caps.
        fp_channels: Per-FP-block MLP channel lists for the backbone; the last width is the embedding dimension.
        num_decoder_layers: Number of transformer decoder layers (each followed by a prediction head).
        num_heads: Number of attention heads.
        mlp_dim: Hidden width of the decoder feed-forward blocks.
        dropout: Dropout probability of the decoder layers.
        act: Activation type or callable.
        act_kwargs: Extra activation arguments.
        norm: Normalization type or callable of the backbone, sampler, position embeddings and heads.
        norm_kwargs: Extra normalization arguments.
    """

    mean_sizes: OptTensor  # registered as a non-persistent buffer in __init__

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        *,
        num_heading_bins: int,
        num_size_clusters: int,
        mean_sizes: Union[Tensor, List[List[float]], None],
        size_class_agnostic: bool,
        num_proposals: int,
        sa_channels: Sequence[Sequence[int]],
        sa_num_points: Sequence[int],
        sa_radii: Sequence[float],
        sa_num_neighbors: Sequence[int],
        fp_channels: Sequence[Sequence[int]],
        num_decoder_layers: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(in_channels=in_channels, num_classes=num_classes)
        if size_class_agnostic:
            mean = None
        else:
            if mean_sizes is None:
                raise ValueError("`mean_sizes` is required unless `size_class_agnostic=True`.")
            mean = torch.as_tensor(mean_sizes, dtype=torch.float32)
            if mean.shape != (num_size_clusters, 3):
                raise ValueError(f"`mean_sizes` must have shape ({num_size_clusters}, 3), got {tuple(mean.shape)}.")

        self.num_heading_bins = num_heading_bins
        self.num_size_clusters = num_size_clusters
        self.size_class_agnostic = size_class_agnostic
        self.num_proposals = num_proposals
        self.num_decoder_layers = num_decoder_layers
        self.act = act
        self.act_kwargs = act_kwargs
        self.norm = norm
        self.norm_kwargs = norm_kwargs

        # Not part of the checkpoint (the reference rebuilds it on the fly); persistent=False
        # keeps it out of the state dict while still moving with `.to(device)`.
        self.register_buffer("mean_sizes", mean, persistent=False)

        self.backbone = VoteNetBackbone(
            in_channels,
            sa_channels=sa_channels,
            sa_num_points=sa_num_points,
            sa_radii=sa_radii,
            sa_num_neighbors=sa_num_neighbors,
            fp_channels=fp_channels,
            act=act,
            act_kwargs=act_kwargs,
            norm=norm,
            norm_kwargs=norm_kwargs,
        )
        embed_dim = self.backbone.out_channels
        self.seed_objectness = MLP(
            [embed_dim, embed_dim, embed_dim, 1],
            act=act,
            act_kwargs=act_kwargs,
            norm=norm,
            norm_kwargs=norm_kwargs,
            plain_last=True,
        )
        self.proposal_head = self.configure_head()
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.query_pos_embeds = nn.ModuleList(
            MLP([6, embed_dim, embed_dim], act=act, act_kwargs=act_kwargs, norm=norm, norm_kwargs=norm_kwargs)
            for _ in range(num_decoder_layers)
        )
        self.key_pos_embeds = nn.ModuleList(
            MLP([3, embed_dim, embed_dim], act=act, act_kwargs=act_kwargs, norm=norm, norm_kwargs=norm_kwargs)
            for _ in range(num_decoder_layers)
        )
        self.decoder = nn.ModuleList(
            TransformerDecoderLayer(
                embed_dim,
                num_heads,
                mlp_dim,
                dropout,
                act=act,
                act_kwargs=act_kwargs,
                norm_first=False,
                pos_in_value=True,
            )
            for _ in range(num_decoder_layers)
        )
        self.prediction_heads = nn.ModuleList(self.configure_head() for _ in range(num_decoder_layers))

        for module in [self.decoder, self.query_pos_embeds, self.key_pos_embeds]:
            for param in module.parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)

    def configure_head(self) -> GroupFree3DPredictionHead:
        """Build one box prediction head."""
        return GroupFree3DPredictionHead(
            self.backbone.out_channels,
            self.num_classes,
            self.num_heading_bins,
            self.num_size_clusters,
            size_class_agnostic=self.size_class_agnostic,
            act=self.act,
            act_kwargs=self.act_kwargs,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
        )

    def reset_classifier(self, num_classes: int) -> None:
        self.num_classes = num_classes
        for head in [self.proposal_head, *self.prediction_heads]:
            assert isinstance(head, GroupFree3DPredictionHead)
            head.num_classes = num_classes
            head.sem_cls_head = nn.Linear(head.in_channels, num_classes)

    @property
    def num_features(self) -> int:
        """Channel count $C$ of the per-seed backbone features."""
        return self.backbone.out_channels

    def forward_features(self, x: OptTensor, pos: Tensor, batch: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        if x is None:
            x = pos.new_zeros((pos.size(0), 0))
        return self.backbone(x, pos, batch)

    def forward_head(self, x: Tensor, pos: Tensor, batch: Tensor, seed_indices: Tensor) -> GroupFree3DOutput:
        seed_logits = self.seed_objectness(x).squeeze(-1)  # (N_s,)
        x_dense, mask = to_dense_batch(x, batch)  # (B, S, C)
        batch_size, num_seeds = mask.shape
        pos_dense, _ = to_dense_batch(pos, batch, batch_size=batch_size, max_num_nodes=num_seeds)  # (B, S, 3)
        # Ranked by logit: the sigmoid saturates to exactly 1 in float32 and would leave the top seeds tied.
        logits_dense, _ = to_dense_batch(
            seed_logits,
            batch,
            fill_value=-math.inf,
            batch_size=batch_size,
            max_num_nodes=num_seeds,
        )  # (B, S)

        topk = logits_dense.topk(self.num_proposals, dim=1).indices  # (B, Q)
        counts = mask.sum(dim=1)
        query_indices = topk + (counts.cumsum(0) - counts).unsqueeze(1)
        x_query = x_dense.gather(1, topk.unsqueeze(-1).expand(-1, -1, x_dense.size(-1)))  # (B, Q, C)
        pos_query = pos_dense.gather(1, topk.unsqueeze(-1).expand(-1, -1, 3))  # (B, Q, 3)

        stages = [self.proposal_head(x_query, pos_query, self.mean_sizes)]
        query = self.query_proj(x_query).transpose(0, 1)  # (Q, B, C)
        key = self.key_proj(x_dense).transpose(0, 1)  # (S, B, C)
        for layer, query_pos_embed, key_pos_embed, head in zip(
            self.decoder, self.query_pos_embeds, self.key_pos_embeds, self.prediction_heads
        ):
            box = torch.cat([stages[-1]["center"], stages[-1]["size"]], dim=-1).detach()  # (B, Q, 6)
            query_pos = query_pos_embed(box.view(-1, 6)).view(batch_size, -1, query.size(-1)).transpose(0, 1)
            key_pos, _ = to_dense_batch(key_pos_embed(pos), batch, batch_size=batch_size, max_num_nodes=num_seeds)
            query = layer(query, key, pos=key_pos.transpose(0, 1), query_pos=query_pos, memory_key_padding_mask=~mask)
            stages.append(head(query.transpose(0, 1), pos_query, self.mean_sizes))

        stacked = {name: torch.stack([stage[name] for stage in stages]) for name in stages[0]}
        return {
            **stacked,  # type: ignore[typeddict-item]
            "query_pos": pos_query,
            "query_indices": query_indices,
            "seed_objectness_logits": seed_logits,
            "pos_seed": pos,
            "batch_seed": batch,
            "seed_indices": seed_indices,
        }

    def forward(self, x: OptTensor, pos: Tensor, batch: Tensor) -> GroupFree3DOutput:
        x_seed, pos_seed, batch_seed, seed_indices = self.forward_features(x, pos, batch)
        return self.forward_head(x_seed, pos_seed, batch_seed, seed_indices)

    @torch.no_grad()
    def decode(self, out: GroupFree3DOutput, stages: Sequence[int] = (-1,)) -> Detection3D:
        r"""Decode the predictions of the selected stages into raw per-candidate detections (no NMS or filtering).

        Builds one oriented box per candidate and stage, scores it by the sigmoid objectness, and labels it by the
        argmax semantic class. Several stages form an ensemble: their boxes are concatenated scene by scene, so one
        NMS pass over the result merges them. Stage $0$ is the proposal head and stage $l$ the prediction head after
        decoder layer $l$; negative indices count from the last stage. The heading head predicts negated angles, so
        the decoded heading is negated to return counter-clockwise headings (the library box convention). The
        evaluation pipeline applies point-count filtering, NMS and the indoor per-class expansion (driven by the
        returned `class_probs`) via the `torch_pointcloud.ops.box3d` utilities.

        Args:
            out: A `GroupFree3DOutput` from `forward`.
            stages: Indices of the stages to decode, in output order; `(-1,)` keeps the last decoder layer.

        Returns:
            Packed candidates `{"boxes", "scores", "labels", "batch", "class_probs"}` (PyG layout), where the
            per-candidate score is objectness, the label is the argmax semantic class, and `class_probs` holds the
            softmaxed semantic-class probabilities.

        Raises:
            IndexError: If `stages` is empty or indexes past the stages of `out`.

        Shape:
            - boxes: $(B \cdot T \cdot Q, 7)$ for $T$ selected stages
            - scores / labels / batch: $(B \cdot T \cdot Q,)$
            - class_probs: $(B \cdot T \cdot Q, C)$
        """
        num_total, batch_size, num_proposals, _ = out["center"].shape
        positions = [stage + num_total if stage < 0 else stage for stage in stages]
        if not positions or any(not 0 <= position < num_total for position in positions):
            raise IndexError(f"`stages` must index the {num_total} stages of the output, got {list(stages)}.")

        index = torch.tensor(positions, device=out["center"].device)
        center = _select_stages(out["center"], index)
        size = _select_stages(out["size"], index)
        heading_class = _select_stages(out["heading_scores"], index).argmax(dim=-1)
        heading_residual = _select_stages(out["heading_residuals"], index).gather(2, heading_class.unsqueeze(-1))
        angle = -class_to_angle(heading_class, heading_residual.squeeze(-1), self.num_heading_bins)

        boxes = torch.cat([center, size, angle.unsqueeze(-1)], dim=-1)
        objectness = _select_stages(out["objectness_scores"], index)[..., 0].sigmoid()
        class_probs = _select_stages(out["sem_cls_scores"], index).softmax(dim=-1)
        batch = torch.arange(batch_size, device=boxes.device).repeat_interleave(len(positions) * num_proposals)
        return {
            "boxes": boxes.reshape(-1, 7),
            "scores": objectness.reshape(-1),
            "labels": class_probs.argmax(dim=-1).reshape(-1),
            "batch": batch,
            "class_probs": class_probs.reshape(-1, class_probs.size(-1)),
        }


def _select_stages(value: Tensor, index: Tensor) -> Tensor:
    r"""Gather the selected stages of a $(T, B, Q, \ldots)$ stage tensor into $(B, T' \cdot Q, \ldots)$ per scene."""
    selected = value.index_select(0, index)
    return selected.transpose(0, 1).flatten(1, 2)


_SCANNET_TRANSFORM = T.Compose(
    [
        T.CopyItems(
            keys=[DataKeys.POS, DataKeys.SEGMENT],
            dst_keys=[DataKeys.ORIGIN_POS, DataKeys.ORIGIN_SEGMENT],
            allow_missing_keys=True,
        ),
        T.RandomSample(
            keys=[DataKeys.POS, DataKeys.COLOR, DataKeys.NORMAL, DataKeys.SEGMENT, DataKeys.INSTANCE],
            num_samples=50000,
            allow_missing_keys=True,
            dst_index_key=DataKeys.INDEX,
        ),
    ]
)

_SUNRGBD_TRANSFORM = T.Compose(
    [
        T.CopyItems(keys=DataKeys.POS, dst_keys=DataKeys.ORIGIN_POS),
        T.RandomSample(
            keys=[DataKeys.POS, DataKeys.COLOR],
            num_samples=20000,
            allow_missing_keys=True,
            dst_index_key=DataKeys.INDEX,
        ),
    ]
)


@register_model(
    "groupfree3d-l6-o256.scannet.ze-liu",
    task="detection",
    weights=WeightsDict(
        url="hf://torch-pointcloud/groupfree3d-l6-o256.scannet.ze-liu/resolve/0000000000000000000000000000000000000000/model.safetensors",
        dataset="scannet",
        metrics={"mAP@0.25": 67.30, "mAP@0.5": 48.90},
        classes=SCANNET_DETECTION_CLASSES,
        author="ze-liu",
        license="MIT",
    ),
    transform=_SCANNET_TRANSFORM,
    hparams=dict(
        in_channels=0,
        num_classes=18,
        num_heading_bins=1,
        num_size_clusters=18,
        mean_sizes=SCANNET_DETECTION_MEAN_SIZES,
        size_class_agnostic=False,
        num_proposals=256,
        sa_channels=[[64, 64, 128], [128, 128, 256], [128, 128, 256], [128, 128, 256]],
        sa_num_points=[2048, 1024, 512, 256],
        sa_radii=[0.2, 0.4, 0.8, 1.2],
        sa_num_neighbors=[64, 32, 16, 16],
        fp_channels=[[256, 256], [256, 288]],
        num_decoder_layers=6,
        num_heads=8,
        mlp_dim=2048,
        dropout=0.1,
    ),
)
def groupfree3d_l6_o256_scannet_ze_liu(**hparams: Any) -> GroupFree3DDetection:
    return GroupFree3DDetection(**hparams)


@register_model(
    "groupfree3d-l12-o256.scannet.ze-liu",
    task="detection",
    weights=WeightsDict(
        url="hf://torch-pointcloud/groupfree3d-l12-o256.scannet.ze-liu/resolve/0000000000000000000000000000000000000000/model.safetensors",
        dataset="scannet",
        metrics={"mAP@0.25": 67.20, "mAP@0.5": 49.70},
        classes=SCANNET_DETECTION_CLASSES,
        author="ze-liu",
        license="MIT",
    ),
    transform=_SCANNET_TRANSFORM,
    hparams=dict(
        in_channels=0,
        num_classes=18,
        num_heading_bins=1,
        num_size_clusters=18,
        mean_sizes=SCANNET_DETECTION_MEAN_SIZES,
        size_class_agnostic=False,
        num_proposals=256,
        sa_channels=[[64, 64, 128], [128, 128, 256], [128, 128, 256], [128, 128, 256]],
        sa_num_points=[2048, 1024, 512, 256],
        sa_radii=[0.2, 0.4, 0.8, 1.2],
        sa_num_neighbors=[64, 32, 16, 16],
        fp_channels=[[256, 256], [256, 288]],
        num_decoder_layers=12,
        num_heads=8,
        mlp_dim=2048,
        dropout=0.1,
    ),
)
def groupfree3d_l12_o256_scannet_ze_liu(**hparams: Any) -> GroupFree3DDetection:
    return GroupFree3DDetection(**hparams)


@register_model(
    "groupfree3d-w2x-l12-o256.scannet.ze-liu",
    task="detection",
    weights=WeightsDict(
        url="hf://torch-pointcloud/groupfree3d-w2x-l12-o256.scannet.ze-liu/resolve/0000000000000000000000000000000000000000/model.safetensors",
        dataset="scannet",
        metrics={"mAP@0.25": 68.80, "mAP@0.5": 52.10},
        classes=SCANNET_DETECTION_CLASSES,
        author="ze-liu",
        license="MIT",
    ),
    transform=_SCANNET_TRANSFORM,
    hparams=dict(
        in_channels=0,
        num_classes=18,
        num_heading_bins=1,
        num_size_clusters=18,
        mean_sizes=SCANNET_DETECTION_MEAN_SIZES,
        size_class_agnostic=False,
        num_proposals=256,
        sa_channels=[[128, 128, 256], [256, 256, 512], [256, 256, 512], [256, 256, 512]],
        sa_num_points=[2048, 1024, 512, 256],
        sa_radii=[0.2, 0.4, 0.8, 1.2],
        sa_num_neighbors=[64, 32, 16, 16],
        fp_channels=[[512, 512], [512, 288]],
        num_decoder_layers=12,
        num_heads=8,
        mlp_dim=2048,
        dropout=0.1,
    ),
)
def groupfree3d_w2x_l12_o256_scannet_ze_liu(**hparams: Any) -> GroupFree3DDetection:
    return GroupFree3DDetection(**hparams)


@register_model(
    "groupfree3d-w2x-l12-o512.scannet.ze-liu",
    task="detection",
    weights=WeightsDict(
        url="hf://torch-pointcloud/groupfree3d-w2x-l12-o512.scannet.ze-liu/resolve/0000000000000000000000000000000000000000/model.safetensors",
        dataset="scannet",
        metrics={"mAP@0.25": 69.10, "mAP@0.5": 52.80},
        classes=SCANNET_DETECTION_CLASSES,
        author="ze-liu",
        license="MIT",
    ),
    transform=_SCANNET_TRANSFORM,
    hparams=dict(
        in_channels=0,
        num_classes=18,
        num_heading_bins=1,
        num_size_clusters=18,
        mean_sizes=SCANNET_DETECTION_MEAN_SIZES,
        size_class_agnostic=False,
        num_proposals=512,
        sa_channels=[[128, 128, 256], [256, 256, 512], [256, 256, 512], [256, 256, 512]],
        sa_num_points=[2048, 1024, 512, 256],
        sa_radii=[0.2, 0.4, 0.8, 1.2],
        sa_num_neighbors=[64, 32, 16, 16],
        fp_channels=[[512, 512], [512, 288]],
        num_decoder_layers=12,
        num_heads=8,
        mlp_dim=2048,
        dropout=0.1,
    ),
)
def groupfree3d_w2x_l12_o512_scannet_ze_liu(**hparams: Any) -> GroupFree3DDetection:
    return GroupFree3DDetection(**hparams)


@register_model(
    "groupfree3d-l6-o256.sunrgbd.ze-liu",
    task="detection",
    weights=WeightsDict(
        url="hf://torch-pointcloud/groupfree3d-l6-o256.sunrgbd.ze-liu/resolve/0000000000000000000000000000000000000000/model.safetensors",
        dataset="sunrgbd",
        metrics={"mAP@0.25": 63.00, "mAP@0.5": 45.20},
        classes=SUNRGBD_CLASSES,
        author="ze-liu",
        license="MIT",
    ),
    transform=_SUNRGBD_TRANSFORM,
    hparams=dict(
        in_channels=0,
        num_classes=10,
        num_heading_bins=12,
        num_size_clusters=0,
        mean_sizes=None,
        size_class_agnostic=True,
        num_proposals=256,
        sa_channels=[[64, 64, 128], [128, 128, 256], [128, 128, 256], [128, 128, 256]],
        sa_num_points=[2048, 1024, 512, 256],
        sa_radii=[0.2, 0.4, 0.8, 1.2],
        sa_num_neighbors=[64, 32, 16, 16],
        fp_channels=[[256, 256], [256, 288]],
        num_decoder_layers=6,
        num_heads=8,
        mlp_dim=2048,
        dropout=0.1,
    ),
)
def groupfree3d_l6_o256_sunrgbd_ze_liu(**hparams: Any) -> GroupFree3DDetection:
    return GroupFree3DDetection(**hparams)


@register_model(
    "groupfree3d-l6-o256.scannet.mmdet3d",
    task="detection",
    weights=WeightsDict(
        url="hf://torch-pointcloud/groupfree3d-l6-o256.scannet.mmdet3d/resolve/0000000000000000000000000000000000000000/model.safetensors",
        dataset="scannet",
        metrics={"mAP@0.25": 66.17, "mAP@0.5": 48.47},
        classes=SCANNET_DETECTION_CLASSES,
        author="mmdet3d",
        license="Apache-2.0",
    ),
    transform=_SCANNET_TRANSFORM,
    hparams=dict(
        in_channels=0,
        num_classes=18,
        num_heading_bins=1,
        num_size_clusters=18,
        mean_sizes=SCANNET_DETECTION_MEAN_SIZES,
        size_class_agnostic=False,
        num_proposals=256,
        sa_channels=[[64, 64, 128], [128, 128, 256], [128, 128, 256], [128, 128, 256]],
        sa_num_points=[2048, 1024, 512, 256],
        sa_radii=[0.2, 0.4, 0.8, 1.2],
        sa_num_neighbors=[64, 32, 16, 16],
        fp_channels=[[256, 256], [256, 288]],
        num_decoder_layers=6,
        num_heads=8,
        mlp_dim=2048,
        dropout=0.1,
    ),
)
def groupfree3d_l6_o256_scannet_mmdet3d(**hparams: Any) -> GroupFree3DDetection:
    return GroupFree3DDetection(**hparams)


@register_model(
    "groupfree3d-l12-o256.scannet.mmdet3d",
    task="detection",
    weights=WeightsDict(
        url="hf://torch-pointcloud/groupfree3d-l12-o256.scannet.mmdet3d/resolve/0000000000000000000000000000000000000000/model.safetensors",
        dataset="scannet",
        metrics={"mAP@0.25": 66.57, "mAP@0.5": 48.21},
        classes=SCANNET_DETECTION_CLASSES,
        author="mmdet3d",
        license="Apache-2.0",
    ),
    transform=_SCANNET_TRANSFORM,
    hparams=dict(
        in_channels=0,
        num_classes=18,
        num_heading_bins=1,
        num_size_clusters=18,
        mean_sizes=SCANNET_DETECTION_MEAN_SIZES,
        size_class_agnostic=False,
        num_proposals=256,
        sa_channels=[[64, 64, 128], [128, 128, 256], [128, 128, 256], [128, 128, 256]],
        sa_num_points=[2048, 1024, 512, 256],
        sa_radii=[0.2, 0.4, 0.8, 1.2],
        sa_num_neighbors=[64, 32, 16, 16],
        fp_channels=[[256, 256], [256, 288]],
        num_decoder_layers=12,
        num_heads=8,
        mlp_dim=2048,
        dropout=0.1,
    ),
)
def groupfree3d_l12_o256_scannet_mmdet3d(**hparams: Any) -> GroupFree3DDetection:
    return GroupFree3DDetection(**hparams)


@register_model(
    "groupfree3d-w2x-l12-o256.scannet.mmdet3d",
    task="detection",
    weights=WeightsDict(
        url="hf://torch-pointcloud/groupfree3d-w2x-l12-o256.scannet.mmdet3d/resolve/0000000000000000000000000000000000000000/model.safetensors",
        dataset="scannet",
        metrics={"mAP@0.25": 68.20, "mAP@0.5": 51.02},
        classes=SCANNET_DETECTION_CLASSES,
        author="mmdet3d",
        license="Apache-2.0",
    ),
    transform=_SCANNET_TRANSFORM,
    hparams=dict(
        in_channels=0,
        num_classes=18,
        num_heading_bins=1,
        num_size_clusters=18,
        mean_sizes=SCANNET_DETECTION_MEAN_SIZES,
        size_class_agnostic=False,
        num_proposals=256,
        sa_channels=[[128, 128, 256], [256, 256, 512], [256, 256, 512], [256, 256, 512]],
        sa_num_points=[2048, 1024, 512, 256],
        sa_radii=[0.2, 0.4, 0.8, 1.2],
        sa_num_neighbors=[64, 32, 16, 16],
        fp_channels=[[512, 512], [512, 288]],
        num_decoder_layers=12,
        num_heads=8,
        mlp_dim=2048,
        dropout=0.1,
    ),
)
def groupfree3d_w2x_l12_o256_scannet_mmdet3d(**hparams: Any) -> GroupFree3DDetection:
    return GroupFree3DDetection(**hparams)


@register_model(
    "groupfree3d-w2x-l12-o512.scannet.mmdet3d",
    task="detection",
    weights=WeightsDict(
        url="hf://torch-pointcloud/groupfree3d-w2x-l12-o512.scannet.mmdet3d/resolve/0000000000000000000000000000000000000000/model.safetensors",
        dataset="scannet",
        metrics={"mAP@0.25": 68.22, "mAP@0.5": 52.61},
        classes=SCANNET_DETECTION_CLASSES,
        author="mmdet3d",
        license="Apache-2.0",
    ),
    transform=_SCANNET_TRANSFORM,
    hparams=dict(
        in_channels=0,
        num_classes=18,
        num_heading_bins=1,
        num_size_clusters=18,
        mean_sizes=SCANNET_DETECTION_MEAN_SIZES,
        size_class_agnostic=False,
        num_proposals=512,
        sa_channels=[[128, 128, 256], [256, 256, 512], [256, 256, 512], [256, 256, 512]],
        sa_num_points=[2048, 1024, 512, 256],
        sa_radii=[0.2, 0.4, 0.8, 1.2],
        sa_num_neighbors=[64, 32, 16, 16],
        fp_channels=[[512, 512], [512, 288]],
        num_decoder_layers=12,
        num_heads=8,
        mlp_dim=2048,
        dropout=0.1,
    ),
)
def groupfree3d_w2x_l12_o512_scannet_mmdet3d(**hparams: Any) -> GroupFree3DDetection:
    return GroupFree3DDetection(**hparams)
