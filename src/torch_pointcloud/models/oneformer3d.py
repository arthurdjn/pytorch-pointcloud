"""OneFormer3D unified semantic and instance segmentation model.

{{ paper("2311.14405") }}
"""

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    TypedDict,
    Union,
    overload,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn.pool.consecutive import consecutive_cluster

import torch_pointcloud.transforms as T
from torch_pointcloud.datasets.s3dis import S3DIS_CLASSES
from torch_pointcloud.datasets.scannet import SCANNET20_CLASSES
from torch_pointcloud.layers.act import create_act
from torch_pointcloud.models._base import SegmentationModel
from torch_pointcloud.models._registry import WeightsDict, register_model
from torch_pointcloud.models.spformer_unet import SPFormerUNetSegmentation
from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.imports import _TORCH_SCATTER_GITHUB_URL, optional_import
from torch_pointcloud.utils.types import OptTensor

if TYPE_CHECKING:
    from torch_scatter import scatter_mean


scatter_mean, _ = optional_import("torch_scatter", "scatter_mean", url=_TORCH_SCATTER_GITHUB_URL)


class OneFormer3DOutput(TypedDict, total=False):
    """Per-scene query decoder predictions: class logits, semantic logits, mask logits, scores and auxiliary layers."""

    cls_preds: List[Tensor]
    sem_preds: List[Tensor]
    masks: List[Tensor]
    scores: List[Optional[Tensor]]
    aux_outputs: List[Dict[str, Any]]


class OneFormer3DCrossAttention(nn.Module):
    r"""Cross-attention block used by the OneFormer3D query decoder.

    Wraps :pytorch: [`nn.MultiheadAttention`](https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html)
    and a residual `LayerNorm`. Operates per-scene on lists of tensors because
    the number of source points and queries differs across the batch.

    Args:
        embed_dim: Embedding dimension.
        num_heads: Number of attention heads.
        dropout: Dropout probability used both inside attention and on its output.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        sources: List[Tensor],
        queries: List[Tensor],
        attn_masks: Optional[List[Tensor]] = None,
    ) -> List[Tensor]:
        outputs: List[Tensor] = []
        for i, (k, q) in enumerate(zip(sources, queries)):
            attn_mask = attn_masks[i] if attn_masks is not None else None
            out, _ = self.attn(q, k, k, attn_mask=attn_mask)
            out = self.dropout(out) + q
            out = self.norm(out)
            outputs.append(out)
        return outputs


class OneFormer3DSelfAttention(nn.Module):
    r"""Self-attention block used by the OneFormer3D query decoder.

    Args:
        embed_dim: Embedding dimension.
        num_heads: Number of attention heads.
        dropout: Dropout probability.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries: List[Tensor]) -> List[Tensor]:
        outputs: List[Tensor] = []
        for q in queries:
            out, _ = self.attn(q, q, q)
            out = self.dropout(out) + q
            out = self.norm(out)
            outputs.append(out)
        return outputs


class OneFormer3DFFN(nn.Module):
    r"""Two-layer feed-forward block of the OneFormer3D decoder.

    Args:
        embed_dim: Embedding dimension.
        mlp_dim: Hidden dimension between the two linear layers.
        dropout: Dropout probability applied after each linear layer.
        act: Activation passed to `create_act`.
        act_kwargs: Extra keyword arguments for the activation.
    """

    def __init__(
        self,
        embed_dim: int,
        mlp_dim: int,
        dropout: float,
        act: Union[str, Callable, None] = "gelu",
        act_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        act_kwargs = act_kwargs or {}
        self.net = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            create_act(act, **act_kwargs) or nn.Identity(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, queries: List[Tensor]) -> List[Tensor]:
        outputs: List[Tensor] = []
        for q in queries:
            out = self.net(q)
            out = out + q
            out = self.norm(out)
            outputs.append(out)
        return outputs


class OneFormer3DQueryDecoder(nn.Module):
    r"""ScanNet query decoder for OneFormer3D.

    Mirrors `ScanNetQueryDecoder` from :github:
    [filaPro/oneformer3d](https://github.com/filaPro/oneformer3d). When
    `num_instance_queries == 0` and `num_semantic_queries == 0` the model uses
    superpoint features themselves as initial queries (the configuration used
    for the released ScanNet checkpoint).

    Args:
        in_channels: Number of channels in the per-superpoint backbone features.
        num_instance_classes: Number of instance / thing classes; an extra
            no-object slot is added inside the model.
        num_semantic_classes: Number of semantic classes including stuff classes.
        embed_dim: Hidden width of the transformer layers.
        num_layers: Number of cross / self / FFN layers.
        num_instance_queries: Number of learned instance queries.
        num_semantic_queries: Number of learned semantic queries.
        num_heads: Number of attention heads.
        mlp_dim: FFN hidden dimension.
        dropout: Dropout probability inside the transformer layers.
        act: FFN activation passed to `create_act`.
        act_kwargs: Extra keyword arguments for the FFN activation.
        iter_pred: If `True`, runs heads after every layer to enable iterative
            mask-attention; otherwise only the final layer is used.
        attn_mask: If `True`, the predicted masks are turned into attention masks
            for the next layer.
        objectness_flag: If `True`, predicts a per-query confidence score.
        semantic_head: If `True`, adds an `out_sem` head that predicts a semantic
            label per query (ScanNet). Set `False` when semantics come from dedicated
            semantic queries instead (S3DIS).
        num_semantic_linears: `1` or `2` linear layers in the semantic head.
    """

    def __init__(
        self,
        in_channels: int,
        num_instance_classes: int,
        num_semantic_classes: int,
        embed_dim: int = 256,
        num_layers: int = 6,
        num_instance_queries: int = 0,
        num_semantic_queries: int = 0,
        num_heads: int = 8,
        mlp_dim: int = 1024,
        dropout: float = 0.0,
        act: Union[str, Callable, None] = "gelu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        iter_pred: bool = True,
        attn_mask: bool = True,
        objectness_flag: bool = False,
        semantic_head: bool = True,
        num_semantic_linears: int = 1,
    ) -> None:
        super().__init__()
        if num_semantic_linears not in (1, 2):
            raise ValueError(f"`num_semantic_linears` must be 1 or 2, got {num_semantic_linears}.")

        self.in_channels = in_channels
        self.num_instance_classes = num_instance_classes
        self.num_semantic_classes = num_semantic_classes
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_instance_queries = num_instance_queries
        self.num_queries = num_instance_queries + num_semantic_queries
        self.iter_pred = iter_pred
        self.attn_mask = attn_mask
        self.objectness_flag = objectness_flag
        self.semantic_head = semantic_head

        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
        )
        if self.num_queries > 0:
            self.query = nn.Embedding(self.num_queries, embed_dim)
        if num_instance_queries == 0:
            self.query_proj = nn.Sequential(
                nn.Linear(in_channels, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim),
            )

        self.cross_attn_layers = nn.ModuleList(
            [OneFormer3DCrossAttention(embed_dim, num_heads, dropout) for _ in range(num_layers)]
        )
        self.self_attn_layers = nn.ModuleList(
            [OneFormer3DSelfAttention(embed_dim, num_heads, dropout) for _ in range(num_layers)]
        )
        self.ffn_layers = nn.ModuleList(
            [OneFormer3DFFN(embed_dim, mlp_dim, dropout, act=act, act_kwargs=act_kwargs) for _ in range(num_layers)]
        )

        self.out_norm = nn.LayerNorm(embed_dim)
        self.out_cls = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, num_instance_classes + 1),
        )
        if objectness_flag:
            self.out_score = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, 1),
            )
        self.x_mask = nn.Sequential(
            nn.Linear(in_channels, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        if semantic_head:
            if num_semantic_linears == 2:
                self.out_sem: nn.Module = nn.Sequential(
                    nn.Linear(embed_dim, embed_dim),
                    nn.ReLU(),
                    nn.Linear(embed_dim, num_semantic_classes + 1),
                )
            else:
                self.out_sem = nn.Linear(embed_dim, num_semantic_classes + 1)

        self.init_weights()

    def init_weights(self) -> None:
        """Applies Xavier uniform initialization to every multi-dimensional parameter."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _get_queries(self, queries: Optional[List[Tensor]], batch_size: int) -> List[Tensor]:
        result: List[Tensor] = []
        for i in range(batch_size):
            parts: List[Tensor] = []
            if hasattr(self, "query"):
                parts.append(self.query.weight)
            if queries is not None:
                parts.append(self.query_proj(queries[i]))
            result.append(torch.cat(parts, dim=0))
        return result

    def _forward_head(
        self,
        queries: List[Tensor],
        mask_feats: List[Tensor],
        last_layer: bool,
    ) -> Tuple[
        List[Tensor],
        Optional[List[Tensor]],
        List[Optional[Tensor]],
        List[Tensor],
        Optional[List[Tensor]],
    ]:
        cls_preds: List[Tensor] = []
        sem_preds: List[Tensor] = []
        pred_scores: List[Optional[Tensor]] = []
        pred_masks: List[Tensor] = []
        attn_masks: List[Tensor] = []
        emit_sem = last_layer and self.semantic_head
        for i, q in enumerate(queries):
            norm_q = self.out_norm(q)
            cls_preds.append(self.out_cls(norm_q))
            if emit_sem:
                sem_preds.append(self.out_sem(norm_q))
            pred_scores.append(self.out_score(norm_q) if self.objectness_flag else None)
            pred_mask = torch.einsum("nd,md->nm", norm_q, mask_feats[i])
            if self.attn_mask:
                attn_mask = (pred_mask.sigmoid() < 0.5).bool()
                full_rows = attn_mask.sum(-1) == attn_mask.shape[-1]
                attn_mask[full_rows] = False
                attn_masks.append(attn_mask.detach())
            pred_masks.append(pred_mask)

        return (
            cls_preds,
            sem_preds if emit_sem else None,
            pred_scores,
            pred_masks,
            attn_masks if self.attn_mask else None,
        )

    def forward(self, x: List[Tensor], queries: Optional[List[Tensor]] = None) -> OneFormer3DOutput:
        inst_feats = [self.input_proj(y) for y in x]
        mask_feats = [self.x_mask(y) for y in x]
        q = self._get_queries(queries, len(x))

        if not self.iter_pred:
            for i in range(self.num_layers):
                q = self.cross_attn_layers[i](inst_feats, q)
                q = self.self_attn_layers[i](q)
                q = self.ffn_layers[i](q)

            cls_preds, sem_preds, pred_scores, pred_masks, _ = self._forward_head(q, mask_feats, last_layer=True)
            out: OneFormer3DOutput = {
                "cls_preds": cls_preds,
                "masks": pred_masks,
                "scores": pred_scores,
            }

            if sem_preds is not None:
                out["sem_preds"] = sem_preds
            return out

        all_cls: List[List[Tensor]] = []
        all_sem: List[Optional[List[Tensor]]] = []
        all_scores: List[List[Optional[Tensor]]] = []
        all_masks: List[List[Tensor]] = []
        cls_pred, sem_pred, pred_score, pred_mask, attn_mask = self._forward_head(q, mask_feats, last_layer=False)
        all_cls.append(cls_pred)
        all_sem.append(sem_pred)
        all_scores.append(pred_score)
        all_masks.append(pred_mask)

        for i in range(self.num_layers):
            q = self.cross_attn_layers[i](inst_feats, q, attn_mask)
            q = self.self_attn_layers[i](q)
            q = self.ffn_layers[i](q)
            last = i == self.num_layers - 1
            cls_pred, sem_pred, pred_score, pred_mask, attn_mask = self._forward_head(q, mask_feats, last)
            all_cls.append(cls_pred)
            all_sem.append(sem_pred)
            all_scores.append(pred_score)
            all_masks.append(pred_mask)

        aux_outputs: List[Dict[str, Any]] = []
        for cls_p, sem_p, sc, m in zip(all_cls[:-1], all_sem[:-1], all_scores[:-1], all_masks[:-1]):
            entry: Dict[str, Any] = {"cls_preds": cls_p, "masks": m, "scores": sc}
            if sem_p is not None:
                entry["sem_preds"] = sem_p
            aux_outputs.append(entry)

        result: OneFormer3DOutput = {
            "cls_preds": all_cls[-1],
            "masks": all_masks[-1],
            "scores": all_scores[-1],
            "aux_outputs": aux_outputs,
        }

        final_sem = all_sem[-1]
        if final_sem is not None:
            result["sem_preds"] = final_sem
        return result


class OneFormer3DSegmentation(SegmentationModel):
    r"""OneFormer3D: one transformer for unified point-cloud segmentation.

    Reference: :arxiv: [Kolodiazhnyi et al., 2024](https://arxiv.org/abs/2311.14405).
    Reference implementation: :github:
    [filaPro/oneformer3d](https://github.com/filaPro/oneformer3d).

    Voxel features go through the [`SPFormerUNetSegmentation`](#) backbone `unet` (built with
    `num_classes=0`, so it returns per-voxel features). With `superpoint_pooling`
    (ScanNet), those features are projected back to points via `inverse` and
    aggregated into superpoints by `scatter_mean`; otherwise (S3DIS) the per-voxel
    features are used directly. The query decoder `head` consumes them and produces
    a dict with `cls_preds`, `masks`, `scores`, optional `sem_preds`, and optional
    `aux_outputs`. Use [`predict_instance`](#) and [`predict_semantic`](#) for the
    final point-level instance and semantic masks.

    Args:
        in_channels: Number of input voxel features (typically $6$: RGB + centered $xyz$).
        num_classes: Number of semantic classes including stuff classes ($20$ on ScanNet).
        num_instance_classes: Number of instance / thing classes ($18$ on ScanNet).
            Defaults to `num_classes` when not given.
        channels: Per-level U-Net channel widths, deepest level last.
        layers: Residual blocks per U-Net level; an `int` is broadcast to every level.
        embed_dim: Decoder embedding dimension.
        num_layers: Number of decoder layers.
        num_instance_queries: Number of learned instance queries.
        num_semantic_queries: Number of learned semantic queries.
        num_heads: Number of attention heads.
        mlp_dim: FFN hidden dimension.
        dropout: Dropout used in the decoder transformer.
        iter_pred: Whether to run iterative mask-attention prediction.
        attn_mask: Whether to mask attention with the previous prediction.
        objectness_flag: Predict a per-query confidence score.
        semantic_head: Add an `out_sem` head over queries (ScanNet); `False` when
            semantics come from dedicated semantic queries (S3DIS).
        superpoint_pooling: Pool voxel features into superpoints before the decoder
            (ScanNet); `False` runs the decoder on voxel features directly (S3DIS).
        num_semantic_linears: Number of linear layers in the semantic head.
        act: Backbone activation passed to `create_act`.
        act_kwargs: Extra keyword arguments for the backbone activation.
        norm: Backbone normalization passed to `create_norm`.
        norm_kwargs: Extra keyword arguments for the backbone normalization (e.g. `eps`, `momentum`).
        spatial_padding: Padding (in voxels) used when constructing the
            `SparseConvTensor`.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        *,
        num_instance_classes: Optional[int] = None,
        channels: Sequence[int] = (32, 64, 96, 128, 160),
        layers: Union[int, Sequence[int]] = 2,
        embed_dim: int = 256,
        num_layers: int = 6,
        num_instance_queries: int = 0,
        num_semantic_queries: int = 0,
        num_heads: int = 8,
        mlp_dim: int = 1024,
        dropout: float = 0.0,
        iter_pred: bool = True,
        attn_mask: bool = True,
        objectness_flag: bool = False,
        semantic_head: bool = True,
        superpoint_pooling: bool = True,
        num_semantic_linears: int = 1,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        spatial_padding: int = 96,
    ) -> None:
        super().__init__(in_channels=in_channels, num_classes=num_classes)
        self.num_semantic_classes = num_classes
        self.num_instance_classes = num_instance_classes if num_instance_classes is not None else num_classes
        self.channels = tuple(channels)
        self.layers = layers
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_instance_queries = num_instance_queries
        self.num_semantic_queries = num_semantic_queries
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.dropout = dropout
        self.iter_pred = iter_pred
        self.attn_mask = attn_mask
        self.objectness_flag = objectness_flag
        self.semantic_head = semantic_head
        self.superpoint_pooling = superpoint_pooling
        self.num_semantic_linears = num_semantic_linears
        self.act = act
        self.act_kwargs = act_kwargs
        self.norm = norm
        self.norm_kwargs = norm_kwargs
        self.spatial_padding = spatial_padding

        self.unet = self.configure_unet()
        self.head = self.configure_head()

    def configure_unet(self) -> SPFormerUNetSegmentation:
        """Builds the headless sparse U-Net backbone producing the per-point features."""
        return SPFormerUNetSegmentation(
            in_channels=self.in_channels,
            num_classes=0,
            channels=self.channels,
            layers=self.layers,
            stem_kernel_size=3,
            spatial_padding=self.spatial_padding,
            act=self.act,
            act_kwargs=self.act_kwargs,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
        )

    def configure_head(self) -> OneFormer3DQueryDecoder:
        return OneFormer3DQueryDecoder(
            in_channels=self.channels[0],
            num_instance_classes=self.num_instance_classes,
            num_semantic_classes=self.num_semantic_classes,
            embed_dim=self.embed_dim,
            num_layers=self.num_layers,
            num_instance_queries=self.num_instance_queries,
            num_semantic_queries=self.num_semantic_queries,
            num_heads=self.num_heads,
            mlp_dim=self.mlp_dim,
            dropout=self.dropout,
            iter_pred=self.iter_pred,
            attn_mask=self.attn_mask,
            objectness_flag=self.objectness_flag,
            semantic_head=self.semantic_head,
            num_semantic_linears=self.num_semantic_linears,
        )

    def reset_classifier(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.num_semantic_classes = num_classes
        self.num_instance_classes = num_classes
        self.head.num_semantic_classes = num_classes
        self.head.num_instance_classes = num_classes
        embed_dim = self.head.embed_dim
        self.head.out_cls = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, num_classes + 1),
        )
        if self.semantic_head:
            if self.num_semantic_linears == 2:
                self.head.out_sem = nn.Sequential(
                    nn.Linear(embed_dim, embed_dim),
                    nn.ReLU(),
                    nn.Linear(embed_dim, num_classes + 1),
                )
            else:
                self.head.out_sem = nn.Linear(embed_dim, num_classes + 1)

    def forward_features(
        self,
        x: Tensor,
        pos_grid: Tensor,
        batch: Tensor,
    ) -> Tensor:
        return self.unet(x, pos_grid, batch)

    def forward_decoder(
        self,
        feats: Tensor,
        batch: Tensor,
        superpoint: OptTensor = None,
        inverse: OptTensor = None,
    ) -> List[Tensor]:
        r"""Split per-voxel features into per-scene decoder sources.

        With `superpoint_pooling` (ScanNet), features are projected back to points
        via `inverse` and aggregated into superpoints by `scatter_mean`. Otherwise
        (S3DIS) the per-voxel features are used directly, split by scene.
        """
        if self.superpoint_pooling:
            if superpoint is None or inverse is None:
                raise ValueError("`superpoint` and `inverse` are required when `superpoint_pooling` is enabled.")
            sp_shift, batch_offsets = _shift_superpoints(superpoint, inverse, batch)
            sp_feat = scatter_mean(feats[inverse], sp_shift, dim=0)
            return [sp_feat[batch_offsets[i] : batch_offsets[i + 1]] for i in range(len(batch_offsets) - 1)]
        batch_size = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
        return [feats[batch == i] for i in range(batch_size)]

    @overload
    def forward_head(self, sources: List[Tensor], pre_logits: Literal[False] = False) -> OneFormer3DOutput: ...

    @overload
    def forward_head(self, sources: List[Tensor], pre_logits: Literal[True]) -> List[Tensor]: ...

    def forward_head(self, sources: List[Tensor], pre_logits: bool = False) -> Union[OneFormer3DOutput, List[Tensor]]:
        if pre_logits:
            return sources
        # With learned instance queries (S3DIS), the decoder ignores the source
        # features as queries; otherwise (ScanNet) the superpoint features seed them.
        queries = sources if self.num_instance_queries == 0 else None
        return self.head(sources, queries)

    def forward(
        self,
        x: Tensor,
        pos_grid: Tensor,
        batch: Tensor,
        superpoint: OptTensor = None,
        inverse: OptTensor = None,
    ) -> OneFormer3DOutput:
        feats = self.forward_features(x, pos_grid, batch)
        sources = self.forward_decoder(feats, batch, superpoint, inverse)
        return self.forward_head(sources)

    @torch.no_grad()
    def predict_semantic(
        self,
        output: OneFormer3DOutput,
        index: Tensor,
        classes: Optional[Sequence[int]] = None,
    ) -> Tensor:
        r"""Per-point semantic predictions for a single scene.

        Args:
            output: Decoder output for a single scene (first batch element).
            index: Maps each output point to its prediction unit, shape $(N,)$.
                With `semantic_head` (ScanNet) this is the per-point superpoint id;
                otherwise (S3DIS) it is the voxel index each point falls into.
            classes: Optional subset of semantic class ids to argmax over.

        Returns:
            Per-point semantic labels of shape $(N,)$.
        """
        if self.semantic_head:
            sem_preds = output["sem_preds"][0]
            cols = list(classes) if classes is not None else list(range(sem_preds.shape[1] - 1))
            return sem_preds[:, cols].argmax(dim=1)[index]

        # S3DIS: semantics come from the last `num_semantic_queries` mask predictions.
        # `argmax` is invariant to the monotonic sigmoid; reducing over voxels before
        # indexing avoids expanding to all points first.
        sem_masks = output["masks"][0][-self.num_semantic_queries :]
        return sem_masks.argmax(dim=0)[index]

    @torch.no_grad()
    def predict_instance(
        self,
        output: OneFormer3DOutput,
        superpoint_per_point: Tensor,
        *,
        topk: int = 600,
        score_threshold: float = 0.0,
        sp_score_threshold: float = 0.4,
        npoint_threshold: int = 100,
        obj_normalization: bool = True,
        obj_normalization_threshold: float = 0.5,
        nms: bool = True,
        nms_kernel: str = "linear",
        nms_sigma: float = 2.0,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        r"""Per-point instance predictions for a single scene.

        With learned semantic queries (S3DIS), the last `num_semantic_queries` queries carry semantics
        and are excluded; only the instance queries are decoded. The defaults are the released ScanNet
        settings; the released S3DIS checkpoint uses `topk=450`, `sp_score_threshold=0.15`,
        `npoint_threshold=300` and `obj_normalization_threshold=0.01`.

        Args:
            output: Decoder output for a single scene (first batch element).
            superpoint_per_point: Per-point superpoint indices for that scene (the voxelization
                inverse when the model runs without superpoint pooling).
            topk: Maximum number of instances to keep before NMS.
            score_threshold: Drop instances with score below this value.
            sp_score_threshold: Threshold applied to the per-superpoint sigmoid
                mask logits.
            npoint_threshold: Drop instances whose mask has fewer points than this.
            obj_normalization: Rescale scores by the average mask probability.
            obj_normalization_threshold: Sigmoid threshold selecting the mask entries averaged by
                `obj_normalization`.
            nms: Apply matrix NMS to the predicted masks.
            nms_kernel: NMS decay kernel; `"linear"` or `"gaussian"`.
            nms_sigma: Sigma used by the Gaussian decay kernel.

        Returns:
            Tuple of per-point boolean instance masks $(P, N)$, instance labels $(P,)$,
            and instance scores $(P,)$.
        """
        cls_preds = output["cls_preds"][0]
        pred_masks = output["masks"][0]
        objectness = output["scores"][0]
        if self.num_semantic_queries > 0:
            cls_preds = cls_preds[: -self.num_semantic_queries]
            pred_masks = pred_masks[: -self.num_semantic_queries]
            objectness = objectness[: -self.num_semantic_queries] if objectness is not None else None
        scores_q = F.softmax(cls_preds, dim=-1)[:, :-1]
        if objectness is not None:
            scores_q = scores_q * objectness
        num_classes = self.num_instance_classes
        labels = torch.arange(num_classes, device=scores_q.device).unsqueeze(0).repeat(len(cls_preds), 1).flatten(0, 1)
        scores_flat, topk_idx = scores_q.flatten(0, 1).topk(min(topk, scores_q.numel()), sorted=False)
        labels = labels[topk_idx]
        topk_idx = torch.div(topk_idx, num_classes, rounding_mode="floor")
        mask_pred = pred_masks[topk_idx]
        mask_sig = mask_pred.sigmoid()

        if obj_normalization:
            pos = mask_sig > obj_normalization_threshold
            mask_scores = (mask_sig * pos).sum(1) / (pos.sum(1) + 1e-6)
            scores_flat = scores_flat * mask_scores

        if nms:
            scores_flat, labels, mask_sig, _ = _mask_matrix_nms(
                mask_sig,
                labels,
                scores_flat,
                kernel=nms_kernel,
                sigma=nms_sigma,
            )

        mask_sig = mask_sig[:, superpoint_per_point]
        mask_pred_bool = mask_sig > sp_score_threshold

        keep = scores_flat > score_threshold
        scores_flat = scores_flat[keep]
        labels = labels[keep]
        mask_pred_bool = mask_pred_bool[keep]

        mask_pointnum = mask_pred_bool.sum(1)
        keep = mask_pointnum > npoint_threshold
        scores_flat = scores_flat[keep]
        labels = labels[keep]
        mask_pred_bool = mask_pred_bool[keep]
        return mask_pred_bool, labels, scores_flat


def _shift_superpoints(
    superpoint: Tensor,
    inverse: Tensor,
    batch: Tensor,
) -> Tuple[Tensor, List[int]]:
    """Relabel per-scene superpoint ids into globally consecutive ids.

    Each scene's superpoints are relabeled to a gap-free range (so backbone
    features `scatter_mean` into a dense per-superpoint tensor) and offset so the
    per-scene ranges are disjoint. Returns the relabeled per-point ids and a
    `batch_offsets` list of length $B + 1$ giving the cumulative superpoint count
    per scene.
    """
    voxel_batch = batch[inverse]
    if superpoint.numel() == 0:
        return superpoint.clone(), [0]

    # Joint key unique per (scene, superpoint); the scene term is high-order so the
    # consecutive ids come out grouped by scene in ascending order, giving disjoint
    # per-scene blocks usable as `batch_offsets` slices.
    key = voxel_batch * (int(superpoint.max()) + 1) + superpoint
    shifted, _ = consecutive_cluster(key)
    scene_of = shifted.new_zeros(int(shifted.max()) + 1).scatter_(0, shifted, voxel_batch)
    counts = torch.bincount(scene_of, minlength=int(voxel_batch.max()) + 1)
    offsets: List[int] = [0, *torch.cumsum(counts, dim=0).tolist()]
    return shifted, offsets


def _mask_matrix_nms(
    masks: Tensor,
    labels: Tensor,
    scores: Tensor,
    kernel: str = "gaussian",
    sigma: float = 2.0,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Mask matrix NMS: decay each mask's score by its overlap with higher-scoring masks of the same class.

    Args:
        masks: Per-instance soft masks $(P, M)$.
        labels: Per-instance class labels $(P,)$.
        scores: Per-instance scores $(P,)$.
        kernel: `"linear"` or `"gaussian"` decay kernel.
        sigma: Sigma used by the gaussian decay kernel.

    Returns:
        Tuple `(scores, labels, masks, keep_inds)` with NMS-adjusted scores.
    """
    if len(labels) == 0:
        zero = scores.new_zeros(0)
        return zero, labels.new_zeros(0), masks.new_zeros(0, *masks.shape[-1:]), labels.new_zeros(0)

    mask_area = masks.sum(1).float()
    scores, sort_inds = torch.sort(scores, descending=True)
    keep_inds = sort_inds
    masks = masks[sort_inds]
    mask_area = mask_area[sort_inds]
    labels = labels[sort_inds]

    num_masks = len(labels)
    flat = masks.reshape(num_masks, -1).float()
    inter = torch.mm(flat, flat.transpose(1, 0))
    area_expand = mask_area.expand(num_masks, num_masks)
    iou = (inter / (area_expand + area_expand.transpose(1, 0) - inter)).triu(diagonal=1)
    labels_expand = labels.expand(num_masks, num_masks)
    same_label = (labels_expand == labels_expand.transpose(1, 0)).triu(diagonal=1)

    compensate_iou, _ = (iou * same_label).max(0)
    compensate_iou = compensate_iou.expand(num_masks, num_masks).transpose(1, 0)
    decay_iou = iou * same_label

    if kernel == "gaussian":
        decay_matrix = torch.exp(-sigma * (decay_iou**2))
        compensate_matrix = torch.exp(-sigma * (compensate_iou**2))
        decay, _ = (decay_matrix / compensate_matrix).min(0)
    elif kernel == "linear":
        decay_matrix = (1 - decay_iou) / (1 - compensate_iou)
        decay, _ = decay_matrix.min(0)
    else:
        raise ValueError(f"Unsupported kernel {kernel!r}. Expected 'linear' or 'gaussian'.")
    scores = scores * decay

    scores, sort_inds = torch.sort(scores, descending=True)
    keep_inds = keep_inds[sort_inds]
    masks = masks[sort_inds]
    labels = labels[sort_inds]
    return scores, labels, masks, keep_inds


_ONEFORMER3D_SCANNET_TRANSFORMS: Callable[..., Any] = T.Compose(
    [
        T.Normalize(keys=DataKeys.COLOR, mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5]),
        T.CopyItems(keys=DataKeys.POS, names="pos_centered"),
        T.Shift(keys="pos_centered", method="centroid"),
        T.Cat(keys=[DataKeys.COLOR, "pos_centered"], dst_key=DataKeys.X, dim=1),
        T.Relabel(keys=DataKeys.SEGMENT, labels=range(1, 21), default=-1),
        T.Shift(keys=DataKeys.POS, method="min"),
        T.Voxelize(
            pos_key=DataKeys.POS,
            pos_reduce="grid",
            keys=[DataKeys.X, DataKeys.SEGMENT],
            reduce=["mean", "first"],
            size=0.02,
            method="fnv",
            dst_inverse_key=DataKeys.INVERSE,
        ),
        T.CopyItems(keys=DataKeys.POS, names=DataKeys.POS_GRID),
    ]
)


@register_model(
    "oneformer3d-base.scannet20.danila-rukhovich",
    task="segmentation",
    weights=WeightsDict(
        url="hf://torch-pointcloud/oneformer3d/oneformer3d-base.scannet20.danila-rukhovich.safetensors",
        dataset="scannet20",
        classes=SCANNET20_CLASSES,
        author="danila-rukhovich",
        license="CC-BY-NC-4.0",
    ),
    transform=_ONEFORMER3D_SCANNET_TRANSFORMS,
    hparams=dict(
        in_channels=6,
        num_classes=20,
        num_instance_classes=18,
        channels=[32, 64, 96, 128, 160],
        layers=2,
        embed_dim=256,
        num_layers=6,
        num_heads=8,
        mlp_dim=1024,
        dropout=0.0,
        iter_pred=True,
        attn_mask=True,
        objectness_flag=False,
        num_semantic_linears=1,
        norm_kwargs=dict(eps=1e-4, momentum=0.1),
        spatial_padding=96,
    ),
)
def oneformer3d_base_scannet20(**hparams: Any) -> OneFormer3DSegmentation:
    return OneFormer3DSegmentation(**hparams)


@register_model(
    "oneformer3d-base.scannet200.danila-rukhovich",
    task="segmentation",
    # The released ScanNet200 checkpoint's backbone (initialized from Mask3D) uses a different sparse-conv
    # engine and cannot be converted to this SpConv backbone, so no pretrained weights are registered.
    weights=None,
    transform=_ONEFORMER3D_SCANNET_TRANSFORMS,
    hparams=dict(
        in_channels=6,
        num_classes=200,
        num_instance_classes=198,
        channels=[32, 64, 96, 128, 160],
        layers=2,
        embed_dim=256,
        num_layers=6,
        num_heads=8,
        mlp_dim=1024,
        dropout=0.0,
        iter_pred=True,
        attn_mask=True,
        objectness_flag=False,
        num_semantic_linears=1,
        norm_kwargs=dict(eps=1e-4, momentum=0.1),
        spatial_padding=96,
    ),
)
def oneformer3d_base_scannet200(**hparams: Any) -> OneFormer3DSegmentation:
    return OneFormer3DSegmentation(**hparams)


_ONEFORMER3D_S3DIS_TRANSFORMS: Callable[..., Any] = T.Compose(
    [
        T.Normalize(keys=DataKeys.COLOR, mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5]),
        T.CopyItems(keys=DataKeys.POS, names="pos_centered"),
        T.Shift(keys="pos_centered", method="centroid"),
        T.Cat(keys=[DataKeys.COLOR, "pos_centered"], dst_key=DataKeys.X, dim=1),
        T.Shift(keys=DataKeys.POS, method="min"),
        T.Voxelize(
            pos_key=DataKeys.POS,
            pos_reduce="grid",
            keys=[DataKeys.X, DataKeys.SEGMENT],
            reduce=["mean", "first"],
            size=0.05,
            method="fnv",
            dst_inverse_key=DataKeys.INVERSE,
        ),
        T.CopyItems(keys=DataKeys.POS, names=DataKeys.POS_GRID),
    ]
)


@register_model(
    "oneformer3d-base.s3dis-area5.danila-rukhovich",
    task="segmentation",
    weights=WeightsDict(
        url="hf://torch-pointcloud/oneformer3d/oneformer3d-base.s3dis-area5.danila-rukhovich.safetensors",
        dataset="s3dis-area5",
        classes=S3DIS_CLASSES,
        author="danila-rukhovich",
        license="CC-BY-NC-4.0",
    ),
    transform=_ONEFORMER3D_S3DIS_TRANSFORMS,
    hparams=dict(
        in_channels=6,
        num_classes=13,
        num_instance_classes=13,
        channels=[64, 128, 192, 256, 320],
        layers=2,
        embed_dim=256,
        num_layers=3,
        num_instance_queries=400,
        num_semantic_queries=13,
        num_heads=8,
        mlp_dim=1024,
        dropout=0.0,
        iter_pred=True,
        attn_mask=True,
        objectness_flag=True,
        semantic_head=False,
        superpoint_pooling=False,
        norm_kwargs=dict(eps=1e-4, momentum=0.1),
        spatial_padding=96,
    ),
)
def oneformer3d_base_s3dis_area5(**hparams: Any) -> OneFormer3DSegmentation:
    return OneFormer3DSegmentation(**hparams)
