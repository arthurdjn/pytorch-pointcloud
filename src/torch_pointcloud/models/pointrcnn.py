"""PointRCNN detection model.

{{ paper("1812.04244") }}
"""

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TypedDict, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import MLP

import torch_pointcloud.transforms as T
from torch_pointcloud.layers.pointnet2_blocks import GlobalSAModule, SAModule
from torch_pointcloud.utils.box3d import boxes_iou3d, decode_box_residuals, nms3d
from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.types import Detection3D, OptTensor

from ._base import DetectionModel
from ._registry import WeightsDict, register_model
from .pointnet2 import PointNet2Decoder, PointNet2Encoder


class PointRCNNOutput(TypedDict):
    r"""Inference-mode PointRCNN output: refined boxes with stage-2 confidences (packed layout).

    Attributes:
        rcnn_cls: Stage-2 confidence logit per ROI, shape $(M, 1)$.
        boxes: Refined boxes $(c_x, c_y, c_z, d_x, d_y, d_z, \theta)$, shape $(M, 7)$.
        roi_labels: Stage-1 ROI class per box ($1$-based), shape $(M,)$.
        roi_scores: Stage-1 sigmoid proposal score per box, shape $(M,)$.
        batch: Per-ROI scene index, shape $(M,)$.
    """

    rcnn_cls: Tensor
    boxes: Tensor
    roi_labels: Tensor
    roi_scores: Tensor
    batch: Tensor


class PointRCNNTrainOutput(TypedDict):
    r"""Training-mode PointRCNN output: stage-1 point predictions plus sampled-ROI stage-2 tensors.

    Attributes:
        point_cls_preds: Stage-1 per-point class logits, shape $(N, \text{num\_classes})$.
        point_box_preds: Stage-1 per-point box residuals, shape $(N, 8)$.
        point_pos: Per-point coordinates, shape $(N, 3)$.
        point_batch: Per-point scene index, shape $(N,)$.
        rcnn_cls: Stage-2 confidence logit per sampled ROI, shape $(M, 1)$.
        rcnn_reg: Stage-2 raw ROI box residuals, shape $(M, 7)$.
        rcnn_boxes: Stage-2 refined boxes in the lidar frame, shape $(M, 7)$.
        rois: Sampled proposal boxes (generated without gradient), shape $(M, 7)$.
        gt_of_rois: ROI-canonical matched ground-truth box, shape $(M, 7)$.
        gt_of_rois_src: Lidar-frame matched ground-truth box, shape $(M, 7)$.
        roi_ious: Per-ROI max IoU with the matched box, shape $(M,)$.
    """

    point_cls_preds: Tensor
    point_box_preds: Tensor
    point_pos: Tensor
    point_batch: Tensor
    rcnn_cls: Tensor
    rcnn_reg: Tensor
    rcnn_boxes: Tensor
    rois: Tensor
    gt_of_rois: Tensor
    gt_of_rois_src: Tensor
    roi_ious: Tensor


def rotate_points_along_z(points: Tensor, angle: Tensor) -> Tensor:
    r"""Rotate point sets about the $+z$ axis (angle increases $x \to y$).

    Args:
        points: Point sets, shape $(B, N, 3 + C)$; only the first three channels are rotated.
        angle: Per-set yaw, shape $(B,)$.

    Returns:
        The rotated point sets, shape $(B, N, 3 + C)$.

    Shape:
        - points: $(B, N, 3 + C)$
        - angle: $(B,)$
        - output: $(B, N, 3 + C)$
    """
    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    ones = angle.new_ones(points.shape[0])
    rot = torch.stack((cosa, sina, zeros, -sina, cosa, zeros, zeros, zeros, ones), dim=1).view(-1, 3, 3)
    pos = torch.matmul(points[:, :, 0:3], rot)
    return torch.cat((pos, points[:, :, 3:]), dim=-1)


def decode_point_residuals(encodings: Tensor, points: Tensor, classes: Tensor, mean_sizes: Tensor) -> Tensor:
    r"""Decode per-point box residuals with class mean-size anchors.

    Decodes a stage-1 prediction $(x_t, y_t, z_t, d_{x,t}, d_{y,t}, d_{z,t}, \cos, \sin)$ at a foreground
    point into an oriented box $(c_x, c_y, c_z, d_x, d_y, d_z, \theta)$, using the predicted class mean size
    as the anchor for the size residuals.

    Args:
        encodings: Box residuals, shape $(N, 8)$.
        points: Anchor point coordinates, shape $(N, 3)$.
        classes: Predicted class index per point ($1 \ldots \text{num\_classes}$), shape $(N,)$.
        mean_sizes: Per-class mean box size $(d_x, d_y, d_z)$, shape $(\text{num\_classes}, 3)$.

    Returns:
        Decoded boxes $(c_x, c_y, c_z, d_x, d_y, d_z, \theta)$, shape $(N, 7)$.

    Shape:
        - encodings: $(N, 8)$
        - points: $(N, 3)$
        - output: $(N, 7)$
    """
    xt, yt, zt, dxt, dyt, dzt, cost, sint = torch.split(encodings, 1, dim=-1)
    xa, ya, za = torch.split(points, 1, dim=-1)

    anchor = mean_sizes[classes - 1]
    dxa, dya, dza = torch.split(anchor, 1, dim=-1)
    diagonal = torch.sqrt(dxa**2 + dya**2)

    xg = xt * diagonal + xa
    yg = yt * diagonal + ya
    zg = zt * dza + za
    dxg = torch.exp(dxt) * dxa
    dyg = torch.exp(dyt) * dya
    dzg = torch.exp(dzt) * dza
    rg = torch.atan2(sint, cost)
    return torch.cat([xg, yg, zg, dxg, dyg, dzg, rg], dim=-1)


class PointHeadBox(nn.Module):
    r"""Stage-1 per-point foreground head + bin-free box proposal generation (`PointHeadBox`).

    Two MLPs over the per-point backbone features predict a per-point class logit and an 8-D box residual.
    At inference every point becomes a proposal: the class score is the sigmoid of the max class logit and
    the box is decoded by
    [`decode_point_residuals`][torch_pointcloud.models.pointrcnn.decode_point_residuals]
    against the point's predicted class mean size.

    Args:
        in_channels: Backbone feature channels per point.
        num_classes: Number of foreground classes.
        cls_channels: Hidden channels of the classification MLP.
        reg_channels: Hidden channels of the box-regression MLP.
        mean_sizes: Per-class mean box size $(d_x, d_y, d_z)$, shape $(\text{num\_classes}, 3)$.
        act: Activation type or callable.
        act_kwargs: Extra activation arguments.
        norm: Normalization type or callable.
        norm_kwargs: Extra normalization arguments.
    """

    mean_sizes: Tensor

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        *,
        cls_channels: Sequence[int],
        reg_channels: Sequence[int],
        mean_sizes: Tensor,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.register_buffer("mean_sizes", mean_sizes, persistent=False)
        self.cls_layers = MLP(
            [in_channels, *cls_channels, num_classes],
            act=act,
            act_kwargs=act_kwargs,
            norm=norm,
            norm_kwargs=norm_kwargs,
            bias=[False] * len(cls_channels) + [True],
            plain_last=True,
        )
        self.box_layers = MLP(
            [in_channels, *reg_channels, 8],
            act=act,
            act_kwargs=act_kwargs,
            norm=norm,
            norm_kwargs=norm_kwargs,
            bias=[False] * len(reg_channels) + [True],
            plain_last=True,
        )

    def forward(self, x: Tensor, pos: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        r"""Predict per-point class scores, raw box residuals and decoded proposal boxes.

        Args:
            x: Per-point backbone features, shape $(N, C)$.
            pos: Per-point coordinates, shape $(N, 3)$.

        Returns:
            A tuple `(point_scores, cls_preds, box_preds, boxes)` of the sigmoid foreground score $(N,)$,
            the raw class logits $(N, \text{num\_classes})$, the raw box residuals $(N, 8)$ (the stage-1
            regression targets are formed against these), and the decoded boxes $(N, 7)$.

        Shape:
            - x: $(N, C)$
            - pos: $(N, 3)$
            - output: $(N,)$, $(N, \text{num\_classes})$, $(N, 8)$, $(N, 7)$
        """
        cls_preds = self.cls_layers(x)
        box_preds = self.box_layers(x)
        point_scores = torch.sigmoid(cls_preds.max(dim=-1).values)
        pred_classes = cls_preds.argmax(dim=-1) + 1
        boxes = decode_point_residuals(box_preds, pos, pred_classes, self.mean_sizes)
        return point_scores, cls_preds, box_preds, boxes


class PointRCNNRefinementHead(nn.Module):
    r"""Stage-2 ROI refinement head (`PointRCNNHead`): point ROI pooling + canonical transform + PointNet++.

    For each proposal it pools a fixed number of input points inside the (optionally enlarged) box, appends
    the per-point foreground score and a depth feature, canonically transforms the pooled points (translate
    to the ROI center, rotate by $-\theta$), lifts the canonical xyz with an MLP, fuses it with the pooled
    point features, and runs a small PointNet++ to produce a confidence logit and a 7-D box refinement.

    Args:
        in_channels: Pooled point-feature channels (the stage-1 backbone feature dim).
        sa_channels: Per-SA-block MLP channel lists.
        sa_npoints: Per-SA-block sample counts; `-1` groups all remaining points.
        sa_radii: Per-SA-block ball-query radii.
        sa_num_neighbors: Per-SA-block neighbor caps.
        xyz_up_channels: Channels of the canonical-xyz lifting MLP.
        cls_channels: Hidden channels of the confidence MLP.
        reg_channels: Hidden channels of the box-refinement MLP.
        num_sampled_points: Points pooled per ROI.
        pool_extra_width: Per-axis enlargement of the pooling box.
        depth_normalizer: Divisor for the point-depth feature.
        act: Activation type or callable.
        act_kwargs: Extra activation arguments.
        norm: Normalization type or callable.
        norm_kwargs: Extra normalization arguments.
    """

    def __init__(
        self,
        in_channels: int,
        *,
        sa_channels: Sequence[Sequence[int]],
        sa_npoints: Sequence[int],
        sa_radii: Sequence[float],
        sa_num_neighbors: Sequence[int],
        xyz_up_channels: Sequence[int],
        cls_channels: Sequence[int],
        reg_channels: Sequence[int],
        num_sampled_points: int = 512,
        pool_extra_width: Sequence[float] = (0.0, 0.0, 0.0),
        depth_normalizer: float = 70.0,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.num_sampled_points = num_sampled_points
        self.pool_extra_width = tuple(pool_extra_width)
        self.depth_normalizer = depth_normalizer
        self.num_prefix_channels = 3 + 2

        xyz_mlps = [self.num_prefix_channels, *xyz_up_channels]
        xyz_layers: List[nn.Module] = []
        for i in range(len(xyz_mlps) - 1):
            xyz_layers.append(nn.Conv2d(xyz_mlps[i], xyz_mlps[i + 1], kernel_size=1, bias=True))
            xyz_layers.append(nn.ReLU())
        self.xyz_up_layer = nn.Sequential(*xyz_layers)

        c_out = xyz_up_channels[-1]
        self.merge_down_layer = nn.Sequential(
            nn.Conv2d(c_out * 2, c_out, kernel_size=1, bias=True),
            nn.ReLU(),
        )

        channel_in = in_channels
        self.sa_modules = nn.ModuleList()
        for channels, npoint, radius, num_neighbors in zip(sa_channels, sa_npoints, sa_radii, sa_num_neighbors):
            if npoint == -1:
                module: nn.Module = GlobalSAModule(
                    channel_in,
                    list(channels),
                    use_pos=True,
                    pos_first=True,
                    act=act,
                    act_kwargs=act_kwargs,
                    norm=norm,
                    norm_kwargs=norm_kwargs,
                )
            else:
                module = SAModule(
                    in_channels=channel_in,
                    channels=list(channels),
                    num_points=npoint,
                    radii=radius,
                    num_neighbors=num_neighbors,
                    use_pos=True,
                    normalize_pos=False,
                    pos_first=True,
                    pool="max",
                    bias=False,
                    act=act,
                    act_kwargs=act_kwargs,
                    norm=norm,
                    norm_kwargs=norm_kwargs,
                )
            self.sa_modules.append(module)
            channel_in = channels[-1]

        self.cls_layers = MLP(
            [channel_in, *cls_channels, 1],
            act=act,
            act_kwargs=act_kwargs,
            norm=norm,
            norm_kwargs=norm_kwargs,
            bias=[False] * len(cls_channels) + [True],
            plain_last=True,
        )
        self.reg_layers = MLP(
            [channel_in, *reg_channels, 7],
            act=act,
            act_kwargs=act_kwargs,
            norm=norm,
            norm_kwargs=norm_kwargs,
            bias=[False] * len(reg_channels) + [True],
            plain_last=True,
        )

    def roipool(self, pos: Tensor, x: Tensor, rois: Tensor) -> Tuple[Tensor, Tensor]:
        r"""Pool a fixed number of in-box points per ROI and canonically transform them.

        Mirrors the reference `roipoint_pool3d` CUDA kernel: for every (optionally enlarged) box, the input
        points are scanned in order and the first `num_sampled_points` that fall inside are kept (cyclically
        duplicated when fewer are found, zeroed when none). Pooled points are translated to the ROI center
        and rotated by $-\theta$ into the box-canonical frame.

        Args:
            pos: Per-point coordinates of one scene, shape $(N, 3)$.
            x: Per-point pooled features (score + depth + backbone), shape $(N, 5 + C)$.
            rois: Proposal boxes $(x, y, z, d_x, d_y, d_z, \theta)$, shape $(M, 7)$.

        Returns:
            A tuple `(pooled, empty)` of the pooled features $(M, S, 3 + (5 + C))$ in canonical xyz and a
            boolean ROI-empty flag $(M,)$.

        Shape:
            - pos: $(N, 3)$
            - x: $(N, 5 + C)$
            - rois: $(M, 7)$
            - output: $(M, S, 3 + 5 + C)$, $(M,)$
        """
        m = rois.shape[0]
        n = pos.shape[0]
        s = self.num_sampled_points
        channels = x.shape[1]

        extra = pos.new_tensor(self.pool_extra_width)
        enlarged = rois.clone()
        enlarged[:, 3:6] = enlarged[:, 3:6] + extra

        center = enlarged[:, 0:3]
        half = enlarged[:, 3:6] / 2.0
        heading = enlarged[:, 6]
        cosa = torch.cos(-heading)
        sina = torch.sin(-heading)

        shift = pos.unsqueeze(0) - center.unsqueeze(1)
        local_x = shift[..., 0] * cosa.unsqueeze(1) + shift[..., 1] * (-sina).unsqueeze(1)
        local_y = shift[..., 0] * sina.unsqueeze(1) + shift[..., 1] * cosa.unsqueeze(1)
        margin = 1e-5
        in_z = shift[..., 2].abs() <= half[:, 2:3]
        in_x = local_x.abs() < half[:, 0:1] + margin
        in_y = local_y.abs() < half[:, 1:2] + margin
        in_box = in_z & in_x & in_y

        order = torch.arange(n, device=pos.device).unsqueeze(0).expand(m, -1)
        ranked = order.masked_fill(~in_box, n)
        ranked, _ = ranked.sort(dim=1)
        counts = in_box.sum(dim=1)
        empty = counts == 0

        positions = torch.arange(s, device=pos.device).unsqueeze(0).expand(m, -1)
        safe_counts = counts.clamp_min(1).unsqueeze(1)
        within = positions < counts.unsqueeze(1)
        gather_rank = torch.where(within, positions, positions % safe_counts)
        sampled_idx = torch.gather(ranked, 1, gather_rank).clamp_max(n - 1)

        pooled_xyz = pos[sampled_idx]
        pooled_feat = x[sampled_idx]
        pooled = torch.cat([pooled_xyz, pooled_feat], dim=2)

        pooled[..., 0:3] = pooled[..., 0:3] - rois[:, None, 0:3]
        flat = pooled.view(m * s, 3 + channels)
        rotated = rotate_points_along_z(flat[:, 0:3].unsqueeze(1), -rois[:, 6].repeat_interleave(s)).squeeze(1)
        pooled = torch.cat([rotated, flat[:, 3:]], dim=1).view(m, s, 3 + channels)
        pooled[empty] = 0
        return pooled, empty

    def forward(
        self,
        pos: Tensor,
        x: Tensor,
        point_scores: Tensor,
        batch: Tensor,
        rois: Tensor,
        roi_batch: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        r"""Refine proposals into confidence logits, raw box residuals and refined boxes.

        Args:
            pos: Per-point coordinates, shape $(N, 3)$.
            x: Per-point backbone features, shape $(N, C)$.
            point_scores: Per-point stage-1 foreground score, shape $(N,)$.
            batch: Per-point scene index, shape $(N,)$.
            rois: Proposal boxes, shape $(M, 7)$.
            roi_batch: Per-ROI scene index, shape $(M,)$.

        Returns:
            A tuple `(rcnn_cls, rcnn_reg, refined_boxes)` of the confidence logit $(M, 1)$, the raw ROI box
            residuals $(M, 7)$ (the stage-2 regression targets are formed against these) and the refined
            boxes $(M, 7)$ in the lidar frame.

        Shape:
            - pos: $(N, 3)$, x: $(N, C)$, point_scores: $(N,)$
            - rois: $(M, 7)$
            - output: $(M, 1)$, $(M, 7)$, $(M, 7)$
        """
        depth = pos.norm(dim=1) / self.depth_normalizer - 0.5
        feat_all = torch.cat([point_scores[:, None], depth[:, None], x], dim=1)

        pooled_list: List[Tensor] = []
        batch_size = int(roi_batch.max().item()) + 1 if roi_batch.numel() else 0
        for b in range(batch_size):
            scene_mask = batch == b
            roi_mask = roi_batch == b
            pooled, _ = self.roipool(pos[scene_mask], feat_all[scene_mask], rois[roi_mask])
            pooled_list.append(pooled)
        pooled = torch.cat(pooled_list, dim=0)

        xyz_input = pooled[..., 0 : self.num_prefix_channels].transpose(1, 2).unsqueeze(dim=3).contiguous()
        xyz_features = self.xyz_up_layer(xyz_input)
        point_features = pooled[..., self.num_prefix_channels :].transpose(1, 2).unsqueeze(dim=3)
        merged = torch.cat([xyz_features, point_features], dim=1)
        merged = self.merge_down_layer(merged).squeeze(dim=3)

        pooled_pos = pooled[..., 0:3].contiguous()
        x, pos_local, batch_local = self._densify(pooled_pos, merged.transpose(1, 2).contiguous())
        for sa_module in self.sa_modules:
            x, pos_local, batch_local = sa_module(x, pos_local, batch_local)

        shared = x
        rcnn_cls = self.cls_layers(shared)
        rcnn_reg = self.reg_layers(shared)
        refined = self._decode_rcnn(rois, rcnn_reg)
        return rcnn_cls, rcnn_reg, refined

    def _densify(self, pos: Tensor, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        r"""Flatten per-ROI point sets into a packed batch (one ROI = one batch element).

        Args:
            pos: Per-ROI pooled coordinates, shape $(M, S, 3)$.
            x: Per-ROI pooled features, shape $(M, S, C)$.

        Returns:
            A tuple `(x, pos, batch)` of packed features $(M \cdot S, C)$, coordinates $(M \cdot S, 3)$, and
            the per-point ROI index $(M \cdot S,)$.

        Shape:
            - pos: $(M, S, 3)$, x: $(M, S, C)$
            - output: $(M \cdot S, C)$, $(M \cdot S, 3)$, $(M \cdot S,)$
        """
        m, s, _ = pos.shape
        batch = torch.arange(m, device=pos.device).repeat_interleave(s)
        return x.reshape(m * s, -1), pos.reshape(m * s, 3), batch

    def _decode_rcnn(self, rois: Tensor, rcnn_reg: Tensor) -> Tensor:
        r"""Decode the stage-2 residual against each ROI and map back to the lidar frame.

        Args:
            rois: Proposal boxes $(M, 7)$.
            rcnn_reg: Per-ROI box residual $(M, 1, 7)$ or $(M, 7)$.

        Returns:
            Refined boxes $(M, 7)$ in the lidar frame.

        Shape:
            - rois: $(M, 7)$
            - rcnn_reg: $(M, 7)$
            - output: $(M, 7)$
        """
        roi_ry = rois[:, 6]
        roi_xyz = rois[:, 0:3]
        local = rois.clone()
        local[:, 0:3] = 0
        boxes = decode_box_residuals(rcnn_reg.view(-1, 7), local)
        boxes = rotate_points_along_z(boxes.unsqueeze(1), roi_ry).squeeze(1)
        boxes[:, 0:3] = boxes[:, 0:3] + roi_xyz
        return boxes


class PointRCNNDetection(DetectionModel):
    r"""PointRCNN two-stage point-based 3D object detector (packed point format).

    Reference: :arxiv: [Shi et al., 2019](https://arxiv.org/abs/1812.04244).
    Reference implementation: :github: [open-mmlab/OpenPCDet](https://github.com/open-mmlab/OpenPCDet).

    Stage 1 runs a multi-scale PointNet++ U-Net
    ([`PointNet2Encoder`][torch_pointcloud.models.pointnet2.PointNet2Encoder] +
    [`PointNet2Decoder`][torch_pointcloud.models.pointnet2.PointNet2Decoder]) over the raw point
    cloud, then a per-point head ([`PointHeadBox`][torch_pointcloud.models.pointrcnn.PointHeadBox]) predicts
    foreground scores and one box proposal per point. The top proposals (after class-agnostic NMS) become
    ROIs that stage 2 ([`PointRCNNRefinementHead`][torch_pointcloud.models.pointrcnn.PointRCNNRefinementHead])
    pools points around, canonically transforms, and refines into a confidence and a box correction.

    Args:
        in_channels: Raw point feature channels including xyz (e.g. $4$ for $x, y, z, \text{intensity}$).
        num_classes: Number of foreground classes.
        mean_sizes: Per-class mean box size $(d_x, d_y, d_z)$, shape $(\text{num\_classes}, 3)$.
        sa_channels: Stage-1 per-SA-block, per-scale MLP channel lists.
        sa_npoints: Stage-1 per-SA-block sample counts.
        sa_radii: Stage-1 per-SA-block, per-scale ball-query radii.
        sa_num_neighbors: Stage-1 per-SA-block, per-scale neighbor caps.
        fp_channels: Stage-1 per-FP-block MLP channel lists, ordered from the coarsest skip level to the
            finest (`PointNet2Decoder` order).
        point_cls_channels: Stage-1 classification MLP hidden channels.
        point_reg_channels: Stage-1 box-regression MLP hidden channels.
        roi_sa_channels: Stage-2 per-SA-block MLP channel lists.
        roi_sa_npoints: Stage-2 per-SA-block sample counts (`-1` groups all).
        roi_sa_radii: Stage-2 per-SA-block ball-query radii.
        roi_sa_num_neighbors: Stage-2 per-SA-block neighbor caps.
        roi_xyz_up_channels: Stage-2 canonical-xyz lifting MLP channels.
        roi_cls_channels: Stage-2 confidence MLP hidden channels.
        roi_reg_channels: Stage-2 box-refinement MLP hidden channels.
        num_sampled_points: Points pooled per ROI in stage 2.
        pool_extra_width: Per-axis enlargement of the stage-2 pooling box.
        depth_normalizer: Divisor for the stage-2 point-depth feature.
        nms_pre_maxsize: Proposals kept before stage-1 NMS.
        nms_post_maxsize: ROIs kept after stage-1 NMS at inference (the stage-2 batch size per scene).
        nms_thresh: Stage-1 proposal NMS IoU threshold at inference.
        train_nms_post_maxsize: Proposals kept after stage-1 NMS during training (before ROI sampling).
        train_nms_thresh: Stage-1 proposal NMS IoU threshold during training.
        roi_per_image: ROIs sampled per scene for stage-2 training.
        fg_ratio: Target fraction of foreground ROIs in the sampled set.
        reg_fg_thresh: ROI-to-GT IoU at or above which a sampled ROI is foreground (box regression valid).
        cls_fg_thresh: ROI-to-GT IoU used with `reg_fg_thresh` to define the foreground sampling threshold.
        cls_bg_thresh_lo: ROI-to-GT IoU below which a ROI is easy background (else hard background).
        hard_bg_ratio: Fraction of the sampled background ROIs drawn from hard (higher-IoU) background.
        act: Activation type or callable.
        act_kwargs: Extra activation arguments.
        norm: Normalization type or callable.
        norm_kwargs: Extra normalization arguments.
    """

    mean_sizes: Tensor

    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 3,
        *,
        mean_sizes: Union[Tensor, Sequence[Sequence[float]]],
        sa_channels: Sequence[Sequence[Sequence[int]]],
        sa_npoints: Sequence[int],
        sa_radii: Sequence[Sequence[float]],
        sa_num_neighbors: Sequence[Sequence[int]],
        fp_channels: Sequence[Sequence[int]],
        point_cls_channels: Sequence[int] = (256, 256),
        point_reg_channels: Sequence[int] = (256, 256),
        roi_sa_channels: Sequence[Sequence[int]],
        roi_sa_npoints: Sequence[int],
        roi_sa_radii: Sequence[float],
        roi_sa_num_neighbors: Sequence[int],
        roi_xyz_up_channels: Sequence[int] = (128, 128),
        roi_cls_channels: Sequence[int] = (256, 256),
        roi_reg_channels: Sequence[int] = (256, 256),
        num_sampled_points: int = 512,
        pool_extra_width: Sequence[float] = (0.0, 0.0, 0.0),
        depth_normalizer: float = 70.0,
        nms_pre_maxsize: int = 9000,
        nms_post_maxsize: int = 100,
        nms_thresh: float = 0.85,
        train_nms_post_maxsize: int = 512,
        train_nms_thresh: float = 0.8,
        roi_per_image: int = 128,
        fg_ratio: float = 0.5,
        reg_fg_thresh: float = 0.55,
        cls_fg_thresh: float = 0.6,
        cls_bg_thresh_lo: float = 0.1,
        hard_bg_ratio: float = 0.8,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(in_channels=in_channels, num_classes=num_classes)
        self.nms_pre_maxsize = nms_pre_maxsize
        self.nms_post_maxsize = nms_post_maxsize
        self.nms_thresh = nms_thresh
        self.train_nms_post_maxsize = train_nms_post_maxsize
        self.train_nms_thresh = train_nms_thresh
        self.roi_per_image = roi_per_image
        self.fg_ratio = fg_ratio
        self.reg_fg_thresh = reg_fg_thresh
        self.cls_fg_thresh = cls_fg_thresh
        self.cls_bg_thresh_lo = cls_bg_thresh_lo
        self.hard_bg_ratio = hard_bg_ratio

        mean = torch.as_tensor(mean_sizes, dtype=torch.float32)
        if mean.shape != (num_classes, 3):
            raise ValueError(f"`mean_sizes` must have shape ({num_classes}, 3), got {tuple(mean.shape)}.")
        self.register_buffer("mean_sizes", mean, persistent=False)

        block_kwargs: Dict[str, Any] = dict(act=act, act_kwargs=act_kwargs, norm=norm, norm_kwargs=norm_kwargs)
        self.encoder = PointNet2Encoder(
            in_channels - 3,
            sa_channels,
            num_points=sa_npoints,
            radii=sa_radii,
            num_neighbors=sa_num_neighbors,
            use_pos=True,
            normalize_pos=False,
            pos_first=True,
            **block_kwargs,
        )
        self.decoder = PointNet2Decoder(
            in_channels=self.encoder.out_channels,
            skip_channels=self.encoder.skip_channels[::-1],
            fp_channels=fp_channels,
            k=3,
            weighting="inverse",
            eps=1e-8,
            **block_kwargs,
        )
        point_channels = fp_channels[-1][-1]
        self.point_head = PointHeadBox(
            point_channels,
            num_classes,
            cls_channels=point_cls_channels,
            reg_channels=point_reg_channels,
            mean_sizes=self.mean_sizes,
            **block_kwargs,
        )
        self.roi_head = PointRCNNRefinementHead(
            point_channels,
            sa_channels=roi_sa_channels,
            sa_npoints=roi_sa_npoints,
            sa_radii=roi_sa_radii,
            sa_num_neighbors=roi_sa_num_neighbors,
            xyz_up_channels=roi_xyz_up_channels,
            cls_channels=roi_cls_channels,
            reg_channels=roi_reg_channels,
            num_sampled_points=num_sampled_points,
            pool_extra_width=pool_extra_width,
            depth_normalizer=depth_normalizer,
            **block_kwargs,
        )

    def reset_classifier(self, num_classes: int) -> None:
        raise NotImplementedError("PointRCNN's class count is fixed by its pretrained box coder mean sizes.")

    def forward_features(self, x: OptTensor, pos: Tensor, batch: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        assert x is not None, "PointRCNN requires input features (got x=None)."
        x, pos_down, batch_down, intermediates = self.encoder(x, pos, batch, return_intermediates=True)
        x, pos, batch = self.decoder(x, pos_down, batch_down, intermediates)
        return x, pos, batch

    def forward(
        self,
        x: OptTensor,
        pos: Tensor,
        batch: Tensor,
        gt_boxes: OptTensor = None,
        gt_labels: OptTensor = None,
        gt_batch: OptTensor = None,
    ) -> Union[PointRCNNTrainOutput, PointRCNNOutput]:
        r"""Run both stages; in train mode the ground truth drives the stage-2 ROI sampling.

        Stage-2 training samples its ROIs by matching stage-1 proposals to ground-truth boxes at forward
        time, so train mode requires the packed ground truth. The GT arguments default to `None` and are
        omitted at inference (a training pipeline passes `box` / `label` / `batch_box` after the point
        inputs, e.g. via `input_keys`).

        Args:
            x: Per-point features including reflectance, shape $(N, \text{in\_channels} - 3)$.
            pos: Per-point coordinates, shape $(N, 3)$.
            batch: Per-point scene index, shape $(N,)$.
            gt_boxes: Ground-truth boxes $(K, 7)$, required in train mode.
            gt_labels: Ground-truth $0$-based classes, shape $(K,)$, required in train mode.
            gt_batch: Per-box scene index, shape $(K,)$, required in train mode.

        Returns:
            A `PointRCNNTrainOutput` in train mode (stage-1 point predictions plus sampled-ROI stage-2
            tensors for the loss), otherwise a `PointRCNNOutput` (refined boxes with confidences).

        Shape:
            - x: $(N, \text{in\_channels} - 3)$, pos: $(N, 3)$, batch: $(N,)$
            - gt_boxes: $(K, 7)$, gt_labels / gt_batch: $(K,)$
        """
        x_point, pos_point, batch_point = self.forward_features(x, pos, batch)
        point_scores, cls_preds, box_preds, boxes = self.point_head(x_point, pos_point)

        if self.training:
            assert gt_boxes is not None and gt_labels is not None and gt_batch is not None, (
                "PointRCNN training needs ground-truth boxes; pass `box` / `label` / `batch_box` via `input_keys`."
            )
            rois, _, roi_labels, roi_batch = self._propose(
                boxes, cls_preds, batch_point, post_maxsize=self.train_nms_post_maxsize, thresh=self.train_nms_thresh
            )
            sampled = self._sample_proposals(rois, roi_labels, roi_batch, gt_boxes, gt_labels.long() + 1, gt_batch)
            rcnn_cls, rcnn_reg, refined = self.roi_head(
                pos_point, x_point, point_scores, batch_point, sampled["rois"], sampled["batch"]
            )
            return {
                "point_cls_preds": cls_preds,
                "point_box_preds": box_preds,
                "point_pos": pos_point,
                "point_batch": batch_point,
                "rcnn_cls": rcnn_cls,
                "rcnn_reg": rcnn_reg,
                "rcnn_boxes": refined,
                "rois": sampled["rois"],
                "gt_of_rois": sampled["gt_of_rois"],
                "gt_of_rois_src": sampled["gt_of_rois_src"],
                "roi_ious": sampled["roi_ious"],
            }

        rois, roi_scores, roi_labels, roi_batch = self._propose(boxes, cls_preds, batch_point)
        rcnn_cls, _, refined = self.roi_head(pos_point, x_point, point_scores, batch_point, rois, roi_batch)
        return {
            "rcnn_cls": rcnn_cls,
            "boxes": refined,
            "roi_labels": roi_labels,
            "roi_scores": roi_scores,
            "batch": roi_batch,
        }

    @torch.no_grad()
    def _propose(
        self,
        boxes: Tensor,
        cls_preds: Tensor,
        batch: Tensor,
        *,
        post_maxsize: Optional[int] = None,
        thresh: Optional[float] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        r"""Class-agnostic NMS over per-point proposals to a fixed number of ROIs per scene.

        Proposal generation carries no gradient: the ROI boxes are detached so the stage-2 loss cannot
        backprop into stage 1 through its own regression targets; stage 2 still receives gradient through
        the pooled backbone features and point scores.

        Args:
            boxes: Per-point decoded boxes, shape $(N, 7)$.
            cls_preds: Per-point class logits, shape $(N, \text{num\_classes})$.
            batch: Per-point scene index, shape $(N,)$.
            post_maxsize: ROIs to keep per scene (defaults to the inference `nms_post_maxsize`).
            thresh: NMS IoU threshold (defaults to the inference `nms_thresh`).

        Returns:
            A tuple `(rois, roi_scores, roi_labels, roi_batch)` of the kept boxes $(M, 7)$, their sigmoid
            scores $(M,)$, class labels $(M,)$, and scene indices $(M,)$.

        Shape:
            - boxes: $(N, 7)$, cls_preds: $(N, \text{num\_classes})$
            - output: $(M, 7)$, $(M,)$, $(M,)$, $(M,)$
        """
        post = self.nms_post_maxsize if post_maxsize is None else post_maxsize
        iou = self.nms_thresh if thresh is None else thresh
        scores = torch.sigmoid(cls_preds)
        roi_score, roi_label = scores.max(dim=1)

        batch_size = int(batch.max().item()) + 1 if batch.numel() else 0
        rois_list, score_list, label_list, batch_list = [], [], [], []
        for b in range(batch_size):
            mask = batch == b
            scene_boxes = boxes[mask]
            scene_scores = roi_score[mask]
            scene_labels = roi_label[mask]
            keep = self._nms_single(scene_boxes, scene_scores, post, iou)
            rois_list.append(scene_boxes[keep])
            score_list.append(scene_scores[keep])
            label_list.append(scene_labels[keep] + 1)
            batch_list.append(torch.full((keep.numel(),), b, dtype=torch.long, device=boxes.device))
        return (
            torch.cat(rois_list),
            torch.cat(score_list),
            torch.cat(label_list),
            torch.cat(batch_list),
        )

    def _nms_single(self, boxes: Tensor, scores: Tensor, post_maxsize: int, thresh: float) -> Tensor:
        r"""Class-agnostic BEV NMS keeping the top proposals (`class_agnostic_nms`).

        Args:
            boxes: Boxes $(N, 7)$.
            scores: Per-box score $(N,)$.
            post_maxsize: Maximum number of boxes to keep.
            thresh: NMS IoU threshold.

        Returns:
            Kept indices into `boxes`, shape $(K,)$ with $K \le$ `post_maxsize`.
        """
        if boxes.numel() == 0:
            return boxes.new_zeros((0,), dtype=torch.long)
        topk = min(self.nms_pre_maxsize, scores.shape[0])
        top_scores, top_idx = torch.topk(scores, k=topk)
        keep = nms3d(boxes[top_idx], top_scores, thresh)
        return top_idx[keep[:post_maxsize]]

    def _sample_proposals(
        self,
        rois: Tensor,
        roi_labels: Tensor,
        roi_batch: Tensor,
        gt_boxes: Tensor,
        gt_labels: Tensor,
        gt_batch: Tensor,
    ) -> Dict[str, Tensor]:
        r"""Sample `roi_per_image` ROIs per scene and match each to a ground-truth box (ProposalTargetLayer).

        Per scene the proposals are matched to same-class ground-truth boxes by 3D IoU
        ([`boxes_iou3d`][torch_pointcloud.utils.box3d.boxes_iou3d]), a foreground / background subset is
        sampled, and each sampled ROI's matched box is returned both in the lidar frame and canonically
        transformed into the ROI frame (translated to the ROI center, rotated by $-\theta$, heading
        wrapped to $[-\pi/2, \pi/2]$).

        Args:
            rois: Proposal boxes $(P, 7)$.
            roi_labels: Per-proposal $1$-based class, shape $(P,)$.
            roi_batch: Per-proposal scene index, shape $(P,)$.
            gt_boxes: Ground-truth boxes $(K, 7)$.
            gt_labels: Ground-truth $1$-based class, shape $(K,)$.
            gt_batch: Per-box scene index, shape $(K,)$.

        Returns:
            A dict of the sampled `rois` $(M, 7)$, per-ROI scene index `batch` $(M,)$, ROI-canonical matched
            box `gt_of_rois` $(M, 7)$, lidar-frame matched box `gt_of_rois_src` $(M, 7)$ and per-ROI max IoU
            `roi_ious` $(M,)$, with $M = \text{roi\_per\_image} \cdot B$.
        """
        batch_size = int(roi_batch.max().item()) + 1 if roi_batch.numel() else 0
        rois_list, batch_list, gt_ct_list, gt_src_list, iou_list = [], [], [], [], []
        for b in range(batch_size):
            roi_mask = roi_batch == b
            cur_rois = rois[roi_mask]
            cur_roi_labels = roi_labels[roi_mask]
            box_mask = gt_batch == b
            cur_gt = gt_boxes[box_mask]
            cur_gt_labels = gt_labels[box_mask]
            if cur_gt.shape[0] == 0:
                cur_gt = cur_gt.new_zeros((1, cur_gt.shape[1]))
                cur_gt_labels = cur_gt_labels.new_zeros(1)

            max_overlaps, gt_assignment = self._roi_gt_iou(cur_rois, cur_roi_labels, cur_gt, cur_gt_labels)
            sampled = self._subsample_rois(max_overlaps)

            sampled_rois = cur_rois[sampled]
            sampled_gt = cur_gt[gt_assignment[sampled]][:, 0:7]
            rois_list.append(sampled_rois)
            batch_list.append(torch.full((sampled.numel(),), b, dtype=torch.long, device=rois.device))
            gt_ct_list.append(self._canonical_gt(sampled_rois, sampled_gt))
            gt_src_list.append(sampled_gt)
            iou_list.append(max_overlaps[sampled])

        return {
            "rois": torch.cat(rois_list),
            "batch": torch.cat(batch_list),
            "gt_of_rois": torch.cat(gt_ct_list),
            "gt_of_rois_src": torch.cat(gt_src_list),
            "roi_ious": torch.cat(iou_list),
        }

    def _roi_gt_iou(
        self, rois: Tensor, roi_labels: Tensor, gt_boxes: Tensor, gt_labels: Tensor
    ) -> Tuple[Tensor, Tensor]:
        r"""Per-proposal max 3D IoU and matched-box index, restricted to same-class ROI / GT pairs."""
        max_overlaps = rois.new_zeros(rois.shape[0])
        gt_assignment = rois.new_zeros(rois.shape[0], dtype=torch.long)
        for cls in torch.unique(gt_labels).tolist():
            roi_mask = roi_labels == cls
            gt_mask = gt_labels == cls
            if roi_mask.any() and gt_mask.any():
                gt_idx = gt_mask.nonzero(as_tuple=False).squeeze(1)
                iou = boxes_iou3d(rois[roi_mask][:, 0:7], gt_boxes[gt_mask][:, 0:7])
                cur_max, cur_arg = iou.max(dim=1)
                max_overlaps[roi_mask] = cur_max
                gt_assignment[roi_mask] = gt_idx[cur_arg]
        return max_overlaps, gt_assignment

    def _subsample_rois(self, max_overlaps: Tensor) -> Tensor:
        r"""Sample foreground / hard-background / easy-background ROI indices to `roi_per_image` total.

        A scene with no proposals at all yields an empty index tensor rather than indices into an empty
        ROI set.
        """
        device = max_overlaps.device
        fg_rois_per_image = int(round(self.fg_ratio * self.roi_per_image))
        fg_thresh = min(self.reg_fg_thresh, self.cls_fg_thresh)

        fg_inds = (max_overlaps >= fg_thresh).nonzero(as_tuple=False).view(-1)
        easy_bg = (max_overlaps < self.cls_bg_thresh_lo).nonzero(as_tuple=False).view(-1)
        hard_bg = (
            ((max_overlaps < self.reg_fg_thresh) & (max_overlaps >= self.cls_bg_thresh_lo))
            .nonzero(as_tuple=False)
            .view(-1)
        )
        fg_num, bg_num = fg_inds.numel(), hard_bg.numel() + easy_bg.numel()

        if fg_num > 0 and bg_num > 0:
            fg_this = min(fg_rois_per_image, fg_num)
            fg_inds = fg_inds[torch.randperm(fg_num, device=device)[:fg_this]]
            bg_inds = self._sample_bg(hard_bg, easy_bg, self.roi_per_image - fg_this)
        elif fg_num > 0:
            fg_inds = fg_inds[torch.randint(0, fg_num, (self.roi_per_image,), device=device)]
            bg_inds = fg_inds.new_empty(0)
        elif bg_num > 0:
            fg_inds = max_overlaps.new_empty(0, dtype=torch.long)
            bg_inds = self._sample_bg(hard_bg, easy_bg, self.roi_per_image)
        else:
            return max_overlaps.new_zeros(0, dtype=torch.long)
        return torch.cat([fg_inds, bg_inds], dim=0)

    def _sample_bg(self, hard_bg: Tensor, easy_bg: Tensor, num: int) -> Tensor:
        r"""Draw `num` background indices, splitting between hard and easy background by `hard_bg_ratio`."""
        device = hard_bg.device
        if hard_bg.numel() > 0 and easy_bg.numel() > 0:
            hard_num = min(int(num * self.hard_bg_ratio), hard_bg.numel())
            hard = hard_bg[torch.randint(0, hard_bg.numel(), (hard_num,), device=device)]
            easy = easy_bg[torch.randint(0, easy_bg.numel(), (num - hard_num,), device=device)]
            return torch.cat([hard, easy])
        if hard_bg.numel() > 0:
            return hard_bg[torch.randint(0, hard_bg.numel(), (num,), device=device)]
        return easy_bg[torch.randint(0, easy_bg.numel(), (num,), device=device)]

    def _canonical_gt(self, rois: Tensor, gt_boxes: Tensor) -> Tensor:
        r"""Transform each matched ground-truth box into its ROI-canonical frame with a wrapped heading."""
        gt = gt_boxes.clone()
        roi_center = rois[:, 0:3]
        roi_ry = rois[:, 6] % (2 * math.pi)
        gt[:, 0:3] = gt[:, 0:3] - roi_center
        gt[:, 6] = gt[:, 6] - roi_ry
        gt = rotate_points_along_z(gt[:, None, :], -roi_ry).squeeze(1)

        heading = gt[:, 6] % (2 * math.pi)
        opposite = (heading > math.pi * 0.5) & (heading < math.pi * 1.5)
        heading[opposite] = (heading[opposite] + math.pi) % (2 * math.pi)
        flag = heading > math.pi
        heading[flag] = heading[flag] - 2 * math.pi
        gt[:, 6] = heading.clamp(min=-math.pi / 2, max=math.pi / 2)
        return gt

    @torch.no_grad()
    def decode(self, out: PointRCNNOutput) -> Detection3D:
        r"""Decode a forward output into raw per-ROI detections (no score threshold or NMS).

        Scores each refined box by its stage-2 confidence (sigmoid of `rcnn_cls`) and labels it by the
        stage-1 ROI label (shifted to 0-indexed). The full per-ROI set is returned; the evaluation
        pipeline applies class-agnostic 3D NMS then score thresholding via the
        `torch_pointcloud.utils.box3d` utilities (see the benchmark example).

        Args:
            out: A forward output `{"rcnn_cls", "boxes", "roi_labels", "roi_scores", "batch"}`.

        Returns:
            Packed per-ROI detections `{"boxes": (R, 7), "scores": (R,), "labels": (R,), "batch": (R,)}`
            (PyG layout).
        """
        return {
            "boxes": out["boxes"],
            "scores": torch.sigmoid(out["rcnn_cls"].view(-1)),
            "labels": out["roi_labels"] - 1,
            "batch": out["batch"],
        }


_KITTI_MEAN_SIZES = [[3.9, 1.6, 1.56], [0.8, 0.6, 1.73], [1.76, 0.6, 1.73]]


@register_model(
    "pointrcnn.kitti.openpcdet",
    task="detection",
    weights=WeightsDict(
        url="hf://torch-pointcloud/pointrcnn/pointrcnn.kitti.openpcdet.safetensors",
        dataset="kitti",
        classes=("Car", "Pedestrian", "Cyclist"),
        author="openpcdet",
        license="Apache-2.0",
    ),
    transform=T.Compose(
        [
            T.Cat(keys=[DataKeys.INTENSITY], dst_key=DataKeys.X, dim=1),
            T.CopyItems(keys=DataKeys.POS, names=DataKeys.ORIGIN_POS),
            T.BoxMask(keys=DataKeys.POS, bbox=(0.0, -40.0, -3.0, 70.4, 40.0, 1.0), dst_keys="range_mask"),
            T.ApplyMask(
                keys=[DataKeys.POS, DataKeys.X, DataKeys.INTENSITY, "range_mask"],
                mask_key="range_mask",
                dst_index_key=DataKeys.INDEX,
            ),
            T.RandomSample(
                keys=[DataKeys.POS, DataKeys.X, DataKeys.INTENSITY, "range_mask"],
                num_samples=16384,
                replace=False,
                dst_index_key=DataKeys.INDEX,
            ),
        ]
    ),
    hparams=dict(
        in_channels=4,
        num_classes=3,
        mean_sizes=_KITTI_MEAN_SIZES,
        sa_channels=[
            [[16, 16, 32], [32, 32, 64]],
            [[64, 64, 128], [64, 96, 128]],
            [[128, 196, 256], [128, 196, 256]],
            [[256, 256, 512], [256, 384, 512]],
        ],
        sa_npoints=[4096, 1024, 256, 64],
        sa_radii=[[0.1, 0.5], [0.5, 1.0], [1.0, 2.0], [2.0, 4.0]],
        sa_num_neighbors=[[16, 32], [16, 32], [16, 32], [16, 32]],
        fp_channels=[[512, 512], [512, 512], [256, 256], [128, 128]],
        roi_sa_channels=[[128, 128, 128], [128, 128, 256], [256, 256, 512]],
        roi_sa_npoints=[128, 32, -1],
        roi_sa_radii=[0.2, 0.4, 100.0],
        roi_sa_num_neighbors=[16, 16, 16],
        roi_xyz_up_channels=[128, 128],
        num_sampled_points=512,
        pool_extra_width=[0.0, 0.0, 0.0],
        depth_normalizer=70.0,
        nms_post_maxsize=100,
        nms_thresh=0.85,
    ),
)
def pointrcnn_openpcdet_kitti(**hparams: Any) -> PointRCNNDetection:
    return PointRCNNDetection(**hparams)
