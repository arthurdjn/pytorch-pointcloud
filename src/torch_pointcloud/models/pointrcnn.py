from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import MLP

import torch_pointcloud.transforms as T
from torch_pointcloud.layers.pointnet2_blocks import GlobalSAModule, SAModule
from torch_pointcloud.utils.box3d import decode_box_residuals, nms3d
from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.types import Detection3D, OptTensor

from ._base import DetectionModel
from ._registry import register_model
from .pointnet2 import PointNet2Decoder, PointNet2Encoder


def rotate_points_along_z(points: Tensor, angle: Tensor) -> Tensor:
    r"""Rotate point sets about the $+z$ axis (angle increases $x \to y$), matching OpenPCDet.

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
    r"""Decode per-point box residuals with class mean-size anchors (OpenPCDet's `PointResidualCoder`).

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

    def forward(self, x: Tensor, pos: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        r"""Predict per-point class scores and decoded proposal boxes.

        Args:
            x: Per-point backbone features, shape $(N, C)$.
            pos: Per-point coordinates, shape $(N, 3)$.

        Returns:
            A tuple `(point_scores, cls_preds, boxes)` of the sigmoid foreground score $(N,)$, the raw
            class logits $(N, \text{num\_classes})$, and the decoded boxes $(N, 7)$.

        Shape:
            - x: $(N, C)$
            - pos: $(N, 3)$
            - output: $(N,)$, $(N, \text{num\_classes})$, $(N, 7)$
        """
        cls_preds = self.cls_layers(x)
        box_preds = self.box_layers(x)
        point_scores = torch.sigmoid(cls_preds.max(dim=-1).values)
        pred_classes = cls_preds.argmax(dim=-1) + 1
        boxes = decode_point_residuals(box_preds, pos, pred_classes, self.mean_sizes)
        return point_scores, cls_preds, boxes


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

    def roipool(self, pos: Tensor, features: Tensor, rois: Tensor) -> Tuple[Tensor, Tensor]:
        r"""Pool a fixed number of in-box points per ROI and canonically transform them.

        Mirrors OpenPCDet's `roipoint_pool3d` CUDA kernel: for every (optionally enlarged) box, the input
        points are scanned in order and the first `num_sampled_points` that fall inside are kept (cyclically
        duplicated when fewer are found, zeroed when none). Pooled points are translated to the ROI center
        and rotated by $-\theta$ into the box-canonical frame.

        Args:
            pos: Per-point coordinates of one scene, shape $(N, 3)$.
            features: Per-point pooled features (score + depth + backbone), shape $(N, 5 + C)$.
            rois: Proposal boxes $(x, y, z, d_x, d_y, d_z, \theta)$, shape $(M, 7)$.

        Returns:
            A tuple `(pooled, empty)` of the pooled features $(M, S, 3 + (5 + C))$ in canonical xyz and a
            boolean ROI-empty flag $(M,)$.

        Shape:
            - pos: $(N, 3)$
            - features: $(N, 5 + C)$
            - rois: $(M, 7)$
            - output: $(M, S, 3 + 5 + C)$, $(M,)$
        """
        m = rois.shape[0]
        n = pos.shape[0]
        s = self.num_sampled_points
        channels = features.shape[1]

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
        pooled_feat = features[sampled_idx]
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
        features: Tensor,
        point_scores: Tensor,
        batch: Tensor,
        rois: Tensor,
        roi_batch: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        r"""Refine proposals into confidence logits and refined boxes.

        Args:
            pos: Per-point coordinates, shape $(N, 3)$.
            features: Per-point backbone features, shape $(N, C)$.
            point_scores: Per-point stage-1 foreground score, shape $(N,)$.
            batch: Per-point scene index, shape $(N,)$.
            rois: Proposal boxes, shape $(M, 7)$.
            roi_batch: Per-ROI scene index, shape $(M,)$.

        Returns:
            A tuple `(rcnn_cls, refined_boxes)` of the confidence logit $(M, 1)$ and the refined boxes
            $(M, 7)$ in the lidar frame.

        Shape:
            - pos: $(N, 3)$, features: $(N, C)$, point_scores: $(N,)$
            - rois: $(M, 7)$
            - output: $(M, 1)$, $(M, 7)$
        """
        depth = pos.norm(dim=1) / self.depth_normalizer - 0.5
        feat_all = torch.cat([point_scores[:, None], depth[:, None], features], dim=1)

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
        return rcnn_cls, refined

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
        nms_post_maxsize: ROIs kept after stage-1 NMS (the stage-2 batch size per scene).
        nms_thresh: Stage-1 proposal NMS IoU threshold.
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
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(in_channels=in_channels, num_classes=num_classes)
        self.nms_pre_maxsize = nms_pre_maxsize
        self.nms_post_maxsize = nms_post_maxsize
        self.nms_thresh = nms_thresh

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

    def forward(self, x: OptTensor, pos: Tensor, batch: Tensor) -> Dict[str, Tensor]:
        x_point, pos_point, batch_point = self.forward_features(x, pos, batch)
        point_scores, cls_preds, boxes = self.point_head(x_point, pos_point)
        rois, roi_scores, roi_labels, roi_batch = self._propose(boxes, cls_preds, batch_point)
        rcnn_cls, refined = self.roi_head(pos_point, x_point, point_scores, batch_point, rois, roi_batch)
        return {
            "rcnn_cls": rcnn_cls,
            "boxes": refined,
            "roi_labels": roi_labels,
            "roi_scores": roi_scores,
            "batch": roi_batch,
        }

    def _propose(self, boxes: Tensor, cls_preds: Tensor, batch: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        r"""Class-agnostic NMS over per-point proposals to a fixed number of ROIs per scene.

        Args:
            boxes: Per-point decoded boxes, shape $(N, 7)$.
            cls_preds: Per-point class logits, shape $(N, \text{num\_classes})$.
            batch: Per-point scene index, shape $(N,)$.

        Returns:
            A tuple `(rois, roi_scores, roi_labels, roi_batch)` of the kept boxes $(M, 7)$, their sigmoid
            scores $(M,)$, class labels $(M,)$, and scene indices $(M,)$.

        Shape:
            - boxes: $(N, 7)$, cls_preds: $(N, \text{num\_classes})$
            - output: $(M, 7)$, $(M,)$, $(M,)$, $(M,)$
        """
        scores = torch.sigmoid(cls_preds)
        roi_score, roi_label = scores.max(dim=1)

        batch_size = int(batch.max().item()) + 1 if batch.numel() else 0
        rois_list, score_list, label_list, batch_list = [], [], [], []
        for b in range(batch_size):
            mask = batch == b
            scene_boxes = boxes[mask]
            scene_scores = roi_score[mask]
            scene_labels = roi_label[mask]
            keep = self._nms_single(scene_boxes, scene_scores)
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

    def _nms_single(self, boxes: Tensor, scores: Tensor) -> Tensor:
        r"""Class-agnostic BEV NMS keeping the top proposals (`class_agnostic_nms`).

        Args:
            boxes: Boxes $(N, 7)$.
            scores: Per-box score $(N,)$.

        Returns:
            Kept indices into `boxes`, shape $(K,)$ with $K \le$ `nms_post_maxsize`.
        """
        if boxes.numel() == 0:
            return boxes.new_zeros((0,), dtype=torch.long)
        topk = min(self.nms_pre_maxsize, scores.shape[0])
        top_scores, top_idx = torch.topk(scores, k=topk)
        keep = nms3d(boxes[top_idx], top_scores, self.nms_thresh)
        return top_idx[keep[: self.nms_post_maxsize]]

    @torch.no_grad()
    def decode(self, out: Dict[str, Tensor]) -> Detection3D:
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
    "pointrcnn-openpcdet.kitti",
    task="detection",
    weights="hf://torch-pointcloud/pointrcnn/pointrcnn-openpcdet.kitti.pt",
    transforms=T.Compose(
        [
            T.Cat(keys=[DataKeys.INTENSITY], dst_key=DataKeys.X, dim=1),
            T.BoxMask(keys=DataKeys.POS, bbox=(0.0, -40.0, -3.0, 70.4, 40.0, 1.0), dst_keys="range_mask"),
            T.ApplyMask(keys=[DataKeys.POS, DataKeys.X], mask_key="range_mask"),
            T.RandomSample(keys=[DataKeys.POS, DataKeys.X], num_samples=16384, replace=False),
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
