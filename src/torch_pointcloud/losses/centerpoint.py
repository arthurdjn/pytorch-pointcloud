r"""Losses of the center-based detection heads, dense (`CenterHead`) and fully sparse (`VoxelNeXtHead`).

Both supervise the heatmap with [`gaussian_focal_loss`][torch_pointcloud.losses.focal.gaussian_focal_loss] and the
gathered regression codes with a masked, code-weighted $L_1$; the dense targets come from
[`draw_heatmap_targets`][torch_pointcloud.ops.heatmap.draw_heatmap_targets].
"""

from typing import TYPE_CHECKING, Any, Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from torch_pointcloud.losses._utils import unbatch_boxes
from torch_pointcloud.losses.focal import gaussian_focal_loss
from torch_pointcloud.ops.box3d import boxes_iou3d
from torch_pointcloud.ops.heatmap import draw_heatmap_targets, gaussian_radius, transpose_gather

if TYPE_CHECKING:
    from torch_pointcloud.models.voxel_mamba import CenterHeadOutput
    from torch_pointcloud.models.voxelnext import VoxelNeXtHeadOutput


def _reg_l1_loss(pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    r"""Masked per-code $L_1$ regression, summed over objects and batch, normalized by the object count.

    Args:
        pred: Gathered predictions, shape $(B, M, \text{code})$.
        target: Regression targets, shape $(B, M, \text{code})$.
        mask: Object validity, shape $(B, M)$.

    Returns:
        Per-code loss, shape $(\text{code},)$.
    """
    num = mask.float().sum()
    expanded = mask.unsqueeze(2).expand_as(target).float() * torch.isfinite(target).float()
    pred = pred * expanded
    target = torch.nan_to_num(target) * expanded
    loss = (pred - target).abs().transpose(2, 0)
    loss = loss.sum(dim=2).sum(dim=1)
    return loss / torch.clamp_min(num, min=1.0)


def _pad_box_columns(boxes: Tensor, num_extra: int) -> Tensor:
    r"""Keep the seven geometry columns plus `num_extra` trailing ones (e.g. velocity), zero-padding if absent."""
    geom = boxes[:, :7]
    if num_extra == 0:
        return geom

    if boxes.shape[1] >= 7 + num_extra:
        extra = boxes[:, 7 : 7 + num_extra]
    else:
        extra = boxes.new_zeros(boxes.shape[0], num_extra)
    return torch.cat([geom, extra], dim=1)


class CenterHeadLoss(nn.Module):
    r"""Loss of the dense `CenterHead` (CenterPoint, Voxel Mamba).

    Reference: :arxiv: [Center-based 3D Object Detection and Tracking](https://arxiv.org/abs/2006.11275) (Yin et al., 2021).

    Ground-truth boxes are splatted onto a per-class BEV Gaussian heatmap and their regression code
    (sub-cell center offset, $z$, log extents and $(\cos\theta, \sin\theta)$) is recorded at each peak
    cell. The heatmap is supervised by the penalty-reduced center focal loss and the regression maps by
    a masked, code-weighted $L_1$ read back at those cells. When the head emits an `iou` map an optional
    $L_1$ term regresses it toward the 3D IoU (rescaled to $[-1, 1]$) between the decoded prediction and
    its matched box.

    Args:
        num_classes: Number of heatmap channels.
        point_cloud_range: Range $(x_\min, y_\min, z_\min, x_\max, y_\max, z_\max)$.
        voxel_size: Voxel size $(v_x, v_y, v_z)$.
        feature_map_stride: Stride from the voxel grid to the BEV feature map.
        code_weights: Per-code regression weight, length $8$ (the head predicts no velocity codes).
        cls_weight: Multiplier on the heatmap focal loss.
        loc_weight: Multiplier on the summed regression loss.
        iou_weight: Multiplier on the optional IoU-branch loss ($0$ disables it).
        gaussian_overlap: Min-overlap passed to the Gaussian-radius solver.
        min_radius: Lower clamp on the integer splat radius.
        num_max_objs: Per-scene object-target capacity.
    """

    code_weights: Tensor

    def __init__(
        self,
        num_classes: int,
        *,
        point_cloud_range: Sequence[float],
        voxel_size: Sequence[float],
        feature_map_stride: int,
        code_weights: Sequence[float],
        cls_weight: float = 1.0,
        loc_weight: float = 0.25,
        iou_weight: float = 0.0,
        gaussian_overlap: float = 0.1,
        min_radius: int = 2,
        num_max_objs: int = 500,
    ) -> None:
        super().__init__()
        if len(code_weights) != 8:
            raise ValueError(f"`code_weights` must have length 8, got {len(code_weights)}.")

        self.num_classes = num_classes
        self.point_cloud_range = tuple(point_cloud_range)
        self.voxel_size = tuple(voxel_size)
        self.feature_map_stride = feature_map_stride
        self.cls_weight = cls_weight
        self.loc_weight = loc_weight
        self.iou_weight = iou_weight
        self.gaussian_overlap = gaussian_overlap
        self.min_radius = min_radius
        self.num_max_objs = num_max_objs

        weights = torch.as_tensor(code_weights, dtype=torch.float32)
        self.register_buffer("code_weights", weights)
        self.code_size = int(weights.numel())

    def forward(self, output: "CenterHeadOutput", data: Dict[str, Any]) -> Dict[str, Tensor]:
        r"""Compute the dense center loss and its terms.

        Args:
            output: Head maps `heatmap` $(B, C, H, W)$, `center` $(B, 2, H, W)$, `center_z` $(B, 1, H, W)$,
                `dim` $(B, 3, H, W)$, `rot` $(B, 2, H, W)$ and optionally `iou` $(B, 1, H, W)$.
            data: Packed GT (`DataKeys.BOX`, `DataKeys.LABEL`, `DataKeys.BATCH_BOX`).

        Returns:
            A dict with the scalar `loss` and detached `heatmap_loss`, `box_loss` (and `iou_loss` when enabled).
        """
        heatmap_pred = output["heatmap"]
        batch_size, _, height, width = heatmap_pred.shape
        boxes_per_scene, labels_per_scene = unbatch_boxes(data, batch_size, heatmap_pred.device)
        boxes_per_scene = [_pad_box_columns(boxes, self.code_size - 8) for boxes in boxes_per_scene]

        hm_targets: List[Tensor] = []
        reg_targets: List[Tensor] = []
        inds: List[Tensor] = []
        masks: List[Tensor] = []
        for b in range(batch_size):
            hm, reg, ind, mask = draw_heatmap_targets(
                boxes_per_scene[b],
                labels_per_scene[b],
                self.num_classes,
                (width, height),
                self.voxel_size,
                self.point_cloud_range,
                self.feature_map_stride,
                num_max_objs=self.num_max_objs,
                gaussian_overlap=self.gaussian_overlap,
                min_radius=self.min_radius,
            )
            hm_targets.append(hm)
            reg_targets.append(reg)
            inds.append(ind)
            masks.append(mask)

        hm_target = torch.stack(hm_targets)
        reg_target = torch.stack(reg_targets)
        ind = torch.stack(inds)
        mask = torch.stack(masks)

        heatmap_loss = gaussian_focal_loss(heatmap_pred.sigmoid(), hm_target) * self.cls_weight

        pred_boxes = torch.cat([output["center"], output["center_z"], output["dim"], output["rot"]], dim=1)
        reg = _reg_l1_loss(transpose_gather(pred_boxes, ind), reg_target, mask)
        box_loss = (reg * self.code_weights).sum() * self.loc_weight

        total = heatmap_loss + box_loss
        result = {"loss": total, "heatmap_loss": heatmap_loss.detach(), "box_loss": box_loss.detach()}

        if self.iou_weight > 0 and "iou" in output:
            iou_loss = self._iou_loss(output["iou"], pred_boxes, ind, mask, boxes_per_scene, width) * self.iou_weight
            result["loss"] = total + iou_loss
            result["iou_loss"] = iou_loss.detach()
        return result

    def _iou_loss(
        self,
        iou_pred: Tensor,
        pred_boxes: Tensor,
        ind: Tensor,
        mask: Tensor,
        boxes_per_scene: List[Tensor],
        width: int,
    ) -> Tensor:
        r"""IoU-branch $L_1$: regress the `iou` map toward $2 \cdot \text{IoU}_{3D} - 1$ at each peak cell."""
        gathered_iou = transpose_gather(iou_pred, ind)
        gathered_box = transpose_gather(pred_boxes, ind)
        vx, vy, _ = self.voxel_size
        pxs = ind % width
        pys = torch.div(ind, width, rounding_mode="floor")

        total = iou_pred.new_zeros(())
        count = iou_pred.new_zeros(())
        for b in range(len(boxes_per_scene)):
            keep = mask[b].bool()
            if keep.sum() == 0:
                continue

            box = gathered_box[b][keep]
            xs = (pxs[b][keep].float() + box[:, 0]) * self.feature_map_stride * vx + self.point_cloud_range[0]
            ys = (pys[b][keep].float() + box[:, 1]) * self.feature_map_stride * vy + self.point_cloud_range[1]
            angle = torch.atan2(box[:, 7], box[:, 6])
            decoded = torch.stack([xs, ys, box[:, 2], *box[:, 3:6].exp().unbind(-1), angle], dim=-1)
            # Target slot k maps to GT row k, but skipped (degenerate / out-of-range) boxes leave holes in
            # the mask, so the kept predictions must be paired with the kept GT rows, not the first rows.
            gt_idx = keep.nonzero(as_tuple=False).squeeze(1)
            gt = boxes_per_scene[b][gt_idx][:, :7]
            iou_target = boxes_iou3d(decoded, gt, aligned=True) * 2 - 1
            total = total + F.l1_loss(gathered_iou[b][keep].view(-1), iou_target, reduction="sum")
            count = count + keep.sum()
        return total / torch.clamp_min(count, min=1.0)


def _voxel_gaussians(distances: Tensor, radius: Tensor) -> Tensor:
    r"""Gaussian in squared voxel distance, one row per object: $\exp(-d^2 / 2\sigma^2)$ with $\sigma = (2r + 1) / 6$."""
    diameter = 2 * radius.to(torch.float64) + 1
    sigma = diameter / 6
    denominator = (2 * sigma * sigma).to(distances.dtype)
    return torch.exp(-distances / denominator[:, None])


def _assign_sparse_scene(
    boxes: Tensor,
    labels: Tensor,
    num_classes: int,
    spatial_xy: Tensor,
    feature_map_size: Tuple[int, int],
    voxel_size: Sequence[float],
    point_cloud_range: Sequence[float],
    feature_map_stride: int,
    *,
    num_max_objs: int,
    gaussian_overlap: float,
    min_radius: int,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    r"""Draw sparse heatmap and regression targets over one scene's occupied voxels.

    Each box center is projected to BEV cells; the target Gaussian is splatted only onto occupied
    voxels (by squared distance to both the box center and the nearest occupied voxel). The regression
    code is anchored to that nearest occupied voxel: the center offset is measured against its real
    (non-integer) index rather than a dense-grid floor. Extents are clamped to $10^{-5}$ before the log
    so a degenerate box does not produce a non-finite target.

    Args:
        boxes: Scene boxes $(K, D)$, $D \ge 7$, without a class column.
        labels: Zero-based group-local class ids, shape $(K,)$.
        num_classes: Number of classes in the group (heatmap channels).
        spatial_xy: Occupied-voxel BEV indices as $(x, y)$ floats, shape $(V, 2)$.
        feature_map_size: BEV size $(W, H)$.
        voxel_size: Voxel size $(v_x, v_y, v_z)$.
        point_cloud_range: Range $(x_\min, y_\min, z_\min, x_\max, y_\max, z_\max)$.
        feature_map_stride: Stride from the voxel grid to the BEV feature map.
        num_max_objs: Per-scene object-target capacity.
        gaussian_overlap: Min-overlap passed to the Gaussian-radius solver.
        min_radius: Lower clamp on the integer splat radius.

    Returns:
        `(heatmap, reg_targets, inds, mask)` with `heatmap` $(\text{num\_classes}, V)$, `reg_targets`
        $(\text{num\_max\_objs}, D + 1)$, `inds`/`mask` $(\text{num\_max\_objs},)$ (long).
    """
    width, height = feature_map_size
    num_voxels = spatial_xy.shape[0]
    code_size = boxes.shape[-1] + 1
    heatmap = boxes.new_zeros(num_classes, num_voxels)
    reg_targets = boxes.new_zeros(num_max_objs, code_size)
    inds = boxes.new_zeros(num_max_objs, dtype=torch.long)
    mask = boxes.new_zeros(num_max_objs, dtype=torch.long)

    boxes, labels = boxes[:num_max_objs], labels[:num_max_objs]
    num_objs = boxes.shape[0]
    if num_objs == 0 or num_voxels == 0:
        return heatmap, reg_targets, inds, mask

    # Project the centers to the feature map and solve the splat radius of every box at once.
    x, y, z = boxes[:, 0], boxes[:, 1], boxes[:, 2]
    pos_x = torch.clamp((x - point_cloud_range[0]) / voxel_size[0] / feature_map_stride, min=0, max=width - 0.5)
    pos_y = torch.clamp((y - point_cloud_range[1]) / voxel_size[1] / feature_map_stride, min=0, max=height - 0.5)
    center = torch.stack([pos_x, pos_y], dim=-1)

    dx = boxes[:, 3] / voxel_size[0] / feature_map_stride
    dy = boxes[:, 4] / voxel_size[1] / feature_map_stride
    radius = torch.clamp_min(gaussian_radius(dy, dx, min_overlap=gaussian_overlap).int(), min_radius)
    valid = (dx > 0) & (dy > 0)

    # Every object's nearest occupied voxel, from the squared distances of all (object, voxel) pairs.
    dist_center = ((spatial_xy[None, :, :] - center[:, None, :]) ** 2).sum(dim=-1)  # (K, V)
    nearest = dist_center.argmin(dim=1)
    nearest_xy = spatial_xy[nearest]
    mask[:num_objs] = valid.long()
    inds[:num_objs] = torch.where(valid, nearest, inds[:num_objs])

    # The two splats of each object (around its center and around its nearest voxel), max-combined per class.
    dist_nearest = ((spatial_xy[None, :, :] - nearest_xy[:, None, :]) ** 2).sum(dim=-1)
    rows = labels[valid, None].expand(-1, num_voxels)
    heatmap.scatter_reduce_(0, rows, _voxel_gaussians(dist_center[valid], radius[valid]), reduce="amax")
    heatmap.scatter_reduce_(0, rows, _voxel_gaussians(dist_nearest[valid], radius[valid]), reduce="amax")

    # The regression code, anchored to the nearest occupied voxel.
    code = torch.cat(
        [
            center - nearest_xy,
            z[:, None],
            boxes[:, 3:6].clamp_min(1e-5).log(),
            torch.cos(boxes[:, 6:7]),
            torch.sin(boxes[:, 6:7]),
            boxes[:, 7:],
        ],
        dim=1,
    )
    reg_targets[:num_objs] = torch.where(valid[:, None], code, reg_targets[:num_objs])

    return heatmap, reg_targets, inds, mask


class VoxelNeXtHeadLoss(nn.Module):
    r"""Loss of the fully sparse `VoxelNeXtHead`.

    Reference: :arxiv: [VoxelNeXt: Fully Sparse VoxelNet for 3D Object Detection and Tracking](https://arxiv.org/abs/2303.11301) (Chen et al., 2023).

    The head predicts CenterPoint-style attributes directly on the occupied BEV voxels rather than a
    dense map, so targets are drawn only at those voxels: the per-class heatmap is a Gaussian in squared
    voxel distance and each object's regression code is anchored to its nearest occupied voxel. The
    heatmap is supervised by the penalty-reduced center focal loss and the gathered regression rows by a
    masked, code-weighted $L_1$. Classes are split into groups, one sparse head each.

    Args:
        class_groups: Zero-based global class-index groups, one per head (e.g. `[[0], [1, 2], ...]`).
        point_cloud_range: Range $(x_\min, y_\min, z_\min, x_\max, y_\max, z_\max)$.
        voxel_size: Voxel size $(v_x, v_y, v_z)$.
        feature_map_stride: Stride from the voxel grid to the BEV feature map.
        code_weights: Per-code regression weight, length $8 + \text{extra}$ (e.g. $10$ with velocity).
        cls_weight: Multiplier on the heatmap focal loss.
        loc_weight: Multiplier on the summed regression loss.
        gaussian_overlap: Min-overlap passed to the Gaussian-radius solver.
        min_radius: Lower clamp on the integer splat radius.
        num_max_objs: Per-scene object-target capacity.
    """

    code_weights: Tensor

    def __init__(
        self,
        class_groups: Sequence[Sequence[int]],
        *,
        point_cloud_range: Sequence[float],
        voxel_size: Sequence[float],
        feature_map_stride: int,
        code_weights: Sequence[float],
        cls_weight: float = 1.0,
        loc_weight: float = 0.25,
        gaussian_overlap: float = 0.1,
        min_radius: int = 2,
        num_max_objs: int = 500,
    ) -> None:
        super().__init__()
        self.class_groups = [list(group) for group in class_groups]
        self.point_cloud_range = tuple(point_cloud_range)
        self.voxel_size = tuple(voxel_size)
        self.feature_map_stride = feature_map_stride
        self.cls_weight = cls_weight
        self.loc_weight = loc_weight
        self.gaussian_overlap = gaussian_overlap
        self.min_radius = min_radius
        self.num_max_objs = num_max_objs

        weights = torch.as_tensor(code_weights, dtype=torch.float32)
        self.register_buffer("code_weights", weights)
        self.code_size = int(weights.numel())

        pcr, vs = self.point_cloud_range, self.voxel_size
        self.feature_map_size = (
            int(round((pcr[3] - pcr[0]) / vs[0] / feature_map_stride)),
            int(round((pcr[4] - pcr[1]) / vs[1] / feature_map_stride)),
        )

    def forward(self, output: "VoxelNeXtHeadOutput", data: Dict[str, Any]) -> Dict[str, Tensor]:
        r"""Compute the sparse center loss summed over class groups.

        Args:
            output: A `VoxelNeXtHeadOutput`: per-group lists `hm` $(V, n_g)$, `center` $(V, 2)$,
                `center_z` $(V, 1)$, `dim` $(V, 3)$, `rot` $(V, 2)$, `vel` $(V, 2)$ and shared
                `voxel_indices` $(V, 3)$ with columns $(\text{batch}, y, x)$.
            data: Packed GT (`DataKeys.BOX`, `DataKeys.LABEL`, `DataKeys.BATCH_BOX`).

        Returns:
            A dict with the scalar `loss` and detached `heatmap_loss`, `box_loss`.
        """
        voxel_indices = output["voxel_indices"]
        batch_index = voxel_indices[:, 0]
        batch_size = int(batch_index.max().item()) + 1 if voxel_indices.numel() else 0
        spatial_xy = voxel_indices[:, [2, 1]].float()

        boxes_per_scene, labels_per_scene = unbatch_boxes(data, batch_size, voxel_indices.device)
        boxes_per_scene = [_pad_box_columns(boxes, self.code_size - 8) for boxes in boxes_per_scene]

        total = voxel_indices.new_zeros((), dtype=torch.float32)
        hm_total = voxel_indices.new_zeros((), dtype=torch.float32)
        box_total = voxel_indices.new_zeros((), dtype=torch.float32)
        for group_idx, group in enumerate(self.class_groups):
            hm_pred = output["hm"][group_idx]
            pred_boxes = torch.cat(
                [
                    output["center"][group_idx],
                    output["center_z"][group_idx],
                    output["dim"][group_idx],
                    output["rot"][group_idx],
                    output["vel"][group_idx],
                ],
                dim=1,
            )

            hm_target = torch.zeros_like(hm_pred)
            reg_targets: List[Tensor] = []
            inds: List[Tensor] = []
            masks: List[Tensor] = []
            for b in range(batch_size):
                voxel_mask = batch_index == b
                group_boxes, group_labels = self._select_group(boxes_per_scene[b], labels_per_scene[b], group)
                hm, reg, ind, mask = _assign_sparse_scene(
                    group_boxes,
                    group_labels,
                    len(group),
                    spatial_xy[voxel_mask],
                    self.feature_map_size,
                    self.voxel_size,
                    self.point_cloud_range,
                    self.feature_map_stride,
                    num_max_objs=self.num_max_objs,
                    gaussian_overlap=self.gaussian_overlap,
                    min_radius=self.min_radius,
                )
                hm_target[voxel_mask] = hm.permute(1, 0)
                reg_targets.append(reg)
                inds.append(ind)
                masks.append(mask)

            hm_loss = gaussian_focal_loss(hm_pred.sigmoid(), hm_target) * self.cls_weight

            ind = torch.stack(inds)
            mask = torch.stack(masks)
            reg_target = torch.stack(reg_targets)
            empty = pred_boxes.new_zeros(ind.shape[1], pred_boxes.shape[1])
            rows: List[Tensor] = []
            for b in range(batch_size):
                scene_pred = pred_boxes[batch_index == b]
                rows.append(scene_pred[ind[b]] if scene_pred.numel() else empty)

            gathered = torch.stack(rows)
            reg = _reg_l1_loss(gathered, reg_target, mask)
            box_loss = (reg * self.code_weights).sum() * self.loc_weight

            total = total + hm_loss + box_loss
            hm_total = hm_total + hm_loss.detach()
            box_total = box_total + box_loss.detach()

        return {"loss": total, "heatmap_loss": hm_total, "box_loss": box_total}

    @staticmethod
    def _select_group(boxes: Tensor, labels: Tensor, group: List[int]) -> Tuple[Tensor, Tensor]:
        r"""Keep the boxes whose global class is in `group` and remap their labels to the group-local index."""
        remap = labels.new_full((max(group) + 1,), -1)
        for local, global_cls in enumerate(group):
            remap[global_cls] = local

        keep = torch.isin(labels, labels.new_tensor(group))
        return boxes[keep], remap[labels[keep]]
