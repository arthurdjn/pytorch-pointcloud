r"""Center-based 3D detection losses: dense (CenterHead) and fully sparse (VoxelNeXt) heatmap objectives."""

from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from torch_pointcloud.utils.box3d import boxes_iou3d
from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.heatmap import draw_heatmap_targets, gaussian_radius

_EPS = 1e-4


def _sigmoid_clamp(x: Tensor) -> Tensor:
    r"""Sigmoid clamped to $[\varepsilon, 1 - \varepsilon]$ so the focal $\log$ terms stay finite."""
    return torch.clamp(x.sigmoid(), min=_EPS, max=1 - _EPS)


def _gaussian_focal_loss(pred: Tensor, target: Tensor) -> Tensor:
    r"""Penalty-reduced center focal loss over a (dense or sparse) heatmap.

    The positive term is the standard $\log(p)(1 - p)^2$ at cells whose Gaussian target is exactly $1$;
    every other cell is a soft negative down-weighted by $(1 - y)^4$ so cells near a peak barely
    contribute. Normalized by the number of positive cells.

    Args:
        pred: Predicted probabilities (post-sigmoid) of any shape.
        target: Gaussian heatmap target of the same shape, values in $[0, 1]$.

    Returns:
        Scalar focal loss.
    """
    pos_inds = target.eq(1).float()
    neg_inds = target.lt(1).float()
    neg_weights = (1 - target).pow(4)

    pos_loss = (torch.log(pred) * (1 - pred).pow(2) * pos_inds).sum()
    neg_loss = (torch.log(1 - pred) * pred.pow(2) * neg_weights * neg_inds).sum()

    num_pos = pos_inds.sum()
    if num_pos == 0:
        return -neg_loss
    return -(pos_loss + neg_loss) / num_pos


def _transpose_gather(feat: Tensor, ind: Tensor) -> Tensor:
    r"""Gather per-object rows from a dense $(B, C, H, W)$ map at flat cell indices `ind` $(B, M)$."""
    b, c = feat.shape[0], feat.shape[1]
    feat = feat.permute(0, 2, 3, 1).reshape(b, -1, c)
    return feat.gather(1, ind.unsqueeze(2).expand(-1, -1, c))


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
    expanded = mask.unsqueeze(2).expand_as(target).float() * (~torch.isnan(target)).float()
    pred = pred * expanded
    target = target * expanded
    loss = (pred - target).abs().transpose(2, 0)
    loss = loss.sum(dim=2).sum(dim=1)
    return loss / torch.clamp_min(num, min=1.0)


def _densify_gt(
    batch: Dict[str, Any], batch_size: int, num_extra: int, device: torch.device
) -> Tuple[List[Tensor], List[Tensor]]:
    r"""Split the packed GT boxes / labels into per-scene lists with a fixed $7 + \text{num\_extra}$ box width.

    The packed batch carries `DataKeys.BOX` $(K, D)$, one-based `DataKeys.LABEL` $(K,)$ and the per-box
    scene index `DataKeys.BATCH_BOX` $(K,)$. Each scene's boxes are sliced (or zero-padded) to seven
    geometry columns plus `num_extra` trailing columns (e.g. velocity); labels are shifted to zero-based.

    Args:
        batch: Packed ground-truth dict.
        batch_size: Number of scenes $B$.
        num_extra: Trailing box columns beyond $(c_x, c_y, c_z, d_x, d_y, d_z, \theta)$.
        device: Device the empty fallbacks are created on.

    Returns:
        `(boxes_per_scene, labels_per_scene)`, each a length-$B$ list.
    """
    box: Tensor = batch[DataKeys.BOX]
    label: Tensor = batch[DataKeys.LABEL].long()
    box_batch: Tensor = batch[DataKeys.BATCH_BOX]

    boxes_per_scene: List[Tensor] = []
    labels_per_scene: List[Tensor] = []
    for b in range(batch_size):
        mask = box_batch == b
        scene = box[mask]
        geom = scene[:, :7]
        if num_extra > 0:
            if scene.shape[1] >= 7 + num_extra:
                extra = scene[:, 7 : 7 + num_extra]
            else:
                extra = scene.new_zeros(scene.shape[0], num_extra)
            geom = torch.cat([geom, extra], dim=1)
        boxes_per_scene.append(geom.to(device))
        labels_per_scene.append(label[mask].to(device))
    return boxes_per_scene, labels_per_scene


class CenterLoss(nn.Module):
    r"""Dense center-based detection loss (CenterHead / Voxel Mamba).

    Reference: :arxiv: [Center-based 3D Object Detection and Tracking](https://arxiv.org/abs/2006.11275).

    Ground-truth boxes are splatted onto a per-class BEV Gaussian heatmap and their regression code
    (sub-cell center offset, $z$, log extents, $(\cos\theta, \sin\theta)$ and any trailing columns) is
    recorded at each peak cell. The heatmap is supervised by the penalty-reduced center focal loss and
    the regression maps by a masked, code-weighted $L_1$ read back at those cells. When the head emits an
    `iou` map an optional $L_1$ term regresses it toward the 3D IoU (rescaled to $[-1, 1]$) between the
    decoded prediction and its matched box.

    Args:
        num_classes: Number of heatmap channels.
        point_cloud_range: Range $(x_\min, y_\min, z_\min, x_\max, y_\max, z_\max)$.
        voxel_size: Voxel size $(v_x, v_y, v_z)$.
        feature_map_stride: Stride from the voxel grid to the BEV feature map.
        code_weights: Per-code regression weight, length $8 + \text{extra}$ (e.g. $8$, or $10$ with velocity).
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
        point_cloud_range: Sequence[float],
        voxel_size: Sequence[float],
        feature_map_stride: int,
        *,
        code_weights: Sequence[float],
        cls_weight: float = 1.0,
        loc_weight: float = 0.25,
        iou_weight: float = 0.0,
        gaussian_overlap: float = 0.1,
        min_radius: int = 2,
        num_max_objs: int = 500,
    ) -> None:
        super().__init__()
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

    def forward(self, output: Dict[str, Tensor], batch: Dict[str, Any]) -> Dict[str, Tensor]:
        r"""Compute the dense center loss and its components.

        Args:
            output: Head maps `heatmap` $(B, C, H, W)$, `center` $(B, 2, H, W)$, `center_z` $(B, 1, H, W)$,
                `dim` $(B, 3, H, W)$, `rot` $(B, 2, H, W)$ and optionally `iou` $(B, 1, H, W)$.
            batch: Packed GT (`DataKeys.BOX`, `DataKeys.LABEL`, `DataKeys.BATCH_BOX`).

        Returns:
            A dict with the scalar `loss` and detached `hm_loss`, `loc_loss` (and `iou_loss` when enabled).
        """
        heatmap_pred = output["heatmap"]
        batch_size, _, height, width = heatmap_pred.shape
        device = heatmap_pred.device
        boxes_per_scene, labels_per_scene = _densify_gt(batch, batch_size, self.code_size - 8, device)

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

        hm_loss = _gaussian_focal_loss(_sigmoid_clamp(heatmap_pred), hm_target) * self.cls_weight

        pred_boxes = torch.cat([output["center"], output["center_z"], output["dim"], output["rot"]], dim=1)
        reg = _reg_l1_loss(_transpose_gather(pred_boxes, ind), reg_target, mask)
        loc_loss = (reg * self.code_weights).sum() * self.loc_weight

        total = hm_loss + loc_loss
        result = {"loss": total, "hm_loss": hm_loss.detach(), "loc_loss": loc_loss.detach()}

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
        gathered_iou = _transpose_gather(iou_pred, ind)
        gathered_box = _transpose_gather(pred_boxes, ind)
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
            gt = boxes_per_scene[b][: box.shape[0], :7]
            iou_target = boxes_iou3d(decoded, gt).diagonal() * 2 - 1
            total = total + F.l1_loss(gathered_iou[b][keep].view(-1), iou_target, reduction="sum")
            count = count + keep.sum()
        return total / torch.clamp_min(count, min=1.0)


def _draw_voxel_gaussian(heatmap: Tensor, distances: Tensor, radius: int) -> None:
    r"""Splat a Gaussian over occupied voxels by squared distance, max-combined in place into `heatmap`."""
    diameter = 2 * radius + 1
    sigma = diameter / 6
    gaussian = torch.exp(-distances / (2 * sigma * sigma))
    torch.max(heatmap, gaussian, out=heatmap)


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
    (non-integer) index rather than a dense-grid floor.

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

    if boxes.shape[0] == 0 or num_voxels == 0:
        return heatmap, reg_targets, inds, mask

    x, y, z = boxes[:, 0], boxes[:, 1], boxes[:, 2]
    coord_x = torch.clamp((x - point_cloud_range[0]) / voxel_size[0] / feature_map_stride, min=0, max=width - 0.5)
    coord_y = torch.clamp((y - point_cloud_range[1]) / voxel_size[1] / feature_map_stride, min=0, max=height - 0.5)
    center = torch.stack([coord_x, coord_y], dim=-1)

    dx = boxes[:, 3] / voxel_size[0] / feature_map_stride
    dy = boxes[:, 4] / voxel_size[1] / feature_map_stride
    radius = torch.clamp_min(gaussian_radius(dx, dy, min_overlap=gaussian_overlap).int(), min_radius)

    for k in range(min(num_max_objs, boxes.shape[0])):
        if dx[k] <= 0 or dy[k] <= 0:
            continue

        dist_center = ((spatial_xy - center[k]) ** 2).sum(dim=-1)
        nearest = int(dist_center.argmin())
        inds[k] = nearest
        mask[k] = 1

        cls = int(labels[k])
        r = int(radius[k].item())
        _draw_voxel_gaussian(heatmap[cls], dist_center, r)
        _draw_voxel_gaussian(heatmap[cls], ((spatial_xy - spatial_xy[nearest]) ** 2).sum(dim=-1), r)

        reg_targets[k, 0:2] = center[k] - spatial_xy[nearest]
        reg_targets[k, 2] = z[k]
        reg_targets[k, 3:6] = boxes[k, 3:6].log()
        reg_targets[k, 6] = torch.cos(boxes[k, 6])
        reg_targets[k, 7] = torch.sin(boxes[k, 6])
        if boxes.shape[1] > 7:
            reg_targets[k, 8:] = boxes[k, 7:]

    return heatmap, reg_targets, inds, mask


class SparseCenterLoss(nn.Module):
    r"""Fully sparse center-based detection loss (VoxelNeXt).

    Reference: :arxiv: [VoxelNeXt](https://arxiv.org/abs/2303.11301).

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
        point_cloud_range: Sequence[float],
        voxel_size: Sequence[float],
        feature_map_stride: int,
        *,
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

    def forward(self, output: Dict[str, Any], batch: Dict[str, Any]) -> Dict[str, Tensor]:
        r"""Compute the sparse center loss summed over class groups.

        Args:
            output: A `VoxelNeXtHeadOutput`: per-group lists `hm` $(V, n_g)$, `center` $(V, 2)$,
                `center_z` $(V, 1)$, `dim` $(V, 3)$, `rot` $(V, 2)$, `vel` $(V, 2)$ and shared
                `voxel_indices` $(V, 3)$ with columns $(\text{batch}, y, x)$.
            batch: Packed GT (`DataKeys.BOX`, `DataKeys.LABEL`, `DataKeys.BATCH_BOX`).

        Returns:
            A dict with the scalar `loss` and detached `hm_loss`, `loc_loss`.
        """
        voxel_indices = output["voxel_indices"]
        batch_index = voxel_indices[:, 0]
        device = voxel_indices.device
        batch_size = int(batch_index.max().item()) + 1 if voxel_indices.numel() else 0
        spatial_xy = voxel_indices[:, [2, 1]].float()

        boxes_per_scene, labels_per_scene = _densify_gt(batch, batch_size, self.code_size - 8, device)

        total = voxel_indices.new_zeros((), dtype=torch.float32)
        hm_total = voxel_indices.new_zeros((), dtype=torch.float32)
        loc_total = voxel_indices.new_zeros((), dtype=torch.float32)
        for group_idx, group in enumerate(self.class_groups):
            hm_pred = output["hm"][group_idx]
            pred_boxes = torch.cat(
                [output[name][group_idx] for name in ("center", "center_z", "dim", "rot", "vel")], dim=1
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

            hm_loss = _gaussian_focal_loss(_sigmoid_clamp(hm_pred), hm_target) * self.cls_weight

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
            loc_loss = (reg * self.code_weights).sum() * self.loc_weight

            total = total + hm_loss + loc_loss
            hm_total = hm_total + hm_loss.detach()
            loc_total = loc_total + loc_loss.detach()

        return {"loss": total, "hm_loss": hm_total, "loc_loss": loc_total}

    @staticmethod
    def _select_group(boxes: Tensor, labels: Tensor, group: List[int]) -> Tuple[Tensor, Tensor]:
        r"""Keep the boxes whose global class is in `group` and remap their labels to the group-local index."""
        remap = labels.new_full((max(group) + 1,), -1)
        for local, global_cls in enumerate(group):
            remap[global_cls] = local
        keep = torch.zeros_like(labels, dtype=torch.bool)
        for global_cls in group:
            keep |= labels == global_cls
        return boxes[keep], remap[labels[keep]]
