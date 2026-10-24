r"""Losses of the center-based detection heads, dense (`CenterHead`) and fully sparse (`VoxelNeXtHead`).

Per class group, both assign the targets with [`draw_heatmap_targets`][torch_pointcloud.ops.heatmap.draw_heatmap_targets]
or [`draw_sparse_heatmap_targets`][torch_pointcloud.ops.heatmap.draw_sparse_heatmap_targets], then supervise the
heatmap with [`gaussian_focal_loss`][torch_pointcloud.losses.focal.gaussian_focal_loss] and the box codes read at the
object centers with [`center_regression_loss`][torch_pointcloud.losses.centerpoint.center_regression_loss]. Another
center head reuses the same pieces on its own outputs:

```python
import torch
from torch_pointcloud.losses.functional import center_regression_loss, gaussian_focal_loss
from torch_pointcloud.ops.heatmap import draw_heatmap_targets, transpose_gather

heatmap_logits, box_maps = torch.zeros(1, 1, 16, 16), torch.zeros(1, 8, 16, 16)
boxes = torch.tensor([[0.0, 0.0, -1.0, 4.0, 2.0, 1.5, 0.3]])
heatmap_target, code_target, inds, mask = draw_heatmap_targets(
    boxes,
    torch.tensor([0]),
    num_classes=1,
    feature_map_size=(16, 16),
    voxel_size=[0.5, 0.5, 0.5],
    point_cloud_range=[-4.0, -4.0, -2.0, 4.0, 4.0, 2.0],
    feature_map_stride=1,
)
heatmap_loss = gaussian_focal_loss(heatmap_logits.sigmoid(), heatmap_target[None])
box_loss = center_regression_loss(transpose_gather(box_maps, inds[None]), code_target[None], mask[None])
loss = heatmap_loss + 0.25 * box_loss
```
"""

import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from torch_pointcloud.losses._utils import unbatch_boxes
from torch_pointcloud.losses.focal import gaussian_focal_loss
from torch_pointcloud.ops.box3d import boxes_iou3d
from torch_pointcloud.ops.heatmap import (
    decode_center_codes,
    draw_heatmap_targets,
    draw_sparse_heatmap_targets,
    transpose_gather,
)
from torch_pointcloud.utils.data import DataKeys

if TYPE_CHECKING:
    from torch_pointcloud.layers.center_head import CenterHeadOutput
    from torch_pointcloud.models.voxelnext import VoxelNeXtHeadOutput


def center_regression_loss(
    pred: Tensor,
    target: Tensor,
    mask: Optional[Tensor] = None,
    weight: Optional[Tensor] = None,
) -> Tensor:
    r"""Code-weighted $L_1$ between the box codes predicted at the object centers and their targets.

    The absolute errors are weighted per code, summed over codes and objects, and divided by the number of
    objects (at least one). The objects outside `mask` and the non-finite target entries (e.g. an unknown
    velocity) add nothing.

    Args:
        pred: Predicted codes at the object centers, shape $(\ldots, M, K)$.
        target: Target codes of the same shape.
        mask: Valid objects, shape $(\ldots, M)$; every object when `None`.
        weight: Per-code weight, shape $(K,)$; one for every code when `None`.

    Returns:
        The scalar loss.

    Shape:
        - pred: $(\ldots, M, K)$
        - target: $(\ldots, M, K)$
        - output: scalar

    Example:
        ```pycon
        >>> target = torch.tensor([[1.0, 0.0], [0.0, float("nan")]])
        >>> center_regression_loss(torch.zeros(2, 2), target)
        tensor(0.5000)

        ```
    """
    if mask is None:
        mask = target.new_ones(target.shape[:-1], dtype=torch.bool)
    valid = mask.bool().unsqueeze(-1) & torch.isfinite(target)
    error = torch.where(valid, pred - torch.nan_to_num(target), 0.0).abs()
    if weight is not None:
        error = error * weight
    return error.sum() / mask.sum().clamp_min(1)


def iou_prediction_loss(
    pred: Tensor,
    boxes: Tensor,
    target_boxes: Tensor,
    mask: Optional[Tensor] = None,
) -> Tensor:
    r"""$L_1$ between a predicted IoU and the 3D IoU of each predicted box with its target.

    The target is $2 \cdot \text{IoU}_{3D} - 1$, the $[-1, 1]$ convention of the IoU-aware center heads. It is a
    label: the boxes are detached, so the term trains the IoU prediction alone. The errors are summed over the
    objects in `mask` and divided by their count (at least one).

    Args:
        pred: Predicted IoU, shape $(\ldots, M)$.
        boxes: Predicted boxes $(c_x, c_y, c_z, d_x, d_y, d_z, \theta)$, shape $(\ldots, M, 7)$.
        target_boxes: Matched ground-truth boxes of the same shape.
        mask: Valid objects, shape $(\ldots, M)$; every object when `None`.

    Returns:
        The scalar loss.

    Shape:
        - pred: $(\ldots, M)$
        - boxes: $(\ldots, M, 7)$
        - target_boxes: $(\ldots, M, 7)$
        - output: scalar

    Example:
        ```pycon
        >>> boxes = torch.tensor([[[0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0]]])
        >>> iou_prediction_loss(torch.tensor([[0.5]]), boxes, boxes)
        tensor(0.5000)

        ```
    """
    if mask is None:
        mask = pred.new_ones(pred.shape, dtype=torch.bool)
    keep = mask.bool()
    if not bool(keep.any()):
        return pred.new_zeros(())

    iou = boxes_iou3d(boxes[keep].detach(), target_boxes[keep], aligned=True)
    return F.l1_loss(pred[keep], iou * 2 - 1, reduction="sum") / keep.sum()


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


def _select_group(boxes: Tensor, labels: Tensor, group: List[int]) -> Tuple[Tensor, Tensor]:
    r"""Keep the boxes whose global class is in `group` and remap their labels to the group-local index."""
    remap = labels.new_full((max(group) + 1,), -1)
    for local, global_cls in enumerate(group):
        remap[global_cls] = local

    keep = torch.isin(labels, labels.new_tensor(group))
    return boxes[keep], remap[labels[keep]]


class CenterHeadLoss(nn.Module):
    r"""Loss of the dense `CenterHead` (CenterPoint, Voxel Mamba).

    Reference: :arxiv: [Center-based 3D Object Detection and Tracking](https://arxiv.org/abs/2006.11275) (Yin et al., 2021).

    Classes are split into groups, one head each. Within a group, the ground-truth boxes are splatted onto a
    per-class BEV Gaussian heatmap and their regression code (sub-cell center offset, $z$, log extents,
    $(\cos\theta, \sin\theta)$ and any extra box columns such as velocity) is recorded at each peak cell. The
    heatmap is supervised by the penalty-reduced center focal loss and the regression maps by a masked,
    code-weighted $L_1$ read back at those cells. When the head emits an `iou` map,
    [`iou_prediction_loss`][torch_pointcloud.losses.centerpoint.iou_prediction_loss] regresses it toward the 3D IoU
    between the decoded prediction and its matched box.

    Args:
        class_groups: Zero-based global class-index groups, one per head (e.g. `[[0], [1, 2], ...]`). A class
            count is accepted until 0.0.20 for one group of every class, with a `FutureWarning`.
        point_cloud_range: Range $(x_\min, y_\min, z_\min, x_\max, y_\max, z_\max)$.
        voxel_size: Voxel size $(v_x, v_y, v_z)$.
        feature_map_stride: Stride from the voxel grid to the BEV feature map.
        code_weights: Per-code regression weight, length $8 + \text{extra}$ (e.g. $10$ with velocity).
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
        class_groups: Union[Sequence[Sequence[int]], int],
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
        if len(code_weights) < 8:
            raise ValueError(f"`code_weights` must have at least 8 entries, got {len(code_weights)}.")

        super().__init__()
        if isinstance(class_groups, int):
            warnings.warn(
                "Passing the class count to `CenterHeadLoss` was replaced by `class_groups` in 0.0.19 and will be "
                f"removed in 0.0.20; pass `[list(range({class_groups}))]` for one group of every class.",
                FutureWarning,
                stacklevel=2,
            )
            class_groups = [list(range(class_groups))]
        self.class_groups = [list(group) for group in class_groups]
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
        r"""Compute the dense center loss summed over class groups.

        Args:
            output: A `CenterHeadOutput`: per-group lists `heatmap` $(B, n_g, H, W)$, `center` $(B, 2, H, W)$,
                `center_z` $(B, 1, H, W)$, `dim` $(B, 3, H, W)$, `rot` $(B, 2, H, W)$ and optionally `vel`
                $(B, 2, H, W)$ and `iou` $(B, 1, H, W)$.
            data: Packed GT (`DataKeys.BOX`, `DataKeys.LABEL`, `DataKeys.BATCH_BOX`).

        Returns:
            A dict with the scalar `loss` and detached `heatmap_loss`, `box_loss` (and `iou_loss` when enabled).
        """
        num_codes = 8 + (output["vel"][0].shape[1] if "vel" in output else 0)
        if num_codes != self.code_size:
            raise ValueError(f"The head regresses {num_codes} box codes but `code_weights` has {self.code_size}.")
        num_columns = data[DataKeys.BOX].shape[1]
        if num_columns < self.code_size - 1:
            raise ValueError(
                f"{self.code_size} code weights need boxes with {self.code_size - 1} columns, got {num_columns}."
            )

        batch_size, _, height, width = output["heatmap"][0].shape
        device = output["heatmap"][0].device
        boxes_per_scene, labels_per_scene = unbatch_boxes(data, batch_size, device)
        boxes_per_scene = [_pad_box_columns(boxes, self.code_size - 8) for boxes in boxes_per_scene]
        use_iou = self.iou_weight > 0 and "iou" in output

        total = output["heatmap"][0].new_zeros(())
        hm_total = output["heatmap"][0].new_zeros(())
        box_total = output["heatmap"][0].new_zeros(())
        iou_total = output["heatmap"][0].new_zeros(())
        for group_idx, group in enumerate(self.class_groups):
            heatmap_pred = output["heatmap"][group_idx]
            maps = [output["center"][group_idx], output["center_z"][group_idx], output["dim"][group_idx]]
            maps.append(output["rot"][group_idx])
            if "vel" in output:
                maps.append(output["vel"][group_idx])
            pred_boxes = torch.cat(maps, dim=1)

            group_boxes: List[Tensor] = []
            hm_targets: List[Tensor] = []
            reg_targets: List[Tensor] = []
            inds: List[Tensor] = []
            masks: List[Tensor] = []
            for b in range(batch_size):
                boxes, labels = _select_group(boxes_per_scene[b], labels_per_scene[b], group)
                hm, reg, ind, mask = draw_heatmap_targets(
                    boxes,
                    labels,
                    len(group),
                    (width, height),
                    self.voxel_size,
                    self.point_cloud_range,
                    self.feature_map_stride,
                    num_max_objs=self.num_max_objs,
                    gaussian_overlap=self.gaussian_overlap,
                    min_radius=self.min_radius,
                )
                slots = boxes.new_zeros(self.num_max_objs, 7)
                slots[: min(len(boxes), self.num_max_objs)] = boxes[: self.num_max_objs, :7]
                group_boxes.append(slots)
                hm_targets.append(hm)
                reg_targets.append(reg)
                inds.append(ind)
                masks.append(mask)

            hm_target = torch.stack(hm_targets)
            reg_target = torch.stack(reg_targets)
            ind = torch.stack(inds)
            mask = torch.stack(masks)

            hm_loss = gaussian_focal_loss(heatmap_pred.sigmoid(), hm_target) * self.cls_weight
            box_codes = transpose_gather(pred_boxes, ind)
            box_loss = center_regression_loss(box_codes, reg_target, mask, weight=self.code_weights) * self.loc_weight

            total = total + hm_loss + box_loss
            hm_total = hm_total + hm_loss.detach()
            box_total = box_total + box_loss.detach()

            if use_iou:
                cell_xy = torch.stack([ind % width, torch.div(ind, width, rounding_mode="floor")], dim=-1).float()
                decoded = decode_center_codes(
                    box_codes,
                    cell_xy,
                    self.voxel_size,
                    self.point_cloud_range,
                    self.feature_map_stride,
                )
                iou_pred = transpose_gather(output["iou"][group_idx], ind).squeeze(-1)
                iou_loss = iou_prediction_loss(iou_pred, decoded, torch.stack(group_boxes), mask) * self.iou_weight
                total = total + iou_loss
                iou_total = iou_total + iou_loss.detach()

        result = {"loss": total, "heatmap_loss": hm_total, "box_loss": box_total}
        if use_iou:
            result["iou_loss"] = iou_total
        return result


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
        voxel_xy = voxel_indices[:, [2, 1]].float()

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
                group_boxes, group_labels = _select_group(boxes_per_scene[b], labels_per_scene[b], group)
                hm, reg, ind, mask = draw_sparse_heatmap_targets(
                    group_boxes,
                    group_labels,
                    len(group),
                    voxel_xy[voxel_mask],
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

            box_codes = torch.stack(rows)
            box_loss = center_regression_loss(box_codes, reg_target, mask, weight=self.code_weights) * self.loc_weight

            total = total + hm_loss + box_loss
            hm_total = hm_total + hm_loss.detach()
            box_total = box_total + box_loss.detach()

        return {"loss": total, "heatmap_loss": hm_total, "box_loss": box_total}
