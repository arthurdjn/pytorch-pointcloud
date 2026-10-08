from typing import Any, Dict, List, Tuple

import pytest
import torch
from torch import Tensor

from torch_pointcloud.losses import CenterHeadLoss, VoxelNeXtHeadLoss
from torch_pointcloud.losses.centerpoint import _reg_l1_loss
from torch_pointcloud.utils.data import DataKeys

_POINT_CLOUD_RANGE = (-12.0, -12.0, -2.0, 12.0, 12.0, 4.0)
_VOXEL_SIZE = (1.0, 1.0, 6.0)


def _dense_data(batch_size: int = 2, num_classes: int = 3, size: int = 24) -> Tuple[Dict[str, Tensor], Dict[str, Any]]:
    torch.manual_seed(0)
    output: Dict[str, Tensor] = {
        "heatmap": torch.randn(batch_size, num_classes, size, size),
        "center": torch.randn(batch_size, 2, size, size),
        "center_z": torch.randn(batch_size, 1, size, size),
        "dim": torch.randn(batch_size, 3, size, size),
        "rot": torch.randn(batch_size, 2, size, size),
        "iou": torch.randn(batch_size, 1, size, size),
    }
    box = torch.tensor(
        [
            [2.0, 3.0, 0.2, 3.5, 2.0, 1.5, 0.4],
            [-5.0, 4.0, 0.1, 4.0, 1.8, 1.6, -0.6],
            [6.0, -3.0, 0.0, 3.2, 1.7, 1.5, 1.1],
        ]
    )
    data: Dict[str, Any] = {
        DataKeys.BOX: box,
        DataKeys.LABEL: torch.tensor([0, 1, 2]),
        DataKeys.BATCH_BOX: torch.tensor([0, 0, 1]),
    }
    return output, data


def _sparse_data(
    groups: List[List[int]],
    num_voxels: int = 400,
    batch_size: int = 2,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    torch.manual_seed(1)
    voxel_indices = torch.stack(
        [
            torch.randint(0, batch_size, (num_voxels,)),
            torch.randint(0, 180, (num_voxels,)),
            torch.randint(0, 180, (num_voxels,)),
        ],
        dim=1,
    )
    output: Dict[str, Any] = {
        "hm": [torch.randn(num_voxels, len(g)) for g in groups],
        "center": [torch.randn(num_voxels, 2) for _ in groups],
        "center_z": [torch.randn(num_voxels, 1) for _ in groups],
        "dim": [torch.randn(num_voxels, 3) for _ in groups],
        "rot": [torch.randn(num_voxels, 2) for _ in groups],
        "vel": [torch.randn(num_voxels, 2) for _ in groups],
        "voxel_indices": voxel_indices,
    }
    box = torch.tensor(
        [
            [1.0, 2.0, 0.0, 4.0, 2.0, 1.5, 0.3, 1.0, 0.5],
            [-4.0, 3.0, 0.5, 4.2, 1.8, 1.6, -0.5, 0.0, 0.0],
            [5.0, -2.0, 0.0, 3.5, 1.7, 1.5, 1.2, -1.0, 0.2],
        ]
    )
    data: Dict[str, Any] = {
        DataKeys.BOX: box,
        DataKeys.LABEL: torch.tensor([0, 6, 8]),
        DataKeys.BATCH_BOX: torch.tensor([0, 0, 1]),
    }
    return output, data


def _perfect_dense_output(box: Tensor, size: int = 24) -> Dict[str, Tensor]:
    """Constant maps that decode exactly to `box` at its peak cell (its center sits mid-cell)."""
    return {
        "heatmap": torch.zeros(1, 1, size, size),
        "center": torch.full((1, 2, size, size), 0.5),
        "center_z": torch.full((1, 1, size, size), float(box[2])),
        "dim": torch.log(box[3:6]).view(1, 3, 1, 1).expand(1, 3, size, size).clone(),
        "rot": torch.stack([torch.cos(box[6]).expand(size, size), torch.sin(box[6]).expand(size, size)]).unsqueeze(0),
        "iou": torch.ones(1, 1, size, size),  # matches the rescaled target 2 * iou3d - 1 = 1 of a perfect box
    }


def test_center_loss_perfect_predictions_regression_terms_zero() -> None:
    loss_fn = CenterHeadLoss(
        1,
        point_cloud_range=_POINT_CLOUD_RANGE,
        voxel_size=_VOXEL_SIZE,
        feature_map_stride=1,
        code_weights=[1.0] * 8,
        iou_weight=1.0,
    )
    box = torch.tensor([2.5, 3.5, 0.2, 3.0, 2.0, 1.5, 0.4])
    data: Dict[str, Any] = {
        DataKeys.BOX: box.unsqueeze(0),
        DataKeys.LABEL: torch.tensor([0]),
        DataKeys.BATCH_BOX: torch.tensor([0]),
    }
    output = _perfect_dense_output(box)
    out = loss_fn(output, data)
    assert out["box_loss"] < 1e-5
    assert out["iou_loss"] < 1e-5

    output["center"] = output["center"] + 0.25
    perturbed = loss_fn(output, data)
    assert perturbed["box_loss"] > out["box_loss"] + 0.01


def test_center_loss_iou_targets_skip_degenerate_boxes() -> None:
    """A degenerate first GT box (skipped by the target assigner) must not shift later IoU targets."""
    loss_fn = CenterHeadLoss(
        1,
        point_cloud_range=_POINT_CLOUD_RANGE,
        voxel_size=_VOXEL_SIZE,
        feature_map_stride=1,
        code_weights=[1.0] * 8,
        iou_weight=1.0,
    )
    box = torch.tensor([2.5, 3.5, 0.2, 3.0, 2.0, 1.5, 0.0])
    degenerate = torch.tensor([0.0, 0.0, 0.0, 0.0, 2.0, 1.5, 0.0])
    data: Dict[str, Any] = {
        DataKeys.BOX: torch.stack([degenerate, box]),
        DataKeys.LABEL: torch.tensor([0, 0]),
        DataKeys.BATCH_BOX: torch.tensor([0, 0]),
    }
    out = loss_fn(_perfect_dense_output(box), data)
    assert out["iou_loss"] < 1e-5  # the prediction decodes exactly to the kept box, so its target is 1


def test_center_loss_returns_scalar_dict() -> None:
    loss_fn = CenterHeadLoss(
        3,
        point_cloud_range=_POINT_CLOUD_RANGE,
        voxel_size=_VOXEL_SIZE,
        feature_map_stride=1,
        code_weights=[1.0] * 8,
    )
    output, data = _dense_data()
    out = loss_fn(output, data)
    for key in ("loss", "heatmap_loss", "box_loss"):
        assert key in out, key
        assert out[key].ndim == 0
        assert torch.isfinite(out[key])


def test_center_loss_iou_branch() -> None:
    loss_fn = CenterHeadLoss(
        3,
        point_cloud_range=_POINT_CLOUD_RANGE,
        voxel_size=_VOXEL_SIZE,
        feature_map_stride=1,
        code_weights=[1.0] * 8,
        iou_weight=1.0,
    )
    output, data = _dense_data()
    out = loss_fn(output, data)
    assert "iou_loss" in out
    assert torch.isfinite(out["iou_loss"])


def test_center_loss_forward_returns_dict() -> None:
    loss_fn = CenterHeadLoss(
        3,
        point_cloud_range=_POINT_CLOUD_RANGE,
        voxel_size=_VOXEL_SIZE,
        feature_map_stride=1,
        code_weights=[1.0] * 8,
    )
    output, data = _dense_data()
    out = loss_fn(output, data)
    assert isinstance(out, dict) and out["loss"].ndim == 0


def test_center_loss_backward() -> None:
    loss_fn = CenterHeadLoss(
        3,
        point_cloud_range=_POINT_CLOUD_RANGE,
        voxel_size=_VOXEL_SIZE,
        feature_map_stride=1,
        code_weights=[1.0] * 8,
    )
    output, data = _dense_data()
    for value in output.values():
        value.requires_grad_(True)
    loss_fn(output, data)["loss"].backward()
    assert output["heatmap"].grad is not None
    assert output["center"].grad is not None


def test_center_loss_rejects_velocity_code_weights() -> None:
    """The dense head predicts no velocity codes, so a length-10 `code_weights` must raise, not crash later."""
    with pytest.raises(ValueError, match="`code_weights` must have length 8"):
        CenterHeadLoss(
            3,
            point_cloud_range=_POINT_CLOUD_RANGE,
            voxel_size=_VOXEL_SIZE,
            feature_map_stride=1,
            code_weights=[1.0] * 8 + [0.2, 0.2],
        )


def test_center_loss_no_boxes_is_finite() -> None:
    loss_fn = CenterHeadLoss(
        3,
        point_cloud_range=_POINT_CLOUD_RANGE,
        voxel_size=_VOXEL_SIZE,
        feature_map_stride=1,
        code_weights=[1.0] * 8,
    )
    output, data = _dense_data()
    data[DataKeys.BOX] = data[DataKeys.BOX][:0]
    data[DataKeys.LABEL] = data[DataKeys.LABEL][:0]
    data[DataKeys.BATCH_BOX] = data[DataKeys.BATCH_BOX][:0]
    assert torch.isfinite(loss_fn(output, data)["loss"])


def test_sparse_center_loss_returns_scalar_dict() -> None:
    groups = [[0], [1, 2], [3, 4], [5], [6, 7], [8, 9]]
    loss_fn = VoxelNeXtHeadLoss(
        groups,
        point_cloud_range=(-54.0, -54.0, -5.0, 54.0, 54.0, 3.0),
        voxel_size=(0.075, 0.075, 0.2),
        feature_map_stride=8,
        code_weights=[1.0] * 8 + [0.2, 0.2],
    )
    output, data = _sparse_data(groups)
    out = loss_fn(output, data)
    for key in ("loss", "heatmap_loss", "box_loss"):
        assert key in out, key
        assert out[key].ndim == 0
        assert torch.isfinite(out[key])


def test_sparse_center_loss_backward() -> None:
    groups = [[0, 1]]
    loss_fn = VoxelNeXtHeadLoss(
        groups,
        point_cloud_range=(-54.0, -54.0, -5.0, 54.0, 54.0, 3.0),
        voxel_size=(0.075, 0.075, 0.2),
        feature_map_stride=8,
        code_weights=[1.0] * 8 + [0.2, 0.2],
    )
    output, data = _sparse_data(groups)
    data[DataKeys.LABEL] = torch.tensor([0, 1, 0])  # 0-based global classes {0, 1}
    for key in ("hm", "center", "center_z", "dim", "rot", "vel"):
        for tensor in output[key]:
            tensor.requires_grad_(True)
    loss_fn(output, data)["loss"].backward()
    assert output["hm"][0].grad is not None
    assert output["center"][0].grad is not None


def test_reg_l1_loss_non_finite_target_codes_are_neutralized() -> None:
    torch.manual_seed(0)
    pred = torch.randn(1, 2, 8)
    target = torch.randn(1, 2, 8)
    target[0, 0, 3] = float("nan")
    target[0, 1, 5] = float("inf")
    mask = torch.ones(1, 2, dtype=torch.long)
    loss = _reg_l1_loss(pred, target, mask)
    assert torch.isfinite(loss).all()
    # A non-finite code contributes exactly zero, as if the target matched the prediction there.
    neutral = target.clone()
    neutral[0, 0, 3] = pred[0, 0, 3]
    neutral[0, 1, 5] = pred[0, 1, 5]
    assert torch.allclose(loss, _reg_l1_loss(pred, neutral, mask))


def test_center_loss_zero_height_box_is_finite() -> None:
    loss_fn = CenterHeadLoss(
        3,
        point_cloud_range=_POINT_CLOUD_RANGE,
        voxel_size=_VOXEL_SIZE,
        feature_map_stride=1,
        code_weights=[1.0] * 8,
    )
    output, data = _dense_data()
    data[DataKeys.BOX][0, 5] = 0.0
    assert torch.isfinite(loss_fn(output, data)["loss"])


def test_sparse_center_loss_zero_height_box_is_finite() -> None:
    groups = [[0, 1]]
    loss_fn = VoxelNeXtHeadLoss(
        groups,
        point_cloud_range=(-54.0, -54.0, -5.0, 54.0, 54.0, 3.0),
        voxel_size=(0.075, 0.075, 0.2),
        feature_map_stride=8,
        code_weights=[1.0] * 8 + [0.2, 0.2],
    )
    output, data = _sparse_data(groups)
    data[DataKeys.LABEL] = torch.tensor([0, 1, 0])
    data[DataKeys.BOX][0, 5] = 0.0
    assert torch.isfinite(loss_fn(output, data)["loss"])


def test_sparse_center_loss_empty_mid_batch_scene_is_finite() -> None:
    """A batch element with zero occupied voxels (an empty or out-of-range cloud) must not crash the gather."""
    groups = [[0, 1]]
    loss_fn = VoxelNeXtHeadLoss(
        groups,
        point_cloud_range=(-54.0, -54.0, -5.0, 54.0, 54.0, 3.0),
        voxel_size=(0.075, 0.075, 0.2),
        feature_map_stride=8,
        code_weights=[1.0] * 8 + [0.2, 0.2],
    )
    output, data = _sparse_data(groups)
    output["voxel_indices"][:, 0] = output["voxel_indices"][:, 0] * 2  # {0, 1} -> {0, 2}, leaving scene 1 empty
    data[DataKeys.LABEL] = torch.tensor([0, 1, 0])
    assert torch.isfinite(loss_fn(output, data)["loss"])


def _assign_sparse_scene_loop(boxes: Tensor, labels: Tensor, **kwargs: Any) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """The per-object loop the vectorized `_assign_sparse_scene` replaced, kept as the reference."""
    from torch_pointcloud.ops.heatmap import gaussian_radius

    num_classes, spatial_xy, feature_map_size = kwargs["num_classes"], kwargs["spatial_xy"], kwargs["feature_map_size"]
    voxel_size, point_cloud_range = kwargs["voxel_size"], kwargs["point_cloud_range"]
    feature_map_stride, num_max_objs = kwargs["feature_map_stride"], kwargs["num_max_objs"]
    gaussian_overlap, min_radius = kwargs["gaussian_overlap"], kwargs["min_radius"]

    width, height = feature_map_size
    num_voxels = spatial_xy.shape[0]
    heatmap = boxes.new_zeros(num_classes, num_voxels)
    reg_targets = boxes.new_zeros(num_max_objs, boxes.shape[-1] + 1)
    inds = boxes.new_zeros(num_max_objs, dtype=torch.long)
    mask = boxes.new_zeros(num_max_objs, dtype=torch.long)
    if boxes.shape[0] == 0 or num_voxels == 0:
        return heatmap, reg_targets, inds, mask

    x, y, z = boxes[:, 0], boxes[:, 1], boxes[:, 2]
    pos_x = torch.clamp((x - point_cloud_range[0]) / voxel_size[0] / feature_map_stride, min=0, max=width - 0.5)
    pos_y = torch.clamp((y - point_cloud_range[1]) / voxel_size[1] / feature_map_stride, min=0, max=height - 0.5)
    center = torch.stack([pos_x, pos_y], dim=-1)
    dx = boxes[:, 3] / voxel_size[0] / feature_map_stride
    dy = boxes[:, 4] / voxel_size[1] / feature_map_stride
    radius = torch.clamp_min(gaussian_radius(dy, dx, min_overlap=gaussian_overlap).int(), min_radius)
    for k in range(min(num_max_objs, boxes.shape[0])):
        if dx[k] <= 0 or dy[k] <= 0:
            continue

        dist_center = ((spatial_xy - center[k]) ** 2).sum(dim=-1)
        nearest = int(dist_center.argmin())
        inds[k] = nearest
        mask[k] = 1
        cls = int(labels[k])
        r = int(radius[k].item())
        for distances in (dist_center, ((spatial_xy - spatial_xy[nearest]) ** 2).sum(dim=-1)):
            sigma = (2 * r + 1) / 6
            torch.max(heatmap[cls], torch.exp(-distances / (2 * sigma * sigma)), out=heatmap[cls])
        reg_targets[k, 0:2] = center[k] - spatial_xy[nearest]
        reg_targets[k, 2] = z[k]
        reg_targets[k, 3:6] = boxes[k, 3:6].clamp_min(1e-5).log()
        reg_targets[k, 6] = torch.cos(boxes[k, 6])
        reg_targets[k, 7] = torch.sin(boxes[k, 6])
        if boxes.shape[1] > 7:
            reg_targets[k, 8:] = boxes[k, 7:]
    return heatmap, reg_targets, inds, mask


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_assign_sparse_scene_matches_per_object_loop(seed: int) -> None:
    """The vectorized sparse assignment equals the per-object reference loop (degenerate boxes, a capped count)."""
    from torch_pointcloud.losses.centerpoint import _assign_sparse_scene

    torch.manual_seed(seed)
    num = 40
    boxes = torch.cat(
        [
            torch.rand(num, 2) * 24 - 12,
            torch.rand(num, 1),
            torch.rand(num, 3) * 6 + 0.2,
            torch.randn(num, 1),
            torch.randn(num, 2),
        ],
        dim=1,
    )
    boxes[:2, 4] = 0.0
    labels = torch.randint(0, 3, (num,))
    spatial_xy = torch.randint(0, 24, (300, 2)).float()
    kwargs: Dict[str, Any] = dict(
        num_classes=3,
        spatial_xy=spatial_xy,
        feature_map_size=(24, 24),
        voxel_size=(1.0, 1.0, 6.0),
        point_cloud_range=(-12.0, -12.0, -2.0, 12.0, 12.0, 4.0),
        feature_map_stride=1,
        num_max_objs=32,
        gaussian_overlap=0.1,
        min_radius=2,
    )
    expected = _assign_sparse_scene_loop(boxes, labels, **kwargs)
    result = _assign_sparse_scene(boxes, labels, **kwargs)
    for got, ref in zip(result, expected):
        assert torch.allclose(got.float(), ref.float(), atol=1e-6, rtol=1e-6), (got - ref).abs().max()
    assert torch.equal(result[2], expected[2]) and torch.equal(result[3], expected[3])
