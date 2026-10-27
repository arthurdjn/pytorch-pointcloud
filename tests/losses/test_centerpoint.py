from typing import Any, Dict, List, Optional, Tuple

import pytest
import torch
from torch import Tensor

from torch_pointcloud.losses import (
    CenterHeadLoss,
    VoxelNeXtHeadLoss,
    center_regression_loss,
    gaussian_focal_loss,
    iou_prediction_loss,
)
from torch_pointcloud.ops.heatmap import draw_heatmap_targets, transpose_gather
from torch_pointcloud.utils.data import DataKeys

_POINT_CLOUD_RANGE = (-12.0, -12.0, -2.0, 12.0, 12.0, 4.0)
_VOXEL_SIZE = (1.0, 1.0, 6.0)


def _dense_data(
    groups: Optional[List[List[int]]] = None,
    batch_size: int = 2,
    size: int = 24,
    use_velocity: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    groups = [[0, 1, 2]] if groups is None else groups
    torch.manual_seed(0)
    output: Dict[str, Any] = {
        "heatmap": [torch.randn(batch_size, len(g), size, size) for g in groups],
        "center": [torch.randn(batch_size, 2, size, size) for _ in groups],
        "center_z": [torch.randn(batch_size, 1, size, size) for _ in groups],
        "dim": [torch.randn(batch_size, 3, size, size) for _ in groups],
        "rot": [torch.randn(batch_size, 2, size, size) for _ in groups],
        "iou": [torch.randn(batch_size, 1, size, size) for _ in groups],
    }
    if use_velocity:
        output["vel"] = [torch.randn(batch_size, 2, size, size) for _ in groups]
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


def _perfect_dense_output(box: Tensor, size: int = 24) -> Dict[str, Any]:
    """Constant maps that decode exactly to `box` at its peak cell (its center sits mid-cell)."""
    return {
        "heatmap": [torch.zeros(1, 1, size, size)],
        "center": [torch.full((1, 2, size, size), 0.5)],
        "center_z": [torch.full((1, 1, size, size), float(box[2]))],
        "dim": [torch.log(box[3:6]).view(1, 3, 1, 1).expand(1, 3, size, size).clone()],
        "rot": [torch.stack([torch.cos(box[6]).expand(size, size), torch.sin(box[6]).expand(size, size)]).unsqueeze(0)],
        "iou": [torch.ones(1, 1, size, size)],  # matches the rescaled target 2 * iou3d - 1 = 1 of a perfect box
    }


def test_center_loss_perfect_predictions_regression_terms_zero() -> None:
    loss_fn = CenterHeadLoss(
        [[0]],
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

    output["center"] = [output["center"][0] + 0.25]
    perturbed = loss_fn(output, data)
    assert perturbed["box_loss"] > out["box_loss"] + 0.01


def test_center_loss_iou_targets_skip_degenerate_boxes() -> None:
    """A degenerate first GT box (skipped by the target assigner) must not shift later IoU targets."""
    loss_fn = CenterHeadLoss(
        [[0]],
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
        [[0, 1, 2]],
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
        [[0, 1, 2]],
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
        [[0, 1, 2]],
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
        [[0, 1, 2]],
        point_cloud_range=_POINT_CLOUD_RANGE,
        voxel_size=_VOXEL_SIZE,
        feature_map_stride=1,
        code_weights=[1.0] * 8,
    )
    output, data = _dense_data()
    for maps in output.values():
        for tensor in maps:
            tensor.requires_grad_(True)
    loss_fn(output, data)["loss"].backward()
    assert output["heatmap"][0].grad is not None
    assert output["center"][0].grad is not None


def test_center_loss_class_groups_and_velocity_codes() -> None:
    """Two groups split the three classes; 9-column boxes supervise the `vel` maps through 10 code weights."""
    groups = [[0], [1, 2]]
    loss_fn = CenterHeadLoss(
        groups,
        point_cloud_range=_POINT_CLOUD_RANGE,
        voxel_size=_VOXEL_SIZE,
        feature_map_stride=1,
        code_weights=[1.0] * 8 + [0.2, 0.2],
    )
    output, data = _dense_data(groups, use_velocity=True)
    data[DataKeys.BOX] = torch.cat([data[DataKeys.BOX], torch.tensor([[1.0, 0.5], [0.0, 0.0], [-1.0, 0.2]])], dim=1)
    for tensor in output["vel"]:
        tensor.requires_grad_(True)
    out = loss_fn(output, data)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    assert output["vel"][0].grad is not None and output["vel"][1].grad is not None


def test_center_loss_rejects_short_code_weights() -> None:
    with pytest.raises(ValueError, match="at least 8"):
        CenterHeadLoss(
            [[0, 1, 2]],
            point_cloud_range=_POINT_CLOUD_RANGE,
            voxel_size=_VOXEL_SIZE,
            feature_map_stride=1,
            code_weights=[1.0] * 7,
        )


def test_center_loss_no_boxes_is_finite() -> None:
    loss_fn = CenterHeadLoss(
        [[0, 1, 2]],
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


def test_center_regression_loss_non_finite_target_codes_are_neutralized() -> None:
    torch.manual_seed(0)
    pred = torch.randn(1, 2, 8)
    target = torch.randn(1, 2, 8)
    target[0, 0, 3] = float("nan")
    target[0, 1, 5] = float("inf")
    loss = center_regression_loss(pred, target)
    assert torch.isfinite(loss)
    # A non-finite code contributes exactly zero, as if the target matched the prediction there.
    neutral = target.clone()
    neutral[0, 0, 3] = pred[0, 0, 3]
    neutral[0, 1, 5] = pred[0, 1, 5]
    assert torch.allclose(loss, center_regression_loss(pred, neutral))


def test_center_regression_loss_averages_the_masked_objects_with_code_weights() -> None:
    pred = torch.zeros(1, 3, 2)
    target = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [100.0, 100.0]]])
    mask = torch.tensor([[1, 1, 0]])
    # The third object is masked out and the two others average (1 + 0.5 * 2 + 3 + 0.5 * 4) / 2.
    loss = center_regression_loss(pred, target, mask, weight=torch.tensor([1.0, 0.5]))
    assert torch.allclose(loss, torch.tensor(3.5))
    assert torch.equal(center_regression_loss(pred[:, :2], target[:, :2]), center_regression_loss(pred, target, mask))


def test_center_head_loss_is_the_two_terms_on_drawn_targets() -> None:
    """`CenterHeadLoss` on one group equals the focal and regression terms on targets drawn outside it."""
    output, data = _dense_data()
    head_loss = CenterHeadLoss(
        [[0, 1, 2]],
        point_cloud_range=_POINT_CLOUD_RANGE,
        voxel_size=_VOXEL_SIZE,
        feature_map_stride=1,
        code_weights=[1.0] * 8,
        loc_weight=0.25,
    )
    batch = data[DataKeys.BATCH_BOX]
    targets = [
        draw_heatmap_targets(
            data[DataKeys.BOX][batch == b],
            data[DataKeys.LABEL][batch == b],
            num_classes=3,
            feature_map_size=(24, 24),
            voxel_size=_VOXEL_SIZE,
            point_cloud_range=_POINT_CLOUD_RANGE,
            feature_map_stride=1,
        )
        for b in range(2)
    ]
    heatmap_target, code_target, inds, mask = (torch.stack(target) for target in zip(*targets))
    box_maps = torch.cat([output[key][0] for key in ("center", "center_z", "dim", "rot")], dim=1)

    heatmap_loss = gaussian_focal_loss(output["heatmap"][0].sigmoid(), heatmap_target)
    box_loss = center_regression_loss(transpose_gather(box_maps, inds), code_target, mask) * 0.25
    expected = head_loss(output, data)
    assert torch.allclose(heatmap_loss, expected["heatmap_loss"])
    assert torch.allclose(box_loss, expected["box_loss"])
    assert torch.allclose(heatmap_loss + box_loss, expected["loss"])


def test_center_loss_zero_height_box_is_finite() -> None:
    loss_fn = CenterHeadLoss(
        [[0, 1, 2]],
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


def _single_box_data(box: Tensor, label: int = 0) -> Dict[str, Any]:
    return {
        DataKeys.BOX: box.unsqueeze(0),
        DataKeys.LABEL: torch.tensor([label]),
        DataKeys.BATCH_BOX: torch.tensor([0]),
    }


def test_center_loss_iou_term_trains_only_the_iou_map() -> None:
    """The IoU target is a label: with only the IoU term on, the box maps get no gradient from it."""
    loss_fn = CenterHeadLoss(
        [[0]],
        point_cloud_range=_POINT_CLOUD_RANGE,
        voxel_size=_VOXEL_SIZE,
        feature_map_stride=1,
        code_weights=[1.0] * 8,
        cls_weight=0.0,
        loc_weight=0.0,
        iou_weight=1.0,
    )
    box = torch.tensor([2.5, 3.5, 0.2, 3.0, 2.0, 1.5, 0.4])
    output = _perfect_dense_output(box)
    # An offset prediction keeps the IoU below one, where a gradient through the target would not vanish.
    output["center"] = [output["center"][0] + 0.25]
    for maps in output.values():
        for tensor in maps:
            tensor.requires_grad_(True)
    loss_fn(output, _single_box_data(box))["loss"].backward()
    assert output["iou"][0].grad is not None and output["iou"][0].grad.abs().sum() > 0
    for key in ("center", "center_z", "dim", "rot"):
        grad = output[key][0].grad
        assert grad is None or grad.abs().sum() == 0, key


def test_center_loss_perfect_velocity_codes_are_zero() -> None:
    loss_fn = CenterHeadLoss(
        [[0]],
        point_cloud_range=_POINT_CLOUD_RANGE,
        voxel_size=_VOXEL_SIZE,
        feature_map_stride=1,
        code_weights=[1.0] * 10,
    )
    box = torch.tensor([2.5, 3.5, 0.2, 3.0, 2.0, 1.5, 0.4, 1.5, -0.5])
    output = _perfect_dense_output(box[:7])
    output["vel"] = [box[7:9].view(1, 2, 1, 1).expand(1, 2, 24, 24).clone()]
    assert loss_fn(output, _single_box_data(box))["box_loss"] < 1e-5

    output["vel"] = [output["vel"][0] + 0.5]
    assert loss_fn(output, _single_box_data(box))["box_loss"] > 0.2


def test_center_loss_maps_a_class_onto_its_group_channel() -> None:
    """Class 2 is the second channel of the group `[1, 2]`, so a peak there costs less than on the first channel."""
    loss_fn = CenterHeadLoss(
        [[0], [1, 2]],
        point_cloud_range=_POINT_CLOUD_RANGE,
        voxel_size=_VOXEL_SIZE,
        feature_map_stride=1,
        code_weights=[1.0] * 8,
    )
    box = torch.tensor([2.5, 3.5, 0.2, 3.0, 2.0, 1.5, 0.4])
    output = {key: maps * 2 for key, maps in _perfect_dense_output(box).items() if key != "iou"}

    def heatmap_loss(channel: int) -> float:
        second = torch.full((1, 2, 24, 24), -10.0)
        second[0, channel, 15, 14] = 10.0  # the peak cell of the box center (2.5, 3.5)
        output["heatmap"] = [torch.full((1, 1, 24, 24), -10.0), second]
        return float(loss_fn(output, _single_box_data(box, label=2))["heatmap_loss"])

    assert heatmap_loss(1) < heatmap_loss(0)


def test_center_loss_class_count_is_deprecated() -> None:
    with pytest.warns(FutureWarning, match="class_groups"):
        loss_fn = CenterHeadLoss(
            3,
            point_cloud_range=_POINT_CLOUD_RANGE,
            voxel_size=_VOXEL_SIZE,
            feature_map_stride=1,
            code_weights=[1.0] * 8,
        )
    assert loss_fn.class_groups == [[0, 1, 2]]


def test_center_loss_rejects_mismatched_codes_and_boxes() -> None:
    kwargs: Dict[str, Any] = dict(point_cloud_range=_POINT_CLOUD_RANGE, voxel_size=_VOXEL_SIZE, feature_map_stride=1)
    output, data = _dense_data(use_velocity=True)
    with pytest.raises(ValueError, match="regresses 10 box codes"):
        CenterHeadLoss([[0, 1, 2]], code_weights=[1.0] * 8, **kwargs)(output, data)
    with pytest.raises(ValueError, match="boxes with 9 columns"):
        CenterHeadLoss([[0, 1, 2]], code_weights=[1.0] * 10, **kwargs)(output, data)


def test_iou_prediction_loss_trains_the_prediction_only() -> None:
    """The IoU target is a label: a perfect prediction costs nothing and the boxes get no gradient."""
    boxes = torch.tensor([[[1.0, 2.0, 0.0, 4.0, 2.0, 1.5, 0.3], [0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0]]])
    shifted = boxes.clone()
    shifted[0, 1, 0] = 1.0  # half overlap along x: IoU 1/3, target -1/3
    pred_boxes = shifted.clone().requires_grad_(True)
    pred = torch.tensor([[1.0, -1.0 / 3.0]], requires_grad=True)
    loss = iou_prediction_loss(pred, pred_boxes, boxes)
    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)

    pred = torch.tensor([[0.0, 0.0]], requires_grad=True)
    loss = iou_prediction_loss(pred, pred_boxes, boxes)
    loss.backward()
    assert torch.allclose(loss, torch.tensor((1.0 + 1.0 / 3.0) / 2), atol=1e-6)
    assert pred.grad is not None and pred.grad.abs().sum() > 0
    assert pred_boxes.grad is None

    masked = iou_prediction_loss(torch.tensor([[0.0, 0.0]]), pred_boxes, boxes, mask=torch.tensor([[1, 0]]))
    assert torch.allclose(masked, torch.tensor(1.0), atol=1e-6)
    assert iou_prediction_loss(torch.tensor([[0.0, 0.0]]), pred_boxes, boxes, mask=torch.tensor([[0, 0]])) == 0
