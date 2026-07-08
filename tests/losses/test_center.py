from typing import Any, Dict, List, Tuple

import torch
from torch import Tensor

from torch_pointcloud.losses import CenterLoss, SparseCenterLoss
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
    batch: Dict[str, Any] = {
        DataKeys.BOX: box,
        DataKeys.LABEL: torch.tensor([0, 1, 2]),
        DataKeys.BATCH_BOX: torch.tensor([0, 0, 1]),
    }
    return output, batch


def _sparse_data(
    groups: List[List[int]], num_voxels: int = 400, batch_size: int = 2
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
    batch: Dict[str, Any] = {
        DataKeys.BOX: box,
        DataKeys.LABEL: torch.tensor([0, 6, 8]),
        DataKeys.BATCH_BOX: torch.tensor([0, 0, 1]),
    }
    return output, batch


def test_center_loss_returns_scalar_dict() -> None:
    loss_fn = CenterLoss(3, _POINT_CLOUD_RANGE, _VOXEL_SIZE, feature_map_stride=1, code_weights=[1.0] * 8)
    output, batch = _dense_data()
    out = loss_fn(output, batch)
    for key in ("loss", "hm_loss", "loc_loss"):
        assert key in out, key
        assert out[key].ndim == 0
        assert torch.isfinite(out[key])


def test_center_loss_iou_branch() -> None:
    loss_fn = CenterLoss(
        3, _POINT_CLOUD_RANGE, _VOXEL_SIZE, feature_map_stride=1, code_weights=[1.0] * 8, iou_weight=1.0
    )
    output, batch = _dense_data()
    out = loss_fn(output, batch)
    assert "iou_loss" in out
    assert torch.isfinite(out["iou_loss"])


def test_center_loss_backward() -> None:
    loss_fn = CenterLoss(3, _POINT_CLOUD_RANGE, _VOXEL_SIZE, feature_map_stride=1, code_weights=[1.0] * 8)
    output, batch = _dense_data()
    for value in output.values():
        value.requires_grad_(True)
    loss_fn(output, batch)["loss"].backward()
    assert output["heatmap"].grad is not None
    assert output["center"].grad is not None


def test_center_loss_no_boxes_is_finite() -> None:
    loss_fn = CenterLoss(3, _POINT_CLOUD_RANGE, _VOXEL_SIZE, feature_map_stride=1, code_weights=[1.0] * 8)
    output, batch = _dense_data()
    batch[DataKeys.BOX] = batch[DataKeys.BOX][:0]
    batch[DataKeys.LABEL] = batch[DataKeys.LABEL][:0]
    batch[DataKeys.BATCH_BOX] = batch[DataKeys.BATCH_BOX][:0]
    assert torch.isfinite(loss_fn(output, batch)["loss"])


def test_sparse_center_loss_returns_scalar_dict() -> None:
    groups = [[0], [1, 2], [3, 4], [5], [6, 7], [8, 9]]
    loss_fn = SparseCenterLoss(
        groups, (-54.0, -54.0, -5.0, 54.0, 54.0, 3.0), (0.075, 0.075, 0.2), 8, code_weights=[1.0] * 8 + [0.2, 0.2]
    )
    output, batch = _sparse_data(groups)
    out = loss_fn(output, batch)
    for key in ("loss", "hm_loss", "loc_loss"):
        assert key in out, key
        assert out[key].ndim == 0
        assert torch.isfinite(out[key])


def test_sparse_center_loss_backward() -> None:
    groups = [[0, 1]]
    loss_fn = SparseCenterLoss(
        groups, (-54.0, -54.0, -5.0, 54.0, 54.0, 3.0), (0.075, 0.075, 0.2), 8, code_weights=[1.0] * 8 + [0.2, 0.2]
    )
    output, batch = _sparse_data(groups)
    batch[DataKeys.LABEL] = torch.tensor([0, 1, 0])  # 0-based global classes {0, 1}
    for key in ("hm", "center", "center_z", "dim", "rot", "vel"):
        for tensor in output[key]:
            tensor.requires_grad_(True)
    loss_fn(output, batch)["loss"].backward()
    assert output["hm"][0].grad is not None
    assert output["center"][0].grad is not None


def test_sparse_center_loss_empty_mid_batch_scene_is_finite() -> None:
    """A batch element with zero occupied voxels (an empty or out-of-range cloud) must not crash the gather."""
    groups = [[0, 1]]
    loss_fn = SparseCenterLoss(
        groups, (-54.0, -54.0, -5.0, 54.0, 54.0, 3.0), (0.075, 0.075, 0.2), 8, code_weights=[1.0] * 8 + [0.2, 0.2]
    )
    output, batch = _sparse_data(groups)
    output["voxel_indices"][:, 0] = output["voxel_indices"][:, 0] * 2  # {0, 1} -> {0, 2}, leaving scene 1 empty
    batch[DataKeys.LABEL] = torch.tensor([0, 1, 0])
    assert torch.isfinite(loss_fn(output, batch)["loss"])
