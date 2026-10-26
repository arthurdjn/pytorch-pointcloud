from typing import Tuple

import pytest
import torch
from torch import Tensor

from torch_pointcloud.models import create_model, list_models
from torch_pointcloud.models.transfusion import TransFusionDetection
from torch_pointcloud.ops.voxelization import hard_voxelize
from torch_pointcloud.utils.imports import _SPCONV_AVAILABLE

# The sparse 3D backbone (spconv SubM/SparseConv3d) needs CUDA in this build.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANGE = (-54.0, -54.0, -5.0, 54.0, 54.0, 3.0)
VOXEL_SIZE = (0.075, 0.075, 0.2)


def _make_inputs(n_per_scene: int = 8000, batch_size: int = 2) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Random nuScenes-like points (x, y, z, intensity, dt) inside the range, hard-voxelized like the transform."""
    torch.manual_seed(0)
    pos = torch.rand(n_per_scene * batch_size, 3)
    for d in range(3):
        pos[:, d] = pos[:, d] * (RANGE[d + 3] - RANGE[d]) + RANGE[d]
    points = torch.cat([pos, torch.rand(n_per_scene * batch_size, 2)], dim=1).to(DEVICE)
    batch = torch.arange(batch_size).repeat_interleave(n_per_scene).to(DEVICE)
    voxels, voxel_indices, num_points = hard_voxelize(points, batch, VOXEL_SIZE, RANGE, 10, 160000)
    return voxels, voxel_indices[:, 1:], num_points, voxel_indices[:, 0].long()


def test_transfusion_registered_variant() -> None:
    assert list_models("transfusion*", task="detection") == ["transfusion-lidar.nuscenes.openpcdet"]


@pytest.mark.skipif(not _SPCONV_AVAILABLE, reason="spconv is not installed")
def test_transfusion_create_model_hparams() -> None:
    model = create_model("transfusion-lidar.nuscenes.openpcdet", task="detection")
    assert isinstance(model, TransFusionDetection)
    assert model.grid_size == (1440, 1440, 40)
    assert model.sparse_shape == [41, 1440, 1440]
    assert model.num_features == 512
    assert model.head.query_radius is None
    assert not model.head.use_iou
    assert model.head.height_code == "center"
    assert model.head.local_max_classes == (8, 9)


@pytest.mark.skipif(not (_SPCONV_AVAILABLE and torch.cuda.is_available()), reason="sparse backbone requires CUDA")
def test_transfusion_forward_and_decode() -> None:
    model = create_model("transfusion-lidar.nuscenes.openpcdet", task="detection").to(DEVICE).eval()
    assert isinstance(model, TransFusionDetection)
    voxels, pos_voxel, num_points, batch = _make_inputs()
    with torch.no_grad():
        features = model.forward_features(voxels, pos_voxel, num_points, batch)
        out = model.forward_head(features)
        det = model.decode(out)

    # BEV feature map 1440 / 8 = 180, 200 queries per scene.
    assert features.shape == (2, 512, 180, 180)
    assert out["dense_heatmap"].shape == (2, 10, 180, 180)
    assert out["heatmap"].shape == (2, 10, 200)
    assert out["center"].shape == (2, 2, 200)
    assert "iou" not in out
    assert det["boxes"].shape[1] == 7
    assert det["velocity"].shape == (det["boxes"].shape[0], 2)
    assert det["scores"].shape == det["labels"].shape == det["batch"].shape == (det["boxes"].shape[0],)
    assert det["boxes"].shape[0] <= 2 * 200
    assert torch.isfinite(det["boxes"]).all()
    assert (det["boxes"][:, :2].abs() <= 61.2).all()
