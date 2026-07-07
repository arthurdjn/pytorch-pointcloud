from typing import Dict

import pytest
import torch
from torch import Tensor

from torch_pointcloud.models import create_model, list_models
from torch_pointcloud.models.second import SECONDDetection, SECONDMultiHeadDetection
from torch_pointcloud.utils.imports import _SPCONV_AVAILABLE
from torch_pointcloud.utils.voxelization import hard_voxelize

pytestmark = pytest.mark.skipif(not _SPCONV_AVAILABLE, reason="spconv is not installed")

# The sparse 3D backbone (spconv SubM/SparseConv3d) needs CUDA in this build.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANGE = (0.0, -40.0, -3.0, 70.4, 40.0, 1.0)


def _voxelize(
    model: SECONDDetection | SECONDMultiHeadDetection, data: Dict[str, Tensor], max_num_points: int, max_num_voxels: int
) -> tuple:
    """Voxelize raw packed points the way the registered `HardVoxelize` transform + collate would."""
    points = torch.cat([data["pos"], data["x"]], dim=1)
    voxels, voxel_indices, num_points = hard_voxelize(
        points, data["batch"], model.voxel_size, model.point_cloud_range, max_num_points, max_num_voxels
    )
    return voxels, voxel_indices[:, 1:], num_points, voxel_indices[:, 0]


def _make_inputs(n_per_scene: int = 8000, batch_size: int = 2) -> Dict[str, Tensor]:
    """Random KITTI-like points inside the point-cloud range, packed across scenes."""
    torch.manual_seed(0)
    pos_list, x_list, batch_list = [], [], []
    for i in range(batch_size):
        p = torch.rand(n_per_scene, 3)
        p[:, 0] = p[:, 0] * (RANGE[3] - RANGE[0]) + RANGE[0]
        p[:, 1] = p[:, 1] * (RANGE[4] - RANGE[1]) + RANGE[1]
        p[:, 2] = p[:, 2] * (RANGE[5] - RANGE[2]) + RANGE[2]
        pos_list.append(p)
        x_list.append(torch.rand(n_per_scene, 1))
        batch_list.append(torch.full((n_per_scene,), i, dtype=torch.long))
    return {
        "pos": torch.cat(pos_list).to(DEVICE),
        "x": torch.cat(x_list).to(DEVICE),
        "batch": torch.cat(batch_list).to(DEVICE),
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="SECOND sparse backbone requires CUDA")
def test_second_forward_shapes() -> None:
    model = create_model("second.kitti.openpcdet", task="detection").to(DEVICE).eval()
    assert isinstance(model, SECONDDetection)
    data = _make_inputs()
    voxels, pos_voxel, num_points, vbatch = _voxelize(model, data, 5, 40000)
    with torch.no_grad():
        out = model(voxels, pos_voxel, num_points, vbatch)

    # KITTI 3-class BEV feature map at stride 8
    b, h, w = 2, 200, 176
    n_anchors = h * w * model.head.num_anchors_per_location
    assert out["cls"].shape == (b, h, w, 18)
    assert out["box"].shape == (b, h, w, 42)
    assert out["dir_cls"].shape == (b, h, w, 12)
    assert out["batch_cls"].shape == (b, n_anchors, 3)
    assert out["batch_box"].shape == (b, n_anchors, 7)
    assert torch.isfinite(out["batch_box"]).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="SECOND sparse backbone requires CUDA")
def test_second_eval_is_deterministic() -> None:
    model = create_model("second.kitti.openpcdet", task="detection").to(DEVICE).eval()
    assert isinstance(model, SECONDDetection)
    data = _make_inputs()
    voxels, pos_voxel, num_points, vbatch = _voxelize(model, data, 5, 40000)
    with torch.no_grad():
        a = model(voxels, pos_voxel, num_points, vbatch)
        b = model(voxels, pos_voxel, num_points, vbatch)
    for key in ("cls", "box", "batch_box"):
        assert torch.allclose(a[key], b[key], atol=1e-5), f"{key} drifted beyond float-atomics on CUDA"


def test_second_registered_variant() -> None:
    assert "second.kitti.openpcdet" in list_models("second*", task="detection")


def test_second_create_model_hparams() -> None:
    model = create_model("second.kitti.openpcdet", task="detection")
    assert isinstance(model, SECONDDetection)
    assert model.in_channels == 4
    assert model.num_classes == 3
    assert model.grid_size == (1408, 1600, 40)
    assert model.sparse_shape == [41, 1600, 1408]
    assert model.head.anchors.shape == (200 * 176 * 6, 7)


NUSCENES_RANGE = (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0)


def _make_nuscenes_inputs(n_per_scene: int = 8000, batch_size: int = 2) -> Dict[str, Tensor]:
    """Random nuScenes-like points (5 features: x, y, z, intensity, dt) inside the range."""
    torch.manual_seed(0)
    pos_list, x_list, batch_list = [], [], []
    for i in range(batch_size):
        p = torch.rand(n_per_scene, 3)
        for d in range(3):
            p[:, d] = p[:, d] * (NUSCENES_RANGE[d + 3] - NUSCENES_RANGE[d]) + NUSCENES_RANGE[d]
        pos_list.append(p)
        x_list.append(torch.rand(n_per_scene, 2))
        batch_list.append(torch.full((n_per_scene,), i, dtype=torch.long))
    return {
        "pos": torch.cat(pos_list).to(DEVICE),
        "x": torch.cat(x_list).to(DEVICE),
        "batch": torch.cat(batch_list).to(DEVICE),
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="SECOND sparse backbone requires CUDA")
def test_second_multihead_forward_shapes() -> None:
    model = create_model("second-multihead.nuscenes.openpcdet", task="detection").to(DEVICE).eval()
    assert isinstance(model, SECONDMultiHeadDetection)
    data = _make_nuscenes_inputs()
    voxels, pos_voxel, num_points, vbatch = _voxelize(model, data, 10, 60000)
    with torch.no_grad():
        out = model(voxels, pos_voxel, num_points, vbatch)

    # 6 RPN groups; BEV feature map 1024/8 = 128, 10 classes x 2 rotations = 327680 anchors.
    assert len(out["cls"]) == 6
    assert [m.tolist() for m in out["multihead_label_mapping"]] == [[1], [2, 3], [4, 5], [6], [7, 8], [9, 10]]
    assert out["cls"][0].shape == (2, 32768, 1)
    assert out["box"][1].shape == (2, 65536, 10)
    assert out["batch_box"].shape == (2, 128 * 128 * 10 * 2, 9)
    assert torch.isfinite(out["batch_box"]).all()


def test_second_multihead_create_model_hparams() -> None:
    model = create_model("second-multihead.nuscenes.openpcdet", task="detection")
    assert isinstance(model, SECONDMultiHeadDetection)
    assert model.in_channels == 5
    assert model.num_classes == 10
    assert model.grid_size == (1024, 1024, 40)
    assert model.sparse_shape == [41, 1024, 1024]
    assert len(model.head.rpn_heads) == 6
    assert "second-multihead.nuscenes.openpcdet" in list_models("second*", task="detection")
