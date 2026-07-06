from pathlib import Path
from typing import Dict

import pytest
import torch
from torch import Tensor

import torch_pointcloud.models.voxelnext  # noqa: F401
from torch_pointcloud.config import MODELS_DIR
from torch_pointcloud.models import create_model, list_models
from torch_pointcloud.models.voxelnext import VoxelNeXtDetection
from torch_pointcloud.utils.imports import _SPCONV_AVAILABLE
from torch_pointcloud.utils.voxelization import hard_voxelize

pytestmark = pytest.mark.skipif(not _SPCONV_AVAILABLE, reason="spconv is not installed")

# The fully sparse 3D backbone (spconv SubM/SparseConv) needs CUDA in this build.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANGE = (-54.0, -54.0, -5.0, 54.0, 54.0, 3.0)
WEIGHTS = Path(MODELS_DIR, "voxelnext", "voxelnext-openpcdet.nuscenes.pt")


def _voxelize(model: VoxelNeXtDetection, data: Dict[str, Tensor], max_num_points: int, max_num_voxels: int) -> tuple:
    """Voxelize raw packed points the way the registered `HardVoxelize` transform + collate would."""
    points = torch.cat([data["pos"], data["x"]], dim=1)
    voxels, voxel_indices, num_points = hard_voxelize(
        points, data["batch"], model.voxel_size, model.point_cloud_range, max_num_points, max_num_voxels
    )
    return voxels, voxel_indices[:, 1:], num_points, voxel_indices[:, 0]


def _make_inputs(n_per_scene: int = 8000, batch_size: int = 2) -> Dict[str, Tensor]:
    """Random nuScenes-like points (5 features: x, y, z, intensity, dt) inside the range."""
    torch.manual_seed(0)
    pos_list, x_list, batch_list = [], [], []
    for i in range(batch_size):
        p = torch.rand(n_per_scene, 3)
        for d in range(3):
            p[:, d] = p[:, d] * (RANGE[d + 3] - RANGE[d]) + RANGE[d]
        pos_list.append(p)
        x_list.append(torch.rand(n_per_scene, 2))
        batch_list.append(torch.full((n_per_scene,), i, dtype=torch.long))
    return {
        "pos": torch.cat(pos_list).to(DEVICE),
        "x": torch.cat(x_list).to(DEVICE),
        "batch": torch.cat(batch_list).to(DEVICE),
    }


def test_voxelnext_create_model_hparams() -> None:
    model = create_model("voxelnext-openpcdet.nuscenes", task="detection")
    assert isinstance(model, VoxelNeXtDetection)
    assert model.in_channels == 5
    assert model.num_classes == 10
    assert model.grid_size == (1440, 1440, 40)
    assert model.sparse_shape == [41, 1440, 1440]
    assert len(model.head.heads_list) == 6
    assert model.head.class_groups == [[0], [1, 2], [3, 4], [5], [6, 7], [8, 9]]


def test_voxelnext_registered_variant() -> None:
    assert "voxelnext-openpcdet.nuscenes" in list_models("voxelnext*", task="detection")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="VoxelNeXt sparse backbone requires CUDA")
def test_voxelnext_forward_shapes() -> None:
    model = create_model("voxelnext-openpcdet.nuscenes", task="detection").to(DEVICE).eval()
    assert isinstance(model, VoxelNeXtDetection)
    data = _make_inputs()
    voxels, pos_voxel, num_points, vbatch = _voxelize(model, data, 10, 160000)
    with torch.no_grad():
        out = model(voxels, pos_voxel, num_points, vbatch)

    num_voxels = out["voxel_indices"].shape[0]
    assert out["voxel_indices"].shape == (num_voxels, 3)
    # 6 class groups; per-group hm channels follow the group sizes [1, 2, 2, 1, 2, 2].
    assert [h.shape[1] for h in out["hm"]] == [1, 2, 2, 1, 2, 2]
    for key, width in (("center", 2), ("center_z", 1), ("dim", 3), ("rot", 2), ("vel", 2)):
        assert all(t.shape == (num_voxels, width) for t in out[key])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="VoxelNeXt sparse backbone requires CUDA")
@pytest.mark.skipif(not WEIGHTS.exists(), reason="VoxelNeXt nuScenes weights not in local cache")
def test_voxelnext_pretrained_forward_decode() -> None:
    model = create_model("voxelnext-openpcdet.nuscenes", task="detection", pretrained=True).to(DEVICE).eval()
    assert isinstance(model, VoxelNeXtDetection)
    data = _make_inputs()
    voxels, pos_voxel, num_points, vbatch = _voxelize(model, data, 10, 160000)
    with torch.no_grad():
        out = model(voxels, pos_voxel, num_points, vbatch)
    det = model.decode(out)
    for key in ("boxes", "scores", "labels", "batch"):
        assert key in det
    assert det["boxes"].shape[1] == 7
    assert det["boxes"].shape[0] == det["scores"].shape[0] == det["labels"].shape[0] == det["batch"].shape[0]
    assert torch.isfinite(det["boxes"]).all()
    if det["labels"].numel():
        assert int(det["labels"].min()) >= 0 and int(det["labels"].max()) < model.num_classes
