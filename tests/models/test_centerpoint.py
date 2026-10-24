from typing import Dict, Union

import pytest
import torch
from torch import Tensor

from torch_pointcloud.layers.center_head import CenterHead, CenterHeadOutput, SeparateHead
from torch_pointcloud.models import create_model, list_models
from torch_pointcloud.models.centerpoint import CenterPointDetection, CenterPointPillarsDetection
from torch_pointcloud.ops.voxelization import hard_voxelize
from torch_pointcloud.utils.imports import _SPCONV_AVAILABLE

# The sparse 3D backbone (spconv SubM/SparseConv3d) needs CUDA in this build.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANGE = (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0)
CLASS_GROUPS = [[0], [1, 2], [3, 4], [5], [6, 7], [8, 9]]


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


def _voxelize(
    model: Union[CenterPointDetection, CenterPointPillarsDetection],
    data: Dict[str, Tensor],
    max_num_points: int,
    max_num_voxels: int,
) -> tuple:
    """Voxelize raw packed points the way the registered `HardVoxelize` transform + collate would."""
    points = torch.cat([data["pos"], data["x"]], dim=1)
    voxels, voxel_indices, num_points = hard_voxelize(
        points, data["batch"], model.voxel_size, model.point_cloud_range, max_num_points, max_num_voxels
    )
    return voxels, voxel_indices[:, 1:], num_points, voxel_indices[:, 0]


def test_centerpoint_registered_variants() -> None:
    names = list_models("centerpoint*", task="detection")
    assert "centerpoint-voxel01.nuscenes.mmdet3d" in names
    assert "centerpoint-voxel0075.nuscenes.mmdet3d" in names
    assert "centerpoint-pillar02.nuscenes.mmdet3d" in names


@pytest.mark.skipif(not _SPCONV_AVAILABLE, reason="spconv is not installed")
def test_centerpoint_create_model_hparams() -> None:
    model = create_model("centerpoint-voxel01.nuscenes.mmdet3d", task="detection")
    assert isinstance(model, CenterPointDetection)
    assert model.in_channels == 5
    assert model.num_classes == 10
    assert model.grid_size == (1024, 1024, 40)
    assert model.sparse_shape == [41, 1024, 1024]
    assert model.num_features == model.backbone.num_bev_features == 512
    assert len(model.head.heads) == 6
    assert model.head.class_groups == CLASS_GROUPS
    assert all(head.vel is not None for head in model.head.heads)

    model = create_model("centerpoint-voxel0075.nuscenes.mmdet3d", task="detection")
    assert isinstance(model, CenterPointDetection)
    assert model.grid_size == (1440, 1440, 40)


def test_centerpoint_pillars_create_model_hparams() -> None:
    model = create_model("centerpoint-pillar02.nuscenes.mmdet3d", task="detection")
    assert isinstance(model, CenterPointPillarsDetection)
    assert model.grid_size == (512, 512, 1)
    assert model.num_features == model.backbone.num_bev_features == 384
    assert model.feature_map_stride == 4
    assert len(model.head.heads) == 6


@pytest.mark.skipif(not (_SPCONV_AVAILABLE and torch.cuda.is_available()), reason="sparse backbone requires CUDA")
def test_centerpoint_forward_and_decode() -> None:
    model = create_model("centerpoint-voxel01.nuscenes.mmdet3d", task="detection").to(DEVICE).eval()
    assert isinstance(model, CenterPointDetection)
    data = _make_inputs()
    voxels, pos_voxel, num_points, vbatch = _voxelize(model, data, 10, 120000)
    with torch.no_grad():
        feat = model.forward_features(voxels, pos_voxel, num_points, vbatch)
        out = model(voxels, pos_voxel, num_points, vbatch)
        det = model.decode(out, top_k=50)

    # BEV feature map 1024 / 8 = 128; one heatmap per class group.
    assert feat.shape == (2, 512, 128, 128)
    assert [t.shape[1] for t in out["heatmap"]] == [len(g) for g in CLASS_GROUPS]
    assert out["center"][0].shape == (2, 2, 128, 128)
    assert out["vel"][0].shape == (2, 2, 128, 128)
    assert torch.isfinite(out["heatmap"][0]).all()

    # 50 peaks per group and scene; labels are global class ids.
    assert det["boxes"].shape == (2 * 6 * 50, 7)
    assert det["velocity"].shape == (2 * 6 * 50, 2)
    assert det["scores"].shape == det["labels"].shape == det["batch"].shape == (600,)
    assert det["labels"].dtype == torch.long
    assert int(det["labels"].max()) < 10 and int(det["batch"].max()) == 1
    assert torch.isfinite(det["boxes"]).all()


@pytest.mark.skipif(not _SPCONV_AVAILABLE, reason="spconv is not installed")
def test_centerpoint_pillars_forward_and_decode() -> None:
    model = create_model("centerpoint-pillar02.nuscenes.mmdet3d", task="detection").to(DEVICE).eval()
    assert isinstance(model, CenterPointPillarsDetection)
    data = _make_inputs(n_per_scene=2000)
    voxels, pos_voxel, num_points, vbatch = _voxelize(model, data, 20, 40000)
    with torch.no_grad():
        out = model(voxels, pos_voxel, num_points, vbatch)
        det = model.decode(out, top_k=20)

    # Pillar grid 512 at stride 4 -> 128.
    assert out["heatmap"][1].shape == (2, 2, 128, 128)
    assert det["boxes"].shape == (2 * 6 * 20, 7)
    assert det["velocity"].shape == (2 * 6 * 20, 2)
    assert torch.isfinite(det["boxes"]).all()


def test_separate_head_optional_branches() -> None:
    head = SeparateHead(8, 2)
    out = head(torch.rand(1, 8, 4, 4))
    assert set(out) == {"heatmap", "center", "center_z", "dim", "rot"}
    assert out["heatmap"].shape == (1, 2, 4, 4)
    final = head.heatmap[-1]
    assert isinstance(final, torch.nn.Conv2d) and final.bias is not None
    assert torch.allclose(final.bias, torch.full((2,), -2.19))

    head = SeparateHead(8, 2, use_velocity=True, use_iou=True)
    out = head(torch.rand(1, 8, 4, 4))
    assert out["vel"].shape == (1, 2, 4, 4) and out["iou"].shape == (1, 1, 4, 4)


def test_center_head_decode_velocity_and_rectifier() -> None:
    torch.manual_seed(0)
    head = CenterHead(
        8,
        [[0], [1, 2]],
        feature_map_stride=1,
        voxel_size=(1.0, 1.0, 1.0),
        point_cloud_range=(-4.0, -4.0, -2.0, 4.0, 4.0, 2.0),
        use_velocity=True,
        use_iou=True,
    ).eval()
    out = head(torch.rand(2, 8, 8, 8))
    assert len(out["heatmap"]) == 2 and out["heatmap"][1].shape == (2, 2, 8, 8)
    assert "vel" in out and "iou" in out

    det = head.decode(out, top_k=5)
    assert det["boxes"].shape == (2 * 2 * 5, 7)
    assert det["velocity"].shape == (20, 2)
    # Candidates come group by group, scene by scene: 5 per scene of group 0, then 5 per scene of group 1.
    assert det["labels"][:10].tolist() == [0] * 10
    assert set(det["labels"][10:].tolist()) <= {1, 2}
    assert det["batch"].tolist() == [0] * 5 + [1] * 5 + [0] * 5 + [1] * 5
    # The decoded centers lie inside the 8 x 8 grid, give or take the (unbounded) sub-cell offsets.
    margin = max(float(t.abs().max()) for t in out["center"])
    assert (det["boxes"][:, :2].abs() <= 4.0 + margin).all()

    with pytest.raises(ValueError, match="iou_rectifier"):
        head.decode(out, iou_rectifier=(0.5, 0.5))
    rescored = head.decode(out, top_k=5, iou_rectifier=(0.5, 0.5, 0.5))
    assert ((rescored["scores"] >= 0) & (rescored["scores"] <= 1)).all()
    assert not torch.allclose(rescored["scores"], det["scores"])

    filtered = head.decode(out, top_k=5, score_threshold=1.0)
    assert filtered["boxes"].shape == (0, 7) and filtered["velocity"].shape == (0, 2)


def test_center_head_decode_places_a_peak_on_a_non_square_map() -> None:
    head = CenterHead(
        8,
        [[0]],
        feature_map_stride=2,
        voxel_size=(0.5, 0.5, 1.0),
        point_cloud_range=(-10.0, -20.0, -3.0, 10.0, 20.0, 1.0),
        use_velocity=True,
    )
    height, width = 6, 4
    heatmap = torch.full((1, 1, height, width), -10.0)
    heatmap[0, 0, 4, 1] = 10.0
    out: CenterHeadOutput = {
        "heatmap": [heatmap],
        "center": [torch.tensor([0.25, 0.75]).view(1, 2, 1, 1).expand(1, 2, height, width)],
        "center_z": [torch.full((1, 1, height, width), 0.5)],
        "dim": [torch.log(torch.tensor([2.0, 1.0, 1.5])).view(1, 3, 1, 1).expand(1, 3, height, width)],
        "rot": [torch.tensor([0.0, 1.0]).view(1, 2, 1, 1).expand(1, 2, height, width)],
        "vel": [torch.tensor([1.0, -2.0]).view(1, 2, 1, 1).expand(1, 2, height, width)],
    }
    det = head.decode(out, top_k=1)
    # Column 1 and row 4 of a map of 2 x 0.5 m cells, shifted by the sub-cell center, from the range corner.
    expected = torch.tensor([[-10.0 + 1.25, -20.0 + 4.75, 0.5, 2.0, 1.0, 1.5, torch.pi / 2]])
    assert torch.allclose(det["boxes"], expected, atol=1e-5)
    assert torch.allclose(det["velocity"], torch.tensor([[1.0, -2.0]]))


def test_center_head_decode_rejects_a_rectifier_without_iou_maps() -> None:
    head = CenterHead(
        8,
        [[0]],
        feature_map_stride=1,
        voxel_size=(1.0, 1.0, 1.0),
        point_cloud_range=(-4.0, -4.0, -2.0, 4.0, 4.0, 2.0),
    ).eval()
    out = head(torch.rand(1, 8, 8, 8))
    with pytest.raises(ValueError, match="use_iou"):
        head.decode(out, iou_rectifier=(0.5,))


def test_centerpoint_rejects_class_groups_that_miss_a_class() -> None:
    with pytest.raises(ValueError, match="every class index below 3"):
        CenterPointDetection(num_classes=3, head_class_groups=[[0], [1]])
    with pytest.raises(ValueError, match="every class index below 3"):
        CenterPointPillarsDetection(num_classes=3, head_class_groups=[[0, 1], [1, 2]])
