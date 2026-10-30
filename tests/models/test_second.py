import math
from typing import Dict

import pytest
import torch
from torch import Tensor

from torch_pointcloud.models import create_model, list_models
from torch_pointcloud.models.second import (
    SECONDDetection,
    SECONDIoUDetection,
    SECONDIoUHead,
    SECONDMultiHeadDetection,
)
from torch_pointcloud.ops.voxelization import hard_voxelize
from torch_pointcloud.utils.imports import _SPCONV_AVAILABLE

SPCONV = pytest.mark.skipif(not _SPCONV_AVAILABLE, reason="spconv is not installed")

# The sparse 3D backbone (spconv SubM/SparseConv3d) needs CUDA in this build.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANGE = (0.0, -40.0, -3.0, 70.4, 40.0, 1.0)


def _voxelize(
    model: SECONDDetection | SECONDIoUDetection | SECONDMultiHeadDetection,
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


@SPCONV
@pytest.mark.skipif(not torch.cuda.is_available(), reason="SECOND sparse backbone requires CUDA")
def test_second_forward_shapes() -> None:
    model = create_model("second.kitti.openpcdet", task="detection").to(DEVICE).eval()
    assert isinstance(model, SECONDDetection)
    data = _make_inputs()
    voxels, pos_voxel, num_points, vbatch = _voxelize(model, data, 5, 40000)
    with torch.no_grad():
        out = model(voxels, pos_voxel, num_points, vbatch)
        feat = model.forward_features(voxels, pos_voxel, num_points, vbatch)
    assert feat.shape[1] == model.num_features

    # KITTI 3-class BEV feature map at stride 8
    b, h, w = 2, 200, 176
    n_anchors = h * w * model.head.num_anchors_per_location
    assert out["cls"].shape == (b, h, w, 18)
    assert out["box"].shape == (b, h, w, 42)
    assert out["dir_cls"].shape == (b, h, w, 12)
    assert out["batch_cls"].shape == (b, n_anchors, 3)
    assert out["batch_box"].shape == (b, n_anchors, 7)
    assert torch.isfinite(out["batch_box"]).all()


@SPCONV
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


@SPCONV
def test_second_create_model_hparams() -> None:
    model = create_model("second.kitti.openpcdet", task="detection")
    assert isinstance(model, SECONDDetection)
    assert model.in_channels == 4
    assert model.num_classes == 3
    assert model.grid_size == (1408, 1600, 40)
    assert model.sparse_shape == [41, 1600, 1408]
    assert model.num_features == model.backbone.num_bev_features
    assert model.head.anchors.shape == (200 * 176 * 6, 7)


@SPCONV
def test_second_iou_create_model_hparams() -> None:
    model = create_model("second-iou.kitti.openpcdet", task="detection")
    assert isinstance(model, SECONDIoUDetection)
    assert model.num_features == 512
    assert model.roi_head.shared.lins[0].in_channels == 512 * 7 * 7
    assert model.roi_head.origin == (0.0, -40.0)
    assert model.roi_head.cell_size == pytest.approx((0.4, 0.4))
    assert model.grid_size == 7 and model.grid_size_3d == (1408, 1600, 40)


def test_second_iou_head_reads_only_the_map_of_each_roi() -> None:
    torch.manual_seed(0)
    head = SECONDIoUHead(8, 3, (0.0, 0.0), (1.0, 1.0), shared_channels=[16], cls_channels=[16]).eval()
    features = torch.randn(2, 8, 10, 10)
    rois = torch.tensor([[4.0, 4.0, 0.0, 3.0, 2.0, 1.0, 0.3], [5.0, 6.0, 0.0, 2.0, 4.0, 1.0, -1.0]]).repeat(2, 1)
    roi_batch = torch.tensor([0, 0, 1, 1])
    with torch.no_grad():
        iou = head(rois, roi_batch, features)
        changed = features.clone()
        changed[1] = torch.randn(8, 10, 10)
        iou_changed = head(rois, roi_batch, changed)
    assert iou.shape == (4, 1)
    assert torch.equal(iou[:2], iou_changed[:2])
    assert not torch.equal(iou[2:], iou_changed[2:])


def _pooling_head(grid_size: int, origin: tuple, cell_size: tuple) -> SECONDIoUHead:
    return SECONDIoUHead(2, grid_size, origin, cell_size, shared_channels=[4], cls_channels=[4])


def test_second_iou_head_pool_reads_the_box_corners() -> None:
    features = torch.arange(32.0).view(1, 2, 4, 4)
    rois = torch.tensor([[1.5, 1.5, 0.0, 3.0, 3.0, 1.0, 0.0]])
    out = _pooling_head(2, (0.0, 0.0), (1.0, 1.0)).pool(rois, torch.zeros(1, dtype=torch.long), features)
    assert out.shape == (1, 2, 2, 2)
    assert out[0, 0].tolist() == [[0.0, 3.0], [12.0, 15.0]]
    assert out[0, 1].tolist() == [[16.0, 19.0], [28.0, 31.0]]


def test_second_iou_head_pool_rotates_the_grid_and_uses_map_coordinates() -> None:
    features = torch.arange(16.0).view(1, 1, 4, 4).repeat(1, 2, 1, 1)
    # A quarter turn transposes the grid of a square box; origin and cell size map the box into pixels.
    rois = torch.tensor([[13.0, 23.0, 0.0, 6.0, 6.0, 1.0, math.pi / 2]])
    out = _pooling_head(2, (10.0, 20.0), (2.0, 2.0)).pool(rois, torch.zeros(1, dtype=torch.long), features)[0, 0]
    assert torch.allclose(out, torch.tensor([[3.0, 15.0], [0.0, 12.0]]), atol=1e-5)


def test_second_iou_head_pool_shears_a_rotated_non_square_box() -> None:
    # Channel 0 holds the map x of every pixel, channel 1 its map y.
    j, i = torch.meshgrid(torch.arange(9.0), torch.arange(9.0), indexing="xy")
    features = torch.stack([j, i]).unsqueeze(0)
    rois = torch.tensor([[4.0, 4.0, 0.0, 4.0, 2.0, 1.0, math.pi / 2]])
    out = _pooling_head(3, (0.0, 0.0), (1.0, 1.0)).pool(rois, torch.zeros(1, dtype=torch.long), features)[0]
    # The grid is rotated first and scaled along the map axes second: x spans the half length 2 (along v), y the
    # half width 1 (along u), where the turned 4 x 2 footprint itself spans 1 in x and 2 in y.
    assert torch.allclose(out[0], torch.tensor([[6.0] * 3, [4.0] * 3, [2.0] * 3]), atol=1e-5)
    assert torch.allclose(out[1], torch.tensor([[3.0, 4.0, 5.0]] * 3), atol=1e-5)


def test_second_iou_head_pool_reads_the_map_of_each_roi() -> None:
    features = torch.stack([torch.zeros(2, 4, 4), torch.ones(2, 4, 4)])
    rois = torch.tensor([[1.5, 1.5, 0.0, 3.0, 3.0, 1.0, 0.0]]).repeat(3, 1)
    out = _pooling_head(3, (0.0, 0.0), (1.0, 1.0)).pool(rois, torch.tensor([1, 0, 1]), features)
    assert out[:, 0].mean(dim=(1, 2)).tolist() == [1.0, 0.0, 1.0]


def test_second_iou_head_pool_is_zero_outside_the_map() -> None:
    rois = torch.tensor([[100.0, 100.0, 0.0, 2.0, 2.0, 1.0, 0.3]])
    out = _pooling_head(3, (0.0, 0.0), (1.0, 1.0)).pool(rois, torch.zeros(1, dtype=torch.long), torch.ones(1, 2, 4, 4))
    assert out.abs().max() == 0.0


def test_second_iou_head_pool_skips_a_map_without_rois() -> None:
    features = torch.stack([torch.zeros(2, 4, 4), torch.ones(2, 4, 4), 2 * torch.ones(2, 4, 4)])
    rois = torch.tensor([[1.5, 1.5, 0.0, 3.0, 3.0, 1.0, 0.0]]).repeat(2, 1)
    out = _pooling_head(3, (0.0, 0.0), (1.0, 1.0)).pool(rois, torch.tensor([2, 0]), features)
    assert out[:, 0].mean(dim=(1, 2)).tolist() == [2.0, 0.0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_second_iou_head_pool_reads_the_map_once_per_scene() -> None:
    # cuDNN's grid sampler makes its input contiguous, so a map broadcast to every RoI is copied once per RoI.
    features = torch.randn(1, 64, 200, 176, device="cuda")
    rois = torch.tensor([[30.0, 0.0, -1.0, 3.9, 1.6, 1.56, 0.3]], device="cuda").repeat(40, 1)
    head = SECONDIoUHead(64, 7, (0.0, -40.0), (0.4, 0.4), shared_channels=[16], cls_channels=[16]).cuda()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.memory_allocated()
    head.pool(rois, torch.zeros(40, dtype=torch.long, device="cuda"), features)
    torch.cuda.synchronize()
    assert torch.cuda.max_memory_allocated() - start < 2 * features.numel() * features.element_size()


@SPCONV
def test_second_iou_heads_score_the_proposals_of_a_bev_map() -> None:
    torch.manual_seed(0)
    model = create_model("second-iou.kitti.openpcdet", task="detection", proposal_post_maxsize=8).eval()
    assert isinstance(model, SECONDIoUDetection)
    features = torch.randn(2, model.num_features, 200, 176)
    with torch.no_grad():
        out = model.forward_head(features)
        rois, roi_scores, roi_labels, roi_batch = model.proposals(out)
        rcnn_cls = model.roi_head(rois, roi_batch, features)
        det = model.decode(
            {
                "rcnn_cls": rcnn_cls,
                "boxes": rois,
                "roi_labels": roi_labels,
                "roi_scores": roi_scores,
                "batch": roi_batch,
            }
        )
    assert 0 < rois.shape[0] <= 2 * 8
    assert torch.bincount(roi_batch).max() <= 8
    assert torch.equal(det["boxes"], rois) and torch.equal(det["labels"], roi_labels)
    assert torch.equal(det["scores"], rcnn_cls.sigmoid().view(-1))
    assert int(det["labels"].max()) < 3


@SPCONV
@pytest.mark.skipif(not torch.cuda.is_available(), reason="SECOND sparse backbone requires CUDA")
def test_second_iou_forward_and_decode() -> None:
    model = create_model("second-iou.kitti.openpcdet", task="detection", proposal_post_maxsize=16).to(DEVICE).eval()
    assert isinstance(model, SECONDIoUDetection)
    data = _make_inputs()
    voxels, pos_voxel, num_points, vbatch = _voxelize(model, data, 5, 40000)
    with torch.no_grad():
        out = model(voxels, pos_voxel, num_points, vbatch)
        det = model.decode(out)
    num_rois = out["boxes"].shape[0]
    assert 0 < num_rois <= 2 * 16
    assert out["rcnn_cls"].shape == (num_rois, 1)
    assert det["scores"].shape == det["labels"].shape == det["batch"].shape == (num_rois,)
    assert torch.isfinite(det["boxes"]).all()


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


@SPCONV
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


@SPCONV
def test_second_multihead_create_model_hparams() -> None:
    model = create_model("second-multihead.nuscenes.openpcdet", task="detection")
    assert isinstance(model, SECONDMultiHeadDetection)
    assert model.in_channels == 5
    assert model.num_classes == 10
    assert model.grid_size == (1024, 1024, 40)
    assert model.sparse_shape == [41, 1024, 1024]
    assert len(model.head.rpn_heads) == 6
    assert "second-multihead.nuscenes.openpcdet" in list_models("second*", task="detection")
