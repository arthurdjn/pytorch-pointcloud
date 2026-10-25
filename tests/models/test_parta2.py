from typing import Tuple

import pytest
import torch
from torch import Tensor

from torch_pointcloud.models import create_model
from torch_pointcloud.models.parta2 import PartA2BoxHead, PartA2Detection, SparseUNetDecoder
from torch_pointcloud.ops.voxelization import hard_voxelize
from torch_pointcloud.utils.imports import _SPCONV_AVAILABLE

RANGE = (0.0, -40.0, -3.0, 70.4, 40.0, 1.0)
CUDA_SPCONV = pytest.mark.skipif(
    not (_SPCONV_AVAILABLE and torch.cuda.is_available()),
    reason="the sparse backbone needs spconv and CUDA",
)


def _voxelize(model: PartA2Detection, n_per_scene: int = 4000, batch_size: int = 2) -> Tuple[Tensor, ...]:
    """Random KITTI-like points (x, y, z, intensity) inside the detection range, voxelized for `model`."""
    torch.manual_seed(0)
    points = torch.rand(batch_size * n_per_scene, 4)
    for d in range(3):
        points[:, d] = points[:, d] * (RANGE[d + 3] - RANGE[d]) + RANGE[d]
    batch = torch.arange(batch_size).repeat_interleave(n_per_scene)
    voxels, voxel_indices, num_points = hard_voxelize(
        points.cuda(),
        batch.cuda(),
        model.voxel_size,
        model.point_cloud_range,
        5,
        40000,
    )
    return voxels, voxel_indices[:, 1:], num_points, voxel_indices[:, 0]


@pytest.mark.skipif(not _SPCONV_AVAILABLE, reason="spconv is not installed")
def test_parta2_create_model_hparams() -> None:
    model = create_model("part-a2.kitti.mmdet3d", task="detection")
    car = create_model("part-a2.kitti-car.mmdet3d", task="detection")
    assert isinstance(model, PartA2Detection) and isinstance(car, PartA2Detection)
    assert model.num_classes == 3 and car.num_classes == 1
    assert model.decoder.out_channels == 16


@CUDA_SPCONV
def test_parta2_decoder_restores_the_input_voxels() -> None:
    model = create_model("part-a2.kitti.mmdet3d", task="detection").cuda().eval()
    assert isinstance(model, PartA2Detection)
    voxels, pos_voxel, num_points, batch = _voxelize(model)
    with torch.no_grad():
        features, stages = model.encode(voxels, pos_voxel, num_points, batch)
        voxel_features = model.decoder(stages)
        bev_features = model.forward_features(voxels, pos_voxel, num_points, batch)
    assert voxel_features.shape == (voxels.shape[0], 16)
    assert features.shape == bev_features.shape == (2, model.num_features, 200, 176)


@CUDA_SPCONV
def test_parta2_forward_and_decode() -> None:
    model = create_model("part-a2.kitti.mmdet3d", task="detection", proposal_post_maxsize=16).cuda().eval()
    assert isinstance(model, PartA2Detection)
    with torch.no_grad():
        out = model(*_voxelize(model))
        det = model.decode(out)

    num_rois = out["boxes"].shape[0]
    assert 0 < num_rois <= 2 * 16
    assert out["rcnn_cls"].shape == (num_rois, 1)
    assert out["roi_class_scores"].shape == (num_rois, 3)
    assert det["boxes"].shape == (num_rois, 7)
    assert det["scores"].shape == det["labels"].shape == det["batch"].shape == (num_rois,)
    assert torch.isfinite(det["boxes"]).all()
    assert int(det["labels"].max()) < 3 and int(det["batch"].max()) <= 1


@CUDA_SPCONV
def test_parta2_box_head_predicts_every_roi_independently() -> None:
    """Permuting the RoIs permutes the predictions: the sparse grids of different RoIs never mix."""
    model = create_model("part-a2.kitti.mmdet3d", task="detection").cuda().eval()
    assert isinstance(model, PartA2Detection)
    torch.manual_seed(0)
    occupied = torch.rand(4, 14, 14, 14, 1, device="cuda") < 0.05
    seg_grid = torch.rand(4, 14, 14, 14, 16, device="cuda") * occupied
    part_grid = (torch.rand(4, 14, 14, 14, 4, device="cuda") + 0.1) * occupied
    order = torch.tensor([2, 0, 3, 1], device="cuda")
    with torch.no_grad():
        cls, reg = model.roi_head(seg_grid, part_grid)
        cls_permuted, reg_permuted = model.roi_head(seg_grid[order], part_grid[order])
    assert torch.allclose(cls[order], cls_permuted, atol=1e-5)
    assert torch.allclose(reg[order], reg_permuted, atol=1e-5)


@pytest.mark.skipif(not _SPCONV_AVAILABLE, reason="spconv is not installed")
def test_sparse_unet_decoder_rejects_mismatched_channels() -> None:
    with pytest.raises(ValueError, match="Decoder level 2"):
        SparseUNetDecoder(((64, 64, 64), (64, 64, 64), (32, 32, 16), (16, 16, 16)))
    with pytest.raises(ValueError, match="one entry per encoder stage"):
        SparseUNetDecoder(((64, 64, 64), (64, 64, 32)))


@CUDA_SPCONV
def test_parta2_box_head_handles_rois_without_points() -> None:
    head = PartA2BoxHead().cuda().eval()
    with torch.no_grad():
        cls, reg = head(torch.zeros(3, 14, 14, 14, 16, device="cuda"), torch.zeros(3, 14, 14, 14, 4, device="cuda"))
    assert cls.shape == (3, 1) and reg.shape == (3, 7)
    assert torch.isfinite(cls).all() and torch.isfinite(reg).all()
