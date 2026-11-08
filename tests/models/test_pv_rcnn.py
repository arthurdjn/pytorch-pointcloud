import math
import warnings
from typing import Dict

import pytest
import torch
from torch import Tensor

from torch_pointcloud.models import create_model
from torch_pointcloud.models.pv_rcnn import PVRCNNDetection, bilinear_interpolate, roi_grid_points
from torch_pointcloud.ops.voxelization import hard_voxelize
from torch_pointcloud.utils.imports import _PYG_LIB_AVAILABLE, _SPCONV_AVAILABLE

RANGE = (0.0, -40.0, -3.0, 70.4, 40.0, 1.0)


def _make_inputs(n_per_scene: int = 4000, batch_size: int = 2) -> Dict[str, Tensor]:
    """Random KITTI-like points (x, y, z, intensity) inside the detection range."""
    torch.manual_seed(0)
    pos = torch.rand(batch_size * n_per_scene, 3)
    for d in range(3):
        pos[:, d] = pos[:, d] * (RANGE[d + 3] - RANGE[d]) + RANGE[d]
    return {
        "pos": pos.cuda(),
        "x": torch.rand(batch_size * n_per_scene, 1).cuda(),
        "batch": torch.arange(batch_size).repeat_interleave(n_per_scene).cuda(),
    }


def test_roi_grid_points_fill_the_box_cells() -> None:
    rois = torch.tensor([[1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 0.0]])
    grid = roi_grid_points(rois, 2)
    assert grid.shape == (1, 8, 3)
    # x varies slowest, z fastest.
    assert grid[0, 0].tolist() == [0.5, 1.0, 1.5]
    assert grid[0, 1].tolist() == [0.5, 1.0, 4.5]
    assert grid[0, 7].tolist() == [1.5, 3.0, 4.5]


def test_roi_grid_points_rotate_with_the_heading() -> None:
    rois = torch.tensor([[0.0, 0.0, 0.0, 2.0, 2.0, 2.0, math.pi / 2]])
    grid = roi_grid_points(rois, 2)
    # The local (+0.5, -0.5) corner of the cells lands at (+0.5, +0.5) after a quarter turn.
    assert torch.allclose(grid[0, 4], torch.tensor([0.5, 0.5, -0.5]), atol=1e-6)


def test_bilinear_interpolate_interior_and_border() -> None:
    image = torch.arange(12.0).view(3, 4, 1)
    inside = bilinear_interpolate(image, torch.tensor([1.5]), torch.tensor([0.5]))
    assert inside.item() == pytest.approx((1 + 2 + 5 + 6) / 4)
    # Past the last column both clamped neighbors are the border cell and their weights cancel out.
    border = bilinear_interpolate(image, torch.tensor([3.5]), torch.tensor([0.0]))
    assert border.item() == 0.0


@pytest.mark.skipif(not _SPCONV_AVAILABLE, reason="spconv is not installed")
def test_pv_rcnn_create_model_hparams() -> None:
    model = create_model("pv-rcnn.kitti.mmdet3d", task="detection")
    assert isinstance(model, PVRCNNDetection)
    assert model.num_classes == 3
    assert model.keypoint_encoder.in_channels == 640
    assert model.keypoint_encoder.out_channels == 128


@pytest.mark.skipif(not _SPCONV_AVAILABLE, reason="spconv is not installed")
def test_pv_rcnn_without_dir_offset_warns() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        create_model("pv-rcnn.kitti.mmdet3d", task="detection")
    with pytest.warns(FutureWarning, match="PVRCNNDetection"):
        model = create_model("pv-rcnn.kitti.mmdet3d", task="detection", dir_offset=None)
    assert isinstance(model, PVRCNNDetection)
    assert model.dir_offset == 0.78539


@pytest.mark.skipif(
    not (_SPCONV_AVAILABLE and _PYG_LIB_AVAILABLE and torch.cuda.is_available()),
    reason="the sparse backbone and the keypoint encoder need spconv, pyg-lib and CUDA",
)
def test_pv_rcnn_forward_and_decode() -> None:
    model = create_model("pv-rcnn.kitti.mmdet3d", task="detection", num_keypoints=256, proposal_post_maxsize=16)
    assert isinstance(model, PVRCNNDetection)
    model = model.cuda().eval()
    data = _make_inputs()
    points = torch.cat([data["pos"], data["x"]], dim=1)
    voxels, voxel_indices, num_points = hard_voxelize(
        points, data["batch"], model.voxel_size, model.point_cloud_range, 5, 40000
    )
    with torch.no_grad():
        out = model(
            voxels,
            voxel_indices[:, 1:],
            num_points,
            voxel_indices[:, 0],
            data["x"],
            data["pos"],
            data["batch"],
        )
        det = model.decode(out)

    num_rois = out["boxes"].shape[0]
    assert 0 < num_rois <= 2 * 16
    assert out["rcnn_cls"].shape == (num_rois, 1)
    assert det["boxes"].shape == (num_rois, 7)
    assert det["scores"].shape == det["labels"].shape == det["batch"].shape == (num_rois,)
    assert torch.isfinite(det["boxes"]).all()
    assert int(det["labels"].max()) < 3 and int(det["batch"].max()) <= 1
