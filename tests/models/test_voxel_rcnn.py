from typing import Dict, List, Tuple

import pytest
import torch
from torch import Tensor

from torch_pointcloud.losses import AnchorHeadLoss, RoIHeadLoss
from torch_pointcloud.models import create_model, model_info
from torch_pointcloud.models.voxel_rcnn import VoxelRCNNBoxHead, VoxelRCNNDetection
from torch_pointcloud.ops.voxelization import hard_voxelize
from torch_pointcloud.utils.data import DataKeys
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


def _make_stages(batch: List[int]) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
    """Three pooled stages of 32, 64 and 64 channels holding the same voxel centers around the origin."""
    torch.manual_seed(0)
    pos = torch.rand(len(batch), 3) - 0.5
    scene = torch.tensor(batch)
    features = [torch.randn(len(batch), channels) for channels in (32, 64, 64)]
    return features, [pos] * 3, [scene] * 3


def _make_head() -> VoxelRCNNBoxHead:
    return VoxelRCNNBoxHead(
        (32, 64, 64),
        grid_size=2,
        pool_channels=(((8, 8),), ((8, 8),), ((8, 8),)),
        pool_radii=((0.4,), (0.8,), (1.6,)),
        pool_num_neighbors=((4,), (4,), (4,)),
        shared_channels=(16,),
        cls_channels=(16,),
        reg_channels=(16,),
    ).eval()


@pytest.mark.skipif(not _PYG_LIB_AVAILABLE, reason="the voxel RoI pooling needs pyg-lib")
def test_voxel_rcnn_box_head_shapes() -> None:
    head = _make_head()
    features, pos, batch = _make_stages([0] * 20 + [1] * 20)
    rois = torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0], [0.0, 0.0, 0.0, 2.0, 1.0, 1.0, 0.5]])
    with torch.no_grad():
        rcnn_cls, rcnn_reg = head(rois, torch.tensor([0, 1]), features, pos, batch)
    assert rcnn_cls.shape == (2, 1)
    assert rcnn_reg.shape == (2, 7)


@pytest.mark.skipif(not _PYG_LIB_AVAILABLE, reason="the voxel RoI pooling needs pyg-lib")
def test_voxel_rcnn_box_head_pools_only_the_voxels_of_its_scene() -> None:
    head = _make_head()
    roi = torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]])
    features, pos, batch = _make_stages([1] * 20)
    with torch.no_grad():
        # The RoI of scene 0 sees no voxel, so it matches a RoI far from every voxel of scene 1.
        empty = head(roi, torch.tensor([0]), features, pos, batch)
        far = head(roi + torch.tensor([100.0, 0, 0, 0, 0, 0, 0]), torch.tensor([1]), features, pos, batch)
        near = head(roi, torch.tensor([1]), features, pos, batch)
    assert torch.equal(empty[0], far[0]) and torch.equal(empty[1], far[1])
    assert not torch.allclose(empty[0], near[0])


@pytest.mark.skipif(not _SPCONV_AVAILABLE, reason="spconv is not installed")
def test_voxel_rcnn_create_model_hparams() -> None:
    model = create_model("voxel-rcnn.kitti-car.openpcdet", task="detection")
    assert isinstance(model, VoxelRCNNDetection)
    assert model.num_classes == 1
    assert model.num_features == 256
    assert [pool.out_channels for pool in model.roi_head.grid_pools] == [32, 32, 32]


@pytest.mark.skipif(
    not (_SPCONV_AVAILABLE and _PYG_LIB_AVAILABLE and torch.cuda.is_available()),
    reason="the sparse backbone and the voxel RoI pooling need spconv, pyg-lib and CUDA",
)
def test_voxel_rcnn_forward_and_decode() -> None:
    model = create_model("voxel-rcnn.kitti-car.openpcdet", task="detection", proposal_post_maxsize=16)
    assert isinstance(model, VoxelRCNNDetection)
    model = model.cuda().eval()
    data = _make_inputs()
    points = torch.cat([data["pos"], data["x"]], dim=1)
    voxels, voxel_indices, num_points = hard_voxelize(
        points, data["batch"], model.voxel_size, model.point_cloud_range, 5, 40000
    )
    with torch.no_grad():
        out = model(voxels, voxel_indices[:, 1:], num_points, voxel_indices[:, 0])
        det = model.decode(out)

    num_rois = out["boxes"].shape[0]
    assert 0 < num_rois <= 2 * 16
    assert out["rcnn_cls"].shape == (num_rois, 1)
    assert det["boxes"].shape == (num_rois, 7)
    assert det["scores"].shape == det["labels"].shape == det["batch"].shape == (num_rois,)
    assert torch.isfinite(det["boxes"]).all()
    assert int(det["labels"].max()) == 0 and int(det["batch"].max()) <= 1


@pytest.mark.skipif(
    not (_SPCONV_AVAILABLE and _PYG_LIB_AVAILABLE and torch.cuda.is_available()),
    reason="the sparse backbone and the voxel RoI pooling need spconv, pyg-lib and CUDA",
)
def test_voxel_rcnn_training_forward_samples_rois_and_backpropagates() -> None:
    model = create_model("voxel-rcnn.kitti-car.openpcdet", task="detection", roi_num_samples=32)
    assert isinstance(model, VoxelRCNNDetection)
    model = model.cuda().train()
    data = _make_inputs()
    points = torch.cat([data["pos"], data["x"]], dim=1)
    voxels, voxel_indices, num_points = hard_voxelize(
        points, data["batch"], model.voxel_size, model.point_cloud_range, 5, 16000
    )
    gt_boxes = torch.tensor([[20.0, 0.0, -1.0, 3.9, 1.6, 1.56, 0.3], [35.0, 10.0, -1.0, 3.9, 1.6, 1.56, -1.0]]).cuda()
    gt_labels, gt_batch = torch.zeros(2, dtype=torch.long).cuda(), torch.tensor([0, 1]).cuda()
    inputs = (voxels, voxel_indices[:, 1:], num_points, voxel_indices[:, 0])
    with pytest.raises(ValueError, match="gt_boxes"):
        model(*inputs)
    out = model(*inputs, gt_boxes, gt_labels, gt_batch)

    assert out["rcnn_cls"].shape == (64, 1) and out["rcnn_reg"].shape == (64, 7)
    assert out["roi_batch"].tolist() == [0] * 32 + [1] * 32
    hparams = model_info("voxel-rcnn.kitti-car.openpcdet", task="detection")["hparams"]
    anchor_loss = AnchorHeadLoss(
        1,
        point_cloud_range=hparams["point_cloud_range"],
        voxel_size=hparams["voxel_size"],
        anchor_sizes=hparams["anchor_sizes"],
        anchor_bottom_heights=hparams["anchor_bottom_heights"],
        feature_map_stride=8,
        matched_thresholds=(0.6,),
        unmatched_thresholds=(0.45,),
    ).cuda()
    roi_loss = RoIHeadLoss(
        cls_foreground_threshold=0.75,
        cls_background_threshold=0.25,
        reg_foreground_threshold=0.55,
    ).cuda()
    target = {DataKeys.BOX: gt_boxes, DataKeys.LABEL: gt_labels, DataKeys.BATCH_BOX: gt_batch}
    roi_losses = roi_loss(
        out["rcnn_cls"].view(-1), out["rcnn_reg"], out["rois"], out["roi_target_boxes"], out["roi_iou"]
    )
    loss = anchor_loss(out, target)["loss"] + roi_losses["loss"]
    loss.backward()
    assert torch.isfinite(loss)
    assert model.roi_head.shared.lins[0].weight.grad is not None
