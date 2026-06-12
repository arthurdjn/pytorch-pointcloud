from pathlib import Path
from typing import Any, Dict

import pytest
import torch
from torch import Tensor

import torch_pointcloud.models.pointrcnn  # noqa: F401  (registers the model)
from torch_pointcloud.config import DATA_DIR, MODELS_DIR
from torch_pointcloud.models import create_model, list_models
from torch_pointcloud.models.pointrcnn import (
    PointHeadBox,
    PointRCNNDetection,
    decode_point_residuals,
    rotate_points_along_z,
)
from torch_pointcloud.utils.box3d import decode_box_residuals
from torch_pointcloud.utils.imports import _TORCH_CLUSTER_AVAILABLE, _TORCH_SCATTER_AVAILABLE

pytestmark = [
    pytest.mark.skipif(not _TORCH_CLUSTER_AVAILABLE, reason="torch-cluster is not installed"),
    pytest.mark.skipif(not _TORCH_SCATTER_AVAILABLE, reason="torch-scatter is not installed"),
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANGE = (0.0, -40.0, -3.0, 70.4, 40.0, 1.0)
KITTI_MEAN_SIZES = [[3.9, 1.6, 1.56], [0.8, 0.6, 1.73], [1.76, 0.6, 1.73]]


def _make_pointrcnn(**overrides: Any) -> PointRCNNDetection:
    """A small two-stage PointRCNN (few SA points / ROIs) so the forward runs fast."""
    kwargs: Dict[str, Any] = dict(
        in_channels=4,
        num_classes=3,
        mean_sizes=KITTI_MEAN_SIZES,
        sa_channels=[[[16, 16, 32], [32, 32, 64]], [[64, 64, 128], [64, 96, 128]]],
        sa_npoints=[512, 128],
        sa_radii=[[0.1, 0.5], [0.5, 1.0]],
        sa_num_neighbors=[[16, 32], [16, 32]],
        fp_channels=[[256, 256], [128, 128]],
        point_cls_channels=[128],
        point_reg_channels=[128],
        roi_sa_channels=[[64, 64, 128], [128, 128, 256]],
        roi_sa_npoints=[64, -1],
        roi_sa_radii=[0.2, 100.0],
        roi_sa_num_neighbors=[16, 16],
        roi_xyz_up_channels=[128, 128],
        roi_cls_channels=[128],
        roi_reg_channels=[128],
        num_sampled_points=64,
        nms_post_maxsize=16,
    )
    kwargs.update(overrides)
    return PointRCNNDetection(**kwargs)


def _make_inputs(n_per_scene: int = 4096, batch_size: int = 2, in_channels: int = 1) -> Dict[str, Tensor]:
    """Random KITTI-like points inside the point-cloud range, packed across scenes."""
    torch.manual_seed(0)
    pos_list, x_list, batch_list = [], [], []
    for i in range(batch_size):
        p = torch.rand(n_per_scene, 3)
        for d in range(3):
            p[:, d] = p[:, d] * (RANGE[d + 3] - RANGE[d]) + RANGE[d]
        pos_list.append(p)
        x_list.append(torch.rand(n_per_scene, in_channels))
        batch_list.append(torch.full((n_per_scene,), i, dtype=torch.long))
    return {
        "pos": torch.cat(pos_list).to(DEVICE),
        "x": torch.cat(x_list).to(DEVICE),
        "batch": torch.cat(batch_list).to(DEVICE),
    }


def test_pointrcnn_forward_shapes() -> None:
    model = _make_pointrcnn().to(DEVICE).eval()
    data = _make_inputs()
    with torch.no_grad():
        out = model(data["x"], data["pos"], data["batch"])
    num_rois = 2 * model.nms_post_maxsize
    assert out["rcnn_cls"].shape == (num_rois, 1)
    assert out["boxes"].shape == (num_rois, 7)
    assert out["roi_labels"].shape == (num_rois,)
    assert out["batch"].shape == (num_rois,)
    assert out["batch"].max().item() == 1
    assert torch.isfinite(out["boxes"]).all()


def test_pointrcnn_decode_packed_detections() -> None:
    model = _make_pointrcnn().to(DEVICE).eval()
    data = _make_inputs()
    with torch.no_grad():
        out = model(data["x"], data["pos"], data["batch"])
        det = model.decode(out, score_threshold=0.0)
    for key in ("boxes", "scores", "labels", "batch"):
        assert key in det
    assert det["boxes"].shape[1] == 7
    assert det["boxes"].shape[0] == det["scores"].shape[0] == det["labels"].shape[0] == det["batch"].shape[0]
    assert det["labels"].numel() == 0 or det["labels"].min().item() >= 0
    assert det["labels"].numel() == 0 or det["labels"].max().item() < model.num_classes


def test_pointrcnn_eval_is_deterministic() -> None:
    model = _make_pointrcnn().to(DEVICE).eval()
    data = _make_inputs()
    with torch.no_grad():
        a = model(data["x"], data["pos"], data["batch"])
        b = model(data["x"], data["pos"], data["batch"])
    if DEVICE == "cpu":
        assert torch.equal(a["boxes"], b["boxes"])
    else:
        assert torch.allclose(a["boxes"], b["boxes"], atol=1e-4)


def test_pointrcnn_backbone_outputs_per_point_features() -> None:
    model = _make_pointrcnn().to(DEVICE).eval()
    data = _make_inputs(n_per_scene=2048)
    with torch.no_grad():
        x, pos, batch = model.forward_features(data["x"], data["pos"], data["batch"])
    # feature propagation returns one feature per input point
    assert x.shape == (data["pos"].shape[0], 128)
    assert pos.shape == data["pos"].shape
    assert torch.equal(batch, data["batch"])


def test_decode_point_residuals_shape() -> None:
    mean = torch.tensor(KITTI_MEAN_SIZES)
    enc = torch.randn(50, 8)
    pos = torch.randn(50, 3)
    cls = torch.randint(1, 4, (50,))
    boxes = decode_point_residuals(enc, pos, cls, mean)
    assert boxes.shape == (50, 7)


def test_decode_box_residuals_zero_residual_recovers_anchor() -> None:
    # heading within (-pi, pi) so the sincos branch's atan2 wrap-around stays a no-op
    anchors = torch.cat([torch.randn(20, 3), torch.rand(20, 3) + 0.5, torch.rand(20, 1) * 6 - 3], dim=1)
    decoded = decode_box_residuals(torch.zeros(20, 7), anchors)
    assert torch.allclose(decoded, anchors, atol=1e-6)
    decoded_sincos = decode_box_residuals(torch.zeros(20, 8), anchors, angle_by_sincos=True)
    assert torch.allclose(decoded_sincos, anchors, atol=1e-6)


def test_rotate_points_along_z_round_trip() -> None:
    pts = torch.randn(30, 1, 3)
    angle = torch.rand(30) * 6.0 - 3.0
    rotated = rotate_points_along_z(pts, angle)
    restored = rotate_points_along_z(rotated, -angle)
    assert torch.allclose(restored, pts, atol=1e-5)


def test_pointrcnn_reset_classifier_unsupported() -> None:
    model = _make_pointrcnn()
    with pytest.raises(NotImplementedError):
        model.reset_classifier(num_classes=5)


def test_pointrcnn_bad_mean_sizes_shape() -> None:
    with pytest.raises(ValueError, match="mean_sizes"):
        _make_pointrcnn(mean_sizes=[[1.0, 1.0, 1.0]])


def test_pointrcnn_mean_sizes_not_persisted() -> None:
    model = _make_pointrcnn()
    assert "mean_sizes" not in model.state_dict()
    assert model.mean_sizes.shape == (3, 3)


def test_point_head_box_returns_scores_logits_boxes() -> None:
    head = (
        PointHeadBox(64, 3, cls_channels=[32], reg_channels=[32], mean_sizes=torch.tensor(KITTI_MEAN_SIZES))
        .to(DEVICE)
        .eval()
    )
    x = torch.randn(100, 64, device=DEVICE)
    pos = torch.randn(100, 3, device=DEVICE)
    with torch.no_grad():
        scores, logits, boxes = head(x, pos)
    assert scores.shape == (100,)
    assert logits.shape == (100, 3)
    assert boxes.shape == (100, 7)
    assert (scores >= 0).all() and (scores <= 1).all()


def test_pointrcnn_registered_variant() -> None:
    assert "pointrcnn-openpcdet.kitti" in list_models("pointrcnn*", task="detection")


def test_pointrcnn_create_model_hparams() -> None:
    model = create_model("pointrcnn-openpcdet.kitti", task="detection")
    assert isinstance(model, PointRCNNDetection)
    assert model.in_channels == 4
    assert model.num_classes == 3
    assert model.encoder.out_channels == 1024
    assert model.nms_post_maxsize == 100
    assert len(model.encoder.sa_blocks) == 4
    assert len(model.decoder.fp_blocks) == 4
    assert len(model.roi_head.sa_modules) == 3


_WEIGHTS = Path(MODELS_DIR, "pointrcnn", "pointrcnn-openpcdet.kitti.pt")
_KITTI_DIR = Path(DATA_DIR, "KITTI")


@pytest.mark.skipif(not _WEIGHTS.exists(), reason="PointRCNN KITTI weights not downloaded")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="PointRCNN backbone exercised on CUDA")
def test_pointrcnn_pretrained_loads_strict() -> None:
    model = create_model("pointrcnn-openpcdet.kitti", task="detection", pretrained=True).to(DEVICE).eval()
    assert isinstance(model, PointRCNNDetection)
    data = _make_inputs(n_per_scene=16384, batch_size=1)
    with torch.no_grad():
        out = model(data["x"], data["pos"], data["batch"])
        det = model.decode(out)
    assert out["boxes"].shape == (model.nms_post_maxsize, 7)
    assert det["boxes"].shape[1] == 7
    assert torch.isfinite(out["boxes"]).all()
