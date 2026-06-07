from typing import Dict

import pytest
import torch
from torch import Tensor

from torch_pointcloud.models import create_model, list_models
from torch_pointcloud.models.pointpillars import PointPillars, PointPillarsMultiHead
from torch_pointcloud.utils.imports import _SPCONV_AVAILABLE

pytestmark = pytest.mark.skipif(not _SPCONV_AVAILABLE, reason="spconv is not installed")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANGE = (0.0, -39.68, -3.0, 69.12, 39.68, 1.0)


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


def test_pointpillars_forward_shapes() -> None:
    model = create_model("pointpillars-openpcdet.kitti", task="detection").to(DEVICE).eval()
    assert isinstance(model, PointPillars)
    data = _make_inputs()
    with torch.no_grad():
        out = model(data["x"], data["pos"], data["batch"])

    # KITTI 3-class BEV feature map at stride 2
    b, h, w = 2, 248, 216
    n_anchors = h * w * model.head.num_anchors_per_location
    assert out["cls_preds"].shape == (b, h, w, 18)
    assert out["box_preds"].shape == (b, h, w, 42)
    assert out["dir_cls_preds"].shape == (b, h, w, 12)
    assert out["batch_cls_preds"].shape == (b, n_anchors, 3)
    assert out["batch_box_preds"].shape == (b, n_anchors, 7)
    assert torch.isfinite(out["batch_box_preds"]).all()


def test_pointpillars_eval_is_deterministic() -> None:
    model = create_model("pointpillars-openpcdet.kitti", task="detection").to(DEVICE).eval()
    data = _make_inputs()
    with torch.no_grad():
        a = model(data["x"], data["pos"], data["batch"])
        b = model(data["x"], data["pos"], data["batch"])
    for key in ("cls_preds", "box_preds", "batch_box_preds"):
        if DEVICE == "cpu":
            assert torch.equal(a[key], b[key]), f"{key} not bit-identical on CPU"
        else:
            assert torch.allclose(a[key], b[key], atol=1e-5), f"{key} drifted beyond float-atomics on CUDA"


def test_pointpillars_registered_variant() -> None:
    assert "pointpillars-openpcdet.kitti" in list_models("pointpillars*", task="detection")


def test_pointpillars_create_model_hparams() -> None:
    model = create_model("pointpillars-openpcdet.kitti", task="detection")
    assert isinstance(model, PointPillars)
    assert model.in_channels == 4
    assert model.num_classes == 3
    assert model.grid_size == (432, 496, 1)
    assert model.head.anchors.shape == (248 * 216 * 6, 7)


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


def test_pp_multihead_forward_shapes() -> None:
    model = create_model("pointpillars-openpcdet-multihead.nuscenes", task="detection").to(DEVICE).eval()
    data = _make_nuscenes_inputs()
    with torch.no_grad():
        out = model(data["x"], data["pos"], data["batch"])

    # 6 RPN groups; BEV feature map 512/4 = 128, 10 classes x 2 rotations = 327680 anchors.
    assert len(out["cls_preds"]) == 6
    assert len(out["box_preds"]) == 6
    assert [m.tolist() for m in out["multihead_label_mapping"]] == [[1], [2, 3], [4, 5], [6], [7, 8], [9, 10]]
    assert out["cls_preds"][0].shape == (2, 32768, 1)
    assert out["cls_preds"][1].shape == (2, 65536, 2)
    assert out["box_preds"][0].shape == (2, 32768, 10)
    assert out["batch_box_preds"].shape == (2, 128 * 128 * 10 * 2, 9)
    assert torch.isfinite(out["batch_box_preds"]).all()


def test_pp_multihead_create_model_hparams() -> None:
    model = create_model("pointpillars-openpcdet-multihead.nuscenes", task="detection")
    assert isinstance(model, PointPillarsMultiHead)
    assert model.in_channels == 5
    assert model.num_classes == 10
    assert model.grid_size == (512, 512, 1)
    assert len(model.head.rpn_heads) == 6
    assert "pointpillars-openpcdet-multihead.nuscenes" in list_models("pointpillars*", task="detection")
