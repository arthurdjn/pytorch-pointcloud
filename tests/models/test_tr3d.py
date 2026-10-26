import math
from typing import Tuple

import pytest
import torch
from torch import Tensor

from torch_pointcloud.models import create_model
from torch_pointcloud.models.tr3d import TR3DDetection, TR3DOutput
from torch_pointcloud.utils.imports import _SPCONV_AVAILABLE

SPCONV = pytest.mark.skipif(not _SPCONV_AVAILABLE, reason="spconv is not installed")
CUDA_SPCONV = pytest.mark.skipif(
    not (_SPCONV_AVAILABLE and torch.cuda.is_available()),
    reason="the sparse encoder needs spconv and CUDA",
)


def _scenes(n_per_scene: int = 6000, batch_size: int = 2) -> Tuple[Tensor, Tensor, Tensor]:
    """Random coloured points on the floor and two walls of a 4 m room, per scene."""
    generator = torch.Generator().manual_seed(0)
    pos = torch.rand(batch_size * n_per_scene, 3, generator=generator) * torch.tensor([4.0, 4.0, 2.5])
    pos[0::3, 2] = 0.0
    pos[1::3, 0] = 0.0
    pos[2::3, 1] = 4.0
    x = torch.rand(batch_size * n_per_scene, 3, generator=generator)
    batch = torch.arange(batch_size).repeat_interleave(n_per_scene)
    return x.cuda(), pos.cuda(), batch.cuda()


@SPCONV
def test_tr3d_registered_variants() -> None:
    expected = {"tr3d.scannet.mmdet3d": (18, 6), "tr3d.sunrgbd.mmdet3d": (10, 8), "tr3d.s3dis-area5.mmdet3d": (5, 6)}
    for name, (num_classes, num_box_outputs) in expected.items():
        model = create_model(name, task="detection")
        assert isinstance(model, TR3DDetection)
        assert model.num_classes == num_classes and model.cls.out_features == num_classes
        assert model.box.out_features == num_box_outputs
        assert model.strides == (16, 32)
        assert model.num_features == 128


@SPCONV
def test_tr3d_decode_boxes_axis_aligned_offsets() -> None:
    model = create_model("tr3d.scannet.mmdet3d", task="detection")
    assert isinstance(model, TR3DDetection)
    box = torch.tensor([[0.5, -0.25, 0.0, math.log(2.0), 0.0, math.log(0.5)]])
    boxes = model.decode_boxes(box, torch.ones(1, 3))
    torch.testing.assert_close(boxes, torch.tensor([[1.5, 0.75, 1.0, 2.0, 1.0, 0.5, 0.0]]))


@SPCONV
def test_tr3d_decode_boxes_oriented_heading_code() -> None:
    model = create_model("tr3d.sunrgbd.mmdet3d", task="detection")
    assert isinstance(model, TR3DDetection)
    ratio, heading = 3.0, -0.4
    code = [math.log(ratio) * math.sin(2 * heading), math.log(ratio) * math.cos(2 * heading)]
    box = torch.tensor([[0.0, 0.0, 0.0, 0.0, math.log(3.0), math.log(0.5), *code]])
    boxes = model.decode_boxes(box, torch.ones(1, 3))

    # The height of an oriented box is the sum of the last two regressed sizes.
    torch.testing.assert_close(boxes, torch.tensor([[1.0, 1.0, 1.0, 1.0, 3.0, 3.5, heading]]))


@SPCONV
def test_tr3d_decode_scores_by_class() -> None:
    model = create_model("tr3d.scannet.mmdet3d", task="detection")
    assert isinstance(model, TR3DDetection)
    cls = torch.full((1, 18), -10.0)
    cls[0, 2] = 1.0
    out: TR3DOutput = {
        "cls_logits": cls,
        "boxes": torch.rand(1, 7),
        "pos": torch.rand(1, 3),
        "batch": torch.tensor([0]),
        "level": torch.tensor([1]),
    }
    det = model.decode(out)

    assert det["labels"].tolist() == [2]
    torch.testing.assert_close(det["scores"], torch.sigmoid(torch.tensor([1.0])))
    torch.testing.assert_close(det["class_probs"], cls.sigmoid())


@SPCONV
def test_tr3d_reset_classifier() -> None:
    model = create_model("tr3d.scannet.mmdet3d", task="detection")
    assert isinstance(model, TR3DDetection)
    model.reset_classifier(4)
    assert model.num_classes == 4 and model.cls.out_features == 4


@CUDA_SPCONV
def test_tr3d_forward_and_decode_shapes() -> None:
    model = create_model("tr3d.scannet.mmdet3d", task="detection").cuda().eval()
    assert isinstance(model, TR3DDetection)
    x, pos, batch = _scenes()
    with torch.no_grad():
        out = model(x, pos, batch)
        det = model.decode(out)

    m = len(out["cls_logits"])
    assert out["cls_logits"].shape == (m, 18) and out["boxes"].shape == (m, 7)
    assert set(out["level"].unique().tolist()) == {0, 1}
    assert set(out["batch"].unique().tolist()) == {0, 1}
    assert (out["boxes"][:, 3:6] > 0).all()
    assert det["boxes"].shape == (m, 7) and det["class_probs"].shape == (m, 18)
    stride = torch.tensor(model.strides, device="cuda")[out["level"]].unsqueeze(1)
    cells = torch.round(out["pos"] / model.voxel_size).long()
    assert (cells % stride == 0).all()
