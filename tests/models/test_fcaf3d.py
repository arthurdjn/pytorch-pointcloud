import math
from typing import Tuple

import pytest
import torch
from torch import Tensor
from torch_geometric.nn import InstanceNorm

from torch_pointcloud.models import create_model
from torch_pointcloud.models.fcaf3d import FCAF3DDetection, FCAF3DOutput
from torch_pointcloud.ops.voxelization import lookup_coordinates
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
def test_fcaf3d_registered_variants() -> None:
    expected = {
        "fcaf3d.scannet.mmdet3d": (18, 6),
        "fcaf3d.sunrgbd.mmdet3d": (10, 8),
        "fcaf3d.s3dis-area5.mmdet3d": (5, 6),
    }
    for name, (num_classes, num_box_outputs) in expected.items():
        model = create_model(name, task="detection")
        assert isinstance(model, FCAF3DDetection)
        assert model.num_classes == num_classes and model.head.cls.out_features == num_classes
        assert model.head.box.out_features == num_box_outputs
        assert isinstance(model.encoder.norm1, InstanceNorm) and model.encoder.supports_norm_batch
        assert model.encoder.strides == (8, 16, 32, 64)
        assert model.num_features == 512


@SPCONV
def test_fcaf3d_decode_boxes_axis_aligned_face_distances() -> None:
    model = create_model("fcaf3d.scannet.mmdet3d", task="detection")
    assert isinstance(model, FCAF3DDetection)
    box = torch.tensor([[1.0, 3.0, 0.5, 0.5, 2.0, 0.0]])
    boxes = model.decode_boxes(box, torch.ones(1, 3))
    torch.testing.assert_close(boxes, torch.tensor([[2.0, 1.0, 0.0, 4.0, 1.0, 2.0, 0.0]]))


@SPCONV
def test_fcaf3d_decode_boxes_oriented_heading_code() -> None:
    model = create_model("fcaf3d.sunrgbd.mmdet3d", task="detection")
    assert isinstance(model, FCAF3DDetection)
    ratio, heading = 2.0, 0.3
    code = [math.log(ratio) * math.sin(2 * heading), math.log(ratio) * math.cos(2 * heading)]
    boxes = model.decode_boxes(torch.tensor([[0.5, 0.5, 1.0, 1.0, 0.25, 0.75, *code]]), torch.ones(1, 3))

    width = 3.0 / (1 + ratio)
    torch.testing.assert_close(boxes, torch.tensor([[1.0, 1.0, 1.25, width, width * ratio, 1.0, heading]]))


@SPCONV
def test_fcaf3d_decode_scores_by_class_and_centerness() -> None:
    model = create_model("fcaf3d.scannet.mmdet3d", task="detection")
    assert isinstance(model, FCAF3DDetection)
    cls = torch.full((2, 18), -10.0)
    cls[0, 4], cls[1, 7] = 2.0, 0.5
    out: FCAF3DOutput = {
        "cls_logits": cls,
        "centerness": torch.tensor([[1.0], [-1.0]]),
        "boxes": torch.rand(2, 7),
        "pos": torch.rand(2, 3),
        "batch": torch.tensor([0, 1]),
        "level": torch.tensor([0, 3]),
    }
    det = model.decode(out)

    assert det["labels"].tolist() == [4, 7]
    assert det["boxes"] is out["boxes"] and det["batch"] is out["batch"]
    torch.testing.assert_close(det["class_probs"], cls.sigmoid() * out["centerness"].sigmoid())
    torch.testing.assert_close(det["scores"], det["class_probs"].max(dim=1).values)


@SPCONV
def test_fcaf3d_reset_classifier() -> None:
    model = create_model("fcaf3d.scannet.mmdet3d", task="detection")
    assert isinstance(model, FCAF3DDetection)
    model.reset_classifier(3)
    assert model.num_classes == 3 and model.head.cls.out_features == 3


@CUDA_SPCONV
def test_fcaf3d_forward_and_decode_shapes() -> None:
    model = create_model("fcaf3d.sunrgbd.mmdet3d", task="detection").cuda().eval()
    assert isinstance(model, FCAF3DDetection)
    x, pos, batch = _scenes()
    with torch.no_grad():
        out = model(x, pos, batch)
        det = model.decode(out)

    m = len(out["cls_logits"])
    assert out["cls_logits"].shape == (m, 10) and out["centerness"].shape == (m, 1) and out["boxes"].shape == (m, 7)
    assert set(out["level"].unique().tolist()) == {0, 1, 2, 3}
    assert set(out["batch"].unique().tolist()) == {0, 1}
    assert (out["boxes"][:, 3:6] > 0).all()
    assert det["boxes"].shape == (m, 7) and det["class_probs"].shape == (m, 10)
    # Every location sits on the grid of its level.
    stride = torch.tensor(model.encoder.strides, device="cuda")[out["level"]].unsqueeze(1)
    cells = torch.round(out["pos"] / model.voxel_size).long()
    assert (cells % stride == 0).all()


@CUDA_SPCONV
def test_fcaf3d_scenes_are_independent_in_a_batch() -> None:
    model = create_model("fcaf3d.scannet.mmdet3d", task="detection").cuda().eval()
    assert isinstance(model, FCAF3DDetection)
    x, pos, batch = _scenes()
    first = batch == 0
    with torch.no_grad():
        both = model(x, pos, batch)
        alone = model(x[first], pos[first], batch[first])

    rows = both["batch"] == 0
    keys_both = torch.cat([both["level"][rows, None], torch.round(both["pos"][rows] / 0.01).long()], 1)
    keys_alone = torch.cat([alone["level"][:, None], torch.round(alone["pos"] / 0.01).long()], 1)
    index = lookup_coordinates(keys_both, keys_alone)
    assert (index >= 0).all() and len(keys_alone) == len(keys_both)
    torch.testing.assert_close(both["cls_logits"][rows][index], alone["cls_logits"], atol=1e-4, rtol=1e-4)


@CUDA_SPCONV
def test_fcaf3d_head_prunes_every_scene_to_max_voxels_per_scene() -> None:
    model = create_model("fcaf3d.scannet.mmdet3d", task="detection", max_voxels_per_scene=200).cuda().eval()
    assert isinstance(model, FCAF3DDetection)
    x, pos, batch = _scenes()
    with torch.no_grad():
        out = model(x, pos, batch)

    for level in range(3):
        for scene in range(2):
            assert int(((out["level"] == level) & (out["batch"] == scene)).sum()) <= 200
