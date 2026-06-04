import math

import torch
from torch import Tensor

import torch_pointcloud.transforms as T
import torch_pointcloud.transforms.functional as F


def _box(heading: float = 0.0, cls: float = 1.0) -> Tensor:
    return torch.tensor([[1.0, 0.5, 0.3, 0.4, 0.3, 0.2, heading, cls]])


def test_flip_boxes_yz_plane() -> None:
    box = _box(heading=0.2)
    flipped = F.flip_boxes(box, axis=0)
    assert torch.allclose(flipped[:, 0], -box[:, 0])
    assert torch.allclose(flipped[:, 6], math.pi - box[:, 6])
    assert torch.allclose(flipped[:, 3:6], box[:, 3:6])


def test_flip_boxes_xz_plane() -> None:
    box = _box(heading=0.2)
    flipped = F.flip_boxes(box, axis=1)
    assert torch.allclose(flipped[:, 1], -box[:, 1])
    assert torch.allclose(flipped[:, 6], -box[:, 6])


def test_rotate_boxes_decrements_heading() -> None:
    box = _box(heading=0.2)
    rotation = F.rotation_matrix(0.5, axis=2)
    rotated = F.rotate_boxes(box, rotation, 0.5)
    assert torch.allclose(rotated[:, 6], box[:, 6] - 0.5)
    assert torch.allclose(rotated[:, 3:6], box[:, 3:6])


def test_scale_boxes_centers_and_sizes() -> None:
    box = _box(heading=0.2)
    scaled = F.scale_boxes(box, 2.0)
    assert torch.allclose(scaled[:, 0:6], box[:, 0:6] * 2.0)
    assert torch.allclose(scaled[:, 6], box[:, 6])


def test_points_in_oriented_box_axis_aligned() -> None:
    box = _box(heading=0.0)
    pts = torch.tensor([[1.0, 0.5, 0.3], [1.4, 0.5, 0.3], [2.0, 0.5, 0.3]])
    mask = F.points_in_oriented_box(pts, box[0])
    assert mask.tolist() == [True, True, False]


def test_points_in_oriented_box_yaw_aware() -> None:
    box = _box(heading=math.pi / 2)
    corner = torch.tensor([[1.0 + 0.3, 0.5 + 0.4, 0.3]])
    outside = torch.tensor([[1.0 + 0.4, 0.5, 0.3]])
    assert F.points_in_oriented_box(corner, box[0]).tolist() == [True]
    assert F.points_in_oriented_box(outside, box[0]).tolist() == [False]


def test_random_flip_boxes_preserves_membership() -> None:
    gen = torch.Generator().manual_seed(0)
    box = _box(heading=0.2, cls=2.0)
    face = torch.tensor([[1.4, 0.5, 0.3]])
    data = {"pos": face.clone(), "box": box.clone()}
    out = T.RandomFlipBoxes(axes=(0,), p=1.0, generator=gen)(data)
    assert F.points_in_oriented_box(out["pos"], out["box"][0]).item()


def test_random_rotate_boxes_preserves_membership() -> None:
    gen = torch.Generator().manual_seed(0)
    box = _box(heading=0.2, cls=2.0)
    face = torch.tensor([[1.4, 0.5, 0.3]])
    data = {"pos": face.clone(), "box": box.clone()}
    out = T.RandomRotateBoxes(angle_range=(25.0, 25.0), p=1.0, generator=gen)(data)
    assert F.points_in_oriented_box(out["pos"], out["box"][0]).item()


def test_random_scale_boxes_preserves_membership() -> None:
    gen = torch.Generator().manual_seed(0)
    box = _box(heading=0.2, cls=2.0)
    face = torch.tensor([[1.4, 0.5, 0.3]])
    data = {"pos": face.clone(), "box": box.clone()}
    out = T.RandomScaleBoxes(scale_range=(1.3, 1.3), p=1.0, generator=gen)(data)
    assert F.points_in_oriented_box(out["pos"], out["box"][0]).item()


def test_random_rotate_boxes_p_zero_is_noop() -> None:
    box = _box(heading=0.2)
    pos = torch.tensor([[1.0, 0.5, 0.3]])
    data = {"pos": pos.clone(), "box": box.clone()}
    out = T.RandomRotateBoxes(p=0.0)(data)
    assert torch.allclose(out["box"], box)
    assert torch.allclose(out["pos"], pos)


def test_random_rotate_boxes_votes_stay_consistent() -> None:
    gen = torch.Generator().manual_seed(0)
    box = _box(heading=0.2)
    pos = torch.tensor([[1.0, 0.5, 0.3]])
    vote = (box[:, 0:3] - pos).repeat(1, 3)
    data = {"pos": pos.clone(), "box": box.clone(), "vote_label": vote.clone()}
    out = T.RandomRotateBoxes(angle_range=(40.0, 40.0), p=1.0, vote_key="vote_label", generator=gen)(data)
    expected = out["box"][:, 0:3] - out["pos"]
    assert torch.allclose(out["vote_label"][:, 0:3], expected, atol=1e-5)
    assert torch.allclose(out["vote_label"][:, 3:6], expected, atol=1e-5)


def test_random_scale_boxes_votes_stay_consistent() -> None:
    gen = torch.Generator().manual_seed(0)
    box = _box(heading=0.2)
    pos = torch.tensor([[1.0, 0.5, 0.3]])
    vote = (box[:, 0:3] - pos).repeat(1, 3)
    data = {"pos": pos.clone(), "box": box.clone(), "vote_label": vote.clone()}
    out = T.RandomScaleBoxes(scale_range=(1.3, 1.3), p=1.0, vote_key="vote_label", generator=gen)(data)
    expected = out["box"][:, 0:3] - out["pos"]
    assert torch.allclose(out["vote_label"][:, 0:3], expected, atol=1e-5)


def test_generate_vote_labels_marks_in_and_out() -> None:
    box = _box(heading=0.0)
    pts = torch.tensor([[1.0, 0.5, 0.3], [5.0, 5.0, 5.0]])
    data = {"pos": pts, "box": box.clone()}
    out = T.GenerateVoteLabels(oriented=True, gt_vote_factor=3)(data)
    assert out["vote_label"].shape == (2, 9)
    assert out["vote_label_mask"].tolist() == [1, 0]
    assert torch.allclose(out["vote_label"][0, 0:3], box[0, 0:3] - pts[0])
    assert torch.allclose(out["vote_label"][0, 0:3], out["vote_label"][0, 3:6])
    assert torch.allclose(out["vote_label"][0, 0:3], out["vote_label"][0, 6:9])
    assert torch.allclose(out["vote_label"][1], torch.zeros(9))


def test_generate_vote_labels_oriented_vs_axis_aligned() -> None:
    box = _box(heading=math.pi / 4)
    corner = torch.tensor([[1.0 + 0.38, 0.5 + 0.28, 0.3]])
    data_axis = {"pos": corner.clone(), "box": box.clone()}
    data_oriented = {"pos": corner.clone(), "box": box.clone()}
    out_axis = T.GenerateVoteLabels(oriented=False)(data_axis)
    out_oriented = T.GenerateVoteLabels(oriented=True)(data_oriented)
    assert out_axis["vote_label_mask"].item() == 1
    assert out_oriented["vote_label_mask"].item() == 0


def test_encode_votenet_targets_shapes_and_roundtrip() -> None:
    mean = torch.ones(10, 3) * 0.5
    box = _box(heading=0.6, cls=2.0)
    data = {"box": box.clone()}
    out = T.EncodeVoteNetTargets(num_heading_bin=12, mean_size_arr=mean, max_num_obj=64)(data)
    assert out["center_label"].shape == (64, 3)
    assert out["heading_class_label"].shape == (64,)
    assert out["heading_residual_label"].shape == (64,)
    assert out["size_class_label"].shape == (64,)
    assert out["size_residual_label"].shape == (64, 3)
    assert out["sem_cls_label"].shape == (64,)
    assert out["box_label_mask"].shape == (64,)
    assert out["box_label_mask"].sum().item() == 1
    assert out["size_class_label"][0].item() == 2
    assert out["sem_cls_label"][0].item() == 2
    assert torch.allclose(out["center_label"][0], box[0, 0:3])
    recovered = F.class2size(out["size_class_label"][:1], out["size_residual_label"][:1], mean)
    assert torch.allclose(recovered[0], box[0, 3:6] * 2, atol=1e-5)


def test_encode_votenet_targets_truncates_to_max_num_obj() -> None:
    mean = torch.ones(10, 3) * 0.5
    boxes = _box(heading=0.0, cls=1.0).repeat(5, 1)
    data = {"box": boxes}
    out = T.EncodeVoteNetTargets(num_heading_bin=12, mean_size_arr=mean, max_num_obj=3)(data)
    assert out["center_label"].shape == (3, 3)
    assert out["box_label_mask"].sum().item() == 3


def test_vote_then_encode_keeps_boxes_and_writes_all_labels() -> None:
    mean = torch.ones(10, 3) * 0.5
    pos = torch.rand(2048, 3) * 4
    boxes = torch.tensor(
        [
            [1.0, 1.0, 1.0, 0.5, 0.4, 0.3, 0.2, 3.0],
            [2.0, 2.0, 1.0, 0.6, 0.5, 0.4, 0.0, 1.0],
        ]
    )
    data = {"pos": pos.clone(), "box": boxes.clone(), "class": boxes[:, 7].long()}
    data = T.GenerateVoteLabels(pos_key="pos", box_key="box")(data)
    out = T.EncodeVoteNetTargets(box_key="box", num_heading_bin=12, mean_size_arr=mean, max_num_obj=64)(data)

    assert torch.equal(out["box"], boxes)
    assert out["vote_label"].shape == (2048, 9)
    assert out["vote_label_mask"].shape == (2048,)

    shapes = {
        "center_label": (64, 3),
        "heading_class_label": (64,),
        "heading_residual_label": (64,),
        "size_class_label": (64,),
        "size_residual_label": (64, 3),
        "sem_cls_label": (64,),
        "box_label_mask": (64,),
    }
    for key, shape in shapes.items():
        assert out[key].shape == shape

    assert out["heading_class_label"].dtype == torch.long
    assert out["size_class_label"].dtype == torch.long
    assert out["sem_cls_label"].dtype == torch.long
    assert out["center_label"].dtype == torch.float32
    assert out["size_residual_label"].dtype == torch.float32
    assert out["box_label_mask"].dtype == torch.float32

    assert out["box_label_mask"][:2].tolist() == [1.0, 1.0]
    assert out["box_label_mask"][2:].sum().item() == 0.0
    assert out["sem_cls_label"][:2].tolist() == [3, 1]
    assert out["size_class_label"][:2].tolist() == [3, 1]


def test_angle2class_roundtrip() -> None:
    angles = torch.tensor([0.0, 0.6, 2.5, math.pi])
    cls, residual = F.angle2class(angles, 12)
    angle_per_class = 2 * math.pi / 12
    recovered = (cls.to(angles.dtype) * angle_per_class + residual) % (2 * math.pi)
    assert torch.allclose(recovered, angles % (2 * math.pi), atol=1e-5)
