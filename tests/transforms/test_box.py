import math
from pathlib import Path
from typing import Any, Dict

import pytest
import torch
from torch import Tensor

import torch_pointcloud.transforms as T
import torch_pointcloud.transforms.functional as F
from torch_pointcloud.ops.box3d import box_corners, points_in_boxes
from torch_pointcloud.utils.data import DataKeys


def _box(heading: float = 0.0) -> Tensor:
    return torch.tensor([[1.0, 0.5, 0.3, 0.8, 0.6, 0.4, heading]])


def _half_extent(box: Tensor) -> Tensor:
    return torch.cat([box[0:3], box[3:6] / 2, box[6:7]])


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


def test_rotate_boxes_increments_heading() -> None:
    box = _box(heading=0.2)
    rotation = F.rotation_matrix(0.5, axis=2)
    rotated = F.rotate_boxes(box, rotation, 0.5)
    assert torch.allclose(rotated[:, 6], box[:, 6] + 0.5)
    assert torch.allclose(rotated[:, 3:6], box[:, 3:6])
    assert torch.allclose(rotated[:, 0:3], box[:, 0:3] @ rotation.T)


def test_scale_boxes_centers_and_sizes() -> None:
    box = _box(heading=0.2)
    scaled = F.scale_boxes(box, 2.0)
    assert torch.allclose(scaled[:, 0:6], box[:, 0:6] * 2.0)
    assert torch.allclose(scaled[:, 6], box[:, 6])


def test_translate_boxes_centers_only() -> None:
    box = _box(heading=0.2)
    shift = torch.tensor([0.5, -1.0, 2.0])
    shifted = F.translate_boxes(box, shift)
    assert torch.allclose(shifted[:, 0:3], box[:, 0:3] + shift)
    assert torch.allclose(shifted[:, 3:7], box[:, 3:7])


def test_points_in_oriented_box_axis_aligned() -> None:
    box = torch.tensor([1.0, 0.5, 0.3, 0.4, 0.3, 0.2, 0.0])  # half extents
    pts = torch.tensor([[1.0, 0.5, 0.3], [1.4, 0.5, 0.3], [2.0, 0.5, 0.3]])
    mask = F.points_in_oriented_box(pts, box)
    assert mask.tolist() == [True, True, False]


def test_points_in_oriented_box_yaw_aware() -> None:
    box = torch.tensor([1.0, 0.5, 0.3, 0.4, 0.3, 0.2, math.pi / 2])  # half extents
    corner = torch.tensor([[1.0 + 0.3, 0.5 + 0.4, 0.3]])
    outside = torch.tensor([[1.0 + 0.4, 0.5, 0.3]])
    assert F.points_in_oriented_box(corner, box).tolist() == [True]
    assert F.points_in_oriented_box(outside, box).tolist() == [False]


def test_random_flip_boxes_preserves_membership() -> None:
    box = _box(heading=0.2)
    face = torch.tensor([[1.4, 0.5, 0.3]])
    data = {"pos": face.clone(), "box": box.clone()}
    out = T.RandomFlip(keys="pos", box_key="box", axes=(0,), p=1.0, seed=0)(data)
    assert F.points_in_oriented_box(out["pos"], _half_extent(out["box"][0])).item()


def test_random_rotate_boxes_preserves_membership() -> None:
    box = _box(heading=0.2)
    face = torch.tensor([[1.4, 0.5, 0.3]])
    data = {"pos": face.clone(), "box": box.clone()}
    out = T.RandomRotate(keys="pos", box_key="box", angle_range=(25.0, 25.0), p=1.0, seed=0)(data)
    assert F.points_in_oriented_box(out["pos"], _half_extent(out["box"][0])).item()


def test_random_rotate_boxes_containment_is_exact() -> None:
    """Rotating points and boxes jointly preserves containment for every point of an oriented box."""
    gen = torch.Generator().manual_seed(0)
    center = torch.tensor([1.0, -2.0, 0.5])
    dims = torch.tensor([1.2, 0.8, 0.6])
    heading = 0.7
    box = torch.cat([center, dims, torch.tensor([heading])]).unsqueeze(0)
    local = (torch.rand(500, 3, generator=gen) - 0.5) * dims * 0.99
    pos = local @ F.rotation_matrix(heading, axis=2).T + center
    assert F.points_in_oriented_box(pos, _half_extent(box[0])).all()

    data = {"pos": pos, "box": box.clone()}
    out = T.RandomRotate(keys="pos", box_key="box", angle_range=(140.0, 140.0), p=1.0, seed=0)(data)
    assert F.points_in_oriented_box(out["pos"], _half_extent(out["box"][0])).all()

    rotation = F.rotation_matrix(math.radians(140.0), axis=2)
    assert torch.allclose(box_corners(out["box"]), box_corners(box) @ rotation.T, atol=1e-5)


def test_random_scale_boxes_preserves_membership() -> None:
    box = _box(heading=0.2)
    face = torch.tensor([[1.4, 0.5, 0.3]])
    data = {"pos": face.clone(), "box": box.clone()}
    out = T.RandomScale(keys="pos", box_key="box", scale_range=(1.3, 1.3), p=1.0, seed=0)(data)
    assert F.points_in_oriented_box(out["pos"], _half_extent(out["box"][0])).item()


def test_random_translate_boxes_move_with_points() -> None:
    box = _box(heading=0.2)
    face = torch.tensor([[1.4, 0.5, 0.3]])
    data = {"pos": face.clone(), "box": box.clone()}
    out = T.RandomTranslate(keys="pos", box_key="box", translation_range=(0.7, 0.7), p=1.0, seed=0)(data)
    assert torch.allclose(out["box"][:, 0:3], box[:, 0:3] + 0.7)
    assert torch.allclose(out["box"][:, 3:7], box[:, 3:7])
    assert F.points_in_oriented_box(out["pos"], _half_extent(out["box"][0])).item()


def test_random_rotate_boxes_p_zero_is_noop() -> None:
    box = _box(heading=0.2)
    pos = torch.tensor([[1.0, 0.5, 0.3]])
    data = {"pos": pos.clone(), "box": box.clone()}
    out = T.RandomRotate(keys="pos", box_key="box", p=0.0)(data)
    assert torch.allclose(out["box"], box)
    assert torch.allclose(out["pos"], pos)


def test_random_rotate_boxes_votes_stay_consistent() -> None:
    box = _box(heading=0.2)
    pos = torch.tensor([[1.0, 0.5, 0.3]])
    vote = (box[:, 0:3] - pos).repeat(1, 3)
    data = {"pos": pos.clone(), "box": box.clone(), "vote_label": vote.clone()}
    out = T.RandomRotate(keys=("pos", "vote_label"), box_key="box", angle_range=(40.0, 40.0), p=1.0, seed=0)(data)
    expected = out["box"][:, 0:3] - out["pos"]
    assert torch.allclose(out["vote_label"][:, 0:3], expected, atol=1e-5)
    assert torch.allclose(out["vote_label"][:, 3:6], expected, atol=1e-5)
    assert F.points_in_oriented_box(out["pos"], _half_extent(out["box"][0])).item()


def test_random_scale_boxes_votes_stay_consistent() -> None:
    box = _box(heading=0.2)
    pos = torch.tensor([[1.0, 0.5, 0.3]])
    vote = (box[:, 0:3] - pos).repeat(1, 3)
    data = {"pos": pos.clone(), "box": box.clone(), "vote_label": vote.clone()}
    out = T.RandomScale(keys=("pos", "vote_label"), box_key="box", scale_range=(1.3, 1.3), p=1.0, seed=0)(data)
    expected = out["box"][:, 0:3] - out["pos"]
    assert torch.allclose(out["vote_label"][:, 0:3], expected, atol=1e-5)


def test_instance_to_box_extents_class_and_drop() -> None:
    # inst 0 (class 2): corners (0,0,0)-(2,1,1); inst 1 (class 5): a y-slab; inst 2: ignore class -> dropped.
    pos = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [2.0, 1.0, 1.0],
            [5.0, 5.0, 5.0],
            [5.0, 6.0, 5.0],
            [9.0, 9.0, 9.0],
        ]
    )
    data = {"pos": pos, "instance": torch.tensor([0, 0, 1, 1, 2]), "segment": torch.tensor([2, 2, 5, 5, -1])}
    out = T.InstanceToBox()(data)
    box = out["box"]
    assert box.shape == (2, 7)
    assert torch.allclose(box[0], torch.tensor([1.0, 0.5, 0.5, 2.0, 1.0, 1.0, 0.0]))
    assert torch.allclose(box[1], torch.tensor([5.0, 5.5, 5.0, 0.0, 1.0, 0.0, 0.0]))
    assert out["label"].tolist() == [2, 5]
    assert out["label"].dtype == torch.long


def test_instance_to_box_excludes_negative_instance_ids() -> None:
    # Unlabeled points (instance -1) span the scene; they must not form a giant box.
    pos = torch.tensor([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0], [1.0, 1.0, 1.0], [2.0, 1.0, 1.0]])
    data = {"pos": pos, "instance": torch.tensor([-1, -1, 0, 0]), "segment": torch.tensor([3, 3, 2, 2])}
    out = T.InstanceToBox()(data)
    assert out["box"].shape == (1, 7)
    assert torch.allclose(out["box"][0], torch.tensor([1.5, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]))
    assert out["label"].tolist() == [2]


def test_instance_to_box_all_ignored_is_empty() -> None:
    data = {"pos": torch.rand(4, 3), "instance": torch.tensor([0, 0, 1, 1]), "segment": torch.tensor([-1, -1, -1, -1])}
    out = T.InstanceToBox()(data)
    assert out["box"].shape == (0, 7)
    assert out["label"].shape == (0,)
    assert out["label"].dtype == torch.long


def test_relabel_boxes_maps_drops_and_ignores() -> None:
    # raw 0 -> det 0, raw 3 -> det 1 (foreground); raw 1 is an ignore class attributed to det 0;
    # raw 7 is dropped.
    data: Dict[str, Any] = {
        DataKeys.BOX: torch.tensor(
            [[0.0, 0, 0, 1, 1, 1, 0], [1, 0, 0, 1, 1, 1, 0], [2, 0, 0, 1, 1, 1, 0], [3, 0, 0, 1, 1, 1, 0]]
        ),
        DataKeys.LABEL: torch.tensor([0, 1, 3, 7]),
        DataKeys.OCCLUSION: torch.tensor([0, 1, 2, 3]),
    }
    out = T.RelabelBoxes(
        keys=(DataKeys.BOX, DataKeys.LABEL, DataKeys.OCCLUSION), mapping={0: 0, 3: 1}, ignore_mapping={1: 0}
    )(data)
    assert out[DataKeys.LABEL].tolist() == [0, 0, 1]
    assert out["ignore_mask"].tolist() == [False, True, False]
    # Every box key is filtered by the same keep mask: the dropped raw-7 row is gone everywhere.
    assert out[DataKeys.BOX].shape == (3, 7)
    assert out[DataKeys.OCCLUSION].tolist() == [0, 1, 2]
    assert torch.equal(out[DataKeys.BOX][:, 0], torch.tensor([0.0, 1.0, 2.0]))


def test_relabel_boxes_ignore_attribution() -> None:
    # Ignore rows carry the detection class they excuse: raw 1 -> det 0 (Van -> Car style),
    # raw 4 -> det 1 (Person_sitting -> Pedestrian style).
    data: Dict[str, Any] = {
        DataKeys.BOX: torch.zeros(3, 7),
        DataKeys.LABEL: torch.tensor([1, 4, 0]),
    }
    out = T.RelabelBoxes(keys=(DataKeys.BOX, DataKeys.LABEL), mapping={0: 0, 3: 1}, ignore_mapping={1: 0, 4: 1})(data)
    assert out[DataKeys.LABEL].tolist() == [0, 1, 0]
    assert out["ignore_mask"].tolist() == [True, True, False]


def test_relabel_boxes_difficulty_to_ignore() -> None:
    # Both boxes are foreground class 0, but the second fails the occlusion <= 1 rule -> ignore region.
    data: Dict[str, Any] = {
        DataKeys.BOX: torch.zeros(2, 7),
        DataKeys.LABEL: torch.tensor([0, 0]),
        DataKeys.OCCLUSION: torch.tensor([0, 2]),
    }
    out = T.RelabelBoxes(
        keys=(DataKeys.BOX, DataKeys.LABEL, DataKeys.OCCLUSION),
        mapping={0: 0},
        ignore_fields={DataKeys.OCCLUSION: (None, 1)},
    )(data)
    assert out[DataKeys.LABEL].tolist() == [0, 0]
    assert out["ignore_mask"].tolist() == [False, True]


def _scene_with_boxes() -> Dict[str, Any]:
    # two boxes with four points each, one lone point outside, intensities equal to the point index
    boxes = torch.tensor([[2.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0], [10.0, 5.0, 0.0, 2.0, 2.0, 2.0, 0.5]])
    inside_first = torch.tensor([[1.5, 0.2, 0.1], [2.5, -0.3, 0.0], [2.0, 0.4, 0.4], [1.8, -0.6, -0.5]])
    inside_second = torch.tensor([[10.0, 5.0, 0.1], [10.3, 5.2, 0.0], [9.8, 4.9, 0.4], [10.1, 5.1, -0.5]])
    outside = torch.tensor([[30.0, 30.0, 0.0]])
    pos = torch.cat([inside_first, inside_second, outside])
    return {
        DataKeys.POS: pos,
        DataKeys.INTENSITY: torch.arange(pos.shape[0], dtype=torch.float32)[:, None],
        DataKeys.BOX: boxes,
        DataKeys.LABEL: torch.tensor([0, 1]),
        DataKeys.OCCLUSION: torch.tensor([0, 2]),
    }


def test_cut_boxes_centers_the_points_of_every_object() -> None:
    scene = _scene_with_boxes()
    objects = F.cut_boxes(scene, keys=[DataKeys.POS, DataKeys.INTENSITY], attribute_keys=DataKeys.OCCLUSION)
    assert len(objects) == 2 and [int(obj[DataKeys.LABEL]) for obj in objects] == [0, 1]
    assert all(type(key) is str for key in objects[0])

    first, second = objects
    assert torch.allclose(first[DataKeys.POS] + first[DataKeys.BOX][:3], scene[DataKeys.POS][:4])
    assert torch.equal(second[DataKeys.INTENSITY][:, 0], torch.arange(4.0, 8.0))
    assert [int(obj[DataKeys.OCCLUSION]) for obj in objects] == [0, 2]

    empty: Dict[str, Any] = {
        DataKeys.POS: torch.zeros(3, 3),
        DataKeys.BOX: torch.zeros(0, 7),
        DataKeys.LABEL: torch.zeros(0),
    }
    assert F.cut_boxes(empty) == []


def test_cut_boxes_objects_round_trip_through_the_safe_loader(tmp_path: Path) -> None:
    objects = F.cut_boxes(_scene_with_boxes(), keys=[DataKeys.POS, DataKeys.INTENSITY])
    torch.save(objects, tmp_path / "objects.pt")
    loaded = torch.load(tmp_path / "objects.pt", weights_only=True)
    assert len(loaded) == 2 and torch.equal(loaded[1][DataKeys.BOX], objects[1][DataKeys.BOX])


def test_paste_boxes_pastes_the_objects_that_do_not_overlap() -> None:
    scene = _scene_with_boxes()
    # three objects of label 0 with two points each: one lands on the scene's first box, two on free ground
    boxes = torch.tensor(
        [
            [2.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0],
            [20.0, -5.0, 0.0, 2.0, 2.0, 2.0, 0.3],
            [40.0, 8.0, 0.0, 2.0, 2.0, 2.0, 1.0],
        ]
    )
    objects: list[Dict[str, Tensor]] = [
        {
            DataKeys.POS: torch.tensor([[0.1, 0.1, 0.1], [-0.2, 0.0, 0.3]]),
            DataKeys.INTENSITY: torch.full((2, 1), 100.0 + k),
            DataKeys.BOX: box,
            DataKeys.LABEL: torch.tensor(0),
        }
        for k, box in enumerate(boxes)
    ]
    transform = T.PasteBoxes(
        objects, num_samples={0: 3}, keys=[DataKeys.POS, DataKeys.INTENSITY], count_existing=False, seed=0
    )
    out = transform(scene)

    # the overlapping object is left out, the two others are appended with their labels
    assert out[DataKeys.BOX].shape[0] == 4 and out[DataKeys.LABEL].tolist() == [0, 1, 0, 0]
    assert torch.allclose(out[DataKeys.BOX][2:].sort(dim=0).values, boxes[1:].sort(dim=0).values)
    # 4 pasted points come first, then the 9 scene points (none fell inside the pasted boxes)
    assert out[DataKeys.POS].shape[0] == 13 and out[DataKeys.INTENSITY].shape == (13, 1)
    assert (out[DataKeys.INTENSITY][:4] >= 100).all() and (out[DataKeys.INTENSITY][4:] < 100).all()
    for box in out[DataKeys.BOX][2:]:
        assert points_in_boxes(out[DataKeys.POS][:4], box[None]).any()

    # counting the single object of label 0 already in the scene lowers the target
    counted = T.PasteBoxes(objects, num_samples={0: 2}, keys=[DataKeys.POS, DataKeys.INTENSITY], seed=0)
    assert counted(scene)[DataKeys.BOX].shape[0] == 3


def test_paste_boxes_clears_the_scene_points_under_the_pasted_object() -> None:
    scene = _scene_with_boxes()
    objects: list[Dict[str, Tensor]] = [
        {
            DataKeys.POS: torch.zeros(3, 3),
            DataKeys.BOX: torch.tensor([30.0, 30.0, 0.0, 2.0, 2.0, 2.0, 0.0]),
            DataKeys.LABEL: torch.tensor(1),
        }
    ]
    out = T.PasteBoxes(objects, num_samples={1: 5}, count_existing=False)(scene)

    # the lone scene point at (30, 30, 0) is inside the pasted box and goes away; the three object points come first
    assert out[DataKeys.POS].shape[0] == 11
    assert torch.allclose(out[DataKeys.POS][:3], torch.tensor([[30.0, 30.0, 0.0]]).expand(3, 3))
    assert out[DataKeys.LABEL].tolist() == [0, 1, 1]


def test_paste_boxes_appends_the_attributes_of_the_pasted_objects() -> None:
    scene = _scene_with_boxes()
    scene[DataKeys.VELOCITY] = torch.tensor([[0.5, 0.0], [0.0, 0.5]])
    # every object moves along x at the speed of its x coordinate, so the pasted rows can be told apart
    objects: list[Dict[str, Tensor]] = [
        {
            DataKeys.POS: torch.zeros(2, 3),
            DataKeys.BOX: torch.tensor([20.0, -5.0, 0.0, 2.0, 2.0, 2.0, 0.3]),
            DataKeys.LABEL: torch.tensor(0),
            DataKeys.VELOCITY: torch.tensor([20.0, 0.0]),
        },
        {
            DataKeys.POS: torch.zeros(2, 3),
            DataKeys.BOX: torch.tensor([40.0, 8.0, 0.0, 2.0, 2.0, 2.0, 1.0]),
            DataKeys.LABEL: torch.tensor(0),
            DataKeys.VELOCITY: torch.tensor([40.0, 0.0]),
        },
    ]
    transform = T.PasteBoxes(
        objects, num_samples={0: 2}, attribute_keys=DataKeys.VELOCITY, count_existing=False, seed=0
    )
    out = transform(scene)

    assert out[DataKeys.BOX].shape[0] == 4 and out[DataKeys.VELOCITY].shape == (4, 2)
    assert torch.equal(out[DataKeys.VELOCITY][:2], scene[DataKeys.VELOCITY])
    assert torch.equal(out[DataKeys.VELOCITY][2:, 0], out[DataKeys.BOX][2:, 0])

    with pytest.raises(ValueError, match="cut_boxes"):
        T.PasteBoxes(objects, num_samples={0: 2}, attribute_keys=DataKeys.OCCLUSION)
