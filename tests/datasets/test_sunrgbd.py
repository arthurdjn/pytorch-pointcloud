"""Tests for the SUN RGB-D dataset helpers."""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import numpy as np
import pytest
import torch

from torch_pointcloud.datasets import PointCloudDataLoader, SunRGBD
from torch_pointcloud.datasets.sunrgbd import (
    SUNRGBD_CLASSES,
    decode_depth,
    parse_boxes,
    rebase_sequence,
    unproject,
)
from torch_pointcloud.utils.data import DETECTION_STACK_KEYS, DataKeys, collate


def test_class_names_count() -> None:
    assert len(SUNRGBD_CLASSES) == 10


def test_class_to_idx_mapping() -> None:
    mapping = {name: i for i, name in enumerate(SUNRGBD_CLASSES)}
    assert mapping["bed"] == 0
    assert mapping["bathtub"] == 9


def test_rebase_sequence_strips_prefix() -> None:
    assert rebase_sequence("/n/fs/sun3d/data/SUNRGBD/kv1/NYUdata/NYU0001") == "kv1/NYUdata/NYU0001"
    assert rebase_sequence("//n/fs/sun3d/data/SUNRGBD/kv1/b3do/img_0063") == "kv1/b3do/img_0063"
    assert rebase_sequence("SUNRGBD/kv1/NYUdata/NYU0001") == "kv1/NYUdata/NYU0001"


def test_rebase_sequence_collapses_internal_double_slash() -> None:
    # Two-thirds of the metadata depth / rgb paths carry a doubled slash (`-resize//depth/...`) that
    # does not match the single-slash zip member, so it must collapse for every scene to resolve.
    assert (
        rebase_sequence("/n/fs/sun3d/data/SUNRGBD/kv2/kinect2data/000385-resize//depth/0000087.png")
        == "kv2/kinect2data/000385-resize/depth/0000087.png"
    )


def test_decode_depth_shape_and_dtype() -> None:
    raw = np.array([[8 << 3, 0], [16 << 3, 32 << 3]], dtype=np.uint16)
    depth = decode_depth(raw)
    assert depth.shape == (2, 2)
    assert depth.dtype == np.float32
    assert depth[0, 0] == pytest.approx(8 / 1000.0)
    assert depth[0, 1] == 0.0
    assert depth[1, 0] == pytest.approx(16 / 1000.0)


def test_decode_depth_truncates_at_8m() -> None:
    raw = np.array([[1]], dtype=np.uint16)
    depth = decode_depth(raw)
    assert depth[0, 0] == 8.0


def test_unproject_identity_intrinsics() -> None:
    depth = np.array([[1.0, 2.0]], dtype=np.float32)
    k = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    rtilt = np.eye(3, dtype=np.float32)
    pts = unproject(depth, k, rtilt)
    assert pts.shape == (2, 3)
    assert pts.dtype == np.float32
    np.testing.assert_allclose(pts[0], [0.0, 1.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(pts[1], [2.0, 2.0, 0.0], atol=1e-6)


def _box(classname: str, orientation: tuple[float, float]) -> SimpleNamespace:
    return SimpleNamespace(
        classname=classname,
        centroid=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        coeffs=np.array([0.5, 0.6, 0.7], dtype=np.float32),
        basis=np.eye(3, dtype=np.float32),
        orientation=np.array(orientation, dtype=np.float32),
        _fieldnames=["classname", "centroid", "coeffs", "basis", "orientation"],
    )


def test_parse_boxes_heading_sign_and_reordered_halfextents() -> None:
    mapping = {name: i for i, name in enumerate(SUNRGBD_CLASSES)}
    box = _box("chair", (0.9302, -0.3650))
    out = parse_boxes(box, mapping)
    assert out.shape == (1, 8)
    np.testing.assert_allclose(out[0, :3], [1.0, 2.0, 3.0], atol=1e-6)
    # coeffs [0.5, 0.6, 0.7] -> [c1, c0, c2] = [0.6, 0.5, 0.7], matching votenet's [l, w, h].
    np.testing.assert_allclose(out[0, 3:6], [0.6, 0.5, 0.7], atol=1e-6)
    assert out[0, 6] == pytest.approx(-math.atan2(-0.3650, 0.9302), abs=1e-5)
    assert int(out[0, 7]) == mapping["chair"]


def test_parse_boxes_filters_unknown_classes() -> None:
    mapping = {name: i for i, name in enumerate(SUNRGBD_CLASSES)}
    boxes = np.array([_box("chair", (1.0, 0.0)), _box("swivelchair", (1.0, 0.0)), _box("wall", (1.0, 0.0))])
    out = parse_boxes(boxes, mapping)
    assert out.shape == (1, 8)
    assert int(out[0, 7]) == mapping["chair"]


def test_parse_boxes_empty_returns_zero_rows() -> None:
    mapping = {name: i for i, name in enumerate(SUNRGBD_CLASSES)}
    assert parse_boxes(None, mapping).shape == (0, 8)
    assert parse_boxes(np.array([_box("wall", (1.0, 0.0))]), mapping).shape == (0, 8)


def _sample(num_points: int, num_boxes: int) -> dict[str, torch.Tensor]:
    return {
        DataKeys.POS: torch.randn(num_points, 3),
        DataKeys.BOX: torch.randn(num_boxes, 8),
        DataKeys.CLASS: torch.randint(0, 10, (num_boxes,)),
    }


def test_collate_index_keys_emits_box_scene_index() -> None:
    samples = [_sample(5, 2), _sample(7, 3)]
    out = collate(samples, index_keys=(DataKeys.BOX,))
    assert out[DataKeys.POS].shape == (12, 3)
    assert out[DataKeys.BATCH].shape == (12,)
    assert out[DataKeys.BOX].shape == (5, 8)
    box_batch = out[f"{DataKeys.BOX}_batch"]
    assert box_batch.shape == (5,)
    assert box_batch.tolist() == [0, 0, 1, 1, 1]


def test_collate_stack_keys_stacks_dense_ground_truth() -> None:
    def scene(n_points: int) -> dict:
        return {
            DataKeys.POS: torch.randn(n_points, 3),
            "center_label": torch.randn(64, 3),
            "box_label_mask": torch.zeros(64),
            "vote_label": torch.randn(n_points, 9),
            "vote_label_mask": torch.zeros(n_points, dtype=torch.long),
        }

    out = collate([scene(20), scene(20)], stack_keys=DETECTION_STACK_KEYS)
    assert out[DataKeys.POS].shape == (40, 3)
    assert out["batch"].shape == (40,)
    assert out["center_label"].shape == (2, 64, 3)
    assert out["box_label_mask"].shape == (2, 64)
    assert out["vote_label"].shape == (2, 20, 9)
    assert out["vote_label_mask"].shape == (2, 20)


def test_collate_index_keys_handles_empty_scene() -> None:
    samples = [_sample(4, 0), _sample(6, 2)]
    out = collate(samples, index_keys=(DataKeys.BOX,))
    box_batch = out[f"{DataKeys.BOX}_batch"]
    assert out[DataKeys.BOX].shape == (2, 8)
    assert box_batch.tolist() == [1, 1]


def test_plain_collate_loses_box_scene_index() -> None:
    samples = [_sample(5, 2), _sample(7, 3)]
    out = collate(samples)
    assert out[DataKeys.BOX].shape == (5, 8)
    assert f"{DataKeys.BOX}_batch" not in out


@pytest.mark.parametrize("split", ["train", "val"])
def test_sunrgbd_loads_processed_fixture(datasets_dir_factory: Callable[..., Path], split: str) -> None:
    datasets_dir = datasets_dir_factory("SunRGBD/processed/**/*")
    dataset = SunRGBD(root=datasets_dir, split=split, show_progress=False)
    assert len(dataset) == 3
    sample = dataset[0]
    assert sample[DataKeys.POS].shape[1] == 3
    assert sample[DataKeys.POS].dtype == torch.float32
    assert sample[DataKeys.BOX].shape[1] == 8
    assert sample[DataKeys.CLASS].shape[0] == sample[DataKeys.BOX].shape[0]
    assert sample[DataKeys.CLASS].dtype == torch.long
    assert sample[DataKeys.COLOR].shape == sample[DataKeys.POS].shape


def test_sunrgbd_dataloader_packs_boxes_with_scene_index(datasets_dir_factory: Callable[..., Path]) -> None:
    datasets_dir = datasets_dir_factory("SunRGBD/processed/**/*")
    dataset = SunRGBD(root=datasets_dir, split="val", show_progress=False)
    num_boxes = sum(int(dataset[i][DataKeys.BOX].shape[0]) for i in range(len(dataset)))

    loader = PointCloudDataLoader(dataset, batch_size=len(dataset), shuffle=False, index_keys=(DataKeys.BOX,))
    batch = next(iter(loader))

    assert batch[DataKeys.POS].shape[0] == batch[DataKeys.BATCH].shape[0]
    assert batch[DataKeys.BOX].shape == (num_boxes, 8)
    box_batch = batch[f"{DataKeys.BOX}_batch"]
    assert box_batch.shape == (num_boxes,)
    assert int(box_batch.min()) >= 0 and int(box_batch.max()) < len(dataset)


def test_sunrgbd_processes_from_raw_fixture(datasets_dir_factory: Callable[..., Path]) -> None:
    datasets_dir = datasets_dir_factory("SunRGBD/raw/**/*")
    dataset = SunRGBD(root=datasets_dir, split="val", show_progress=False)
    assert len(dataset) == 3
    sample = dataset[0]
    assert sample[DataKeys.POS].shape[1] == 3
    assert sample[DataKeys.POS].shape[0] > 2048
    assert sample[DataKeys.BOX].shape[1] == 8
    assert sample[DataKeys.BOX].shape[0] == sample[DataKeys.CLASS].shape[0]
    assert sample[DataKeys.BOX].shape[0] >= 1
