from unittest.mock import sentinel

import pytest
import torch

import torch_pointcloud.transforms as T


def test_remove_near_origin_filters_near_points() -> None:
    pos = torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0001, 0.0, 0.0]])
    labels = torch.tensor([0, 1, 2])
    data = {"pos": pos, "label": labels, "other": sentinel.other}

    result = T.RemoveNearOrigin(pos_key="pos", keys=["label"], radius=0.01)(data)

    # Only point 1 (10, 0, 0) is far enough from origin
    assert result["pos"].shape == (1, 3)
    assert torch.equal(result["pos"][0], pos[1])
    assert torch.equal(result["label"], torch.tensor([1]))
    assert result["other"] is sentinel.other


def test_remove_near_origin_default_radius_keeps_all() -> None:
    pos = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    result = T.RemoveNearOrigin(pos_key="pos")({"pos": pos})
    assert torch.equal(result["pos"], pos)


def test_box_mask_basic() -> None:
    pos = torch.tensor([[0.5, 0.5], [2.0, 0.5], [0.5, 2.0]])
    result = T.BoxMask(keys=["pos"], bbox=(0.0, 0.0, 1.0, 1.0))({"pos": pos})
    # in-place overwrite: result["pos"] is the mask now
    assert result["pos"].dtype == torch.bool
    assert result["pos"].tolist() == [True, False, False]


def test_box_mask_with_dst_keys() -> None:
    pos = torch.tensor([[0.5, 0.5], [2.0, 0.5]])
    result = T.BoxMask(keys=["pos"], bbox=(0.0, 0.0, 1.0, 1.0), dst_keys=["mask"])({"pos": pos})
    assert "mask" in result
    assert result["mask"].dtype == torch.bool
    # source pos preserved
    assert torch.equal(result["pos"], pos)


def test_box_mask_default_includes_boundary() -> None:
    pos = torch.tensor([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]])
    result = T.BoxMask(keys=["pos"], bbox=(0.0, 0.0, 1.0, 1.0), dst_keys=["mask"])({"pos": pos})
    assert result["mask"].tolist() == [True, True, True]


def test_box_mask_strict_excludes_boundary() -> None:
    pos = torch.tensor([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]])
    result = T.BoxMask(keys=["pos"], bbox=(0.0, 0.0, 1.0, 1.0), dst_keys=["mask"], strict=True)({"pos": pos})
    assert result["mask"].tolist() == [False, False, True]


def test_apply_mask_basic() -> None:
    pos = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    mask = torch.tensor([True, False, True])
    result = T.ApplyMask(keys=["pos"], mask_key="mask")({"pos": pos, "mask": mask, "other": sentinel.other})

    assert torch.equal(result["pos"], pos[mask])
    assert result["other"] is sentinel.other


def test_apply_mask_with_dst_keys() -> None:
    pos = torch.tensor([[1.0], [2.0], [3.0]])
    mask = torch.tensor([True, False, True])
    result = T.ApplyMask(keys=["pos"], mask_key="mask", dst_keys=["filtered"])({"pos": pos, "mask": mask})
    assert torch.equal(result["filtered"], pos[mask])
    # source untouched
    assert torch.equal(result["pos"], pos)


def test_apply_mask_missing_key_raises() -> None:
    with pytest.raises(KeyError, match="mask"):
        T.ApplyMask(keys=["pos"], mask_key="mask")({"pos": torch.zeros(3)})


def test_apply_mask_missing_key_allowed() -> None:
    data = {"pos": torch.tensor([[1.0], [2.0]])}
    result = T.ApplyMask(keys=["pos"], mask_key="mask", allow_missing_keys=True)(data)
    assert torch.equal(result["pos"], data["pos"])


def test_cube_mask() -> None:
    pos = torch.tensor([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]])
    data = {"pos": pos}
    transform = T.CubeMask(keys=["pos"], center=[0.0, 0.0, 0.0], radius=1.0, dst_keys=["mask"])
    result = transform(data)

    assert result["mask"][0].item() is True
    assert result["mask"][1].item() is False


def test_sphere_mask_basic() -> None:
    pos = torch.tensor(
        [
            [0.0, 0.0, 0.0],  # center
            [0.5, 0.0, 0.0],  # inside (L2 = 0.5)
            [10.0, 0.0, 0.0],  # outside
        ]
    )
    transform = T.SphereMask(keys=["pos"], center=[0.0, 0.0, 0.0], radius=1.0, dst_keys=["mask"])
    result = transform({"pos": pos})
    assert result["mask"].dtype == torch.bool
    assert result["mask"].tolist() == [True, True, False]


def test_sphere_mask_l2_vs_cube_mask_l_infinity_corner() -> None:
    """At a unit-cube corner, CubeMask says inside (L∞=1) but SphereMask says outside (L2≈√3)."""
    pos = torch.tensor([[1.0, 1.0, 1.0]])
    data = {"pos": pos}
    cube = T.CubeMask(keys=["pos"], center=[0.0, 0.0, 0.0], radius=1.0, dst_keys=["mask"])(data)
    sphere = T.SphereMask(keys=["pos"], center=[0.0, 0.0, 0.0], radius=1.0, dst_keys=["mask"])(data)
    assert cube["mask"].item() is True
    assert sphere["mask"].item() is False


def test_sphere_crop_fixed_center_drops_outside() -> None:
    pos = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [100.0, 0.0, 0.0],  # far outside
        ]
    )
    out = T.SphereCrop(pos_key="pos", radius=1.0, center=(0.0, 0.0, 0.0))({"pos": pos})
    assert out["pos"].shape[0] == 2


def test_sphere_crop_centroid_center() -> None:
    pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    # Centroid = (1, 0, 0); radius 1.5 should keep all three.
    out = T.SphereCrop(pos_key="pos", radius=1.5, center="centroid")({"pos": pos})
    assert out["pos"].shape[0] == 3


def test_sphere_crop_applies_mask_to_other_keys() -> None:
    pos = torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    color = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    out = T.SphereCrop(pos_key="pos", keys=("color",), radius=1.0, center=(0.0, 0.0, 0.0))({"pos": pos, "color": color})
    assert out["pos"].shape == out["color"].shape


def test_sphere_crop_max_nodes_keeps_nearest() -> None:
    # Centroid of the five points is (2, 0, 0); a wide radius keeps all five, so
    # max_nodes=3 must keep the three nearest the centroid: x = 1, 2, 3.
    pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    out = T.SphereCrop(pos_key="pos", radius=10.0, max_nodes=3, center="centroid")({"pos": pos})
    assert out["pos"].shape[0] == 3
    assert set(out["pos"][:, 0].tolist()) == {1.0, 2.0, 3.0}


def test_sphere_crop_max_nodes_above_count_is_noop() -> None:
    # Only three points fall in the sphere; max_nodes=10 leaves them untouched.
    pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    out = T.SphereCrop(pos_key="pos", radius=10.0, max_nodes=10, center="centroid")({"pos": pos})
    assert out["pos"].shape[0] == 3


def test_sphere_crop_accepts_integer_coords() -> None:
    # Grid coordinates are integer-typed; SphereCrop must not choke on them.
    pos = torch.tensor([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=torch.long)
    out = T.SphereCrop(pos_key="pos", radius=1000.0, max_nodes=2, center="centroid")({"pos": pos})
    assert out["pos"].shape[0] == 2
    assert out["pos"].dtype == torch.long
