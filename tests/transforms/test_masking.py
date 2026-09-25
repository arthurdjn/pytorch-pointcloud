from unittest.mock import sentinel

import pytest
import torch

import torch_pointcloud.transforms as T
import torch_pointcloud.transforms.functional as F


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


def test_sphere_crop_accepts_integer_pos() -> None:
    # Grid coordinates are integer-typed; SphereCrop must not choke on them.
    pos = torch.tensor([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=torch.long)
    out = T.SphereCrop(pos_key="pos", radius=1000.0, max_nodes=2, center="centroid")({"pos": pos})
    assert out["pos"].shape[0] == 2
    assert out["pos"].dtype == torch.long


def test_remove_near_origin_removes_close_points() -> None:
    """Test that points within the given radius of the origin are removed."""
    pos = torch.tensor(
        [
            [0.0001, 0.0001, 0.0001],  # near origin
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],  # exactly at origin
        ]
    )
    result = F.remove_near_origin(pos, radius=1e-3)
    assert result.shape == (2, 3)
    assert torch.allclose(result, pos[1:3])


def test_remove_near_origin_keeps_all_when_none_near() -> None:
    """Test that all points are kept when none are near the origin."""
    pos = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    result = F.remove_near_origin(pos, radius=1e-3)
    assert torch.equal(result, pos)


def test_remove_near_origin_return_mask() -> None:
    """Test that the mask is correctly returned when return_mask=True."""
    pos = torch.tensor(
        [
            [0.0, 0.0, 0.0],  # at origin
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0001],  # near origin
            [0.0, 2.0, 0.0],
        ]
    )
    result, mask = F.remove_near_origin(pos, radius=1e-3, return_mask=True)
    expected_mask = torch.tensor([False, True, False, True])
    assert torch.equal(mask, expected_mask)
    assert result.shape == (2, 3)
    assert torch.allclose(result, pos[mask])


def test_remove_near_origin_custom_radius() -> None:
    """Test remove_near_origin with a custom radius."""
    pos = torch.tensor(
        [
            [0.5, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [2.5, 0.0, 0.0],
        ]
    )
    result = F.remove_near_origin(pos, radius=1.0)
    assert result.shape == (2, 3)
    assert torch.allclose(result, pos[1:])


def test_remove_near_origin_all_removed() -> None:
    """Test remove_near_origin when all points are near the origin."""
    pos = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0001, 0.0001, 0.0],
        ]
    )
    result = F.remove_near_origin(pos, radius=1.0)
    assert result.shape == (0, 3)


def test_remove_near_origin_empty_tensor() -> None:
    """Test remove_near_origin with an empty input tensor."""
    pos = torch.zeros(0, 3)
    result = F.remove_near_origin(pos, radius=1e-3)
    assert result.shape == (0, 3)


def test_bounding_box_basic() -> None:
    """Test bounding_box returns correct min and max values."""
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    result = F.bounding_box(x, dim=0)
    assert result == (1.0, 2.0, 3.0, 7.0, 8.0, 9.0)


def test_bounding_box_default_dim() -> None:
    """Test bounding_box with default dim=0."""
    x = torch.tensor([[0.0, 10.0], [-5.0, 5.0]])
    result = F.bounding_box(x)
    assert result == (-5.0, 5.0, 0.0, 10.0)


def test_bounding_box_dim0() -> None:
    """Test bounding_box along dimension 0."""
    x = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 0.0, 6.0],
        ]
    )
    result = F.bounding_box(x, dim=0)
    assert result == (1.0, 0.0, 3.0, 4.0, 2.0, 6.0)


def test_bounding_box_single_point() -> None:
    """Test bounding_box with a single point returns that point as min and max."""
    x = torch.tensor([[3.0, 5.0, 7.0]])
    result = F.bounding_box(x, dim=0)
    assert result == (3.0, 5.0, 7.0, 3.0, 5.0, 7.0)


def test_bounding_box_negative_positions() -> None:
    """Test bounding_box with all-negative positions."""
    x = torch.tensor([[-5.0, -3.0, -1.0], [-10.0, -7.0, -2.0]])
    result = F.bounding_box(x, dim=0)
    assert result == (-10.0, -7.0, -2.0, -5.0, -3.0, -1.0)


def test_bounding_box_composable_with_box_mask() -> None:
    """Test that bounding_box output can be directly fed into box_mask."""
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [0.0, -1.0, 7.0]])
    bbox = F.bounding_box(x, dim=0)
    mask = F.box_mask(x, bbox, dim=-1)
    assert mask.shape == (3,)


def test_box_mask_all_inside() -> None:
    """Test that all points inside the bounding box produce True mask."""
    x = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    bbox = (0.0, 0.0, 0.0, 3.0, 3.0, 3.0)  # min x,y,z=0, max x,y,z=3
    result = F.box_mask(x, bbox, dim=-1)
    assert torch.equal(result, torch.tensor([True, True]))


def test_box_mask_some_outside() -> None:
    """Test that points outside the bounding box produce False mask."""
    x = torch.tensor(
        [
            [1.0, 1.0, 1.0],
            [5.0, 5.0, 5.0],
            [2.0, 2.0, 2.0],
        ]
    )
    bbox = (0.0, 0.0, 0.0, 3.0, 3.0, 3.0)
    result = F.box_mask(x, bbox, dim=-1)
    assert torch.equal(result, torch.tensor([True, False, True]))


def test_box_mask_boundary_inclusive_by_default() -> None:
    """Points exactly on the boundary are included when `strict=False` (the default)."""
    x = torch.tensor(
        [
            [0.0, 0.0, 0.0],  # on min boundary
            [3.0, 3.0, 3.0],  # on max boundary
            [1.5, 1.5, 1.5],  # inside
        ]
    )
    bbox = (0.0, 0.0, 0.0, 3.0, 3.0, 3.0)
    result = F.box_mask(x, bbox, dim=-1)
    assert torch.equal(result, torch.tensor([True, True, True]))


def test_box_mask_boundary_exclusive_when_strict() -> None:
    """Points exactly on the boundary are excluded with `strict=True`."""
    x = torch.tensor(
        [
            [0.0, 0.0, 0.0],  # on min boundary
            [3.0, 3.0, 3.0],  # on max boundary
            [1.5, 1.5, 1.5],  # inside
        ]
    )
    bbox = (0.0, 0.0, 0.0, 3.0, 3.0, 3.0)
    result = F.box_mask(x, bbox, dim=-1, strict=True)
    assert torch.equal(result, torch.tensor([False, False, True]))


def test_box_mask_2d() -> None:
    """Test box_mask with 2D data."""
    x = torch.tensor(
        [
            [1.0, 1.0],
            [5.0, 1.0],
            [1.0, 5.0],
        ]
    )
    bbox = (0.0, 0.0, 3.0, 3.0)  # min x,y=0, max x,y=3
    result = F.box_mask(x, bbox, dim=-1)
    assert torch.equal(result, torch.tensor([True, False, False]))


def test_box_mask_invalid_bbox_size() -> None:
    """Test that box_mask raises ValueError for mismatched bbox size."""
    x = torch.tensor([[1.0, 2.0, 3.0]])
    bbox = (0.0, 0.0, 3.0, 3.0)  # size 4, but dim size is 3 -> expects 6
    with pytest.raises(ValueError, match="Bounding box size mismatch"):
        F.box_mask(x, bbox, dim=-1)


def test_functional_apply_mask_basic() -> None:
    """Test that apply_mask filters elements correctly."""
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    mask = torch.tensor([True, False, True, False])
    result = F.apply_mask(x, mask)
    expected = torch.tensor([1.0, 3.0])
    assert torch.equal(result, expected)


def test_apply_mask_all_true() -> None:
    """Test apply_mask with all True mask returns original tensor."""
    x = torch.tensor([1.0, 2.0, 3.0])
    mask = torch.tensor([True, True, True])
    result = F.apply_mask(x, mask)
    assert torch.equal(result, x)


def test_apply_mask_all_false() -> None:
    """Test apply_mask with all False mask returns empty tensor."""
    x = torch.tensor([1.0, 2.0, 3.0])
    mask = torch.tensor([False, False, False])
    result = F.apply_mask(x, mask)
    assert result.shape == (0,)


def test_apply_mask_2d() -> None:
    """Test apply_mask on a 2D tensor (row selection)."""
    x = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ]
    )
    mask = torch.tensor([True, False, True])
    result = F.apply_mask(x, mask)
    expected = torch.tensor([[1.0, 2.0], [5.0, 6.0]])
    assert torch.equal(result, expected)


def test_cube_mask_keeps_points_inside_chebyshev_ball() -> None:
    x = torch.tensor(
        [
            [0.0, 0.0, 0.0],  # at center
            [0.5, 0.5, 0.5],  # inside (Linf 0.5 <= 1)
            [1.0, 1.0, 1.0],  # on boundary (inclusive)
            [1.5, 0.0, 0.0],  # outside on axis X
        ]
    )
    mask = F.cube_mask(x, center=[0.0, 0.0, 0.0], radius=1.0, dim=-1)
    assert mask.dtype == torch.bool
    assert mask.tolist() == [True, True, True, False]


def test_cube_mask_off_center() -> None:
    x = torch.tensor([[5.0, 5.0, 5.0], [4.0, 4.0, 4.0]])
    mask = F.cube_mask(x, center=[5.0, 5.0, 5.0], radius=0.5, dim=-1)
    assert mask.tolist() == [True, False]


def test_cube_mask_empty() -> None:
    x = torch.empty(0, 3)
    mask = F.cube_mask(x, center=[0.0, 0.0, 0.0], radius=1.0)
    assert mask.shape == (0,)
    assert mask.dtype == torch.bool


def test_sphere_mask_keeps_points_inside_euclidean_ball() -> None:
    x = torch.tensor(
        [
            [0.0, 0.0, 0.0],  # at center
            [0.5, 0.5, 0.5],  # inside (L2 ≈ 0.87 <= 1)
            [1.0, 0.0, 0.0],  # on boundary
            [1.0, 1.0, 1.0],  # outside (L2 ≈ 1.73)
        ]
    )
    mask = F.sphere_mask(x, center=[0.0, 0.0, 0.0], radius=1.0, dim=-1)
    assert mask.dtype == torch.bool
    assert mask.tolist() == [True, True, True, False]


def test_sphere_mask_differs_from_cube_mask_in_corners() -> None:
    """A corner of the unit cube (L∞=1) is outside the unit sphere (L2≈√3)."""
    x = torch.tensor([[1.0, 1.0, 1.0]])  # L∞=1, L2=√3
    assert F.cube_mask(x, [0.0, 0.0, 0.0], 1.0).item() is True
    assert F.sphere_mask(x, [0.0, 0.0, 0.0], 1.0).item() is False


def test_sphere_mask_empty() -> None:
    x = torch.empty(0, 3)
    mask = F.sphere_mask(x, center=[0.0, 0.0, 0.0], radius=1.0)
    assert mask.shape == (0,)
    assert mask.dtype == torch.bool


def test_remove_near_origin_uses_sphere_mask_semantics() -> None:
    """remove_near_origin now delegates to sphere_mask; verify L2 semantics preserved."""
    pos = torch.tensor(
        [
            [0.5, 0.0, 0.0],  # close (L2=0.5)
            [2.0, 0.0, 0.0],  # far (L2=2.0)
            [0.6, 0.6, 0.6],  # L2 ≈ 1.04, borderline
        ]
    )
    filtered = F.remove_near_origin(pos, radius=1.0)
    # Points with L2 > 1.0 survive: the (2, 0, 0) and (0.6, 0.6, 0.6)
    assert filtered.shape == (2, 3)
