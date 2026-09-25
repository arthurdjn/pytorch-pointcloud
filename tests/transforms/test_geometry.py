import math

import pytest
import torch

import torch_pointcloud.transforms as T
import torch_pointcloud.transforms.functional as F
from torch_pointcloud.transforms import Voxelize
from torch_pointcloud.utils.imports import _PYG_LIB_AVAILABLE


def test_center_bbox() -> None:
    pos = torch.tensor([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]])
    data = {"pos": pos}
    transform = T.Shift(keys=["pos"], method="bbox")
    result = transform(data)

    expected = pos - torch.tensor([1.0, 1.0, 1.0])
    assert torch.allclose(result["pos"], expected)


def test_center_centroid() -> None:
    pos = torch.tensor([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]])
    data = {"pos": pos}
    transform = T.Shift(keys=["pos"], method="centroid")
    result = transform(data)

    expected = pos - pos.mean(dim=0)
    assert torch.allclose(result["pos"], expected)


def test_axis_min_offset() -> None:
    pos = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    data = {"pos": pos}
    transform = T.AxisMinOffset(keys=["pos"], axis=2, dst_keys=["h"])
    result = transform(data)

    assert result["h"].shape == (2, 1)
    assert result["h"][0].item() == 0.0
    assert result["h"][1].item() == 3.0


def test_quantize() -> None:
    pos = torch.tensor([[0.0, 0.0, 0.0], [0.03, 0.0, 0.0], [0.05, 0.0, 0.0]])
    data = {"pos": pos, "x": torch.randn(3, 2)}
    result = T.Quantize(keys="pos", size=0.02, dst_keys="pos_grid")(data)

    assert torch.equal(result["pos_grid"], torch.tensor([[0, 0, 0], [1, 0, 0], [2, 0, 0]]))
    assert result["pos"] is pos
    assert result["x"] is data["x"]

    in_place = T.Quantize(keys="pos", size=0.02)(data)
    assert torch.equal(in_place["pos"], result["pos_grid"])


@pytest.mark.skipif(not _PYG_LIB_AVAILABLE, reason="pyg-lib is not installed")
def test_estimate_normals() -> None:
    grid = torch.linspace(-1.0, 1.0, 20)
    xx, yy = torch.meshgrid(grid, grid, indexing="ij")
    plane = torch.stack([xx.reshape(-1), yy.reshape(-1), torch.zeros(400)], dim=1)
    data = {"pos": plane}

    result = T.EstimateNormals(keys="pos", dst_keys="normal", k=16)(data)

    assert result["normal"].shape == (400, 3)
    # The z=0 plane's normal is the unit z axis.
    assert torch.allclose(result["normal"][:, 2].abs(), torch.ones(400), atol=1e-5)


def test_shift_min_method() -> None:
    pos = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    result = T.Shift(keys=["pos"], method="min")({"pos": pos})
    assert torch.allclose(result["pos"], pos - pos.min(dim=0).values)


def test_shift_axes_subset() -> None:
    pos = torch.tensor([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]])
    # bbox midrange is (1, 1, 1); axes=[0] shifts only X.
    result = T.Shift(keys=["pos"], method="bbox", axes=[0])({"pos": pos})
    expected = torch.tensor([[-1.0, 0.0, 0.0], [1.0, 2.0, 2.0]])
    assert torch.allclose(result["pos"], expected)


def test_shift_invalid_method_raises() -> None:
    with pytest.raises(ValueError, match="Invalid method"):
        T.Shift(keys=["pos"], method="bogus")  # type: ignore[arg-type]


def test_shift_dst_keys() -> None:
    pos = torch.tensor([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]])
    result = T.Shift(keys=["pos"], method="bbox", dst_keys=["shifted"])({"pos": pos})
    assert "shifted" in result
    assert torch.allclose(result["pos"], pos)  # source untouched


def test_shift_pointcept_centering_with_z() -> None:
    """Pointcept-style centering (XY bbox + Z min) via Compose, replacing the
    old CenterShift(apply_z=True)."""
    pos = torch.tensor([[0.0, 0.0, 0.0], [2.0, 4.0, 6.0]])
    transform = T.Compose(
        [
            T.Shift(keys=["pos"], method="bbox", axes=[0, 1]),  # XY: bbox midrange
            T.Shift(keys=["pos"], method="min", axes=[2]),  # Z:  min
        ]
    )
    result = transform({"pos": pos})
    expected = torch.tensor([[-1.0, -2.0, 0.0], [1.0, 2.0, 6.0]])
    assert torch.allclose(result["pos"], expected)


def test_shift_pointcept_centering_without_z() -> None:
    """Pointcept-style centering with Z untouched, replacing CenterShift(apply_z=False)."""
    pos = torch.tensor([[0.0, 0.0, 1.0], [2.0, 4.0, 7.0]])
    result = T.Shift(keys=["pos"], method="bbox", axes=[0, 1])({"pos": pos})
    # Z is left unchanged
    assert torch.allclose(result["pos"][:, 2], pos[:, 2])
    # XY are bbox-centered
    assert torch.allclose(result["pos"][:, :2], pos[:, :2] - torch.tensor([1.0, 2.0]))


def test_shift_disjoint_axes_commute() -> None:
    """Two Shift calls on disjoint axes commute: the second min/max sees the
    first's mutation, but only on axes the second ignores, so the result is
    invariant to ordering."""
    pos = torch.tensor([[0.0, 0.0, 0.0], [4.0, 6.0, 8.0]])
    a = T.Compose(
        [
            T.Shift(keys=["pos"], method="bbox", axes=[0, 1]),
            T.Shift(keys=["pos"], method="min", axes=[2]),
        ]
    )({"pos": pos})
    b = T.Compose(
        [
            T.Shift(keys=["pos"], method="min", axes=[2]),
            T.Shift(keys=["pos"], method="bbox", axes=[0, 1]),
        ]
    )({"pos": pos})
    assert torch.allclose(a["pos"], b["pos"])


def test_shift_empty_passthrough(empty_scene: dict) -> None:
    out = T.Shift(keys=["pos"], method="bbox")(empty_scene)
    assert out["pos"].shape == (0, 3)


def test_axis_min_offset_empty_passthrough(empty_scene: dict) -> None:
    out = T.AxisMinOffset(keys=["pos"], axis=2, dst_keys=["h"])(empty_scene)
    assert out["h"].shape == (0, 1)


def test_bbox_center_midpoint() -> None:
    data = {"bbox": torch.tensor([0.0, 2.0, -1.0, 4.0, 6.0, 3.0])}
    out = T.BBoxCenter(keys="bbox", dst_keys="center")(data)
    assert torch.allclose(out["center"], torch.tensor([2.0, 4.0, 1.0]))
    assert torch.equal(out["bbox"], data["bbox"])


def test_bbox_center_odd_length_raises() -> None:
    with pytest.raises(ValueError, match="even"):
        T.BBoxCenter(keys="bbox")({"bbox": torch.zeros(5)})


def test_estimate_normals_planar_patch() -> None:
    pytest.importorskip("pyg_lib")
    grid = torch.linspace(-1.0, 1.0, 20)
    xx, yy = torch.meshgrid(grid, grid, indexing="ij")
    plane = torch.stack([xx.reshape(-1), yy.reshape(-1), torch.zeros(400)], dim=1)

    normals = F.estimate_normals(plane, k=16)

    assert normals.shape == (400, 3)
    assert torch.allclose(normals.norm(dim=1), torch.ones(400), atol=1e-5)
    # The z=0 plane's normal is the z axis.
    assert torch.allclose(normals[:, 2].abs(), torch.ones(400), atol=1e-5)


def test_estimate_normals_respects_batch() -> None:
    pytest.importorskip("pyg_lib")
    grid = torch.linspace(-1.0, 1.0, 20)
    xx, yy = torch.meshgrid(grid, grid, indexing="ij")
    plane = torch.stack([xx.reshape(-1), yy.reshape(-1), torch.zeros(400)], dim=1)
    # Second cloud is the same plane translated far along x; neighbors must not cross.
    pos = torch.cat([plane, plane + torch.tensor([100.0, 0.0, 0.0])])
    batch = torch.cat([torch.zeros(400, dtype=torch.long), torch.ones(400, dtype=torch.long)])

    normals = F.estimate_normals(pos, k=16, batch=batch)

    assert normals.shape == (800, 3)
    assert torch.allclose(normals[:, 2].abs(), torch.ones(800), atol=1e-5)


def test_estimate_normals_fewer_points_than_k_raises() -> None:
    with pytest.raises(ValueError, match=r"N=5, k=10"):
        F.estimate_normals(torch.randn(5, 3), k=10)


def test_shift_bbox_centers_on_midrange() -> None:
    x = torch.tensor([[0.0, 0.0, 0.0], [2.0, 4.0, 6.0]])
    out = F.shift(x, method="bbox")
    expected = x - torch.tensor([1.0, 2.0, 3.0])  # bbox midrange
    assert torch.allclose(out, expected)


def test_shift_centroid_subtracts_mean() -> None:
    x = torch.tensor([[0.0, 0.0, 0.0], [2.0, 4.0, 6.0]])
    out = F.shift(x, method="centroid")
    assert torch.allclose(out, x - x.mean(dim=0))


def test_shift_min_aligns_positive_octant() -> None:
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    out = F.shift(x, method="min")
    assert torch.allclose(out, x - x.min(dim=0).values)
    assert out.min().item() == pytest.approx(0.0)


def test_shift_axes_subset_leaves_other_axes_untouched() -> None:
    x = torch.tensor([[0.0, 0.0, 0.0], [2.0, 4.0, 6.0]])
    out = F.shift(x, method="bbox", axes=[0, 1])  # XY only
    # XY shifted by their midranges (1, 2); Z unchanged.
    assert torch.allclose(out[:, :2], x[:, :2] - torch.tensor([1.0, 2.0]))
    assert torch.allclose(out[:, 2], x[:, 2])


def test_shift_chained_disjoint_axes_match_pointcept_centering() -> None:
    """F.shift composes the way the Pointcept-style centering recipe expects."""
    x = torch.tensor([[0.0, 0.0, 0.0], [2.0, 4.0, 6.0]])
    out = F.shift(x, method="bbox", axes=[0, 1])
    out = F.shift(out, method="min", axes=[2])
    expected = torch.tensor([[-1.0, -2.0, 0.0], [1.0, 2.0, 6.0]])
    assert torch.allclose(out, expected)


def test_functional_shift_empty_passthrough() -> None:
    x = torch.empty(0, 3)
    out = F.shift(x, method="bbox")
    assert out.shape == (0, 3)


def test_functional_shift_invalid_method_raises() -> None:
    x = torch.zeros(5, 3)
    with pytest.raises(ValueError, match="Invalid method"):
        F.shift(x, method="typo")  # type: ignore[arg-type]


def test_axis_min_offset_height_feature() -> None:
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [0.0, 0.0, 9.0]])
    out = F.axis_min_offset(x, axis=2)
    assert out.shape == (3, 1)
    # Z column [3, 6, 9], min=3; offsets are [0, 3, 6]
    assert torch.allclose(out, torch.tensor([[0.0], [3.0], [6.0]]))


def test_axis_min_offset_quantile_floor() -> None:
    # Z column 0..100; the q=0.25 quantile is 25, so offsets are z - 25 (VoteNet-style robust floor).
    z = torch.arange(101, dtype=torch.float32)
    x = torch.stack([torch.zeros_like(z), torch.zeros_like(z), z], dim=1)
    out = F.axis_min_offset(x, axis=2, quantile=0.25)
    assert out.shape == (101, 1)
    assert torch.allclose(out[:, 0], z - 25.0)


def test_axis_min_offset_preserves_dtype() -> None:
    x = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64)
    out = F.axis_min_offset(x, axis=0)
    assert out.dtype == torch.float64


def test_quantize_grid_coordinates() -> None:
    pos = torch.tensor([[-0.05, 0.0, 0.0], [0.03, 0.0, 0.0], [0.05, 0.02, 0.0], [0.031, 0.0, 0.0]])
    out = F.quantize(pos, size=0.02)
    assert out.dtype == torch.long
    assert out.shape == pos.shape
    assert torch.equal(out, torch.tensor([[0, 0, 0], [4, 0, 0], [5, 1, 0], [4, 0, 0]]))


@pytest.mark.skipif(not _PYG_LIB_AVAILABLE, reason="pyg-lib is not installed")
def test_quantize_matches_voxelize_grid_representatives() -> None:
    """Every kept voxel representative of `Voxelize(pos_reduce="grid")` carries the coordinates `quantize` gives it."""
    pos = torch.rand(200, 3) * 2.0 - 1.0
    grid = F.quantize(pos, size=0.1)
    voxelized = Voxelize(pos_key="pos", pos_reduce="grid", size=0.1, dst_inverse_key="inverse")({"pos": pos})
    assert torch.equal(voxelized["pos"][voxelized["inverse"]], grid)


def test_quantize_empty_and_invalid_size() -> None:
    assert F.quantize(torch.empty(0, 3), size=0.1).shape == (0, 3)
    with pytest.raises(ValueError, match="size"):
        F.quantize(torch.zeros(2, 3), size=0.0)


def test_axis_min_offset_empty() -> None:
    x = torch.empty(0, 3)
    out = F.axis_min_offset(x, axis=2)
    assert out.shape == (0, 1)


def test_rotation_matrix_z_90deg() -> None:
    """Rotation matrix around z by 90deg maps (1, 0, 0) to (0, 1, 0)."""

    R = F.rotation_matrix(math.pi / 2, axis=2)
    v = torch.tensor([1.0, 0.0, 0.0])
    rotated = F.rotate_vectors(v, R)
    assert torch.allclose(rotated, torch.tensor([0.0, 1.0, 0.0]), atol=1e-5)


def test_rotation_matrix_is_orthonormal_for_every_axis() -> None:
    """Rotation matrices are orthonormal: R @ R.T = I with det = 1."""
    for axis in (0, 1, 2):
        R = F.rotation_matrix(1.234, axis=axis)
        assert torch.allclose(R @ R.T, torch.eye(3), atol=1e-5)
        assert torch.allclose(torch.det(R), torch.tensor(1.0), atol=1e-5)


def test_rotation_matrix_invalid_axis_raises() -> None:
    with pytest.raises(ValueError, match="axis"):
        F.rotation_matrix(0.0, axis=3)
    with pytest.raises(ValueError, match="axis"):
        F.rotation_matrix(0.0, axis=-1)
