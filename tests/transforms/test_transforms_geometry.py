import pytest
import torch

import torch_pointcloud.transforms as T
from torch_pointcloud.utils.imports import _TORCH_CLUSTER_AVAILABLE


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


def test_align_axis() -> None:
    pos = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    data = {"pos": pos}
    transform = T.AlignAxis(keys=["pos"], dim=-1)
    result = transform(data)

    assert result["pos"][:, -1].min() == 0.0


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


@pytest.mark.skipif(not _TORCH_CLUSTER_AVAILABLE, reason="torch-cluster not installed")
def test_estimate_normals() -> None:
    grid = torch.linspace(-1.0, 1.0, 20)
    xx, yy = torch.meshgrid(grid, grid, indexing="ij")
    plane = torch.stack([xx.reshape(-1), yy.reshape(-1), torch.zeros(400)], dim=1)
    data = {"pos": plane}

    result = T.EstimateNormals(keys="pos", normal_key="normal", k=16)(data)

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


def test_align_axis_empty_passthrough(empty_scene: dict) -> None:
    out = T.AlignAxis(keys=["pos"], dim=2)(empty_scene)
    assert out["pos"].shape == (0, 3)


def test_align_axis_inplace_on_non_contiguous_does_not_raise() -> None:
    pos = torch.arange(12, dtype=torch.float32).reshape(3, 4)[:, :3]
    assert not pos.is_contiguous()
    T.AlignAxis(keys=["pos"], dim=2, inplace=True)({"pos": pos})  # should not crash


def test_bbox_center_midpoint() -> None:
    data = {"bbox": torch.tensor([0.0, 2.0, -1.0, 4.0, 6.0, 3.0])}
    out = T.BBoxCenter(keys="bbox", dst_keys="center")(data)
    assert torch.allclose(out["center"], torch.tensor([2.0, 4.0, 1.0]))
    assert torch.equal(out["bbox"], data["bbox"])


def test_bbox_center_odd_length_raises() -> None:
    with pytest.raises(ValueError, match="even"):
        T.BBoxCenter(keys="bbox")({"bbox": torch.zeros(5)})
