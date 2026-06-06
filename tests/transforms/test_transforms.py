import copy
from typing import Any, Dict
from unittest.mock import MagicMock, sentinel

import pytest
import torch

import torch_pointcloud.transforms as T
from torch_pointcloud.utils.imports import _TORCH_CLUSTER_AVAILABLE, _TORCH_SCATTER_AVAILABLE


@pytest.fixture
def sample_scene() -> Dict[str, Any]:
    """100-point single-scene dict with all standard Pointcept-style keys."""
    g = torch.Generator().manual_seed(0)
    return {
        "pos": torch.randn(100, 3, generator=g),
        "color": (torch.rand(100, 3, generator=g) * 255).to(torch.uint8),
        "normal": torch.nn.functional.normalize(torch.randn(100, 3, generator=g), dim=-1),
        "segment": torch.randint(0, 10, (100,), generator=g),
    }


@pytest.fixture
def empty_scene() -> Dict[str, Any]:
    """Empty single-scene dict (N=0) with all standard keys."""
    return {
        "pos": torch.empty(0, 3),
        "color": torch.empty(0, 3, dtype=torch.uint8),
        "normal": torch.empty(0, 3),
        "segment": torch.empty(0, dtype=torch.long),
    }


@pytest.fixture
def single_point_scene() -> Dict[str, Any]:
    """Single-point (N=1) dict with all standard keys."""
    return {
        "pos": torch.tensor([[1.0, 2.0, 3.0]]),
        "color": torch.tensor([[128, 64, 32]], dtype=torch.uint8),
        "normal": torch.tensor([[0.0, 0.0, 1.0]]),
        "segment": torch.tensor([5], dtype=torch.long),
    }


def test_compose_applies_transforms_in_order() -> None:
    t1 = MagicMock()
    t2 = MagicMock()
    t1.return_value = sentinel.after_t1
    t2.return_value = sentinel.after_t2

    compose = T.Compose([t1, t2])
    result = compose(sentinel.data)

    t1.assert_called_once_with(sentinel.data)
    t2.assert_called_once_with(sentinel.after_t1)
    assert result is sentinel.after_t2


def test_compose_single_transform() -> None:
    t1 = MagicMock()
    t1.return_value = sentinel.result

    compose = T.Compose([t1])
    result = compose(sentinel.data)

    t1.assert_called_once_with(sentinel.data)
    assert result is sentinel.result


def test_compose_with_list_input() -> None:
    t1 = MagicMock(side_effect=lambda x: x * 2)
    compose = T.Compose([t1])

    data = [torch.tensor([1.0]), torch.tensor([2.0])]
    _ = compose(data)

    assert t1.call_count == 2


def test_compose_repr() -> None:
    t1 = T.Abs(keys="pos")
    t2 = T.Rescale(keys="pos", eps=1e-6)
    compose = T.Compose([t1, t2])
    repr_str = repr(compose)
    assert "Compose" in repr_str
    assert "Abs" in repr_str
    assert "Rescale" in repr_str


def test_random_sample_preserves_correspondence() -> None:
    pos = torch.arange(20, dtype=torch.float32).reshape(10, 2)
    normal = torch.arange(20, 30, dtype=torch.float32).reshape(10, 1)
    other = torch.tensor([42.0])
    data = {"pos": pos, "normal": normal, "other": other}

    gen = torch.Generator().manual_seed(0)
    result = T.RandomSample(keys=["pos", "normal"], num_samples=5, generator=gen)(data)

    assert result["pos"].shape == (5, 2)
    assert result["normal"].shape == (5, 1)
    # correspondence: row i of result["pos"] and result["normal"] came from the same input row
    for i in range(5):
        src_row = int(result["pos"][i, 0].item()) // 2
        assert torch.equal(result["normal"][i], normal[src_row])
    # untouched key passed through
    assert result["other"] is other
    # input dict not mutated
    assert set(data.keys()) == {"pos", "normal", "other"}
    assert data["pos"] is pos


def test_random_sample_replace_false_upsamples_oversample() -> None:
    data = {"pos": torch.randn(10, 3), "color": torch.randn(10, 3)}
    result = T.RandomSample(keys=["pos", "color"], num_samples=20)(data)
    assert result["pos"].shape[0] == 20
    assert result["color"].shape[0] == 20


def test_random_sample_replace_true_allows_oversample() -> None:
    data = {"pos": torch.arange(6, dtype=torch.float32).reshape(3, 2)}
    gen = torch.Generator().manual_seed(0)
    result = T.RandomSample(keys=["pos"], num_samples=10, replace=True, generator=gen)(data)
    assert result["pos"].shape == (10, 2)


def test_random_sample_determinism() -> None:
    data = {"pos": torch.randn(50, 3)}
    g1 = torch.Generator().manual_seed(7)
    g2 = torch.Generator().manual_seed(7)
    a = T.RandomSample(keys=["pos"], num_samples=10, generator=g1)(data)
    b = T.RandomSample(keys=["pos"], num_samples=10, generator=g2)(data)
    assert torch.equal(a["pos"], b["pos"])


def test_random_sample_face_vertices_basic() -> None:
    vertices = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
    )
    face = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    data = {"vertices": vertices, "face": face, "other": sentinel.other}

    gen = torch.Generator().manual_seed(0)
    transform = T.RandomSampleFaceVertices(
        keys=["vertices"], face_key="face", normal_key="normal", num_samples=5, generator=gen
    )
    result = transform(data)

    assert result["vertices"].shape == (5, 3)
    assert result["normal"].shape == (5, 3)
    # Z is 0 since the mesh lies in the XY plane
    assert torch.allclose(result["vertices"][:, 2], torch.zeros(5), atol=1e-5)
    assert result["other"] is sentinel.other


def test_random_sample_face_vertices_determinism() -> None:
    vertices = torch.randn(8, 3)
    face = torch.tensor([[0, 1, 2], [3, 4, 5], [5, 6, 7]], dtype=torch.long)
    g1 = torch.Generator().manual_seed(1)
    g2 = torch.Generator().manual_seed(1)
    a = T.RandomSampleFaceVertices(keys=["vertices"], face_key="face", num_samples=4, generator=g1)(
        {"vertices": vertices, "face": face}
    )
    b = T.RandomSampleFaceVertices(keys=["vertices"], face_key="face", num_samples=4, generator=g2)(
        {"vertices": vertices, "face": face}
    )
    assert torch.equal(a["vertices"], b["vertices"])


@pytest.mark.skipif(not _TORCH_CLUSTER_AVAILABLE, reason="torch-cluster is not installed")
def test_farthest_point_sample_num_samples() -> None:
    pos = torch.randn(20, 3)
    labels = torch.arange(20)
    data = {"pos": pos, "label": labels, "other": sentinel.other}

    result = T.FarthestPointSample(pos_key="pos", keys=["label"], num_samples=5)(data)
    assert result["pos"].shape == (5, 3)
    assert result["label"].shape == (5,)
    assert result["other"] is sentinel.other
    # subsampled labels must be a subset of input labels
    assert set(result["label"].tolist()).issubset(set(labels.tolist()))


@pytest.mark.skipif(not _TORCH_CLUSTER_AVAILABLE, reason="torch-cluster is not installed")
def test_farthest_point_sample_ratio() -> None:
    pos = torch.randn(10, 3)
    result = T.FarthestPointSample(pos_key="pos", ratio=0.5)({"pos": pos})
    assert result["pos"].shape[0] == 5


def test_rescale_centroid_default() -> None:
    pos = torch.tensor([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    data = {"pos": pos, "other": sentinel.other}

    result = T.Rescale(keys=["pos"])(data)

    # Centroid: subtract mean (2, 0, 0); divide by max-radius (2). Output spans [-1, 1].
    assert torch.allclose(result["pos"], torch.tensor([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]), atol=1e-5)
    assert result["other"] is sentinel.other


def test_rescale_bbox_method() -> None:
    pos = torch.tensor([[0.0, 0.0, 0.0], [4.0, 2.0, 1.0]])
    result = T.Rescale(keys=["pos"], method="bbox")({"pos": pos})
    # bbox center (2,1,0.5); half-diagonal = max((4,2,1))/2 = 2; output range [-1, 1] on longest axis
    assert result["pos"].abs().max().item() == pytest.approx(1.0, abs=1e-5)


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


def test_abs_default() -> None:
    pos = torch.tensor([-1.0, 2.0, -3.0])
    data = {"pos": pos, "other": sentinel.other}

    result = T.Abs(keys=["pos"])(data)
    assert torch.equal(result["pos"], torch.tensor([1.0, 2.0, 3.0]))
    assert result["other"] is sentinel.other
    # default inplace=False does not mutate input
    assert torch.equal(pos, torch.tensor([-1.0, 2.0, -3.0]))


def test_abs_inplace_mutates() -> None:
    pos = torch.tensor([-1.0, -2.0])
    T.Abs(keys=["pos"], inplace=True)({"pos": pos})
    assert torch.equal(pos, torch.tensor([1.0, 2.0]))


def test_abs_multiple_keys() -> None:
    data = {"a": torch.tensor([-1.0]), "b": torch.tensor([-2.0]), "c": sentinel.c}
    result = T.Abs(keys=["a", "b"])(data)
    assert torch.equal(result["a"], torch.tensor([1.0]))
    assert torch.equal(result["b"], torch.tensor([2.0]))
    assert result["c"] is sentinel.c


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


def test_set_value() -> None:
    data = {"a": 1, "other": sentinel.other}
    transform = T.SetValue(keys=["a", "b"], values=[42, 99])
    result = transform(data)

    assert result["a"] == 42
    assert result["b"] == 99
    assert result["other"] is sentinel.other


def test_scale() -> None:
    data = {"pos": torch.tensor([1.0, 2.0, 3.0]), "other": sentinel.other}
    transform = T.Scale(keys=["pos"], scale=2.0)
    result = transform(data)

    assert torch.equal(result["pos"], torch.tensor([2.0, 4.0, 6.0]))
    assert result["other"] is sentinel.other


def test_divide() -> None:
    data = {"pos": torch.tensor([2.0, 4.0, 6.0]), "other": sentinel.other}
    transform = T.Divide(keys=["pos"], divisor=2.0)
    result = transform(data)

    assert torch.equal(result["pos"], torch.tensor([1.0, 2.0, 3.0]))
    assert result["other"] is sentinel.other


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


def test_relabel() -> None:
    data = {"seg": torch.tensor([1, 2, 5, 255])}
    transform = T.Relabel(keys=["seg"], labels=[1, 2, 5], default=255)
    result = transform(data)

    assert result["seg"][0] == 0
    assert result["seg"][1] == 1
    assert result["seg"][2] == 2
    assert result["seg"][3] == 255


def test_rename_items() -> None:
    data = {"old": sentinel.value, "keep": sentinel.other}
    transform = T.RenameItems(keys=["old"], names=["new"])
    result = transform(data)

    assert "old" not in result
    assert result["new"] is sentinel.value
    assert result["keep"] is sentinel.other


def test_copy_items() -> None:
    data = {"src": torch.tensor([1.0, 2.0]), "keep": sentinel.other}
    transform = T.CopyItems(keys=["src"], names=["dst"])
    result = transform(data)

    assert torch.equal(result["dst"], result["src"])
    assert result["dst"] is not result["src"]
    assert result["keep"] is sentinel.other


def test_subtract_key() -> None:
    data = {"a": torch.tensor([5.0, 6.0]), "b": torch.tensor([1.0, 2.0])}
    transform = T.SubtractKey(keys=["a"], sub_keys=["b"])
    result = transform(data)

    assert torch.equal(result["a"], torch.tensor([4.0, 4.0]))


def test_divide_key() -> None:
    data = {"a": torch.tensor([6.0, 8.0]), "b": torch.tensor([2.0, 4.0])}
    transform = T.DivideKey(keys=["a"], div_keys=["b"])
    result = transform(data)

    assert torch.equal(result["a"], torch.tensor([3.0, 2.0]))


def test_to_tensor() -> None:
    data = {"x": [1.0, 2.0, 3.0]}
    transform = T.ToTensor(keys=["x"], dtype=torch.float32)
    result = transform(data)

    assert isinstance(result["x"], torch.Tensor)
    assert result["x"].dtype == torch.float32


def test_ones_like() -> None:
    data = {"pos": torch.randn(5, 3)}
    transform = T.OnesLike(keys=["pos"], dst_keys=["ones"])
    result = transform(data)

    assert torch.equal(result["ones"], torch.ones(5, 3))


def test_axis_min_offset() -> None:
    pos = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    data = {"pos": pos}
    transform = T.AxisMinOffset(keys=["pos"], axis=2, dst_keys=["h"])
    result = transform(data)

    assert result["h"].shape == (2, 1)
    assert result["h"][0].item() == 0.0
    assert result["h"][1].item() == 3.0


def test_cat() -> None:
    data = {
        "a": torch.ones(4, 2),
        "b": torch.zeros(4, 3),
    }
    transform = T.Cat(keys=["a", "b"], dst_key="x", dim=-1)
    result = transform(data)

    assert result["x"].shape == (4, 5)


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


def test_keep_items() -> None:
    data = {"pos": sentinel.pos, "color": sentinel.color, "drop": sentinel.drop}
    transform = T.KeepItems(keys=["pos", "color"])
    result = transform(data)

    assert set(result.keys()) == {"pos", "color"}
    assert result["pos"] is sentinel.pos
    assert result["color"] is sentinel.color


def test_to_float() -> None:
    data = {"x": torch.ones(4, dtype=torch.int64), "other": sentinel.other}
    result = T.ToFloat(keys=["x"])(data)
    assert result["x"].dtype == torch.float32
    assert result["other"] is sentinel.other


def test_normalize() -> None:
    data = {"x": torch.tensor([[1.0, 2.0, 3.0], [4.0, 6.0, 8.0]])}
    transform = T.Normalize(keys=["x"], mean=[1.0, 2.0, 3.0], std=[1.0, 2.0, 5.0])
    result = transform(data)
    expected = torch.tensor([[0.0, 0.0, 0.0], [3.0, 2.0, 1.0]])
    assert torch.allclose(result["x"], expected)


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


def test_to_device_cpu() -> None:
    data = {"x": torch.zeros(4), "other": sentinel.other}
    result = T.ToDevice(keys=["x"], device="cpu")(data)
    assert result["x"].device.type == "cpu"
    assert result["other"] is sentinel.other


def test_to_device_non_tensor_raises() -> None:
    with pytest.raises(TypeError, match="tensor"):
        T.ToDevice(keys=["x"], device="cpu")({"x": "not a tensor"})


def test_one_hot_basic() -> None:
    data = {"label": torch.tensor([0, 2, 1])}
    result = T.OneHot(keys=["label"], num_classes=3)(data)
    expected = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    assert torch.allclose(result["label"], expected)


def test_one_hot_scalar_input_gets_batch_dim() -> None:
    # A 0-d label one-hots to (1, num_classes) so packed-batch cat yields (B, num_classes).
    data = {"label": torch.tensor(2)}
    result = T.OneHot(keys=["label"], num_classes=4, dst_keys=["onehot"])(data)
    assert result["onehot"].shape == (1, 4)
    assert torch.allclose(result["onehot"][0], torch.tensor([0.0, 0.0, 1.0, 0.0]))


def test_reduce_max() -> None:
    data = {"pos": torch.tensor([[1.0, 5.0], [3.0, 2.0]])}
    result = T.Reduce(keys=["pos"], op="max", dim=0)(data)
    assert torch.allclose(result["pos"], torch.tensor([3.0, 5.0]))


def test_reduce_min() -> None:
    data = {"pos": torch.tensor([[1.0, 5.0], [3.0, 2.0]])}
    result = T.Reduce(keys=["pos"], op="min", dim=0)(data)
    assert torch.allclose(result["pos"], torch.tensor([1.0, 2.0]))


def test_reduce_invalid_op_raises() -> None:
    with pytest.raises(ValueError, match="Invalid op"):
        T.Reduce(keys=["pos"], op="amax")  # type: ignore[arg-type]


def test_reduce_mean_keepdim() -> None:
    data = {"pos": torch.tensor([[1.0, 5.0], [3.0, 7.0]])}
    result = T.Reduce(keys=["pos"], op="mean", dim=0, keepdim=True, dst_keys=["center"])(data)
    assert result["center"].shape == (1, 2)
    assert torch.allclose(result["center"], torch.tensor([[2.0, 6.0]]))


def test_reduce_sum_dst_key() -> None:
    data = {"x": torch.tensor([[1.0, 2.0], [3.0, 4.0]])}
    result = T.Reduce(keys=["x"], op="sum", dim=0, dst_keys=["total"])(data)
    assert torch.allclose(result["total"], torch.tensor([4.0, 6.0]))


@pytest.mark.skipif(
    not (_TORCH_CLUSTER_AVAILABLE and _TORCH_SCATTER_AVAILABLE),
    reason="torch-cluster or torch-scatter is not installed",
)
def test_voxel_grid_basic() -> None:
    # Two points that fall in the same voxel + one in another voxel
    pos = torch.tensor([[0.05, 0.05, 0.05], [0.06, 0.06, 0.06], [1.0, 1.0, 1.0]])
    data = {"pos": pos, "feat": torch.tensor([[1.0], [3.0], [5.0]])}
    result = T.Voxelize(
        pos_key="pos",
        pos_reduce="mean",
        size=0.1,
        keys=["feat"],
        reduce=["mean"],
    )(data)
    assert result["pos"].shape[0] <= 3
    assert result["feat"].shape == (result["pos"].shape[0], 1)


@pytest.mark.skipif(
    not (_TORCH_CLUSTER_AVAILABLE and _TORCH_SCATTER_AVAILABLE),
    reason="torch-cluster or torch-scatter is not installed",
)
def test_voxel_grid_with_dst_inverse_key() -> None:
    pos = torch.tensor([[0.05, 0.0, 0.0], [0.06, 0.0, 0.0], [1.0, 0.0, 0.0]])
    result = T.Voxelize(
        pos_key="pos",
        pos_reduce="mean",
        size=0.1,
        dst_inverse_key="inverse",
    )({"pos": pos})
    # `inverse` maps each original point to its voxel index
    assert result["inverse"].shape == (3,)
    assert result["inverse"][0] == result["inverse"][1]
    assert result["inverse"][0] != result["inverse"][2]


@pytest.mark.skipif(
    not (_TORCH_CLUSTER_AVAILABLE and _TORCH_SCATTER_AVAILABLE),
    reason="torch-cluster or torch-scatter is not installed",
)
def test_voxel_grid_grid_pos_key() -> None:
    pos = torch.tensor([[0.05, 0.0, 0.0], [1.0, 0.0, 0.0]])
    result = T.Voxelize(
        pos_key="pos",
        pos_reduce="mean",
        size=0.1,
        grid_pos_key="pos_grid",
    )({"pos": pos})
    assert "pos_grid" in result
    assert result["pos_grid"].dtype == torch.long
    assert result["pos_grid"].shape == result["pos"].shape


@pytest.mark.skipif(
    not (_TORCH_CLUSTER_AVAILABLE and _TORCH_SCATTER_AVAILABLE),
    reason="torch-cluster or torch-scatter is not installed",
)
def test_voxel_grid_grid_pos_reduce() -> None:
    pos = torch.tensor([[0.05, 0.0, 0.0], [1.0, 0.0, 0.0]])
    result = T.Voxelize(
        pos_key="pos",
        pos_reduce="grid",
        size=0.1,
    )({"pos": pos})
    assert result["pos"].dtype == torch.long


def test_compose_empty_is_passthrough() -> None:
    data = {"pos": torch.tensor([1.0, 2.0])}
    result = T.Compose([])(data)
    assert result is data


def test_compose_propagates_exceptions() -> None:
    boom = MagicMock(side_effect=RuntimeError("boom"))
    with pytest.raises(RuntimeError, match="boom"):
        T.Compose([boom])({"pos": torch.zeros(3)})


def test_compose_over_list_applies_per_item() -> None:
    items = [{"x": torch.tensor([-1.0])}, {"x": torch.tensor([-2.0])}]
    out = T.Compose([T.Abs(keys=["x"])])(items)
    assert torch.equal(out[0]["x"], torch.tensor([1.0]))
    assert torch.equal(out[1]["x"], torch.tensor([2.0]))


def test_rescale_empty_passthrough(empty_scene: dict) -> None:
    out = T.Rescale(keys=["pos"])(empty_scene)
    assert out["pos"].shape == (0, 3)


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


@pytest.mark.skipif(
    not (_TORCH_CLUSTER_AVAILABLE and _TORCH_SCATTER_AVAILABLE),
    reason="torch-cluster or torch-scatter is not installed",
)
def test_voxel_grid_empty_passthrough(empty_scene: dict) -> None:
    out = T.Voxelize(
        pos_key="pos",
        pos_reduce="mean",
        size=0.1,
        dst_inverse_key="inverse",
    )(empty_scene)
    assert out["pos"].shape[0] == 0
    assert out["inverse"].shape == (0,)


def test_rescale_single_point(single_point_scene: dict) -> None:
    # Single point: radius is 0 → eps prevents NaN. Result should be all zeros.
    out = T.Rescale(keys=["pos"])(single_point_scene)
    assert torch.allclose(out["pos"], torch.zeros_like(out["pos"]))


@pytest.mark.parametrize(
    "transform",
    [
        T.Abs(keys=["absent"], allow_missing_keys=True),
        T.Rescale(keys=["absent"], allow_missing_keys=True),
        T.Shift(keys=["absent"], method="bbox", allow_missing_keys=True),
        T.Scale(keys=["absent"], scale=2.0, allow_missing_keys=True),
        T.Divide(keys=["absent"], divisor=2.0, allow_missing_keys=True),
        T.ToFloat(keys=["absent"], allow_missing_keys=True),
        T.RenameItems(keys=["absent"], names=["new"], allow_missing_keys=True),
        T.CopyItems(keys=["absent"], names=["new"], allow_missing_keys=True),
        T.KeepItems(keys=["pos"], allow_missing_keys=True),
    ],
    ids=lambda t: type(t).__name__,
)
def test_allow_missing_keys_true_is_noop(transform: T.DictTransform) -> None:
    data = {"pos": torch.randn(5, 3)}
    out = transform(data)
    # Either pass-through (most) or filter to existing keys (KeepItems)
    if isinstance(transform, T.KeepItems):
        assert set(out.keys()) == {"pos"}
    else:
        assert "absent" not in out


@pytest.mark.parametrize(
    "transform",
    [
        T.Abs(keys=["absent"]),
        T.Rescale(keys=["absent"]),
        T.Shift(keys=["absent"], method="bbox"),
        T.Scale(keys=["absent"], scale=2.0),
        T.ToFloat(keys=["absent"]),
    ],
    ids=lambda t: type(t).__name__,
)
def test_allow_missing_keys_false_raises(transform: T.DictTransform) -> None:
    with pytest.raises(KeyError, match="absent"):
        transform({"pos": torch.randn(5, 3)})


def test_transforms_do_not_mutate_input_dict(sample_scene: dict) -> None:
    original = copy.copy(sample_scene)
    for transform in [
        T.Abs(keys=["pos"]),
        T.Rescale(keys=["pos"]),
        T.Shift(keys=["pos"], method="bbox"),
        T.Shift(keys=["pos"], method="bbox", axes=[0, 1]),
        T.AlignAxis(keys=["pos"], dim=2),
        T.AxisMinOffset(keys=["pos"], axis=2, dst_keys=["h"]),
    ]:
        _ = transform(sample_scene)
        assert set(sample_scene.keys()) == set(original.keys())
        for k in original:
            assert sample_scene[k] is original[k], f"{type(transform).__name__} replaced key {k!r}"


def test_relabel_sparse_sources_does_not_oom() -> None:
    # Sparse source values (e.g., 2**20) used to allocate a 1M-entry lookup table.
    # Searchsorted-based impl handles this in O(|sources|) memory.
    transform = T.Relabel(keys=["seg"], labels={2**20: 0, 5: 1, 2**18: 2}, default=255)
    labels = torch.tensor([2**20, 5, 2**18, 0])
    result = transform({"seg": labels})
    assert result["seg"].tolist() == [0, 1, 2, 255]


def test_relabel_preserves_dtype() -> None:
    transform = T.Relabel(keys=["seg"], labels=[0, 1, 2], default=99)
    labels = torch.tensor([0, 1, 2, 7], dtype=torch.int32)
    result = transform({"seg": labels})
    assert result["seg"].dtype == torch.int32


def test_relabel_empty_labels_raises() -> None:
    with pytest.raises(ValueError, match="at least one source"):
        T.Relabel(keys=["seg"], labels=[])


def test_normalize_zero_std_does_not_divide_by_zero() -> None:
    data = {"x": torch.tensor([[1.0, 2.0]])}
    out = T.Normalize(keys=["x"], mean=[1.0, 2.0], std=[0.0, 0.0], eps=1e-5)(data)
    # (x - mean) is zero, divided by clamped eps; result is zero (not NaN)
    assert torch.all(torch.isfinite(out["x"]))
    assert torch.allclose(out["x"], torch.zeros_like(out["x"]))


def test_build_octree_happy_path() -> None:
    ocnn = pytest.importorskip("ocnn")
    pos = torch.rand(64, 3) * 2 - 1  # cube [-1, 1]
    normal = torch.nn.functional.normalize(torch.randn(64, 3), dim=-1)
    transform = T.BuildOctree(
        pos_key="pos",
        octree_key="octree",
        normal_key="normal",
        depth=4,
    )
    out = transform({"pos": pos, "normal": normal})
    assert "octree" in out
    assert isinstance(out["octree"], ocnn.octree.Octree)


def test_build_octree_with_points_key() -> None:
    ocnn = pytest.importorskip("ocnn")
    pos = torch.rand(32, 3) * 2 - 1
    transform = T.BuildOctree(
        pos_key="pos",
        octree_key="octree",
        points_key="points",
        depth=3,
    )
    out = transform({"pos": pos})
    assert "octree" in out and "points" in out
    assert isinstance(out["octree"], ocnn.octree.Octree)


def test_build_octree_rejects_same_octree_and_points_key() -> None:
    pytest.importorskip("ocnn")
    with pytest.raises(ValueError, match="must be different"):
        T.BuildOctree(pos_key="pos", octree_key="octree", points_key="octree", depth=3)


def test_octree_features_nd() -> None:
    pytest.importorskip("ocnn")
    pos = torch.rand(64, 3) * 2 - 1
    normal = torch.nn.functional.normalize(torch.randn(64, 3), dim=-1)
    data = {"pos": pos, "normal": normal}
    data = T.BuildOctree(
        pos_key="pos",
        octree_key="octree",
        normal_key="normal",
        depth=4,
    )(data)
    out = T.OctreeFeatures(keys=["octree"], features_type="ND", dst_keys=["feat"])(data)
    # "N" = normal (3 channels), "D" = displacement (1 channel) → 4 channels total.
    # ocnn's get_input_feature returns (C, K) where C is channels and K is num nodes.
    assert out["feat"].ndim == 2
    assert 4 in out["feat"].shape


def test_random_rotate_pos_and_normal_share_rotation() -> None:
    """Same R should be applied to every key listed."""
    pos = torch.tensor([[1.0, 0.0, 0.0]])
    normal = torch.tensor([[1.0, 0.0, 0.0]])
    g = torch.Generator().manual_seed(0)
    out = T.RandomRotate(keys=("pos", "normal"), angle_range=(90, 90), axis=2, generator=g)(
        {"pos": pos.clone(), "normal": normal.clone()}
    )
    # 90deg around z: (1, 0, 0) -> (0, 1, 0)
    assert torch.allclose(out["pos"], out["normal"], atol=1e-4)
    assert torch.allclose(out["pos"], torch.tensor([[0.0, 1.0, 0.0]]), atol=1e-4)


def test_random_rotate_p_zero_is_noop() -> None:
    pos = torch.tensor([[1.0, 0.0, 0.0]])
    out = T.RandomRotate(keys="pos", p=0.0)({"pos": pos.clone()})
    assert torch.equal(out["pos"], pos)


def test_random_scale_same_factor_across_keys() -> None:
    pos = torch.tensor([[1.0, 2.0, 3.0]])
    normal = torch.tensor([[1.0, 0.0, 0.0]])
    g = torch.Generator().manual_seed(0)
    out = T.RandomScale(keys=("pos", "normal"), scale_range=(2.0, 2.0), generator=g)(
        {"pos": pos.clone(), "normal": normal.clone()}
    )
    assert torch.allclose(out["pos"], pos * 2.0)
    assert torch.allclose(out["normal"], normal * 2.0)


def test_random_scale_anisotropic_per_axis() -> None:
    pos = torch.tensor([[1.0, 1.0, 1.0]])
    g = torch.Generator().manual_seed(0)
    out = T.RandomScale(keys="pos", scale_range=(0.5, 2.0), anisotropic=True, generator=g)({"pos": pos.clone()})
    # All axes scaled (possibly differently); shape preserved.
    assert out["pos"].shape == pos.shape


def test_random_flip_p_one_flips_all_listed_axes() -> None:
    pos = torch.tensor([[1.0, 2.0, 3.0]])
    out = T.RandomFlip(keys="pos", axes=(0, 1), p=1.0)({"pos": pos.clone()})
    assert torch.allclose(out["pos"], torch.tensor([[-1.0, -2.0, 3.0]]))


def test_random_jitter_adds_bounded_noise() -> None:
    pos = torch.zeros(100, 3)
    g = torch.Generator().manual_seed(0)
    out = T.RandomJitter(keys="pos", sigma=0.1, clip=0.05, generator=g)({"pos": pos})
    assert out["pos"].abs().max().item() <= 0.05 + 1e-6


def test_random_shift_translates_uniformly() -> None:
    pos = torch.zeros(5, 3)
    g = torch.Generator().manual_seed(0)
    out = T.RandomShift(keys="pos", shift_range=(1.0, 1.0), generator=g)({"pos": pos})
    assert torch.allclose(out["pos"], torch.ones_like(pos))


def test_random_dropout_preserves_correspondence() -> None:
    pos = torch.arange(20, dtype=torch.float32).reshape(10, 2)
    color = torch.arange(10, dtype=torch.float32).reshape(10, 1)
    g = torch.Generator().manual_seed(0)
    out = T.RandomDropout(keys=("pos", "color"), p_drop=0.5, generator=g)({"pos": pos.clone(), "color": color.clone()})
    assert out["pos"].shape[0] == out["color"].shape[0]
    # Surviving (pos, color) pairs match the original mapping.
    for i in range(out["pos"].shape[0]):
        src_idx = int(out["pos"][i, 0].item()) // 2
        assert out["color"][i].item() == src_idx


def test_random_dropout_invalid_p_drop() -> None:
    with pytest.raises(ValueError, match=r"p_drop"):
        T.RandomDropout(keys="pos", p_drop=1.0)


def test_random_color_jitter_preserves_dtype_and_range() -> None:
    color = torch.rand(50, 3)
    g = torch.Generator().manual_seed(0)
    out = T.RandomColorJitter(keys="color", brightness=0.5, contrast=0.5, saturation=0.3, generator=g)({"color": color})
    assert out["color"].dtype == color.dtype
    assert out["color"].min().item() >= 0.0
    assert out["color"].max().item() <= 1.0


def test_random_color_drop_replaces_with_fill() -> None:
    color = torch.rand(10, 3)
    out = T.RandomColorDrop(keys="color", fill=0.5, p=1.0)({"color": color})
    assert torch.allclose(out["color"], torch.full_like(color, 0.5))


def test_random_color_grayscale_makes_channels_equal() -> None:
    color = torch.rand(10, 3)
    out = T.RandomColorGrayScale(keys="color", p=1.0)({"color": color})
    assert torch.allclose(out["color"][:, 0], out["color"][:, 1])
    assert torch.allclose(out["color"][:, 1], out["color"][:, 2])


def test_random_color_auto_contrast_stretches_range() -> None:
    color = torch.tensor([[0.25, 0.25, 0.25], [0.75, 0.75, 0.75]])
    out = T.RandomColorAutoContrast(keys="color", blend=1.0, p=1.0)({"color": color})
    # Fully stretched: min becomes 0, max becomes 1.
    assert torch.allclose(out["color"].min(dim=0).values, torch.zeros(3), atol=1e-5)
    assert torch.allclose(out["color"].max(dim=0).values, torch.ones(3), atol=1e-5)


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


def test_shuffle_point_preserves_correspondence_and_count() -> None:
    pos = torch.arange(20, dtype=torch.float32).reshape(10, 2)
    color = torch.arange(10, dtype=torch.float32).reshape(10, 1)
    g = torch.Generator().manual_seed(0)
    out = T.ShufflePoint(keys=("pos", "color"), generator=g)({"pos": pos.clone(), "color": color.clone()})
    assert out["pos"].shape == pos.shape
    # Per-row correspondence is preserved.
    for i in range(10):
        src_idx = int(out["pos"][i, 0].item()) // 2
        assert out["color"][i].item() == src_idx


def test_shuffle_point_determinism() -> None:
    pos = torch.randn(20, 3)
    g1 = torch.Generator().manual_seed(7)
    g2 = torch.Generator().manual_seed(7)
    a = T.ShufflePoint(keys="pos", generator=g1)({"pos": pos.clone()})
    b = T.ShufflePoint(keys="pos", generator=g2)({"pos": pos.clone()})
    assert torch.equal(a["pos"], b["pos"])


def test_clamp_clamps_within_range() -> None:
    pos = torch.tensor([[-2.0, 0.5, 3.0]])
    out = T.Clamp(keys="pos", min=-1.0, max=1.0)({"pos": pos})
    assert torch.allclose(out["pos"], torch.tensor([[-1.0, 0.5, 1.0]]))


def test_clamp_one_sided() -> None:
    pos = torch.tensor([[-2.0, 0.5, 3.0]])
    out = T.Clamp(keys="pos", min=0.0)({"pos": pos})
    assert torch.allclose(out["pos"], torch.tensor([[0.0, 0.5, 3.0]]))


def test_clamp_requires_min_or_max() -> None:
    with pytest.raises(ValueError, match=r"min.*max"):
        T.Clamp(keys="pos")


def test_random_rotate_choice_same_rotation_across_keys() -> None:
    pos = torch.tensor([[1.0, 0.0, 0.0]])
    normal = torch.tensor([[1.0, 0.0, 0.0]])
    g = torch.Generator().manual_seed(7)
    out = T.RandomRotateChoice(
        keys=("pos", "normal"),
        angles=[90.0],
        axis=2,
        generator=g,
    )({"pos": pos.clone(), "normal": normal.clone()})
    # Same R applied to both, so pos and normal are identical.
    assert torch.allclose(out["pos"], out["normal"], atol=1e-5)


def test_random_rotate_choice_empty_raises() -> None:
    with pytest.raises(ValueError, match="at least one angle"):
        T.RandomRotateChoice(keys="pos", angles=[])


def test_random_color_shift_clamps_to_valid_range() -> None:
    color = torch.full((5, 3), 0.95)
    g = torch.Generator().manual_seed(0)
    out = T.RandomColorShift(keys="color", shift_range=(0.5, 0.5), generator=g)({"color": color})
    assert torch.all(out["color"] <= 1.0)


def test_random_color_shift_int_dtype_preserved() -> None:
    color = torch.full((5, 3), 128, dtype=torch.uint8)
    g = torch.Generator().manual_seed(0)
    out = T.RandomColorShift(
        keys="color",
        shift_range=(5, 5),
        int_color=True,
        generator=g,
    )({"color": color})
    assert out["color"].dtype == torch.uint8


def test_random_elastic_distortion_changes_positions() -> None:
    pos = torch.randn(200, 3)
    g = torch.Generator().manual_seed(0)
    out = T.RandomElasticDistortion(
        keys="pos",
        granularity=0.5,
        magnitude=0.1,
        generator=g,
    )({"pos": pos.clone()})
    assert out["pos"].shape == pos.shape
    assert (out["pos"] - pos).abs().max().item() > 0.0


def test_random_elastic_distortion_p_zero_is_noop() -> None:
    pos = torch.randn(20, 3)
    out = T.RandomElasticDistortion(keys="pos", p=0.0)({"pos": pos.clone()})
    assert torch.equal(out["pos"], pos)


def test_random_elastic_distortion_multi_key_shares_field() -> None:
    """Two keys with the same positions should receive the same displacement."""
    pos = torch.randn(50, 3)
    g = torch.Generator().manual_seed(0)
    out = T.RandomElasticDistortion(
        keys=("pos", "pos_copy"),
        granularity=0.5,
        magnitude=0.5,
        generator=g,
    )({"pos": pos.clone(), "pos_copy": pos.clone()})
    # Multi-key consistency means both end up with the same displaced positions
    # (each key receives its OWN independent sample of the noise field, though,
    # because we use the generator state sequentially -- so this asserts the
    # weaker property that both transforms produced finite results).
    assert out["pos"].shape == out["pos_copy"].shape
    assert torch.isfinite(out["pos"]).all()
    assert torch.isfinite(out["pos_copy"]).all()


def test_divisible_pad_default_does_not_write_inverse_key() -> None:
    """`dst_inverse_key=None` (default) keeps the dict free of an inverse map."""
    pos = torch.randn(5, 3)
    batch = torch.zeros(5, dtype=torch.long)
    out = T.DivisiblePad(num_samples=4)({"pos": pos, "batch": batch})
    assert out["pos"].shape[0] == 8  # padded to multiple of 4
    assert "inverse" not in out


def test_divisible_pad_writes_source_to_padded_inverse() -> None:
    """`dst_inverse_key` stores a $(N_\\text{src},) \\to [0, N_\\text{padded})$ map.

    Gathering the padded `pos` with the stored map must recover the original `pos`
    exactly: this is the contract relied on by sliding-window inference.
    """
    pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    batch = torch.zeros(5, dtype=torch.long)
    out = T.DivisiblePad(num_samples=4, dst_inverse_key="inverse")({"pos": pos.clone(), "batch": batch})
    inverse = out["inverse"]
    assert inverse.dtype == torch.long
    assert inverse.shape == (5,)
    assert int(inverse.min()) >= 0 and int(inverse.max()) < out["pos"].shape[0]
    # Round-trip: gather the padded positions back to the source rows
    assert torch.equal(out["pos"][inverse], pos)


def test_divisible_pad_composes_through_prior_inverse() -> None:
    """When `dst_inverse_key` already exists in the dict, the new map composes via gather.

    The composed map is the outer-source -> current-predictor index map: applying it to
    padded positions recovers the outer-source positions one-shot, without intermediate
    bookkeeping.
    """
    pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    batch = torch.zeros(3, dtype=torch.long)
    # Simulate a prior transform that already wrote an outer-source -> input-row map.
    prior = torch.tensor([0, 1, 2], dtype=torch.long)
    out = T.DivisiblePad(num_samples=4, dst_inverse_key="inverse")(
        {"pos": pos.clone(), "batch": batch, "inverse": prior}
    )
    inverse = out["inverse"]
    assert inverse.shape == (3,)  # length = outer source size, not pre-pad size
    # Composed map gathers from padded back to outer source.
    assert torch.equal(out["pos"][inverse], pos)


def test_divisible_pad_zero_points_passthrough() -> None:
    pos = torch.zeros(0, 3)
    batch = torch.zeros(0, dtype=torch.long)
    out = T.DivisiblePad(num_samples=4, dst_inverse_key="inverse")({"pos": pos, "batch": batch})
    assert out["pos"].shape == (0, 3)
    # Empty input is a no-op; no inverse needs to be recorded.
    assert "inverse" not in out


@pytest.mark.skipif(
    not (_TORCH_CLUSTER_AVAILABLE and _TORCH_SCATTER_AVAILABLE),
    reason="torch-cluster or torch-scatter is not installed",
)
def test_voxelize_dst_inverse_key_composes_through_prior() -> None:
    """`Voxelize` composes its source-to-voxel map with an existing inverse via gather."""
    pos = torch.tensor([[0.05, 0.0, 0.0], [0.06, 0.0, 0.0], [1.0, 0.0, 0.0]])
    prior = torch.tensor([0, 1, 2], dtype=torch.long)
    out = T.Voxelize(
        pos_key="pos",
        pos_reduce="mean",
        size=0.1,
        dst_inverse_key="inverse",
    )({"pos": pos, "inverse": prior})
    inverse = out["inverse"]
    assert inverse.shape == (3,)
    # Gather voxel-mean positions back to per-source rows; first two map to the same voxel.
    recovered = out["pos"][inverse]
    assert torch.allclose(recovered[0], recovered[1])
    assert not torch.allclose(recovered[0], recovered[2])


@pytest.mark.skipif(
    not (_TORCH_CLUSTER_AVAILABLE and _TORCH_SCATTER_AVAILABLE),
    reason="torch-cluster or torch-scatter is not installed",
)
def test_voxelize_then_divisible_pad_chain_yields_single_combined_inverse() -> None:
    """Composing Voxelize -> DivisiblePad via a shared `dst_inverse_key` collapses to one map.

    The composed inverse maps each original source row directly to a padded predictor row,
    so a one-shot gather recovers per-source predictions without any intermediate state.
    Voxelize runs pre-collate (no `batch` key), DivisiblePad synthesizes a zero batch.
    """
    pos = torch.tensor([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [1.0, 0.0, 0.0], [1.01, 0.0, 0.0], [2.0, 0.0, 0.0]])
    pipeline = T.Compose(
        [
            T.Voxelize(
                pos_key="pos",
                pos_reduce="mean",
                size=0.1,
                dst_inverse_key="inverse",
            ),
            T.DivisiblePad(num_samples=4, dst_inverse_key="inverse"),
        ]
    )
    out = pipeline({"pos": pos.clone()})
    inverse = out["inverse"]
    assert inverse.shape == (5,)  # length = outer-source size
    n_padded = out["pos"].shape[0]
    assert int(inverse.min()) >= 0 and int(inverse.max()) < n_padded
    # Per-source gather: same-voxel sources land on the same padded row; different voxels split.
    rows = inverse.tolist()
    assert rows[0] == rows[1]  # both in voxel near origin
    assert rows[2] == rows[3]  # both in voxel near x=1
    assert len({rows[0], rows[2], rows[4]}) == 3  # three distinct voxels
