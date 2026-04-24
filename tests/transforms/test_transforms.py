from unittest.mock import MagicMock, Mock, patch, sentinel

import torch

from torch_pointcloud.transforms.transforms import (
    Abs,
    AlignAxis,
    ApplyMask,
    AxisMinOffset,
    BallMask,
    Cat,
    Center,
    Compose,
    CopyItems,
    Divide,
    DivideKey,
    InboxMask,
    KeepItems,
    NormalizeScale,
    OnesLike,
    RandomSample,
    RandomSampleFaceVertices,
    Relabel,
    RemoveNearOrigin,
    RenameItems,
    SampleFarthestPoints,
    Scale,
    SetValue,
    SubtractKey,
    ToTensor,
)


def test_compose_applies_transforms_in_order() -> None:
    t1 = MagicMock()
    t2 = MagicMock()
    t1.return_value = sentinel.after_t1
    t2.return_value = sentinel.after_t2

    compose = Compose([t1, t2])
    result = compose(sentinel.data)

    t1.assert_called_once_with(sentinel.data)
    t2.assert_called_once_with(sentinel.after_t1)
    assert result is sentinel.after_t2


def test_compose_single_transform() -> None:
    t1 = MagicMock()
    t1.return_value = sentinel.result

    compose = Compose([t1])
    result = compose(sentinel.data)

    t1.assert_called_once_with(sentinel.data)
    assert result is sentinel.result


def test_compose_with_list_input() -> None:
    t1 = MagicMock(side_effect=lambda x: x * 2)
    compose = Compose([t1])

    data = [torch.tensor([1.0]), torch.tensor([2.0])]
    _ = compose(data)

    assert t1.call_count == 2


def test_compose_repr() -> None:
    t1 = Abs(keys="pos")
    t2 = NormalizeScale(keys="pos", eps=1e-6)
    compose = Compose([t1, t2])
    repr_str = repr(compose)
    assert "Compose" in repr_str
    assert "Abs" in repr_str
    assert "NormalizeScale" in repr_str


@patch("torch_pointcloud.transforms.transforms.F.random_sample")
def test_random_sample(mock_fn: Mock) -> None:
    sampled_tensor = sentinel.sampled_tensor
    sampled_indices = sentinel.indices
    mock_fn.return_value = (sampled_tensor, sampled_indices)

    data = {"pos": MagicMock(), "normal": MagicMock(), "other": MagicMock()}
    transform = RandomSample(keys=["pos", "normal"], num_samples=10)
    result = transform(data)

    mock_fn.assert_called_once_with(data["pos"], 10, return_indices=True, generator=None)
    assert result["pos"] is sampled_tensor
    assert result["normal"] is data["normal"][sampled_indices]
    assert result["other"] is data["other"]


@patch("torch_pointcloud.transforms.transforms.F.random_sample_face_vertices")
def test_random_sample_face_vertices(mock_fn: Mock) -> None:
    mock_fn.return_value = (sentinel.sampled_vertices, sentinel.sampled_normals)

    data = {"vertices": MagicMock(), "face": MagicMock(), "other": MagicMock()}
    transform = RandomSampleFaceVertices(keys=["vertices"], face_key="face", normal_key="normal", num_samples=5)
    result = transform(data)

    mock_fn.assert_called_once_with(data["vertices"], data["face"], 5, generator=None, return_normals=True)
    assert result["vertices"] is sentinel.sampled_vertices
    assert result["normal"] is sentinel.sampled_normals
    assert result["other"] is data["other"]


@patch("torch_pointcloud.transforms.transforms.F.sample_farthest_points")
def test_sample_farthest_points(mock_fn: Mock) -> None:
    indices = torch.tensor([0, 3, 7])
    mock_fn.return_value = indices

    pos = torch.randn(10, 3)
    labels = torch.arange(10)
    data = {"pos": pos, "label": labels, "other": sentinel.other}

    transform = SampleFarthestPoints(pos_key="pos", keys=["label"], num_samples=3)
    result = transform(data)

    mock_fn.assert_called_once_with(pos, num_samples=3, ratio=None, random_start=False)
    assert torch.equal(result["pos"], pos[indices])
    assert torch.equal(result["label"], labels[indices])
    assert result["other"] is sentinel.other


@patch("torch_pointcloud.transforms.transforms.F.sample_farthest_points")
def test_sample_farthest_points_ratio(mock_fn: Mock) -> None:
    mock_fn.return_value = torch.tensor([0, 2])
    data = {"pos": torch.randn(5, 3)}

    transform = SampleFarthestPoints(pos_key="pos", ratio=0.5)
    transform(data)

    mock_fn.assert_called_once_with(data["pos"], num_samples=None, ratio=0.5, random_start=False)


@patch("torch_pointcloud.transforms.transforms.F.sample_farthest_points")
def test_sample_farthest_points_random_start(mock_fn: Mock) -> None:
    mock_fn.return_value = torch.tensor([0])
    data = {"pos": torch.randn(5, 3)}

    transform = SampleFarthestPoints(pos_key="pos", num_samples=1, random_start=True)
    transform(data)

    mock_fn.assert_called_once_with(data["pos"], num_samples=1, ratio=None, random_start=True)


@patch("torch_pointcloud.transforms.transforms.F.normalize_scale")
def test_normalize_scale(mock_fn: Mock) -> None:
    mock_fn.return_value = sentinel.normalized
    data = {"pos": MagicMock(), "other": sentinel.other}

    transform = NormalizeScale(keys=["pos"])
    result = transform(data)

    mock_fn.assert_called_once_with(data["pos"], eps=1e-6, method="centroid")
    assert result["pos"] is sentinel.normalized
    assert result["other"] is sentinel.other


@patch("torch_pointcloud.transforms.transforms.F.normalize_scale")
def test_normalize_scale_bbox_method(mock_fn: Mock) -> None:
    mock_fn.return_value = sentinel.normalized
    data = {"pos": MagicMock()}

    transform = NormalizeScale(keys=["pos"], method="bbox", eps=1e-8)
    transform(data)

    mock_fn.assert_called_once_with(data["pos"], eps=1e-8, method="bbox")


@patch("torch_pointcloud.transforms.transforms.F.remove_near_origin")
def test_remove_near_origin(mock_fn: Mock) -> None:
    mask = torch.tensor([False, True, True, False])
    mock_fn.return_value = (sentinel.filtered_pos, mask)

    pos = torch.randn(4, 3)
    labels = torch.tensor([0, 1, 2, 3])
    data = {"pos": pos, "label": labels, "other": sentinel.other}

    transform = RemoveNearOrigin(pos_key="pos", keys=["label"], radius=0.01)
    result = transform(data)

    mock_fn.assert_called_once_with(pos, radius=0.01, return_mask=True)
    assert torch.equal(result["pos"], pos[mask])
    assert torch.equal(result["label"], labels[mask])
    assert result["other"] is sentinel.other


@patch("torch_pointcloud.transforms.transforms.F.remove_near_origin")
def test_remove_near_origin_defaults(mock_fn: Mock) -> None:
    mock_fn.return_value = (sentinel.filtered, torch.tensor([True]))
    data = {"pos": torch.randn(1, 3)}

    transform = RemoveNearOrigin(pos_key="pos")
    transform(data)

    mock_fn.assert_called_once_with(data["pos"], radius=1e-3, return_mask=True)


@patch("torch_pointcloud.transforms.transforms.F.abs")
def test_abs_default(mock_fn: Mock) -> None:
    mock_fn.return_value = sentinel.abs_result
    data = {"pos": MagicMock(), "other": sentinel.other}

    transform = Abs(keys=["pos"])
    result = transform(data)

    mock_fn.assert_called_once_with(data["pos"], inplace=False)
    assert result["pos"] is sentinel.abs_result
    assert result["other"] is sentinel.other


@patch("torch_pointcloud.transforms.transforms.F.abs")
def test_abs_inplace(mock_fn: Mock) -> None:
    mock_fn.return_value = sentinel.result
    data = {"pos": MagicMock()}

    transform = Abs(keys=["pos"], inplace=True)
    transform(data)

    mock_fn.assert_called_once_with(data["pos"], inplace=True)


@patch("torch_pointcloud.transforms.transforms.F.abs")
def test_abs_multiple_keys(mock_fn: Mock) -> None:
    mock_fn.side_effect = [sentinel.abs_a, sentinel.abs_b]
    data = {"a": MagicMock(), "b": MagicMock(), "c": sentinel.c}

    transform = Abs(keys=["a", "b"])
    result = transform(data)

    assert mock_fn.call_count == 2
    assert result["a"] is sentinel.abs_a
    assert result["b"] is sentinel.abs_b
    assert result["c"] is sentinel.c


@patch("torch_pointcloud.transforms.transforms.F.inbox_mask")
def test_inbox_mask(mock_fn: Mock) -> None:
    mock_fn.return_value = sentinel.mask
    data = {"pos": MagicMock()}

    transform = InboxMask(keys=["pos"], bbox=(0.0, 0.0, 1.0, 1.0))
    result = transform(data)

    mock_fn.assert_called_once_with(data["pos"], (0.0, 0.0, 1.0, 1.0), dim=-1)
    assert result["pos"] is sentinel.mask


@patch("torch_pointcloud.transforms.transforms.F.inbox_mask")
def test_inbox_mask_with_dst_keys(mock_fn: Mock) -> None:
    mock_fn.return_value = sentinel.mask
    data = {"pos": MagicMock()}

    transform = InboxMask(keys=["pos"], bbox=(0.0, 1.0), dst_keys=["mask"], dim=0)
    result = transform(data)

    mock_fn.assert_called_once_with(data["pos"], (0.0, 1.0), dim=0)
    assert result["mask"] is sentinel.mask
    assert result["pos"] is data["pos"]


@patch("torch_pointcloud.transforms.transforms.F.apply_mask")
def test_apply_mask(mock_fn: Mock) -> None:
    mock_fn.return_value = sentinel.masked
    mask = sentinel.mask
    data = {"pos": MagicMock(), "mask": mask, "other": sentinel.other}

    transform = ApplyMask(keys=["pos"], mask_key="mask")
    result = transform(data)

    mock_fn.assert_called_once_with(data["pos"], mask)
    assert result["pos"] is sentinel.masked
    assert result["other"] is sentinel.other


@patch("torch_pointcloud.transforms.transforms.F.apply_mask")
def test_apply_mask_with_dst_keys(mock_fn: Mock) -> None:
    mock_fn.return_value = sentinel.masked
    data = {"pos": MagicMock(), "mask": sentinel.mask}

    transform = ApplyMask(keys=["pos"], mask_key="mask", dst_keys=["filtered"])
    result = transform(data)

    assert result["filtered"] is sentinel.masked
    assert result["pos"] is data["pos"]


def test_set_value() -> None:
    data = {"a": 1, "other": sentinel.other}
    transform = SetValue(keys=["a", "b"], values=[42, 99])
    result = transform(data)

    assert result["a"] == 42
    assert result["b"] == 99
    assert result["other"] is sentinel.other


def test_scale() -> None:
    data = {"pos": torch.tensor([1.0, 2.0, 3.0]), "other": sentinel.other}
    transform = Scale(keys=["pos"], scale=2.0)
    result = transform(data)

    assert torch.equal(result["pos"], torch.tensor([2.0, 4.0, 6.0]))
    assert result["other"] is sentinel.other


def test_divide() -> None:
    data = {"pos": torch.tensor([2.0, 4.0, 6.0]), "other": sentinel.other}
    transform = Divide(keys=["pos"], divisor=2.0)
    result = transform(data)

    assert torch.equal(result["pos"], torch.tensor([1.0, 2.0, 3.0]))
    assert result["other"] is sentinel.other


def test_center_bbox() -> None:
    pos = torch.tensor([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]])
    data = {"pos": pos}
    transform = Center(keys=["pos"], method="bbox")
    result = transform(data)

    expected = pos - torch.tensor([1.0, 1.0, 1.0])
    assert torch.allclose(result["pos"], expected)


def test_center_mean() -> None:
    pos = torch.tensor([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]])
    data = {"pos": pos}
    transform = Center(keys=["pos"], method="mean")
    result = transform(data)

    expected = pos - pos.mean(dim=0)
    assert torch.allclose(result["pos"], expected)


def test_align_axis() -> None:
    pos = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    data = {"pos": pos}
    transform = AlignAxis(keys=["pos"], dim=-1)
    result = transform(data)

    assert result["pos"][:, -1].min() == 0.0


def test_ball_mask() -> None:
    pos = torch.tensor([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]])
    data = {"pos": pos}
    transform = BallMask(keys=["pos"], center=[0.0, 0.0, 0.0], radius=1.0, dst_keys=["mask"])
    result = transform(data)

    assert result["mask"][0].item() is True
    assert result["mask"][1].item() is False


def test_relabel() -> None:
    data = {"seg": torch.tensor([1, 2, 5, 255])}
    transform = Relabel(keys=["seg"], labels=[1, 2, 5], default=255)
    result = transform(data)

    assert result["seg"][0] == 0
    assert result["seg"][1] == 1
    assert result["seg"][2] == 2
    assert result["seg"][3] == 255


def test_rename_items() -> None:
    data = {"old": sentinel.value, "keep": sentinel.other}
    transform = RenameItems(keys=["old"], names=["new"])
    result = transform(data)

    assert "old" not in result
    assert result["new"] is sentinel.value
    assert result["keep"] is sentinel.other


def test_copy_items() -> None:
    data = {"src": torch.tensor([1.0, 2.0]), "keep": sentinel.other}
    transform = CopyItems(keys=["src"], names=["dst"])
    result = transform(data)

    assert torch.equal(result["dst"], result["src"])
    assert result["dst"] is not result["src"]
    assert result["keep"] is sentinel.other


def test_subtract_key() -> None:
    data = {"a": torch.tensor([5.0, 6.0]), "b": torch.tensor([1.0, 2.0])}
    transform = SubtractKey(keys=["a"], sub_keys=["b"])
    result = transform(data)

    assert torch.equal(result["a"], torch.tensor([4.0, 4.0]))


def test_divide_key() -> None:
    data = {"a": torch.tensor([6.0, 8.0]), "b": torch.tensor([2.0, 4.0])}
    transform = DivideKey(keys=["a"], div_keys=["b"])
    result = transform(data)

    assert torch.equal(result["a"], torch.tensor([3.0, 2.0]))


def test_to_tensor() -> None:
    data = {"x": [1.0, 2.0, 3.0]}
    transform = ToTensor(keys=["x"], dtype=torch.float32)
    result = transform(data)

    assert isinstance(result["x"], torch.Tensor)
    assert result["x"].dtype == torch.float32


def test_ones_like() -> None:
    data = {"pos": torch.randn(5, 3)}
    transform = OnesLike(keys=["pos"], dst_keys=["ones"])
    result = transform(data)

    assert torch.equal(result["ones"], torch.ones(5, 3))


def test_axis_min_offset() -> None:
    pos = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    data = {"pos": pos}
    transform = AxisMinOffset(keys=["pos"], axis=2, dst_keys=["h"])
    result = transform(data)

    assert result["h"].shape == (2, 1)
    assert result["h"][0].item() == 0.0
    assert result["h"][1].item() == 3.0


def test_cat() -> None:
    data = {
        "a": torch.ones(4, 2),
        "b": torch.zeros(4, 3),
    }
    transform = Cat(keys=["a", "b"], dst_key="x", dim=-1)
    result = transform(data)

    assert result["x"].shape == (4, 5)


def test_keep_items() -> None:
    data = {"pos": sentinel.pos, "color": sentinel.color, "drop": sentinel.drop}
    transform = KeepItems(keys=["pos", "color"])
    result = transform(data)

    assert set(result.keys()) == {"pos", "color"}
    assert result["pos"] is sentinel.pos
    assert result["color"] is sentinel.color
