import copy
from typing import Any, Callable
from unittest.mock import MagicMock, sentinel

import pytest
import torch

import torch_pointcloud.transforms as T


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
        T.FarthestPointSample(pos_key="absent", num_samples=2, allow_missing_keys=True),
        T.RemoveNearOrigin(pos_key="absent", allow_missing_keys=True),
        T.SphereCrop(pos_key="absent", radius=1.0, allow_missing_keys=True),
        T.Voxelize(pos_key="absent", pos_reduce="mean", size=0.1, allow_missing_keys=True),
        T.Cat(keys=["absent"], dst_key="feat", allow_missing_keys=True),
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
        T.FarthestPointSample(pos_key="absent", num_samples=2),
        T.RemoveNearOrigin(pos_key="absent"),
        T.SphereCrop(pos_key="absent", radius=1.0),
        T.Voxelize(pos_key="absent", pos_reduce="mean", size=0.1),
        T.Cat(keys=["absent"], dst_key="feat"),
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


@pytest.mark.parametrize("p", [-0.1, 1.5])
@pytest.mark.parametrize(
    "make_transform",
    [
        pytest.param(lambda p: T.RandomRotate(keys="pos", p=p), id="RandomRotate"),
        pytest.param(lambda p: T.RandomScale(keys="pos", p=p), id="RandomScale"),
        pytest.param(lambda p: T.RandomFlip(keys="pos", p=p), id="RandomFlip"),
        pytest.param(lambda p: T.RandomJitter(keys="pos", p=p), id="RandomJitter"),
        pytest.param(lambda p: T.RandomShift(keys="pos", p=p), id="RandomShift"),
        pytest.param(lambda p: T.RandomDropout(keys="pos", p=p), id="RandomDropout"),
        pytest.param(lambda p: T.RandomColorJitter(keys="color", p=p), id="RandomColorJitter"),
        pytest.param(lambda p: T.RandomColorDrop(keys="color", p=p), id="RandomColorDrop"),
        pytest.param(lambda p: T.RandomColorGrayScale(keys="color", p=p), id="RandomColorGrayScale"),
        pytest.param(lambda p: T.RandomColorAutoContrast(keys="color", p=p), id="RandomColorAutoContrast"),
        pytest.param(lambda p: T.SphereCrop(pos_key="pos", radius=1.0, p=p), id="SphereCrop"),
        pytest.param(lambda p: T.ShufflePoint(keys="pos", p=p), id="ShufflePoint"),
        pytest.param(lambda p: T.RandomRotateChoice(keys="pos", angles=[90.0], p=p), id="RandomRotateChoice"),
        pytest.param(lambda p: T.RandomColorShift(keys="color", p=p), id="RandomColorShift"),
        pytest.param(lambda p: T.RandomElasticDistortion(keys="pos", p=p), id="RandomElasticDistortion"),
        pytest.param(lambda p: T.Mix3D(keys="pos", p=p), id="Mix3D"),
        pytest.param(lambda p: T.LaserMix(keys="pos", num_areas=(3,), pitch_range=(-25.0, 3.0), p=p), id="LaserMix"),
        pytest.param(lambda p: T.PolarMix(keys="pos", instance_classes=(1,), p=p), id="PolarMix"),
    ],
)
def test_random_transform_out_of_range_p_raises(make_transform: Callable[[float], T.Transform], p: float) -> None:
    with pytest.raises(ValueError, match=r"p must be in \[0, 1\]"):
        make_transform(p)


def test_compose_allow_missing_keys_propagates_to_children() -> None:
    child = T.Abs(keys=["absent"])
    compose = T.Compose([child], allow_missing_keys=True)
    assert child.allow_missing_keys is True
    assert compose({"pos": torch.randn(5, 3)})["pos"].shape == (5, 3)


def test_compose_allow_missing_keys_propagates_through_nested_compose() -> None:
    child = T.Abs(keys=["absent"])
    inner = T.Compose([child])
    outer = T.Compose([inner], allow_missing_keys=True)
    assert inner.allow_missing_keys is True
    assert child.allow_missing_keys is True
    outer.allow_missing_keys = False
    assert inner.allow_missing_keys is False
    assert child.allow_missing_keys is False


def test_compose_allow_missing_keys_none_leaves_children() -> None:
    lenient = T.Abs(keys=["absent"], allow_missing_keys=True)
    strict = T.Abs(keys=["absent"])
    compose = T.Compose([lenient, strict])
    assert compose.allow_missing_keys is None
    assert lenient.allow_missing_keys is True
    assert strict.allow_missing_keys is False


def test_compose_allow_missing_keys_skips_non_dict_transforms() -> None:
    class Plain(T.Transform):
        def transform(self, data: Any) -> Any:
            return data

    plain = Plain()
    T.Compose([plain], allow_missing_keys=True)
    assert not hasattr(plain, "allow_missing_keys")
