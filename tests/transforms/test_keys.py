from unittest.mock import sentinel

import pytest
import torch

import torch_pointcloud.transforms as T


def test_set_value() -> None:
    data = {"a": 1, "other": sentinel.other}
    transform = T.SetValue(keys=["a", "b"], values=[42, 99])
    result = transform(data)

    assert result["a"] == 42
    assert result["b"] == 99
    assert result["other"] is sentinel.other
    assert data == {"a": 1, "other": sentinel.other}


def test_set_value_broadcast_constant() -> None:
    data = {"other": sentinel.other}
    transform = T.SetValue(keys=["condition", "extra"], values="ScanNet")
    result = transform(data)

    assert result["condition"] == "ScanNet"
    assert result["extra"] == "ScanNet"
    assert "condition" not in data


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


def test_cat() -> None:
    data = {
        "a": torch.ones(4, 2),
        "b": torch.zeros(4, 3),
    }
    transform = T.Cat(keys=["a", "b"], dst_key="x", dim=-1)
    result = transform(data)

    assert result["x"].shape == (4, 5)


def test_cat_preserves_float64() -> None:
    data = {
        "a": torch.ones(4, 2, dtype=torch.float64),
        "b": torch.zeros(4, 3, dtype=torch.float64),
    }
    result = T.Cat(keys=["a", "b"], dst_key="x", dim=-1)(data)
    assert result["x"].dtype == torch.float64


def test_cat_promotes_mixed_float_dtypes() -> None:
    data = {
        "a": torch.ones(4, 2, dtype=torch.float32),
        "b": torch.zeros(4, 3, dtype=torch.float64),
    }
    result = T.Cat(keys=["a", "b"], dst_key="x", dim=-1)(data)
    assert result["x"].dtype == torch.float64


def test_cat_casts_integer_inputs_to_float32() -> None:
    data = {
        "a": torch.ones(4, 2, dtype=torch.long),
        "b": torch.zeros(4, 3),
    }
    result = T.Cat(keys=["a", "b"], dst_key="x", dim=-1)(data)
    assert result["x"].dtype == torch.float32


def test_keep_items() -> None:
    data = {"pos": sentinel.pos, "color": sentinel.color, "drop": sentinel.drop}
    transform = T.KeepItems(keys=["pos", "color"])
    result = transform(data)

    assert set(result.keys()) == {"pos", "color"}
    assert result["pos"] is sentinel.pos
    assert result["color"] is sentinel.color


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


def test_reduce_mean_preserves_float64() -> None:
    data = {"pos": torch.tensor([[1.0, 5.0], [3.0, 7.0]], dtype=torch.float64)}
    result = T.Reduce(keys=["pos"], op="mean", dim=0, dst_keys=["center"])(data)
    assert result["center"].dtype == torch.float64
    assert torch.allclose(result["center"], torch.tensor([2.0, 6.0], dtype=torch.float64))


def test_reduce_sum_dst_key() -> None:
    data = {"x": torch.tensor([[1.0, 2.0], [3.0, 4.0]])}
    result = T.Reduce(keys=["x"], op="sum", dim=0, dst_keys=["total"])(data)
    assert torch.allclose(result["total"], torch.tensor([4.0, 6.0]))


def test_relabel_sparse_sources_does_not_oom() -> None:
    # The searchsorted-based impl handles sparse source values (e.g. 2**20) in O(|sources|) memory,
    # instead of a dense max-value lookup table.
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


def test_to_float() -> None:
    data = {"x": torch.ones(4, dtype=torch.int64), "other": sentinel.other}
    result = T.ToFloat(keys=["x"])(data)
    assert result["x"].dtype == torch.float32
    assert result["other"] is sentinel.other


def test_to_device_cpu() -> None:
    data = {"x": torch.zeros(4), "other": sentinel.other}
    result = T.ToDevice(keys=["x"], device="cpu")(data)
    assert result["x"].device.type == "cpu"
    assert result["other"] is sentinel.other


def test_to_device_non_tensor_raises() -> None:
    with pytest.raises(TypeError, match="tensor"):
        T.ToDevice(keys=["x"], device="cpu")({"x": "not a tensor"})
