import functools
import pickle
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, Tuple, Type
from unittest.mock import Mock, call, sentinel

import pytest
import torch
from torch import nn

from torch_pointcloud.utils.state_dict import load_state_dict, read_state_dict, transform_state_dict


def test_state_dict_empty() -> None:
    """Empty state dict returns empty ordered dict."""
    result = transform_state_dict({}, {"a": "b"})

    assert result == OrderedDict()
    assert isinstance(result, OrderedDict)


def test_state_dict_empty_mapping_preserves_keys() -> None:
    """With no mapping rules, keys are unchanged."""
    state_dict = {
        "layer.weight": sentinel.weight,
        "layer.bias": sentinel.bias,
    }
    result = transform_state_dict(state_dict, {})

    assert list(result.keys()) == ["layer.weight", "layer.bias"]
    assert result["layer.weight"] is sentinel.weight
    assert result["layer.bias"] is sentinel.bias


def test_state_dict_simple_key_remapping() -> None:
    """Literal key mapping (no placeholders) works when key matches exactly."""
    state_dict = {"old_name": sentinel.value}
    mapping = {"old_name": "new_name"}
    result = transform_state_dict(state_dict, mapping)

    assert list(result.keys()) == ["new_name"]
    assert result["new_name"] is sentinel.value


def test_state_dict_pattern_remapping_with_index_arithmetic() -> None:
    """Pattern with {module}, {i} and arithmetic in destination (e.g. i+1)."""
    state_dict = {
        "encoder.conv.0.weight": sentinel.weight0,
        "encoder.conv.0.bias": sentinel.bias0,
        "encoder.norm.1.weight": sentinel.weight1,
        "encoder.norm.1.bias": sentinel.bias1,
        "encoder.norm.1.running_mean": sentinel.running_mean1,
        "encoder.norm.1.running_var": sentinel.running_var1,
    }
    mapping = {
        "encoder.{module}.{i}.weight": "backbone.{module}.{i+1}.weight",
        "encoder.{module}.{i}.bias": "backbone.{module}.{i+1}.bias",
        "encoder.{module}.{i}.running_{stat}": "backbone.{module}.{i+1}.running_{stat}",
    }
    result = transform_state_dict(state_dict, mapping)

    assert set(result.keys()) == {
        "backbone.conv.1.weight",
        "backbone.conv.1.bias",
        "backbone.norm.2.weight",
        "backbone.norm.2.bias",
        "backbone.norm.2.running_mean",
        "backbone.norm.2.running_var",
    }
    assert result["backbone.conv.1.weight"] is sentinel.weight0
    assert result["backbone.conv.1.bias"] is sentinel.bias0
    assert result["backbone.norm.2.weight"] is sentinel.weight1
    assert result["backbone.norm.2.bias"] is sentinel.bias1
    assert result["backbone.norm.2.running_mean"] is sentinel.running_mean1
    assert result["backbone.norm.2.running_var"] is sentinel.running_var1


def test_state_dict_unmatched_keys_preserved() -> None:
    """Keys that match no rule are left unchanged."""
    state_dict = {
        "mapped.key": sentinel.value,
        "unmapped.other.key": sentinel.other_value,
    }
    mapping = {"mapped.key": "new.key"}
    result = transform_state_dict(state_dict, mapping)

    assert result["new.key"] is sentinel.value
    assert result["unmapped.other.key"] is sentinel.other_value


def test_state_dict_value_transform_applied() -> None:
    """value_transform is applied to each value."""
    state_dict = {"a": sentinel.a, "b": sentinel.b}
    mapping = {"a": "new_a", "b": "new_b"}
    value_transform = Mock(side_effect=[sentinel.new_a, sentinel.new_b])
    result = transform_state_dict(state_dict, mapping, value_transform=value_transform)

    assert list(result.keys()) == ["new_a", "new_b"]
    assert result["new_a"] is sentinel.new_a
    assert result["new_b"] is sentinel.new_b
    assert value_transform.call_args_list == [call(sentinel.a), call(sentinel.b)]


def test_state_dict_order_preserved() -> None:
    """Output order follows input state_dict order."""
    state_dict = OrderedDict(
        [
            ("first", sentinel.first),
            ("second", sentinel.second),
            ("third", sentinel.third),
        ]
    )
    mapping = {"third": "x", "second": "y", "first": "z"}
    result = transform_state_dict(state_dict, mapping)

    assert list(result.keys()) == ["z", "y", "x"]
    assert result["z"] is sentinel.first
    assert result["y"] is sentinel.second
    assert result["x"] is sentinel.third


def test_state_dict_int_placeholder_explicit() -> None:
    """Placeholder {i:int} matches only digits."""
    state_dict = {"block.0.weight": sentinel.weight}
    mapping = {"block.{i:int}.weight": "backbone.{i+1}.weight"}
    result = transform_state_dict(state_dict, mapping)

    assert list(result.keys()) == ["backbone.1.weight"]
    assert result["backbone.1.weight"] is sentinel.weight


def test_state_dict_multiple_rules_first_match_used() -> None:
    """When several rules could match, the first in mapping order is used."""
    state_dict = {"encoder.conv.0.weight": sentinel.weight0}
    mapping = {
        "encoder.{module}.0.weight": "backbone.{module}.1.weight",
        "encoder.{module}.{i}.weight": "other.{module}.{i}.weight",
    }
    result = transform_state_dict(state_dict, mapping)

    assert list(result.keys()) == ["backbone.conv.1.weight"]


def test_state_dict_placeholder_spanning_multiple_segments_ignored() -> None:
    """Placeholders that span on multiple segments are ignored."""
    state_dict = {"layer.foo.bar": sentinel.value}
    mapping = {"layer.{name}": "new.{name}"}
    result = transform_state_dict(state_dict, mapping)

    assert list(result.keys()) == ["layer.foo.bar"]


def test_state_dict_collision_raises_when_strict() -> None:
    """Two source keys mapping to the same destination raise instead of silently overwriting."""
    state_dict = {"a.weight": sentinel.a, "b.weight": sentinel.b}
    mapping = {"a.weight": "c.weight", "b.weight": "c.weight"}

    with pytest.raises(ValueError, match="Colliding keys"):
        transform_state_dict(state_dict, mapping, strict=True)


def test_state_dict_collision_warns_when_not_strict() -> None:
    """Without strict, a collision warns and the last source value wins."""
    state_dict = {"a.weight": sentinel.a, "b.weight": sentinel.b}
    mapping = {"a.weight": "c.weight", "b.weight": "c.weight"}

    with pytest.warns(UserWarning, match="Colliding keys"):
        result = transform_state_dict(state_dict, mapping)

    assert list(result.keys()) == ["c.weight"]
    assert result["c.weight"] is sentinel.b


def test_state_dict_unused_mapping_raises_when_strict() -> None:
    """A mapping pattern that matches no key raises under strict instead of silently doing nothing."""
    state_dict = {"a.weight": sentinel.a}
    mapping = {"missing.{i}.weight": "new.{i}.weight"}

    with pytest.raises(ValueError, match="matched no key"):
        transform_state_dict(state_dict, mapping, strict=True)


def test_load_state_dict_supports_lazy_parameters() -> None:
    """Uninitialized lazy parameters are skipped by the shape check and materialize on load."""
    source = nn.LazyLinear(4)
    source(torch.randn(2, 3))
    target = nn.LazyLinear(4)

    load_state_dict(target, dict(source.state_dict()), source="lazy.ckpt")

    assert torch.equal(target.weight, source.weight)
    assert target(torch.randn(2, 3)).shape == (2, 4)


@pytest.mark.filterwarnings("error::UserWarning")
def test_load_state_dict_mismatched_shapes_keep_initialization() -> None:
    model = nn.Linear(4, 2)
    weight = model.weight.clone()
    load_state_dict(model, {"weight": torch.zeros(2, 3), "bias": torch.ones(2)}, source="test")
    assert torch.equal(model.weight, weight)
    assert torch.equal(model.bias, torch.ones(2))


def test_load_state_dict_missing_keys_raise_by_default() -> None:
    model = nn.Linear(3, 4)

    with pytest.raises(RuntimeError, match="missing model keys"):
        load_state_dict(model, {}, source="empty.ckpt")


def test_load_state_dict_strict_false_warm_starts_from_partial_checkpoint() -> None:
    """A backbone-only checkpoint loads its keys and keeps the head's fresh initialization."""
    backbone = nn.Linear(3, 4)
    model = nn.Sequential(OrderedDict([("backbone", nn.Linear(3, 4)), ("head", nn.Linear(4, 2))]))
    head_weight = model.state_dict()["head.weight"].clone()
    checkpoint = {f"backbone.{key}": value for key, value in backbone.state_dict().items()}

    with pytest.warns(UserWarning, match="missing model keys"):
        load_state_dict(model, checkpoint, source="backbone.ckpt", strict=False)

    assert torch.equal(model.state_dict()["backbone.weight"], backbone.weight)
    assert torch.equal(model.state_dict()["head.weight"], head_weight)


class _FailsIfRestored:
    def __init__(self) -> None:
        self.saved = True

    def __setstate__(self, state: Dict[str, bool]) -> None:
        pytest.fail("unpickling restored an object of the checkpoint")


class _RestoredByCallingAnObject:
    def __reduce__(self) -> Tuple[Callable[[], Dict[Any, Any]], Tuple[()]]:
        return functools.partial(dict), ()


@pytest.mark.parametrize(
    ("checkpoint", "zip_format", "error", "match"),
    [
        pytest.param(torch.zeros(3), True, ValueError, "not a state dict", id="tensor"),
        pytest.param(nn.Linear(2, 2), True, ValueError, "got torch.nn.modules.linear.Linear", id="pickled-module"),
        pytest.param(
            {"state_dict": nn.Linear(2, 2)},
            True,
            ValueError,
            "got torch.nn.modules.linear.Linear",
            id="pickled-module-under-state-dict",
        ),
        pytest.param(
            {"model": nn.Linear(2, 2), "epoch": 3},
            True,
            ValueError,
            r"'model' \(torch.nn.modules.linear.Linear\)",
            id="pickled-module-among-parameters",
        ),
        pytest.param(
            {"lin.weight": torch.ones(2, 2), "optimizer": functools.partial(torch.optim.Adam, lr=0.01)},
            False,
            pickle.UnpicklingError,
            "Weights only load failed",
            id="legacy-format-with-objects",
        ),
        pytest.param(
            {"lin.weight": torch.ones(2, 2), "hook": _RestoredByCallingAnObject()},
            True,
            pickle.UnpicklingError,
            "Weights only load failed",
            id="restored-by-calling-an-object",
        ),
    ],
)
def test_read_state_dict_unreadable_checkpoint_raises(
    tmp_path: Path,
    checkpoint: Any,
    zip_format: bool,
    error: Type[Exception],
    match: str,
) -> None:
    path = tmp_path / "model.pt"
    torch.save(checkpoint, path, _use_new_zipfile_serialization=zip_format)

    with pytest.raises(error, match=match):
        read_state_dict(path)


def test_read_state_dict_unwraps_model_state_dict(tmp_path: Path) -> None:
    path = tmp_path / "training.pth"
    torch.save({"epoch": 3, "model_state_dict": {"lin.weight": torch.zeros(2, 2)}, "optimizer_state_dict": {}}, path)

    result = read_state_dict(path)

    assert list(result.keys()) == ["lin.weight"]


def test_read_state_dict_lightning_checkpoint_without_model_prefix_raises(tmp_path: Path) -> None:
    path = tmp_path / "lit.ckpt"
    torch.save({"pytorch-lightning_version": "2.0", "state_dict": {"net.lin.weight": torch.zeros(2, 2)}}, path)

    with pytest.raises(ValueError, match="no 'model.'-prefixed keys"):
        read_state_dict(path)


def test_read_state_dict_lightning_checkpoint_strips_model_prefix(tmp_path: Path) -> None:
    path = tmp_path / "lit.ckpt"
    torch.save({"pytorch-lightning_version": "2.0", "state_dict": {"model.lin.weight": torch.zeros(2, 2)}}, path)

    result = read_state_dict(path)

    assert list(result.keys()) == ["lin.weight"]


def test_read_state_dict_reads_other_objects_as_inert_placeholders(tmp_path: Path) -> None:
    path = tmp_path / "lit.ckpt"
    checkpoint = {
        "pytorch-lightning_version": "2.0",
        "state_dict": {"model.lin.weight": torch.ones(2, 2)},
        "hyper_parameters": {"optimizer": functools.partial(torch.optim.Adam, lr=0.01), "hook": _FailsIfRestored()},
    }
    torch.save(checkpoint, path)

    with pytest.raises(pytest.fail.Exception, match="restored"):
        torch.load(path, weights_only=False)

    result = read_state_dict(path)
    assert list(result.keys()) == ["lin.weight"]
    assert torch.equal(result["lin.weight"], torch.ones(2, 2))
