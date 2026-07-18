from collections import OrderedDict
from unittest.mock import Mock, call, sentinel

import pytest
import torch
from torch import nn

from torch_pointcloud.utils.state_dict import load_state_dict, transform_state_dict


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


def test_load_state_dict_supports_lazy_parameters() -> None:
    """Uninitialized lazy parameters are skipped by the shape check and materialize on load."""
    source = nn.LazyLinear(4)
    source(torch.randn(2, 3))
    target = nn.LazyLinear(4)

    load_state_dict(target, dict(source.state_dict()), source="lazy.ckpt")

    assert torch.equal(target.weight, source.weight)
    assert target(torch.randn(2, 3)).shape == (2, 4)
