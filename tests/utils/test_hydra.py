import pytest

from torch_pointcloud.utils.hydra import instantiate_list


def test_none_returns_empty_list() -> None:
    assert instantiate_list(None) == []


def test_empty_dict_and_list_return_empty() -> None:
    assert instantiate_list({}) == []
    assert instantiate_list([]) == []


def test_dict_of_targets_instantiates_each_value() -> None:
    cfg = {
        "linear": {"_target_": "torch.nn.Linear", "in_features": 3, "out_features": 4},
        "relu": {"_target_": "torch.nn.ReLU"},
    }
    instances = instantiate_list(cfg)
    assert len(instances) == 2
    import torch

    assert isinstance(instances[0], torch.nn.Linear)
    assert isinstance(instances[1], torch.nn.ReLU)


def test_list_of_targets_instantiates_each_element() -> None:
    cfg = [
        {"_target_": "torch.nn.Linear", "in_features": 2, "out_features": 5},
        {"_target_": "torch.nn.Sigmoid"},
    ]
    instances = instantiate_list(cfg)
    assert len(instances) == 2


def test_skips_none_entries() -> None:
    cfg = {
        "linear": {"_target_": "torch.nn.Linear", "in_features": 2, "out_features": 2},
        "skipped": None,
    }
    assert len(instantiate_list(cfg)) == 1


def test_invalid_input_type_raises() -> None:
    with pytest.raises(TypeError, match="mapping, sequence, or None"):
        instantiate_list(42)
