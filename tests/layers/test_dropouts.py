import pytest
import torch
import torch.nn as nn

from torch_pointcloud.layers.dropouts import DropPath, create_dropout, drop_path


def test_drop_path_function_zero_prob_is_identity() -> None:
    x = torch.randn(8, 16)
    out = drop_path(x, drop_prob=0.0, training=True)
    torch.testing.assert_close(out, x)


def test_drop_path_function_eval_is_identity() -> None:
    x = torch.randn(8, 16)
    out = drop_path(x, drop_prob=0.5, training=False)
    torch.testing.assert_close(out, x)


def test_drop_path_module_train_eval() -> None:
    layer = DropPath(drop_prob=0.5, scale_by_keep=True)
    layer.eval()
    x = torch.randn(8, 16)
    torch.testing.assert_close(layer(x), x)

    layer.train()
    torch.manual_seed(0)
    out = layer(x)
    assert out.shape == x.shape


def test_drop_path_extra_repr() -> None:
    layer = DropPath(drop_prob=0.123)
    assert "drop_prob" in repr(layer)


@pytest.mark.parametrize(
    "name", ["dropout", "dropout2d", "dropout3d", "alpha_dropout", "feature_alpha_dropout", "drop_path"]
)
def test_create_dropout_by_name(name: str) -> None:
    layer = create_dropout(name)  # type: ignore[arg-type]
    assert isinstance(layer, nn.Module)


def test_create_dropout_from_float() -> None:
    layer = create_dropout(0.25)
    assert isinstance(layer, nn.Dropout)
    assert layer.p == 0.25
