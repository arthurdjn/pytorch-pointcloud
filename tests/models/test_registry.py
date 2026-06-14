from typing import Any, Iterator

import pytest
from torch import Tensor, nn

from torch_pointcloud.models import ClassificationModel, create_model, register_model
from torch_pointcloud.models._registry import _REGISTERED_MODELS


class DummyClassificationModel(ClassificationModel):
    def __init__(self, in_channels: int = 3, num_classes: int = 5) -> None:
        super().__init__(in_channels=in_channels, num_classes=num_classes)
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x: Tensor, pos: Tensor, batch: Tensor) -> Tensor:
        return self.fc(x)


def _dummy_classification(**kwargs: Any) -> DummyClassificationModel:
    return DummyClassificationModel(**kwargs)


@pytest.fixture(autouse=True)
def _register_dummy() -> Iterator[None]:
    """Register a dummy model with no pretrained weights, then pop it on teardown to keep the
    global registry clean for `test_registered_models`."""
    register_model("dummy.classification", task="classification")(_dummy_classification)
    yield
    _REGISTERED_MODELS["classification"].pop("dummy.classification", None)


def test_create_model_returns_model() -> None:
    model = create_model("dummy.classification", task="classification")
    assert isinstance(model, DummyClassificationModel)


def test_create_model_return_info_returns_tuple() -> None:
    model, info = create_model("dummy.classification", task="classification", return_info=True)
    assert isinstance(model, DummyClassificationModel)
    assert info["name"] == "dummy.classification"


def test_create_model_pretrained_without_weights_warns_and_returns_model() -> None:
    with pytest.warns(UserWarning, match="No pretrained weights"):
        model = create_model("dummy.classification", task="classification", pretrained=True)
    assert isinstance(model, DummyClassificationModel)


def test_create_model_pretrained_without_weights_respects_return_info() -> None:
    with pytest.warns(UserWarning, match="No pretrained weights"):
        result = create_model("dummy.classification", task="classification", pretrained=True, return_info=True)
    assert isinstance(result, tuple)
    model, info = result
    assert isinstance(model, DummyClassificationModel)
    assert info["name"] == "dummy.classification"


def test_create_model_unknown_name_raises() -> None:
    with pytest.raises(ValueError, match="not found"):
        create_model("does-not-exist", task="classification")


def test_create_model_invalid_task_raises() -> None:
    with pytest.raises(ValueError, match="Invalid model task"):
        create_model("dummy.classification", task="invalid")
