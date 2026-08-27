from pathlib import Path
from typing import Any, Dict, Iterator

import pytest
import torch
from safetensors.torch import save_file
from torch import Tensor, nn

from torch_pointcloud.models import (
    ClassificationModel,
    SegmentationModel,
    WeightsDict,
    create_model,
    list_models,
    register_model,
)
from torch_pointcloud.models._registry import _REGISTERED_MODELS


class DummyClassificationModel(ClassificationModel):
    def __init__(self, in_channels: int = 3, num_classes: int = 5) -> None:
        super().__init__(in_channels=in_channels, num_classes=num_classes)
        self.encoder = nn.Linear(in_channels, self.num_features)
        self.fc = self.configure_head()

    @property
    def num_features(self) -> int:
        return 8

    def configure_head(self) -> nn.Module:
        return nn.Identity() if self.num_classes == 0 else nn.Linear(self.num_features, self.num_classes)

    def reset_classifier(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.fc = self.configure_head()

    def forward_features(self, x: Tensor, pos: Tensor, batch: Tensor) -> Tensor:
        return self.encoder(x)

    def forward_head(self, x: Tensor, pre_logits: bool = False) -> Tensor:
        return x if pre_logits else self.fc(x)

    def forward(self, x: Tensor, pos: Tensor, batch: Tensor) -> Tensor:
        return self.forward_head(self.forward_features(x, pos, batch))


class DummySegmentationModel(SegmentationModel):
    def __init__(self, in_channels: int = 3, num_classes: int = 5) -> None:
        super().__init__(in_channels=in_channels, num_classes=num_classes)
        self.fc = self.configure_head()

    @property
    def num_features(self) -> int:
        return self.in_channels

    def configure_head(self) -> nn.Module:
        return nn.Identity() if self.num_classes == 0 else nn.Linear(self.num_features, self.num_classes)

    def reset_classifier(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.fc = self.configure_head()

    def forward_features(self, x: Tensor, pos: Tensor, batch: Tensor) -> Tensor:
        return x

    def forward_head(self, x: Tensor, pre_logits: bool = False) -> Tensor:
        return x if pre_logits else self.fc(x)

    def forward(self, x: Tensor, pos: Tensor, batch: Tensor) -> Tensor:
        return self.forward_head(self.forward_features(x, pos, batch))


def _dummy_classification(**kwargs: Any) -> DummyClassificationModel:
    return DummyClassificationModel(**kwargs)


def _dummy_segmentation(**kwargs: Any) -> DummySegmentationModel:
    return DummySegmentationModel(**kwargs)


@pytest.fixture(autouse=True)
def _register_dummy() -> Iterator[None]:
    """Register a dummy model with no pretrained weights, then pop it on teardown to keep the
    global registry clean for `test_registered_models`."""
    register_model("dummy.classification", task="classification")(_dummy_classification)
    yield
    _REGISTERED_MODELS["classification"].pop("dummy.classification", None)


@pytest.fixture
def _register_pretrained_dummy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Register a dummy with weights resolving under a temporary `MODELS_DIR`; yields the weight path."""
    monkeypatch.setattr("torch_pointcloud.models._registry.MODELS_DIR", tmp_path)
    register_model(
        "dummy-pretrained.classification",
        task="classification",
        weights="hf://torch-pointcloud/dummy/dummy-pretrained.pt",
    )(_dummy_classification)
    weights_path = tmp_path / "dummy" / "dummy-pretrained.pt"
    weights_path.parent.mkdir(parents=True)
    torch.save(DummyClassificationModel().state_dict(), weights_path)
    yield weights_path
    _REGISTERED_MODELS["classification"].pop("dummy-pretrained.classification", None)


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


def test_create_model_unknown_name_suggests_close_matches() -> None:
    with pytest.raises(ValueError, match="Did you mean 'dummy.classification'"):
        create_model("dummy.classifiction", task="classification")


def test_create_model_unknown_name_hints_other_task() -> None:
    with pytest.raises(ValueError, match="registered under task 'classification'; pass task='classification'"):
        create_model("dummy.classification", task="segmentation")


def test_create_model_segmentation_name_as_classification_hints_task() -> None:
    register_model("dummy-seg-only.segmentation", task="segmentation")(_dummy_segmentation)
    try:
        with pytest.raises(ValueError, match="registered under task 'segmentation'; pass task='segmentation'"):
            create_model("dummy-seg-only.segmentation", task="classification")
    finally:
        _REGISTERED_MODELS["segmentation"].pop("dummy-seg-only.segmentation", None)


def test_register_model_returns_the_registered_callable() -> None:
    decorated = register_model("dummy-identity.classification", task="classification")(_dummy_classification)
    try:
        assert decorated is _dummy_classification
    finally:
        _REGISTERED_MODELS["classification"].pop("dummy-identity.classification", None)


def test_create_model_invalid_task_raises() -> None:
    with pytest.raises(ValueError, match="Invalid model task"):
        create_model("dummy.classification", task="invalid")  # type: ignore[call-overload]


def test_register_model_normalizes_weights_url_string(_register_pretrained_dummy: Path) -> None:
    entry = _REGISTERED_MODELS["classification"]["dummy-pretrained.classification"]
    assert entry["weights"] == {"url": "hf://torch-pointcloud/dummy/dummy-pretrained.pt"}


def test_register_model_keeps_weights_dict() -> None:
    weights = WeightsDict(
        url="hf://torch-pointcloud/dummy/dummy-meta.pt",
        dataset="modelnet40",
        metrics={"OA": 92.46},
        classes=("airplane", "bathtub"),
    )
    register_model("dummy-meta.classification", task="classification", weights=weights)(_dummy_classification)
    try:
        assert _REGISTERED_MODELS["classification"]["dummy-meta.classification"]["weights"] == weights
    finally:
        _REGISTERED_MODELS["classification"].pop("dummy-meta.classification", None)


def test_list_models_pretrained_filter(_register_pretrained_dummy: Path) -> None:
    assert list_models("dummy*", task="classification") == ["dummy-pretrained.classification", "dummy.classification"]
    assert list_models("dummy*", task="classification", pretrained=True) == ["dummy-pretrained.classification"]


def test_list_models_across_tasks() -> None:
    assert "dummy.classification" in list_models("dummy*")


def test_list_models_invalid_task_raises() -> None:
    with pytest.raises(ValueError, match="Invalid model task"):
        list_models(task="invalid")  # type: ignore[arg-type]


def test_create_model_pretrained_loads_weights(_register_pretrained_dummy: Path) -> None:
    model = create_model("dummy-pretrained.classification", task="classification", pretrained=True)
    assert isinstance(model, DummyClassificationModel)
    state_dict = torch.load(_register_pretrained_dummy, weights_only=True)
    assert torch.equal(model.encoder.weight, state_dict["encoder.weight"])
    assert isinstance(model.fc, nn.Linear)
    assert torch.equal(model.fc.weight, state_dict["fc.weight"])


def test_create_model_pretrained_num_classes_override_adapts_head(_register_pretrained_dummy: Path) -> None:
    with pytest.warns(UserWarning, match="mismatched shapes"):
        model = create_model("dummy-pretrained.classification", task="classification", pretrained=True, num_classes=3)
    assert isinstance(model, DummyClassificationModel)
    state_dict = torch.load(_register_pretrained_dummy, weights_only=True)
    assert model.fc.weight.shape == (3, 8)
    assert torch.equal(model.encoder.weight, state_dict["encoder.weight"])


def test_create_model_pretrained_ignores_unexpected_keys(_register_pretrained_dummy: Path) -> None:
    state_dict = torch.load(_register_pretrained_dummy, weights_only=True)
    state_dict["extra.weight"] = torch.ones(2)
    torch.save(state_dict, _register_pretrained_dummy)
    with pytest.warns(UserWarning, match="absent from the model"):
        model = create_model("dummy-pretrained.classification", task="classification", pretrained=True)
    assert isinstance(model, DummyClassificationModel)
    assert torch.equal(model.encoder.weight, state_dict["encoder.weight"])


def test_create_model_pretrained_missing_keys_raise(_register_pretrained_dummy: Path) -> None:
    state_dict = torch.load(_register_pretrained_dummy, weights_only=True)
    del state_dict["encoder.weight"]
    torch.save(state_dict, _register_pretrained_dummy)
    with pytest.raises(RuntimeError, match="missing model keys"):
        create_model("dummy-pretrained.classification", task="classification", pretrained=True)


def test_create_model_checkpoint_path(tmp_path: Path) -> None:
    reference = DummyClassificationModel()
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(reference.state_dict(), checkpoint_path)
    model = create_model("dummy.classification", task="classification", checkpoint_path=checkpoint_path)
    assert isinstance(model, DummyClassificationModel)
    assert torch.equal(model.encoder.weight, reference.encoder.weight)


def test_create_model_checkpoint_path_safetensors(tmp_path: Path) -> None:
    reference = DummyClassificationModel()
    checkpoint_path = tmp_path / "checkpoint.safetensors"
    save_file(reference.state_dict(), checkpoint_path)
    model = create_model("dummy.classification", task="classification", checkpoint_path=checkpoint_path)
    assert isinstance(model, DummyClassificationModel)
    assert torch.equal(model.encoder.weight, reference.encoder.weight)


def test_create_model_checkpoint_path_lightning(tmp_path: Path) -> None:
    reference = DummyClassificationModel()
    checkpoint: Dict[str, Any] = {
        "pytorch-lightning_version": "2.0.0",
        "state_dict": {f"model.{key}": value for key, value in reference.state_dict().items()},
    }
    checkpoint["state_dict"]["criterion.weight"] = torch.ones(2)
    checkpoint_path = tmp_path / "checkpoint.ckpt"
    torch.save(checkpoint, checkpoint_path)
    model = create_model("dummy.classification", task="classification", checkpoint_path=checkpoint_path)
    assert isinstance(model, DummyClassificationModel)
    assert torch.equal(model.encoder.weight, reference.encoder.weight)


def test_create_model_checkpoint_path_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        create_model("dummy.classification", task="classification", checkpoint_path=tmp_path / "missing.pt")


def test_create_model_pretrained_and_checkpoint_path_raise(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        create_model(
            "dummy.classification", task="classification", pretrained=True, checkpoint_path=tmp_path / "any.pt"
        )


def test_create_model_arch_only_entry_raises_actionable_type_error() -> None:
    """Arch-only entries register no `in_channels` / `num_classes`; the bare call must say what to pass."""
    with pytest.raises(TypeError, match="in_channels"):
        create_model("pointnext-sm", task="classification")
    model = create_model("pointnext-sm", task="classification", in_channels=4, num_classes=15)
    assert model.num_classes == 15


def test_create_model_pretrained_rejects_weight_path_escaping_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("torch_pointcloud.models._registry.MODELS_DIR", tmp_path)
    register_model(
        "dummy-escape.classification",
        task="classification",
        weights="hf://torch-pointcloud/../outside/dummy-escape.pt",
    )(_dummy_classification)
    try:
        with pytest.raises(ValueError, match="outside the models cache"):
            create_model("dummy-escape.classification", task="classification", pretrained=True)
    finally:
        _REGISTERED_MODELS["classification"].pop("dummy-escape.classification", None)
