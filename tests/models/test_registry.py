import subprocess
import sys
from email.message import Message
from pathlib import Path
from typing import Any, Dict, Iterator
from unittest.mock import Mock
from urllib.error import HTTPError

import pytest
import torch
from safetensors.torch import save_file
from torch import Tensor, nn

from torch_pointcloud.models import (
    ClassificationModel,
    ModelInfoDict,
    SemanticSegmentationModel,
    WeightsDict,
    create_model,
    deprecated_model,
    list_models,
    model_info,
    register_model,
)
from torch_pointcloud.models._registry import (
    _DEPRECATED_MODELS,
    _REGISTERED_MODELS,
    cache_path,
)
from torch_pointcloud.utils.imports import _SPCONV_AVAILABLE


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
        features = self.forward_features(x, pos, batch)
        return self.forward_head(features)


class DummySemanticSegmentationModel(SemanticSegmentationModel):
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
        features = self.forward_features(x, pos, batch)
        return self.forward_head(features)


def _dummy_classification(**kwargs: Any) -> DummyClassificationModel:
    return DummyClassificationModel(**kwargs)


def _dummy_segmentation(**kwargs: Any) -> DummySemanticSegmentationModel:
    return DummySemanticSegmentationModel(**kwargs)


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
        weights="hf://torch-pointcloud/dummy-pretrained/resolve/main/model.pt",
    )(_dummy_classification)
    weights_path = tmp_path / "dummy-pretrained" / "main" / "model.pt"
    weights_path.parent.mkdir(parents=True)
    torch.save(DummyClassificationModel().state_dict(), weights_path)
    yield weights_path
    _REGISTERED_MODELS["classification"].pop("dummy-pretrained.classification", None)


@pytest.fixture
def _register_renamed_dummy() -> Iterator[None]:
    """Register `dummy-renamed.classification` with the former name `dummy-former.classification`, then drop both."""

    @deprecated_model("dummy-former.classification", deprecated_in="0.0.17", removed_in="0.1.0")
    @register_model("dummy-renamed.classification", task="classification")
    def dummy_renamed(**kwargs: Any) -> DummyClassificationModel:
        return DummyClassificationModel(**kwargs)

    yield
    _REGISTERED_MODELS["classification"].pop("dummy-renamed.classification", None)
    _DEPRECATED_MODELS["classification"].pop("dummy-former.classification", None)


@pytest.fixture
def _register_renamed_in_one_task_dummy() -> Iterator[None]:
    """Register `dummy-shared.demo` for semantic segmentation and, renamed from that name for classification only,
    `dummy-shared-v2.demo`, then drop them."""
    register_model("dummy-shared.demo", task="semantic-segmentation")(_dummy_segmentation)

    @deprecated_model("dummy-shared.demo", deprecated_in="0.0.17")
    @register_model("dummy-shared-v2.demo", task="classification")
    def dummy_shared_v2(**kwargs: Any) -> DummyClassificationModel:
        return DummyClassificationModel(**kwargs)

    yield
    _REGISTERED_MODELS["semantic-segmentation"].pop("dummy-shared.demo", None)
    _REGISTERED_MODELS["classification"].pop("dummy-shared-v2.demo", None)
    _DEPRECATED_MODELS["classification"].pop("dummy-shared.demo", None)


def test_create_model_returns_model() -> None:
    model = create_model("dummy.classification", task="classification")
    assert isinstance(model, DummyClassificationModel)


def test_create_model_return_info_returns_tuple() -> None:
    model, info = create_model("dummy.classification", task="classification", return_info=True)
    assert isinstance(model, DummyClassificationModel)
    assert info["name"] == "dummy.classification"


def test_create_model_infers_the_task_of_a_unique_name() -> None:
    model, info = create_model("dummy.classification", return_info=True)
    assert isinstance(model, DummyClassificationModel)
    assert info["task"] == "classification"


def test_create_model_return_info_is_a_model_info_dict() -> None:
    _, info = create_model("dummy.classification", return_info=True)
    assert set(info) == set(ModelInfoDict.__annotations__)
    assert info["input_keys"] == ("x", "pos", "batch")


def test_create_model_without_task_raises_for_a_name_under_several_tasks() -> None:
    register_model("dummy.classification", task="semantic-segmentation")(_dummy_segmentation)
    try:
        with pytest.raises(ValueError, match="registered under tasks 'classification' and 'semantic-segmentation'"):
            create_model("dummy.classification")
    finally:
        _REGISTERED_MODELS["semantic-segmentation"].pop("dummy.classification", None)


def test_create_model_without_task_unknown_name_suggests_close_matches() -> None:
    with pytest.raises(ValueError, match="is not registered. Did you mean 'dummy.classification'"):
        create_model("dummy.classifiction")


def test_register_model_stores_input_keys() -> None:
    register_model(
        "dummy-keys.classification",
        task="classification",
        input_keys=["x", "pos_grid", "batch"],
    )(_dummy_classification)
    try:
        entry = _REGISTERED_MODELS["classification"]["dummy-keys.classification"]
        assert entry["input_keys"] == ("x", "pos_grid", "batch")
    finally:
        _REGISTERED_MODELS["classification"].pop("dummy-keys.classification", None)


def test_create_model_pretrained_without_weights_raises() -> None:
    with pytest.raises(ValueError, match="No pretrained weights are registered"):
        create_model("dummy.classification", task="classification", pretrained=True)


def test_create_model_pretrained_without_weights_raises_with_return_info() -> None:
    with pytest.raises(ValueError, match="No pretrained weights are registered"):
        create_model("dummy.classification", task="classification", pretrained=True, return_info=True)


def test_create_model_unknown_name_raises() -> None:
    with pytest.raises(ValueError, match="not found"):
        create_model("does-not-exist", task="classification")


def test_create_model_unknown_name_suggests_close_matches() -> None:
    with pytest.raises(ValueError, match="Did you mean 'dummy.classification'"):
        create_model("dummy.classifiction", task="classification")


def test_create_model_unknown_name_hints_other_task() -> None:
    with pytest.raises(ValueError, match="registered under task 'classification'; pass task='classification'"):
        create_model("dummy.classification", task="semantic-segmentation")


def test_create_model_segmentation_name_as_classification_hints_task() -> None:
    register_model("dummy-seg-only.segmentation", task="semantic-segmentation")(_dummy_segmentation)
    try:
        with pytest.raises(
            ValueError, match="registered under task 'semantic-segmentation'; pass task='semantic-segmentation'"
        ):
            create_model("dummy-seg-only.segmentation", task="classification")
    finally:
        _REGISTERED_MODELS["semantic-segmentation"].pop("dummy-seg-only.segmentation", None)


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
    assert entry["weights"] == {"url": "hf://torch-pointcloud/dummy-pretrained/resolve/main/model.pt"}


def test_register_model_keeps_weights_dict() -> None:
    weights = WeightsDict(
        url="hf://torch-pointcloud/dummy-meta/resolve/main/model.pt",
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


def test_create_model_former_name_warns_and_builds_the_renamed_model(_register_renamed_dummy: None) -> None:
    with pytest.warns(
        FutureWarning, match="'dummy-former.classification' was renamed to 'dummy-renamed.classification'"
    ):
        model, info = create_model("dummy-former.classification", return_info=True)
    assert isinstance(model, DummyClassificationModel)
    assert info["name"] == "dummy-renamed.classification"


def test_model_info_former_name_warns_and_returns_the_renamed_entry(_register_renamed_dummy: None) -> None:
    with pytest.warns(FutureWarning, match="renamed"):
        info = model_info("dummy-former.classification")
    assert info["name"] == "dummy-renamed.classification"


def test_list_models_skips_former_names(_register_renamed_dummy: None) -> None:
    assert list_models("dummy*", task="classification") == ["dummy-renamed.classification", "dummy.classification"]


def test_deprecated_model_stacks_one_decorator_per_former_name() -> None:
    @deprecated_model("dummy-former.classification", deprecated_in="0.0.17", removed_in="0.1.0")
    @deprecated_model("dummy-older.classification", deprecated_in="0.0.15")
    @register_model("dummy-renamed.classification", task="classification")
    def dummy_renamed(**kwargs: Any) -> DummyClassificationModel:
        return DummyClassificationModel(**kwargs)

    try:
        with pytest.warns(
            FutureWarning,
            match=r"'dummy-former.classification' was renamed to 'dummy-renamed.classification' in 0.0.17; the former "
            r"name will be removed in 0.1.0\.$",
        ):
            assert model_info("dummy-former.classification")["name"] == "dummy-renamed.classification"
        with pytest.warns(
            FutureWarning,
            match=r"'dummy-older.classification' was renamed to 'dummy-renamed.classification' in 0.0.15; the former "
            r"name will be removed in a future release\.$",
        ):
            assert model_info("dummy-older.classification")["name"] == "dummy-renamed.classification"
        assert _REGISTERED_MODELS["classification"]["dummy-renamed.classification"]["deprecated_names"] == (
            {"name": "dummy-older.classification", "deprecated_in": "0.0.15", "removed_in": None, "message": None},
            {"name": "dummy-former.classification", "deprecated_in": "0.0.17", "removed_in": "0.1.0", "message": None},
        )
    finally:
        _REGISTERED_MODELS["classification"].pop("dummy-renamed.classification", None)
        _DEPRECATED_MODELS["classification"].pop("dummy-former.classification", None)
        _DEPRECATED_MODELS["classification"].pop("dummy-older.classification", None)


def test_deprecated_model_appends_the_message_to_the_warning() -> None:
    @deprecated_model("dummy-former.classification", deprecated_in="0.0.17", message="The weights changed too.")
    @register_model("dummy-renamed.classification", task="classification")
    def dummy_renamed(**kwargs: Any) -> DummyClassificationModel:
        return DummyClassificationModel(**kwargs)

    try:
        with pytest.warns(FutureWarning, match=r"a future release\. The weights changed too\.$"):
            model_info("dummy-former.classification")
    finally:
        _REGISTERED_MODELS["classification"].pop("dummy-renamed.classification", None)
        _DEPRECATED_MODELS["classification"].pop("dummy-former.classification", None)


def test_deprecated_model_registers_a_class() -> None:
    @deprecated_model("dummy-class-former.classification", deprecated_in="0.0.17")
    @register_model("dummy-class.classification", task="classification")
    class DummyRenamedClass(DummyClassificationModel):
        pass

    try:
        with pytest.warns(FutureWarning, match="renamed to 'dummy-class.classification'"):
            model = create_model("dummy-class-former.classification")
        assert isinstance(model, DummyRenamedClass)
    finally:
        _REGISTERED_MODELS["classification"].pop("dummy-class.classification", None)
        _DEPRECATED_MODELS["classification"].pop("dummy-class-former.classification", None)


def test_create_model_former_name_of_a_popped_entry_is_not_found(_register_renamed_dummy: None) -> None:
    _REGISTERED_MODELS["classification"].pop("dummy-renamed.classification")
    for task in (None, "classification"):
        with pytest.raises(
            ValueError, match="'dummy-former.classification' not found in the 'classification' registry"
        ):
            create_model("dummy-former.classification", task=task)

    register_model("dummy-renamed.classification", task="classification")(_dummy_classification)
    with pytest.raises(ValueError, match="'dummy-former.classification' not found in the 'classification' registry"):
        create_model("dummy-former.classification")


def test_model_info_lists_the_former_names(_register_renamed_dummy: None) -> None:
    assert model_info("dummy-renamed.classification")["deprecated_names"] == (
        {"name": "dummy-former.classification", "deprecated_in": "0.0.17", "removed_in": "0.1.0", "message": None},
    )


@pytest.mark.parametrize(
    "kwargs,match",
    [
        pytest.param(dict(deprecated_in="soon"), "version string for `deprecated_in`", id="deprecated_in"),
        pytest.param(
            dict(deprecated_in="0.0.17", removed_in="later"), "version string for `removed_in`", id="removed_in"
        ),
        pytest.param(
            dict(deprecated_in="0.0.17", removed_in="0.0.17"),
            "removed in '0.0.17', which is not a later release than its deprecation in '0.0.17'",
            id="same-release",
        ),
        pytest.param(
            dict(deprecated_in="0.0.17", removed_in="0.0.16"),
            "removed in '0.0.16', which is not a later release than its deprecation in '0.0.17'",
            id="earlier-release",
        ),
    ],
)
def test_deprecated_model_invalid_schedule_raises(kwargs: Dict[str, str], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        deprecated_model("dummy-former.classification", **kwargs)


def test_deprecated_model_below_register_model_raises() -> None:
    with pytest.raises(ValueError, match="is not registered. Place `@deprecated_model` above `@register_model`"):

        @deprecated_model("dummy-former.classification", deprecated_in="0.0.17")
        def dummy_unregistered(**kwargs: Any) -> DummyClassificationModel:
            return DummyClassificationModel(**kwargs)


def test_deprecated_model_registered_under_several_names_raises() -> None:
    register_model("dummy-twice.classification", task="classification")(_dummy_classification)
    try:
        with pytest.raises(
            ValueError,
            match=r"registered as 'dummy.classification' \('classification'\), 'dummy-twice.classification' \('classification'\)",
        ):
            deprecated_model("dummy-former.classification", deprecated_in="0.0.17")(_dummy_classification)
    finally:
        _REGISTERED_MODELS["classification"].pop("dummy-twice.classification", None)


def test_deprecated_model_same_former_name_twice_raises() -> None:
    try:
        with pytest.raises(
            ValueError,
            match="'dummy-former.classification' is already a deprecated name of 'dummy-renamed.classification'",
        ):

            @deprecated_model("dummy-former.classification", deprecated_in="0.0.17")
            @deprecated_model("dummy-former.classification", deprecated_in="0.0.16")
            @register_model("dummy-renamed.classification", task="classification")
            def dummy_renamed(**kwargs: Any) -> DummyClassificationModel:
                return DummyClassificationModel(**kwargs)

    finally:
        _REGISTERED_MODELS["classification"].pop("dummy-renamed.classification", None)
        _DEPRECATED_MODELS["classification"].pop("dummy-former.classification", None)


def test_register_model_overwrite_drops_the_former_names(_register_renamed_dummy: None) -> None:
    with pytest.warns(UserWarning, match="overwriting the existing entry"):
        register_model("dummy-renamed.classification", task="classification")(_dummy_classification)
    with pytest.raises(ValueError, match="'dummy-former.classification' is not registered"):
        model_info("dummy-former.classification")


def test_deprecated_model_name_of_a_registered_model_raises() -> None:
    try:
        with pytest.raises(
            ValueError, match="'dummy.classification' is already a registered 'classification' model name"
        ):

            @deprecated_model("dummy.classification", deprecated_in="0.0.17")
            @register_model("dummy-other.classification", task="classification")
            def dummy_other(**kwargs: Any) -> DummyClassificationModel:
                return DummyClassificationModel(**kwargs)

    finally:
        _REGISTERED_MODELS["classification"].pop("dummy-other.classification", None)


def test_register_model_under_a_deprecated_name_raises(_register_renamed_dummy: None) -> None:
    with pytest.raises(ValueError, match="'dummy-former.classification' is a deprecated name of"):
        register_model("dummy-former.classification", task="classification")(_dummy_classification)


def test_create_model_former_name_resolves_under_its_task(_register_renamed_in_one_task_dummy: None) -> None:
    with pytest.warns(FutureWarning, match="'dummy-shared.demo' was renamed to 'dummy-shared-v2.demo'"):
        model = create_model("dummy-shared.demo", task="classification")
    assert isinstance(model, DummyClassificationModel)


@pytest.mark.filterwarnings("error::FutureWarning")
def test_create_model_name_kept_under_another_task_does_not_warn(_register_renamed_in_one_task_dummy: None) -> None:
    model = create_model("dummy-shared.demo", task="semantic-segmentation")
    assert isinstance(model, DummySemanticSegmentationModel)


def test_create_model_name_renamed_under_one_task_only_needs_task(_register_renamed_in_one_task_dummy: None) -> None:
    with pytest.raises(ValueError, match="pass `task=` to pick one"):
        create_model("dummy-shared.demo")


def test_create_model_former_name_under_another_task_hints_the_rename(
    _register_renamed_in_one_task_dummy: None,
) -> None:
    with pytest.raises(
        ValueError,
        match="was renamed to 'dummy-shared-v2.demo' under task 'classification'; pass task='classification'",
    ):
        create_model("dummy-shared.demo", task="pretraining")


def test_deprecated_model_name_equal_to_the_registered_name_raises() -> None:
    try:
        with pytest.raises(ValueError, match="'dummy-self.classification' is the registered name itself"):

            @deprecated_model("dummy-self.classification", deprecated_in="0.0.17")
            @register_model("dummy-self.classification", task="classification")
            def dummy_self(**kwargs: Any) -> DummyClassificationModel:
                return DummyClassificationModel(**kwargs)

    finally:
        _REGISTERED_MODELS["classification"].pop("dummy-self.classification", None)


def test_deprecated_model_name_of_another_model_raises(_register_renamed_dummy: None) -> None:
    try:
        with pytest.raises(
            ValueError,
            match="'dummy-former.classification' is already a deprecated name of 'dummy-renamed.classification'",
        ):

            @deprecated_model("dummy-former.classification", deprecated_in="0.0.17")
            @register_model("dummy-other.classification", task="classification")
            def dummy_other(**kwargs: Any) -> DummyClassificationModel:
                return DummyClassificationModel(**kwargs)

    finally:
        _REGISTERED_MODELS["classification"].pop("dummy-other.classification", None)


@pytest.mark.filterwarnings("error::UserWarning")
def test_create_model_pretrained_loads_weights(_register_pretrained_dummy: Path) -> None:
    model = create_model("dummy-pretrained.classification", task="classification", pretrained=True)
    assert isinstance(model, DummyClassificationModel)
    state_dict = torch.load(_register_pretrained_dummy, weights_only=True)
    assert torch.equal(model.encoder.weight, state_dict["encoder.weight"])
    assert isinstance(model.fc, nn.Linear)
    assert torch.equal(model.fc.weight, state_dict["fc.weight"])


def test_create_model_pretrained_non_commercial_license_warns(_register_pretrained_dummy: Path) -> None:
    _REGISTERED_MODELS["classification"]["dummy-pretrained.classification"]["weights"] = WeightsDict(
        url="hf://torch-pointcloud/dummy-pretrained/resolve/main/model.pt",
        license="CC-BY-NC-4.0",
    )
    with pytest.warns(UserWarning, match=r"'dummy-pretrained.classification' are released under CC-BY-NC-4.0"):
        create_model("dummy-pretrained.classification", task="classification", pretrained=True)


@pytest.mark.filterwarnings("error::UserWarning")
@pytest.mark.parametrize(
    "weights",
    [
        pytest.param(
            WeightsDict(url="hf://torch-pointcloud/dummy-pretrained/resolve/main/model.pt", license="MIT"), id="mit"
        ),
        pytest.param(WeightsDict(url="hf://torch-pointcloud/dummy-pretrained/resolve/main/model.pt"), id="no-license"),
    ],
)
def test_create_model_pretrained_other_license_does_not_warn(
    _register_pretrained_dummy: Path,
    weights: WeightsDict,
) -> None:
    _REGISTERED_MODELS["classification"]["dummy-pretrained.classification"]["weights"] = weights
    create_model("dummy-pretrained.classification", task="classification", pretrained=True)


@pytest.mark.filterwarnings("error::UserWarning")
def test_create_model_pretrained_num_classes_override_rebuilds_head(_register_pretrained_dummy: Path) -> None:
    model = create_model("dummy-pretrained.classification", task="classification", pretrained=True, num_classes=3)
    assert isinstance(model, DummyClassificationModel)
    state_dict = torch.load(_register_pretrained_dummy, weights_only=True)
    assert model.fc.weight.shape == (3, 8)
    assert torch.equal(model.encoder.weight, state_dict["encoder.weight"])


@pytest.mark.filterwarnings("error::UserWarning")
def test_create_model_pretrained_registered_num_classes_keeps_head(_register_pretrained_dummy: Path) -> None:
    model = create_model("dummy-pretrained.classification", task="classification", pretrained=True, num_classes=5)
    assert isinstance(model, DummyClassificationModel)
    state_dict = torch.load(_register_pretrained_dummy, weights_only=True)
    assert isinstance(model.fc, nn.Linear)
    assert torch.equal(model.fc.weight, state_dict["fc.weight"])


@pytest.mark.filterwarnings("error::UserWarning")
def test_create_model_pretrained_registered_num_classes_as_a_sequence_keeps_head(
    _register_pretrained_dummy: Path,
) -> None:
    model = create_model("dummy-pretrained.classification", task="classification", pretrained=True, num_classes=[5])
    assert isinstance(model, DummyClassificationModel)
    state_dict = torch.load(_register_pretrained_dummy, weights_only=True)
    assert isinstance(model.fc, nn.Linear)
    assert torch.equal(model.fc.weight, state_dict["fc.weight"])


@pytest.mark.filterwarnings("error::UserWarning")
def test_create_model_pretrained_in_channels_override_keeps_fresh_stem(_register_pretrained_dummy: Path) -> None:
    model = create_model("dummy-pretrained.classification", task="classification", pretrained=True, in_channels=4)
    assert isinstance(model, DummyClassificationModel)
    assert model.encoder.weight.shape == (8, 4)
    state_dict = torch.load(_register_pretrained_dummy, weights_only=True)
    assert torch.equal(model.encoder.bias, state_dict["encoder.bias"])
    assert isinstance(model.fc, nn.Linear)
    assert torch.equal(model.fc.weight, state_dict["fc.weight"])


@pytest.mark.filterwarnings("error::UserWarning")
def test_create_model_pretrained_headless_loads_backbone_without_warning(_register_pretrained_dummy: Path) -> None:
    model = create_model("dummy-pretrained.classification", task="classification", pretrained=True, num_classes=0)
    assert isinstance(model, DummyClassificationModel)
    assert isinstance(model.fc, nn.Identity)
    state_dict = torch.load(_register_pretrained_dummy, weights_only=True)
    assert torch.equal(model.encoder.weight, state_dict["encoder.weight"])


def test_create_model_pretrained_headless_warns_about_unexpected_keys(_register_pretrained_dummy: Path) -> None:
    state_dict = torch.load(_register_pretrained_dummy, weights_only=True)
    state_dict["extra.weight"] = torch.ones(2)
    torch.save(state_dict, _register_pretrained_dummy)
    with pytest.warns(UserWarning, match=r"absent from the model: extra\.weight\.$"):
        model = create_model("dummy-pretrained.classification", task="classification", pretrained=True, num_classes=0)
    assert isinstance(model, DummyClassificationModel)
    assert isinstance(model.fc, nn.Identity)
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


def _strip_backbone_prefix(state_dict: Dict[str, Any], model: nn.Module) -> Dict[str, Any]:
    return {key.removeprefix("backbone."): value for key, value in state_dict.items()}


@pytest.mark.filterwarnings("error::UserWarning")
def test_create_model_pretrained_applies_pretrained_filter_fn(_register_pretrained_dummy: Path) -> None:
    state_dict = torch.load(_register_pretrained_dummy, weights_only=True)
    torch.save({f"backbone.{key}": value for key, value in state_dict.items()}, _register_pretrained_dummy)
    filter_fn = Mock(wraps=_strip_backbone_prefix)
    _REGISTERED_MODELS["classification"]["dummy-pretrained.classification"]["weights"] = WeightsDict(
        url="hf://torch-pointcloud/dummy-pretrained/resolve/main/model.pt",
        pretrained_filter_fn=filter_fn,
    )
    model = create_model("dummy-pretrained.classification", task="classification", pretrained=True)
    assert isinstance(model, DummyClassificationModel)
    filter_fn.assert_called_once()
    assert filter_fn.call_args.args[1] is model
    assert torch.equal(model.encoder.weight, state_dict["encoder.weight"])
    assert isinstance(model.fc, nn.Linear)
    assert torch.equal(model.fc.weight, state_dict["fc.weight"])


@pytest.mark.filterwarnings("error::UserWarning")
def test_create_model_pretrained_reads_several_urls_into_one_state_dict(_register_pretrained_dummy: Path) -> None:
    state_dict = torch.load(_register_pretrained_dummy, weights_only=True)
    head_path = _register_pretrained_dummy.with_name("head.pt")
    torch.save({key: value for key, value in state_dict.items() if key.startswith("fc.")}, head_path)
    torch.save(
        {key: value for key, value in state_dict.items() if key.startswith("encoder.")},
        _register_pretrained_dummy,
    )
    _REGISTERED_MODELS["classification"]["dummy-pretrained.classification"]["weights"] = WeightsDict(
        url=(
            "hf://torch-pointcloud/dummy-pretrained/resolve/main/model.pt",
            "hf://torch-pointcloud/dummy-pretrained/resolve/main/head.pt",
        ),
    )
    model = create_model("dummy-pretrained.classification", task="classification", pretrained=True)
    assert isinstance(model, DummyClassificationModel)
    assert torch.equal(model.encoder.weight, state_dict["encoder.weight"])
    assert isinstance(model.fc, nn.Linear)
    assert torch.equal(model.fc.weight, state_dict["fc.weight"])


def test_create_model_pretrained_warns_when_a_later_url_overrides_keys(_register_pretrained_dummy: Path) -> None:
    state_dict = torch.load(_register_pretrained_dummy, weights_only=True)
    head_path = _register_pretrained_dummy.with_name("head.pt")
    torch.save({**state_dict, "fc.weight": torch.zeros_like(state_dict["fc.weight"])}, head_path)
    _REGISTERED_MODELS["classification"]["dummy-pretrained.classification"]["weights"] = WeightsDict(
        url=(
            "hf://torch-pointcloud/dummy-pretrained/resolve/main/model.pt",
            "hf://torch-pointcloud/dummy-pretrained/resolve/main/head.pt",
        ),
    )
    with pytest.warns(UserWarning, match=r"head\.pt' of 'dummy-pretrained.classification' overrides keys .*fc\.weight"):
        model = create_model("dummy-pretrained.classification", task="classification", pretrained=True)
    assert isinstance(model, DummyClassificationModel)
    assert isinstance(model.fc, nn.Linear)
    assert torch.equal(model.fc.weight, torch.zeros_like(state_dict["fc.weight"]))


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


def test_create_model_missing_optional_dependency_raises_actionable_import_error() -> None:
    """With `spconv` absent, building a model that needs it must name the model, the dependency and the
    installation page, and not the generic requirement wording."""
    code = (
        "import sys\n"
        "sys.modules['spconv'] = None\n"
        "from torch_pointcloud.models import create_model\n"
        "try:\n"
        "    create_model('spunet-v1m1.scannet20.pointcept')\n"
        "except ImportError as err:\n"
        "    print(err)\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    message = result.stdout.strip()
    assert "Model 'spunet-v1m1.scannet20.pointcept'" in message
    assert "Optional module 'spconv.pytorch' is not installed." in message
    assert "https://pytorch-pointcloud.org/latest/installation/" in message
    assert "does not meet the requirement" not in message


@pytest.mark.skipif(not _SPCONV_AVAILABLE, reason="spconv is not installed")
def test_create_model_missing_flash_attn_hints_use_flash_attn_false(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("torch_pointcloud.layers.serialized_attention._FLASH_ATTN_AVAILABLE", False)
    with pytest.raises(ImportError, match="or pass `use_flash_attn=False`") as exc_info:
        create_model("ptv3-base.scannet20.pointcept", use_flash_attn=True)
    assert "Model 'ptv3-base.scannet20.pointcept'" in str(exc_info.value)
    assert "https://pytorch-pointcloud.org/latest/installation/" in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, ImportError)


def test_create_model_pretrained_rejects_weight_path_escaping_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("torch_pointcloud.models._registry.MODELS_DIR", tmp_path)
    register_model(
        "dummy-escape.classification",
        task="classification",
        weights="hf://torch-pointcloud/../resolve/main/model.pt",
    )(_dummy_classification)

    try:
        with pytest.raises(ValueError, match="outside the models cache"):
            create_model("dummy-escape.classification", task="classification", pretrained=True)
    finally:
        _REGISTERED_MODELS["classification"].pop("dummy-escape.classification", None)


def test_create_model_pretrained_downloads_missing_weights(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("torch_pointcloud.models._registry.MODELS_DIR", tmp_path)
    register_model(
        "dummy-download.classification",
        task="classification",
        weights="hf://torch-pointcloud/dummy-download/resolve/v1.0/model.pt",
    )(_dummy_classification)

    def fake_download(url: str, file_path: Path, **kwargs: Any) -> str:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(DummyClassificationModel().state_dict(), file_path)
        return file_path.as_posix()

    download = Mock(side_effect=fake_download)
    monkeypatch.setattr("torch_pointcloud.models._registry.download_url", download)
    monkeypatch.setenv("HF_TOKEN", "hf_secret")

    try:
        model = create_model("dummy-download.classification", task="classification", pretrained=True)
    finally:
        _REGISTERED_MODELS["classification"].pop("dummy-download.classification", None)

    assert isinstance(model, DummyClassificationModel)
    assert download.call_args.args[0] == "https://huggingface.co/torch-pointcloud/dummy-download/resolve/v1.0/model.pt"
    assert download.call_args.kwargs["headers"] == {"Authorization": "Bearer hf_secret"}
    assert (tmp_path / "dummy-download" / "v1.0" / "model.pt").exists()
    assert not list(tmp_path.rglob("*.part"))


def test_create_model_pretrained_download_failure_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("torch_pointcloud.models._registry.MODELS_DIR", tmp_path)
    register_model(
        "dummy-offline.classification",
        task="classification",
        weights="hf://torch-pointcloud/dummy-offline/resolve/main/model.pt",
    )(_dummy_classification)
    monkeypatch.setattr("torch_pointcloud.models._registry.download_url", Mock(side_effect=OSError("offline")))

    try:
        with pytest.raises(FileNotFoundError, match="Could not download"):
            create_model("dummy-offline.classification", task="classification", pretrained=True)
    finally:
        _REGISTERED_MODELS["classification"].pop("dummy-offline.classification", None)

    assert not (tmp_path / "dummy-offline" / "main" / "model.pt").exists()


def test_create_model_pretrained_gated_download_names_the_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("torch_pointcloud.models._registry.MODELS_DIR", tmp_path)
    register_model(
        "dummy-gated.classification",
        task="classification",
        weights="hf://authors/dummy-gated/resolve/main/model.pt",
    )(_dummy_classification)
    url = "https://huggingface.co/authors/dummy-gated/resolve/main/model.pt"
    error = HTTPError(url, 401, "Unauthorized", Message(), None)
    monkeypatch.setattr("torch_pointcloud.models._registry.download_url", Mock(side_effect=error))

    try:
        with pytest.raises(
            FileNotFoundError, match="gated: accept its terms at https://huggingface.co/authors/dummy-gated"
        ):
            create_model("dummy-gated.classification", task="classification", pretrained=True)
    finally:
        _REGISTERED_MODELS["classification"].pop("dummy-gated.classification", None)

    assert not list(tmp_path.rglob("*.part"))


def test_cache_path_rejects_several_urls() -> None:
    with pytest.raises(TypeError, match="Expected a single weight URL string, got tuple"):
        cache_path(("hf://torch-pointcloud/ckpt/resolve/aaa/model.safetensors",))  # type: ignore[arg-type]


def test_cache_path_is_keyed_by_revision() -> None:
    """A checkpoint pinned to a new revision must not reuse the file cached for the previous one."""
    old = cache_path("hf://torch-pointcloud/ckpt/resolve/aaa/model.safetensors")
    new = cache_path("hf://torch-pointcloud/ckpt/resolve/bbb/model.safetensors")
    assert old.parent.name == "aaa"
    assert old != new


def test_model_info_matches_the_entry_of_create_model() -> None:
    _, built = create_model("pointnet.modelnet40", return_info=True)
    assert model_info("pointnet.modelnet40") == built
    with pytest.raises(ValueError, match="not registered"):
        model_info("not-a-model")
