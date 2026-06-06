from functools import partial
from typing import Any, Dict, Iterator
from unittest.mock import Mock

import pytest
import torch
from torch import Tensor, nn

from torch_pointcloud.lightning import (
    LitClassificationModel,
    LitDetectionModel,
    MeanAveragePrecision3D,
    MetricCallback,
)
from torch_pointcloud.models import ClassificationModel, DetectionModel, SegmentationModel, register_model
from torch_pointcloud.models._registry import _REGISTERED_MODELS

pytest.importorskip("lightning.pytorch")

import lightning.pytorch as L  # noqa: E402
from torchmetrics import Accuracy, JaccardIndex  # noqa: E402
from torchmetrics.classification import MulticlassJaccardIndex  # noqa: E402


class DummyClassificationModel(ClassificationModel):
    def __init__(self, in_channels: int = 3, num_classes: int = 3) -> None:
        super().__init__(in_channels=in_channels, num_classes=num_classes)
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x: Tensor, pos: Tensor, batch: Tensor) -> Tensor:
        return self.fc(x)


class DummySegmentationModel(SegmentationModel):
    def __init__(self, in_channels: int = 3, num_classes: int = 5) -> None:
        super().__init__(in_channels=in_channels, num_classes=num_classes)
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x: Tensor, pos: Tensor, batch: Tensor) -> Tensor:
        return self.fc(x)


class DummyDetectionModel(DetectionModel):
    def __init__(self, in_channels: int = 1, num_classes: int = 1) -> None:
        super().__init__(in_channels=in_channels, num_classes=num_classes)
        self.register_buffer("mean_sizes", torch.ones(num_classes, 3))

    def forward(self, x: Tensor, pos: Tensor, batch: Tensor) -> Dict[str, Tensor]:
        return {}

    def decode(self, out: Dict[str, Tensor], pos: Tensor, batch: Tensor) -> Dict[str, Tensor]:
        return out


def _dummy_classification(**kwargs: Any) -> DummyClassificationModel:
    return DummyClassificationModel(**kwargs)


def _dummy_segmentation(**kwargs: Any) -> DummySegmentationModel:
    return DummySegmentationModel(**kwargs)


def _dummy_detection(**kwargs: Any) -> DummyDetectionModel:
    return DummyDetectionModel(**kwargs)


@pytest.fixture(autouse=True)
def _register_dummies() -> Iterator[None]:
    """The LightningModules build their model via `create_model(name, ...)`, so the test doubles are
    registered here (and removed afterwards, to keep the global registry clean for other tests)."""
    register_model("dummy.classification", task="classification")(_dummy_classification)
    register_model("dummy.segmentation", task="segmentation")(_dummy_segmentation)
    register_model("dummy.detection", task="detection")(_dummy_detection)
    yield
    for task, name in (
        ("classification", "dummy.classification"),
        ("segmentation", "dummy.segmentation"),
        ("detection", "dummy.detection"),
    ):
        _REGISTERED_MODELS[task].pop(name, None)


@pytest.fixture
def trainer() -> L.Trainer:
    return L.Trainer(logger=False, enable_checkpointing=False, enable_progress_bar=False, accelerator="cpu", devices=1)


def _cls_module(num_classes: int = 3) -> LitClassificationModel:
    return LitClassificationModel(
        name="dummy.classification",
        num_classes=num_classes,
        optimizer=partial(torch.optim.AdamW, lr=0.01),
    )


def _det_module() -> LitDetectionModel:
    return LitDetectionModel(
        name="dummy.detection",
        num_classes=1,
        optimizer=partial(torch.optim.Adam, lr=1e-3),
        criterion=Mock(),
    )


def test_metric_callback_holds_ready_metric() -> None:
    """The callback holds the provided metric as-is (Hydra builds it; there is no factory/setup step)."""
    metric = JaccardIndex(task="multiclass", num_classes=5, ignore_index=-1)
    callback = MetricCallback(metric=metric, name="mIoU")
    assert callback.metric is metric
    assert isinstance(callback.metric, MulticlassJaccardIndex)
    assert callback.metric.ignore_index == -1


def test_metric_callback_logs_epoch_accuracy(trainer: L.Trainer, monkeypatch: pytest.MonkeyPatch) -> None:
    """A passed-in metric is updated each batch and the epoch value is logged as `{stage}/{name}`."""
    module = _cls_module(num_classes=3)
    log = Mock()
    monkeypatch.setattr(module, "log", log)
    callback = MetricCallback(metric=Accuracy(task="multiclass", num_classes=3), name="accuracy", stages=("val",))
    preds = torch.tensor([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0], [2.0, 0.0, 0.0]])
    target = torch.tensor([0, 1, 2, 1])

    callback.on_validation_epoch_start(trainer, module)
    callback.on_validation_batch_end(trainer, module, {"preds": preds, "target": target}, {}, 0)
    callback.on_validation_epoch_end(trainer, module)

    logged = {call.args[0]: call.args[1] for call in log.call_args_list}
    assert logged["val/accuracy"].item() == pytest.approx(0.75)


def test_metric_callback_skips_unlisted_stage(trainer: L.Trainer, monkeypatch: pytest.MonkeyPatch) -> None:
    """A stage not in `stages` is left untouched (no update, no logging)."""
    module = _cls_module(num_classes=3)
    log = Mock()
    monkeypatch.setattr(module, "log", log)
    callback = MetricCallback(metric=Accuracy(task="multiclass", num_classes=3), name="accuracy", stages=("val",))

    callback.on_test_epoch_start(trainer, module)
    callback.on_test_batch_end(trainer, module, {"preds": torch.randn(2, 3), "target": torch.tensor([0, 1])}, {}, 0)
    callback.on_test_epoch_end(trainer, module)

    log.assert_not_called()


def test_metric_callback_logs_detection_map(trainer: L.Trainer, monkeypatch: pytest.MonkeyPatch) -> None:
    """A perfect detection scored through `MetricCallback` + `MeanAveragePrecision3D` logs `mAP@t = 1.0`."""
    module = _det_module()
    log = Mock()
    monkeypatch.setattr(module, "log", log)
    callback = MetricCallback(metric=MeanAveragePrecision3D(iou_thresholds=(0.25, 0.5)), name="mAP", stages=("val",))
    preds = {
        "boxes": torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]]),
        "scores": torch.tensor([0.9]),
        "labels": torch.tensor([0]),
        "batch": torch.tensor([0]),
    }
    target = {
        "boxes": torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]]),
        "labels": torch.tensor([0]),
        "batch": torch.tensor([0]),
    }

    callback.on_validation_epoch_start(trainer, module)
    callback.on_validation_batch_end(trainer, module, {"preds": preds, "target": target}, {}, 0)
    callback.on_validation_epoch_end(trainer, module)

    logged = {call.args[0]: call.args[1] for call in log.call_args_list}
    assert logged["val/mAP@0.25"] == pytest.approx(1.0)
    assert logged["val/mAP@0.5"] == pytest.approx(1.0)
