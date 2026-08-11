from functools import partial
from typing import Any, Dict, Iterator, Tuple
from unittest.mock import Mock

import pytest
import torch
from torch import Tensor, nn

from torch_pointcloud.lightning import (
    BNMomentumScheduler,
    LitClassificationModel,
    LitDetectionModel,
    MeanAveragePrecision3D,
    MetricCallback,
)
from torch_pointcloud.models import ClassificationModel, DetectionModel, SegmentationModel, register_model
from torch_pointcloud.models._registry import _REGISTERED_MODELS, Task
from torch_pointcloud.utils.types import Detection3D

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
    def __init__(
        self, in_channels: int = 1, num_classes: int = 10, num_heading_bin: int = 12, num_size_cluster: int = 10
    ) -> None:
        super().__init__(in_channels=in_channels, num_classes=num_classes)
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.register_buffer("mean_sizes", torch.ones(num_size_cluster, 3))
        self.fc = nn.Linear(in_channels, num_classes)

    def forward_features(self, x: Tensor, pos: Tensor, batch: Tensor) -> Tensor:
        return x

    def forward(self, x: Tensor, pos: Tensor, batch: Tensor) -> Dict[str, Tensor]:
        return {"objectness_scores": self.fc(x)}

    def decode(self, output: Dict[str, Tensor]) -> Detection3D:
        scores = output["objectness_scores"]
        return {
            "boxes": scores.new_zeros(0, 7),
            "scores": scores.new_zeros(0),
            "labels": scores.new_zeros(0, dtype=torch.long),
            "batch": scores.new_zeros(0, dtype=torch.long),
        }


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
    dummies: Tuple[Tuple[Task, str], ...] = (
        ("classification", "dummy.classification"),
        ("segmentation", "dummy.segmentation"),
        ("detection", "dummy.detection"),
    )
    for task, name in dummies:
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


def test_metric_callback_batch_key_forwards_third_update_arg(trainer: L.Trainer) -> None:
    """With `batch_key` set, `metric.update` receives the step output's batch index as a third positional
    argument (packed multi-shape metrics need it); without it, the two-argument call is unchanged."""
    module = _cls_module(num_classes=3)
    metric = Mock()
    callback = MetricCallback(metric=metric, name="ins_mIoU", stages=("val",), batch_key="batch")
    preds = torch.randn(4, 3)
    target = torch.tensor([0, 1, 2, 1])
    batch_index = torch.tensor([0, 0, 1, 1])

    callback.on_validation_batch_end(trainer, module, {"preds": preds, "target": target, "batch": batch_index}, {}, 0)

    metric.update.assert_called_once()
    args = metric.update.call_args.args
    assert args[0] is preds
    assert args[1] is target
    assert args[2] is batch_index


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


def test_metric_callback_raises_on_non_dict_step_output(trainer: L.Trainer) -> None:
    """A `*_step` returning a bare tensor would silently skip the metric update; it must fail loudly."""
    module = _cls_module(num_classes=3)
    callback = MetricCallback(metric=Accuracy(task="multiclass", num_classes=3), name="accuracy", stages=("val",))

    with pytest.raises(TypeError, match="step output"):
        callback.on_validation_batch_end(trainer, module, torch.randn(2, 3), {}, 0)


def test_metric_callback_logs_on_test_stage(trainer: L.Trainer, monkeypatch: pytest.MonkeyPatch) -> None:
    """With `stages=("test",)` the test hooks accumulate and log the metric as `test/{name}`."""
    module = _cls_module(num_classes=3)
    log = Mock()
    monkeypatch.setattr(module, "log", log)
    callback = MetricCallback(metric=Accuracy(task="multiclass", num_classes=3), name="accuracy", stages=("test",))
    preds = torch.tensor([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    target = torch.tensor([0, 0])

    callback.on_test_epoch_start(trainer, module)
    callback.on_test_batch_end(trainer, module, {"preds": preds, "target": target}, {}, 0)
    callback.on_test_epoch_end(trainer, module)

    logged = {call.args[0]: call.args[1] for call in log.call_args_list}
    assert logged["test/accuracy"].item() == pytest.approx(0.5)


def test_metric_callback_test_stage_ignores_validation_hooks(
    trainer: L.Trainer, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _cls_module(num_classes=3)
    log = Mock()
    monkeypatch.setattr(module, "log", log)
    metric = Mock()
    callback = MetricCallback(metric=metric, name="accuracy", stages=("test",))

    callback.on_validation_epoch_start(trainer, module)
    callback.on_validation_batch_end(
        trainer, module, {"preds": torch.randn(2, 3), "target": torch.tensor([0, 1])}, {}, 0
    )
    callback.on_validation_epoch_end(trainer, module)

    metric.update.assert_not_called()
    log.assert_not_called()


@pytest.mark.parametrize(
    ("epoch", "expected"),
    [
        pytest.param(0, 0.5, id="epoch-0"),
        pytest.param(19, 0.5, id="before-first-decay"),
        pytest.param(20, 0.25, id="first-decay"),
        pytest.param(45, 0.125, id="second-decay"),
        pytest.param(1000, 0.001, id="clipped-at-floor"),
    ],
)
def test_bn_momentum_scheduler_decays_batchnorm_momentum(epoch: int, expected: float) -> None:
    callback = BNMomentumScheduler(bn_momentum_init=0.5, bn_decay_rate=0.5, bn_decay_step=20, bn_momentum_clip=0.001)
    model = nn.Sequential(nn.Linear(3, 4), nn.BatchNorm1d(4), nn.BatchNorm2d(4), nn.BatchNorm3d(4))
    pl_module = Mock(model=model)

    callback.on_train_epoch_start(Mock(current_epoch=epoch), pl_module)

    assert model[1].momentum == pytest.approx(expected)
    assert model[2].momentum == pytest.approx(expected)
    assert model[3].momentum == pytest.approx(expected)
