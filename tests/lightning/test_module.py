from functools import partial
from typing import Any, Dict, Iterator, Tuple
from unittest.mock import Mock

import pytest
import torch
from torch import Tensor, nn
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import Dataset
from torch_geometric.utils import scatter

from torch_pointcloud.lightning import (
    LitClassificationModel,
    LitDetectionModel,
    LitSegmentationModel,
    PointCloudDataModule,
)
from torch_pointcloud.models import ClassificationModel, DetectionModel, SegmentationModel, register_model
from torch_pointcloud.models._registry import _REGISTERED_MODELS, Task

pytest.importorskip("lightning.pytorch")

import lightning.pytorch as L  # noqa: E402


class DummyClassificationModel(ClassificationModel):
    def __init__(self, in_channels: int = 3, num_classes: int = 5) -> None:
        super().__init__(in_channels=in_channels, num_classes=num_classes)
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x: Tensor, pos: Tensor, batch: Tensor) -> Tensor:
        pooled = scatter(x, batch, dim=0, reduce="mean")
        return self.fc(pooled)


class DummySegmentationModel(SegmentationModel):
    def __init__(self, in_channels: int = 3, num_classes: int = 4) -> None:
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

    def forward(self, x: Tensor, pos: Tensor, batch: Tensor) -> Dict[str, Tensor]:
        return {"objectness_scores": self.fc(x)}


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


class DummySegmentationDataset(Dataset):
    def __init__(self, n: int = 4) -> None:
        self._n = n

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        return {
            "x": torch.randn(6, 3),
            "pos": torch.randn(6, 3),
            "segment": torch.randint(0, 4, (6,)),
        }


def _make_seg_module(*, scheduler: Any = None, param_groups: Any = None) -> LitSegmentationModel:
    return LitSegmentationModel(
        name="dummy.segmentation",
        target_key="segment",
        optimizer=partial(torch.optim.AdamW, lr=0.01),
        scheduler=scheduler,
        scheduler_interval="step",
        param_groups=param_groups,
    )


def _make_cls_module() -> LitClassificationModel:
    return LitClassificationModel(
        name="dummy.classification",
        optimizer=partial(torch.optim.AdamW, lr=0.01),
    )


def _make_det_module(*, scheduler: Any = None, param_groups: Any = None) -> LitDetectionModel:
    return LitDetectionModel(
        name="dummy.detection",
        optimizer=partial(torch.optim.AdamW, lr=0.01),
        criterion=Mock(),
        scheduler=scheduler,
        param_groups=param_groups,
    )


def test_seg_forward_shapes() -> None:
    lit = _make_seg_module()
    batch = {
        "x": torch.randn(12, 3),
        "pos": torch.randn(12, 3),
        "batch": torch.cat([torch.zeros(6, dtype=torch.long), torch.ones(6, dtype=torch.long)]),
        "segment": torch.randint(0, 4, (12,)),
    }
    logits = lit(batch)
    assert logits.shape == (12, 4)


def test_seg_training_step_returns_scalar_loss() -> None:
    lit = _make_seg_module()
    batch = {
        "x": torch.randn(12, 3),
        "pos": torch.randn(12, 3),
        "batch": torch.cat([torch.zeros(6, dtype=torch.long), torch.ones(6, dtype=torch.long)]),
        "segment": torch.randint(0, 4, (12,)),
    }
    loss = lit.training_step(batch, batch_idx=0)
    assert isinstance(loss, Tensor) and loss.dim() == 0


def test_cls_module_forward_shapes() -> None:
    lit = _make_cls_module()
    batch = {
        "x": torch.randn(12, 3),
        "pos": torch.randn(12, 3),
        "batch": torch.cat([torch.zeros(6, dtype=torch.long), torch.ones(6, dtype=torch.long)]),
        "label": torch.tensor([0, 3]),
    }
    logits = lit(batch)
    assert logits.shape == (2, 5)


def test_configure_optimizers_without_scheduler_returns_optimizer() -> None:
    lit = _make_seg_module(scheduler=None)
    out = lit.configure_optimizers()
    assert isinstance(out, torch.optim.Optimizer)


def test_configure_optimizers_uses_param_groups() -> None:
    lit = _make_seg_module(
        param_groups={
            "layer_matches": [lambda name: name.startswith("model.fc")],
            "match_types": "filter",
            "lr_values": 0.0001,
        },
    )
    optim = lit.configure_optimizers()
    # 2 groups: matched + others (others is empty since fc is the only sub-module).
    assert len(optim.param_groups) == 2
    assert optim.param_groups[0]["lr"] == 0.0001
    matched_ids = {id(p) for p in optim.param_groups[0]["params"]}
    expected_ids = {id(p) for n, p in lit.named_parameters() if n.startswith("model.fc")}
    assert matched_ids == expected_ids


def test_fit_smoke_with_explicit_scheduler() -> None:
    """End-to-end: a real `fit` runs train + val steps with a scheduler whose `total_steps`
    is set explicitly (the LitModule no longer auto-injects it)."""
    lit = _make_seg_module(scheduler=partial(OneCycleLR, max_lr=0.01, total_steps=10))
    dm = PointCloudDataModule(
        train_dataset=DummySegmentationDataset(4),
        val_dataset=DummySegmentationDataset(2),
        batch_size=2,
        num_workers=0,
    )
    trainer = L.Trainer(
        fast_dev_run=True,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_progress_bar=False,
        enable_checkpointing=False,
    )
    trainer.fit(lit, datamodule=dm)
    assert "train/loss" in trainer.callback_metrics


def test_mix3d_halves_batch_index_during_training(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mix3D with `mix_prob=1.0` always merges adjacent scene pairs."""
    lit = LitSegmentationModel(
        name="dummy.segmentation",
        optimizer=partial(torch.optim.AdamW, lr=0.01),
        mix_prob=1.0,
    )
    assert lit.mix_prob == 1.0
    trainer = Mock()
    trainer.training = True
    monkeypatch.setattr(lit, "_trainer", trainer)
    batch = {"batch": torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])}
    out = lit.on_after_batch_transfer(batch, dataloader_idx=0)
    # 0,0,1,1,2,2,3,3 -> 0,0,0,0,1,1,1,1 after `// 2`.
    assert torch.equal(out["batch"], torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]))


def test_detection_criterion_completed_with_model_params() -> None:
    """`__init__` forwards the model's head-geometry params (not the model itself) to the criterion factory."""
    criterion = Mock()
    lit = LitDetectionModel(name="dummy.detection", optimizer=partial(torch.optim.AdamW, lr=0.01), criterion=criterion)
    criterion.assert_called_once()
    kwargs = criterion.call_args.kwargs
    assert kwargs["num_heading_bin"] == lit.model.num_heading_bin
    assert kwargs["num_size_cluster"] == lit.model.num_size_cluster
    assert kwargs["num_classes"] == lit.model.num_classes
    assert kwargs["mean_sizes"] is lit.model.mean_sizes


def test_detection_forward_delegates_to_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """`forward` resolves the input keys and passes them positionally to the model."""
    module = _make_det_module()
    output = {"objectness_scores": torch.randn(2, 10)}
    forward = Mock(return_value=output)
    monkeypatch.setattr(module.model, "forward", forward)
    batch = {"x": torch.rand(6, 1), "pos": torch.rand(6, 3), "batch": torch.zeros(6, dtype=torch.long)}
    out = module(batch)
    assert out is output
    args = forward.call_args.args
    assert args[0] is batch["x"]
    assert args[1] is batch["pos"]
    assert args[2] is batch["batch"]


def test_detection_step_feeds_output_and_batch_to_criterion(monkeypatch: pytest.MonkeyPatch) -> None:
    """`step` calls the criterion with `(output, batch)`, logs each component, and returns output + total loss."""
    module = _make_det_module()
    output = {"objectness_scores": torch.randn(2, 10)}
    loss_dict = {"loss": torch.tensor(1.5, requires_grad=True), "obj_acc": torch.tensor(0.5)}
    forward = Mock(return_value=output)
    criterion = Mock(return_value=loss_dict)
    log = Mock()
    monkeypatch.setattr(module, "forward", forward)
    monkeypatch.setattr(module, "criterion", criterion)
    monkeypatch.setattr(module, "log", log)
    batch = {"batch": torch.tensor([0, 0, 1, 1])}
    result = module.step(batch, "train")
    forward.assert_called_once_with(batch)
    criterion.assert_called_once_with(output, batch)
    assert result["output"] is output
    assert result["loss"] is loss_dict["loss"]
    logged = {call.args[0]: call.kwargs["prog_bar"] for call in log.call_args_list}
    assert logged == {"train/loss": True, "train/obj_acc": True}


def test_detection_training_step_delegates_to_step(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _make_det_module()
    sentinel = torch.tensor(1.0)
    step = Mock(return_value={"output": {}, "loss": sentinel})
    monkeypatch.setattr(module, "step", step)
    batch = {"batch": torch.tensor([0, 0, 1, 1])}
    out = module.training_step(batch, batch_idx=0)
    step.assert_called_once_with(batch, "train")
    assert out is sentinel


def test_detection_validation_step_decodes_filters_and_targets(monkeypatch: pytest.MonkeyPatch) -> None:
    """`validation_step` runs raw `decode`, drops low-confidence boxes + per-class NMS, and pairs with the GT."""
    module = _make_det_module()
    output = {"objectness_scores": torch.randn(2, 10)}
    decoded = {
        "boxes": torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0], [9.0, 9.0, 9.0, 1.0, 1.0, 1.0, 0.0]]),
        "scores": torch.tensor([0.9, 0.01]),
        "labels": torch.tensor([2, 5]),
        "batch": torch.tensor([0, 1]),
    }
    step = Mock(return_value={"output": output, "loss": torch.tensor(1.0)})
    decode = Mock(return_value=decoded)
    monkeypatch.setattr(module, "step", step)
    monkeypatch.setattr(module.model, "decode", decode)
    batch = {
        "pos": torch.zeros(4, 3),
        "batch": torch.tensor([0, 0, 1, 1]),
        "box": torch.tensor([[0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.0, 3.0]]),
        "batch_box": torch.tensor([0]),
    }
    out = module.validation_step(batch, batch_idx=0)
    step.assert_called_once_with(batch, "val")
    assert decode.call_args.args[0] is output
    # the 0.01-score box is below the default 0.05 threshold (dropped); the 0.9 box survives NMS.
    assert out["preds"]["labels"].tolist() == [2]
    assert out["preds"]["boxes"].shape[0] == 1
    assert out["target"]["labels"].tolist() == [3]
    assert torch.allclose(out["target"]["boxes"][0], torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]))


def test_detection_score_threshold_kwarg_filters(monkeypatch: pytest.MonkeyPatch) -> None:
    """The `score_threshold` / `nms_iou` __init__ kwargs control the inlined eval postprocess."""
    module = LitDetectionModel(
        name="dummy.detection",
        optimizer=partial(torch.optim.AdamW, lr=0.01),
        criterion=Mock(),
        score_threshold=0.5,
        nms_iou=0.1,
    )
    assert module.score_threshold == 0.5 and module.nms_iou == 0.1
    output = {"objectness_scores": torch.randn(2, 10)}
    decoded = {
        "boxes": torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]]),
        "scores": torch.tensor([0.3]),
        "labels": torch.tensor([1]),
        "batch": torch.tensor([0]),
    }
    monkeypatch.setattr(module, "step", Mock(return_value={"output": output, "loss": torch.tensor(1.0)}))
    monkeypatch.setattr(module.model, "decode", Mock(return_value=decoded))
    batch = {
        "pos": torch.zeros(2, 3),
        "batch": torch.tensor([0, 0]),
        "box": torch.tensor([[0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.0, 1.0]]),
        "batch_box": torch.tensor([0]),
    }
    out = module.validation_step(batch, batch_idx=0)
    # the 0.3-score box is below score_threshold=0.5, so nothing is kept.
    assert out["preds"]["boxes"].shape[0] == 0


def test_detection_configure_optimizers_without_scheduler_returns_optimizer() -> None:
    module = _make_det_module(scheduler=None)
    out = module.configure_optimizers()
    assert isinstance(out, torch.optim.Optimizer)


def test_detection_configure_optimizers_with_scheduler_returns_dict() -> None:
    module = _make_det_module(scheduler=partial(torch.optim.lr_scheduler.StepLR, step_size=1))
    out = module.configure_optimizers()
    assert set(out) == {"optimizer", "lr_scheduler"}
    assert out["lr_scheduler"]["interval"] == "epoch"
