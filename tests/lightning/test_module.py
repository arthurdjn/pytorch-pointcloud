from functools import partial
from typing import Any, Dict, Iterator, Tuple
from unittest.mock import Mock

import pytest
import torch
from torch import Tensor, nn
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import Dataset
from torch_geometric.utils import scatter

from torch_pointcloud.inferers import SimpleInferer
from torch_pointcloud.lightning import (
    LitClassificationModel,
    LitDetectionModel,
    LitSegmentationModel,
    PointCloudDataModule,
)
from torch_pointcloud.lightning.metrics import AveragePrecision3D
from torch_pointcloud.models import ClassificationModel, DetectionModel, SegmentationModel, register_model
from torch_pointcloud.models._registry import _REGISTERED_MODELS, Task
from torch_pointcloud.utils.box3d import projected_ignore_mask
from torch_pointcloud.utils.metrics import average_precision3d
from torch_pointcloud.utils.types import Boxes3D, Detection3D

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


def test_seg_target_key_defaults_to_segment() -> None:
    lit = LitSegmentationModel(name="dummy.segmentation")
    batch = {
        "x": torch.randn(12, 3),
        "pos": torch.randn(12, 3),
        "batch": torch.cat([torch.zeros(6, dtype=torch.long), torch.ones(6, dtype=torch.long)]),
        "segment": torch.randint(0, 4, (12,)),
    }
    assert lit.hparams["target_key"] == "segment"
    out = lit.step(batch, "val")
    assert torch.equal(out["target"], batch["segment"])


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


@pytest.mark.parametrize(
    ("stage", "expected"),
    [
        pytest.param("train", False, id="train"),
        pytest.param("val", True, id="val"),
        pytest.param("test", True, id="test"),
    ],
)
def test_step_syncs_logged_loss_on_eval_stages_only(
    stage: str, expected: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    lit = _make_seg_module()
    log = Mock()
    monkeypatch.setattr(lit, "log", log)
    batch = {
        "x": torch.randn(12, 3),
        "pos": torch.randn(12, 3),
        "batch": torch.cat([torch.zeros(6, dtype=torch.long), torch.ones(6, dtype=torch.long)]),
        "segment": torch.randint(0, 4, (12,)),
    }
    lit.step(batch, stage)
    assert log.call_args.kwargs["sync_dist"] is expected


def test_forward_missing_input_key_raises() -> None:
    lit = LitSegmentationModel(name="dummy.segmentation", target_key="segment", input_keys=("x", "pos", "wrong"))
    batch = {
        "x": torch.randn(6, 3),
        "pos": torch.randn(6, 3),
        "batch": torch.zeros(6, dtype=torch.long),
        "segment": torch.randint(0, 4, (6,)),
    }
    with pytest.raises(KeyError, match="wrong"):
        lit(batch)


def test_forward_missing_x_resolves_to_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """`x` is the only optional input key: a batch without point features passes `x=None` to the model."""
    lit = _make_cls_module()
    forward = Mock(return_value=torch.randn(2, 5))
    monkeypatch.setattr(lit.model, "forward", forward)
    batch = {
        "pos": torch.randn(12, 3),
        "batch": torch.cat([torch.zeros(6, dtype=torch.long), torch.ones(6, dtype=torch.long)]),
    }
    lit(batch)
    args = forward.call_args.args
    assert args[0] is None
    assert args[1] is batch["pos"]
    assert args[2] is batch["batch"]


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


def test_configure_optimizers_without_optimizer_raises() -> None:
    """Without an `optimizer` the module is evaluation-only, so `Trainer.fit` must fail loudly."""
    lit = LitClassificationModel(name="dummy.classification")
    with pytest.raises(RuntimeError, match="evaluation-only"):
        lit.configure_optimizers()


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
    is set explicitly (the LitModule does not auto-inject it)."""
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


def test_fit_smoke_train_only() -> None:
    """A datamodule without a `val_dataset` fits train-only: its `val_dataloader` returns an empty
    list, which tells Lightning to skip validation."""
    lit = _make_seg_module()
    dm = PointCloudDataModule(train_dataset=DummySegmentationDataset(4), batch_size=2, num_workers=0)
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
    assert "val/loss" not in trainer.callback_metrics


def test_seg_eval_params_saved_to_hparams() -> None:
    lit = LitSegmentationModel(name="dummy.segmentation", target_key="segment", inverse_key="inverse")
    assert lit.hparams_initial["inverse_key"] == "inverse"
    assert lit.hparams_initial["origin_target_key"] == "origin_segment"


def test_detection_eval_params_saved_to_hparams() -> None:
    module = LitDetectionModel(name="dummy.detection", score_threshold=0.5, nms_iou=0.1, min_points=5)
    assert module.hparams_initial["score_threshold"] == 0.5
    assert module.hparams_initial["nms_iou"] == 0.1
    assert module.hparams_initial["min_points"] == 5
    assert module.hparams_initial["label_key"] == "label"
    assert module.hparams_initial["ignore_mask_key"] is None


def test_seg_inferer_runs_on_test_step() -> None:
    """`test_step` delegates the forward to the inferer, passing the batch and the module's forward."""
    preds = torch.randn(12, 4)
    inferer = Mock(return_value=preds)
    lit = LitSegmentationModel(name="dummy.segmentation", target_key="segment", inferer=inferer)
    batch = {
        "x": torch.randn(12, 3),
        "pos": torch.randn(12, 3),
        "batch": torch.cat([torch.zeros(6, dtype=torch.long), torch.ones(6, dtype=torch.long)]),
        "segment": torch.randint(0, 4, (12,)),
    }
    out = lit.test_step(batch, batch_idx=0)
    inferer.assert_called_once()
    assert inferer.call_args.args[0] is batch
    assert inferer.call_args.kwargs["predictor"] == lit.predict
    assert out["preds"] is preds
    assert torch.equal(out["target"], batch["segment"])


def test_inferer_defaults_to_simple() -> None:
    """Every module runs its test predictions through an inferer; the default is a plain forward."""
    assert isinstance(_make_cls_module().inferer, SimpleInferer)
    assert isinstance(_make_seg_module().inferer, SimpleInferer)


def test_cls_inferer_runs_on_test_step() -> None:
    preds = torch.randn(2, 4)
    inferer = Mock(return_value=preds)
    lit = LitClassificationModel(name="dummy.classification", inferer=inferer)
    batch = {
        "x": torch.randn(12, 3),
        "pos": torch.randn(12, 3),
        "batch": torch.cat([torch.zeros(6, dtype=torch.long), torch.ones(6, dtype=torch.long)]),
        "label": torch.tensor([0, 3]),
    }
    out = lit.test_step(batch, batch_idx=0)
    inferer.assert_called_once()
    assert inferer.call_args.args[0] is batch
    assert inferer.call_args.kwargs["predictor"] == lit.predict
    assert out["preds"] is preds
    assert torch.equal(out["target"], batch["label"])
    lit.validation_step(batch, batch_idx=0)
    inferer.assert_called_once()


def test_seg_eval_step_returns_per_point_batch() -> None:
    """Eval steps expose the packed per-point shape index so multi-shape metrics can score per shape."""
    lit = _make_seg_module()
    batch = {
        "x": torch.randn(12, 3),
        "pos": torch.randn(12, 3),
        "batch": torch.cat([torch.zeros(6, dtype=torch.long), torch.ones(6, dtype=torch.long)]),
        "segment": torch.randint(0, 4, (12,)),
    }
    out = lit.validation_step(batch, batch_idx=0)
    assert out["batch"] is batch["batch"]


def test_metric_input_keys_default_leaves_eval_output_unchanged() -> None:
    lit = _make_cls_module()
    batch = {
        "x": torch.randn(12, 3),
        "pos": torch.randn(12, 3),
        "batch": torch.cat([torch.zeros(6, dtype=torch.long), torch.ones(6, dtype=torch.long)]),
        "label": torch.tensor([0, 3]),
    }
    assert lit.hparams["metric_input_keys"] == []
    out = lit.validation_step(batch, batch_idx=0)
    assert set(out) == {"preds", "target"}


def test_metric_input_keys_saved_to_hparams() -> None:
    lit = LitClassificationModel(name="dummy.classification", metric_input_keys=("velocity",))
    assert lit.hparams_initial["metric_input_keys"] == ["velocity"]


@pytest.mark.parametrize(
    "step",
    [pytest.param("validation_step", id="val"), pytest.param("test_step", id="test")],
)
def test_metric_input_keys_passthrough_on_eval_steps(step: str) -> None:
    """Listed batch keys are copied as-is into the eval step output for `MetricCallback` to forward."""
    lit = LitClassificationModel(name="dummy.classification", metric_input_keys=("velocity", "num_points"))
    batch = {
        "x": torch.randn(12, 3),
        "pos": torch.randn(12, 3),
        "batch": torch.cat([torch.zeros(6, dtype=torch.long), torch.ones(6, dtype=torch.long)]),
        "label": torch.tensor([0, 3]),
        "velocity": torch.randn(2, 2),
        "num_points": torch.tensor([6, 6]),
    }
    out = getattr(lit, step)(batch, batch_idx=0)
    assert set(out) == {"preds", "target", "velocity", "num_points"}
    assert out["velocity"] is batch["velocity"]
    assert out["num_points"] is batch["num_points"]


def test_metric_input_keys_missing_from_batch_raises() -> None:
    lit = LitClassificationModel(name="dummy.classification", metric_input_keys=("velocity",))
    batch = {
        "x": torch.randn(12, 3),
        "pos": torch.randn(12, 3),
        "batch": torch.cat([torch.zeros(6, dtype=torch.long), torch.ones(6, dtype=torch.long)]),
        "label": torch.tensor([0, 3]),
    }
    with pytest.raises(KeyError, match="metric_input_keys"):
        lit.validation_step(batch, batch_idx=0)


def test_seg_metric_input_keys_extend_step_output() -> None:
    """The passthrough adds to the seg eval output without touching its `preds` / `target` / `batch`."""
    lit = LitSegmentationModel(name="dummy.segmentation", target_key="segment", metric_input_keys=("pos",))
    batch = {
        "x": torch.randn(12, 3),
        "pos": torch.randn(12, 3),
        "batch": torch.cat([torch.zeros(6, dtype=torch.long), torch.ones(6, dtype=torch.long)]),
        "segment": torch.randint(0, 4, (12,)),
    }
    out = lit.validation_step(batch, batch_idx=0)
    assert set(out) == {"preds", "target", "batch", "pos"}
    assert out["pos"] is batch["pos"]
    assert out["batch"] is batch["batch"]


def test_seg_inferer_not_used_on_validation_step() -> None:
    inferer = Mock(return_value=torch.randn(12, 4))
    lit = LitSegmentationModel(name="dummy.segmentation", target_key="segment", inferer=inferer)
    batch = {
        "x": torch.randn(12, 3),
        "pos": torch.randn(12, 3),
        "batch": torch.cat([torch.zeros(6, dtype=torch.long), torch.ones(6, dtype=torch.long)]),
        "segment": torch.randint(0, 4, (12,)),
    }
    out = lit.validation_step(batch, batch_idx=0)
    inferer.assert_not_called()
    assert out["preds"].shape == (12, 4)


def test_seg_inverse_key_broadcasts_preds_to_raw_resolution() -> None:
    """With `inverse_key`, eval preds are gathered to raw resolution and scored against `origin_segment`."""
    lit = LitSegmentationModel(name="dummy.segmentation", target_key="segment", inverse_key="inverse")
    batch = {
        "x": torch.randn(6, 3),
        "pos": torch.randn(6, 3),
        "batch": torch.zeros(6, dtype=torch.long),
        "segment": torch.randint(0, 4, (6,)),
        "inverse": torch.randint(0, 6, (12,)),
        "origin_segment": torch.randint(0, 4, (12,)),
    }
    logits = lit(batch)
    out = lit.test_step(batch, batch_idx=0)
    assert out["preds"].shape == (12, 4)
    assert torch.equal(out["preds"], logits[batch["inverse"]])
    assert torch.equal(out["target"], batch["origin_segment"])


def test_seg_inverse_key_missing_from_batch_raises() -> None:
    lit = LitSegmentationModel(name="dummy.segmentation", target_key="segment", inverse_key="inverse")
    batch = {
        "x": torch.randn(6, 3),
        "pos": torch.randn(6, 3),
        "batch": torch.zeros(6, dtype=torch.long),
        "segment": torch.randint(0, 4, (6,)),
    }
    with pytest.raises(KeyError):
        lit.test_step(batch, batch_idx=0)


def test_seg_inverse_key_offsets_per_scene_with_batch_index() -> None:
    """Multi-scene batches offset each scene's inverse map by the voxel rows of the scenes before it."""
    lit = LitSegmentationModel(name="dummy.segmentation", target_key="segment", inverse_key="inverse")
    batch = {
        "x": torch.randn(5, 3),
        "pos": torch.randn(5, 3),
        "batch": torch.tensor([0, 0, 1, 1, 1]),
        "segment": torch.randint(0, 4, (5,)),
        "inverse": torch.tensor([0, 1, 1, 0, 2, 2, 1]),
        "batch_inverse": torch.tensor([0, 0, 0, 1, 1, 1, 1]),
        "origin_segment": torch.randint(0, 4, (7,)),
    }
    logits = lit(batch)
    out = lit.test_step(batch, batch_idx=0)
    expected_rows = torch.tensor([0, 1, 1, 2, 4, 4, 3])
    assert torch.equal(out["preds"], logits[expected_rows])
    assert torch.equal(out["target"], batch["origin_segment"])
    assert torch.equal(out["batch"], batch["batch"][expected_rows])


def test_seg_inverse_key_multi_scene_without_batch_index_raises() -> None:
    lit = LitSegmentationModel(name="dummy.segmentation", target_key="segment", inverse_key="inverse")
    batch = {
        "x": torch.randn(5, 3),
        "pos": torch.randn(5, 3),
        "batch": torch.tensor([0, 0, 1, 1, 1]),
        "segment": torch.randint(0, 4, (5,)),
        "inverse": torch.tensor([0, 1, 1, 0, 2, 2, 1]),
        "origin_segment": torch.randint(0, 4, (7,)),
    }
    with pytest.raises(RuntimeError, match="cat_keys"):
        lit.test_step(batch, batch_idx=0)


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


def test_detection_criterion_prebuilt_module_used_as_is() -> None:
    """A ready-built `nn.Module` criterion is used as-is, not called as a head-geometry factory."""
    criterion = nn.Identity()
    lit = LitDetectionModel(name="dummy.detection", optimizer=partial(torch.optim.AdamW, lr=0.01), criterion=criterion)
    assert lit.criterion is criterion


def test_detection_criterion_none_skips_head_geometry_completion() -> None:
    """Without a `criterion` the head-geometry params are never read, so a model may not expose them."""

    def factory(**kwargs: Any) -> DummyDetectionModel:
        model = DummyDetectionModel(**kwargs)
        del model.num_heading_bin
        del model.num_size_cluster
        del model.mean_sizes
        return model

    register_model("dummy-headless.detection", task="detection")(factory)
    try:
        module = LitDetectionModel(name="dummy-headless.detection")
        assert module.criterion is None
    finally:
        _REGISTERED_MODELS["detection"].pop("dummy-headless.detection", None)


def test_detection_criterion_none_eval_step_returns_preds_without_loss(monkeypatch: pytest.MonkeyPatch) -> None:
    module = LitDetectionModel(name="dummy.detection")
    decoded = {
        "boxes": torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]]),
        "scores": torch.tensor([0.9]),
        "labels": torch.tensor([2]),
        "batch": torch.tensor([0]),
    }
    monkeypatch.setattr(module.model, "decode", Mock(return_value=decoded))
    log = Mock()
    monkeypatch.setattr(module, "log", log)
    batch = {
        "x": torch.rand(4, 1),
        "pos": torch.zeros(4, 3),
        "batch": torch.zeros(4, dtype=torch.long),
        "box": torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]]),
        "batch_box": torch.tensor([0]),
        "label": torch.tensor([3]),
    }
    out = module.test_step(batch, batch_idx=0)
    log.assert_not_called()
    assert out["preds"]["labels"].tolist() == [2]
    assert out["target"]["labels"].tolist() == [3]


def test_detection_criterion_none_training_step_raises() -> None:
    module = LitDetectionModel(name="dummy.detection")
    batch = {
        "x": torch.rand(4, 1),
        "pos": torch.rand(4, 3),
        "batch": torch.zeros(4, dtype=torch.long),
    }
    with pytest.raises(RuntimeError, match="evaluation-only"):
        module.training_step(batch, batch_idx=0)


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
    forward = Mock(return_value=output)
    decode = Mock(return_value=decoded)
    monkeypatch.setattr(module, "forward", forward)
    monkeypatch.setattr(module.model, "decode", decode)
    batch = {
        "pos": torch.zeros(4, 3),
        "batch": torch.tensor([0, 0, 1, 1]),
        "box": torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]]),
        "batch_box": torch.tensor([0]),
        "label": torch.tensor([3]),
    }
    out = module.validation_step(batch, batch_idx=0)
    forward.assert_called_once_with(batch)
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
    monkeypatch.setattr(module, "forward", Mock(return_value=output))
    monkeypatch.setattr(module.model, "decode", Mock(return_value=decoded))
    batch = {
        "pos": torch.zeros(2, 3),
        "batch": torch.tensor([0, 0]),
        "box": torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]]),
        "batch_box": torch.tensor([0]),
        "label": torch.tensor([1]),
    }
    out = module.validation_step(batch, batch_idx=0)
    # the 0.3-score box is below score_threshold=0.5, so nothing is kept.
    assert out["preds"]["boxes"].shape[0] == 0


def test_detection_min_points_filters_boxes_without_points(monkeypatch: pytest.MonkeyPatch) -> None:
    module = LitDetectionModel(
        name="dummy.detection",
        optimizer=partial(torch.optim.AdamW, lr=0.01),
        criterion=Mock(),
        min_points=2,
    )
    decoded = {
        "boxes": torch.tensor([[0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0], [9.0, 9.0, 9.0, 1.0, 1.0, 1.0, 0.0]]),
        "scores": torch.tensor([0.9, 0.8]),
        "labels": torch.tensor([1, 2]),
        "batch": torch.tensor([0, 0]),
    }
    monkeypatch.setattr(module, "forward", Mock(return_value={}))
    monkeypatch.setattr(module.model, "decode", Mock(return_value=decoded))
    batch = {
        "pos": torch.zeros(4, 3),
        "batch": torch.zeros(4, dtype=torch.long),
        "box": torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]]),
        "batch_box": torch.tensor([0]),
        "label": torch.tensor([1]),
    }
    out = module.validation_step(batch, batch_idx=0)
    # all 4 points fall in the first box; the far-away box holds 0 < min_points=2 and is dropped.
    assert out["preds"]["labels"].tolist() == [1]
    assert out["preds"]["boxes"].shape[0] == 1


def test_detection_class_probs_expands_boxes_per_class(monkeypatch: pytest.MonkeyPatch) -> None:
    """When `decode` emits `class_probs`, each surviving box is scored once per class (indoor AP protocol)."""
    module = _make_det_module()
    probs = torch.tensor([[0.5, 0.3, 0.2], [0.1, 0.6, 0.3]])
    decoded = {
        "boxes": torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0], [5.0, 5.0, 5.0, 1.0, 1.0, 1.0, 0.0]]),
        "scores": torch.tensor([0.9, 0.8]),
        "labels": torch.tensor([0, 1]),
        "batch": torch.tensor([0, 1]),
        "class_probs": probs,
    }
    monkeypatch.setattr(module, "forward", Mock(return_value={}))
    monkeypatch.setattr(module.model, "decode", Mock(return_value=decoded))
    batch = {
        "pos": torch.zeros(4, 3),
        "batch": torch.tensor([0, 0, 1, 1]),
        "box": torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]]),
        "batch_box": torch.tensor([0]),
        "label": torch.tensor([1]),
    }
    out = module.validation_step(batch, batch_idx=0)
    assert out["preds"]["boxes"].shape == (6, 7)
    assert torch.equal(out["preds"]["boxes"][:3], decoded["boxes"][0].expand(3, 7))
    assert torch.allclose(out["preds"]["scores"], (probs * torch.tensor([[0.9], [0.8]])).reshape(-1))
    assert out["preds"]["labels"].tolist() == [0, 1, 2, 0, 1, 2]
    assert out["preds"]["batch"].tolist() == [0, 0, 0, 1, 1, 1]


def test_detection_label_key_passes_full_extent_target_through(monkeypatch: pytest.MonkeyPatch) -> None:
    """`label_key` overrides the per-box class key; the $(K, 7)$ boxes and ignore mask pass through unmodified."""
    module = LitDetectionModel(
        name="dummy.detection",
        optimizer=partial(torch.optim.AdamW, lr=0.01),
        criterion=Mock(),
        label_key="label",
        ignore_mask_key="ignore_mask",
    )
    decoded = {
        "boxes": torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]]),
        "scores": torch.tensor([0.9]),
        "labels": torch.tensor([0]),
        "batch": torch.tensor([0]),
    }
    monkeypatch.setattr(module, "forward", Mock(return_value={}))
    monkeypatch.setattr(module.model, "decode", Mock(return_value=decoded))
    batch = {
        "pos": torch.zeros(2, 3),
        "batch": torch.zeros(2, dtype=torch.long),
        "box": torch.tensor([[1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 0.3]]),
        "batch_box": torch.tensor([0]),
        "label": torch.tensor([2]),
        "ignore_mask": torch.tensor([False]),
    }
    out = module.validation_step(batch, batch_idx=0)
    assert out["target"]["boxes"] is batch["box"]
    assert out["target"]["labels"].tolist() == [2]
    assert out["target"]["batch"] is batch["batch_box"]
    assert out["target"]["ignore_mask"] is batch["ignore_mask"]


def test_detection_metric_input_keys_passthrough(monkeypatch: pytest.MonkeyPatch) -> None:
    module = LitDetectionModel(name="dummy.detection", metric_input_keys=("calib",))
    decoded = {
        "boxes": torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]]),
        "scores": torch.tensor([0.9]),
        "labels": torch.tensor([2]),
        "batch": torch.tensor([0]),
    }
    monkeypatch.setattr(module.model, "decode", Mock(return_value=decoded))
    batch = {
        "x": torch.rand(4, 1),
        "pos": torch.zeros(4, 3),
        "batch": torch.zeros(4, dtype=torch.long),
        "box": torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]]),
        "batch_box": torch.tensor([0]),
        "label": torch.tensor([3]),
        "calib": torch.rand(1, 3, 4),
    }
    out = module.test_step(batch, batch_idx=0)
    assert set(out) == {"preds", "target", "calib"}
    assert out["calib"] is batch["calib"]


def test_detection_configure_optimizers_without_scheduler_returns_optimizer() -> None:
    module = _make_det_module(scheduler=None)
    out = module.configure_optimizers()
    assert isinstance(out, torch.optim.Optimizer)


def test_detection_configure_optimizers_with_scheduler_returns_dict() -> None:
    module = _make_det_module(scheduler=partial(torch.optim.lr_scheduler.StepLR, step_size=1))
    out = module.configure_optimizers()
    assert set(out) == {"optimizer", "lr_scheduler"}
    assert out["lr_scheduler"]["interval"] == "epoch"


def test_detection_eval_step_passes_decode_velocity_through(monkeypatch: pytest.MonkeyPatch) -> None:
    """Extra per-box `decode` entries (here `velocity`) follow the boxes through the score filter and
    NMS reordering into the predictions dict; without calib keys no `ignore_mask` is attached."""
    module = LitDetectionModel(name="dummy.detection")
    velocity = torch.tensor([[1.5, 0.5], [3.0, 0.0]])
    decoded = {
        "boxes": torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0], [5.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]]),
        "scores": torch.tensor([0.5, 0.9]),
        "labels": torch.tensor([1, 1]),
        "batch": torch.tensor([0, 0]),
        "velocity": velocity,
    }
    monkeypatch.setattr(module, "forward", Mock(return_value={}))
    monkeypatch.setattr(module.model, "decode", Mock(return_value=decoded))
    batch = {
        "pos": torch.zeros(2, 3),
        "batch": torch.zeros(2, dtype=torch.long),
        "box": torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]]),
        "batch_box": torch.tensor([0]),
        "label": torch.tensor([1]),
    }
    out = module.validation_step(batch, batch_idx=0)
    # NMS orders by descending score, so the second decoded box comes first; velocity follows.
    assert torch.equal(out["preds"]["boxes"], decoded["boxes"][[1, 0]])
    assert torch.equal(out["preds"]["velocity"], velocity[[1, 0]])
    assert "ignore_mask" not in out["preds"]


def test_detection_eval_ignore_mask_built_from_calib_per_scene(monkeypatch: pytest.MonkeyPatch) -> None:
    """With `calib` / `image_shape` in the batch, the eval step attaches `projected_ignore_mask` built
    from each box's own frame (the stacked calib indexed by the boxes' scene index)."""
    module = LitDetectionModel(name="dummy.detection")
    # Same box in both scenes; scene 0's focal projects it under 25 px, scene 1's five-fold focal does not.
    decoded = {
        "boxes": torch.tensor([[10.0, 0.0, 0.0, 2.0, 2.0, 1.0, 0.0], [10.0, 0.0, 0.0, 2.0, 2.0, 1.0, 0.0]]),
        "scores": torch.tensor([0.9, 0.9]),
        "labels": torch.tensor([0, 0]),
        "batch": torch.tensor([0, 1]),
    }
    monkeypatch.setattr(module, "forward", Mock(return_value={}))
    monkeypatch.setattr(module.model, "decode", Mock(return_value=decoded))
    calib = torch.stack(
        [
            torch.tensor([[50.0, -100.0, 0.0, 0.0], [50.0, 0.0, -100.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
            torch.tensor([[50.0, -100.0, 0.0, 0.0], [50.0, 0.0, -500.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
        ]
    )
    image_shape = torch.tensor([[100, 200], [100, 200]])
    batch = {
        "pos": torch.zeros(4, 3),
        "batch": torch.tensor([0, 0, 1, 1]),
        "box": torch.tensor([[10.0, 0.0, 0.0, 2.0, 2.0, 1.0, 0.0], [10.0, 0.0, 0.0, 2.0, 2.0, 1.0, 0.0]]),
        "batch_box": torch.tensor([0, 1]),
        "label": torch.tensor([0, 0]),
        "calib": calib,
        "image_shape": image_shape,
    }
    out = module.validation_step(batch, batch_idx=0)
    preds = out["preds"]
    assert preds["ignore_mask"].tolist() == [True, False]
    for scene in range(2):
        rows = preds["batch"] == scene
        expected = projected_ignore_mask(preds["boxes"][rows], calib[scene], image_shape[scene])
        assert torch.equal(preds["ignore_mask"][rows], expected)


def test_detection_eval_ignore_mask_flows_into_ap_as_functional(monkeypatch: pytest.MonkeyPatch) -> None:
    """The eval-step mask reaches the AP exactly as the functional path: the flagged far box (a false
    positive without the mask) is excluded from scoring, so the AP recovers to 1."""
    module = LitDetectionModel(name="dummy.detection")
    decoded = {
        "boxes": torch.tensor([[10.0, 0.0, 0.0, 2.0, 2.0, 1.0, 0.0], [2.0, 0.0, 0.0, 2.0, 2.0, 1.0, 0.0]]),
        "scores": torch.tensor([0.9, 0.8]),
        "labels": torch.tensor([0, 0]),
        "batch": torch.tensor([0, 0]),
    }
    monkeypatch.setattr(module, "forward", Mock(return_value={}))
    monkeypatch.setattr(module.model, "decode", Mock(return_value=decoded))
    calib = torch.tensor([[50.0, -100.0, 0.0, 0.0], [50.0, 0.0, -100.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    image_shape = torch.tensor([100, 200])
    batch = {
        "pos": torch.zeros(2, 3),
        "batch": torch.zeros(2, dtype=torch.long),
        "box": torch.tensor([[2.0, 0.0, 0.0, 2.0, 2.0, 1.0, 0.0]]),
        "batch_box": torch.tensor([0]),
        "label": torch.tensor([0]),
        "calib": calib[None],
        "image_shape": image_shape[None],
    }
    out = module.validation_step(batch, batch_idx=0)
    metric = AveragePrecision3D(iou_per_class={0: 0.5})
    metric.update(out["preds"], out["target"])

    unmasked: Detection3D = {
        "boxes": decoded["boxes"],
        "scores": decoded["scores"],
        "labels": decoded["labels"],
        "batch": decoded["batch"],
    }
    masked: Detection3D = {**unmasked, "ignore_mask": projected_ignore_mask(decoded["boxes"], calib, image_shape)}
    target: Boxes3D = {"boxes": batch["box"], "labels": batch["label"], "batch": batch["batch_box"]}
    assert metric.compute() == average_precision3d([masked], [target], iou_per_class={0: 0.5})
    assert metric.compute()["AP/0"] == pytest.approx(1.0)
    assert average_precision3d([unmasked], [target], iou_per_class={0: 0.5})["AP/0"] == pytest.approx(0.5)


def test_detection_nms_rotated_saved_to_hparams_default_false() -> None:
    module = LitDetectionModel(name="dummy.detection")
    assert module.hparams_initial["nms_rotated"] is False
    assert module.nms_rotated is False
    module = LitDetectionModel(name="dummy.detection", nms_rotated=True)
    assert module.hparams_initial["nms_rotated"] is True
    assert module.nms_rotated is True


def test_detection_nms_rotated_threaded_into_eval_nms(monkeypatch: pytest.MonkeyPatch) -> None:
    """`nms_rotated=True` switches the eval postprocess NMS to the rotated-BEV criterion: parallel
    45-degree neighbors with disjoint rotated footprints survive, while their overlapping AABBs are
    suppressed at the same threshold by default."""
    decoded = {
        "boxes": torch.tensor(
            [
                [0.0, 0.0, 0.0, 4.0, 0.5, 1.0, 0.7853981633974483],
                [-0.7071, 0.7071, 0.0, 4.0, 0.5, 1.0, 0.7853981633974483],
            ]
        ),
        "scores": torch.tensor([0.9, 0.8]),
        "labels": torch.tensor([1, 1]),
        "batch": torch.tensor([0, 0]),
    }
    batch = {
        "pos": torch.zeros(2, 3),
        "batch": torch.zeros(2, dtype=torch.long),
        "box": torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]]),
        "batch_box": torch.tensor([0]),
        "label": torch.tensor([1]),
    }
    for nms_rotated, expected_boxes in ((False, 1), (True, 2)):
        module = LitDetectionModel(name="dummy.detection", nms_iou=0.01, nms_rotated=nms_rotated)
        monkeypatch.setattr(module, "forward", Mock(return_value={}))
        monkeypatch.setattr(module.model, "decode", Mock(return_value=dict(decoded)))
        out = module.validation_step(batch, batch_idx=0)
        assert out["preds"]["boxes"].shape[0] == expected_boxes
