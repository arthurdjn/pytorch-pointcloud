from functools import partial
from typing import Any, Dict
from unittest.mock import Mock

import pytest
import torch
from torch import Tensor, nn
from torch.utils.data import Dataset

from torch_pointcloud.models import ClassificationModel, SegmentationModel
from torch_pointcloud.utils.imports import _LIGHTNING_AVAILABLE

pytestmark = pytest.mark.skipif(not _LIGHTNING_AVAILABLE, reason="lightning is not installed")


class _StubSegModel(SegmentationModel):
    """Per-point linear classifier — segmentation contract."""

    def __init__(self, in_channels: int = 3, num_classes: int = 4) -> None:
        super().__init__(in_channels=in_channels, num_classes=num_classes)
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x: Tensor, pos: Tensor, batch: Tensor) -> Tensor:
        return self.fc(x)


class _StubClsModel(ClassificationModel):
    """Per-cloud linear classifier — scatter-mean pool then classify."""

    def __init__(self, in_channels: int = 3, num_classes: int = 5) -> None:
        super().__init__(in_channels=in_channels, num_classes=num_classes)
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x: Tensor, pos: Tensor, batch: Tensor) -> Tensor:
        from torch_geometric.utils import scatter

        pooled = scatter(x, batch, dim=0, reduce="mean")
        return self.fc(pooled)


class _SegDataset(Dataset):
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


class _ClsDataset(Dataset):
    def __init__(self, n: int = 4) -> None:
        self._n = n

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        return {
            "x": torch.randn(6, 3),
            "pos": torch.randn(6, 3),
            "label": torch.tensor(index % 5),
        }


def _make_seg_module(*, scheduler: Any = None, param_groups: Any = None) -> Any:
    from torch_pointcloud.lightning import LitSegmentationModel

    return LitSegmentationModel(
        model=_StubSegModel(),
        optimizer=partial(torch.optim.AdamW, lr=0.01),
        scheduler=scheduler,
        scheduler_interval="step",
        param_groups=param_groups,
    )


def _make_cls_module() -> Any:
    from torch_pointcloud.lightning import LitClassificationModel

    return LitClassificationModel(
        model=_StubClsModel(),
        optimizer=partial(torch.optim.AdamW, lr=0.01),
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
    """The MONAI-style `param_groups` dict reaches `generate_param_groups`."""
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
    import lightning.pytorch as L
    from torch.optim.lr_scheduler import OneCycleLR

    from torch_pointcloud.lightning import PointCloudDataModule

    lit = _make_seg_module(scheduler=partial(OneCycleLR, max_lr=0.01, total_steps=10))
    dm = PointCloudDataModule(
        train_dataset=_SegDataset(4),
        val_dataset=_SegDataset(2),
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


def test_mix3d_halves_batch_index_during_training() -> None:
    """Mix3D with `mix_prob=1.0` always merges adjacent scene pairs."""
    from torch_pointcloud.lightning import LitSegmentationModel

    lit = LitSegmentationModel(
        model=_StubSegModel(),
        optimizer=partial(torch.optim.AdamW, lr=0.01),
        mix_prob=1.0,
    )
    assert lit.mix_prob == 1.0
    trainer = Mock()
    trainer.training = True
    lit._trainer = trainer
    batch = {"batch": torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])}
    out = lit.on_after_batch_transfer(batch, dataloader_idx=0)
    # 0,0,1,1,2,2,3,3 -> 0,0,0,0,1,1,1,1 after `// 2`.
    assert torch.equal(out["batch"], torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]))
