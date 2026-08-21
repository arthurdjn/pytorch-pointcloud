from typing import Any, Callable, Dict, Optional
from unittest.mock import Mock

import pytest
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from torch_pointcloud.utils.imports import _LIGHTNING_AVAILABLE

pytestmark = pytest.mark.skipif(not _LIGHTNING_AVAILABLE, reason="lightning is not installed")


class DummySegmentationDataset(Dataset):
    def __init__(self, n: int = 4, value: float = 0.0) -> None:
        self._n = n
        self._value = value
        self.transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        return {
            "x": torch.full((8, 3), self._value),
            "pos": torch.randn(8, 3),
            "segment": torch.randint(0, 4, (8,)),
        }


def test_datamodule_returns_packed_batches() -> None:
    from torch_pointcloud.lightning import PointCloudDataModule

    dm = PointCloudDataModule(
        train_dataset=DummySegmentationDataset(4),
        val_dataset=DummySegmentationDataset(2),
        batch_size=2,
        num_workers=0,
    )
    batch = next(iter(dm.train_dataloader()))
    # 2 clouds of 8 points each collated into a packed batch of N=16.
    assert batch["x"].shape == (16, 3)
    assert batch["pos"].shape == (16, 3)
    assert batch["segment"].shape == (16,)
    assert batch["batch"].shape == (16,)
    assert int(batch["batch"].max().item()) == 1


def test_datamodule_eval_batch_size_applies_to_val_and_test_only() -> None:
    from torch_pointcloud.lightning import PointCloudDataModule

    dm = PointCloudDataModule(
        train_dataset=DummySegmentationDataset(4),
        val_dataset=DummySegmentationDataset(4),
        batch_size=2,
        eval_batch_size=1,
        num_workers=0,
    )
    val_loader = dm.val_dataloader()
    assert isinstance(val_loader, DataLoader)
    assert next(iter(dm.train_dataloader()))["x"].shape == (16, 3)
    assert next(iter(val_loader))["x"].shape == (8, 3)
    assert next(iter(dm.test_dataloader()))["x"].shape == (8, 3)


def test_datamodule_eval_batch_size_defaults_to_batch_size() -> None:
    from torch_pointcloud.lightning import PointCloudDataModule

    dm = PointCloudDataModule(val_dataset=DummySegmentationDataset(4), batch_size=2, num_workers=0)
    val_loader = dm.val_dataloader()
    assert isinstance(val_loader, DataLoader)
    assert next(iter(val_loader))["x"].shape == (16, 3)


def test_repeat_dataset_lengthens_epoch() -> None:
    """`RepeatDataset(dataset, loop=k)` lengthens an epoch by k. The datamodule
    treats it as any other dataset; the loop concept is owned by the dataset wrapper."""
    from torch_pointcloud.datasets import RepeatDataset
    from torch_pointcloud.lightning import PointCloudDataModule

    base = DummySegmentationDataset(3)
    dm = PointCloudDataModule(train_dataset=RepeatDataset(base, loop=4), batch_size=1, num_workers=0)
    loader = dm.train_dataloader()
    # 3 samples * 4 loops = 12, divided into batches of 1, with drop_last=True.
    assert len(loader) == 12


def test_val_and_test_dataloaders() -> None:
    from torch_pointcloud.lightning import PointCloudDataModule

    dm = PointCloudDataModule(
        val_dataset=DummySegmentationDataset(3),
        test_dataset=DummySegmentationDataset(2),
        batch_size=1,
        num_workers=0,
    )
    assert len(dm.val_dataloader()) == 3
    assert len(dm.test_dataloader()) == 2


def test_val_dataloader_without_val_dataset_returns_empty_list() -> None:
    """An empty list tells Lightning to skip validation, so a train-only experiment fits cleanly."""
    from torch_pointcloud.lightning import PointCloudDataModule

    dm = PointCloudDataModule(train_dataset=DummySegmentationDataset(4), batch_size=1, num_workers=0)
    assert dm.val_dataloader() == []


def test_missing_dataset_raises() -> None:
    from torch_pointcloud.lightning import PointCloudDataModule

    dm = PointCloudDataModule(batch_size=1, num_workers=0)
    with pytest.raises(ValueError, match="not provided"):
        dm.train_dataloader()


def test_setup_applies_model_transform_to_datasets_without_one() -> None:
    """`setup` copies the LightningModule's registered eval transform onto datasets whose `transform`
    is unset; an explicitly set dataset transform is kept."""
    from torch_pointcloud.lightning import PointCloudDataModule

    transform = Mock()
    existing = Mock()
    train = DummySegmentationDataset(2)
    val = DummySegmentationDataset(2)
    val.transform = existing
    dm = PointCloudDataModule(train_dataset=train, val_dataset=val, batch_size=1, num_workers=0)
    dm.trainer = Mock(lightning_module=Mock(transform=transform))
    dm.setup("fit")
    assert train.transform is transform
    assert val.transform is existing


def test_setup_leaves_wrapper_alone_when_wrapped_dataset_has_a_transform() -> None:
    """A `MixDataset` over a dataset that already carries the recipe keeps its own `transform` unset, so the
    registered transform is not applied a second time on the mixed output."""
    from torch_pointcloud.datasets import MixDataset
    from torch_pointcloud.lightning import PointCloudDataModule

    inner = DummySegmentationDataset(2)
    inner.transform = Mock()
    mixed = MixDataset(inner, mix=Mock())
    bare = MixDataset(DummySegmentationDataset(2), mix=Mock())
    dm = PointCloudDataModule(train_dataset=mixed, val_dataset=bare, batch_size=1, num_workers=0)
    transform = Mock()
    dm.trainer = Mock(lightning_module=Mock(transform=transform))
    dm.setup("fit")
    assert mixed.transform is None
    assert bare.transform is transform


def test_setup_without_model_transform_is_noop() -> None:
    from torch_pointcloud.lightning import PointCloudDataModule

    train = DummySegmentationDataset(2)
    dm = PointCloudDataModule(train_dataset=train, batch_size=1, num_workers=0)
    dm.setup("fit")
    assert getattr(train, "transform", None) is None


def test_train_ratios_without_concat_sizes_raises() -> None:
    from torch_pointcloud.lightning import PointCloudDataModule

    dm = PointCloudDataModule(train_dataset=DummySegmentationDataset(4), train_ratios=(1,), batch_size=2)
    with pytest.raises(ValueError, match="ConcatDataset"):
        dm.train_dataloader()


def test_train_ratios_yields_single_dataset_batches() -> None:
    """With `train_ratios`, every train batch is drawn from a single child dataset of the concat."""
    from torch_pointcloud.datasets import ConcatDataset
    from torch_pointcloud.lightning import PointCloudDataModule

    dataset = ConcatDataset([DummySegmentationDataset(4, value=0.0), DummySegmentationDataset(4, value=1.0)])
    dm = PointCloudDataModule(train_dataset=dataset, train_ratios=(1, 1), batch_size=2, num_workers=0)
    batches = list(dm.train_dataloader())
    assert len(batches) == 4
    for batch in batches:
        assert batch["x"].unique().numel() == 1


def test_setup_threads_transform_through_repeat_and_concat_wrappers() -> None:
    """`RepeatDataset` / `ConcatDataset` apply no transform themselves, so the registered transform must
    land on the wrapped datasets that actually run it."""
    from torch_pointcloud.datasets import ConcatDataset, RepeatDataset
    from torch_pointcloud.lightning import PointCloudDataModule

    inner = DummySegmentationDataset(2)
    first, second = DummySegmentationDataset(2), DummySegmentationDataset(2)
    dm = PointCloudDataModule(
        train_dataset=RepeatDataset(inner, loop=2),
        val_dataset=ConcatDataset([first, second]),
        batch_size=1,
        num_workers=0,
    )
    transform = Mock()
    dm.trainer = Mock(lightning_module=Mock(transform=transform))
    dm.setup("fit")
    assert inner.transform is transform
    assert first.transform is transform
    assert second.transform is transform
