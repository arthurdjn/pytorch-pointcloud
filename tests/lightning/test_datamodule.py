from typing import Dict

import pytest
import torch
from torch import Tensor
from torch.utils.data import Dataset

from torch_pointcloud.utils.imports import _LIGHTNING_AVAILABLE

pytestmark = pytest.mark.skipif(not _LIGHTNING_AVAILABLE, reason="lightning is not installed")


class _StubDataset(Dataset):
    """Tiny dataset of `n` packed point-cloud samples."""

    def __init__(self, n: int = 4) -> None:
        self._n = n

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        return {
            "x": torch.randn(8, 3),
            "pos": torch.randn(8, 3),
            "segment": torch.randint(0, 4, (8,)),
        }


def test_datamodule_returns_packed_batches() -> None:
    from torch_pointcloud.lightning import PointCloudDataModule

    dm = PointCloudDataModule(train_dataset=_StubDataset(4), val_dataset=_StubDataset(2), batch_size=2, num_workers=0)
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
        train_dataset=_StubDataset(4),
        val_dataset=_StubDataset(4),
        batch_size=2,
        eval_batch_size=1,
        num_workers=0,
    )
    assert next(iter(dm.train_dataloader()))["x"].shape == (16, 3)
    assert next(iter(dm.val_dataloader()))["x"].shape == (8, 3)
    assert next(iter(dm.test_dataloader()))["x"].shape == (8, 3)


def test_datamodule_eval_batch_size_defaults_to_batch_size() -> None:
    from torch_pointcloud.lightning import PointCloudDataModule

    dm = PointCloudDataModule(val_dataset=_StubDataset(4), batch_size=2, num_workers=0)
    assert next(iter(dm.val_dataloader()))["x"].shape == (16, 3)


def test_repeat_dataset_lengthens_epoch() -> None:
    """`RepeatDataset(dataset, loop=k)` lengthens an epoch by k. The datamodule
    treats it as any other dataset; the loop concept is owned by the dataset wrapper."""
    from torch_pointcloud.datasets import RepeatDataset
    from torch_pointcloud.lightning import PointCloudDataModule

    base = _StubDataset(3)
    dm = PointCloudDataModule(train_dataset=RepeatDataset(base, loop=4), batch_size=1, num_workers=0)
    loader = dm.train_dataloader()
    # 3 samples * 4 loops = 12, divided into batches of 1, with drop_last=True.
    assert len(loader) == 12


def test_val_and_test_dataloaders() -> None:
    from torch_pointcloud.lightning import PointCloudDataModule

    dm = PointCloudDataModule(
        val_dataset=_StubDataset(3),
        test_dataset=_StubDataset(2),
        batch_size=1,
        num_workers=0,
    )
    assert len(dm.val_dataloader()) == 3
    assert len(dm.test_dataloader()) == 2


def test_missing_dataset_raises() -> None:
    from torch_pointcloud.lightning import PointCloudDataModule

    dm = PointCloudDataModule(batch_size=1, num_workers=0)
    with pytest.raises(ValueError, match="not provided"):
        dm.train_dataloader()
