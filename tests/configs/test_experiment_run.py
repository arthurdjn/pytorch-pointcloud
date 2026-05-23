from pathlib import Path
from typing import Callable, Dict, Iterator, List

import lightning.pytorch as L
import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch import Tensor
from torch.utils.data import Dataset

from torch_pointcloud.utils.imports import (
    _CUDA_AVAILABLE,
    _LIGHTNING_AVAILABLE,
    _SPCONV_AVAILABLE,
    _TORCH_CLUSTER_AVAILABLE,
    _TORCH_SCATTER_AVAILABLE,
)

SampleFn = Callable[[], Dict[str, Tensor]]
CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"


pytestmark = pytest.mark.skipif(not _LIGHTNING_AVAILABLE, reason="lightning is not installed")


_REQUIRES_SPCONV = pytest.mark.skipif(not _SPCONV_AVAILABLE, reason="spconv is not installed")
_REQUIRES_CUDA = pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA is not available")
_REQUIRES_TORCH_CLUSTER = pytest.mark.skipif(not _TORCH_CLUSTER_AVAILABLE, reason="torch-cluster is not installed")
_REQUIRES_TORCH_SCATTER = pytest.mark.skipif(not _TORCH_SCATTER_AVAILABLE, reason="torch-scatter is not installed")


def _scannet_sample(n: int = 256, dtype: torch.dtype = torch.float32) -> Dict[str, Tensor]:
    return {
        "x": torch.randn(n, 6, dtype=dtype),
        "pos": (torch.randn(n, 3, dtype=dtype) * 10).round(),  # integer-ish grid coords
        "segment": torch.randint(0, 20, (n,), dtype=torch.long),
    }


def _shapenetpart_sample(n: int = 2048) -> Dict[str, Tensor]:
    # `dgcnn-antao.shapenetpart` is registered with `in_channels=0`: the model
    # concatenates `x` with `pos` only if x is non-None, otherwise uses `pos`
    # alone. The lightning module reads input_keys, so `x` MUST be present in
    # the batch dict; passing an empty tensor preserves the same effective input.
    # OneHot in the transforms unsqueezes scalar labels to `(1, num_classes)` so
    # collate stacks them to `(B, num_classes)` — we mirror that here.
    return {
        "x": torch.empty(n, 0),
        "pos": torch.randn(n, 3),
        "segment": torch.randint(0, 50, (n,), dtype=torch.long),
        "category": torch.nn.functional.one_hot(torch.tensor(0), num_classes=16).float().unsqueeze(0),
    }


EXPERIMENTS = [
    pytest.param(
        "scannet_spunet",
        _scannet_sample,
        marks=[_REQUIRES_SPCONV, _REQUIRES_CUDA],
        id="scannet_spunet",
    ),
    pytest.param(
        "scannet_ptv3",
        _scannet_sample,
        marks=[_REQUIRES_SPCONV, _REQUIRES_CUDA],
        id="scannet_ptv3",
    ),
    pytest.param(
        "scannet_ptv3_v2",
        _scannet_sample,
        marks=[_REQUIRES_SPCONV, _REQUIRES_CUDA],
        id="scannet_ptv3_v2",
    ),
    pytest.param(
        "scannet_ptv3_v3",
        _scannet_sample,
        marks=[_REQUIRES_SPCONV, _REQUIRES_CUDA],
        id="scannet_ptv3_v3",
    ),
    pytest.param(
        "shapenetpart_dgcnn",
        _shapenetpart_sample,
        marks=[_REQUIRES_TORCH_CLUSTER, _REQUIRES_TORCH_SCATTER],
        id="shapenetpart_dgcnn",
    ),
]


class _StubDataset(Dataset):
    """Yields a fixed number of fake packed-batch samples."""

    def __init__(self, n: int, sample_fn: SampleFn) -> None:
        self._n = n
        self._sample_fn = sample_fn

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        # Re-seed per-sample so each call returns deterministic shapes (but distinct content).
        gen = torch.Generator().manual_seed(index)
        torch.manual_seed(int(gen.initial_seed()))
        return self._sample_fn()


@pytest.fixture(autouse=True)
def _register_eval_resolver() -> None:
    OmegaConf.register_new_resolver("eval", eval, replace=True)


@pytest.fixture(autouse=True)
def _clear_hydra() -> Iterator[None]:
    GlobalHydra.instance().clear()
    yield
    GlobalHydra.instance().clear()


@pytest.fixture(autouse=True)
def _fake_data_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TORCH_POINTCLOUD_DATA_DIR", str(tmp_path))


@pytest.mark.parametrize("experiment,sample_fn", EXPERIMENTS)
def test_experiment_fit_two_epochs(experiment: str, sample_fn: SampleFn, tmp_path: Path) -> None:
    """Compose the experiment, swap in a stub datamodule, fit 2 epochs on CPU."""
    from torch_pointcloud.lightning import PointCloudDataModule

    overrides: List[str] = [
        f"experiment={experiment}",
        f"paths.root_dir={tmp_path}",
        # `accelerator=auto` lets spconv use CUDA when available; CPU-only runs
        # gate via the `_REQUIRES_CUDA` skipif.
        "trainer.accelerator=auto",
        "trainer.devices=1",
        "trainer.max_epochs=2",
        "trainer.precision=32-true",
        "+trainer.limit_train_batches=1",
        "+trainer.limit_val_batches=1",
        "+trainer.logger=false",
        "+trainer.enable_checkpointing=false",
        "+trainer.enable_progress_bar=false",
        # Hydra needs to write its `.hydra/` dir somewhere writable.
        f"hydra.run.dir={tmp_path / 'run'}",
    ]

    with initialize_config_dir(config_dir=str(CONFIGS_DIR), version_base=None):
        cfg = compose(config_name="train", overrides=overrides, return_hydra_config=True)
        HydraConfig.instance().set_config(cfg)

    model = instantiate(cfg.model)
    trainer: L.Trainer = instantiate(cfg.trainer, callbacks=[], logger=False)
    dm = PointCloudDataModule(
        train_dataset=_StubDataset(n=4, sample_fn=sample_fn),
        val_dataset=_StubDataset(n=2, sample_fn=sample_fn),
        batch_size=2,
        num_workers=0,
        drop_last=False,
    )

    trainer.fit(model=model, datamodule=dm)
    assert trainer.current_epoch == 2  # 2 epochs completed
    assert "train/loss" in trainer.callback_metrics
