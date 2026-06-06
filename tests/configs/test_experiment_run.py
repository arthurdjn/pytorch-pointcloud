from pathlib import Path
from typing import Iterator, List

import pytest

from torch_pointcloud.utils.imports import (
    _CUDA_AVAILABLE,
    _DWCONV_AVAILABLE,
    _HYDRA_AVAILABLE,
    _LIGHTNING_AVAILABLE,
    _MAMBA_SSM_AVAILABLE,
    _OCNN_AVAILABLE,
    _SPCONV_AVAILABLE,
    _TORCH_CLUSTER_AVAILABLE,
    _TORCH_SCATTER_AVAILABLE,
    _TORCHSPARSE_AVAILABLE,
)

# Module-level skip: hydra and lightning are dev/optional deps. Their imports
# (and `import lightning.pytorch as L` in particular) would otherwise run at
# collection time on a bare install and crash pytest discovery.
pytestmark = [
    pytest.mark.skipif(not _LIGHTNING_AVAILABLE, reason="lightning is not installed"),
    pytest.mark.skipif(not _HYDRA_AVAILABLE, reason="hydra-core is not installed"),
    pytest.mark.experiment,
]

CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"

_REQUIRES_CUDA = pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA is not available")
_REQUIRES_SPCONV = pytest.mark.skipif(not _SPCONV_AVAILABLE, reason="spconv is not installed")
_REQUIRES_TORCHSPARSE = pytest.mark.skipif(not _TORCHSPARSE_AVAILABLE, reason="torchsparse is not installed")
_REQUIRES_OCNN = pytest.mark.skipif(not _OCNN_AVAILABLE, reason="ocnn is not installed")
_REQUIRES_DWCONV = pytest.mark.skipif(not _DWCONV_AVAILABLE, reason="dwconv is not installed")
_REQUIRES_MAMBA = pytest.mark.skipif(not _MAMBA_SSM_AVAILABLE, reason="mamba-ssm is not installed")
_REQUIRES_TORCH_CLUSTER = pytest.mark.skipif(not _TORCH_CLUSTER_AVAILABLE, reason="torch-cluster is not installed")
_REQUIRES_TORCH_SCATTER = pytest.mark.skipif(not _TORCH_SCATTER_AVAILABLE, reason="torch-scatter is not installed")


# Each row is (experiment, trainer accelerator) plus that model's optional-dependency skips. The config
# already names its dataset; the `datasets_dir` fixture points TORCH_POINTCLOUD_DATA_DIR at the whole
# dummy-data tree, so each experiment loads its own dataset from there. Point models run on CPU (as CI
# does, and so a heavy one cannot exhaust GPU memory); models needing a CUDA-only dependency run on the
# GPU and are skipped on CI. scannet200 experiments are omitted (no scannet200 dummy data); they stay
# covered by tests/configs/test_experiment_config.py.
EXPERIMENTS = [
    pytest.param(
        "dgcnn/dgcnn_modelnet40",
        "cpu",
        marks=[_REQUIRES_TORCH_CLUSTER],
        id="dgcnn/dgcnn_modelnet40",
    ),
    pytest.param(
        "dgcnn/dgcnn-2048_modelnet40",
        "cpu",
        marks=[_REQUIRES_TORCH_CLUSTER],
        id="dgcnn/dgcnn-2048_modelnet40",
    ),
    pytest.param(
        "dgcnn/dgcnn_shapenetpart",
        "cpu",
        marks=[_REQUIRES_TORCH_CLUSTER],
        id="dgcnn/dgcnn_shapenetpart",
    ),
    pytest.param(
        "pointnet2/pointnet2_modelnet40",
        "cpu",
        marks=[_REQUIRES_TORCH_CLUSTER],
        id="pointnet2/pointnet2_modelnet40",
    ),
    pytest.param(
        "pointnet2/pointnet2-msg_modelnet40",
        "cpu",
        marks=[_REQUIRES_TORCH_CLUSTER],
        id="pointnet2/pointnet2-msg_modelnet40",
    ),
    pytest.param(
        "pointnet2/pointnet2-openpoints_modelnet40",
        "cpu",
        marks=[_REQUIRES_TORCH_CLUSTER],
        id="pointnet2/pointnet2-openpoints_modelnet40",
    ),
    pytest.param(
        "pointnet2/pointnet2-openpoints_scanobjectnn",
        "cpu",
        marks=[_REQUIRES_TORCH_CLUSTER],
        id="pointnet2/pointnet2-openpoints_scanobjectnn",
    ),
    pytest.param(
        "pointnet2/pointnet2-openpoints_s3dis-area5",
        "cpu",
        marks=[_REQUIRES_TORCH_CLUSTER],
        id="pointnet2/pointnet2-openpoints_s3dis-area5",
    ),
    pytest.param(
        "pointconv/pointconv_modelnet40",
        "cpu",
        marks=[_REQUIRES_TORCH_CLUSTER, _REQUIRES_TORCH_SCATTER],
        id="pointconv/pointconv_modelnet40",
    ),
    pytest.param(
        "pointmlp/pointmlp_modelnet40",
        "cpu",
        marks=[_REQUIRES_TORCH_CLUSTER, _REQUIRES_TORCH_SCATTER],
        id="pointmlp/pointmlp_modelnet40",
    ),
    pytest.param(
        "pointmlp/pointmlp-elite_modelnet40",
        "cpu",
        marks=[_REQUIRES_TORCH_CLUSTER, _REQUIRES_TORCH_SCATTER],
        id="pointmlp/pointmlp-elite_modelnet40",
    ),
    pytest.param(
        "pointmlp/pointmlp_scanobjectnn",
        "cpu",
        marks=[_REQUIRES_TORCH_CLUSTER, _REQUIRES_TORCH_SCATTER],
        id="pointmlp/pointmlp_scanobjectnn",
    ),
    pytest.param(
        "pointmlp/pointmlp-elite_scanobjectnn",
        "cpu",
        marks=[_REQUIRES_TORCH_CLUSTER, _REQUIRES_TORCH_SCATTER],
        id="pointmlp/pointmlp-elite_scanobjectnn",
    ),
    pytest.param(
        "pointnext/pointnext_modelnet40",
        "cpu",
        marks=[_REQUIRES_TORCH_CLUSTER, _REQUIRES_TORCH_SCATTER],
        id="pointnext/pointnext_modelnet40",
    ),
    pytest.param(
        "pointnext/pointnext_scanobjectnn",
        "cpu",
        marks=[_REQUIRES_TORCH_CLUSTER, _REQUIRES_TORCH_SCATTER],
        id="pointnext/pointnext_scanobjectnn",
    ),
    pytest.param(
        "pointnext/pointnext-base_s3dis-area5",
        "cpu",
        marks=[_REQUIRES_TORCH_CLUSTER, _REQUIRES_TORCH_SCATTER],
        id="pointnext/pointnext-base_s3dis-area5",
    ),
    pytest.param(
        "pointnext/pointnext-sm_s3dis-area5",
        "cpu",
        marks=[_REQUIRES_TORCH_CLUSTER, _REQUIRES_TORCH_SCATTER],
        id="pointnext/pointnext-sm_s3dis-area5",
    ),
    pytest.param(
        "pointnext/pointnext-lg_s3dis-area5",
        "cpu",
        marks=[_REQUIRES_TORCH_CLUSTER, _REQUIRES_TORCH_SCATTER],
        id="pointnext/pointnext-lg_s3dis-area5",
    ),
    pytest.param(
        "pointnext/pointnext-xl_s3dis-area5",
        "cpu",
        marks=[_REQUIRES_TORCH_CLUSTER, _REQUIRES_TORCH_SCATTER],
        id="pointnext/pointnext-xl_s3dis-area5",
    ),
    pytest.param(
        "kpfcnn/kpfcnn-base_s3dis",
        "cpu",
        marks=[_REQUIRES_TORCH_CLUSTER, _REQUIRES_TORCH_SCATTER],
        id="kpfcnn/kpfcnn-base_s3dis",
    ),
    pytest.param(
        "kpfcnn/kpfcnn-base-deform_s3dis",
        "cpu",
        marks=[_REQUIRES_TORCH_CLUSTER, _REQUIRES_TORCH_SCATTER],
        id="kpfcnn/kpfcnn-base-deform_s3dis",
    ),
    pytest.param(
        "kpfcnn/kpfcnn-base-sm_s3dis",
        "cpu",
        marks=[_REQUIRES_TORCH_CLUSTER, _REQUIRES_TORCH_SCATTER],
        id="kpfcnn/kpfcnn-base-sm_s3dis",
    ),
    pytest.param(
        "kpfcnn/kpfcnn-base-sm-deform_s3dis",
        "cpu",
        marks=[_REQUIRES_TORCH_CLUSTER, _REQUIRES_TORCH_SCATTER],
        id="kpfcnn/kpfcnn-base-sm-deform_s3dis",
    ),
    pytest.param(
        "pvcnn/pvcnn_s3dis",
        "cpu",
        marks=[_REQUIRES_TORCH_SCATTER],
        id="pvcnn/pvcnn_s3dis",
    ),
    pytest.param(
        "randlanet/randlanet_semantickitti",
        "cpu",
        marks=[_REQUIRES_TORCH_CLUSTER, _REQUIRES_TORCH_SCATTER],
        id="randlanet/randlanet_semantickitti",
    ),
    pytest.param(
        "votenet/votenet_sunrgbd",
        "cpu",
        marks=[_REQUIRES_TORCH_CLUSTER, _REQUIRES_TORCH_SCATTER],
        id="votenet/votenet_sunrgbd",
    ),
    pytest.param(
        "spunet/spunet_scannet",
        "auto",
        marks=[_REQUIRES_SPCONV, _REQUIRES_CUDA],
        id="spunet/spunet_scannet",
    ),
    pytest.param(
        "point_transformer_v3/point_transformer_v3_scannet",
        "auto",
        marks=[_REQUIRES_SPCONV, _REQUIRES_TORCH_SCATTER, _REQUIRES_CUDA],
        id="point_transformer_v3/point_transformer_v3_scannet",
    ),
    pytest.param(
        "point_transformer_v3/point_transformer_v3_s3dis",
        "auto",
        marks=[_REQUIRES_SPCONV, _REQUIRES_TORCH_SCATTER, _REQUIRES_CUDA],
        id="point_transformer_v3/point_transformer_v3_s3dis",
    ),
    pytest.param(
        "sonata/sonata_scannet",
        "auto",
        marks=[_REQUIRES_SPCONV, _REQUIRES_TORCH_SCATTER, _REQUIRES_CUDA],
        id="sonata/sonata_scannet",
    ),
    pytest.param(
        "concerto/concerto_scannet",
        "auto",
        marks=[_REQUIRES_SPCONV, _REQUIRES_TORCH_SCATTER, _REQUIRES_CUDA],
        id="concerto/concerto_scannet",
    ),
    pytest.param(
        "utonia/utonia_scannet",
        "auto",
        marks=[_REQUIRES_SPCONV, _REQUIRES_TORCH_SCATTER, _REQUIRES_CUDA],
        id="utonia/utonia_scannet",
    ),
    pytest.param(
        "spvcnn/spvcnn-119gmacs_semantickitti",
        "auto",
        marks=[_REQUIRES_TORCHSPARSE, _REQUIRES_CUDA],
        id="spvcnn/spvcnn-119gmacs_semantickitti",
    ),
    pytest.param(
        "spvcnn/spvcnn-30gmacs_semantickitti",
        "auto",
        marks=[_REQUIRES_TORCHSPARSE, _REQUIRES_CUDA],
        id="spvcnn/spvcnn-30gmacs_semantickitti",
    ),
    pytest.param(
        "spvcnn/spvcnn-47gmacs_semantickitti",
        "auto",
        marks=[_REQUIRES_TORCHSPARSE, _REQUIRES_CUDA],
        id="spvcnn/spvcnn-47gmacs_semantickitti",
    ),
    pytest.param(
        "point_mamba/point_mamba_modelnet40",
        "auto",
        marks=[_REQUIRES_MAMBA, _REQUIRES_CUDA, _REQUIRES_TORCH_CLUSTER, _REQUIRES_TORCH_SCATTER],
        id="point_mamba/point_mamba_modelnet40",
    ),
    pytest.param(
        "point_mamba/point_mamba_scanobjectnn",
        "auto",
        marks=[_REQUIRES_MAMBA, _REQUIRES_CUDA, _REQUIRES_TORCH_CLUSTER, _REQUIRES_TORCH_SCATTER],
        id="point_mamba/point_mamba_scanobjectnn",
    ),
    pytest.param(
        "point_mamba/point_mamba-arot_scanobjectnn",
        "auto",
        marks=[_REQUIRES_MAMBA, _REQUIRES_CUDA, _REQUIRES_TORCH_CLUSTER, _REQUIRES_TORCH_SCATTER],
        id="point_mamba/point_mamba-arot_scanobjectnn",
    ),
    pytest.param(
        "point_mamba/point_mamba-nobg_scanobjectnn",
        "auto",
        marks=[_REQUIRES_MAMBA, _REQUIRES_CUDA, _REQUIRES_TORCH_CLUSTER, _REQUIRES_TORCH_SCATTER],
        id="point_mamba/point_mamba-nobg_scanobjectnn",
    ),
    pytest.param(
        "octformer/octformer_modelnet40",
        "auto",
        marks=[_REQUIRES_OCNN, _REQUIRES_DWCONV, _REQUIRES_CUDA],
        id="octformer/octformer_modelnet40",
    ),
    pytest.param(
        "octformer/octformer_scannet",
        "auto",
        marks=[_REQUIRES_OCNN, _REQUIRES_DWCONV, _REQUIRES_CUDA],
        id="octformer/octformer_scannet",
    ),
]


@pytest.fixture(autouse=True)
def _register_eval_resolver() -> None:
    from omegaconf import OmegaConf

    OmegaConf.register_new_resolver("eval", eval, replace=True)


@pytest.fixture(autouse=True)
def _clear_hydra() -> Iterator[None]:
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    yield
    GlobalHydra.instance().clear()


@pytest.fixture(autouse=True)
def _release_cuda_memory() -> Iterator[None]:
    """Fitting every experiment in one process accumulates GPU memory (the spconv models hold several
    GB); reclaim it between tests so the later CUDA fits do not run out of memory locally."""
    import gc

    import torch

    yield
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.mark.parametrize("experiment,accelerator", EXPERIMENTS)
def test_experiment_fit_two_epochs(
    experiment: str,
    accelerator: str,
    datasets_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compose the experiment, fit 2 epochs on the real datamodule pointed at the dummy datasets."""
    import lightning.pytorch as L  # noqa: N812
    from hydra import compose, initialize_config_dir
    from hydra.core.hydra_config import HydraConfig
    from hydra.utils import instantiate

    monkeypatch.setenv("TORCH_POINTCLOUD_DATA_DIR", str(datasets_dir))

    overrides: List[str] = [
        f"experiment={experiment}",
        f"paths.root_dir={tmp_path}",
        f"trainer.accelerator={accelerator}",
        "trainer.devices=1",
        "trainer.max_epochs=2",
        "trainer.precision=32-true",
        "+trainer.limit_train_batches=1",
        "+trainer.limit_val_batches=1",
        "+trainer.logger=false",
        "+trainer.enable_checkpointing=false",
        "+trainer.enable_progress_bar=false",
        # The dummy datasets only have a handful of samples; shrink the batch and keep the partial one.
        "data.batch_size=2",
        "data.num_workers=0",
        "+data.drop_last=false",
        f"hydra.run.dir={tmp_path / 'run'}",
    ]

    with initialize_config_dir(config_dir=str(CONFIGS_DIR), version_base=None):
        cfg = compose(config_name="train", overrides=overrides, return_hydra_config=True)
        HydraConfig.instance().set_config(cfg)

    model = instantiate(cfg.model)
    trainer: L.Trainer = instantiate(cfg.trainer, callbacks=[], logger=False)
    dm = instantiate(cfg.data)

    trainer.fit(model=model, datamodule=dm)
    assert trainer.current_epoch == 2  # 2 epochs completed
    assert "train/loss" in trainer.callback_metrics
