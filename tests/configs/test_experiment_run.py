from pathlib import Path
from typing import Iterator, List, NamedTuple, Tuple

import pytest

from torch_pointcloud.utils.imports import (
    _CUDA_AVAILABLE,
    _DWCONV_AVAILABLE,
    _HYDRA_AVAILABLE,
    _LIGHTNING_AVAILABLE,
    _MAMBA_SSM_AVAILABLE,
    _OCNN_AVAILABLE,
    _SPCONV_AVAILABLE,
    _SPTR_AVAILABLE,
    _TORCH_CLUSTER_AVAILABLE,
    _TORCH_SCATTER_AVAILABLE,
    _TORCHSPARSE_AVAILABLE,
)

# Hydra and lightning are dev/optional deps; their imports would crash pytest discovery on a bare install.
pytestmark = [
    pytest.mark.skipif(not _LIGHTNING_AVAILABLE, reason="lightning is not installed"),
    pytest.mark.skipif(not _HYDRA_AVAILABLE, reason="hydra-core is not installed"),
]

CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"

_REQUIRES_CUDA = pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA is not available")
_REQUIRES_SPCONV = pytest.mark.skipif(not _SPCONV_AVAILABLE, reason="spconv is not installed")
_REQUIRES_TORCHSPARSE = pytest.mark.skipif(not _TORCHSPARSE_AVAILABLE, reason="torchsparse is not installed")
_REQUIRES_OCNN = pytest.mark.skipif(not _OCNN_AVAILABLE, reason="ocnn is not installed")
_REQUIRES_DWCONV = pytest.mark.skipif(not _DWCONV_AVAILABLE, reason="dwconv is not installed")
_REQUIRES_MAMBA = pytest.mark.skipif(not _MAMBA_SSM_AVAILABLE, reason="mamba-ssm is not installed")
_REQUIRES_SPTR = pytest.mark.skipif(not _SPTR_AVAILABLE, reason="sptr is not installed")
_REQUIRES_TORCH_CLUSTER = pytest.mark.skipif(not _TORCH_CLUSTER_AVAILABLE, reason="torch-cluster is not installed")
_REQUIRES_TORCH_SCATTER = pytest.mark.skipif(not _TORCH_SCATTER_AVAILABLE, reason="torch-scatter is not installed")

# Shorthands for the KITTI rows: the dummy tree ships no ImageSets/ frame lists and no image_2/, so the
# split-file selection and the front-camera FOV filter are disabled for the smoke run.
_KITTI_DUMMY_OVERRIDES = ("datamodule.val_dataset.split_file=null", "datamodule.val_dataset.fov=false")
# The fit path builds the train split too, so it also needs the train dataset fixups.
_KITTI_DUMMY_TRAIN_OVERRIDES = (
    "datamodule.train_dataset.split_file=null",
    "datamodule.train_dataset.fov=false",
    *_KITTI_DUMMY_OVERRIDES,
)
# Benchmark mode mirrors the documented invocation (`test.py experiment=... model.pretrained=true`): rows
# whose model ships registry weights append this so it composes after the base random-weights override
# and wins; rows without fetchable weights keep the random weights.
_PRETRAINED = ("model.pretrained=true",)


class Experiment(NamedTuple):
    """Runtime-test row for one experiment config.

    `train` fits 2 epochs on the dummy datasets (requires an `optimizer` in the config) and `test` runs the
    eval-only `test.py` path. Both `False` is an explicit opt-out and must say why in a comment on the row.
    """

    experiment: str
    accelerator: str
    train: bool
    test: bool
    marks: Tuple[pytest.MarkDecorator, ...] = ()
    benchmark_overrides: Tuple[str, ...] = ()
    train_overrides: Tuple[str, ...] = ()


_CPU_CLUSTER = (_REQUIRES_TORCH_CLUSTER,)
_CPU_CLUSTER_SCATTER = (_REQUIRES_TORCH_CLUSTER, _REQUIRES_TORCH_SCATTER)
_GPU_SPCONV = (_REQUIRES_SPCONV, _REQUIRES_CUDA)
_GPU_SPCONV_SCATTER = (_REQUIRES_SPCONV, _REQUIRES_TORCH_SCATTER, _REQUIRES_CUDA)
_GPU_OCTREE = (_REQUIRES_OCNN, _REQUIRES_DWCONV, _REQUIRES_CUDA)
_GPU_MAMBA = (_REQUIRES_MAMBA, _REQUIRES_CUDA, _REQUIRES_TORCH_CLUSTER, _REQUIRES_TORCH_SCATTER)

EXPERIMENTS = (
    Experiment(
        "3detr/scannet", "cpu", train=True, test=True, marks=_CPU_CLUSTER_SCATTER, benchmark_overrides=_PRETRAINED
    ),
    Experiment(
        "3detr/sunrgbd", "cpu", train=True, test=True, marks=_CPU_CLUSTER_SCATTER, benchmark_overrides=_PRETRAINED
    ),
    Experiment(
        "concerto/scannet", "auto", train=True, test=True, marks=_GPU_SPCONV_SCATTER, benchmark_overrides=_PRETRAINED
    ),
    Experiment("dgcnn/modelnet40", "cpu", train=True, test=True, marks=_CPU_CLUSTER, benchmark_overrides=_PRETRAINED),
    Experiment(
        "dgcnn/modelnet40-2048", "cpu", train=True, test=True, marks=_CPU_CLUSTER, benchmark_overrides=_PRETRAINED
    ),
    Experiment("dgcnn/s3dis", "cpu", train=False, test=True, marks=_CPU_CLUSTER, benchmark_overrides=_PRETRAINED),
    Experiment("dgcnn/scannet", "cpu", train=False, test=True, marks=_CPU_CLUSTER, benchmark_overrides=_PRETRAINED),
    Experiment("dgcnn/shapenetpart", "cpu", train=True, test=True, marks=_CPU_CLUSTER, benchmark_overrides=_PRETRAINED),
    Experiment(
        "kpfcnn/s3dis", "cpu", train=True, test=True, marks=_CPU_CLUSTER_SCATTER, benchmark_overrides=_PRETRAINED
    ),
    Experiment(
        "lion/nuscenes",
        "auto",
        train=True,
        test=True,
        marks=_GPU_MAMBA + (_REQUIRES_SPCONV,),
        benchmark_overrides=(*_PRETRAINED, "model.score_threshold=0.99"),
        train_overrides=("model.score_threshold=0.99",),
    ),
    Experiment(
        "octformer/modelnet40", "auto", train=True, test=True, marks=_GPU_OCTREE, benchmark_overrides=_PRETRAINED
    ),
    Experiment("octformer/scannet", "auto", train=True, test=True, marks=_GPU_OCTREE, benchmark_overrides=_PRETRAINED),
    # No scannet200 dummy dataset; compose-tested only.
    Experiment("octformer/scannet200", "auto", train=False, test=False, marks=_GPU_OCTREE),
    Experiment(
        "point_bert/modelnet40",
        "cpu",
        train=False,
        test=True,
        marks=_CPU_CLUSTER_SCATTER,
        benchmark_overrides=_PRETRAINED,
    ),
    Experiment(
        "point_bert/scanobjectnn-hardest",
        "cpu",
        train=False,
        test=True,
        marks=_CPU_CLUSTER_SCATTER,
        benchmark_overrides=_PRETRAINED,
    ),
    Experiment(
        "point_bert/scanobjectnn-objbg",
        "cpu",
        train=False,
        test=True,
        marks=_CPU_CLUSTER_SCATTER,
        benchmark_overrides=_PRETRAINED,
    ),
    Experiment(
        "point_bert/scanobjectnn-objonly",
        "cpu",
        train=False,
        test=True,
        marks=_CPU_CLUSTER_SCATTER,
        benchmark_overrides=_PRETRAINED,
    ),
    Experiment(
        "point_m2ae/modelnet40",
        "cpu",
        train=False,
        test=True,
        marks=_CPU_CLUSTER_SCATTER,
        benchmark_overrides=_PRETRAINED,
    ),
    Experiment(
        "point_m2ae/scanobjectnn-hardest",
        "cpu",
        train=False,
        test=True,
        marks=_CPU_CLUSTER_SCATTER,
        benchmark_overrides=_PRETRAINED,
    ),
    Experiment(
        "point_m2ae/scanobjectnn-objbg",
        "cpu",
        train=False,
        test=True,
        marks=_CPU_CLUSTER_SCATTER,
        benchmark_overrides=_PRETRAINED,
    ),
    Experiment(
        "point_m2ae/shapenetpart",
        "cpu",
        train=False,
        test=True,
        marks=_CPU_CLUSTER_SCATTER,
        benchmark_overrides=_PRETRAINED,
    ),
    Experiment(
        "point_mae/modelnet40",
        "cpu",
        train=False,
        test=True,
        marks=_CPU_CLUSTER_SCATTER,
        benchmark_overrides=_PRETRAINED,
    ),
    Experiment(
        "point_mae/scanobjectnn-hardest",
        "cpu",
        train=False,
        test=True,
        marks=_CPU_CLUSTER_SCATTER,
        benchmark_overrides=_PRETRAINED,
    ),
    Experiment(
        "point_mae/scanobjectnn-objbg",
        "cpu",
        train=False,
        test=True,
        marks=_CPU_CLUSTER_SCATTER,
        benchmark_overrides=_PRETRAINED,
    ),
    Experiment(
        "point_mae/scanobjectnn-objonly",
        "cpu",
        train=False,
        test=True,
        marks=_CPU_CLUSTER_SCATTER,
        benchmark_overrides=_PRETRAINED,
    ),
    Experiment(
        "point_mae/shapenetpart",
        "cpu",
        train=False,
        test=True,
        marks=_CPU_CLUSTER_SCATTER,
        benchmark_overrides=_PRETRAINED,
    ),
    Experiment(
        "point_mamba/modelnet40", "auto", train=True, test=True, marks=_GPU_MAMBA, benchmark_overrides=_PRETRAINED
    ),
    Experiment(
        "point_mamba/scanobjectnn", "auto", train=True, test=True, marks=_GPU_MAMBA, benchmark_overrides=_PRETRAINED
    ),
    Experiment(
        "point_mamba/scanobjectnn-augmentedrot-scale75",
        "auto",
        train=True,
        test=True,
        marks=_GPU_MAMBA,
        benchmark_overrides=_PRETRAINED,
    ),
    Experiment(
        "point_mamba/scanobjectnn-nobg",
        "auto",
        train=True,
        test=True,
        marks=_GPU_MAMBA,
        benchmark_overrides=_PRETRAINED,
    ),
    # pointcnn-base has no registered weights, so the pretrained benchmark is not a supported workflow;
    # the train recipe is the point of these configs.
    Experiment("pointcnn/modelnet40", "cpu", train=True, test=False, marks=_CPU_CLUSTER),
    Experiment("pointcnn/shapenetpart", "cpu", train=True, test=False, marks=_CPU_CLUSTER),
    Experiment(
        "pointconv/modelnet40",
        "cpu",
        train=True,
        test=True,
        marks=_CPU_CLUSTER_SCATTER,
        benchmark_overrides=_PRETRAINED,
    ),
    Experiment(
        "pointgpt/modelnet40",
        "cpu",
        train=False,
        test=True,
        marks=_CPU_CLUSTER_SCATTER,
        benchmark_overrides=_PRETRAINED,
    ),
    Experiment(
        "pointgpt/scanobjectnn-hardest",
        "cpu",
        train=False,
        test=True,
        marks=_CPU_CLUSTER_SCATTER,
        benchmark_overrides=_PRETRAINED,
    ),
    Experiment(
        "pointgpt/scanobjectnn-objbg",
        "cpu",
        train=False,
        test=True,
        marks=_CPU_CLUSTER_SCATTER,
        benchmark_overrides=_PRETRAINED,
    ),
    Experiment(
        "pointgpt/scanobjectnn-objonly",
        "cpu",
        train=False,
        test=True,
        marks=_CPU_CLUSTER_SCATTER,
        benchmark_overrides=_PRETRAINED,
    ),
    Experiment(
        "pointmlp/modelnet40", "cpu", train=True, test=True, marks=_CPU_CLUSTER_SCATTER, benchmark_overrides=_PRETRAINED
    ),
    Experiment(
        "pointmlp/scanobjectnn",
        "cpu",
        train=True,
        test=True,
        marks=_CPU_CLUSTER_SCATTER,
        benchmark_overrides=_PRETRAINED,
    ),
    # PointNet v1 has no registered weights, so the pretrained benchmark is not a supported workflow;
    # the train recipe is the point of these configs.
    Experiment("pointnet/modelnet40", "cpu", train=True, test=False, marks=_CPU_CLUSTER_SCATTER),
    Experiment("pointnet/s3dis", "cpu", train=True, test=False, marks=(_REQUIRES_TORCH_SCATTER,)),
    Experiment("pointnet/shapenetpart", "cpu", train=True, test=False, marks=(_REQUIRES_TORCH_SCATTER,)),
    Experiment(
        "pointnet2/modelnet40", "cpu", train=True, test=True, marks=_CPU_CLUSTER, benchmark_overrides=_PRETRAINED
    ),
    Experiment(
        "pointnet2/msg_modelnet40", "cpu", train=True, test=True, marks=_CPU_CLUSTER, benchmark_overrides=_PRETRAINED
    ),
    Experiment(
        "pointnet2/openpoints_modelnet40",
        "cpu",
        train=True,
        test=True,
        marks=_CPU_CLUSTER,
        benchmark_overrides=_PRETRAINED,
    ),
    Experiment(
        "pointnet2/openpoints_s3dis", "cpu", train=True, test=True, marks=_CPU_CLUSTER, benchmark_overrides=_PRETRAINED
    ),
    Experiment(
        "pointnet2/openpoints_scanobjectnn",
        "cpu",
        train=True,
        test=True,
        marks=_CPU_CLUSTER,
        benchmark_overrides=_PRETRAINED,
    ),
    Experiment(
        "pointnet2/yanx27_s3dis", "cpu", train=True, test=True, marks=_CPU_CLUSTER, benchmark_overrides=_PRETRAINED
    ),
    Experiment(
        "pointnext/modelnet40",
        "cpu",
        train=True,
        test=True,
        marks=_CPU_CLUSTER_SCATTER,
        benchmark_overrides=_PRETRAINED,
    ),
    Experiment(
        "pointnext/s3dis", "cpu", train=True, test=True, marks=_CPU_CLUSTER_SCATTER, benchmark_overrides=_PRETRAINED
    ),
    Experiment(
        "pointnext/scanobjectnn",
        "cpu",
        train=True,
        test=True,
        marks=_CPU_CLUSTER_SCATTER,
        benchmark_overrides=_PRETRAINED,
    ),
    Experiment(
        "pointnext/shapenetpart",
        "cpu",
        train=False,
        test=True,
        marks=_CPU_CLUSTER_SCATTER,
        benchmark_overrides=_PRETRAINED,
    ),
    # Random weights leave most dense anchors above the production score threshold; the pairwise
    # rotated-BEV NMS over that set exhausts host memory.
    Experiment(
        "pointpillars/kitti",
        "cpu",
        train=True,
        test=True,
        marks=(_REQUIRES_TORCH_SCATTER,),
        benchmark_overrides=(*_PRETRAINED, *_KITTI_DUMMY_OVERRIDES),
        train_overrides=(*_KITTI_DUMMY_TRAIN_OVERRIDES, "model.score_threshold=0.99"),
    ),
    # Random weights score ~0.5 on every dense nuScenes anchor; at the production score threshold the
    # pairwise per-class NMS over the full ~500k-anchor set exhausts host memory.
    Experiment(
        "pointpillars/nuscenes",
        "cpu",
        train=True,
        test=True,
        marks=(_REQUIRES_TORCH_SCATTER,),
        benchmark_overrides=(*_PRETRAINED, "model.score_threshold=0.99"),
        train_overrides=("model.score_threshold=0.99",),
    ),
    Experiment(
        "pointrcnn/kitti",
        "cpu",
        train=True,
        test=True,
        marks=_CPU_CLUSTER_SCATTER,
        benchmark_overrides=(*_PRETRAINED, *_KITTI_DUMMY_OVERRIDES),
        train_overrides=_KITTI_DUMMY_TRAIN_OVERRIDES,
    ),
    # Point Transformer v1/v2 have no registered weights; the train recipe is the point of these configs.
    Experiment("point_transformer/modelnet40", "cpu", train=True, test=False, marks=_CPU_CLUSTER_SCATTER),
    Experiment("point_transformer/s3dis", "cpu", train=True, test=False, marks=_CPU_CLUSTER_SCATTER),
    Experiment("point_transformer/scannet", "cpu", train=True, test=False, marks=_CPU_CLUSTER_SCATTER),
    Experiment("point_transformer_v2/scannet", "cpu", train=True, test=False, marks=_CPU_CLUSTER_SCATTER),
    # No scannet200 dummy dataset; compose-tested only.
    Experiment("point_transformer_v2/scannet200", "cpu", train=False, test=False, marks=_CPU_CLUSTER_SCATTER),
    Experiment(
        "point_transformer_v3/s3dis",
        "auto",
        train=True,
        test=True,
        marks=_GPU_SPCONV_SCATTER,
        benchmark_overrides=_PRETRAINED,
    ),
    Experiment(
        "point_transformer_v3/scannet",
        "auto",
        train=True,
        test=True,
        marks=_GPU_SPCONV_SCATTER,
        benchmark_overrides=_PRETRAINED,
    ),
    # No scannet200 dummy dataset; compose-tested only.
    Experiment("point_transformer_v3/scannet200", "auto", train=False, test=False, marks=_GPU_SPCONV_SCATTER),
    # Joint ScanNet20 + S3DIS training with PDNorm; the multi-dataset train recipe is the point (no benchmark).
    Experiment("point_transformer_v3/scannet_s3dis_joint", "auto", train=True, test=False, marks=_GPU_SPCONV_SCATTER),
    Experiment(
        "pvcnn/s3dis", "cpu", train=True, test=True, marks=(_REQUIRES_TORCH_SCATTER,), benchmark_overrides=_PRETRAINED
    ),
    # PVCNN++ has no registered weights; the train recipe is the point of this config.
    Experiment("pvcnn2/s3dis", "cpu", train=True, test=False, marks=_CPU_CLUSTER_SCATTER),
    # The dummy SemanticKITTI tree ships sequences 00 / 08 / 11 only; the train split wants 00-10 minus 08.
    Experiment(
        "randlanet/semantickitti",
        "cpu",
        train=True,
        test=True,
        marks=_CPU_CLUSTER_SCATTER,
        benchmark_overrides=_PRETRAINED,
        train_overrides=('+datamodule.train_dataset.sequences=["00"]',),
    ),
    # Random weights leave most dense anchors above the production score threshold; the pairwise
    # rotated-BEV NMS over that set exhausts host memory.
    Experiment(
        "second/kitti",
        "auto",
        train=True,
        test=True,
        marks=_GPU_SPCONV,
        benchmark_overrides=(*_PRETRAINED, *_KITTI_DUMMY_OVERRIDES),
        train_overrides=(*_KITTI_DUMMY_TRAIN_OVERRIDES, "model.score_threshold=0.99"),
    ),
    # Same dense-anchor memory blow-up as pointpillars/nuscenes under random weights.
    Experiment(
        "second/nuscenes",
        "auto",
        train=True,
        test=True,
        marks=_GPU_SPCONV,
        benchmark_overrides=(*_PRETRAINED, "model.score_threshold=0.99"),
        train_overrides=("model.score_threshold=0.99",),
    ),
    Experiment(
        "sonata/scannet", "auto", train=True, test=True, marks=_GPU_SPCONV_SCATTER, benchmark_overrides=_PRETRAINED
    ),
    # spformer-unet.scannet20 has no registered weights, so the pretrained benchmark is not a supported
    # workflow; the train recipe is the point of this config.
    Experiment("spformer_unet/scannet", "auto", train=True, test=False, marks=_GPU_SPCONV),
    # sphereformer has no registered weights (dead upstream links), so benchmark mode keeps random weights.
    Experiment("sphereformer/semantickitti", "auto", train=False, test=True, marks=_GPU_SPCONV + (_REQUIRES_SPTR,)),
    Experiment("spunet/scannet", "auto", train=True, test=True, marks=_GPU_SPCONV, benchmark_overrides=_PRETRAINED),
    Experiment(
        "spvcnn/semantickitti",
        "auto",
        train=True,
        test=True,
        marks=(_REQUIRES_TORCHSPARSE, _REQUIRES_CUDA),
        benchmark_overrides=_PRETRAINED,
        train_overrides=('+datamodule.train_dataset.sequences=["00"]',),
    ),
    Experiment(
        "utonia/scannet", "auto", train=True, test=True, marks=_GPU_SPCONV_SCATTER, benchmark_overrides=_PRETRAINED
    ),
    # Eval-only config: no train recipe (the checkpoint comes from the reference implementation).
    Experiment(
        "votenet/scannet", "cpu", train=False, test=True, marks=_CPU_CLUSTER_SCATTER, benchmark_overrides=_PRETRAINED
    ),
    Experiment(
        "votenet/sunrgbd", "cpu", train=True, test=True, marks=_CPU_CLUSTER_SCATTER, benchmark_overrides=_PRETRAINED
    ),
    Experiment("voxelnext/nuscenes", "auto", train=True, test=True, marks=_GPU_SPCONV, benchmark_overrides=_PRETRAINED),
)

TRAIN_EXPERIMENTS = [pytest.param(run, marks=list(run.marks), id=run.experiment) for run in EXPERIMENTS if run.train]
BENCHMARK_EXPERIMENTS = [pytest.param(run, marks=list(run.marks), id=run.experiment) for run in EXPERIMENTS if run.test]


def test_experiment_runs_cover_all_configs() -> None:
    """Every experiment YAML has a row in `EXPERIMENTS` declaring its runtime coverage (an empty
    `modes` is an explicit, commented opt-out), so a new config cannot skip runtime testing silently."""
    experiment_dir = CONFIGS_DIR / "experiment"
    on_disk = {p.relative_to(experiment_dir).with_suffix("").as_posix() for p in experiment_dir.rglob("*.yaml")}
    assert {run.experiment for run in EXPERIMENTS} == on_disk


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


@pytest.mark.experiment
@pytest.mark.parametrize("run", TRAIN_EXPERIMENTS)
def test_experiment_fit_two_epochs(
    run: Experiment,
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
        f"experiment={run.experiment}",
        f"paths.root_dir={tmp_path}",
        f"trainer.accelerator={run.accelerator}",
        "trainer.devices=1",
        "trainer.max_epochs=2",
        "trainer.precision=32-true",
        "+trainer.limit_train_batches=1",
        "+trainer.limit_val_batches=1",
        "+trainer.logger=false",
        "+trainer.enable_checkpointing=false",
        "+trainer.enable_progress_bar=false",
        # The dummy datasets only have a handful of samples; shrink the batch and keep the partial one.
        "datamodule.batch_size=2",
        "datamodule.num_workers=0",
        "+datamodule.drop_last=false",
        f"hydra.run.dir={tmp_path / 'run'}",
        *run.train_overrides,
    ]

    with initialize_config_dir(config_dir=str(CONFIGS_DIR), version_base=None):
        cfg = compose(config_name="train", overrides=overrides, return_hydra_config=True)
        HydraConfig.instance().set_config(cfg)

    model = instantiate(cfg.model)
    trainer: L.Trainer = instantiate(cfg.trainer, callbacks=[], logger=False)
    dm = instantiate(cfg.datamodule)

    trainer.fit(model=model, datamodule=dm)
    assert trainer.current_epoch == 2  # 2 epochs completed
    assert "train/loss" in trainer.callback_metrics


@pytest.mark.experiment
@pytest.mark.parametrize("run", BENCHMARK_EXPERIMENTS)
def test_experiment_benchmark_mode(
    run: Experiment,
    datasets_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compose the `test.py` config, run `Trainer.test` on the dummy datasets, and check the metric
    callback logged a `test/` metric. The base overrides pin random weights; rows whose model ships
    registry weights re-enable the documented `model.pretrained=true` load through `benchmark_overrides`."""
    import lightning.pytorch as L  # noqa: N812
    from hydra import compose, initialize_config_dir
    from hydra.core.hydra_config import HydraConfig
    from hydra.utils import instantiate

    monkeypatch.setenv("TORCH_POINTCLOUD_DATA_DIR", str(datasets_dir))

    overrides: List[str] = [
        f"experiment={run.experiment}",
        "model.pretrained=false",
        # test.py nulls the train split; mirror it so only the eval split is built.
        "datamodule.train_dataset=null",
        f"paths.root_dir={tmp_path}",
        f"trainer.accelerator={run.accelerator}",
        "trainer.devices=1",
        "trainer.precision=32-true",
        "+trainer.limit_test_batches=1",
        "+trainer.logger=false",
        "+trainer.enable_checkpointing=false",
        "+trainer.enable_progress_bar=false",
        "datamodule.batch_size=2",
        "datamodule.num_workers=0",
        "+datamodule.drop_last=false",
        f"hydra.run.dir={tmp_path / 'run'}",
        *run.benchmark_overrides,
    ]

    with initialize_config_dir(config_dir=str(CONFIGS_DIR), version_base=None):
        cfg = compose(config_name="test", overrides=overrides, return_hydra_config=True)
        HydraConfig.instance().set_config(cfg)

    model = instantiate(cfg.model)
    # Only the metric callbacks: model_checkpoint/lr_monitor interpolate hydra runtime paths and are
    # irrelevant to an eval-only smoke run.
    callbacks = [instantiate(cfg.callbacks[key]) for key in cfg.callbacks if key.startswith("val_")]
    trainer: L.Trainer = instantiate(cfg.trainer, callbacks=callbacks, logger=False)
    dm = instantiate(cfg.datamodule)

    trainer.test(model=model, datamodule=dm)
    metric_keys = [key for key in trainer.callback_metrics if key.startswith("test/") and key != "test/loss"]
    assert metric_keys, f"no test metric logged for {run.experiment!r}: {sorted(trainer.callback_metrics)}"
