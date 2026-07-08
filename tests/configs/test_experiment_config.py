from pathlib import Path
from typing import Dict, Iterator

import pytest

from torch_pointcloud.models import list_models
from torch_pointcloud.models._registry import Task
from torch_pointcloud.utils.imports import _HYDRA_AVAILABLE

# Hydra is an optional dev dependency, so gate the whole module on it being
# importable. The `hydra` / `omegaconf` imports live inside fixtures and tests.
pytestmark = pytest.mark.skipif(not _HYDRA_AVAILABLE, reason="hydra-core is not installed")

CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"
_EXPERIMENT_DIR = CONFIGS_DIR / "experiment"
# One explicit entry per experiment YAML (the group-relative path `experiment=` expects).
# `test_experiment_list_is_exhaustive` asserts this list matches the files on disk, so a new
# experiment cannot land untested; add it here when adding the YAML.
EXPERIMENTS = (
    "3detr/scannet",
    "3detr/sunrgbd",
    "concerto/scannet",
    "dgcnn/modelnet40",
    "dgcnn/modelnet40-2048",
    "dgcnn/s3dis",
    "dgcnn/scannet",
    "dgcnn/shapenetpart",
    "kpfcnn/s3dis",
    "lion/nuscenes",
    "octformer/modelnet40",
    "octformer/scannet",
    "octformer/scannet200",
    "point_bert/modelnet40",
    "point_bert/scanobjectnn-hardest",
    "point_bert/scanobjectnn-objbg",
    "point_bert/scanobjectnn-objonly",
    "point_m2ae/modelnet40",
    "point_m2ae/scanobjectnn-hardest",
    "point_m2ae/scanobjectnn-objbg",
    "point_m2ae/shapenetpart",
    "point_mae/modelnet40",
    "point_mae/scanobjectnn-hardest",
    "point_mae/scanobjectnn-objbg",
    "point_mae/scanobjectnn-objonly",
    "point_mae/shapenetpart",
    "point_mamba/modelnet40",
    "point_mamba/scanobjectnn",
    "point_mamba/scanobjectnn-augmentedrot-scale75",
    "point_mamba/scanobjectnn-nobg",
    "pointcnn/modelnet40",
    "pointcnn/shapenetpart",
    "pointconv/modelnet40",
    "pointgpt/modelnet40",
    "pointgpt/scanobjectnn-hardest",
    "pointgpt/scanobjectnn-objbg",
    "pointgpt/scanobjectnn-objonly",
    "pointmlp/modelnet40",
    "pointmlp/scanobjectnn",
    "pointnet/modelnet40",
    "pointnet/s3dis",
    "pointnet/shapenetpart",
    "pointnet2/modelnet40",
    "pointnet2/msg_modelnet40",
    "pointnet2/openpoints_modelnet40",
    "pointnet2/openpoints_s3dis",
    "pointnet2/openpoints_scanobjectnn",
    "pointnet2/yanx27_s3dis",
    "pointnext/modelnet40",
    "pointnext/s3dis",
    "pointnext/scanobjectnn",
    "pointnext/shapenetpart",
    "pointpillars/kitti",
    "pointpillars/nuscenes",
    "pointrcnn/kitti",
    "point_transformer/modelnet40",
    "point_transformer/s3dis",
    "point_transformer/scannet",
    "point_transformer_v2/scannet",
    "point_transformer_v2/scannet200",
    "point_transformer_v3/s3dis",
    "point_transformer_v3/scannet",
    "point_transformer_v3/scannet200",
    "point_transformer_v3/scannet_s3dis_joint",
    "pvcnn/s3dis",
    "pvcnn2/s3dis",
    "randlanet/semantickitti",
    "second/kitti",
    "second/nuscenes",
    "sonata/scannet",
    "sphereformer/semantickitti",
    "spunet/scannet",
    "spvcnn/semantickitti",
    "utonia/scannet",
    "votenet/sunrgbd",
    "voxelnext/nuscenes",
)


@pytest.fixture(autouse=True)
def _register_eval_resolver() -> None:
    """train.py registers this at startup; tests need it too."""
    from omegaconf import OmegaConf

    OmegaConf.register_new_resolver("eval", eval, replace=True)


@pytest.fixture(autouse=True)
def _clear_hydra() -> Iterator[None]:
    """Hydra refuses to initialize twice in the same process."""
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    yield
    GlobalHydra.instance().clear()


@pytest.fixture(autouse=True)
def _fake_data_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """`paths.data_dir = ${oc.env:TORCH_POINTCLOUD_DATA_DIR}` would otherwise fail to resolve."""
    monkeypatch.setenv("TORCH_POINTCLOUD_DATA_DIR", str(tmp_path))


def test_experiment_list_is_exhaustive() -> None:
    """The explicit `EXPERIMENTS` list covers exactly the YAML files on disk, so adding an experiment
    without registering it here (or deleting one without removing it) fails loudly."""
    on_disk = {p.relative_to(_EXPERIMENT_DIR).with_suffix("").as_posix() for p in _EXPERIMENT_DIR.rglob("*.yaml")}
    assert on_disk == set(EXPERIMENTS)


@pytest.mark.parametrize("experiment", EXPERIMENTS)
def test_experiment_composes(experiment: str) -> None:
    """`hydra.compose(config_name='train', overrides=[f'experiment={exp}'])` succeeds."""
    from hydra import compose, initialize_config_dir
    from hydra.core.hydra_config import HydraConfig

    with initialize_config_dir(config_dir=str(CONFIGS_DIR), version_base=None):
        cfg = compose(
            config_name="train",
            overrides=[f"experiment={experiment}"],
            return_hydra_config=True,
        )
        HydraConfig.instance().set_config(cfg)
        # Required top-level shape.
        for key in ("paths", "trainer", "callbacks", "logger", "datamodule", "model", "task_name", "run_id"):
            assert key in cfg, f"experiment {experiment!r} is missing `{key}` after compose"
        # Required nested targets.
        assert cfg.trainer._target_ == "lightning.Trainer"
        assert "_target_" in cfg.model
        assert "_target_" in cfg.datamodule
        # Training runs `Trainer.test` on the best checkpoint afterwards.
        assert cfg.test is True


@pytest.mark.parametrize("experiment", EXPERIMENTS)
def test_experiment_composes_for_test_entrypoint(experiment: str) -> None:
    """Every experiment also composes under `test.py`'s config, and the train split exists for
    `test.py` to null out."""
    from hydra import compose, initialize_config_dir
    from hydra.core.hydra_config import HydraConfig

    with initialize_config_dir(config_dir=str(CONFIGS_DIR), version_base=None):
        cfg = compose(
            config_name="test",
            overrides=[f"experiment={experiment}"],
            return_hydra_config=True,
        )
        HydraConfig.instance().set_config(cfg)
        assert cfg.task_name == "test"
        assert cfg.ckpt_path is None
        assert "train_dataset" in cfg.datamodule


def test_inferer_group_composes_into_model() -> None:
    """The `inferer` config group packages into `model.inferer` on top of any segmentation experiment,
    and the TTA variant keeps its pass transforms mandatory."""
    from hydra import compose, initialize_config_dir
    from hydra.core.hydra_config import HydraConfig
    from omegaconf import MissingMandatoryValue, OmegaConf

    with initialize_config_dir(config_dir=str(CONFIGS_DIR), version_base=None):
        cfg = compose(
            config_name="train",
            overrides=["experiment=pvcnn/s3dis", "inferer=sliding_window", "model.inferer.block_size=2.0"],
            return_hydra_config=True,
        )
        HydraConfig.instance().set_config(cfg)
        assert cfg.model.inferer._target_ == "torch_pointcloud.inferers.SlidingWindowInferer"
        assert cfg.model.inferer.block_size == 2.0

    with initialize_config_dir(config_dir=str(CONFIGS_DIR), version_base=None):
        cfg = compose(
            config_name="train",
            overrides=["experiment=pvcnn/s3dis", "inferer=tta"],
            return_hydra_config=True,
        )
        HydraConfig.instance().set_config(cfg)
        assert cfg.model.inferer._target_ == "torch_pointcloud.inferers.TTAInferer"
        with pytest.raises(MissingMandatoryValue):
            OmegaConf.to_container(cfg.model.inferer, resolve=True, throw_on_missing=True)


# The LightningModule `_target_` implies the registry task it builds its model under.
_TARGET_TO_TASK: Dict[str, Task] = {
    "torch_pointcloud.lightning.LitClassificationModel": "classification",
    "torch_pointcloud.lightning.LitSegmentationModel": "segmentation",
    "torch_pointcloud.lightning.LitDetectionModel": "detection",
}


@pytest.mark.parametrize("experiment", EXPERIMENTS)
def test_experiment_model_name_registered(experiment: str) -> None:
    """The composed `model.name` is a real registry entry for the LightningModule's task. This catches a
    renamed or typo'd model in any experiment without instantiating it, so it covers GPU-only configs too."""
    from hydra import compose, initialize_config_dir
    from hydra.core.hydra_config import HydraConfig

    with initialize_config_dir(config_dir=str(CONFIGS_DIR), version_base=None):
        cfg = compose(
            config_name="train",
            overrides=[f"experiment={experiment}"],
            return_hydra_config=True,
        )
        HydraConfig.instance().set_config(cfg)
        name = cfg.model.name
        task = _TARGET_TO_TASK[cfg.model._target_]
        assert name in list_models(task=task), (
            f"experiment {experiment!r}: model {name!r} not registered for task {task!r}"
        )


@pytest.mark.parametrize("experiment", EXPERIMENTS)
def test_experiment_metric_callbacks_interpolate(experiment: str) -> None:
    """The metric callbacks resolve their `${model.num_classes}` interpolation to a real class count
    (and the per-class-IoU AP metric gets its mandatory `iou_per_class`)."""
    from hydra import compose, initialize_config_dir
    from hydra.core.hydra_config import HydraConfig

    with initialize_config_dir(config_dir=str(CONFIGS_DIR), version_base=None):
        cfg = compose(
            config_name="train",
            overrides=[f"experiment={experiment}"],
            return_hydra_config=True,
        )
        HydraConfig.instance().set_config(cfg)
        for callback in ("val_miou", "val_accuracy"):
            if callback in cfg.callbacks:
                num_classes = cfg.callbacks[callback].metric.num_classes
                assert isinstance(num_classes, int) and num_classes > 0
                assert num_classes == cfg.model.num_classes
        if "val_ap" in cfg.callbacks:
            iou_per_class = cfg.callbacks.val_ap.metric.iou_per_class
            assert len(iou_per_class) > 0


@pytest.mark.parametrize("experiment", EXPERIMENTS)
def test_experiment_resolves_user_config(experiment: str) -> None:
    """Every user-side interpolation (paths, run_id, model targets, etc.) resolves
    without `MissingMandatoryValue`. The `hydra.*` namespace is excluded because
    sweep-only fields like `hydra.job.num` are legitimately missing outside a sweep."""
    from hydra import compose, initialize_config_dir
    from hydra.core.hydra_config import HydraConfig
    from omegaconf import OmegaConf

    with initialize_config_dir(config_dir=str(CONFIGS_DIR), version_base=None):
        cfg = compose(
            config_name="train",
            overrides=[f"experiment={experiment}"],
            return_hydra_config=True,
        )
        HydraConfig.instance().set_config(cfg)
        # Resolve only the user-facing tree (without `hydra.*`).
        user_cfg = OmegaConf.create({k: cfg[k] for k in cfg if k != "hydra"})
        resolved = OmegaConf.to_container(user_cfg, resolve=True, throw_on_missing=True)
        assert isinstance(resolved, dict)
        assert resolved["run_id"].startswith(resolved["run_name"]), (
            f"run_id should embed run_name; got run_id={resolved['run_id']!r} run_name={resolved['run_name']!r}"
        )


@pytest.mark.parametrize("experiment", EXPERIMENTS)
def test_experiment_run_dir_under_task_runs(experiment: str) -> None:
    """`hydra.run.dir` lands in the expected `logs/<task>/runs/<run_name>/<timestamp>/` layout."""
    from hydra import compose, initialize_config_dir
    from hydra.core.hydra_config import HydraConfig

    with initialize_config_dir(config_dir=str(CONFIGS_DIR), version_base=None):
        cfg = compose(
            config_name="train",
            overrides=[f"experiment={experiment}", "run_name=under_test"],
            return_hydra_config=True,
        )
        HydraConfig.instance().set_config(cfg)
        run_dir = Path(cfg.hydra.run.dir)
        # logs/${task_name}/runs/${run_name}/${timestamp}
        assert run_dir.parts[-3:][0] == "runs"
        assert run_dir.parts[-3:][1] == "under_test"
        # timestamp is the trailing dir
        assert run_dir.parts[-3:][2] == cfg.timestamp
