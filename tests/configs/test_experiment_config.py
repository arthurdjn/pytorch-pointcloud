from pathlib import Path
from typing import Iterator

import pytest

from torch_pointcloud.models import list_models
from torch_pointcloud.utils.imports import _HYDRA_AVAILABLE

# Hydra is an optional dev dependency, so gate the whole module on it being
# importable. The `hydra` / `omegaconf` imports live inside fixtures and tests.
pytestmark = pytest.mark.skipif(not _HYDRA_AVAILABLE, reason="hydra-core is not installed")

CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"
_EXPERIMENT_DIR = CONFIGS_DIR / "experiment"
# Experiments live in per-model subdirectories (`dgcnn/dgcnn_shapenetpart`, ...), so recurse and
# keep the group-relative path that `experiment=` expects, not just the file stem.
EXPERIMENTS = sorted(p.relative_to(_EXPERIMENT_DIR).with_suffix("").as_posix() for p in _EXPERIMENT_DIR.rglob("*.yaml"))


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
        for key in ("paths", "trainer", "callbacks", "logger", "data", "model", "task_name", "run_id"):
            assert key in cfg, f"experiment {experiment!r} is missing `{key}` after compose"
        # Required nested targets.
        assert cfg.trainer._target_ == "lightning.Trainer"
        assert "_target_" in cfg.model
        assert "_target_" in cfg.data


# The LightningModule `_target_` implies the registry task it builds its model under.
_TARGET_TO_TASK = {
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
