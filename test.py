"""Hydra entrypoint for reproducible point cloud evaluation.

Composes the same experiment configs as `train.py` but runs `Trainer.test` only, on the held-out
split (the validation set when no test set is defined). Weights come from the registry
(`model.pretrained=true` by default) to reproduce the documented benchmark numbers, or from
`ckpt_path=...` to evaluate your own training run (an explicit checkpoint takes precedence over the
registry weights). Evaluation runs in full precision with TF32 pinned off, matching how the reference
numbers were measured. Repo-only dev tooling, not part of the installed `torch_pointcloud` package.

Usage:
    # reproduce the documented benchmark number (registry pretrained weights)
    uv run --no-sync python test.py experiment=spunet/scannet
    # evaluate your own training run instead of the registry weights
    uv run --no-sync python test.py experiment=spunet/scannet ckpt_path=logs/.../last.ckpt
    # quick smoke
    uv run --no-sync python test.py experiment=spunet/scannet +trainer.limit_test_batches=5
"""

import logging
from pathlib import Path
from typing import List

import hydra
import lightning as L
from dotenv import load_dotenv
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict

# Load .env before the torch_pointcloud imports: the configs' `${oc.env:...}` interpolations and the
# library's `TORCH_POINTCLOUD_*` settings (read at import time) both resolve from the environment.
load_dotenv()

from torch_pointcloud import list_models  # noqa: E402
from torch_pointcloud.utils.hydra import instantiate_list  # noqa: E402
from torch_pointcloud.utils.random import seed_everything, set_determinism  # noqa: E402

log = logging.getLogger(__name__)

OmegaConf.register_new_resolver("eval", eval, replace=True)


def _pretrained_experiments() -> List[str]:
    """Experiment ids (e.g. `spunet/scannet`) whose `model.name` ships registry weights."""
    experiments_dir = Path(__file__).resolve().parent / "configs" / "experiment"
    pretrained = set(list_models(pretrained=True))
    names: List[str] = []
    for path in sorted(experiments_dir.rglob("*.yaml")):
        experiment = OmegaConf.load(path)
        model = experiment.get("model")
        if isinstance(model, DictConfig) and model.get("name") in pretrained:
            names.append(path.relative_to(experiments_dir).with_suffix("").as_posix())
    return names


@hydra.main(config_path="configs", config_name="test", version_base=None)
def main(cfg: DictConfig) -> None:
    """Build the eval objects from `cfg` and run `Trainer.test`."""
    seed_everything(cfg.seed)
    # Reference numbers are measured with TF32 off; the Trainer precision flag does not cover it.
    set_determinism(tf32=False)

    with open_dict(cfg):
        # Only the held-out split is evaluated.
        cfg.datamodule.train_dataset = None
        # An explicit checkpoint takes precedence over the registry weights (no double load, and
        # models without registry weights stay evaluable from a checkpoint).
        cfg.model.pretrained = not cfg.get("ckpt_path")
        # Experiments pin their own precision where the architecture requires it; the training
        # default (16-mixed) shifts benchmark numbers, so evaluation upgrades it to full precision.
        if cfg.trainer.get("precision") == "16-mixed":
            cfg.trainer.precision = "32-true"

    if cfg.model.pretrained and cfg.model.name not in list_models(pretrained=True):
        hint = "\n  ".join(_pretrained_experiments())
        raise SystemExit(
            f"Model {cfg.model.name!r} has no registered pretrained weights, so there is no benchmark to "
            "reproduce. Pass ckpt_path=... to evaluate your own checkpoint, or pick an experiment whose "
            f"model ships registry weights:\n  {hint}"
        )

    model: L.LightningModule = instantiate(cfg.model)
    datamodule: L.LightningDataModule = instantiate(cfg.datamodule)
    callbacks = instantiate_list(cfg.get("callbacks"))
    loggers = instantiate_list(cfg.get("logger"))
    trainer: L.Trainer = instantiate(cfg.trainer, callbacks=callbacks, logger=loggers)

    cfg_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)
    for backend in loggers:
        if hasattr(backend, "log_hyperparams"):
            backend.log_hyperparams(cfg_dict)

    results = trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))
    # Lightning prints its metrics table to stdout only; mirror it through `logging` so the
    # run dir's ${task_name}.log records the final numbers.
    for metrics in results:
        for name, value in sorted(metrics.items()):
            log.info("%s: %s", name, value)


if __name__ == "__main__":
    main()
