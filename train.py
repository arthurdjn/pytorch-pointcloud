"""Hydra entrypoint for reproducible point cloud training.

Composes a LightningModule, datamodule, callbacks, loggers, and `Trainer` from the `configs/` tree,
runs `Trainer.fit`, and (when `test=true`) `Trainer.test` on the best checkpoint (falling back to the
validation set when no test set is defined). Evaluation of pretrained weights or an existing
checkpoint lives in `test.py`. Repo-only dev tooling, not part of the installed `torch_pointcloud`
package.

Usage:
    # train from scratch, then test the best checkpoint
    uv run --no-sync python train.py experiment=spunet/scannet
    # resume an interrupted fit
    uv run --no-sync python train.py experiment=spunet/scannet ckpt_path=logs/.../last.ckpt
"""

import logging

import hydra
import lightning as L
from dotenv import load_dotenv
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

# Load .env before the torch_pointcloud imports: the configs' `${oc.env:...}` interpolations and the
# library's `TORCH_POINTCLOUD_*` settings (read at import time) both resolve from the environment.
load_dotenv()

from torch_pointcloud.utils.hydra import instantiate_list  # noqa: E402
from torch_pointcloud.utils.random import seed_everything  # noqa: E402

log = logging.getLogger(__name__)

# `${eval:"<expr>"}` for inline arithmetic in YAML (e.g. compute total_steps from
# epoch count, scene count, batch size). Executes a Python expression at compose
# time, so only pass trusted configs.
OmegaConf.register_new_resolver("eval", eval, replace=True)


@hydra.main(config_path="configs", config_name="train", version_base=None)
def main(cfg: DictConfig) -> None:
    """Build the training objects from `cfg`, fit, and optionally test."""
    seed_everything(cfg.seed)

    model: L.LightningModule = instantiate(cfg.model)
    datamodule: L.LightningDataModule = instantiate(cfg.datamodule)
    callbacks = instantiate_list(cfg.get("callbacks"))
    loggers = instantiate_list(cfg.get("logger"))
    trainer: L.Trainer = instantiate(cfg.trainer, callbacks=callbacks, logger=loggers)

    # Push the resolved config to every logger so each tracking UI shows it.
    cfg_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)
    for backend in loggers:
        if hasattr(backend, "log_hyperparams"):
            backend.log_hyperparams(cfg_dict)

    trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    if cfg.get("test", False):
        ckpt_path = "best" if trainer.checkpoint_callback else None
        results = trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        # Lightning prints its metrics table to stdout only; mirror it through `logging` so the
        # run dir's ${task_name}.log records the final numbers.
        for metrics in results:
            for name, value in sorted(metrics.items()):
                log.info("%s: %s", name, value)


if __name__ == "__main__":
    main()
