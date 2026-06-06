"""Hydra entrypoint for reproducible point cloud training.

Composes a LightningModule, datamodule, callbacks, loggers, and `Trainer` from
the `configs/` tree, runs `Trainer.fit`, and (when `test=true`) `Trainer.test`
(falling back to the validation set when no test set is defined). Repo-only dev
tooling, not part of the installed `torch_pointcloud` package.

Usage:
    # train
    uv run --no-sync python train.py experiment=spunet/spunet_scannet
    # benchmark pretrained weights (no training): evaluate on the held-out set
    uv run --no-sync python train.py experiment=spunet/spunet_scannet train=false model.pretrained=true
    uv run --no-sync python train.py experiment=point_transformer_v3/point_transformer_v3_scannet \
        ckpt_path=logs/train/runs/foo/checkpoints/last.ckpt
"""

import hydra
import lightning as L
from dotenv import load_dotenv
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict

from torch_pointcloud.utils.hydra import instantiate_list
from torch_pointcloud.utils.random import seed_everything

# Load .env so the configs' `${oc.env:...}` interpolations resolve.
load_dotenv()

# `${eval:"<expr>"}` for inline arithmetic in YAML (e.g. compute total_steps from
# epoch count, scene count, batch size). Executes a Python expression at compose
# time, so only pass trusted configs.
OmegaConf.register_new_resolver("eval", eval, replace=True)


@hydra.main(config_path="configs", config_name="train", version_base=None)
def main(cfg: DictConfig) -> None:
    """Build the training objects from `cfg`, fit, and optionally test."""
    seed_everything(cfg.seed)
    # Benchmark mode (`train=false`): drop the training split so only the held-out (val) set is built.
    if not cfg.get("train", True):
        with open_dict(cfg):
            cfg.data.train_dataset = None
    model: L.LightningModule = instantiate(cfg.model)
    datamodule: L.LightningDataModule = instantiate(cfg.data)
    callbacks = instantiate_list(cfg.get("callbacks"))
    loggers = instantiate_list(cfg.get("logger"))
    trainer: L.Trainer = instantiate(cfg.trainer, callbacks=callbacks, logger=loggers)

    # Push the resolved config to every logger so each tracking UI shows it.
    cfg_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)
    for log in loggers:
        if hasattr(log, "log_hyperparams"):
            log.log_hyperparams(cfg_dict)

    if cfg.get("train", True):
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    if cfg.get("test", False):
        # After fit, test the best checkpoint Lightning tracked; otherwise use the user's ckpt_path.
        ckpt_path = "best" if cfg.get("train", True) and trainer.checkpoint_callback else cfg.get("ckpt_path")
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
