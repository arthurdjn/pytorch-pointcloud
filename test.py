"""Hydra entrypoint: load a checkpoint and run `Trainer.test`.

Usage:
    uv run --no-sync python test.py \
        experiment=scannet_ptv3 \
        ckpt_path=logs/train/runs/scannet_ptv3_2026-05-23_15-30-00/checkpoints/last.ckpt
"""

from typing import Any, List

import hydra
import lightning as L
from dotenv import load_dotenv
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from torch_pointcloud.utils.random import seed_everything

load_dotenv()


def _instantiate_group(cfg: Any) -> List[Any]:
    """Instantiate every entry of a config group into a list."""
    if not cfg:
        return []
    return [instantiate(node) for node in cfg.values()]


@hydra.main(config_path="configs", config_name="test", version_base=None)
def main(cfg: DictConfig) -> None:
    """Build trainer + datamodule from `cfg`, then run `Trainer.test` from `ckpt_path`."""
    seed_everything(cfg.seed)
    model: L.LightningModule = instantiate(cfg.model)
    datamodule: L.LightningDataModule = instantiate(cfg.data)
    loggers = _instantiate_group(cfg.get("logger"))
    trainer: L.Trainer = instantiate(cfg.trainer, logger=loggers)

    cfg_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)
    for log in loggers:
        if hasattr(log, "log_hyperparams"):
            log.log_hyperparams(cfg_dict)

    trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)


if __name__ == "__main__":
    main()
