"""Hydra entrypoint: load a checkpoint and run `Trainer.test`.

Usage:
    uv run --no-sync python test.py \
        experiment=point_transformer_v3/point_transformer_v3_scannet \
        ckpt_path=logs/train/runs/point_transformer_v3_segmentation_scannet_2026-05-23_15-30-00/checkpoints/last.ckpt
"""

import hydra
import lightning as L
from dotenv import load_dotenv
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from torch_pointcloud.utils.hydra import instantiate_list
from torch_pointcloud.utils.random import seed_everything

load_dotenv()


@hydra.main(config_path="configs", config_name="test", version_base=None)
def main(cfg: DictConfig) -> None:
    """Build trainer + datamodule from `cfg`, then run `Trainer.test` from `ckpt_path`."""
    seed_everything(cfg.seed)

    model: L.LightningModule = instantiate(cfg.model)
    datamodule: L.LightningDataModule = instantiate(cfg.data)
    loggers = instantiate_list(cfg.get("logger"))
    trainer: L.Trainer = instantiate(cfg.trainer, logger=loggers)

    cfg_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)
    for log in loggers:
        if hasattr(log, "log_hyperparams"):
            log.log_hyperparams(cfg_dict)

    trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)


if __name__ == "__main__":
    main()
