r"""VoteNet training on SUN RGB-D.

Usage:
    uv run --no-sync python examples/votenet_train_sunrgbd.py --root /path/to/data
"""

import os
from argparse import ArgumentParser, Namespace
from functools import partial

import lightning.pytorch as L
import torch

import torch_pointcloud.transforms as T
from torch_pointcloud.config import DATA_DIR
from torch_pointcloud.datasets import SunRGBD
from torch_pointcloud.lightning import LitDetectionModel
from torch_pointcloud.losses import VoteNetLoss
from torch_pointcloud.models import VoteNetDetection, create_model
from torch_pointcloud.utils.data import DataKeys, PointCloudDataLoader

CPU_COUNT = os.cpu_count()
NUM_WORKERS = CPU_COUNT // 2 if CPU_COUNT is not None else 0
TARGET_KEYS = [
    "center_label",
    "heading_class_label",
    "heading_residual_label",
    "size_class_label",
    "size_residual_label",
    "sem_cls_label",
    "box_label_mask",
    "vote_label",
    "vote_label_mask",
]


def main() -> None:
    args = parse_args()
    L.seed_everything(args.seed)

    model = create_model("votenet.sunrgbd.fair", task="detection")
    assert isinstance(model, VoteNetDetection)
    dataset = SunRGBD(
        root=args.root,
        split="train",
        transform=T.Compose(
            [
                T.AxisMinOffset(keys="pos", axis=2, quantile=0.0099, dst_keys="height"),
                T.RandomSample(keys=["pos", "height"], num_samples=20000),
                T.GenerateVoteLabels(pos_key="pos", box_key=DataKeys.BOX),
                T.RandomFlip(keys=("pos", "vote_label"), box_key=DataKeys.BOX, axes=(0,)),
                T.RandomRotate(keys=("pos", "vote_label"), box_key=DataKeys.BOX, angle_range=(-30.0, 30.0)),
                T.RandomScale(keys=("pos", "vote_label"), box_key=DataKeys.BOX, scale_range=(0.85, 1.15)),
                T.EncodeVoteNetTargets(
                    box_key=DataKeys.BOX,
                    num_heading_bin=model.num_heading_bin,
                    max_num_obj=64,
                    mean_sizes=model.mean_sizes,
                ),
                T.Cat(keys=["height"], dst_key="x", dim=1),
            ]
        ),
    )

    dataloader = PointCloudDataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        stack_keys=TARGET_KEYS,
    )

    lit_model = LitDetectionModel(
        name="votenet.sunrgbd.fair",
        optimizer=partial(torch.optim.Adam, lr=args.lr),
        criterion=VoteNetLoss,
    )

    trainer = L.Trainer(
        max_epochs=args.epochs,
        limit_train_batches=args.steps,
        enable_checkpointing=False,
        logger=False,
    )
    trainer.fit(lit_model, dataloader)


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Minimal VoteNet SUN RGB-D training example.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--root", type=str, default=DATA_DIR, help="Dataset root directory.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=None, help="Cap on train batches per epoch (default: full epoch).")
    parser.add_argument("--lr", type=float, default=1e-3)
    return parser.parse_args()


if __name__ == "__main__":
    main()
