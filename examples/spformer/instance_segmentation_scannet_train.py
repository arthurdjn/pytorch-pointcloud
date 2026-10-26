"""Train SPFormer on ScanNet following the authors' `spf_scannet.yaml` recipe."""

import math
import os
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

import torch_pointcloud.transforms as T
from torch_pointcloud.config import DATA_DIR
from torch_pointcloud.datasets import ScanNet20
from torch_pointcloud.losses import SPFormerLoss
from torch_pointcloud.metrics import instance_average_precision, instance_matches
from torch_pointcloud.metrics.instance_segmentation import InstanceMatches
from torch_pointcloud.models import SPFormerInstanceSegmentation, create_model, model_info
from torch_pointcloud.utils.data import DataKeys, PointCloudDataLoader
from torch_pointcloud.utils.random import seed_everything

CPU_COUNT = os.cpu_count()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = CPU_COUNT // 2 if CPU_COUNT is not None else 0
MODEL = "spformer.scannet.jiahao-sun"
NUM_CLASSES = 18
POINT_KEYS = [DataKeys.X, DataKeys.SEGMENT, DataKeys.INSTANCE, DataKeys.SUPERPOINT]

TRAIN_TRANSFORM = T.Compose(
    [
        T.Relabel(keys=DataKeys.SEGMENT, labels=range(3, 21), default=-1),
        # rgb += N(0, 0.1) per channel on [-1, 1] colors: a uniform shift of the same standard deviation.
        T.RandomColorShift(keys=DataKeys.COLOR, shift_range=(-22.08, 22.08), int_color=True),
        T.Normalize(keys=DataKeys.COLOR, mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5]),
        T.Shift(keys=DataKeys.POS, method="centroid"),
        T.RandomLinearJitter(keys=DataKeys.POS, sigma=0.1),
        T.RandomFlip(keys=DataKeys.POS, axes=(0,), p=0.5),
        T.RandomRotate(keys=DataKeys.POS, axis=2, angle_range=(-180.0, 180.0)),
        T.Cat(keys=[DataKeys.COLOR, DataKeys.POS], dst_key=DataKeys.X, dim=1),
        # Elastic distortion in 2 cm voxel units; the reference's noise grid spacing is twice its granularity.
        T.Scale(keys=DataKeys.POS, scale=50.0),
        T.RandomElasticDistortion(keys=DataKeys.POS, granularity=(12.0, 40.0), magnitude=(40.0, 160.0)),
        T.SphereCrop(
            pos_key=DataKeys.POS,
            radius=math.inf,
            max_nodes=250_000,
            keys=POINT_KEYS,
            center="random_point",
        ),
        T.Shift(keys=DataKeys.POS, method="min"),
        T.Voxelize(
            pos_key=DataKeys.POS,
            pos_reduce="grid",
            dst_pos_grid_key=DataKeys.POS_GRID,
            keys=[DataKeys.X],
            size=1.0,
            dst_inverse_key=DataKeys.INVERSE,
        ),
        T.KeepItems(keys=[*POINT_KEYS, DataKeys.POS, DataKeys.POS_GRID, DataKeys.INVERSE]),
    ]
)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    train_loader, val_loader = configure_dataloaders(args)
    model = configure_model(args)
    criterion = configure_criterion(args)
    optimizer, scheduler = configure_optimizers(args, model)

    num_train, num_val = len(train_loader.dataset), len(val_loader.dataset)  # type: ignore[arg-type]
    print(f"Training {MODEL!r} on {num_train} scenes, validating on {num_val}.")
    output: Optional[Path] = Path(args.output) if args.output else None
    for epoch in range(args.epochs):
        losses = train_one_epoch(model, criterion, optimizer, train_loader, args.device)
        scheduler.step()
        print(
            f"Epoch {epoch + 1}/{args.epochs}  lr={scheduler.get_last_lr()[0]:.2e}  "
            + "  ".join(f"train/{key} {value:.4f}" for key, value in losses.items())
        )

        if (epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs:
            val = evaluate(model, val_loader, args.device)
            print("  " + " | ".join(f"val/{key} {value * 100:.2f}" for key, value in val.items()))

        if output is not None:
            output.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), output / "last.pt")


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Train SPFormer instance segmentation on ScanNet.")
    parser.add_argument("--root", default=DATA_DIR, help="Dataset root directory.")
    parser.add_argument("--output", default=None, help="Directory for the checkpoints (default: none saved).")
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--seed", default=1999, type=int)
    parser.add_argument("--epochs", default=512, type=int)
    parser.add_argument("--batch-size", default=4, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--weight-decay", default=0.05, type=float)
    parser.add_argument("--num-workers", default=NUM_WORKERS, type=int)
    parser.add_argument("--eval-every", default=16, type=int, help="Validate every this many epochs.")
    parser.add_argument("--limit-train-batches", default=None, type=int)
    parser.add_argument("--limit-val-batches", default=None, type=int)
    return parser.parse_args()


def configure_datasets(args: Namespace) -> tuple[Dataset, Dataset]:
    common: Dict[str, Any] = dict(
        root=args.root, use_axis_alignment=False, return_superpoint=True, num_workers=args.num_workers
    )
    val_transform = model_info(MODEL, task="instance-segmentation")["transform"]

    train_dataset: Dataset
    val_dataset: Dataset
    train_dataset = ScanNet20(split="train", transform=TRAIN_TRANSFORM, **common)
    val_dataset = ScanNet20(split="val", transform=val_transform, **common)

    if args.limit_train_batches is not None:
        train_dataset = Subset(
            train_dataset, range(min(args.limit_train_batches * args.batch_size, len(train_dataset)))
        )
    if args.limit_val_batches is not None:
        val_dataset = Subset(val_dataset, range(min(args.limit_val_batches, len(val_dataset))))

    return train_dataset, val_dataset


def configure_dataloaders(args: Namespace) -> tuple[DataLoader, DataLoader]:
    train_dataset, val_dataset = configure_datasets(args)

    train_loader = PointCloudDataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=len(train_dataset) > args.batch_size,  # type: ignore[arg-type]
        num_workers=args.num_workers,
        cat_keys=[DataKeys.INVERSE],
    )
    val_loader = PointCloudDataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        cat_keys=[DataKeys.INVERSE],
    )

    return train_loader, val_loader


def configure_model(args: Namespace) -> SPFormerInstanceSegmentation:
    model = create_model(MODEL, task="instance-segmentation", pretrained=False)
    assert isinstance(model, SPFormerInstanceSegmentation)
    return model.to(args.device)


def configure_optimizers(args: Namespace, model: nn.Module) -> tuple[Optimizer, LRScheduler]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Stepped per epoch, decaying to zero at the last one.
    scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=args.epochs, power=0.9)
    return optimizer, scheduler


def configure_criterion(args: Namespace) -> nn.Module:
    return SPFormerLoss(num_classes=NUM_CLASSES).to(args.device)


def train_one_epoch(
    model: SPFormerInstanceSegmentation,
    criterion: nn.Module,
    optimizer: Optimizer,
    loader: DataLoader,
    device: str,
) -> Dict[str, float]:
    model.train()
    totals: Dict[str, float] = {}
    for data in tqdm(loader, desc="Training", leave=False):
        data = {key: value.to(device) if isinstance(value, Tensor) else value for key, value in data.items()}
        output = model(
            data[DataKeys.X],
            data[DataKeys.POS_GRID],
            data[DataKeys.BATCH],
            data[DataKeys.SUPERPOINT],
            data[DataKeys.INVERSE],
            data[DataKeys.BATCH_INVERSE],
        )
        losses = criterion(output, data)

        optimizer.zero_grad(set_to_none=True)
        losses["loss"].backward()
        optimizer.step()
        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + value.item()

    return {key: value / len(loader) for key, value in totals.items()}


@torch.no_grad()
def evaluate(model: SPFormerInstanceSegmentation, loader: DataLoader, device: str) -> Dict[str, float]:
    model.eval()
    matches: List[InstanceMatches] = []
    for data in tqdm(loader, desc="Evaluating", leave=False):
        data = {key: value.to(device) if isinstance(value, Tensor) else value for key, value in data.items()}
        output = model(
            data[DataKeys.X],
            data[DataKeys.POS_GRID],
            data[DataKeys.BATCH],
            data[DataKeys.SUPERPOINT],
            data[DataKeys.INVERSE],
            data[DataKeys.BATCH_INVERSE],
        )
        instances = model.decode(output)
        masks, labels, scores = instances["masks"], instances["labels"], instances["scores"]
        keep = (scores > 0) & (masks.sum(dim=1) > 100)
        match = instance_matches(
            masks[keep],
            labels[keep],
            scores[keep],
            data[DataKeys.ORIGIN_INSTANCE],
            data[DataKeys.ORIGIN_SEGMENT],
            batch=data[DataKeys.BATCH_INVERSE],
        )
        matches.append(match)

    return {
        "mAP": instance_average_precision(matches, num_classes=NUM_CLASSES),
        "mAP@0.5": instance_average_precision(matches, iou_threshold=0.5, num_classes=NUM_CLASSES),
        "mAP@0.25": instance_average_precision(matches, iou_threshold=0.25, num_classes=NUM_CLASSES),
    }


if __name__ == "__main__":
    main()
