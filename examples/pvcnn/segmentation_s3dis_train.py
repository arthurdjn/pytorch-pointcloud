"""Train PVCNN or PVCNN++ on S3DIS following the pvcnn `configs/s3dis` configs."""

import os
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Optional

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

import torch_pointcloud.transforms as T
from torch_pointcloud.config import DATA_DIR
from torch_pointcloud.datasets import S3DIS
from torch_pointcloud.datasets.s3dis import S3DIS_AREAS, S3DISArea
from torch_pointcloud.metrics import confusion_matrix, intersection_over_union
from torch_pointcloud.models import create_model
from torch_pointcloud.utils.data import DataKeys, PointCloudDataLoader
from torch_pointcloud.utils.random import seed_everything

CPU_COUNT = os.cpu_count()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = CPU_COUNT // 2 if CPU_COUNT is not None else 0
# Points per block of each model.
MODELS = {"pvcnn.s3dis-area5.mit-han-lab": 4096, "pvcnn2.s3dis-area5": 8192}
NUM_CLASSES = 13
BLOCK_SIZE = 1.5
BLOCK_STRIDE = 0.75
POINT_KEYS = [DataKeys.POS, DataKeys.COLOR, DataKeys.SEGMENT, DataKeys.NORM_POS]


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    train_loader, val_loader = configure_dataloaders(args)
    model = configure_model(args)
    optimizer, scheduler = configure_optimizers(args, model)
    criterion = configure_criterion(args)

    num_train, num_val = len(train_loader.dataset), len(val_loader.dataset)  # type: ignore[arg-type]
    print(f"Training {args.model!r} on {num_train} blocks, validating on {num_val}.")
    output: Optional[Path] = Path(args.output) if args.output else None
    best_loss = float("inf")
    for epoch in range(args.epochs):
        loss = train_one_epoch(model, criterion, optimizer, train_loader, args.device)
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        print(f"Epoch {epoch + 1}/{args.epochs}  lr={lr:.2e}  train/loss {loss:.4f}")

        if (epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs:
            val = evaluate(model, val_loader, args.device)
            print("  " + " | ".join(f"val/{key} {value * 100:.2f}" for key, value in val.items()))

        if output is not None:
            output.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), output / "last.pt")
            if loss < best_loss:
                best_loss = loss
                torch.save(model.state_dict(), output / "best.pt")


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Train PVCNN on S3DIS with the reference recipe.")
    parser.add_argument("--model", default=next(iter(MODELS)), choices=list(MODELS))
    parser.add_argument("--area", default=5, type=int, choices=range(1, 7), help="Held-out area.")
    parser.add_argument("--root", default=DATA_DIR, help="Dataset root directory.")
    parser.add_argument("--output", default=None, help="Directory for the checkpoints (default: none saved).")
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--val-batch-size", default=10, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--weight-decay", default=1e-5, type=float)
    parser.add_argument("--num-workers", default=NUM_WORKERS, type=int)
    parser.add_argument("--eval-every", default=1, type=int, help="Validate every this many epochs.")
    parser.add_argument("--limit-train-batches", default=None, type=int)
    parser.add_argument("--limit-val-batches", default=None, type=int)
    parser.add_argument(
        "--download", action="store_true", help="Download S3DIS if missing (requires accepting its terms)."
    )
    parser.add_argument("--force-process", action="store_true", help="Force re-processing the dataset.")
    return parser.parse_args()


def configure_train_transform(args: Namespace) -> T.Compose:
    """The block features: xy centered on the block, z, rgb / 255, xyz / room extent."""
    return T.Compose(
        [
            T.KeepItems(keys=[DataKeys.POS, DataKeys.COLOR, DataKeys.SEGMENT, DataKeys.SCENE_MAX]),
            T.CopyItems(keys=DataKeys.POS, dst_keys=DataKeys.NORM_POS),
            T.DivideItems(keys=DataKeys.NORM_POS, div_keys=DataKeys.SCENE_MAX),  # xyz / room extent
            T.RandomSample(keys=POINT_KEYS, num_samples=MODELS[args.model]),
            T.Shift(keys=DataKeys.POS, method="min", axes=[0, 1]),
            T.Translate(keys=DataKeys.POS, offset=(-BLOCK_SIZE / 2, -BLOCK_SIZE / 2, 0.0)),  # x - (min x + size / 2)
            T.ToFloat(keys=DataKeys.COLOR),
            T.Divide(keys=DataKeys.COLOR, divisor=255.0),
            T.Cat(keys=[DataKeys.POS, DataKeys.COLOR, DataKeys.NORM_POS], dst_key=DataKeys.X, dim=1),
        ]
    )


def configure_val_transform(args: Namespace) -> T.Compose:
    # The validation points are drawn from a fixed seed, so every validation scores the same points.
    return T.Compose(
        [
            T.KeepItems(keys=[DataKeys.POS, DataKeys.COLOR, DataKeys.SEGMENT, DataKeys.SCENE_MAX]),
            T.CopyItems(keys=DataKeys.POS, dst_keys=DataKeys.NORM_POS),
            T.DivideItems(keys=DataKeys.NORM_POS, div_keys=DataKeys.SCENE_MAX),  # xyz / room extent
            T.RandomSample(keys=POINT_KEYS, num_samples=MODELS[args.model], seed=0),
            T.Shift(keys=DataKeys.POS, method="min", axes=[0, 1]),
            T.Translate(keys=DataKeys.POS, offset=(-BLOCK_SIZE / 2, -BLOCK_SIZE / 2, 0.0)),  # x - (min x + size / 2)
            T.ToFloat(keys=DataKeys.COLOR),
            T.Divide(keys=DataKeys.COLOR, divisor=255.0),
            T.Cat(keys=[DataKeys.POS, DataKeys.COLOR, DataKeys.NORM_POS], dst_key=DataKeys.X, dim=1),
        ]
    )


def configure_datasets(args: Namespace) -> tuple[Dataset, Dataset]:
    val_areas: list[S3DISArea] = [area for area in S3DIS_AREAS if area == f"Area_{args.area}"]
    train_areas: list[S3DISArea] = [area for area in S3DIS_AREAS if area not in val_areas]
    common: dict[str, Any] = dict(
        root=args.root,
        block_size=BLOCK_SIZE,
        block_stride=BLOCK_STRIDE,
        num_nodes=None,  # every point of a block; the transform draws the model's point count
        min_num_nodes=1,
        download=args.download,
        force_process=args.force_process,
        num_workers=args.num_workers,
    )

    train_dataset: Dataset
    val_dataset: Dataset
    train_transform = configure_train_transform(args)
    train_dataset = S3DIS(areas=train_areas, transform=train_transform, **common)
    val_transform = configure_val_transform(args)
    val_dataset = S3DIS(areas=val_areas, transform=val_transform, **common)

    if args.limit_train_batches is not None:
        train_dataset = Subset(
            train_dataset, range(min(args.limit_train_batches * args.batch_size, len(train_dataset)))
        )
    if args.limit_val_batches is not None:
        val_dataset = Subset(val_dataset, range(min(args.limit_val_batches * args.val_batch_size, len(val_dataset))))

    return train_dataset, val_dataset


def configure_dataloaders(args: Namespace) -> tuple[DataLoader, DataLoader]:
    train_dataset, val_dataset = configure_datasets(args)

    train_loader = PointCloudDataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = PointCloudDataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    return train_loader, val_loader


def configure_model(args: Namespace) -> nn.Module:
    if args.area != 5:
        raise SystemExit("The PVCNN registry entries are defined for Area 5 only.")
    return create_model(args.model, task="semantic-segmentation", pretrained=False).to(args.device)


def configure_optimizers(args: Namespace, model: nn.Module) -> tuple[Optimizer, LRScheduler]:
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Stepped once per epoch.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    return optimizer, scheduler


def configure_criterion(args: Namespace) -> nn.Module:
    return nn.CrossEntropyLoss()


def train_one_epoch(
    model: nn.Module, criterion: nn.Module, optimizer: Optimizer, loader: DataLoader, device: str
) -> float:
    model.train()
    total_loss = 0.0
    for data in tqdm(loader, desc="Training", leave=False):
        data = {key: value.to(device) if isinstance(value, Tensor) else value for key, value in data.items()}
        loss = criterion(model(data[DataKeys.X], data[DataKeys.POS], data[DataKeys.BATCH]), data[DataKeys.SEGMENT])

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> dict[str, float]:
    model.eval()
    confusion = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long, device=device)
    for data in tqdm(loader, desc="Evaluating", leave=False):
        data = {key: value.to(device) if isinstance(value, Tensor) else value for key, value in data.items()}
        preds = model(data[DataKeys.X], data[DataKeys.POS], data[DataKeys.BATCH]).argmax(dim=1)
        target = data[DataKeys.SEGMENT]
        confusion += confusion_matrix(preds, target, NUM_CLASSES, ignore_index=-1)

    per_class = confusion.diagonal().float() / confusion.sum(dim=1).clamp(min=1).float()
    present = confusion.sum(dim=1) > 0
    return {
        "mIoU": float(intersection_over_union(confusion)),
        "OA": float(confusion.diagonal().sum()) / float(confusion.sum().clamp(min=1)),
        "mAcc": float(per_class[present].mean()),
    }


if __name__ == "__main__":
    main()
