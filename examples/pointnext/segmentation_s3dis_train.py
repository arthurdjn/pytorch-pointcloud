"""Train PointNeXt (S / B / L / XL) or the openpoints PointNet++ on S3DIS, following openpoints' `s3dis` configs."""

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
from torch_pointcloud.datasets import S3DIS, RepeatDataset
from torch_pointcloud.datasets.s3dis import S3DIS_AREAS, S3DISArea
from torch_pointcloud.metrics import confusion_matrix, intersection_over_union
from torch_pointcloud.models import create_model
from torch_pointcloud.optim import CosineWarmupLR, param_groups
from torch_pointcloud.utils.data import DataKeys, PointCloudDataLoader
from torch_pointcloud.utils.random import seed_everything

CPU_COUNT = os.cpu_count()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = CPU_COUNT // 2 if CPU_COUNT is not None else 0
# Reference batch size per model.
MODELS = {"pointnext-sm": 32, "pointnext-base": 32, "pointnext-lg": 8, "pointnext-xl": 8, "pointnet2": 32}
NUM_CLASSES = 13
VOXEL_SIZE = 0.04
VOXEL_MAX = 24_000
LOOP = 30
POINT_KEYS = [DataKeys.POS, DataKeys.COLOR, DataKeys.SEGMENT]
# The released checkpoints order two class pairs differently from the dataset (chair / table, bookcase / sofa).
S3DIS_LABEL_ORDER = [0, 1, 2, 3, 4, 5, 6, 8, 7, 10, 9, 11, 12]
COLOR_MEAN = [0.5136457, 0.49523646, 0.44921124]
COLOR_STD = [0.18308958, 0.18415008, 0.19252081]

TRAIN_TRANSFORM = T.Compose(
    [
        T.KeepItems(keys=POINT_KEYS),
        T.Relabel(keys=DataKeys.SEGMENT, labels=S3DIS_LABEL_ORDER),
        T.Divide(keys=DataKeys.COLOR, divisor=255.0),  # colors in [0, 1]; the chromatic ops below are linear
        T.Shift(keys=DataKeys.POS, method="min"),
        T.Voxelize(  # `voxelize(mode=0)`: one random point per voxel keeps its own coordinates
            pos_key=DataKeys.POS,
            pos_reduce="first",
            size=VOXEL_SIZE,
            method="fnv",
            reduce="first",
            keys=[DataKeys.COLOR, DataKeys.SEGMENT],
            random_first=True,
        ),
        T.SphereCrop(  # `crop_pc`: the `voxel_max` points nearest to a random one
            pos_key=DataKeys.POS,
            radius=math.inf,
            max_nodes=VOXEL_MAX,
            keys=[DataKeys.COLOR, DataKeys.SEGMENT],
            center="random_point",
        ),
        T.ShufflePoint(keys=POINT_KEYS),
        T.Shift(keys=DataKeys.POS, method="min"),
        T.AxisMinOffset(keys=DataKeys.POS, axis=2, dst_keys="height"),  # the height of the un-augmented room
        T.RandomColorAutoContrast(keys=DataKeys.COLOR, blend=None, p=0.2),
        T.RandomScale(keys=DataKeys.POS, scale_range=(0.9, 1.1), anisotropic=True),
        T.Shift(keys=DataKeys.POS, method="centroid"),  # PointCloudXYZAlign
        T.Shift(keys=DataKeys.POS, method="min", axes=[2]),
        T.RandomJitter(keys=DataKeys.POS, sigma=0.005, clip=0.02),
        T.RandomColorDrop(keys=DataKeys.COLOR, fill=0.0, p=0.2),  # ChromaticDropGPU, before the normalization
        T.Normalize(keys=DataKeys.COLOR, mean=COLOR_MEAN, std=COLOR_STD),
        T.Cat(keys=[DataKeys.COLOR, "height"], dst_key=DataKeys.X),
    ]
)
VAL_TRANSFORM = T.Compose(
    [
        T.KeepItems(keys=POINT_KEYS),
        T.Relabel(keys=DataKeys.SEGMENT, labels=S3DIS_LABEL_ORDER),
        T.Divide(keys=DataKeys.COLOR, divisor=255.0),
        T.Shift(keys=DataKeys.POS, method="min"),
        T.Voxelize(
            pos_key=DataKeys.POS,
            pos_reduce="first",
            size=VOXEL_SIZE,
            method="fnv",
            reduce="first",
            keys=[DataKeys.COLOR, DataKeys.SEGMENT],
        ),
        T.AxisMinOffset(keys=DataKeys.POS, axis=2, dst_keys="height"),
        T.Shift(keys=DataKeys.POS, method="centroid"),
        T.Shift(keys=DataKeys.POS, method="min", axes=[2]),
        T.Normalize(keys=DataKeys.COLOR, mean=COLOR_MEAN, std=COLOR_STD),
        T.Cat(keys=[DataKeys.COLOR, "height"], dst_key=DataKeys.X),
    ]
)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    train_loader, val_loader = configure_dataloaders(args)
    model = configure_model(args)
    optimizer, scheduler = configure_optimizers(args, model)
    criterion = configure_criterion(args)

    num_train, num_val = len(train_loader.dataset), len(val_loader.dataset)  # type: ignore[arg-type]
    print(f"Training {args.model!r} on {num_train} room crops per epoch, validating on {num_val}.")
    output: Optional[Path] = Path(args.output) if args.output else None
    best_loss = float("inf")
    for epoch in range(args.epochs):
        loss = train_one_epoch(model, criterion, optimizer, train_loader, args.device, args.grad_clip_norm)
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
    parser = ArgumentParser(description="Train PointNeXt on S3DIS.")
    parser.add_argument("--model", default="pointnext-sm", choices=list(MODELS))
    parser.add_argument("--area", default=5, type=int, choices=range(1, 7), help="Held-out area.")
    parser.add_argument("--root", default=DATA_DIR, help="Dataset root directory.")
    parser.add_argument("--output", default=None, help="Directory for the checkpoints (default: none saved).")
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--loop", default=LOOP, type=int, help="Times each room is drawn per epoch.")
    parser.add_argument("--batch-size", default=None, type=int, help="Default: the reference batch size of the model.")
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--min-lr", default=1e-5, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--grad-clip-norm", default=10.0, type=float)
    parser.add_argument("--num-workers", default=NUM_WORKERS, type=int)
    parser.add_argument("--eval-every", default=1, type=int, help="Validate every this many epochs.")
    parser.add_argument("--limit-train-batches", default=None, type=int)
    parser.add_argument("--limit-val-batches", default=None, type=int)
    parser.add_argument(
        "--download", action="store_true", help="Download S3DIS if missing (requires accepting its terms)."
    )
    parser.add_argument("--force-process", action="store_true", help="Force re-processing the dataset.")
    args = parser.parse_args()
    if args.batch_size is None:
        args.batch_size = MODELS[args.model]
    return args


def configure_datasets(args: Namespace) -> tuple[Dataset, Dataset]:
    val_areas: List[S3DISArea] = [area for area in S3DIS_AREAS if area == f"Area_{args.area}"]
    train_areas: List[S3DISArea] = [area for area in S3DIS_AREAS if area not in val_areas]
    common: Dict[str, Any] = dict(
        root=args.root, download=args.download, force_process=args.force_process, num_workers=args.num_workers
    )

    # Each room is drawn `loop` times per epoch, as a new random crop every time.
    train_dataset: Dataset
    val_dataset: Dataset
    train_dataset = RepeatDataset(S3DIS(areas=train_areas, transform=TRAIN_TRANSFORM, **common), k=args.loop)
    val_dataset = S3DIS(areas=val_areas, transform=VAL_TRANSFORM, **common)

    if args.limit_train_batches is not None:
        train_dataset = Subset(
            train_dataset, range(min(args.limit_train_batches * args.batch_size, len(train_dataset)))
        )
    if args.limit_val_batches is not None:
        val_dataset = Subset(val_dataset, range(min(args.limit_val_batches * 1, len(val_dataset))))

    return train_dataset, val_dataset


def configure_dataloaders(args: Namespace) -> tuple[DataLoader, DataLoader]:
    train_dataset, val_dataset = configure_datasets(args)

    train_loader = PointCloudDataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=len(train_dataset) >= args.batch_size,  # type: ignore[arg-type]
        num_workers=args.num_workers,
    )
    val_loader = PointCloudDataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
    )

    return train_loader, val_loader


def configure_model(args: Namespace) -> nn.Module:
    name = f"{args.model}.s3dis-area{args.area}.openpoints"
    return create_model(name, task="semantic-segmentation", pretrained=False).to(args.device)


def configure_optimizers(args: Namespace, model: nn.Module) -> tuple[Optimizer, LRScheduler]:
    groups = param_groups(model, weight_decay=args.weight_decay, no_decay=True)
    optimizer = torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay)

    # Stepped once per epoch.
    scheduler = CosineWarmupLR(optimizer, total_steps=args.epochs, lr_min=args.min_lr)
    return optimizer, scheduler


def configure_criterion(args: Namespace) -> nn.Module:
    return nn.CrossEntropyLoss(label_smoothing=0.2)


def train_one_epoch(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: Optimizer,
    loader: DataLoader,
    device: str,
    grad_clip_norm: float,
) -> float:
    model.train()
    total_loss = 0.0
    for data in tqdm(loader, desc="Training", leave=False):
        data = {key: value.to(device) for key, value in data.items() if isinstance(value, Tensor)}
        logits = model(data[DataKeys.X], data[DataKeys.POS], data[DataKeys.BATCH])
        loss = criterion(logits, data[DataKeys.SEGMENT])

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> dict[str, float]:
    model.eval()
    confusion = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long, device=device)
    for data in tqdm(loader, desc="Evaluating", leave=False):
        data = {key: value.to(device) for key, value in data.items() if isinstance(value, Tensor)}
        logits = model(data[DataKeys.X], data[DataKeys.POS], data[DataKeys.BATCH])
        confusion += confusion_matrix(logits.argmax(dim=1), data[DataKeys.SEGMENT], NUM_CLASSES, ignore_index=-1)

    present = confusion.sum(dim=1) > 0
    accuracy = confusion.diag().sum() / confusion.sum()
    mean_accuracy = (confusion.diag() / confusion.sum(dim=1).clamp_min(1))[present].mean()
    return {"mIoU": float(intersection_over_union(confusion)), "OA": accuracy.item(), "mAcc": mean_accuracy.item()}


if __name__ == "__main__":
    main()
