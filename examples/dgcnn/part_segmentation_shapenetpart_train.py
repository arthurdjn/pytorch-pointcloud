"""Train DGCNN on ShapeNetPart, following dgcnn.pytorch's `main_partseg.py`."""

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
from torch_pointcloud.datasets import ConcatDataset, ShapeNetPart
from torch_pointcloud.metrics import part_intersection_over_union, part_mean_intersection_over_union
from torch_pointcloud.models import create_model
from torch_pointcloud.utils.data import DataKeys, PointCloudDataLoader
from torch_pointcloud.utils.random import seed_everything

CPU_COUNT = os.cpu_count()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = CPU_COUNT // 2 if CPU_COUNT is not None else 0
MODEL = "dgcnn.shapenetpart.an-tao"
NUM_CLASSES = 50
NUM_CATEGORIES = 16
NUM_POINTS = 2048
POINT_KEYS = [DataKeys.POS, DataKeys.NORMAL, DataKeys.SEGMENT]

TRAIN_TRANSFORM = T.Compose(
    [
        T.Rescale(keys=DataKeys.POS, method="centroid"),
        T.FarthestPointSample(pos_key=DataKeys.POS, keys=POINT_KEYS, num_samples=NUM_POINTS),
        T.ShufflePoint(keys=POINT_KEYS),
        T.OneHot(keys=DataKeys.CATEGORY, num_classes=NUM_CATEGORIES),
    ]
)

VAL_TRANSFORM = T.Compose(
    [
        T.Rescale(keys=DataKeys.POS, method="centroid"),
        T.FarthestPointSample(pos_key=DataKeys.POS, keys=POINT_KEYS, num_samples=NUM_POINTS),
        T.OneHot(keys=DataKeys.CATEGORY, num_classes=NUM_CATEGORIES),
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
    print(f"Training {MODEL!r} on {num_train} shapes, validating on {num_val}.")
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
    parser = ArgumentParser(description="Train DGCNN on ShapeNetPart.")
    parser.add_argument("--root", default=DATA_DIR, help="Dataset root directory.")
    parser.add_argument("--output", default=None, help="Directory for the checkpoints (default: none saved).")
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--val-batch-size", default=16, type=int)
    parser.add_argument("--lr", default=0.1, type=float)
    parser.add_argument("--min-lr", default=1e-3, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--label-smoothing", default=0.2, type=float, help="Smoothing ε (0 disables it).")
    parser.add_argument("--num-workers", default=NUM_WORKERS, type=int)
    parser.add_argument("--eval-every", default=1, type=int, help="Validate every this many epochs.")
    parser.add_argument("--limit-train-batches", default=None, type=int)
    parser.add_argument("--limit-val-batches", default=None, type=int)
    parser.add_argument("--force-process", action="store_true", help="Force re-processing the dataset.")
    return parser.parse_args()


def configure_datasets(args: Namespace) -> tuple[Dataset, Dataset]:
    common: Dict[str, Any] = dict(root=args.root, force_process=args.force_process, num_workers=args.num_workers)

    # Trained on the train + val splits, validated on the test split.
    train_dataset: Dataset
    val_dataset: Dataset
    train_dataset = ConcatDataset(
        [
            ShapeNetPart(split="train", transform=TRAIN_TRANSFORM, **common),
            ShapeNetPart(split="val", transform=TRAIN_TRANSFORM, **common),
        ]
    )
    val_dataset = ShapeNetPart(split="test", transform=VAL_TRANSFORM, **common)

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
        drop_last=len(train_dataset) >= args.batch_size,  # type: ignore[arg-type]
        num_workers=args.num_workers,
    )
    val_loader = PointCloudDataLoader(
        val_dataset, batch_size=args.val_batch_size, shuffle=False, num_workers=args.num_workers
    )

    return train_loader, val_loader


def configure_model(args: Namespace) -> nn.Module:
    return create_model(MODEL, task="part-segmentation", pretrained=False).to(args.device)


def configure_optimizers(args: Namespace, model: nn.Module) -> tuple[Optimizer, LRScheduler]:
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    # Stepped once per epoch.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    return optimizer, scheduler


def configure_criterion(args: Namespace) -> nn.Module:
    # `1 - ε` on the target class and `ε / (C - 1)` elsewhere is torch's `label_smoothing = ε C / (C - 1)`.
    return nn.CrossEntropyLoss(label_smoothing=args.label_smoothing * NUM_CLASSES / (NUM_CLASSES - 1))


def train_one_epoch(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: Optimizer,
    loader: DataLoader,
    device: str,
) -> float:
    model.train()
    total_loss = 0.0
    for data in tqdm(loader, desc="Training", leave=False):
        data = {key: value.to(device) for key, value in data.items() if isinstance(value, Tensor)}
        logits = model(None, data[DataKeys.POS], data[DataKeys.BATCH], data[DataKeys.CATEGORY])
        loss = criterion(logits, data[DataKeys.SEGMENT])

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> dict[str, float]:
    """Instance and class mIoU of the predicted parts."""
    model.eval()
    part_ids = list(ShapeNetPart.seg_ids.values())
    ious: List[Tensor] = []
    categories: List[Tensor] = []
    for data in tqdm(loader, desc="Evaluating", leave=False):
        data = {key: value.to(device) for key, value in data.items() if isinstance(value, Tensor)}
        logits = model(None, data[DataKeys.POS], data[DataKeys.BATCH], data[DataKeys.CATEGORY])
        preds, category = logits.argmax(dim=1).cpu(), data[DataKeys.CATEGORY].argmax(dim=1).cpu()
        ious.append(
            part_intersection_over_union(
                preds, data[DataKeys.SEGMENT].cpu(), part_ids, category, data[DataKeys.BATCH].cpu()
            )
        )
        categories.append(category)

    shape_ious, category = torch.cat(ious), torch.cat(categories)
    return {
        "ins_mIoU": float(part_mean_intersection_over_union(shape_ious, category)),
        "cls_mIoU": float(part_mean_intersection_over_union(shape_ious, category, average="macro")),
    }


if __name__ == "__main__":
    main()
