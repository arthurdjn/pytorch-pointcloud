"""Train PointNeXt-S (C=64) or the openpoints PointNet++ on ModelNet40 (HDF5 release), following openpoints'
`modelnet40ply2048` configs."""

import os
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

import torch_pointcloud.transforms as T
from torch_pointcloud.config import DATA_DIR
from torch_pointcloud.datasets import ModelNet40Hdf5
from torch_pointcloud.metrics import confusion_matrix
from torch_pointcloud.models import create_model
from torch_pointcloud.optim import CosineWarmupLR, param_groups
from torch_pointcloud.utils.data import DataKeys, PointCloudDataLoader
from torch_pointcloud.utils.random import seed_everything

CPU_COUNT = os.cpu_count()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = CPU_COUNT // 2 if CPU_COUNT is not None else 0
# Reference epochs per model.
MODELS = {"pointnext-sm-c64.modelnet40.openpoints": 600, "pointnet2.modelnet40.openpoints": 250}
NUM_CLASSES = 40
NUM_POINTS = 1024
LABEL_SMOOTHING = 0.2
POINT_KEYS = [DataKeys.POS, DataKeys.NORMAL]

TRAIN_TRANSFORM = T.Compose(
    [
        T.Slice(keys=POINT_KEYS, stop=NUM_POINTS),
        T.ShufflePoint(keys=POINT_KEYS),
        # PointCloudScaleAndTranslate(scale=[2/3, 3/2], shift=[0.2, 0.2, 0.2])
        T.RandomScale(keys=DataKeys.POS, scale_range=(2.0 / 3.0, 3.0 / 2.0), anisotropic=True),
        T.RandomTranslate(keys=DataKeys.POS, translation_range=(-0.2, 0.2)),
        T.CopyItems(keys=DataKeys.POS, dst_keys=DataKeys.X),
    ]
)
VAL_TRANSFORM = T.Compose(
    [
        T.Slice(keys=POINT_KEYS, stop=NUM_POINTS),
        T.CopyItems(keys=DataKeys.POS, dst_keys=DataKeys.X),
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
    print(f"Training {args.model!r} on {num_train} shapes, validating on {num_val}.")
    output: Optional[Path] = Path(args.output) if args.output else None
    best_loss = float("inf")
    for epoch in range(args.epochs):
        loss = train_one_epoch(model, criterion, optimizer, train_loader, args.device, args.grad_clip_norm)
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        print(f"Epoch {epoch + 1}/{args.epochs}  lr={lr:.2e}  train/loss {loss:.4f}")

        if (epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs:
            val = evaluate(model, val_loader, args.device)
            print(f"  val/OA {val['acc'] * 100:.2f} | val/mAcc {val['mean_acc'] * 100:.2f}")

        if output is not None:
            output.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), output / "last.pt")
            if loss < best_loss:
                best_loss = loss
                torch.save(model.state_dict(), output / "best.pt")


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Train PointNeXt on ModelNet40.")
    parser.add_argument("--model", default=next(iter(MODELS)), choices=list(MODELS))
    parser.add_argument("--root", default=DATA_DIR, help="Dataset root directory.")
    parser.add_argument("--output", default=None, help="Directory for the checkpoints (default: none saved).")
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--epochs", default=None, type=int, help="Default: the reference epochs of the model.")
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--val-batch-size", default=64, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--min-lr", default=None, type=float, help="Default: lr / 1000.")
    parser.add_argument("--weight-decay", default=0.05, type=float)
    parser.add_argument("--grad-clip-norm", default=1.0, type=float)
    parser.add_argument("--num-workers", default=NUM_WORKERS, type=int)
    parser.add_argument("--eval-every", default=1, type=int, help="Validate every this many epochs.")
    parser.add_argument("--limit-train-batches", default=None, type=int)
    parser.add_argument("--limit-val-batches", default=None, type=int)
    parser.add_argument("--download", action="store_true", help="Download the HDF5 release if missing.")
    args = parser.parse_args()
    if args.epochs is None:
        args.epochs = MODELS[args.model]
    if args.min_lr is None:
        args.min_lr = args.lr / 1000
    return args


def configure_datasets(args: Namespace) -> tuple[Dataset, Dataset]:
    train_dataset: Dataset
    val_dataset: Dataset
    train_dataset = ModelNet40Hdf5(root=args.root, train=True, transform=TRAIN_TRANSFORM, download=args.download)
    val_dataset = ModelNet40Hdf5(root=args.root, train=False, transform=VAL_TRANSFORM, download=args.download)

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
    return create_model(args.model, task="classification", pretrained=False).to(args.device)


def configure_optimizers(args: Namespace, model: nn.Module) -> tuple[Optimizer, LRScheduler]:
    groups = param_groups(model, weight_decay=args.weight_decay, no_decay=True)
    optimizer = torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay)

    # Stepped once per epoch.
    scheduler = CosineWarmupLR(optimizer, total_steps=args.epochs, lr_min=args.min_lr)
    return optimizer, scheduler


def configure_criterion(args: Namespace) -> nn.Module:
    # The checkpoints were trained with ε spread over the other classes: torch's label smoothing of ε C / (C - 1).
    return nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING * NUM_CLASSES / (NUM_CLASSES - 1))


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
        loss = criterion(logits, data[DataKeys.LABEL])

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
        confusion += confusion_matrix(logits.argmax(dim=1), data[DataKeys.LABEL], NUM_CLASSES)

    present = confusion.sum(dim=1) > 0
    accuracy = confusion.diag().sum() / confusion.sum()
    mean_accuracy = (confusion.diag() / confusion.sum(dim=1).clamp_min(1))[present].mean()
    return {"acc": accuracy.item(), "mean_acc": mean_accuracy.item()}


if __name__ == "__main__":
    main()
