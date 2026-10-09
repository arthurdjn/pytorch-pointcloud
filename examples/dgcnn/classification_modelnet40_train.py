"""Train DGCNN on ModelNet40 (HDF5 release) at 1024 or 2048 points, following dgcnn.pytorch's `main_cls.py`."""

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
from torch_pointcloud.utils.data import DataKeys, PointCloudDataLoader
from torch_pointcloud.utils.random import seed_everything

CPU_COUNT = os.cpu_count()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = CPU_COUNT // 2 if CPU_COUNT is not None else 0
# Points per shape of each model.
MODELS = {"dgcnn.modelnet40-1024.an-tao": 1024, "dgcnn.modelnet40-2048.an-tao": 2048}
NUM_CLASSES = 40
POINT_KEYS = [DataKeys.POS, DataKeys.NORMAL]


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
        loss = train_one_epoch(model, criterion, optimizer, train_loader, args.device)
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
    parser = ArgumentParser(description="Train DGCNN on ModelNet40.")
    parser.add_argument("--model", default=next(iter(MODELS)), choices=list(MODELS))
    parser.add_argument("--root", default=DATA_DIR, help="Dataset root directory.")
    parser.add_argument("--output", default=None, help="Directory for the checkpoints (default: none saved).")
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--epochs", default=250, type=int)
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
    parser.add_argument("--download", action="store_true", help="Download the HDF5 release if missing.")
    return parser.parse_args()


def configure_train_transform(args: Namespace) -> T.Transform:
    return T.Compose(
        [
            T.Slice(keys=POINT_KEYS, stop=MODELS[args.model]),
            # translate_pointcloud: per-axis scale in [2/3, 3/2] and shift in [-0.2, 0.2]
            T.RandomScale(keys=DataKeys.POS, scale_range=(2.0 / 3.0, 3.0 / 2.0), anisotropic=True),
            T.RandomTranslate(keys=DataKeys.POS, translation_range=(-0.2, 0.2)),
            T.ShufflePoint(keys=POINT_KEYS),
        ]
    )


def configure_val_transform(args: Namespace) -> T.Transform:
    return T.Compose([T.Slice(keys=POINT_KEYS, stop=MODELS[args.model])])


def configure_datasets(args: Namespace) -> tuple[Dataset, Dataset]:
    train_dataset: Dataset
    val_dataset: Dataset
    train_transform = configure_train_transform(args)
    train_dataset = ModelNet40Hdf5(root=args.root, train=True, transform=train_transform, download=args.download)
    val_transform = configure_val_transform(args)
    val_dataset = ModelNet40Hdf5(root=args.root, train=False, transform=val_transform, download=args.download)

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
        logits = model(None, data[DataKeys.POS], data[DataKeys.BATCH])
        loss = criterion(logits, data[DataKeys.LABEL])

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
        data = {key: value.to(device) for key, value in data.items() if isinstance(value, Tensor)}
        logits = model(None, data[DataKeys.POS], data[DataKeys.BATCH])
        confusion += confusion_matrix(logits.argmax(dim=1), data[DataKeys.LABEL], NUM_CLASSES)

    present = confusion.sum(dim=1) > 0
    accuracy = confusion.diag().sum() / confusion.sum()
    mean_accuracy = (confusion.diag() / confusion.sum(dim=1).clamp_min(1))[present].mean()
    return {"acc": accuracy.item(), "mean_acc": mean_accuracy.item()}


if __name__ == "__main__":
    main()
