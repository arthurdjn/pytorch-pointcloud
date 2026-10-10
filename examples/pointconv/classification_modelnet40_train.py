"""Train PointConv on ModelNet40 with normals, following pointconv_pytorch's `train_cls_conv.py --normal`."""

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
from torch_pointcloud.datasets import ModelNetNormalResampled
from torch_pointcloud.metrics import confusion_matrix
from torch_pointcloud.models import create_model, model_info
from torch_pointcloud.utils.data import DataKeys, PointCloudDataLoader
from torch_pointcloud.utils.random import seed_everything

CPU_COUNT = os.cpu_count()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = CPU_COUNT // 2 if CPU_COUNT is not None else 0
MODELS = ["pointconv-density-base.modelnet40.wenxuan-wu"]
NUM_CLASSES = 40
NUM_POINTS = 1024
POINT_KEYS = [DataKeys.POS, DataKeys.NORMAL]
SCALE_RANGE = (2.0 / 3.0, 3.0 / 2.0)
TRANSLATION_RANGE = (-0.2, 0.2)
MAX_DROPOUT_RATIO = 0.875
MOMENTUM = 0.9
LR_STEP_EPOCHS = 30
LR_GAMMA = 0.7

TRAIN_TRANSFORM = T.Compose(
    [
        T.Slice(keys=POINT_KEYS, stop=NUM_POINTS),
        T.Rescale(keys=DataKeys.POS),
        T.RandomScale(keys=DataKeys.POS, scale_range=SCALE_RANGE),
        T.RandomTranslate(keys=DataKeys.POS, translation_range=TRANSLATION_RANGE),
        # The reference overwrites the dropped points with copies of kept ones, so every shape keeps 1024 points.
        T.RandomDropout(keys=POINT_KEYS, drop_ratio_range=(0.0, MAX_DROPOUT_RATIO)),
        T.ShufflePoint(keys=POINT_KEYS),
        T.CopyItems(keys=DataKeys.NORMAL, dst_keys=DataKeys.X),
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
        loss = train_one_epoch(model, criterion, optimizer, train_loader, args.device)
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        print(f"Epoch {epoch + 1}/{args.epochs}  lr={lr:.2e}  train/loss {loss:.4f}")

        if (epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs:
            val = evaluate(model, val_loader, args.device)
            print(f"  val/OA {val['acc'] * 100:.2f} | val/mAcc {val['mean_acc'] * 100:.2f}")

        # The reference keeps the checkpoint of the best test accuracy after the fifth epoch.
        if output is not None:
            output.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), output / "last.pt")
            if loss < best_loss:
                best_loss = loss
                torch.save(model.state_dict(), output / "best.pt")


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Train PointConv on ModelNet40.")
    parser.add_argument("--model", default=MODELS[0], choices=MODELS)
    parser.add_argument("--root", default=DATA_DIR, help="Dataset root directory.")
    parser.add_argument("--output", default=None, help="Directory for the checkpoints (default: none saved).")
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--epochs", default=400, type=int)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--val-batch-size", default=32, type=int)
    parser.add_argument("--lr", default=1e-2, type=float)
    parser.add_argument("--weight-decay", default=0.0, type=float)
    parser.add_argument("--num-workers", default=NUM_WORKERS, type=int)
    parser.add_argument("--eval-every", default=1, type=int, help="Validate every this many epochs.")
    parser.add_argument("--limit-train-batches", default=None, type=int)
    parser.add_argument("--limit-val-batches", default=None, type=int)
    parser.add_argument("--download", action="store_true", help="Download the dataset if missing.")
    parser.add_argument("--force-process", action="store_true", help="Reprocess the dataset.")
    return parser.parse_args()


def configure_datasets(args: Namespace) -> tuple[Dataset, Dataset]:
    train_dataset: Dataset
    val_dataset: Dataset
    train_dataset = ModelNetNormalResampled(
        root=args.root,
        variant="40",
        train=True,
        transform=TRAIN_TRANSFORM,
        download=args.download,
        force_process=args.force_process,
    )
    # The reference validates on the first 1024 points of each shape; the benchmark protocol samples them farthest-first.
    val_dataset = ModelNetNormalResampled(
        root=args.root,
        variant="40",
        train=False,
        transform=model_info(args.model)["transform"],
        download=args.download,
        force_process=args.force_process,
    )

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
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = PointCloudDataLoader(
        val_dataset, batch_size=args.val_batch_size, shuffle=False, num_workers=args.num_workers
    )

    return train_loader, val_loader


def configure_model(args: Namespace) -> nn.Module:
    return create_model(args.model, task="classification", pretrained=False).to(args.device)


def configure_optimizers(args: Namespace, model: nn.Module) -> tuple[Optimizer, LRScheduler]:
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=MOMENTUM, weight_decay=args.weight_decay)

    # The reference steps the schedule at the start of every epoch; here it is stepped at the end.
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP_EPOCHS, gamma=LR_GAMMA)
    return optimizer, scheduler


def configure_criterion(args: Namespace) -> nn.Module:
    return nn.CrossEntropyLoss()


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
        logits = model(data[DataKeys.X], data[DataKeys.POS], data[DataKeys.BATCH])
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
        logits = model(data[DataKeys.X], data[DataKeys.POS], data[DataKeys.BATCH])
        confusion += confusion_matrix(logits.argmax(dim=1), data[DataKeys.LABEL], NUM_CLASSES)

    present = confusion.sum(dim=1) > 0
    accuracy = confusion.diag().sum() / confusion.sum()
    mean_accuracy = (confusion.diag() / confusion.sum(dim=1).clamp_min(1))[present].mean()
    return {"acc": accuracy.item(), "mean_acc": mean_accuracy.item()}


if __name__ == "__main__":
    main()
