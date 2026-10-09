"""Train PointNet++ (SSG / MSG) or PointNet on ModelNet40 (normal-resampled release), following yanx27's
`train_classification.py`."""

import os
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

import torch_pointcloud.transforms as T
from torch_pointcloud.config import DATA_DIR
from torch_pointcloud.datasets import ModelNetNormalResampled
from torch_pointcloud.losses import TNetOrthogonalityRegularizer
from torch_pointcloud.metrics import confusion_matrix
from torch_pointcloud.models import create_model
from torch_pointcloud.utils.data import DataKeys, PointCloudDataLoader
from torch_pointcloud.utils.random import seed_everything

CPU_COUNT = os.cpu_count()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = CPU_COUNT // 2 if CPU_COUNT is not None else 0
MODELS = ["pointnet2-ssg.modelnet40.xu-yan", "pointnet2-msg.modelnet40.xu-yan", "pointnet.modelnet40"]
NUM_CLASSES = 40
NUM_POINTS = 1024
REGULARIZER_WEIGHT = 0.001
POINT_KEYS = [DataKeys.POS, DataKeys.NORMAL]

TRAIN_TRANSFORM = T.Compose(
    [
        T.Slice(keys=POINT_KEYS, stop=NUM_POINTS),
        T.Rescale(keys=DataKeys.POS, method="centroid"),  # pc_normalize
        T.RandomDropout(keys=POINT_KEYS, drop_ratio_range=(0.0, 0.875)),  # random_point_dropout
        T.RandomScale(keys=DataKeys.POS, scale_range=(0.8, 1.25)),  # random_scale_point_cloud
        T.RandomTranslate(keys=DataKeys.POS, translation_range=(-0.1, 0.1)),  # shift_point_cloud
    ]
)
VAL_TRANSFORM = T.Compose(
    [
        T.Slice(keys=POINT_KEYS, stop=NUM_POINTS),
        T.Rescale(keys=DataKeys.POS, method="centroid"),
    ]
)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    use_normals = "msg" in args.model

    train_loader, val_loader = configure_dataloaders(args)
    model = configure_model(args)
    optimizer, scheduler = configure_optimizers(args, model)
    criterion = configure_criterion(args)
    regularizer = configure_regularizer(args)

    num_train, num_val = len(train_loader.dataset), len(val_loader.dataset)  # type: ignore[arg-type]
    print(f"Training {args.model!r} on {num_train} shapes, validating on {num_val}.")
    output: Optional[Path] = Path(args.output) if args.output else None
    best_loss = float("inf")
    for epoch in range(args.epochs):
        scheduler.step()  # the reference steps the schedule before each epoch, so epoch 20 is the first at 0.7x
        loss = train_one_epoch(model, criterion, regularizer, optimizer, train_loader, args.device, use_normals)
        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch + 1}/{args.epochs}  lr={lr:.2e}  train/loss {loss:.4f}")

        if (epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs:
            val = evaluate(model, val_loader, args.device, use_normals)
            print(f"  val/OA {val['acc'] * 100:.2f} | val/mAcc {val['mean_acc'] * 100:.2f}")

        if output is not None:
            output.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), output / "last.pt")
            if loss < best_loss:
                best_loss = loss
                torch.save(model.state_dict(), output / "best.pt")


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Train PointNet++ on ModelNet40.")
    parser.add_argument("--model", default=MODELS[0], choices=MODELS)
    parser.add_argument("--root", default=DATA_DIR, help="Dataset root directory.")
    parser.add_argument("--output", default=None, help="Directory for the checkpoints (default: none saved).")
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--batch-size", default=24, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--num-workers", default=NUM_WORKERS, type=int)
    parser.add_argument("--eval-every", default=1, type=int, help="Validate every this many epochs.")
    parser.add_argument("--limit-train-batches", default=None, type=int)
    parser.add_argument("--limit-val-batches", default=None, type=int)
    parser.add_argument("--download", action="store_true", help="Download the normal-resampled release if missing.")
    parser.add_argument("--force-process", action="store_true", help="Force re-processing the dataset.")
    return parser.parse_args()


def configure_datasets(args: Namespace) -> tuple[Dataset, Dataset]:
    common: Dict[str, Any] = dict(
        root=args.root,
        variant="40",
        download=args.download,
        force_process=args.force_process,
        num_workers=args.num_workers,
    )

    train_dataset: Dataset
    val_dataset: Dataset
    train_dataset = ModelNetNormalResampled(train=True, transform=TRAIN_TRANSFORM, **common)
    val_dataset = ModelNetNormalResampled(train=False, transform=VAL_TRANSFORM, **common)

    if args.limit_train_batches is not None:
        train_dataset = Subset(
            train_dataset, range(min(args.limit_train_batches * args.batch_size, len(train_dataset)))
        )
    if args.limit_val_batches is not None:
        val_dataset = Subset(val_dataset, range(min(args.limit_val_batches * args.batch_size, len(val_dataset))))

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
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    return train_loader, val_loader


def configure_model(args: Namespace) -> nn.Module:
    return create_model(args.model, task="classification", pretrained=False).to(args.device)


def configure_optimizers(args: Namespace, model: nn.Module) -> tuple[Optimizer, LRScheduler]:
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.weight_decay
    )

    # Stepped once per epoch, before the epoch as the reference does.
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    return optimizer, scheduler


def configure_criterion(args: Namespace) -> nn.Module:
    return nn.CrossEntropyLoss()


def configure_regularizer(args: Namespace) -> Optional[nn.Module]:
    # The reference regularizes the feature transform of PointNet, not its input transform.
    return TNetOrthogonalityRegularizer() if args.model.startswith("pointnet.") else None


def train_one_epoch(
    model: nn.Module,
    criterion: nn.Module,
    regularizer: Optional[nn.Module],
    optimizer: Optimizer,
    loader: DataLoader,
    device: str,
    use_normals: bool,
) -> float:
    model.train()
    total_loss = 0.0
    for data in tqdm(loader, desc="Training", leave=False):
        data = {key: value.to(device) for key, value in data.items() if isinstance(value, Tensor)}
        x = data[DataKeys.NORMAL] if use_normals else None
        logits = model(x, data[DataKeys.POS], data[DataKeys.BATCH])
        loss = criterion(logits, data[DataKeys.LABEL])
        if regularizer is not None:
            loss = loss + REGULARIZER_WEIGHT * regularizer(model.get_submodule("encoder.ftnet"))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str, use_normals: bool) -> dict[str, float]:
    model.eval()
    confusion = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long, device=device)
    for data in tqdm(loader, desc="Evaluating", leave=False):
        data = {key: value.to(device) for key, value in data.items() if isinstance(value, Tensor)}
        x = data[DataKeys.NORMAL] if use_normals else None
        logits = model(x, data[DataKeys.POS], data[DataKeys.BATCH])
        confusion += confusion_matrix(logits.argmax(dim=1), data[DataKeys.LABEL], NUM_CLASSES)

    present = confusion.sum(dim=1) > 0
    accuracy = confusion.diag().sum() / confusion.sum()
    mean_accuracy = (confusion.diag() / confusion.sum(dim=1).clamp_min(1))[present].mean()
    return {"acc": accuracy.item(), "mean_acc": mean_accuracy.item()}


if __name__ == "__main__":
    main()
