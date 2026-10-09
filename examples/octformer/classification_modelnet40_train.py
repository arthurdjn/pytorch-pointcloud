"""Train OctFormer on ModelNet40, following octformer's `cls_m40.yaml`."""

import os
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

import torch_pointcloud.transforms as T
from torch_pointcloud.config import DATA_DIR
from torch_pointcloud.datasets import ModelNet40
from torch_pointcloud.metrics import confusion_matrix
from torch_pointcloud.models import create_model, model_info
from torch_pointcloud.utils.data import DataKeys, PointCloudDataLoader
from torch_pointcloud.utils.random import seed_everything

CPU_COUNT = os.cpu_count()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = CPU_COUNT // 2 if CPU_COUNT is not None else 0
MODELS = ["octformer-base.modelnet40.octree-nn"]
NUM_CLASSES = 40
NUM_SAMPLES = 8000
POINT_KEYS = [DataKeys.POS, DataKeys.NORMAL]
DEPTH = 6
FULL_DEPTH = 2
# `angle: (0, 0, 5)` with `interval: (1, 1, 1)`: a whole number of degrees about z.
ROTATION_DEGREES = list(range(-5, 6))
SCALE_RANGE = (0.75, 1.25)
JITTER_RANGE = (-0.125, 0.125)
MILESTONES = (120, 160)
LR_GAMMA = 0.1

TRAIN_TRANSFORM = T.Compose(
    [
        # The reference samples the 8000 surface points of every mesh once, offline.
        T.RandomSampleFaceVertices(
            keys=DataKeys.POS,
            face_key=DataKeys.FACE,
            dst_normal_key=DataKeys.NORMAL,
            num_samples=NUM_SAMPLES,
        ),
        T.ToTensor(keys=POINT_KEYS, dtype=torch.float32),
        T.Shift(keys=DataKeys.POS, method="bbox"),
        T.Rescale(keys=DataKeys.POS, method="bbox"),
        T.RandomRotateChoice(keys=POINT_KEYS, angles=ROTATION_DEGREES, axis=2),
        T.RandomTranslate(keys=DataKeys.POS, translation_range=JITTER_RANGE),
        T.RandomScale(keys=DataKeys.POS, scale_range=SCALE_RANGE, anisotropic=True),
        T.Abs(keys=DataKeys.NORMAL),
        T.BoxMask(keys=DataKeys.POS, bbox=(-1.0, -1.0, -1.0, 1.0, 1.0, 1.0), dst_keys=DataKeys.BOX_MASK),
        T.ApplyMask(keys=POINT_KEYS, mask_key=DataKeys.BOX_MASK),
        T.BuildOctree(
            pos_key=DataKeys.POS,
            dst_octree_key=DataKeys.OCTREE,
            depth=DEPTH,
            full_depth=FULL_DEPTH,
            batch_size=1,
            normal_key=DataKeys.NORMAL,
        ),
        T.OctreeFeatures(keys=DataKeys.OCTREE, features_type="ND", nempty=False, dst_keys=DataKeys.X),
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

        # The reference keeps the checkpoint of the best validation accuracy.
        if output is not None:
            output.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), output / "last.pt")
            if loss < best_loss:
                best_loss = loss
                torch.save(model.state_dict(), output / "best.pt")


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Train OctFormer on ModelNet40.")
    parser.add_argument("--model", default=MODELS[0], choices=MODELS)
    parser.add_argument("--root", default=DATA_DIR, help="Dataset root directory.")
    parser.add_argument("--output", default=None, help="Directory for the checkpoints (default: none saved).")
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--val-batch-size", default=32, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--weight-decay", default=0.05, type=float)
    parser.add_argument("--num-workers", default=NUM_WORKERS, type=int)
    parser.add_argument("--eval-every", default=5, type=int, help="Validate every this many epochs.")
    parser.add_argument("--limit-train-batches", default=None, type=int)
    parser.add_argument("--limit-val-batches", default=None, type=int)
    parser.add_argument("--download", action="store_true", help="Download the dataset if missing.")
    parser.add_argument("--force-process", action="store_true", help="Reprocess the dataset.")
    return parser.parse_args()


def configure_datasets(args: Namespace) -> tuple[Dataset, Dataset]:
    train_dataset: Dataset
    val_dataset: Dataset
    train_dataset = ModelNet40(
        root=args.root,
        train=True,
        transform=TRAIN_TRANSFORM,
        download=args.download,
        force_process=args.force_process,
    )
    # The reference validates on its own once-sampled points; the benchmark protocol is used here.
    val_dataset = ModelNet40(
        root=args.root,
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=list(MILESTONES), gamma=LR_GAMMA)
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
        octree = data[DataKeys.OCTREE].to(device)
        logits = model(data[DataKeys.X].to(device), octree, octree.depth)
        loss = criterion(logits, data[DataKeys.LABEL].to(device))

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
        octree = data[DataKeys.OCTREE].to(device)
        logits = model(data[DataKeys.X].to(device), octree, octree.depth)
        confusion += confusion_matrix(logits.argmax(dim=1), data[DataKeys.LABEL].to(device), NUM_CLASSES)

    present = confusion.sum(dim=1) > 0
    accuracy = confusion.diag().sum() / confusion.sum()
    mean_accuracy = (confusion.diag() / confusion.sum(dim=1).clamp_min(1))[present].mean()
    return {"acc": accuracy.item(), "mean_acc": mean_accuracy.item()}


if __name__ == "__main__":
    main()
