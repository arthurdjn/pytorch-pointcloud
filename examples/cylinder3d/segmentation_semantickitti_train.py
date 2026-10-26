"""Train Cylinder3D on SemanticKITTI following mmdet3d's `cylinder3d_4xb4-3x_semantickitti` config."""

import math
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
from torch_pointcloud.datasets import SemanticKITTI
from torch_pointcloud.datasets.semantickitti import SEMANTIC_KITTI_LEARNING_MAP
from torch_pointcloud.losses import LovaszLoss, SumLoss
from torch_pointcloud.metrics import accuracy, confusion_matrix, intersection_over_union
from torch_pointcloud.models import SemanticSegmentationModel, create_model
from torch_pointcloud.ops.voxelization import majority_label
from torch_pointcloud.utils.data import DataKeys, PointCloudDataLoader
from torch_pointcloud.utils.random import seed_everything

CPU_COUNT = os.cpu_count()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = CPU_COUNT // 2 if CPU_COUNT is not None else 0
MODELS = ["cylinder3d.semantickitti.mmdet3d"]
NUM_CLASSES = 19
IGNORE_INDEX = 255
TRANSLATION_STD = 0.1
POINT_KEYS = [DataKeys.POS, DataKeys.INTENSITY, DataKeys.SEGMENT]

TRAIN_TRANSFORM = T.Compose(
    [
        T.KeepItems(keys=POINT_KEYS),
        T.Relabel(keys=DataKeys.SEGMENT, labels=SEMANTIC_KITTI_LEARNING_MAP, default=IGNORE_INDEX),
        T.RandomFlip(keys=DataKeys.POS, axes=(0, 1), p=0.5),
        T.RandomRotate(keys=DataKeys.POS, axis=2, angle_range=(-45.0, 45.0)),
        T.RandomScale(keys=DataKeys.POS, scale_range=(0.95, 1.05)),
        # The reference draws the shift of each axis from a Gaussian of standard deviation TRANSLATION_STD.
        T.RandomTranslate(
            keys=DataKeys.POS,
            translation_range=(-TRANSLATION_STD * math.sqrt(3), TRANSLATION_STD * math.sqrt(3)),
        ),
        T.Cat(keys=[DataKeys.INTENSITY], dst_key=DataKeys.X, dim=1),
    ]
)

VAL_TRANSFORM = T.Compose(
    [
        T.KeepItems(keys=POINT_KEYS),
        T.Relabel(keys=DataKeys.SEGMENT, labels=SEMANTIC_KITTI_LEARNING_MAP, default=IGNORE_INDEX),
        T.Cat(keys=[DataKeys.INTENSITY], dst_key=DataKeys.X, dim=1),
    ]
)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    train_loader, val_loader = configure_dataloaders(args)
    model = configure_model(args)
    criterion = configure_criterion(args)
    steps_per_epoch = len(train_loader)
    optimizer, scheduler = configure_optimizers(args, model, steps_per_epoch)

    num_train, num_val = len(train_loader.dataset), len(val_loader.dataset)  # type: ignore[arg-type]
    print(f"Training {args.model!r} on {num_train} scans, validating on {num_val}.")
    output: Optional[Path] = Path(args.output) if args.output else None
    best_miou = -1.0
    for epoch in range(args.epochs):
        loss = train_one_epoch(model, criterion, optimizer, scheduler, train_loader, args.device)
        print(f"Epoch {epoch + 1}/{args.epochs}  lr={scheduler.get_last_lr()[0]:.2e}  train/loss {loss:.4f}")

        if (epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs:
            val = evaluate(model, val_loader, args.device)
            print("  " + " | ".join(f"val/{key} {value * 100:.2f}" for key, value in val.items()))
            if output is not None and val["mIoU"] > best_miou:
                best_miou = val["mIoU"]
                output.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), output / "best.pt")

        if output is not None:
            output.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), output / "last.pt")


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Train Cylinder3D on SemanticKITTI.")
    parser.add_argument("--model", default=MODELS[0], choices=MODELS)
    parser.add_argument("--root", default=DATA_DIR, help="Dataset root directory.")
    parser.add_argument(
        "--train-sequences", nargs="+", default=None, help="Training sequences (default: the split's 00-10 but 08)."
    )
    parser.add_argument("--val-sequences", nargs="+", default=None, help="Validation sequences (default: 08).")
    parser.add_argument("--output", default=None, help="Directory for the checkpoints (default: none saved).")
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--epochs", default=36, type=int)
    parser.add_argument(
        "--milestone", default=30, type=int, help="Epoch from which the learning rate is divided by 10."
    )
    parser.add_argument("--batch-size", default=16, type=int, help="The reference's 4 per GPU on 4 GPUs.")
    parser.add_argument("--val-batch-size", default=1, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--weight-decay", default=0.01, type=float)
    parser.add_argument("--warmup-iters", default=1000, type=int)
    parser.add_argument("--num-workers", default=NUM_WORKERS, type=int)
    parser.add_argument("--eval-every", default=1, type=int, help="Validate every this many epochs.")
    parser.add_argument("--limit-train-batches", default=None, type=int)
    parser.add_argument("--limit-val-batches", default=None, type=int)
    return parser.parse_args()


def configure_datasets(args: Namespace) -> tuple[Dataset, Dataset]:
    train_dataset: Dataset
    val_dataset: Dataset
    train_dataset = SemanticKITTI(
        root=args.root, split="train", sequences=args.train_sequences, transform=TRAIN_TRANSFORM
    )
    val_dataset = SemanticKITTI(root=args.root, split="val", sequences=args.val_sequences, transform=VAL_TRANSFORM)

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


def configure_model(args: Namespace) -> SemanticSegmentationModel:
    return create_model(args.model, task="semantic-segmentation", pretrained=False).to(args.device)


def configure_optimizers(args: Namespace, model: nn.Module, steps_per_epoch: int) -> tuple[Optimizer, LRScheduler]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Stepped per iteration: a linear warm-up from 0.001 of the rate, then a step decay at the milestone epoch.
    def lr_lambda(iteration: int) -> float:
        warmup = 0.001 + 0.999 * min(iteration / args.warmup_iters, 1.0)
        decay = 0.1 if iteration >= args.milestone * steps_per_epoch else 1.0
        return warmup * decay

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler


def configure_criterion(args: Namespace) -> nn.Module:
    return SumLoss([nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX), LovaszLoss(ignore_index=IGNORE_INDEX)])


def train_one_epoch(
    model: SemanticSegmentationModel,
    criterion: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    loader: DataLoader,
    device: str,
) -> float:
    model.train()
    total_loss = 0.0
    for data in tqdm(loader, desc="Training", leave=False):
        data = {key: value.to(device) if isinstance(value, Tensor) else value for key, value in data.items()}
        # The reference scores the voxels against the majority label of their points.
        sparse_x, inverse, skips = model.forward_features(
            data[DataKeys.X],
            data[DataKeys.POS],
            data[DataKeys.BATCH],
            return_intermediates=True,
        )
        sparse_x = model.forward_decoder(sparse_x, skips)
        logits = model.forward_head(sparse_x)
        target = majority_label(data[DataKeys.SEGMENT], inverse, NUM_CLASSES, logits.shape[0], IGNORE_INDEX)
        loss = criterion(logits, target)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> dict[str, float]:
    model.eval()
    cm = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long, device=device)
    for data in tqdm(loader, desc="Evaluating", leave=False):
        data = {key: value.to(device) if isinstance(value, Tensor) else value for key, value in data.items()}
        preds = model(data[DataKeys.X], data[DataKeys.POS], data[DataKeys.BATCH]).argmax(dim=1)
        cm += confusion_matrix(preds, data[DataKeys.SEGMENT], NUM_CLASSES, ignore_index=IGNORE_INDEX)

    return {
        "mIoU": intersection_over_union(cm),
        "OA": accuracy(cm),
    }


if __name__ == "__main__":
    main()
