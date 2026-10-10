"""Train RandLA-Net on SemanticKITTI following the authors' `main_SemanticKITTI.py` configuration."""

import math
import os
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, Dataset, RandomSampler, Subset
from tqdm import tqdm

import torch_pointcloud.transforms as T
from torch_pointcloud.config import DATA_DIR
from torch_pointcloud.datasets import SemanticKITTI
from torch_pointcloud.metrics import confusion_matrix, intersection_over_union
from torch_pointcloud.models import create_model
from torch_pointcloud.utils.data import DataKeys, PointCloudDataLoader
from torch_pointcloud.utils.random import seed_everything

CPU_COUNT = os.cpu_count()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = CPU_COUNT // 2 if CPU_COUNT is not None else 0
MODEL = "randlanet.semantickitti.tsung-han-wu"
NUM_CLASSES = 19
IGNORE_INDEX = 255
GRID_SIZE = 0.06  # sub_grid_size
NUM_POINTS = 4096 * 11
TRAIN_STEPS, VAL_STEPS = 500, 100
# The 19-class learning map: moving classes merge into their static counterpart.
LEARNING_MAP = {
    10: 0,
    252: 0,
    11: 1,
    15: 2,
    18: 3,
    258: 3,
    20: 4,
    259: 4,
    30: 5,
    254: 5,
    31: 6,
    253: 6,
    32: 7,
    255: 7,
    40: 8,
    44: 9,
    48: 10,
    49: 11,
    50: 12,
    51: 13,
    70: 14,
    71: 15,
    72: 16,
    80: 17,
    81: 18,
}
# Points per class of the training sequences, as hard-coded by the authors (`get_class_weights`).
NUM_PER_CLASS = [
    55437630, 320797, 541736, 2578735, 3274484, 552662, 184064, 78858, 240942562, 17294618,
    170599734, 6369672, 230413074, 101130274, 476491114, 9833174, 129609852, 4506626, 1168181,
]  # fmt: skip

# Grid subsampling, then the `NUM_POINTS` nearest neighbors of a random point, shuffled.
TRAIN_TRANSFORM = T.Compose(
    [
        T.KeepItems(keys=[DataKeys.POS, DataKeys.SEGMENT]),
        T.Relabel(keys=DataKeys.SEGMENT, labels=LEARNING_MAP, default=IGNORE_INDEX),
        T.Voxelize(pos_key=DataKeys.POS, pos_reduce="mean", keys=DataKeys.SEGMENT, reduce="first", size=GRID_SIZE),
        T.SphereCrop(
            pos_key=DataKeys.POS,
            radius=math.inf,
            max_nodes=NUM_POINTS,
            keys=DataKeys.SEGMENT,
            center="random_point",
        ),
        T.ShufflePoint(keys=[DataKeys.POS, DataKeys.SEGMENT]),
    ]
)
# The validation crops are drawn from a fixed seed, so every validation scores the same crops.
VAL_TRANSFORM = T.Compose(
    [
        T.KeepItems(keys=[DataKeys.POS, DataKeys.SEGMENT]),
        T.Relabel(keys=DataKeys.SEGMENT, labels=LEARNING_MAP, default=IGNORE_INDEX),
        T.Voxelize(pos_key=DataKeys.POS, pos_reduce="mean", keys=DataKeys.SEGMENT, reduce="first", size=GRID_SIZE),
        T.SphereCrop(
            pos_key=DataKeys.POS,
            radius=math.inf,
            max_nodes=NUM_POINTS,
            keys=DataKeys.SEGMENT,
            center="random_point",
            seed=0,
        ),
        T.ShufflePoint(keys=[DataKeys.POS, DataKeys.SEGMENT], seed=0),
    ]
)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    train_loader, val_loader = configure_dataloaders(args)
    model = configure_model(args)
    optimizer, scheduler = configure_optimizers(args, model)
    criterion = configure_criterion(args)

    num_train = len(train_loader) * args.batch_size
    num_val = len(val_loader.dataset)  # type: ignore[arg-type]
    print(f"Training {MODEL!r} on {num_train} scans per epoch, validating on {num_val}.")
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
    parser = ArgumentParser(description="Train RandLA-Net on SemanticKITTI with the reference recipe.")
    parser.add_argument("--root", default=DATA_DIR, help="Dataset root directory.")
    parser.add_argument(
        "--train-sequences", nargs="+", default=None, help="Training sequences (default: the split's 00-10 but 08)."
    )
    parser.add_argument("--val-sequences", nargs="+", default=None, help="Validation sequences (default: 08).")
    parser.add_argument("--output", default=None, help="Directory for the checkpoints (default: none saved).")
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--batch-size", default=6, type=int)
    parser.add_argument("--val-batch-size", default=20, type=int)
    parser.add_argument("--lr", default=1e-2, type=float)
    parser.add_argument("--lr-decay", default=0.95, type=float, help="Learning-rate factor per epoch.")
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

    # An epoch draws `TRAIN_STEPS` batches of scans at random (see `configure_dataloaders`); the validation scans are
    # a fixed random subset.
    val_batches = args.limit_val_batches if args.limit_val_batches is not None else VAL_STEPS
    num_val = min(val_batches * args.val_batch_size, len(val_dataset))
    val_indices = torch.randperm(len(val_dataset), generator=torch.Generator().manual_seed(0))[:num_val]
    val_dataset = Subset(val_dataset, val_indices.tolist())

    return train_dataset, val_dataset


def configure_dataloaders(args: Namespace) -> tuple[DataLoader, DataLoader]:
    train_dataset, val_dataset = configure_datasets(args)

    train_batches = args.limit_train_batches if args.limit_train_batches is not None else TRAIN_STEPS
    num_train = min(train_batches * args.batch_size, len(train_dataset))  # type: ignore[arg-type]
    train_sampler = RandomSampler(train_dataset, num_samples=num_train)  # type: ignore[arg-type]

    train_loader = PointCloudDataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers
    )
    val_loader = PointCloudDataLoader(
        val_dataset, batch_size=args.val_batch_size, shuffle=False, num_workers=args.num_workers
    )

    return train_loader, val_loader


def configure_model(args: Namespace) -> nn.Module:
    return create_model(MODEL, task="semantic-segmentation", pretrained=False).to(args.device)


def configure_optimizers(args: Namespace, model: nn.Module) -> tuple[Optimizer, LRScheduler]:
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Stepped once per epoch: lr x 0.95 per epoch.
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay)
    return optimizer, scheduler


def configure_criterion(args: Namespace) -> nn.Module:
    """Cross-entropy weighted per class by 1 / (frequency + 0.02), from the authors' hard-coded point counts."""
    frequency = torch.tensor(NUM_PER_CLASS, dtype=torch.float64)
    frequency = frequency / frequency.sum()
    weights = (1.0 / (frequency + 0.02)).float()

    return nn.CrossEntropyLoss(weight=weights.to(args.device), ignore_index=IGNORE_INDEX)


def train_one_epoch(
    model: nn.Module, criterion: nn.Module, optimizer: Optimizer, loader: DataLoader, device: str
) -> float:
    model.train()
    total_loss = 0.0
    for data in tqdm(loader, desc="Training", leave=False):
        data = {key: value.to(device) if isinstance(value, Tensor) else value for key, value in data.items()}
        loss = criterion(model(None, data[DataKeys.POS], data[DataKeys.BATCH]), data[DataKeys.SEGMENT])

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
        preds = model(None, data[DataKeys.POS], data[DataKeys.BATCH]).argmax(dim=1)
        target = data[DataKeys.SEGMENT]
        confusion += confusion_matrix(preds, target, NUM_CLASSES, ignore_index=IGNORE_INDEX)

    per_class = confusion.diagonal().float() / confusion.sum(dim=1).clamp(min=1).float()
    present = confusion.sum(dim=1) > 0
    return {
        "mIoU": float(intersection_over_union(confusion)),
        "OA": float(confusion.diagonal().sum()) / float(confusion.sum().clamp(min=1)),
        "mAcc": float(per_class[present].mean()),
    }


if __name__ == "__main__":
    main()
