"""Train SPVCNN on SemanticKITTI following spvnas' `configs/semantic_kitti/spvcnn` configs."""

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
from torch_pointcloud.metrics import confusion_matrix, intersection_over_union
from torch_pointcloud.models import create_model
from torch_pointcloud.utils.data import DataKeys, PointCloudDataLoader
from torch_pointcloud.utils.random import seed_everything

CPU_COUNT = os.cpu_count()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = CPU_COUNT // 2 if CPU_COUNT is not None else 0
MODELS = [
    "spvcnn-30gmacs.semantickitti.mit-han-lab",
    "spvcnn-47gmacs.semantickitti.mit-han-lab",
    "spvcnn-119gmacs.semantickitti.mit-han-lab",
]
NUM_CLASSES = 19
IGNORE_INDEX = 255
VOXEL_SIZE = 0.05
MAX_VOXELS = 80_000
POINT_KEYS = [DataKeys.POS, DataKeys.INTENSITY, DataKeys.SEGMENT]
# The 19-class learning map: moving classes merge into their static counterpart, the rest is ignored.
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


TRAIN_TRANSFORM = T.Compose(
    [
        T.KeepItems(keys=POINT_KEYS),
        T.Relabel(keys=DataKeys.SEGMENT, labels=LEARNING_MAP, default=IGNORE_INDEX),
        T.RandomRotate(keys=DataKeys.POS, axis=2, angle_range=(0.0, 360.0)),
        T.RandomScale(keys=DataKeys.POS, scale_range=(0.95, 1.05)),
        T.Cat(keys=[DataKeys.POS, DataKeys.INTENSITY], dst_key=DataKeys.X, dim=1),  # the augmented coordinates
        T.Voxelize(  # `sparse_quantize`: integer grid coordinates, the first point of each voxel
            pos_key=DataKeys.POS,
            pos_reduce="grid",
            keys=[DataKeys.X, DataKeys.SEGMENT],
            reduce="first",
            size=VOXEL_SIZE,
        ),
        T.RandomSample(keys=[DataKeys.POS, DataKeys.X, DataKeys.SEGMENT], num_samples=MAX_VOXELS, allow_fewer=True),
    ]
)

VAL_TRANSFORM = T.Compose(
    [
        T.KeepItems(keys=POINT_KEYS),
        T.Relabel(keys=DataKeys.SEGMENT, labels=LEARNING_MAP, default=IGNORE_INDEX),
        T.Cat(keys=[DataKeys.POS, DataKeys.INTENSITY], dst_key=DataKeys.X, dim=1),
        T.CopyItems(keys=[DataKeys.POS, DataKeys.SEGMENT], dst_keys=[DataKeys.ORIGIN_POS, DataKeys.ORIGIN_SEGMENT]),
        T.Voxelize(
            pos_key=DataKeys.POS,
            pos_reduce="grid",
            keys=[DataKeys.X, DataKeys.SEGMENT],
            reduce="first",
            size=VOXEL_SIZE,
            dst_inverse_key=DataKeys.INVERSE,
        ),
    ]
)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    train_loader, val_loader = configure_dataloaders(args)
    model = configure_model(args)
    criterion = configure_criterion(args)
    # The reference counts the epoch in whole batches of the scans, dropped last batch included.
    steps_per_epoch = math.ceil(len(train_loader.dataset) / args.batch_size)  # type: ignore[arg-type]
    optimizer, scheduler = configure_optimizers(args, model, steps_per_epoch)
    amp = not args.no_amp and torch.device(args.device).type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if amp else None

    num_train, num_val = len(train_loader.dataset), len(val_loader.dataset)  # type: ignore[arg-type]
    print(f"Training {args.model!r} on {num_train} scans, validating on {num_val}.")
    output: Optional[Path] = Path(args.output) if args.output else None
    best_loss = float("inf")
    for epoch in range(args.epochs):
        loss = train_one_epoch(model, criterion, optimizer, scheduler, train_loader, args.device, scaler)
        print(f"Epoch {epoch + 1}/{args.epochs}  lr={scheduler.get_last_lr()[0]:.2e}  train/loss {loss:.4f}")

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
    parser = ArgumentParser(description="Train SPVCNN on SemanticKITTI.")
    parser.add_argument("--model", default=MODELS[1], choices=MODELS)
    parser.add_argument("--root", default=DATA_DIR, help="Dataset root directory.")
    parser.add_argument(
        "--train-sequences", nargs="+", default=None, help="Training sequences (default: the split's 00-10 but 08)."
    )
    parser.add_argument("--val-sequences", nargs="+", default=None, help="Validation sequences (default: 08).")
    parser.add_argument("--output", default=None, help="Directory for the checkpoints (default: none saved).")
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--epochs", default=15, type=int)
    parser.add_argument("--batch-size", default=16, type=int, help="The reference's 2 per GPU on 8 GPUs.")
    parser.add_argument("--val-batch-size", default=2, type=int)
    parser.add_argument("--lr", default=0.24, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--warmup-iters", default=125, type=int, help="The reference's 1000 / #GPUs on 8 GPUs.")
    parser.add_argument("--no-amp", action="store_true", help="Disable the float16 autocast of the reference.")
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


def configure_model(args: Namespace) -> nn.Module:
    return create_model(args.model, task="semantic-segmentation", pretrained=False).to(args.device)


def configure_optimizers(args: Namespace, model: nn.Module, steps_per_epoch: int) -> tuple[Optimizer, LRScheduler]:
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay, nesterov=True
    )

    # Stepped per iteration: cosine with a linear warm-up.
    def lr_lambda(iteration: int) -> float:
        if iteration < args.warmup_iters:
            return (iteration + 1) / args.warmup_iters

        ratio = (iteration - args.warmup_iters) / (args.epochs * steps_per_epoch)
        return 0.5 * (1 + math.cos(math.pi * ratio))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler


def configure_criterion(args: Namespace) -> nn.Module:
    return nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)


def train_one_epoch(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    loader: DataLoader,
    device: str,
    scaler: Optional[torch.amp.GradScaler],
) -> float:
    model.train()
    total_loss = 0.0
    for data in tqdm(loader, desc="Training", leave=False):
        data = {key: value.to(device) if isinstance(value, Tensor) else value for key, value in data.items()}
        with torch.autocast(torch.device(device).type, dtype=torch.float16, enabled=scaler is not None):
            loss = criterion(model(data[DataKeys.X], data[DataKeys.POS], data[DataKeys.BATCH]), data[DataKeys.SEGMENT])

        optimizer.zero_grad(set_to_none=True)
        if scaler is None:
            loss.backward()
            optimizer.step()
            scheduler.step()
        else:
            scaler.scale(loss).backward()
            scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scale <= scaler.get_scale():  # the scaler skips the optimizer step on an overflow
                scheduler.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> dict[str, float]:
    model.eval()
    confusion = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long, device=device)
    for data in tqdm(loader, desc="Evaluating", leave=False):
        data = {key: value.to(device) if isinstance(value, Tensor) else value for key, value in data.items()}
        preds = model(data[DataKeys.X], data[DataKeys.POS], data[DataKeys.BATCH]).argmax(dim=1)
        preds = preds[data[DataKeys.INVERSE]]
        target = data[DataKeys.ORIGIN_SEGMENT]
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
