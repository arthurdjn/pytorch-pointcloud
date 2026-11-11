"""Train MinkUNet on SemanticKITTI following MMDetection3D's `minkunet18_w32_torchsparse_8xb2-amp-15e` config."""

import math
import os
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor, nn
from torch.amp.grad_scaler import GradScaler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

import torch_pointcloud.transforms as T
from torch_pointcloud.config import DATA_DIR
from torch_pointcloud.datasets import SemanticKITTI
from torch_pointcloud.datasets.semantickitti import SEMANTIC_KITTI_LEARNING_MAP
from torch_pointcloud.metrics import accuracy, confusion_matrix, intersection_over_union
from torch_pointcloud.models import create_model, model_info
from torch_pointcloud.utils.data import DataKeys, PointCloudDataLoader
from torch_pointcloud.utils.random import seed_everything

CPU_COUNT = os.cpu_count()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = CPU_COUNT // 2 if CPU_COUNT is not None else 0
MODELS = [
    "minkunet18-w16.semantickitti.mmdet3d",
    "minkunet18-w20.semantickitti.mmdet3d",
    "minkunet18-w32.semantickitti.mmdet3d",
]
NUM_CLASSES = 19
IGNORE_INDEX = 255
VOXEL_SIZE = 0.05
MAX_VOXELS = 80_000


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    train_loader, val_loader = configure_dataloaders(args)
    model = configure_model(args)
    criterion = configure_criterion(args)
    optimizer, scheduler = configure_optimizers(args, model, len(train_loader))
    amp = not args.no_amp and torch.device(args.device).type == "cuda"
    scaler = GradScaler("cuda") if amp else None

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
    parser = ArgumentParser(description="Train MinkUNet on SemanticKITTI.")
    parser.add_argument("--model", default=MODELS[2], choices=MODELS)
    parser.add_argument("--root", default=DATA_DIR, help="Dataset root directory.")
    parser.add_argument(
        "--train-sequences", nargs="+", default=None, help="Training sequences (default: the split's 00-10 but 08)."
    )
    parser.add_argument("--val-sequences", nargs="+", default=None, help="Validation sequences (default: 08).")
    parser.add_argument("--output", default=None, help="Directory for the checkpoints (default: none saved).")
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--epochs", default=15, type=int)
    parser.add_argument("--batch-size", default=16, type=int, help="The reference's 2 per GPU on 8 GPUs.")
    parser.add_argument("--lr", default=0.24, type=float)
    parser.add_argument("--min-lr", default=1e-5, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--warmup-iters", default=125, type=int)
    parser.add_argument("--warmup-factor", default=0.008, type=float, help="Learning-rate factor of the first step.")
    parser.add_argument("--max-voxels", default=MAX_VOXELS, type=int, help="Voxels kept per training scan.")
    parser.add_argument("--no-amp", action="store_true", help="Disable the float16 autocast of the reference.")
    parser.add_argument("--num-workers", default=NUM_WORKERS, type=int)
    parser.add_argument("--eval-every", default=1, type=int, help="Validate every this many epochs.")
    parser.add_argument("--limit-train-batches", default=None, type=int)
    parser.add_argument("--limit-val-batches", default=None, type=int)
    return parser.parse_args()


def configure_train_transform(args: Namespace) -> T.Transform:
    return T.Compose(
        [
            T.KeepItems(keys=[DataKeys.POS, DataKeys.INTENSITY, DataKeys.SEGMENT]),
            T.Relabel(keys=DataKeys.SEGMENT, labels=SEMANTIC_KITTI_LEARNING_MAP, default=IGNORE_INDEX),
            T.RandomRotate(keys=DataKeys.POS, axis=2, angle_range=(0.0, 360.0)),
            T.RandomScale(keys=DataKeys.POS, scale_range=(0.95, 1.05)),
            T.Cat(keys=[DataKeys.POS, DataKeys.INTENSITY], dst_key=DataKeys.X, dim=1),  # the augmented coordinates
            T.Translate(keys=DataKeys.POS, offset=VOXEL_SIZE / 2),  # round to the grid instead of flooring
            T.Voxelize(
                pos_key=DataKeys.POS,
                pos_reduce="first",
                dst_pos_grid_key=DataKeys.POS_GRID,
                keys=[DataKeys.X, DataKeys.SEGMENT],
                reduce="first",
                size=VOXEL_SIZE,
            ),
            T.RandomSample(
                keys=[DataKeys.POS, DataKeys.POS_GRID, DataKeys.X, DataKeys.SEGMENT],
                num_samples=args.max_voxels,
                allow_fewer=True,
            ),
        ]
    )


def configure_datasets(args: Namespace) -> tuple[Dataset, Dataset]:
    train_dataset: Dataset
    val_dataset: Dataset
    val_transform = model_info(args.model, task="semantic-segmentation")["transform"]
    train_dataset = SemanticKITTI(
        root=args.root, split="train", sequences=args.train_sequences, transform=configure_train_transform(args)
    )
    val_dataset = SemanticKITTI(root=args.root, split="val", sequences=args.val_sequences, transform=val_transform)

    if args.limit_train_batches is not None:
        train_dataset = Subset(
            train_dataset, range(min(args.limit_train_batches * args.batch_size, len(train_dataset)))
        )
    if args.limit_val_batches is not None:
        val_dataset = Subset(val_dataset, range(min(args.limit_val_batches, len(val_dataset))))

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
    val_loader = PointCloudDataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    return train_loader, val_loader


def configure_model(args: Namespace) -> nn.Module:
    return create_model(args.model, task="semantic-segmentation", pretrained=False).to(args.device)


def configure_optimizers(args: Namespace, model: nn.Module, steps_per_epoch: int) -> tuple[Optimizer, LRScheduler]:
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay, nesterov=True
    )
    total_iters = args.epochs * steps_per_epoch

    # Stepped per iteration: a cosine decay to `min_lr` over the whole run, scaled by a linear warm-up.
    def lr_lambda(iteration: int) -> float:
        warmup = min(1.0, args.warmup_factor + (1 - args.warmup_factor) * iteration / args.warmup_iters)
        cosine = 0.5 * (1 + math.cos(math.pi * min(iteration, total_iters) / total_iters))
        return warmup * (args.min_lr + (args.lr - args.min_lr) * cosine) / args.lr

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
    scaler: Optional[GradScaler],
) -> float:
    model.train()
    total_loss = 0.0
    for data in tqdm(loader, desc="Training", leave=False):
        data = {key: value.to(device) if isinstance(value, Tensor) else value for key, value in data.items()}
        with torch.autocast(torch.device(device).type, dtype=torch.float16, enabled=scaler is not None):
            logits = model(data[DataKeys.X], data[DataKeys.POS_GRID], data[DataKeys.BATCH])
            loss = criterion(logits, data[DataKeys.SEGMENT])

        optimizer.zero_grad(set_to_none=True)
        if scaler is None:
            loss.backward()
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> dict[str, float]:
    model.eval()
    cm = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long, device=device)
    for data in tqdm(loader, desc="Evaluating", leave=False):
        data = {key: value.to(device) if isinstance(value, Tensor) else value for key, value in data.items()}
        preds = model(data[DataKeys.X], data[DataKeys.POS_GRID], data[DataKeys.BATCH]).argmax(dim=1)
        preds = preds[data[DataKeys.INVERSE]]
        cm += confusion_matrix(preds, data[DataKeys.ORIGIN_SEGMENT], NUM_CLASSES, ignore_index=IGNORE_INDEX)

    return {"mIoU": float(intersection_over_union(cm)), "OA": float(accuracy(cm))}


if __name__ == "__main__":
    main()
