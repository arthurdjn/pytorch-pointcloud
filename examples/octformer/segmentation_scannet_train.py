"""Train OctFormer on ScanNet, 20 classes, following its `seg_scannet.yaml` configuration."""

import math
import os
from argparse import ArgumentParser, Namespace
from bisect import bisect_right
from pathlib import Path
from typing import Any, Optional

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

import torch_pointcloud.transforms as T
from torch_pointcloud.config import DATA_DIR
from torch_pointcloud.datasets import ScanNet20
from torch_pointcloud.metrics import accuracy, confusion_matrix, intersection_over_union
from torch_pointcloud.models import create_model, model_info
from torch_pointcloud.utils.data import DataKeys, PointCloudDataLoader
from torch_pointcloud.utils.random import seed_everything

CPU_COUNT = os.cpu_count()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = CPU_COUNT // 2 if CPU_COUNT is not None else 0
MODEL = "octformer-base.scannet20.octree-nn"
NUM_CLASSES = 21
IGNORE_INDEX = 0
DEPTH = 11
FULL_DEPTH = 2
SCALE = 10.24  # the scene fits in [-1, 1] at depth 11 with 1 cm voxels
MAX_POINTS = 120_000
CROP_RATIO = 0.8
MILESTONES = (360, 520)
POINT_KEYS = [DataKeys.POS, DataKeys.NORMAL, DataKeys.COLOR, DataKeys.SEGMENT]

TRAIN_TRANSFORM = T.Compose(
    [
        T.Shift(keys=DataKeys.POS, method="bbox"),
        T.Divide(keys=[DataKeys.POS, DataKeys.COLOR], divisor=[SCALE, 255]),
        T.RandomColorAutoContrast(keys=DataKeys.COLOR, blend=None, p=1.0),
        T.RandomColorShift(keys=DataKeys.COLOR, shift_range=(-0.1, 0.1), p=0.95),
        T.RandomJitter(keys=DataKeys.COLOR, sigma=0.05, clip=None, p=0.95),
        T.Clamp(keys=DataKeys.COLOR, min=0.0, max=1.0),
        T.RandomElasticDistortion(
            keys=DataKeys.POS, granularity=(0.05, 0.1, 0.2, 0.4), magnitude=(0.1, 0.2, 0.4, 0.8), p=0.95
        ),
        T.RandomFlip(keys=[DataKeys.POS, DataKeys.NORMAL], axes=(0,), p=0.5),
        T.RandomFlip(keys=[DataKeys.POS, DataKeys.NORMAL], axes=(1,), p=0.5),
        # The reference draws whole degrees: +-5 about x and y, +-180 about z.
        T.RandomRotate(keys=DataKeys.POS, vector_keys=DataKeys.NORMAL, axis=0, angle_range=(-5.0, 5.0)),
        T.RandomRotate(keys=DataKeys.POS, vector_keys=DataKeys.NORMAL, axis=1, angle_range=(-5.0, 5.0)),
        T.RandomRotate(keys=DataKeys.POS, vector_keys=DataKeys.NORMAL, axis=2, angle_range=(-180.0, 180.0)),
        T.RandomTranslate(keys=DataKeys.POS, translation_range=(-0.1, 0.1)),
        T.RandomScale(keys=DataKeys.POS, scale_range=(0.8, 1.2), anisotropic=True),
        T.BoxMask(keys=DataKeys.POS, bbox=(-1.0, -1.0, -1.0, 1.0, 1.0, 1.0), dst_keys=DataKeys.BOX_MASK),
        T.ApplyMask(keys=POINT_KEYS, mask_key=DataKeys.BOX_MASK),
        T.SphereCrop(
            pos_key=DataKeys.POS,
            radius=math.inf,
            max_nodes=MAX_POINTS,
            max_ratio=CROP_RATIO,
            keys=[DataKeys.NORMAL, DataKeys.COLOR, DataKeys.SEGMENT],
            center="random_point",
        ),
        T.Shift(keys=DataKeys.POS, method="min", axes=[2]),
        T.BuildOctree(
            pos_key=DataKeys.POS,
            normal_key=DataKeys.NORMAL,
            feature_key=DataKeys.COLOR,
            label_key=DataKeys.SEGMENT,
            dst_points_key=DataKeys.OCTREE_POINTS,
            dst_octree_key=DataKeys.OCTREE,
            depth=DEPTH,
            full_depth=FULL_DEPTH,
            batch_size=1,
        ),
        T.OctreeFeatures(keys=DataKeys.OCTREE, features_type="NDFP", nempty=True, dst_keys=DataKeys.X),
    ]
)

# The registered inference transform of the model: the normalized scene, its octree and the NDFP features.
VAL_TRANSFORM = model_info(MODEL, task="semantic-segmentation")["transform"]


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    train_loader, val_loader = configure_dataloaders(args)
    model = configure_model(args)
    criterion = configure_criterion(args)
    optimizer, scheduler = configure_optimizers(args, model)

    num_train, num_val = len(train_loader.dataset), len(val_loader.dataset)  # type: ignore[arg-type]
    print(f"Training {MODEL!r} on {num_train} scenes, validating on {num_val}.")
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
    parser = ArgumentParser(description="Train OctFormer on ScanNet (20 classes).")
    parser.add_argument("--model", default=MODEL, choices=[MODEL])
    parser.add_argument("--root", default=DATA_DIR, help="Dataset root directory.")
    parser.add_argument("--output", default=None, help="Directory for the checkpoints (default: none saved).")
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--epochs", default=600, type=int)
    # The reference trains 4 scenes per process on 4 processes with 0.0015 per process: 16 scenes at 0.006 here.
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument(
        "--lr", default=6e-3, type=float, help="Peak learning rate (the transformer blocks use a tenth)."
    )
    parser.add_argument("--weight-decay", default=0.05, type=float)
    parser.add_argument("--warmup-epochs", default=20, type=int)
    parser.add_argument("--num-workers", default=NUM_WORKERS, type=int)
    parser.add_argument("--eval-every", default=10, type=int, help="Validate every this many epochs.")
    parser.add_argument("--limit-train-batches", default=None, type=int)
    parser.add_argument("--limit-val-batches", default=None, type=int)
    parser.add_argument(
        "--download", action="store_true", help="Download ScanNet if missing (requires accepting its terms)."
    )
    parser.add_argument("--force-process", action="store_true", help="Force re-processing the dataset.")
    return parser.parse_args()


def configure_datasets(args: Namespace) -> tuple[Dataset, Dataset]:
    common: dict[str, Any] = dict(
        root=args.root, download=args.download, force_process=args.force_process, num_workers=args.num_workers
    )

    # The scenes stay in their scan frame, as the benchmark evaluates the released weights.
    train_dataset: Dataset
    val_dataset: Dataset
    train_dataset = ScanNet20(split="train", transform=TRAIN_TRANSFORM, use_axis_alignment=False, **common)
    val_dataset = ScanNet20(split="val", transform=VAL_TRANSFORM, use_axis_alignment=False, **common)

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
        drop_last=len(train_dataset) > args.batch_size,  # type: ignore[arg-type]
        num_workers=args.num_workers,
    )
    val_loader = PointCloudDataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    return train_loader, val_loader


def configure_model(args: Namespace) -> nn.Module:
    return create_model(args.model, task="semantic-segmentation", pretrained=False).to(args.device)


def configure_criterion(args: Namespace) -> nn.Module:
    return nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)


def configure_optimizers(args: Namespace, model: nn.Module) -> tuple[Optimizer, LRScheduler]:
    # `adamw_attn`: the transformer blocks run at a tenth of the learning rate, every parameter decays.
    blocks = [param for name, param in model.named_parameters() if "blocks" in name]
    others = [param for name, param in model.named_parameters() if "blocks" not in name]
    optimizer = torch.optim.AdamW(
        [{"params": others}, {"params": blocks, "lr": args.lr / 10}], lr=args.lr, weight_decay=args.weight_decay
    )

    # `step_warmup`, stepped once per epoch: linear warm-up from a thousandth, then x0.1 at each milestone.
    def factor(epoch: int) -> float:
        if epoch <= args.warmup_epochs:
            return (1 - 1e-3) * epoch / args.warmup_epochs + 1e-3
        return 0.1 ** bisect_right(MILESTONES, epoch)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    return optimizer, scheduler


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
        x, pos, batch = data[DataKeys.X].to(device), data[DataKeys.POS].to(device), data[DataKeys.BATCH].to(device)
        logits = model(x, octree, octree.depth, pos, batch)
        loss = criterion(logits, data[DataKeys.SEGMENT].to(device).long())

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
        x, pos, batch = data[DataKeys.X].to(device), data[DataKeys.POS].to(device), data[DataKeys.BATCH].to(device)
        preds = model(x, octree, octree.depth, pos, batch).argmax(dim=1)
        target: Tensor = data[DataKeys.SEGMENT].to(device)
        confusion += confusion_matrix(preds, target, NUM_CLASSES, ignore_index=IGNORE_INDEX)

    return {
        "mIoU": float(intersection_over_union(confusion, ignore_index=IGNORE_INDEX)),
        "OA": float(accuracy(confusion)),
    }


if __name__ == "__main__":
    main()
