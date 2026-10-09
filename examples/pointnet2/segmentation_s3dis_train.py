"""Train PointNet++ or PointNet on S3DIS, following yanx27's `train_semseg.py`."""

import os
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

import torch_pointcloud.transforms as T
from torch_pointcloud.config import DATA_DIR
from torch_pointcloud.datasets import S3DIS, RepeatSampler
from torch_pointcloud.datasets.s3dis import S3DIS_AREAS, S3DISArea
from torch_pointcloud.losses import TNetOrthogonalityRegularizer
from torch_pointcloud.metrics import confusion_matrix, intersection_over_union
from torch_pointcloud.models import create_model
from torch_pointcloud.optim import set_bn_momentum
from torch_pointcloud.utils.data import DataKeys, PointCloudDataLoader
from torch_pointcloud.utils.random import seed_everything

CPU_COUNT = os.cpu_count()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = CPU_COUNT // 2 if CPU_COUNT is not None else 0
MODELS = ["pointnet2.s3dis-area5.xu-yan", "pointnet.s3dis-area5"]
NUM_CLASSES = 13
NUM_POINTS = 4096
BLOCK_SIZE = 1.0
MIN_BLOCK_POINTS = 1024
REGULARIZER_WEIGHT = 0.001
POINT_KEYS = [DataKeys.POS, DataKeys.COLOR, DataKeys.SEGMENT]

# A random 1 m block of 4096 points per room, with the room features.
TRAIN_TRANSFORM = T.Compose(
    [
        T.KeepItems(keys=POINT_KEYS),
        T.Shift(keys=DataKeys.POS, method="min"),
        T.Reduce(keys=DataKeys.POS, op="max", dim=0, dst_keys=DataKeys.SCENE_MAX),
        T.CopyItems(keys=DataKeys.POS, dst_keys=DataKeys.NORM_POS),
        T.DivideItems(keys=DataKeys.NORM_POS, div_keys=DataKeys.SCENE_MAX),
        T.Divide(keys=DataKeys.COLOR, divisor=255.0),
        T.RandomBlockCrop(
            pos_key=DataKeys.POS,
            keys=[DataKeys.COLOR, DataKeys.SEGMENT, DataKeys.NORM_POS],
            block_size=BLOCK_SIZE,
            min_nodes=MIN_BLOCK_POINTS,
            dst_center_key=DataKeys.BLOCK_CENTER,
        ),
        T.RandomSample(keys=[*POINT_KEYS, DataKeys.NORM_POS], num_samples=NUM_POINTS),
        T.SubtractItems(keys=DataKeys.POS, sub_keys=DataKeys.BLOCK_CENTER, axes=[0, 1]),
        T.RandomRotate(keys=DataKeys.POS, axis=2, angle_range=(0.0, 360.0)),  # rotate_point_cloud_z
        T.Cat(keys=[DataKeys.POS, DataKeys.COLOR, DataKeys.NORM_POS], dst_key=DataKeys.X),
    ]
)
# The validation blocks are drawn from a fixed seed, so every validation scores the same blocks.
VAL_TRANSFORM = T.Compose(
    [
        T.KeepItems(keys=POINT_KEYS),
        T.Shift(keys=DataKeys.POS, method="min"),
        T.Reduce(keys=DataKeys.POS, op="max", dim=0, dst_keys=DataKeys.SCENE_MAX),
        T.CopyItems(keys=DataKeys.POS, dst_keys=DataKeys.NORM_POS),
        T.DivideItems(keys=DataKeys.NORM_POS, div_keys=DataKeys.SCENE_MAX),
        T.Divide(keys=DataKeys.COLOR, divisor=255.0),
        T.RandomBlockCrop(
            pos_key=DataKeys.POS,
            keys=[DataKeys.COLOR, DataKeys.SEGMENT, DataKeys.NORM_POS],
            block_size=BLOCK_SIZE,
            min_nodes=MIN_BLOCK_POINTS,
            dst_center_key=DataKeys.BLOCK_CENTER,
            seed=0,
        ),
        T.RandomSample(keys=[*POINT_KEYS, DataKeys.NORM_POS], num_samples=NUM_POINTS, seed=0),
        T.SubtractItems(keys=DataKeys.POS, sub_keys=DataKeys.BLOCK_CENTER, axes=[0, 1]),
        T.Cat(keys=[DataKeys.POS, DataKeys.COLOR, DataKeys.NORM_POS], dst_key=DataKeys.X),
    ]
)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    train_loader, val_loader = configure_dataloaders(args)
    model = configure_model(args)
    optimizer, scheduler = configure_optimizers(args, model)
    criterion = configure_criterion(args, _rooms(train_loader.dataset))
    regularizer = configure_regularizer(args)

    num_train, num_val = len(train_loader.sampler), len(val_loader.sampler)  # type: ignore[arg-type]
    print(f"Training {args.model!r} on {num_train} blocks per epoch, validating on {num_val}.")
    output: Optional[Path] = Path(args.output) if args.output else None
    best_loss = float("inf")
    for epoch in range(args.epochs):
        set_bn_momentum(model, max(0.1 * 0.5 ** (epoch // 10), 0.01))
        loss = train_one_epoch(model, criterion, regularizer, optimizer, train_loader, args.device)
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
    parser = ArgumentParser(description="Train PointNet++ on S3DIS.")
    parser.add_argument("--model", default=MODELS[0], choices=MODELS)
    parser.add_argument("--area", default=5, type=int, choices=range(1, 7), help="Held-out area.")
    parser.add_argument("--root", default=DATA_DIR, help="Dataset root directory.")
    parser.add_argument("--output", default=None, help="Directory for the checkpoints (default: none saved).")
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--epochs", default=32, type=int)
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--min-lr", default=1e-5, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--num-workers", default=NUM_WORKERS, type=int)
    parser.add_argument("--eval-every", default=1, type=int, help="Validate every this many epochs.")
    parser.add_argument("--limit-train-batches", default=None, type=int)
    parser.add_argument("--limit-val-batches", default=None, type=int)
    parser.add_argument(
        "--download", action="store_true", help="Download S3DIS if missing (requires accepting its terms)."
    )
    parser.add_argument("--force-process", action="store_true", help="Force re-processing the dataset.")
    return parser.parse_args()


def configure_datasets(args: Namespace) -> tuple[Dataset, Dataset]:
    val_areas: List[S3DISArea] = [area for area in S3DIS_AREAS if area == f"Area_{args.area}"]
    train_areas: List[S3DISArea] = [area for area in S3DIS_AREAS if area not in val_areas]
    common: Dict[str, Any] = dict(
        root=args.root, download=args.download, force_process=args.force_process, num_workers=args.num_workers
    )

    train_dataset: Dataset
    val_dataset: Dataset
    train_dataset = S3DIS(areas=train_areas, transform=TRAIN_TRANSFORM, **common)
    val_dataset = S3DIS(areas=val_areas, transform=VAL_TRANSFORM, **common)

    if args.limit_train_batches is not None:
        train_dataset = Subset(
            train_dataset, range(min(args.limit_train_batches * args.batch_size, len(train_dataset)))
        )
    if args.limit_val_batches is not None:
        val_dataset = Subset(val_dataset, range(min(args.limit_val_batches * args.batch_size, len(val_dataset))))

    return train_dataset, val_dataset


def configure_dataloaders(args: Namespace) -> tuple[DataLoader, DataLoader]:
    train_dataset, val_dataset = configure_datasets(args)

    # Rooms are drawn in proportion to their size.
    train_sampler = RepeatSampler(draws_per_room(train_dataset))
    val_sampler = RepeatSampler(draws_per_room(val_dataset), shuffle=False)
    train_loader = PointCloudDataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        drop_last=len(train_sampler) >= args.batch_size,
        num_workers=args.num_workers,
    )
    val_loader = PointCloudDataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        drop_last=len(val_sampler) >= args.batch_size,
        num_workers=args.num_workers,
    )

    return train_loader, val_loader


def configure_model(args: Namespace) -> nn.Module:
    if args.model.startswith("pointnet2") and args.area != 5:
        raise SystemExit("The PointNet++ registry entry is defined for Area 5 only.")
    return create_model(args.model, task="semantic-segmentation", pretrained=False).to(args.device)


def configure_optimizers(args: Namespace, model: nn.Module) -> tuple[Optimizer, LRScheduler]:
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.weight_decay
    )

    # Stepped once per epoch: x0.7 every 10 epochs with a floor, as the reference's manual decay.
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda epoch: max(0.7 ** (epoch // 10), args.min_lr / args.lr)
    )
    return optimizer, scheduler


def configure_criterion(args: Namespace, dataset: S3DIS) -> nn.Module:
    """Class weights of the reference: (max class frequency / class frequency)^(1/3) over the training points."""
    counts = torch.zeros(NUM_CLASSES)
    for room in dataset.data:
        counts += torch.bincount(room[DataKeys.SEGMENT], minlength=NUM_CLASSES)[:NUM_CLASSES].float()
    frequency = counts / counts.sum()
    weights = (frequency.max() / frequency) ** (1.0 / 3.0)

    return nn.CrossEntropyLoss(weight=weights.to(args.device))


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
) -> float:
    model.train()
    total_loss = 0.0
    for data in tqdm(loader, desc="Training", leave=False):
        data = {key: value.to(device) for key, value in data.items() if isinstance(value, Tensor)}
        logits = model(data[DataKeys.X], data[DataKeys.POS], data[DataKeys.BATCH])
        loss = criterion(logits, data[DataKeys.SEGMENT])
        if regularizer is not None:
            loss = loss + REGULARIZER_WEIGHT * regularizer(model.get_submodule("encoder.ftnet"))

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
        confusion += confusion_matrix(logits.argmax(dim=1), data[DataKeys.SEGMENT], NUM_CLASSES, ignore_index=-1)

    present = confusion.sum(dim=1) > 0
    accuracy = confusion.diag().sum() / confusion.sum()
    mean_accuracy = (confusion.diag() / confusion.sum(dim=1).clamp_min(1))[present].mean()
    return {"mIoU": float(intersection_over_union(confusion)), "OA": accuracy.item(), "mAcc": mean_accuracy.item()}


def _rooms(dataset: Dataset) -> S3DIS:
    """The `S3DIS` rooms behind the (subset) dataset."""
    while not isinstance(dataset, S3DIS):
        dataset = dataset.dataset  # type: ignore[attr-defined]
    return dataset


def draws_per_room(dataset: Dataset) -> List[int]:
    """Each room is drawn about once per `NUM_POINTS` of its points (at least once)."""
    if isinstance(dataset, Subset):
        counts = draws_per_room(dataset.dataset)
        return [counts[index] for index in dataset.indices]
    return [max(1, round(room[DataKeys.POS].shape[0] / NUM_POINTS)) for room in _rooms(dataset).data]


if __name__ == "__main__":
    main()
