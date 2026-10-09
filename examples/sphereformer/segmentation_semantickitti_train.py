"""Train SphereFormer on SemanticKITTI following its `semantic_kitti_unet32_spherical_transformer` config."""

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
from torch_pointcloud.optim import PolyLR, param_groups
from torch_pointcloud.utils.data import DataKeys, PointCloudDataLoader
from torch_pointcloud.utils.random import seed_everything

CPU_COUNT = os.cpu_count()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = CPU_COUNT // 2 if CPU_COUNT is not None else 0
MODELS = ["sphereformer.semantickitti"]
NUM_CLASSES = 19
IGNORE_INDEX = 255
VOXEL_SIZE = 0.05
VOXEL_MAX = 120_000
POINT_RANGE_MIN = (-51.2, -51.2, -4.0)
POINT_RANGE_MAX = (51.2, 51.2, 2.4)
SCALE_RANGE = (0.95, 1.05)
TRANSLATION_STD = 0.1
DROP_PATH = 0.3
TRANSFORMER_LR_SCALE = 0.1
POLY_POWER = 0.9
CLASS_WEIGHTS = [
    3.1557,
    8.7029,
    7.8281,
    6.1354,
    6.3161,
    7.9937,
    8.9704,
    10.1922,
    1.6155,
    4.2187,
    1.9385,
    5.5455,
    2.0198,
    2.6261,
    1.3212,
    5.1102,
    2.5492,
    5.8585,
    7.3929,
]
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
        T.RandomRotate(keys=DataKeys.POS, axis=2, angle_range=(-180.0, 180.0)),
        T.RandomFlip(keys=DataKeys.POS, axes=(0, 1), p=0.5),  # none, x, y or both, equally likely
        T.RandomScale(keys=DataKeys.POS, scale_range=SCALE_RANGE, axes=(0, 1)),
        # The reference draws the shift of each axis from a Gaussian of standard deviation TRANSLATION_STD.
        T.RandomTranslate(
            keys=DataKeys.POS,
            translation_range=(-TRANSLATION_STD * math.sqrt(3), TRANSLATION_STD * math.sqrt(3)),
        ),
        T.Cat(keys=[DataKeys.POS, DataKeys.INTENSITY], dst_key=DataKeys.X, dim=1),  # the unclipped coordinates
        T.Clamp(keys=DataKeys.POS, min=POINT_RANGE_MIN, max=POINT_RANGE_MAX),
        T.Voxelize(  # a random point represents each voxel
            pos_key=DataKeys.POS,
            pos_reduce="first",
            keys=[DataKeys.X, DataKeys.SEGMENT],
            reduce="first",
            random_first=True,
            size=VOXEL_SIZE,
            dst_pos_grid_key=DataKeys.POS_GRID,
        ),
        T.SphereCrop(  # the VOXEL_MAX voxels nearest a random one
            pos_key=DataKeys.POS,
            radius=math.inf,
            max_nodes=VOXEL_MAX,
            center="random_point",
            keys=[DataKeys.X, DataKeys.SEGMENT, DataKeys.POS_GRID],
        ),
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
            pos_reduce="mean",
            keys=[DataKeys.X, DataKeys.SEGMENT],
            size=VOXEL_SIZE,
            dst_pos_grid_key=DataKeys.POS_GRID,
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
    optimizer, scheduler = configure_optimizers(args, model, len(train_loader))
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
    parser = ArgumentParser(description="Train SphereFormer on SemanticKITTI with its published configuration.")
    parser.add_argument("--model", default=MODELS[0], choices=MODELS)
    parser.add_argument("--root", default=DATA_DIR, help="Dataset root directory.")
    parser.add_argument(
        "--train-sequences", nargs="+", default=None, help="Training sequences (default: the split's 00-10 but 08)."
    )
    parser.add_argument("--val-sequences", nargs="+", default=None, help="Validation sequences (default: 08).")
    parser.add_argument("--output", default=None, help="Directory for the checkpoints (default: none saved).")
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--seed", default=123, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--batch-size", default=8, type=int, help="The reference's 2 per GPU on 4 GPUs.")
    parser.add_argument("--val-batch-size", default=8, type=int)
    parser.add_argument("--lr", default=0.006, type=float)
    parser.add_argument("--weight-decay", default=0.02, type=float)
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

    # The reference also caps a batch at 1,000,000 points, which 8 scans of at most VOXEL_MAX voxels never reach.
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
    return create_model(args.model, task="semantic-segmentation", pretrained=False, drop_path=DROP_PATH).to(args.device)


def configure_optimizers(args: Namespace, model: nn.Module, steps_per_epoch: int) -> tuple[Optimizer, LRScheduler]:
    # The transformer blocks train at a tenth of the base learning rate.
    groups = param_groups(
        model,
        weight_decay=args.weight_decay,
        overrides=[dict(keyword="transformer_block", lr=args.lr * TRANSFORMER_LR_SCALE)],
    )
    optimizer = torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay)
    # Stepped per iteration; `total_steps + 1` keeps the reference's non-zero rate on the last step.
    scheduler = PolyLR(optimizer, total_steps=args.epochs * steps_per_epoch + 1, power=POLY_POWER)
    return optimizer, scheduler


def configure_criterion(args: Namespace) -> nn.Module:
    return nn.CrossEntropyLoss(weight=torch.tensor(CLASS_WEIGHTS), ignore_index=IGNORE_INDEX).to(args.device)


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
        # The reference shifts the voxel grid of the whole batch by zero or one voxel per axis.
        data[DataKeys.POS_GRID] = data[DataKeys.POS_GRID] + torch.randint(0, 2, (3,), device=device)
        with torch.autocast(torch.device(device).type, dtype=torch.float16, enabled=scaler is not None):
            logits = model(data[DataKeys.X], data[DataKeys.POS], data[DataKeys.POS_GRID], data[DataKeys.BATCH])
            loss = criterion(logits, data[DataKeys.SEGMENT])

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
        logits = model(data[DataKeys.X], data[DataKeys.POS], data[DataKeys.POS_GRID], data[DataKeys.BATCH])
        preds = logits.argmax(dim=1)[data[DataKeys.INVERSE]]
        confusion += confusion_matrix(preds, data[DataKeys.ORIGIN_SEGMENT], NUM_CLASSES, ignore_index=IGNORE_INDEX)

    per_class = confusion.diagonal().float() / confusion.sum(dim=1).clamp(min=1).float()
    present = confusion.sum(dim=1) > 0
    return {
        "mIoU": float(intersection_over_union(confusion)),
        "OA": float(confusion.diagonal().sum()) / float(confusion.sum().clamp(min=1)),
        "mAcc": float(per_class[present].mean()),
    }


if __name__ == "__main__":
    main()
