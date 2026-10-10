"""Linear probing of Sonata on ScanNet following Pointcept's `semseg-sonata-v1m1-0a-scannet-lin` config."""

import math
import os
from argparse import ArgumentParser, Namespace
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
from torch_pointcloud.losses import LovaszLoss, SumLoss
from torch_pointcloud.metrics import confusion_matrix, intersection_over_union
from torch_pointcloud.models import create_model
from torch_pointcloud.utils.data import DataKeys, PointCloudDataLoader
from torch_pointcloud.utils.random import seed_everything

CPU_COUNT = os.cpu_count()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = CPU_COUNT // 2 if CPU_COUNT is not None else 0
MODEL = "sonata-lp.scannet20.fair"
PRETRAIN = "sonata-base.pretrain.fair"
GRID_SIZE = 0.02
NUM_CLASSES = 20
POINT_KEYS = [DataKeys.POS, DataKeys.COLOR, DataKeys.NORMAL, DataKeys.SEGMENT]
# The reference's tilt of +-1/64 is in units of pi: +-180/64 degrees.
TILT = 180.0 / 64


TRAIN_TRANSFORM = T.Compose(
    [
        T.Relabel(keys=DataKeys.SEGMENT, labels=range(1, NUM_CLASSES + 1), default=-1),
        T.Divide(keys=DataKeys.COLOR, divisor=255.0),  # NormalizeColor; the chromatic ratios below are relative
        T.Shift(keys=DataKeys.POS, method="bbox", axes=[0, 1]),  # CenterShift(apply_z=True)
        T.Shift(keys=DataKeys.POS, method="min", axes=[2]),
        T.RandomDropout(keys=POINT_KEYS, drop_ratio_range=(0.2, 0.2), p=1.0),
        T.RandomRotate(keys=DataKeys.POS, vector_keys=DataKeys.NORMAL, axis=2, angle_range=(-180.0, 180.0), p=0.5),
        T.RandomRotate(
            keys=DataKeys.POS, vector_keys=DataKeys.NORMAL, axis=0, angle_range=(-TILT, TILT), center="bbox", p=0.5
        ),
        T.RandomRotate(
            keys=DataKeys.POS, vector_keys=DataKeys.NORMAL, axis=1, angle_range=(-TILT, TILT), center="bbox", p=0.5
        ),
        T.RandomScale(keys=DataKeys.POS, scale_range=(0.9, 1.1)),
        T.RandomFlip(keys=[DataKeys.POS, DataKeys.NORMAL], axes=(0, 1), p=0.5),
        T.RandomJitter(keys=DataKeys.POS, sigma=0.005, clip=0.02),
        T.RandomElasticDistortion(keys=DataKeys.POS, granularity=(0.2, 0.8), magnitude=(0.4, 1.6), p=1.0),
        T.RandomColorAutoContrast(keys=DataKeys.COLOR, blend=None, p=0.2),
        T.RandomColorShift(keys=DataKeys.COLOR, shift_range=(-0.05, 0.05), p=0.95),  # ChromaticTranslation
        T.RandomJitter(keys=DataKeys.COLOR, sigma=0.05, clip=None, p=0.95),  # ChromaticJitter
        T.Clamp(keys=DataKeys.COLOR, min=0.0, max=1.0),
        T.Voxelize(  # GridSample(mode="train"): one random point per voxel keeps its own coordinates
            pos_key=DataKeys.POS,
            pos_reduce="first",
            size=GRID_SIZE,
            method="fnv",
            reduce="first",
            keys=[DataKeys.COLOR, DataKeys.NORMAL, DataKeys.SEGMENT],
            dst_pos_grid_key=DataKeys.POS_GRID,
            random_first=True,
        ),
        T.SphereCrop(
            pos_key=DataKeys.POS,
            radius=math.inf,
            max_nodes=102_400,
            keys=[DataKeys.COLOR, DataKeys.NORMAL, DataKeys.SEGMENT, DataKeys.POS_GRID],
            center="random_point",
        ),
        T.Shift(keys=DataKeys.POS, method="bbox", axes=[0, 1]),  # CenterShift(apply_z=False)
        T.Cat(keys=[DataKeys.POS, DataKeys.COLOR, DataKeys.NORMAL], dst_key=DataKeys.X, dim=1),
    ]
)

VAL_TRANSFORM = T.Compose(
    [
        T.Relabel(keys=DataKeys.SEGMENT, labels=range(1, NUM_CLASSES + 1), default=-1),
        T.Divide(keys=DataKeys.COLOR, divisor=255.0),
        T.Shift(keys=DataKeys.POS, method="bbox", axes=[0, 1]),
        T.Shift(keys=DataKeys.POS, method="min", axes=[2]),
        T.CopyItems(keys=[DataKeys.POS, DataKeys.SEGMENT], dst_keys=[DataKeys.ORIGIN_POS, DataKeys.ORIGIN_SEGMENT]),
        T.Voxelize(
            pos_key=DataKeys.POS,
            pos_reduce="first",
            size=GRID_SIZE,
            method="fnv",
            reduce="first",
            keys=[DataKeys.COLOR, DataKeys.NORMAL, DataKeys.SEGMENT],
            dst_pos_grid_key=DataKeys.POS_GRID,
            dst_inverse_key=DataKeys.INVERSE,
        ),
        T.Shift(keys=DataKeys.POS, method="bbox", axes=[0, 1]),
        T.Cat(keys=[DataKeys.POS, DataKeys.COLOR, DataKeys.NORMAL], dst_key=DataKeys.X, dim=1),
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
    print(f"Training the head of {args.model!r} on {num_train} scenes, validating on {num_val}.")
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
    parser = ArgumentParser(description="Linear-probe the Sonata encoder on ScanNet.")
    parser.add_argument("--model", default=MODEL, choices=[MODEL])
    parser.add_argument("--root", default=DATA_DIR, help="Dataset root directory.")
    parser.add_argument("--output", default=None, help="Directory for the checkpoints (default: none saved).")
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--epochs", default=800, type=int)
    parser.add_argument("--batch-size", default=24, type=int)
    parser.add_argument("--lr", default=2e-3, type=float)
    parser.add_argument("--weight-decay", default=0.02, type=float)
    parser.add_argument("--mix-prob", default=0.8, type=float, help="Probability of Mix3D-merging a batch pairwise.")
    parser.add_argument(
        "--from-scratch", action="store_true", help="Do not load the pretraining encoder (still frozen)."
    )
    parser.add_argument("--no-amp", action="store_true", help="Disable the float16 autocast of the reference.")
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

    # The processed scenes are kept in their scan frame, as the benchmark evaluates the released weights.
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
        mix=T.Mix3D(keys=[DataKeys.POS, DataKeys.POS_GRID, DataKeys.X, DataKeys.SEGMENT], instance_key=None, p=1.0),
        mix_prob=args.mix_prob,
    )
    val_loader = PointCloudDataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    return train_loader, val_loader


def configure_model(args: Namespace) -> nn.Module:
    model = create_model(args.model, task="semantic-segmentation", pretrained=False)
    if not args.from_scratch:
        load_encoder(model, PRETRAIN)

    for parameter in model.get_submodule("encoder").parameters():  # freeze_backbone
        parameter.requires_grad_(False)

    return model.to(args.device)


def configure_optimizers(args: Namespace, model: nn.Module, steps_per_epoch: int) -> tuple[Optimizer, LRScheduler]:
    head_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(head_parameters, lr=args.lr, weight_decay=args.weight_decay)

    # Stepped per iteration: one cycle over the run.
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=args.epochs * steps_per_epoch,
        pct_start=0.05,
        anneal_strategy="cos",
        div_factor=10.0,
        final_div_factor=1000.0,
    )
    return optimizer, scheduler


def configure_criterion(args: Namespace) -> nn.Module:
    return SumLoss([nn.CrossEntropyLoss(ignore_index=-1), LovaszLoss(ignore_index=-1)])


def load_encoder(model: nn.Module, name: str) -> None:
    """Load the pretraining entry into the probe's `encoder`; the linear head starts fresh."""
    state = {"encoder." + key: value for key, value in create_model(name, pretrained=True).state_dict().items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or any(not key.startswith("head.") for key in missing):
        raise RuntimeError(
            f"The {name!r} weights do not match the encoder: missing {missing}, unexpected {unexpected}."
        )

    print(f"Loaded the encoder of {name!r}; freshly initialized: head.")


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
            loss = criterion(
                model(data[DataKeys.X], data[DataKeys.POS_GRID], data[DataKeys.BATCH]), data[DataKeys.SEGMENT].long()
            )

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
        preds = model(data[DataKeys.X], data[DataKeys.POS_GRID], data[DataKeys.BATCH]).argmax(dim=1)
        preds = preds[data[DataKeys.INVERSE]]
        target = data[DataKeys.ORIGIN_SEGMENT]
        confusion += confusion_matrix(preds, target, NUM_CLASSES, ignore_index=-1)

    per_class = confusion.diagonal().float() / confusion.sum(dim=1).clamp(min=1).float()
    present = confusion.sum(dim=1) > 0
    return {
        "mIoU": float(intersection_over_union(confusion)),
        "OA": float(confusion.diagonal().sum()) / float(confusion.sum().clamp(min=1)),
        "mAcc": float(per_class[present].mean()),
    }


if __name__ == "__main__":
    main()
