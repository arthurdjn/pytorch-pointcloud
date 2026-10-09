"""Finetune Point-MAE for part segmentation on ShapeNetPart, following its `segmentation/main.py` recipe."""

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
from torch_pointcloud.datasets import ConcatDataset, ShapeNetPart
from torch_pointcloud.metrics import part_intersection_over_union, part_mean_intersection_over_union
from torch_pointcloud.models import create_model
from torch_pointcloud.optim import CosineWarmupLR, param_groups
from torch_pointcloud.utils.data import DataKeys, PointCloudDataLoader
from torch_pointcloud.utils.random import seed_everything

CPU_COUNT = os.cpu_count()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = CPU_COUNT // 2 if CPU_COUNT is not None else 0
MODELS = ["point-mae-base.shapenetpart.yatian-pang"]
PRETRAIN = "point-mae-base.pretrain.yatian-pang"
NUM_CATEGORIES = 16
NUM_PARTS = 50
NUM_POINTS = 2048
EPOCHS = 300
BATCH_SIZE = 16
LR = 2e-4
WEIGHT_DECAY = 0.05
WARMUP_EPOCHS = 10
WARMUP_LR_INIT = 1e-6
MIN_LR = 1e-6
GRAD_CLIP_NORM = 10.0
POINT_KEYS = [DataKeys.POS, DataKeys.SEGMENT]

# `pc_normalize`, a random resample with replacement, then the per-shape scale and shift of `provider`.
TRAIN_TRANSFORM = T.Compose(
    [
        T.KeepItems(keys=[*POINT_KEYS, DataKeys.CATEGORY]),
        T.Rescale(keys=DataKeys.POS, method="centroid"),
        T.RandomSample(keys=POINT_KEYS, num_samples=NUM_POINTS, replace=True),
        T.RandomScale(keys=DataKeys.POS, scale_range=(0.8, 1.25)),
        T.RandomTranslate(keys=DataKeys.POS, translation_range=(-0.1, 0.1)),
        T.OneHot(keys=DataKeys.CATEGORY, num_classes=NUM_CATEGORIES),
    ]
)
VAL_TRANSFORM = T.Compose(
    [
        T.KeepItems(keys=[*POINT_KEYS, DataKeys.CATEGORY]),
        T.Rescale(keys=DataKeys.POS, method="centroid"),
        T.RandomSample(keys=POINT_KEYS, num_samples=NUM_POINTS, replace=True),
        T.OneHot(keys=DataKeys.CATEGORY, num_classes=NUM_CATEGORIES),
    ]
)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    train_loader, val_loader = configure_dataloaders(args)
    model = configure_model(args)
    criterion = configure_criterion(args)
    optimizer, scheduler = configure_optimizers(args, model)

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
            print("  " + " | ".join(f"val/{key} {value * 100:.2f}" for key, value in val.items()))

        if output is not None:
            output.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), output / "last.pt")
            if loss < best_loss:
                best_loss = loss
                torch.save(model.state_dict(), output / "best.pt")


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Finetune Point-MAE for part segmentation on ShapeNetPart.")
    parser.add_argument("--model", default=MODELS[0], choices=MODELS)
    parser.add_argument("--root", default=DATA_DIR, help="Dataset root directory.")
    parser.add_argument("--output", default=None, help="Directory for the checkpoints (default: none saved).")
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--epochs", default=EPOCHS, type=int)
    parser.add_argument("--batch-size", default=BATCH_SIZE, type=int)
    parser.add_argument("--lr", default=LR, type=float)
    parser.add_argument("--weight-decay", default=WEIGHT_DECAY, type=float)
    parser.add_argument("--warmup-epochs", default=WARMUP_EPOCHS, type=int)
    parser.add_argument("--from-scratch", action="store_true", help="Do not load the pretraining encoder.")
    parser.add_argument("--num-workers", default=NUM_WORKERS, type=int)
    parser.add_argument("--eval-every", default=1, type=int, help="Validate every this many epochs.")
    parser.add_argument("--limit-train-batches", default=None, type=int)
    parser.add_argument("--limit-val-batches", default=None, type=int)
    parser.add_argument("--force-process", action="store_true", help="Force re-processing the dataset.")
    args = parser.parse_args()
    args.warmup_epochs = min(args.warmup_epochs, max(args.epochs - 1, 0))  # a shortened run keeps a valid schedule
    return args


def configure_datasets(args: Namespace) -> tuple[Dataset, Dataset]:
    common: Dict[str, Any] = dict(root=args.root, force_process=args.force_process, num_workers=args.num_workers)

    # The reference trains on the train + val splits and validates on the test split.
    train_dataset: Dataset
    val_dataset: Dataset
    train_dataset = ConcatDataset(
        [
            ShapeNetPart(split="train", transform=TRAIN_TRANSFORM, **common),
            ShapeNetPart(split="val", transform=TRAIN_TRANSFORM, **common),
        ]
    )
    val_dataset = ShapeNetPart(split="test", transform=VAL_TRANSFORM, **common)

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
    model = create_model(args.model, task="part-segmentation", pretrained=False)
    if not args.from_scratch:
        load_backbone(model, PRETRAIN)

    return model.to(args.device)


def configure_criterion(args: Namespace) -> nn.Module:
    return nn.CrossEntropyLoss()


def configure_optimizers(args: Namespace, model: nn.Module) -> tuple[Optimizer, LRScheduler]:
    groups = param_groups(model, weight_decay=args.weight_decay, no_decay=["token"])
    optimizer = torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay)

    # Stepped once per epoch: cosine with a linear warm-up.
    scheduler = CosineWarmupLR(
        optimizer,
        total_steps=args.epochs,
        warmup_steps=args.warmup_epochs,
        warmup_lr_init=WARMUP_LR_INIT,
        lr_min=MIN_LR,
    )
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
        data = {key: value.to(device) for key, value in data.items() if isinstance(value, Tensor)}
        logits = model(None, data[DataKeys.POS], data[DataKeys.BATCH], data[DataKeys.CATEGORY])
        loss = criterion(logits, data[DataKeys.SEGMENT])

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> dict[str, float]:
    """Instance and class mIoU, each shape predicted among the parts of its own category as the reference does."""
    model.eval()
    part_ids = list(ShapeNetPart.seg_ids.values())
    category_parts = torch.zeros(NUM_CATEGORIES, NUM_PARTS, dtype=torch.bool, device=device)
    for index, parts in enumerate(part_ids):
        category_parts[index, parts] = True

    ious: List[Tensor] = []
    categories: List[Tensor] = []
    for data in tqdm(loader, desc="Evaluating", leave=False):
        data = {key: value.to(device) for key, value in data.items() if isinstance(value, Tensor)}
        logits = model(None, data[DataKeys.POS], data[DataKeys.BATCH], data[DataKeys.CATEGORY])
        category = data[DataKeys.CATEGORY].argmax(dim=1)
        allowed = category_parts[category][data[DataKeys.BATCH]]
        preds = logits.masked_fill(~allowed, float("-inf")).argmax(dim=1).cpu()
        ious.append(
            part_intersection_over_union(
                preds, data[DataKeys.SEGMENT].cpu(), part_ids, category.cpu(), data[DataKeys.BATCH].cpu()
            )
        )
        categories.append(category.cpu())

    shape_ious, category = torch.cat(ious), torch.cat(categories)
    return {
        "ins_mIoU": float(part_mean_intersection_over_union(shape_ious, category)),
        "cls_mIoU": float(part_mean_intersection_over_union(shape_ious, category, average="macro")),
    }


def rename_backbone(state: Dict[str, Tensor]) -> Dict[str, Tensor]:
    """`MAE_encoder.*` of the pretraining model is the segmentation model's encoder, its final norm named `norm_f`."""
    prefix = "MAE_encoder."
    renamed = {}
    for key, value in state.items():
        if not key.startswith(prefix):
            continue

        key = key[len(prefix) :]
        renamed["norm_f." + key[len("norm.") :] if key.startswith("norm.") else key] = value
    return renamed


def load_backbone(model: nn.Module, name: str) -> None:
    """Load the pretraining entry's encoder into the segmentation model; its head and propagation start fresh."""
    state = rename_backbone(create_model(name, pretrained=True).state_dict())
    missing, _ = model.load_state_dict(state, strict=False)
    fresh = sorted({key.split(".")[0] for key in missing})
    print(f"Loaded the encoder of {name!r}; freshly initialized: {', '.join(fresh)}.")


if __name__ == "__main__":
    main()
