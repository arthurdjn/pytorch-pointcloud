"""Finetune Point-MAE on ScanObjectNN following its `finetune_scan_*` configs."""

import os
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

import torch_pointcloud.transforms as T
from torch_pointcloud.config import DATA_DIR
from torch_pointcloud.datasets import ScanObjectNN
from torch_pointcloud.metrics import confusion_matrix
from torch_pointcloud.models import create_model
from torch_pointcloud.optim import CosineWarmupLR, param_groups
from torch_pointcloud.utils.data import DataKeys, PointCloudDataLoader
from torch_pointcloud.utils.random import seed_everything

CPU_COUNT = os.cpu_count()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = CPU_COUNT // 2 if CPU_COUNT is not None else 0
# The ScanObjectNN variant of every entry.
MODELS: Dict[str, Dict[str, Any]] = {
    "point-mae-base.scanobjectnn-hardest.yatian-pang": dict(background=True, variant="augmentedrot_scale75"),
    "point-mae-base.scanobjectnn-objbg.yatian-pang": dict(background=True, variant=None),
    "point-mae-base.scanobjectnn-objonly.yatian-pang": dict(background=False, variant=None),
}
PRETRAIN = "point-mae-base.pretrain.yatian-pang"
NUM_CLASSES = 15
NUM_POINTS = 2048  # the whole object
EPOCHS = 300
BATCH_SIZE = 32
LR = 5e-4
WEIGHT_DECAY = 0.05
WARMUP_EPOCHS = 10
LABEL_SMOOTHING = 0.0
WARMUP_LR_INIT = 1e-6
MIN_LR = 1e-6
GRAD_CLIP_NORM = 10.0

# The whole object is used: a shuffle stands in for the random subset of the farthest-point sample.
TRAIN_TRANSFORM = T.Compose(
    [
        T.KeepItems(keys=[DataKeys.POS, DataKeys.LABEL]),
        T.ShufflePoint(keys=DataKeys.POS),
        # `PointcloudScaleAndTranslate`: per-axis scale in [2/3, 3/2] and shift in [-0.2, 0.2].
        T.RandomScale(keys=DataKeys.POS, scale_range=(2.0 / 3.0, 3.0 / 2.0), anisotropic=True),
        T.RandomTranslate(keys=DataKeys.POS, translation_range=(-0.2, 0.2)),
    ]
)
VAL_TRANSFORM = T.Compose([T.KeepItems(keys=[DataKeys.POS, DataKeys.LABEL])])


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
            print(f"  val/OA {val['acc'] * 100:.2f} | val/mAcc {val['mean_acc'] * 100:.2f}")

        if output is not None:
            output.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), output / "last.pt")
            if loss < best_loss:
                best_loss = loss
                torch.save(model.state_dict(), output / "best.pt")


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Finetune Point-MAE on ScanObjectNN from its pretrained encoder.")
    parser.add_argument("--model", default="point-mae-base.scanobjectnn-hardest.yatian-pang", choices=list(MODELS))
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
    parser.add_argument("--download", action="store_true", help="Download the dataset if missing.")
    args = parser.parse_args()
    args.warmup_epochs = min(args.warmup_epochs, max(args.epochs - 1, 0))  # a shortened run keeps a valid schedule
    return args


def configure_datasets(args: Namespace) -> tuple[Dataset, Dataset]:
    entry = MODELS[args.model]
    common: Dict[str, Any] = dict(
        root=args.root,
        partition="main",
        background=entry["background"],
        variant=entry["variant"],
        download=args.download,
    )
    train_dataset: Dataset
    val_dataset: Dataset
    train_dataset = ScanObjectNN(train=True, transform=TRAIN_TRANSFORM, **common)
    val_dataset = ScanObjectNN(train=False, transform=VAL_TRANSFORM, **common)

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
    model = create_model(args.model, task="classification", pretrained=False)
    if not args.from_scratch:
        load_backbone(model, PRETRAIN)

    return model.to(args.device)


def configure_criterion(args: Namespace) -> nn.Module:
    # The label smoothing of the configs puts `1 - eps` on the target class and `eps / (C - 1)` elsewhere: torch's
    # `label_smoothing = eps * C / (C - 1)`.
    return nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING * NUM_CLASSES / (NUM_CLASSES - 1))


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
        logits = model(None, data[DataKeys.POS], data[DataKeys.BATCH])
        loss = criterion(logits, data[DataKeys.LABEL])

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> dict[str, float]:
    model.eval()
    confusion = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long, device=device)
    for data in tqdm(loader, desc="Evaluating", leave=False):
        data = {key: value.to(device) for key, value in data.items() if isinstance(value, Tensor)}
        logits = model(None, data[DataKeys.POS], data[DataKeys.BATCH])
        confusion += confusion_matrix(logits.argmax(dim=1), data[DataKeys.LABEL], num_classes=NUM_CLASSES)

    present = confusion.sum(dim=1) > 0
    accuracy = confusion.diag().sum() / confusion.sum()
    mean_accuracy = (confusion.diag() / confusion.sum(dim=1).clamp_min(1))[present].mean()
    return {"acc": accuracy.item(), "mean_acc": mean_accuracy.item()}


def rename_backbone(state: Dict[str, Tensor]) -> Dict[str, Tensor]:
    """`MAE_encoder.*` of the pretraining model is the classifier's encoder, its final norm named `norm_f`."""
    prefix = "MAE_encoder."
    renamed = {}
    for key, value in state.items():
        if not key.startswith(prefix):
            continue

        key = key[len(prefix) :]
        renamed["norm_f." + key[len("norm.") :] if key.startswith("norm.") else key] = value
    return renamed


def load_backbone(model: nn.Module, name: str) -> None:
    """Load the pretraining entry's encoder into the classifier; its head and class tokens start fresh."""
    state = rename_backbone(create_model(name, pretrained=True).state_dict())
    missing, _ = model.load_state_dict(state, strict=False)
    fresh = sorted({key.split(".")[0] for key in missing})
    print(f"Loaded the encoder of {name!r}; freshly initialized: {', '.join(fresh)}.")


if __name__ == "__main__":
    main()
