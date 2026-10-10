"""Finetune PointGPT (S / B / L) on ModelNet40 (1k or 8k points) from its pretrained encoder, following PointGPT's
`finetune_modelnet*` configs; the hyper-parameters of each entry default to the config's."""

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
from torch_pointcloud.datasets import ModelNetNormalResampled
from torch_pointcloud.metrics import confusion_matrix
from torch_pointcloud.models import create_model
from torch_pointcloud.optim import CosineWarmupLR, param_groups
from torch_pointcloud.utils.data import DataKeys, PointCloudDataLoader
from torch_pointcloud.utils.random import seed_everything

CPU_COUNT = os.cpu_count()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = CPU_COUNT // 2 if CPU_COUNT is not None else 0
NUM_CLASSES = 40
WARMUP_EPOCHS = 10
WARMUP_LR_INIT = 1e-6
MIN_LR = 1e-6
GRAD_CLIP_NORM = 10.0
# Every entry: the pretraining entry its encoder starts from, its points per shape (`point_all` is the
# farthest-point sample the training points are drawn from) and its config's optimizer and schedule.
MODELS: Dict[str, Dict[str, Any]] = {
    "pointgpt-s.modelnet40.guangyan-chen": dict(
        pretrain="pointgpt-s.pretrain.guangyan-chen",
        num_points=1024,
        point_all=1200,
        epochs=300,
        batch_size=128,
        lr=1e-4,
        weight_decay=0.05,
    ),
    "pointgpt-s.modelnet40-8k.guangyan-chen": dict(
        pretrain="pointgpt-s.pretrain.guangyan-chen",
        num_points=8192,
        point_all=8192,
        epochs=300,
        batch_size=32,
        lr=1e-4,
        weight_decay=0.005,
    ),
    "pointgpt-b.modelnet40.guangyan-chen": dict(
        pretrain="pointgpt-b.pretrain.guangyan-chen",
        num_points=1024,
        point_all=1200,
        epochs=50,
        batch_size=128,
        lr=1e-4,
        weight_decay=0.05,
    ),
    "pointgpt-b.modelnet40-8k.guangyan-chen": dict(
        pretrain="pointgpt-b.pretrain.guangyan-chen",
        num_points=8192,
        point_all=8192,
        epochs=50,
        batch_size=32,
        lr=5e-5,
        weight_decay=0.005,
    ),
    "pointgpt-l.modelnet40.guangyan-chen": dict(
        pretrain="pointgpt-l.pretrain.guangyan-chen",
        num_points=1024,
        point_all=1200,
        epochs=50,
        batch_size=128,
        lr=1e-4,
        weight_decay=0.05,
    ),
    "pointgpt-l.modelnet40-8k.guangyan-chen": dict(
        pretrain="pointgpt-l.pretrain.guangyan-chen",
        num_points=8192,
        point_all=8192,
        epochs=50,
        batch_size=32,
        lr=5e-5,
        weight_decay=0.005,
    ),
}


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    train_loader, val_loader = configure_dataloaders(args)
    model = configure_model(args)
    optimizer, scheduler = configure_optimizers(args, model)
    criterion = configure_criterion(args)

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
    parser = ArgumentParser(description="Finetune PointGPT on ModelNet40.")
    parser.add_argument("--model", default="pointgpt-s.modelnet40.guangyan-chen", choices=list(MODELS))
    parser.add_argument("--root", default=DATA_DIR, help="Dataset root directory.")
    parser.add_argument("--output", default=None, help="Directory for the checkpoints (default: none saved).")
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--epochs", default=None, type=int, help="Default: the reference epochs of the model.")
    parser.add_argument("--batch-size", default=None, type=int, help="Default: the reference batch size of the model.")
    parser.add_argument("--lr", default=None, type=float, help="Default: the reference learning rate of the model.")
    parser.add_argument("--weight-decay", default=None, type=float, help="Default: the reference weight decay.")
    parser.add_argument("--warmup-epochs", default=WARMUP_EPOCHS, type=int)
    parser.add_argument("--from-scratch", action="store_true", help="Do not load the pretraining encoder.")
    parser.add_argument("--num-workers", default=NUM_WORKERS, type=int)
    parser.add_argument("--eval-every", default=1, type=int, help="Validate every this many epochs.")
    parser.add_argument("--limit-train-batches", default=None, type=int)
    parser.add_argument("--limit-val-batches", default=None, type=int)
    parser.add_argument("--download", action="store_true", help="Download the dataset if missing.")
    parser.add_argument("--force-process", action="store_true", help="Force re-processing ModelNet40.")
    args = parser.parse_args()

    entry = MODELS[args.model]
    if args.epochs is None:
        args.epochs = entry["epochs"]
    if args.batch_size is None:
        args.batch_size = entry["batch_size"]
    if args.lr is None:
        args.lr = entry["lr"]
    if args.weight_decay is None:
        args.weight_decay = entry["weight_decay"]
    args.warmup_epochs = min(args.warmup_epochs, max(args.epochs - 1, 0))  # a shortened run keeps a valid schedule
    return args


def configure_train_transform(args: Namespace) -> T.Transform:
    entry = MODELS[args.model]
    num_points, point_all = entry["num_points"], entry["point_all"]

    # Farthest-point sample the 10 000-point shape to `point_all`, then a random subset of `num_points` of them.
    steps: List[T.Transform] = [
        T.KeepItems(keys=[DataKeys.POS, DataKeys.LABEL]),
        T.Rescale(keys=DataKeys.POS, method="centroid"),  # pc_normalize
        T.FarthestPointSample(pos_key=DataKeys.POS, num_samples=point_all, random_start=True),
    ]
    if point_all > num_points:
        steps.append(T.RandomSample(keys=DataKeys.POS, num_samples=num_points))

    # `PointcloudScaleAndTranslate`: per-axis scale in [2/3, 3/2] and shift in [-0.2, 0.2].
    steps.append(T.RandomScale(keys=DataKeys.POS, scale_range=(2.0 / 3.0, 3.0 / 2.0), anisotropic=True))
    steps.append(T.RandomTranslate(keys=DataKeys.POS, translation_range=(-0.2, 0.2)))
    return T.Compose(steps)


def configure_val_transform(args: Namespace) -> T.Transform:
    num_points = MODELS[args.model]["num_points"]

    # Farthest-point sample the 10 000-point shape.
    return T.Compose(
        [
            T.KeepItems(keys=[DataKeys.POS, DataKeys.LABEL]),
            T.Rescale(keys=DataKeys.POS, method="centroid"),  # pc_normalize
            T.FarthestPointSample(pos_key=DataKeys.POS, num_samples=num_points, random_start=False),
        ]
    )


def configure_datasets(args: Namespace) -> tuple[Dataset, Dataset]:
    common: Dict[str, Any] = dict(
        root=args.root,
        variant="40",
        download=args.download,
        force_process=args.force_process,
        num_workers=args.num_workers,
    )

    train_dataset: Dataset
    val_dataset: Dataset
    train_transform = configure_train_transform(args)
    train_dataset = ModelNetNormalResampled(train=True, transform=train_transform, **common)
    val_transform = configure_val_transform(args)
    val_dataset = ModelNetNormalResampled(train=False, transform=val_transform, **common)

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
        load_backbone(model, MODELS[args.model]["pretrain"])

    return model.to(args.device)


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


def configure_criterion(args: Namespace) -> nn.Module:
    return nn.CrossEntropyLoss()


def load_backbone(model: nn.Module, name: str) -> None:
    """Load the pretraining entry's encoder into the classifier; its head and class tokens start fresh."""
    state = rename_backbone(create_model(name, pretrained=True).state_dict())
    missing, _ = model.load_state_dict(state, strict=False)
    fresh = sorted({key.split(".")[0] for key in missing})
    print(f"Loaded the encoder of {name!r}; freshly initialized: {', '.join(fresh)}.")


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
    """The pretraining model's encoder keys are the classifier's."""
    return state


if __name__ == "__main__":
    main()
