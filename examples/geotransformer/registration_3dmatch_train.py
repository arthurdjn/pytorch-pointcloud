"""Train GeoTransformer on 3DMatch following the authors' `trainval.py` configuration."""

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
from torch_pointcloud.datasets import ThreeDMatch
from torch_pointcloud.losses import GeoTransformerLoss
from torch_pointcloud.metrics import feature_matching_recall, inlier_ratio, rotation_error, translation_error
from torch_pointcloud.models import GeoTransformerRegistration, create_model
from torch_pointcloud.utils.data import DataKeys, PointCloudDataLoader
from torch_pointcloud.utils.random import seed_everything

CUDA_AVAILABLE = torch.cuda.is_available()
CPU_COUNT = os.cpu_count()
DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"
NUM_WORKERS = CPU_COUNT // 2 if CPU_COUNT is not None else 0
SEED = 42
POINT_LIMIT = 30000
INLIER_RADIUS = 0.1
INLIER_RATIO_THRESHOLD = 0.05
POSITIVE_RADIUS = 0.05

TRAIN_TRANSFORM = T.Compose(
    [
        T.RandomSample(keys=DataKeys.POS, num_samples=POINT_LIMIT, allow_fewer=True),
        T.RandomSample(keys=DataKeys.REF_POS, num_samples=POINT_LIMIT, allow_fewer=True),
        T.RandomPairRotate(src_keys=DataKeys.POS, ref_keys=DataKeys.REF_POS, transform_key=DataKeys.TRANSFORM),
        T.RandomJitter(keys=[DataKeys.POS, DataKeys.REF_POS], sigma=0.0025, clip=None, distribution="uniform"),
    ]
)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    train_loader, val_loader = configure_dataloaders(args)
    model = configure_model(args)
    optimizer, scheduler = configure_optimizers(args, model)
    criterion = configure_criterion(args)

    num_train, num_val = len(train_loader.dataset), len(val_loader.dataset)  # type: ignore[arg-type]
    print(f"Training {args.model!r} on {num_train} pairs, validating on {num_val}.")
    output: Optional[Path] = Path(args.output) if args.output else None
    best_loss = float("inf")
    for epoch in range(args.epochs):
        loss = train_one_epoch(model, criterion, optimizer, train_loader, args.device)
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        print(f"Epoch {epoch + 1}/{args.epochs}  lr={lr:.2e}  train/loss {loss:.4f}")

        if (epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs:
            val = evaluate(model, val_loader, args.device)
            print("  " + " | ".join(f"val/{key} {value:.2f}" for key, value in val.items()))

        if output is not None:
            output.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), output / "last.pt")
            if loss < best_loss:
                best_loss = loss
                torch.save(model.state_dict(), output / "best.pt")


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Train GeoTransformer on 3DMatch.")
    parser.add_argument("--model", default="geotransformer.3dmatch.zheng-qin", help="Registered registration model")
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--root", default=DATA_DIR, help="Dataset root directory.")
    parser.add_argument("--output", default=None, help="Directory for the checkpoints (default: none saved).")
    parser.add_argument("--seed", default=SEED, type=int)
    parser.add_argument("--epochs", default=40, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--weight-decay", default=1e-6, type=float)
    parser.add_argument("--lr-decay", default=0.95, type=float, help="Learning rate factor applied every epoch.")
    parser.add_argument("--num-workers", default=NUM_WORKERS, type=int)
    parser.add_argument("--eval-every", default=1, type=int, help="Validate every this many epochs.")
    parser.add_argument("--limit-train-batches", default=None, type=int)
    parser.add_argument("--limit-val-batches", default=None, type=int)
    parser.add_argument("--download", action="store_true", help="Download the dataset if missing.")
    parser.add_argument("--force-process", action="store_true", help="Force re-processing the dataset.")
    return parser.parse_args()


def configure_datasets(args: Namespace) -> tuple[Dataset, Dataset]:
    common: Dict[str, Any] = dict(
        root=args.root, download=args.download, force_process=args.force_process, num_workers=args.num_workers
    )
    train_dataset: Dataset
    val_dataset: Dataset
    train_dataset = ThreeDMatch(split="train", transform=TRAIN_TRANSFORM, **common)
    val_dataset = ThreeDMatch(split="val", **common)

    if args.limit_train_batches is not None:
        train_dataset = Subset(train_dataset, range(min(args.limit_train_batches, len(train_dataset))))
    if args.limit_val_batches is not None:
        val_dataset = Subset(val_dataset, range(min(args.limit_val_batches, len(val_dataset))))

    return train_dataset, val_dataset


def configure_dataloaders(args: Namespace) -> tuple[DataLoader, DataLoader]:
    train_dataset, val_dataset = configure_datasets(args)
    # The authors train one pair at a time: the superpoint transformer runs per pair.
    train_loader = PointCloudDataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=args.num_workers,
        cat_keys=[DataKeys.REF_POS],
        stack_keys=[DataKeys.TRANSFORM],
    )
    val_loader = PointCloudDataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        cat_keys=[DataKeys.REF_POS],
        stack_keys=[DataKeys.TRANSFORM],
    )
    return train_loader, val_loader


def configure_model(args: Namespace) -> GeoTransformerRegistration:
    model = create_model(args.model, task="registration", pretrained=False)
    assert isinstance(model, GeoTransformerRegistration)
    return model.to(args.device)


def configure_optimizers(args: Namespace, model: nn.Module) -> tuple[Optimizer, LRScheduler]:
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay)
    return optimizer, scheduler


def configure_criterion(args: Namespace) -> GeoTransformerLoss:
    return GeoTransformerLoss(positive_radius=POSITIVE_RADIUS)


def train_one_epoch(
    model: GeoTransformerRegistration,
    criterion: GeoTransformerLoss,
    optimizer: Optimizer,
    loader: DataLoader,
    device: str,
) -> float:
    model.train()
    total_loss = 0.0
    for data in tqdm(loader, desc="Training", leave=False):
        data = {key: value.to(device) if isinstance(value, Tensor) else value for key, value in data.items()}
        output = model(
            None,
            data[DataKeys.POS],
            data[DataKeys.BATCH],
            None,
            data[DataKeys.REF_POS],
            data[DataKeys.BATCH_REF_POS],
            transform=data[DataKeys.TRANSFORM],
        )
        loss = criterion(output, data)["loss"]

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model: GeoTransformerRegistration, loader: DataLoader, device: str) -> Dict[str, float]:
    model.eval()
    inlier_ratios: List[Tensor] = []
    rotation_errors: List[Tensor] = []
    translation_errors: List[Tensor] = []
    for data in tqdm(loader, desc="Evaluating", leave=False):
        data = {key: value.to(device) if isinstance(value, Tensor) else value for key, value in data.items()}
        output = model(
            None,
            data[DataKeys.POS],
            data[DataKeys.BATCH],
            None,
            data[DataKeys.REF_POS],
            data[DataKeys.BATCH_REF_POS],
        )
        registration = model.decode(output)
        target = data[DataKeys.TRANSFORM]
        inlier_ratios.append(
            inlier_ratio(
                registration["src_points"],
                registration["ref_points"],
                target,
                INLIER_RADIUS,
                registration["batch"],
            )
        )
        rotation_errors.append(rotation_error(registration["transform"], target))
        translation_errors.append(translation_error(registration["transform"], target))

    return {
        "IR": 100 * torch.cat(inlier_ratios).mean().item(),
        "FMR": 100 * feature_matching_recall(torch.cat(inlier_ratios), INLIER_RATIO_THRESHOLD),
        "RRE": torch.cat(rotation_errors).mean().item(),
        "RTE": 100 * torch.cat(translation_errors).mean().item(),
    }


if __name__ == "__main__":
    main()
