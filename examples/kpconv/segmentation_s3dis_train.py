"""Train KP-FCNN (rigid or deformable) on S3DIS, following KPConv-PyTorch's `S3DISConfig`."""

import os
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import torch_pointcloud.transforms as T
from torch_pointcloud.config import DATA_DIR
from torch_pointcloud.datasets import S3DIS, RepeatSampler
from torch_pointcloud.datasets.s3dis import S3DIS_AREAS, S3DISArea
from torch_pointcloud.losses import KPConvDeformRegularizer
from torch_pointcloud.metrics import confusion_matrix, intersection_over_union
from torch_pointcloud.models import create_model
from torch_pointcloud.optim import param_groups
from torch_pointcloud.utils.data import DataKeys, PointCloudDataLoader
from torch_pointcloud.utils.random import seed_everything

CPU_COUNT = os.cpu_count()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = CPU_COUNT // 2 if CPU_COUNT is not None else 0
# Input sphere radius of each released checkpoint (`in_radius`).
MODELS = {
    "kpfcnn-base.s3dis-area5.hugues-thomas": 1.8,
    "kpfcnn-base-sm.s3dis-area5.hugues-thomas": 1.2,
    "kpfcnn-base-deform.s3dis-area5.hugues-thomas": 1.5,
    "kpfcnn-base-sm-deform.s3dis-area5.hugues-thomas": 1.2,
}
NUM_CLASSES = 13
GRID_SIZE = 0.03  # first_subsampling_dl
SPHERES_PER_EPOCH = 500 * 6  # epoch_steps x batch_num
VAL_SPHERES = 50 * 6  # validation_size x batch_num
POINT_KEYS = [DataKeys.POS, DataKeys.COLOR, DataKeys.SEGMENT]

# Features = ones + rgb + z of the (augmented) sphere.
FEATURE_TRANSFORMS = [
    T.Slice(keys=DataKeys.POS, start=2, stop=3, dim=1, dst_keys="height"),
    T.OnesLike(keys="height", dst_keys="ones"),
    T.Cat(keys=["ones", DataKeys.COLOR, "height"], dst_key=DataKeys.X),
]


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    train_loader, val_loader = configure_dataloaders(args)
    model = configure_model(args)
    optimizer, scheduler = configure_optimizers(args, model)
    criterion = configure_criterion(args)
    regularizer = configure_regularizer(args)

    num_train, num_val = len(train_loader.sampler), len(val_loader.sampler)  # type: ignore[arg-type]
    print(f"Training {args.model!r} (radius {args.radius}) on {num_train} spheres per epoch, validating on {num_val}.")
    output: Optional[Path] = Path(args.output) if args.output else None
    best_loss = float("inf")
    for epoch in range(args.epochs):
        loss = train_one_epoch(
            model, criterion, regularizer, optimizer, train_loader, args.device, args.grad_clip_value
        )
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
    parser = ArgumentParser(description="Train KP-FCNN on S3DIS.")
    parser.add_argument("--model", default=next(iter(MODELS)), choices=list(MODELS))
    parser.add_argument("--area", default=5, type=int, choices=range(1, 7), help="Held-out area.")
    parser.add_argument("--radius", default=None, type=float, help="Input sphere radius (default: the checkpoint's).")
    parser.add_argument("--root", default=DATA_DIR, help="Dataset root directory.")
    parser.add_argument("--output", default=None, help="Directory for the checkpoints (default: none saved).")
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--epochs", default=500, type=int)
    parser.add_argument("--batch-size", default=6, type=int)
    parser.add_argument("--lr", default=1e-2, type=float)
    parser.add_argument("--momentum", default=0.98, type=float)
    parser.add_argument("--lr-decay", default=0.1 ** (1 / 150), type=float, help="Learning-rate factor per epoch.")
    parser.add_argument("--grad-clip-value", default=100.0, type=float)
    parser.add_argument("--num-workers", default=NUM_WORKERS, type=int)
    parser.add_argument("--eval-every", default=1, type=int, help="Validate every this many epochs.")
    parser.add_argument("--limit-train-batches", default=None, type=int)
    parser.add_argument("--limit-val-batches", default=None, type=int)
    parser.add_argument(
        "--download", action="store_true", help="Download S3DIS if missing (requires accepting its terms)."
    )
    parser.add_argument("--force-process", action="store_true", help="Force re-processing the dataset.")
    args = parser.parse_args()
    if args.radius is None:
        args.radius = MODELS[args.model]
    return args


def configure_train_transform(args: Namespace) -> T.Transform:
    # Grid subsampling and a random input sphere centered on its center point, augmented, then the features.
    return T.Compose(
        [
            T.KeepItems(keys=POINT_KEYS),
            T.Voxelize(  # grid subsampling: averaged coordinates and colors
                pos_key=DataKeys.POS,
                pos_reduce="mean",
                keys=[DataKeys.COLOR, DataKeys.SEGMENT],
                reduce=["mean", "first"],
                size=GRID_SIZE,
                method="grid",
            ),
            T.Scale(keys=DataKeys.COLOR, scale=1.0 / 255),
            T.SphereCrop(
                pos_key=DataKeys.POS,
                radius=args.radius,
                keys=[DataKeys.COLOR, DataKeys.SEGMENT],
                center="random_point",
                dst_center_key="center",
                seed=None,
            ),
            T.SubtractItems(keys=DataKeys.POS, sub_keys="center"),
            T.RandomRotate(keys=DataKeys.POS, axis=2, angle_range=(0.0, 360.0)),  # augment_rotation 'vertical'
            T.RandomScale(keys=DataKeys.POS, scale_range=(0.9, 1.1), anisotropic=True),
            T.RandomFlip(keys=DataKeys.POS, axes=[0], p=0.5),  # augment_symmetries [True, False, False]
            T.RandomJitter(keys=DataKeys.POS, sigma=0.001, clip=None),  # augment_noise
            T.RandomApply([T.Scale(keys=DataKeys.COLOR, scale=0.0)], p=0.2),  # augment_color 0.8
            *FEATURE_TRANSFORMS,
        ]
    )


def configure_val_transform(args: Namespace) -> T.Transform:
    # The validation spheres are drawn from a fixed seed, so every validation scores the same spheres.
    return T.Compose(
        [
            T.KeepItems(keys=POINT_KEYS),
            T.Voxelize(  # grid subsampling: averaged coordinates and colors
                pos_key=DataKeys.POS,
                pos_reduce="mean",
                keys=[DataKeys.COLOR, DataKeys.SEGMENT],
                reduce=["mean", "first"],
                size=GRID_SIZE,
                method="grid",
            ),
            T.Scale(keys=DataKeys.COLOR, scale=1.0 / 255),
            T.SphereCrop(
                pos_key=DataKeys.POS,
                radius=args.radius,
                keys=[DataKeys.COLOR, DataKeys.SEGMENT],
                center="random_point",
                dst_center_key="center",
                seed=0,
            ),
            T.SubtractItems(keys=DataKeys.POS, sub_keys="center"),
            *FEATURE_TRANSFORMS,
        ]
    )


def configure_datasets(args: Namespace) -> tuple[Dataset, Dataset]:
    val_areas: List[S3DISArea] = [area for area in S3DIS_AREAS if area == f"Area_{args.area}"]
    train_areas: List[S3DISArea] = [area for area in S3DIS_AREAS if area not in val_areas]
    common: Dict[str, Any] = dict(
        root=args.root, download=args.download, force_process=args.force_process, num_workers=args.num_workers
    )

    train_dataset: Dataset
    val_dataset: Dataset
    train_transform = configure_train_transform(args)
    train_dataset = S3DIS(areas=train_areas, transform=train_transform, **common)
    val_transform = configure_val_transform(args)
    val_dataset = S3DIS(areas=val_areas, transform=val_transform, **common)
    return train_dataset, val_dataset


def configure_dataloaders(args: Namespace) -> tuple[DataLoader, DataLoader]:
    train_dataset, val_dataset = configure_datasets(args)

    # Each room is drawn in proportion to its size; a batch limit keeps the first draws of the epoch.
    train_draws = draws_per_room(train_dataset, SPHERES_PER_EPOCH)
    val_draws = draws_per_room(val_dataset, VAL_SPHERES)
    if args.limit_train_batches is not None:
        train_draws = first_draws(train_draws, args.limit_train_batches * args.batch_size)
    if args.limit_val_batches is not None:
        val_draws = first_draws(val_draws, args.limit_val_batches * args.batch_size)

    train_loader = PointCloudDataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=RepeatSampler(train_draws),
        num_workers=args.num_workers,
    )
    val_loader = PointCloudDataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=RepeatSampler(val_draws, shuffle=False),
        num_workers=args.num_workers,
    )

    return train_loader, val_loader


def configure_model(args: Namespace) -> nn.Module:
    if args.area != 5:
        raise SystemExit("The KP-FCNN registry entries are defined for Area 5 only.")
    return create_model(args.model, task="semantic-segmentation", pretrained=False).to(args.device)


def configure_optimizers(args: Namespace, model: nn.Module) -> tuple[Optimizer, LRScheduler]:
    groups = param_groups(model, overrides=[dict(keyword="offset", lr=args.lr * 0.1)])  # deform_lr_factor
    optimizer = torch.optim.SGD(groups, lr=args.lr, momentum=args.momentum)

    # Stepped once per epoch: lr x 0.1^(1/150) per epoch.
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay)
    return optimizer, scheduler


def configure_criterion(args: Namespace) -> nn.Module:
    return nn.CrossEntropyLoss()


def configure_regularizer(args: Namespace) -> nn.Module:
    return KPConvDeformRegularizer(fitting_power=1.0, repulse_extent=1.2)


def train_one_epoch(
    model: nn.Module,
    criterion: nn.Module,
    regularizer: nn.Module,
    optimizer: Optimizer,
    loader: DataLoader,
    device: str,
    grad_clip_value: float,
) -> float:
    model.train()
    total_loss = 0.0
    for data in tqdm(loader, desc="Training", leave=False):
        data = {key: value.to(device) for key, value in data.items() if isinstance(value, Tensor)}
        logits = model(data[DataKeys.X], data[DataKeys.POS], data[DataKeys.BATCH])
        loss = criterion(logits, data[DataKeys.SEGMENT]) + regularizer(model)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), grad_clip_value)
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


def draws_per_room(dataset: Dataset, total: int) -> List[int]:
    """Spheres drawn from each room per epoch, in proportion to its points."""
    rooms = dataset
    while not isinstance(rooms, S3DIS):
        rooms = rooms.dataset  # type: ignore[attr-defined]

    counts = torch.tensor([float(room[DataKeys.POS].shape[0]) for room in rooms.data])
    return [max(1, int(round(float(count) / float(counts.sum()) * total))) for count in counts]


def first_draws(draws: List[int], budget: int) -> List[int]:
    """The per-room draws truncated to the first `budget` draws, rooms in order."""
    kept = []
    for count in draws:
        kept.append(min(count, budget))
        budget -= kept[-1]
    return kept


if __name__ == "__main__":
    main()
