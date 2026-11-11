"""Train TR3D on ScanNet, SUN RGB-D or S3DIS following the authors' `tr3d_1xb16_*` configurations."""

import os
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import List, Optional, cast

import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, Dataset, Subset
from torch_geometric.nn.pool.select.topk import topk
from tqdm import tqdm

import torch_pointcloud.transforms as T
from torch_pointcloud.config import DATA_DIR
from torch_pointcloud.datasets import S3DIS, RepeatDataset, ScanNet, SunRGBD
from torch_pointcloud.datasets.s3dis import S3DIS_DETECTION_LABELS, S3DISArea
from torch_pointcloud.datasets.scannet import SCANNET_DETECTION_LABELS
from torch_pointcloud.losses import TR3DLoss
from torch_pointcloud.metrics import box_average_precision, box_matches
from torch_pointcloud.metrics.detection import BoxMatches
from torch_pointcloud.models import TR3DDetection, create_model, model_info
from torch_pointcloud.ops.box3d import nms3d
from torch_pointcloud.utils.data import DataKeys, PointCloudDataLoader
from torch_pointcloud.utils.random import seed_everything
from torch_pointcloud.utils.types import Boxes3D, Detection3D

CUDA_AVAILABLE = torch.cuda.is_available()
CPU_COUNT = os.cpu_count()
DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"
NUM_WORKERS = CPU_COUNT // 2 if CPU_COUNT is not None else 0
SEED = 42
MODELS = {"scannet": "tr3d.scannet.mmdet3d", "sunrgbd": "tr3d.sunrgbd.mmdet3d", "s3dis": "tr3d.s3dis-area5.mmdet3d"}
# Passes over the training scenes per epoch.
REPEATS = {"scannet": 15, "sunrgbd": 5, "s3dis": 13}
NUM_POINTS = 100000
# Pyramid level of every class: 0 for the stride-16 level, 1 for the stride-32 one.
LABEL_LEVELS = {
    "scannet": [0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0],
    "sunrgbd": [1, 1, 1, 0, 0, 1, 0, 0, 1, 0],
    "s3dis": [1, 0, 1, 1, 0],
}
NUM_POSITIVES = 6
# Box columns of the distance-IoU center distance: the oriented loss of the reference measures (x, y, width).
DISTANCE_DIMS = {"scannet": (0, 1, 2), "sunrgbd": (0, 1, 3), "s3dis": (0, 1, 2)}
NMS_PRE = 1000
SCORE_THRESHOLD = 0.01
NMS_IOU = 0.5
IOU_THRESHOLDS = [0.25, 0.5]


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    train_loader, val_loader = configure_dataloaders(args)
    model = configure_model(args)
    optimizer, scheduler = configure_optimizers(args, model)
    criterion = TR3DLoss(
        model.num_classes,
        label_levels=LABEL_LEVELS[args.dataset],
        num_positives=NUM_POSITIVES,
        distance_dims=DISTANCE_DIMS[args.dataset],
    )

    num_train, num_val = len(train_loader.dataset), len(val_loader.dataset)  # type: ignore[arg-type]
    print(f"Training {args.model!r} on {num_train} scenes, validating on {num_val}.")
    output: Optional[Path] = Path(args.output) if args.output else None
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


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Train TR3D 3D detection on an indoor dataset.")
    parser.add_argument("--dataset", default="scannet", choices=list(MODELS))
    parser.add_argument("--model", default=None, help="Registered detection model name (default: the dataset's).")
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--root", default=DATA_DIR, help="Dataset root directory.")
    parser.add_argument("--output", default=None, help="Directory for the checkpoints (default: none saved).")
    parser.add_argument("--seed", default=SEED, type=int)
    parser.add_argument("--epochs", default=12, type=int)
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--num-workers", default=NUM_WORKERS, type=int)
    parser.add_argument("--eval-every", default=12, type=int, help="Validate every this many epochs.")
    parser.add_argument("--limit-train-batches", default=None, type=int)
    parser.add_argument("--limit-val-batches", default=None, type=int)
    args = parser.parse_args()
    args.model = args.model or MODELS[args.dataset]
    return args


def configure_train_transform(args: Namespace) -> T.Transform:
    """Random sampling, flips, a rotation, a scaling and a translation of the scene, then the colours in $[0, 1]$."""
    colors = [
        T.Normalize(keys=DataKeys.COLOR, mean=[0.0, 0.0, 0.0], std=[255.0, 255.0, 255.0]),
        T.Cat(keys=[DataKeys.COLOR], dst_key=DataKeys.X, dim=1),
        T.KeepItems(keys=[DataKeys.POS, DataKeys.X, DataKeys.BOX, DataKeys.LABEL]),
    ]
    if args.dataset == "sunrgbd":
        # The reference SUN RGB-D scenes keep 50000 random points, resampled to 100000 with replacement.
        steps: List[T.Transform] = [
            T.RandomSample(keys=[DataKeys.POS, DataKeys.COLOR], num_samples=50000),
            T.RandomSample(keys=[DataKeys.POS, DataKeys.COLOR], num_samples=NUM_POINTS),
            T.RandomFlip(keys=DataKeys.POS, axes=(0,), p=0.5, box_key=DataKeys.BOX),
            T.RandomRotate(keys=DataKeys.POS, axis=2, angle_range=(-30.0, 30.0), box_key=DataKeys.BOX),
            T.RandomScale(keys=DataKeys.POS, scale_range=(0.85, 1.15), box_key=DataKeys.BOX),
            T.RandomTranslate(
                keys=DataKeys.POS,
                translation_range=(-0.1, 0.1),
                box_key=DataKeys.BOX,
                distribution="normal",
            ),
        ]
        return T.Compose([*steps, *colors])

    # The axis-aligned boxes are refitted on the instances after the rotation.
    keys = [DataKeys.POS, DataKeys.COLOR, DataKeys.SEGMENT, DataKeys.INSTANCE]
    if args.dataset == "scannet":
        steps = [
            T.Relabel(keys=DataKeys.SEGMENT, labels=SCANNET_DETECTION_LABELS, default=-1),
            # Keep a random 33% to 100% of the points.
            T.RandomDropout(keys=keys, drop_ratio_range=(0.0, 0.67)),
            T.RandomFlip(keys=DataKeys.POS, axes=(0,), p=0.5),
            T.RandomFlip(keys=DataKeys.POS, axes=(1,), p=0.5),
            T.RandomRotate(keys=DataKeys.POS, axis=2, angle_range=(-1.146, 1.146)),
            T.RandomScale(keys=DataKeys.POS, scale_range=(0.9, 1.1)),
        ]
    else:
        steps = [
            # The reference rooms start at their minimum corner.
            T.Shift(keys=DataKeys.POS, method="min"),
            T.Relabel(keys=DataKeys.SEGMENT, labels=S3DIS_DETECTION_LABELS, default=-1),
            T.RandomSample(keys=keys, num_samples=NUM_POINTS),
            T.RandomFlip(keys=DataKeys.POS, axes=(0,), p=0.5),
            T.RandomFlip(keys=DataKeys.POS, axes=(1,), p=0.5),
            T.RandomScale(keys=DataKeys.POS, scale_range=(0.95, 1.05)),
        ]
    steps += [
        T.RandomTranslate(keys=DataKeys.POS, translation_range=(-0.1, 0.1), distribution="normal"),
        T.InstanceToBox(ignore_index=-1),
    ]
    return T.Compose([*steps, *colors])


def configure_val_transform(args: Namespace) -> T.Transform:
    """The ground-truth boxes, then the registered inference transform of the model."""
    transform = model_info(args.model, task="detection")["transform"]
    if transform is None:
        raise ValueError(f"{args.model!r} registers no transform.")

    if args.dataset == "scannet":
        steps: List[T.Transform] = [
            T.Relabel(keys=DataKeys.SEGMENT, labels=SCANNET_DETECTION_LABELS, default=-1),
            T.InstanceToBox(ignore_index=-1),
        ]
    elif args.dataset == "s3dis":
        steps = [
            T.Shift(keys=DataKeys.POS, method="min"),
            T.Relabel(keys=DataKeys.SEGMENT, labels=S3DIS_DETECTION_LABELS, default=-1),
            T.InstanceToBox(ignore_index=-1),
        ]
    else:
        steps = [T.RandomSample(keys=[DataKeys.POS, DataKeys.COLOR], num_samples=50000)]

    return T.Compose(
        [*steps, T.KeepItems(keys=[DataKeys.POS, DataKeys.COLOR, DataKeys.BOX, DataKeys.LABEL]), transform]
    )


def configure_datasets(args: Namespace) -> tuple[Dataset, Dataset]:
    train_transform = configure_train_transform(args)
    val_transform = configure_val_transform(args)

    train_dataset: Dataset
    val_dataset: Dataset
    if args.dataset == "scannet":
        train_dataset = ScanNet(root=args.root, split="train", transform=train_transform)
        val_dataset = ScanNet(root=args.root, split="val", transform=val_transform)
    elif args.dataset == "s3dis":
        train_areas: List[S3DISArea] = ["Area_1", "Area_2", "Area_3", "Area_4", "Area_6"]
        train_dataset = S3DIS(root=args.root, areas=train_areas, transform=train_transform)
        val_dataset = S3DIS(root=args.root, areas=["Area_5"], transform=val_transform)
    else:
        train_dataset = SunRGBD(root=args.root, train=True, transform=train_transform)
        val_dataset = SunRGBD(root=args.root, train=False, transform=val_transform)

    if args.limit_train_batches is not None:
        n = min(args.limit_train_batches * args.batch_size, len(train_dataset))
        train_dataset = Subset(train_dataset, range(n))
    else:
        train_dataset = RepeatDataset(train_dataset, REPEATS[args.dataset])
    if args.limit_val_batches is not None:
        val_dataset = Subset(val_dataset, range(min(args.limit_val_batches, len(val_dataset))))

    return train_dataset, val_dataset


def configure_dataloaders(args: Namespace) -> tuple[DataLoader, DataLoader]:
    train_dataset, val_dataset = configure_datasets(args)
    train_loader = PointCloudDataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        cat_keys=[DataKeys.BOX, DataKeys.LABEL],
    )
    val_loader = PointCloudDataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        cat_keys=[DataKeys.BOX, DataKeys.LABEL],
    )
    return train_loader, val_loader


def configure_model(args: Namespace) -> TR3DDetection:
    model = cast(TR3DDetection, create_model(args.model, task="detection", pretrained=False))
    return model.to(args.device)


def configure_optimizers(args: Namespace, model: TR3DDetection) -> tuple[Optimizer, LRScheduler]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Stepped once per epoch: x0.1 after epochs 8 and 11.
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[8, 11], gamma=0.1)
    return optimizer, scheduler


def train_one_epoch(
    model: TR3DDetection,
    criterion: TR3DLoss,
    optimizer: Optimizer,
    loader: DataLoader,
    device: str,
) -> float:
    model.train()
    total_loss = 0.0
    for data in tqdm(loader, desc="Training", leave=False):
        data = {key: value.to(device) if isinstance(value, Tensor) else value for key, value in data.items()}
        loss = criterion(model(data[DataKeys.X], data[DataKeys.POS], data[DataKeys.BATCH]), data)["loss"]

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model: TR3DDetection, loader: DataLoader, device: str) -> dict[str, float]:
    model.eval()
    matches: list[BoxMatches] = []
    for data in tqdm(loader, desc="Evaluating", leave=False):
        data = {key: value.to(device) if isinstance(value, Tensor) else value for key, value in data.items()}
        out = model(data[DataKeys.X], data[DataKeys.POS], data[DataKeys.BATCH])
        det = model.decode(out)
        # Keep the best locations of every scene, then score each of them against every class.
        keep = topk(det["scores"], NMS_PRE, det["batch"])
        box, label = (det["class_scores"][keep] > SCORE_THRESHOLD).nonzero(as_tuple=True)
        boxes, batch = det["boxes"][keep][box], det["batch"][keep][box]
        scores = det["class_scores"][keep][box, label]
        kept = nms3d(boxes, scores, NMS_IOU, labels=label, batch=batch, rotated=True)
        preds: Detection3D = {"boxes": boxes[kept], "scores": scores[kept], "labels": label[kept], "batch": batch[kept]}
        target: Boxes3D = {
            "boxes": data[DataKeys.BOX],
            "labels": data[DataKeys.LABEL],
            "batch": data[DataKeys.BATCH_BOX],
        }
        matches.append(box_matches(preds, target))

    return {
        f"mAP@{threshold:g}": box_average_precision(matches, iou_threshold=threshold) for threshold in IOU_THRESHOLDS
    }


if __name__ == "__main__":
    main()
