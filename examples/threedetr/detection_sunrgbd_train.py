"""Train 3DETR on SUN RGB-D following the authors' `sunrgbd_ep1080` configuration."""

import os
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Optional, cast

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

import torch_pointcloud.transforms as T
from torch_pointcloud.config import DATA_DIR
from torch_pointcloud.datasets import SunRGBD
from torch_pointcloud.losses import ThreeDETRLoss
from torch_pointcloud.metrics import box_average_precision, box_matches
from torch_pointcloud.metrics.detection import BoxMatches
from torch_pointcloud.models import ThreeDETRDetection, create_model, model_info
from torch_pointcloud.ops.box3d import count_points_in_boxes, nms3d
from torch_pointcloud.optim import CosineWarmupLR
from torch_pointcloud.utils.data import DataKeys, PointCloudDataLoader
from torch_pointcloud.utils.random import seed_everything
from torch_pointcloud.utils.types import Boxes3D, Detection3D

CUDA_AVAILABLE = torch.cuda.is_available()
CPU_COUNT = os.cpu_count()
DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"
NUM_WORKERS = CPU_COUNT // 2 if CPU_COUNT is not None else 0
SEED = 42
MODELS = ["3detr.sunrgbd.fair"]
NUM_POINTS = 20000
WARMUP_LR = 1e-6
FINAL_LR = 1e-6
GRAD_CLIP_NORM = 0.1
MATCHER_COSTS = dict(matcher_cls_cost=1.0, matcher_giou_cost=3.0, matcher_center_cost=5.0, matcher_objectness_cost=5.0)
LOSS_WEIGHTS = dict(giou_weight=0.0, no_object_weight=0.1)
SCORE_THRESHOLD = 0.05
NMS_IOU = 0.25
MIN_POINTS = 5
IOU_THRESHOLDS = [0.25, 0.5]


TRAIN_TRANSFORM = T.Compose(
    [
        T.KeepItems(keys=[DataKeys.POS, DataKeys.BOX, DataKeys.LABEL]),
        # The reference also crops a random cuboid holding at least one box before sampling.
        T.RandomSample(keys=DataKeys.POS, num_samples=NUM_POINTS),
        T.RandomFlip(keys=DataKeys.POS, box_key=DataKeys.BOX, axes=(0,), p=0.5),
        T.RandomRotate(keys=DataKeys.POS, box_key=DataKeys.BOX, angle_range=(-30.0, 30.0)),
        T.RandomScale(keys=DataKeys.POS, box_key=DataKeys.BOX, scale_range=(0.85, 1.15)),
    ]
)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    train_loader, val_loader = configure_dataloaders(args)
    model = configure_model(args)
    optimizer, scheduler = configure_optimizers(args, model, len(train_loader))
    criterion = configure_criterion(args, model)

    num_train, num_val = len(train_loader.dataset), len(val_loader.dataset)  # type: ignore[arg-type]
    print(f"Training {args.model!r} on {num_train} scenes, validating on {num_val}.")
    output: Optional[Path] = Path(args.output) if args.output else None
    best_loss = float("inf")
    for epoch in range(args.epochs):
        loss = train_one_epoch(model, criterion, optimizer, scheduler, train_loader, args.device)
        lr = optimizer.param_groups[0]["lr"]
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
    parser = ArgumentParser(description="Train 3DETR on SUN RGB-D.")
    parser.add_argument("--model", default=MODELS[0], choices=MODELS)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--root", default=DATA_DIR, help="Dataset root directory.")
    parser.add_argument("--output", default=None, help="Directory for the checkpoints (default: none saved).")
    parser.add_argument("--seed", default=SEED, type=int)
    parser.add_argument("--epochs", default=1080, type=int)
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--lr", default=7e-4, type=float)
    parser.add_argument("--weight-decay", default=0.1, type=float)
    parser.add_argument("--warmup-epochs", default=9, type=int)
    parser.add_argument("--num-workers", default=NUM_WORKERS, type=int)
    parser.add_argument("--eval-every", default=10, type=int, help="Validate every this many epochs.")
    parser.add_argument("--limit-train-batches", default=None, type=int)
    parser.add_argument("--limit-val-batches", default=None, type=int)
    parser.add_argument("--download", action="store_true", help="Download SUN RGB-D if missing.")
    parser.add_argument("--force-process", action="store_true", help="Force re-processing the dataset.")
    return parser.parse_args()


def configure_val_transform(args: Namespace) -> T.Transform:
    """The registered inference transform of the model (the random 20 000 points)."""
    transform = model_info(args.model, task="detection")["transform"]
    if transform is None:
        raise ValueError(f"{args.model!r} registers no transform.")

    return transform


def configure_datasets(args: Namespace) -> tuple[Dataset, Dataset]:
    common: dict[str, Any] = dict(root=args.root, download=args.download, force_process=args.force_process)

    train_dataset: Dataset
    val_dataset: Dataset
    train_dataset = SunRGBD(train=True, transform=TRAIN_TRANSFORM, **common)
    val_transform = configure_val_transform(args)
    val_dataset = SunRGBD(train=False, transform=val_transform, **common)

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
        num_workers=args.num_workers,
        cat_keys=[DataKeys.BOX, DataKeys.LABEL],
    )
    val_loader = PointCloudDataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        cat_keys=[DataKeys.BOX, DataKeys.LABEL],
    )

    return train_loader, val_loader


def configure_model(args: Namespace) -> ThreeDETRDetection:
    model = cast(ThreeDETRDetection, create_model(args.model, task="detection", pretrained=False))
    return model.to(args.device)


def configure_optimizers(args: Namespace, model: nn.Module, steps_per_epoch: int) -> tuple[Optimizer, LRScheduler]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Stepped every iteration: linear warm-up, then a cosine from the first step down to the final rate.
    warmup_epochs = min(args.warmup_epochs, args.epochs - 1)
    scheduler = CosineWarmupLR(
        optimizer,
        total_steps=args.epochs * steps_per_epoch,
        warmup_steps=warmup_epochs * steps_per_epoch,
        warmup_lr_init=WARMUP_LR,
        lr_min=FINAL_LR,
    )
    return optimizer, scheduler


def configure_criterion(args: Namespace, model: ThreeDETRDetection) -> ThreeDETRLoss:
    """The set-prediction loss, completed with the head geometry of the model (classes, heading bins)."""
    criterion = ThreeDETRLoss(
        num_classes=model.num_classes,
        num_heading_bins=model.num_heading_bins,
        **MATCHER_COSTS,
        **LOSS_WEIGHTS,
    )
    return criterion.to(args.device)


def train_one_epoch(
    model: ThreeDETRDetection,
    criterion: ThreeDETRLoss,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    loader: DataLoader,
    device: str,
) -> float:
    model.train()
    total_loss = 0.0
    for data in tqdm(loader, desc="Training", leave=False):
        data = {key: value.to(device) if isinstance(value, Tensor) else value for key, value in data.items()}
        loss = criterion(model(None, data[DataKeys.POS], data[DataKeys.BATCH]), data)["loss"]

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model: ThreeDETRDetection, loader: DataLoader, device: str) -> dict[str, float]:
    model.eval()
    num_classes = model.num_classes
    matches: list[BoxMatches] = []
    for data in tqdm(loader, desc="Evaluating", leave=False):
        data = {key: value.to(device) if isinstance(value, Tensor) else value for key, value in data.items()}
        pos, batch = data[DataKeys.POS], data[DataKeys.BATCH]
        det = model.decode(model(None, pos, batch))
        boxes, scores, labels, det_batch = det["boxes"], det["scores"], det["labels"], det["batch"]
        counts = count_points_in_boxes(pos, boxes, pos_batch=batch, box_batch=det_batch)
        cand = (counts >= MIN_POINTS).nonzero(as_tuple=False).squeeze(-1)
        keep = cand[nms3d(boxes[cand], scores[cand], NMS_IOU, labels=labels[cand], batch=det_batch[cand])]
        keep = keep[scores[keep] > SCORE_THRESHOLD]

        # Indoor AP convention: score every surviving box against each class by its class probability.
        class_probs = det["class_probs"][keep]
        preds: Detection3D = {
            "boxes": boxes[keep].repeat_interleave(num_classes, dim=0),
            "scores": (class_probs * scores[keep, None]).reshape(-1),
            "labels": torch.arange(num_classes, device=device).repeat(keep.numel()),
            "batch": det_batch[keep].repeat_interleave(num_classes),
        }
        target: Boxes3D = {
            "boxes": data[DataKeys.BOX],
            "labels": data[DataKeys.LABEL].long(),
            "batch": data[DataKeys.BATCH_BOX],
        }
        matches.append(box_matches(preds, target))

    return {
        f"mAP@{threshold:g}": float(box_average_precision(matches, iou_threshold=threshold))
        for threshold in IOU_THRESHOLDS
    }


if __name__ == "__main__":
    main()
