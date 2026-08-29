"""Train VoteNet on SUN RGB-D with the reference recipe.

NOTE: the reference recipe: 180 epochs, Adam 1e-3 stepped tenfold at 80 / 120 / 160, BatchNorm momentum halved every
20 epochs from 0.5, 20000 points, x-flip / rotation / scale augmentation, validation with the benchmark protocol.

Usage:
    uv run --no-sync python examples/votenet_detection.py --root /path/to/data --output runs/votenet
    uv run --no-sync python examples/votenet_detection.py --epochs 1 --limit-train-batches 2 --limit-val-batches 2
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch import nn
from torch.utils.data import Dataset, Subset
from tqdm import tqdm

import torch_pointcloud.transforms as T
from torch_pointcloud.config import DATA_DIR
from torch_pointcloud.datasets import SunRGBD
from torch_pointcloud.losses import VoteNetLoss
from torch_pointcloud.models import VoteNetDetection, create_model
from torch_pointcloud.utils.box3d import count_points_in_boxes, nms3d
from torch_pointcloud.utils.data import DataKeys, PointCloudDataLoader
from torch_pointcloud.utils.metrics import mean_average_precision3d
from torch_pointcloud.utils.random import seed_everything
from torch_pointcloud.utils.types import Boxes3D, Detection3D

CUDA_AVAILABLE = torch.cuda.is_available()
CPU_COUNT = os.cpu_count()
DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"
NUM_WORKERS = CPU_COUNT // 2 if CPU_COUNT is not None else 0
SEED = 42
NUM_POINTS = 20000
TARGET_KEYS = [
    "center_label",
    "heading_class_label",
    "heading_residual_label",
    "size_class_label",
    "size_residual_label",
    "sem_cls_label",
    "box_label_mask",
    "vote_label",
    "vote_label_mask",
]
SCORE_THRESHOLD = 0.05
NMS_IOU = 0.25
MIN_POINTS = 5
IOU_THRESHOLDS = [0.25, 0.5]


def train_transform(model: VoteNetDetection) -> T.Compose:
    return T.Compose(
        [
            T.AxisMinOffset(keys=DataKeys.POS, axis=2, quantile=0.0099, dst_keys="height"),
            T.RandomSample(keys=[DataKeys.POS, "height"], num_samples=NUM_POINTS),
            T.GenerateVoteLabels(pos_key=DataKeys.POS, box_key=DataKeys.BOX),
            T.RandomFlip(keys=[DataKeys.POS, "vote_label"], box_key=DataKeys.BOX, axes=(0,)),
            T.RandomRotate(keys=[DataKeys.POS, "vote_label"], box_key=DataKeys.BOX, angle_range=(-30.0, 30.0)),
            T.RandomScale(keys=[DataKeys.POS, "vote_label"], box_key=DataKeys.BOX, scale_range=(0.85, 1.15)),
            T.EncodeVoteNetTargets(
                box_key=DataKeys.BOX,
                num_heading_bin=model.num_heading_bin,
                max_num_obj=64,
                mean_sizes=model.mean_sizes,
            ),
            T.Cat(keys=["height"], dst_key=DataKeys.X, dim=1),
        ]
    )


def set_bn_momentum(model: nn.Module, epoch: int) -> float:
    momentum = max(0.5 * 0.5 ** (epoch // 20), 0.001)
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.momentum = momentum
    return momentum


def train_one_epoch(
    model: VoteNetDetection,
    criterion: VoteNetLoss,
    optimizer: torch.optim.Optimizer,
    dataloader: PointCloudDataLoader,
    device: str,
) -> float:
    model.train()
    total_loss = 0.0
    pbar = tqdm(dataloader, total=len(dataloader), desc="Training")
    for data in pbar:
        data = {key: value.to(device) if torch.is_tensor(value) else value for key, value in data.items()}
        optimizer.zero_grad()
        losses = criterion(model(data[DataKeys.X], data[DataKeys.POS], data[DataKeys.BATCH]), data)
        losses["loss"].backward()
        optimizer.step()
        total_loss += losses["loss"].item()
        pbar.set_postfix({"loss": f"{losses['loss'].item():.3f}"})
    return total_loss / max(len(dataloader), 1)


@torch.no_grad()
def evaluate(model: VoteNetDetection, dataloader: PointCloudDataLoader, device: str) -> Dict[str, float]:
    model.eval()
    num_classes = model.num_classes
    preds: List[Detection3D] = []
    targets: List[Boxes3D] = []

    for data in tqdm(dataloader, total=len(dataloader), desc="Evaluating"):
        data = {key: value.to(device) if torch.is_tensor(value) else value for key, value in data.items()}
        pos, batch = data[DataKeys.POS], data[DataKeys.BATCH]
        det = model.decode(model(data[DataKeys.X], pos, batch))
        boxes, scores, labels, det_batch = det["boxes"], det["scores"], det["labels"], det["batch"]
        counts = count_points_in_boxes(pos, boxes, pos_batch=batch, box_batch=det_batch)
        cand = (counts >= MIN_POINTS).nonzero(as_tuple=False).squeeze(-1)
        keep = cand[nms3d(boxes[cand], scores[cand], NMS_IOU, labels=labels[cand], batch=det_batch[cand])]
        keep = keep[scores[keep] > SCORE_THRESHOLD]
        # Indoor AP convention: score every surviving box against each class by its class probability.
        class_probs = det["class_probs"][keep]
        preds.append(
            {
                "boxes": boxes[keep].repeat_interleave(num_classes, dim=0),
                "scores": (class_probs * scores[keep, None]).reshape(-1),
                "labels": torch.arange(num_classes, device=device).repeat(keep.numel()),
                "batch": det_batch[keep].repeat_interleave(num_classes),
            }
        )
        targets.append({"boxes": data[DataKeys.BOX], "labels": data[DataKeys.LABEL], "batch": data[DataKeys.BATCH_BOX]})

    return mean_average_precision3d(preds, targets, iou_thresholds=IOU_THRESHOLDS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train VoteNet on SUN RGB-D.")
    parser.add_argument("--model", default="votenet.sunrgbd.fair", help="Registered detection model name")
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--root", default=DATA_DIR, help="Dataset root directory.")
    parser.add_argument("--output", default=None, help="Directory for the checkpoints (default: none saved).")
    parser.add_argument("--seed", default=SEED, type=int)
    parser.add_argument("--epochs", default=180, type=int)
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--num-workers", default=NUM_WORKERS, type=int)
    parser.add_argument("--eval-every", default=10, type=int, help="Validate every this many epochs.")
    parser.add_argument("--limit-train-batches", default=None, type=int)
    parser.add_argument("--limit-val-batches", default=None, type=int)
    parser.add_argument("--download", action="store_true", help="Download SUN RGB-D if missing.")
    parser.add_argument("--force-process", action="store_true", help="Force re-processing the dataset.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    model, model_info = create_model(args.model, task="detection", pretrained=False, return_info=True)
    assert isinstance(model, VoteNetDetection)
    model.to(args.device)
    criterion = VoteNetLoss(
        num_heading_bin=model.num_heading_bin,
        num_size_cluster=model.num_size_cluster,
        num_classes=model.num_classes,
        mean_sizes=model.mean_sizes,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 120, 160], gamma=0.1)

    train_dataset: Dataset = SunRGBD(
        root=args.root,
        train=True,
        transform=train_transform(model),
        download=args.download,
        force_process=args.force_process,
    )
    val_dataset: Dataset = SunRGBD(
        root=args.root,
        train=False,
        transform=model_info["transform"],
        download=args.download,
        force_process=args.force_process,
    )
    if args.limit_train_batches is not None:
        n = min(args.limit_train_batches * args.batch_size, len(train_dataset))  # type: ignore[arg-type]
        train_dataset = Subset(train_dataset, range(n))
    if args.limit_val_batches is not None:
        n = min(args.limit_val_batches * args.batch_size, len(val_dataset))  # type: ignore[arg-type]
        val_dataset = Subset(val_dataset, range(n))
    train_dataloader = PointCloudDataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        stack_keys=TARGET_KEYS,
        cat_keys=[DataKeys.BOX],
    )
    val_dataloader = PointCloudDataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        cat_keys=[DataKeys.BOX, DataKeys.LABEL],
    )
    print(f"Training {args.model!r} on {len(train_dataset)} scenes, validating on {len(val_dataset)}.")  # type: ignore[arg-type]

    output: Optional[Path] = Path(args.output) if args.output else None
    for epoch in range(args.epochs):
        momentum = set_bn_momentum(model, epoch)
        print(f"Epoch {epoch + 1}/{args.epochs}  lr={scheduler.get_last_lr()[0]:.2e}  bn_momentum={momentum:.4f}")
        loss = train_one_epoch(model, criterion, optimizer, train_dataloader, args.device)
        scheduler.step()
        print(f"  train/loss {loss:.4f}")
        if (epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs:
            metrics = evaluate(model, val_dataloader, args.device)
            print("  " + " | ".join(f"val/{key} {value * 100:.2f}" for key, value in metrics.items()))
        if output is not None:
            output.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), output / "last.pt")


if __name__ == "__main__":
    main()
