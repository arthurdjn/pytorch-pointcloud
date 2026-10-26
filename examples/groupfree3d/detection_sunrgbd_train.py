"""Train Group-Free 3D on SUN RGB-D following the authors' `train_dist.py --dataset sunrgbd` configuration."""

import os
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Optional, cast

import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

import torch_pointcloud.transforms as T
from torch_pointcloud.config import DATA_DIR
from torch_pointcloud.datasets import SunRGBD
from torch_pointcloud.losses import GroupFree3DLoss
from torch_pointcloud.metrics import box_average_precision, box_matches
from torch_pointcloud.metrics.detection import BoxMatches
from torch_pointcloud.models import GroupFree3DDetection, create_model, model_info
from torch_pointcloud.ops.box3d import count_points_in_boxes, nms3d
from torch_pointcloud.optim import generate_param_groups
from torch_pointcloud.utils.data import DataKeys, PointCloudDataLoader
from torch_pointcloud.utils.random import seed_everything
from torch_pointcloud.utils.types import Boxes3D, Detection3D

CUDA_AVAILABLE = torch.cuda.is_available()
CPU_COUNT = os.cpu_count()
DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"
NUM_WORKERS = CPU_COUNT // 2 if CPU_COUNT is not None else 0
SEED = 42
NUM_POINTS = 20000
NMS_IOU = 0.25
MIN_POINTS = 5
IOU_THRESHOLDS = [0.25, 0.5]
DECODER_KEYWORDS = ["decoder", "query_proj", "key_proj", "pos_embeds"]


# The Group-Free targets (sampled seeds, heading bins) are assigned inside `GroupFree3DLoss` from the packed boxes,
# so the pipeline only augments the points and boxes together.
TRAIN_TRANSFORM = T.Compose(
    [
        T.RandomSample(keys=DataKeys.POS, num_samples=NUM_POINTS),
        T.RandomFlip(keys=DataKeys.POS, box_key=DataKeys.BOX, axes=(0,)),
        T.RandomRotate(keys=DataKeys.POS, box_key=DataKeys.BOX, angle_range=(-30.0, 30.0)),
        T.RandomScale(keys=DataKeys.POS, box_key=DataKeys.BOX, scale_range=(0.85, 1.15)),
        T.KeepItems(keys=[DataKeys.POS, DataKeys.BOX, DataKeys.LABEL]),
    ]
)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    train_loader, val_loader = configure_dataloaders(args)
    model = configure_model(args)
    optimizer, scheduler = configure_optimizers(args, model)
    criterion = configure_criterion(args, model)

    num_train, num_val = len(train_loader.dataset), len(val_loader.dataset)  # type: ignore[arg-type]
    print(f"Training {args.model!r} on {num_train} scenes, validating on {num_val}.")
    output: Optional[Path] = Path(args.output) if args.output else None
    best_loss = float("inf")
    for epoch in range(args.epochs):
        loss = train_one_epoch(model, criterion, optimizer, train_loader, args.clip_norm, args.device)
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
    parser = ArgumentParser(description="Train Group-Free 3D on SUN RGB-D.")
    parser.add_argument("--model", default="groupfree3d-l6-o256.sunrgbd.ze-liu", help="Registered detection model name")
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--root", default=DATA_DIR, help="Dataset root directory.")
    parser.add_argument("--output", default=None, help="Directory for the checkpoints (default: none saved).")
    parser.add_argument("--seed", default=SEED, type=int)
    parser.add_argument("--epochs", default=600, type=int)
    parser.add_argument("--batch-size", default=32, type=int, help="Scenes per step (the authors use 4 GPUs x 8).")
    parser.add_argument("--lr", default=4e-3, type=float)
    parser.add_argument("--decoder-lr", default=2e-4, type=float, help="Learning rate of the transformer decoder.")
    parser.add_argument("--weight-decay", default=1e-8, type=float)
    parser.add_argument("--clip-norm", default=0.1, type=float, help="Gradient-norm clipping threshold.")
    parser.add_argument("--num-workers", default=NUM_WORKERS, type=int)
    parser.add_argument("--eval-every", default=50, type=int, help="Validate every this many epochs.")
    parser.add_argument("--limit-train-batches", default=None, type=int)
    parser.add_argument("--limit-val-batches", default=None, type=int)
    parser.add_argument("--download", action="store_true", help="Download SUN RGB-D if missing.")
    parser.add_argument("--force-process", action="store_true", help="Force re-processing the dataset.")
    return parser.parse_args()


def configure_val_transform(args: Namespace) -> T.Transform:
    """The registered inference transform of the model (20 000 random points)."""
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


def configure_model(args: Namespace) -> GroupFree3DDetection:
    model = cast(GroupFree3DDetection, create_model(args.model, task="detection", pretrained=False))
    return model.to(args.device)


def configure_optimizers(args: Namespace, model: GroupFree3DDetection) -> tuple[Optimizer, LRScheduler]:
    """AdamW with a lower learning rate on the transformer decoder, decayed x0.1 after epochs 420, 480 and 540."""
    overrides = [dict(keyword=keyword, lr=args.decoder_lr) for keyword in DECODER_KEYWORDS]
    groups = generate_param_groups(model, weight_decay=args.weight_decay, overrides=overrides)
    optimizer = torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[420, 480, 540], gamma=0.1)
    return optimizer, scheduler


def configure_criterion(args: Namespace, model: GroupFree3DDetection) -> GroupFree3DLoss:
    """The Group-Free 3D loss with the authors' SUN RGB-D weights and smooth-L1 transition points."""
    criterion = GroupFree3DLoss(
        num_classes=model.num_classes,
        num_heading_bins=model.num_heading_bins,
        num_size_clusters=model.num_size_clusters,
        mean_sizes=model.mean_sizes,
        size_class_agnostic=model.size_class_agnostic,
        sampling_weight=0.2,
        objectness_weight=0.4,
        center_delta=0.111111111111,
        size_delta=0.0625,
        heading_delta=0.04,
    )
    return criterion.to(args.device)


def train_one_epoch(
    model: GroupFree3DDetection,
    criterion: GroupFree3DLoss,
    optimizer: Optimizer,
    loader: DataLoader,
    clip_norm: float,
    device: str,
) -> float:
    model.train()
    total_loss = 0.0
    for data in tqdm(loader, desc="Training", leave=False):
        data = {key: value.to(device) if isinstance(value, Tensor) else value for key, value in data.items()}
        loss = criterion(model(None, data[DataKeys.POS], data[DataKeys.BATCH]), data)["loss"]

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model: GroupFree3DDetection, loader: DataLoader, device: str) -> dict[str, float]:
    model.eval()
    num_classes = model.num_classes
    matches: list[BoxMatches] = []
    for data in tqdm(loader, desc="Evaluating", leave=False):
        data = {key: value.to(device) if isinstance(value, Tensor) else value for key, value in data.items()}
        pos, batch = data[DataKeys.POS], data[DataKeys.BATCH]
        det = model.decode(model(None, pos, batch), stages=range(model.num_decoder_layers + 1))
        boxes, scores, labels, det_batch = det["boxes"], det["scores"], det["labels"], det["batch"]
        counts = count_points_in_boxes(pos, boxes, pos_batch=batch, box_batch=det_batch)
        cand = (counts >= MIN_POINTS).nonzero(as_tuple=False).squeeze(-1)
        keep = cand[nms3d(boxes[cand], scores[cand], NMS_IOU, labels=labels[cand], batch=det_batch[cand])]

        # Indoor AP convention: every surviving box is scored against each class by its class probability.
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
