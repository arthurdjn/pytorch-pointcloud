"""Train PointRCNN on KITTI following OpenPCDet's `kitti_models/pointrcnn.yaml` configuration."""

import os
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

import torch_pointcloud.transforms as T
import torch_pointcloud.transforms.functional as F
from torch_pointcloud.config import DATA_DIR
from torch_pointcloud.datasets import KITTI
from torch_pointcloud.datasets.kitti import KITTI_CLASSES
from torch_pointcloud.losses import PointRCNNLoss
from torch_pointcloud.metrics import box_average_precision, box_matches
from torch_pointcloud.metrics.detection import BoxMatches
from torch_pointcloud.models import DetectionModel, create_model, model_info
from torch_pointcloud.ops.box3d import nms3d, projected_ignore_mask
from torch_pointcloud.utils.data import DataKeys, PointCloudDataLoader
from torch_pointcloud.utils.random import seed_everything
from torch_pointcloud.utils.types import Boxes3D, Detection3D

CUDA_AVAILABLE = torch.cuda.is_available()
CPU_COUNT = os.cpu_count()
DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"
NUM_WORKERS = CPU_COUNT // 2 if CPU_COUNT is not None else 0
SEED = 42
MODEL = "pointrcnn.kitti.openpcdet"
KITTI_DETECTION_CLASSES = ("Car", "Pedestrian", "Cyclist")
POINT_CLOUD_RANGE = (0.0, -40.0, -3.0, 70.4, 40.0, 1.0)
NUM_POINTS = 16384
# Objects pasted per scene and class (Car, Pedestrian, Cyclist), counting the ones already there.
SAMPLES_PER_CLASS = {0: 20, 1: 15, 2: 15}
MIN_OBJECT_POINTS = 5
GRAD_CLIP_NORM = 10.0
SCORE_THRESHOLD = 0.1
NMS_IOU = 0.1
KITTI_IOU = {0: 0.7, 1: 0.5, 2: 0.5}
# Training keeps the boxes of the three classes whatever their difficulty.
KITTI_TRAIN_TRANSFORM = T.RelabelBoxes(
    keys=(DataKeys.BOX, DataKeys.LABEL, DataKeys.TRUNCATION, DataKeys.OCCLUSION, DataKeys.BBOX_HEIGHT),
    mapping={KITTI_CLASSES.index(name): index for index, name in enumerate(KITTI_DETECTION_CLASSES)},
)
# Validation scores the moderate split: Van / Person_sitting and harder boxes (occlusion > 1, truncation > 0.3,
# 2D height < 25 px) are ignored, and predictions projecting to under 25 px are dropped.
KITTI_EVAL_TRANSFORM = T.RelabelBoxes(
    keys=(DataKeys.BOX, DataKeys.LABEL, DataKeys.TRUNCATION, DataKeys.OCCLUSION, DataKeys.BBOX_HEIGHT),
    mapping={KITTI_CLASSES.index(name): index for index, name in enumerate(KITTI_DETECTION_CLASSES)},
    ignore_mapping={KITTI_CLASSES.index("Van"): 0, KITTI_CLASSES.index("Person_sitting"): 1},
    ignore_fields={DataKeys.OCCLUSION: (None, 1), DataKeys.TRUNCATION: (None, 0.3), DataKeys.BBOX_HEIGHT: (25, None)},
)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    train_loader, val_loader = configure_dataloaders(args)
    model = configure_model(args)
    optimizer, scheduler = configure_optimizers(args, model, len(train_loader))
    criterion = configure_criterion(args)

    num_train, num_val = len(train_loader.dataset), len(val_loader.dataset)  # type: ignore[arg-type]
    print(f"Training {args.model!r} on {num_train} frames, validating on {num_val}.")
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
    parser = ArgumentParser(description="Train PointRCNN on KITTI.")
    parser.add_argument("--model", default=MODEL, choices=[MODEL])
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--root", default=DATA_DIR, help="Dataset root directory.")
    parser.add_argument(
        "--train-split-file", default=None, help="Frame ids of the training split (ImageSets/train.txt)."
    )
    parser.add_argument("--val-split-file", default=None, help="Frame ids of the validation split (ImageSets/val.txt).")
    parser.add_argument("--output", default=None, help="Directory for the checkpoints (default: none saved).")
    parser.add_argument("--seed", default=SEED, type=int)
    parser.add_argument("--epochs", default=80, type=int)
    # The reference trains on 8 devices with 2 frames each; `--lr` is its learning rate for that total batch.
    parser.add_argument("--batch-size", default=2, type=int)
    parser.add_argument("--lr", default=1e-2, type=float)
    parser.add_argument("--weight-decay", default=0.01, type=float)
    parser.add_argument("--num-workers", default=NUM_WORKERS, type=int)
    parser.add_argument("--eval-every", default=10, type=int, help="Validate every this many epochs.")
    parser.add_argument("--limit-train-batches", default=None, type=int)
    parser.add_argument("--limit-val-batches", default=None, type=int)
    parser.add_argument("--force-process", action="store_true", help="Force re-processing the dataset.")
    return parser.parse_args()


def configure_datasets(args: Namespace) -> tuple[Dataset, Dataset]:
    train_dataset: Dataset
    val_dataset: Dataset
    objects = configure_objects(args)
    train_transform = configure_train_transform(args, objects)
    train_dataset = KITTI(
        root=args.root,
        train=True,
        split_file=args.train_split_file,
        fov=True,
        transform=train_transform,
        force_process=args.force_process,
    )
    val_transform = configure_val_transform(args)
    val_dataset = KITTI(
        root=args.root,
        train=True,
        split_file=args.val_split_file,
        fov=True,
        return_calib=True,
        transform=val_transform,
    )

    if args.limit_train_batches is not None:
        train_dataset = Subset(
            train_dataset, range(min(args.limit_train_batches * args.batch_size, len(train_dataset)))
        )
    if args.limit_val_batches is not None:
        val_dataset = Subset(val_dataset, range(min(args.limit_val_batches * args.batch_size, len(val_dataset))))

    return train_dataset, val_dataset


def configure_objects(args: Namespace) -> list[dict[str, Tensor]]:
    """The training objects, cut out once and cached next to the processed split, filtered as the reference does."""
    dataset = KITTI(
        root=args.root,
        train=True,
        split_file=args.train_split_file,
        fov=True,
        transform=KITTI_TRAIN_TRANSFORM,
        force_process=args.force_process,
    )
    split = Path(args.train_split_file).stem if args.train_split_file else "training"
    path = Path(dataset.processed_split_dir).parent / f"objects_{split}.pt"
    if path.exists() and not args.force_process:
        objects = torch.load(path, weights_only=True)
    else:
        objects = []
        for data in tqdm(dataset, desc="Cutting out the objects"):
            objects.extend(
                F.cut_boxes(
                    data,
                    keys=[DataKeys.POS, DataKeys.INTENSITY],
                    attribute_keys=[DataKeys.TRUNCATION, DataKeys.OCCLUSION, DataKeys.BBOX_HEIGHT],
                )
            )
        torch.save(objects, path)

    kept = []
    for obj in objects:
        difficulty = kitti_difficulty(obj[DataKeys.BBOX_HEIGHT], obj[DataKeys.TRUNCATION], obj[DataKeys.OCCLUSION])
        if len(obj[DataKeys.POS]) >= MIN_OBJECT_POINTS and difficulty >= 0:
            kept.append(obj)
    return kept


def configure_dataloaders(args: Namespace) -> tuple[DataLoader, DataLoader]:
    train_dataset, val_dataset = configure_datasets(args)

    train_loader = PointCloudDataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        cat_keys=[DataKeys.BOX],
    )
    val_loader = PointCloudDataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        stack_keys=[DataKeys.CALIB, DataKeys.IMAGE_SHAPE],
        cat_keys=[DataKeys.BOX],
    )

    return train_loader, val_loader


def configure_model(args: Namespace) -> DetectionModel:
    model = create_model(args.model, task="detection", pretrained=False)
    assert isinstance(model, DetectionModel)
    return model.to(args.device)


def configure_optimizers(args: Namespace, model: nn.Module, steps_per_epoch: int) -> tuple[Optimizer, LRScheduler]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)

    # One cycle per training, stepped every iteration: cosine rise over the first 40% from lr / 10, cosine decay
    # to lr / 1e5, while beta1 moves from 0.95 to 0.85 and back.
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=steps_per_epoch * args.epochs,
        pct_start=0.4,
        anneal_strategy="cos",
        div_factor=10.0,
        final_div_factor=1e4,
        base_momentum=0.85,
        max_momentum=0.95,
    )
    return optimizer, scheduler


def configure_criterion(args: Namespace) -> PointRCNNLoss:
    """The two-stage loss on the size clusters of the registered model."""
    hparams = model_info(args.model, task="detection")["hparams"]
    return PointRCNNLoss(hparams["num_classes"], mean_sizes=hparams["mean_sizes"]).to(args.device)


def configure_train_transform(args: Namespace, objects: list[dict[str, Tensor]]) -> T.Compose:
    return T.Compose(
        [
            KITTI_TRAIN_TRANSFORM,
            T.KeepItems(keys=[DataKeys.POS, DataKeys.INTENSITY, DataKeys.BOX, DataKeys.LABEL]),
            # Pasted objects keep the height they were cut at: the road-plane fit of the reference is not available.
            T.PasteBoxes(objects, num_samples=SAMPLES_PER_CLASS, keys=[DataKeys.POS, DataKeys.INTENSITY]),
            T.RandomFlip(keys=DataKeys.POS, box_key=DataKeys.BOX, axes=(1,)),
            T.RandomRotate(keys=DataKeys.POS, box_key=DataKeys.BOX, angle_range=(-45.0, 45.0)),
            T.RandomScale(keys=DataKeys.POS, box_key=DataKeys.BOX, scale_range=(0.95, 1.05)),
            T.BoxMask(keys=[DataKeys.POS, DataKeys.BOX], bbox=POINT_CLOUD_RANGE, dst_keys=["point_mask", "box_mask"]),
            T.ApplyMask(keys=[DataKeys.POS, DataKeys.INTENSITY], mask_key="point_mask"),
            T.ApplyMask(keys=[DataKeys.BOX, DataKeys.LABEL], mask_key="box_mask"),
            # Uniform draw of the points, where the reference favors the far ones when the scene has more.
            T.RandomSample(keys=[DataKeys.POS, DataKeys.INTENSITY], num_samples=NUM_POINTS),
            T.ShufflePoint(keys=[DataKeys.POS, DataKeys.INTENSITY]),
            T.Cat(keys=[DataKeys.INTENSITY], dst_key=DataKeys.X, dim=1),
        ]
    )


def configure_val_transform(args: Namespace) -> T.Compose:
    """The evaluation relabeling followed by the registered inference transform of the model."""
    transform = model_info(args.model, task="detection")["transform"]
    assert transform is not None
    return T.Compose([KITTI_EVAL_TRANSFORM, transform])


def kitti_difficulty(height: Tensor, truncation: Tensor, occlusion: Tensor) -> Tensor:
    """KITTI difficulty of every box: 0 easy, 1 moderate, 2 hard, -1 beyond the hard limits."""
    easy = (height >= 40) & (truncation <= 0.15) & (occlusion <= 0)
    moderate = (height >= 25) & (truncation <= 0.3) & (occlusion <= 1)
    hard = (height >= 25) & (truncation <= 0.5) & (occlusion <= 2)
    level = torch.full_like(height, -1, dtype=torch.long)
    level[hard] = 2
    level[moderate] = 1
    level[easy] = 0
    return level


def train_one_epoch(
    model: DetectionModel,
    criterion: PointRCNNLoss,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    loader: DataLoader,
    device: str,
) -> float:
    model.train()
    total_loss = 0.0
    for data in tqdm(loader, desc="Training", leave=False):
        data = {key: value.to(device) if isinstance(value, Tensor) else value for key, value in data.items()}
        output = model(
            data[DataKeys.X],
            data[DataKeys.POS],
            data[DataKeys.BATCH],
            gt_boxes=data[DataKeys.BOX],
            gt_labels=data[DataKeys.LABEL],
            gt_batch=data[DataKeys.BATCH_BOX],
        )
        loss = criterion(output, data)["loss"]

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model: DetectionModel, loader: DataLoader, device: str) -> dict[str, float]:
    model.eval()
    matches: list[BoxMatches] = []
    for data in tqdm(loader, desc="Evaluating", leave=False):
        data = {key: value.to(device) if isinstance(value, Tensor) else value for key, value in data.items()}
        det = model.decode(model(data[DataKeys.X], data[DataKeys.POS], data[DataKeys.BATCH]))
        keep = det["scores"] >= SCORE_THRESHOLD
        boxes, scores, labels, batch = det["boxes"][keep], det["scores"][keep], det["labels"][keep], det["batch"][keep]
        idx = nms3d(boxes, scores, NMS_IOU, batch=batch, rotated=True)
        boxes, scores, labels, batch = boxes[idx], scores[idx], labels[idx], batch[idx]
        ignore_mask = projected_ignore_mask(boxes, data[DataKeys.CALIB][batch], data[DataKeys.IMAGE_SHAPE][batch])
        preds: Detection3D = {
            "boxes": boxes,
            "scores": scores,
            "labels": labels,
            "batch": batch,
            "ignore_mask": ignore_mask,
        }
        target: Boxes3D = {
            "boxes": data[DataKeys.BOX],
            "labels": data[DataKeys.LABEL],
            "batch": data[DataKeys.BATCH_BOX],
            "ignore_mask": data["ignore_mask"],
        }
        matches.append(box_matches(preds, target))

    per_class = box_average_precision(
        matches,
        iou_threshold=KITTI_IOU,
        average="none",
        class_names=KITTI_DETECTION_CLASSES,
        interpolation="r11",
    )
    metrics = {f"AP/{name}": ap for name, ap in per_class.items()}
    metrics["mAP"] = box_average_precision(matches, iou_threshold=KITTI_IOU, interpolation="r11")
    return metrics


if __name__ == "__main__":
    main()
