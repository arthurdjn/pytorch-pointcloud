"""Train LION on nuScenes following its `lion_models/lion_mamba_nusc_8x_1f_1x_one_stride_128dim.yaml` configuration."""

import math
import os
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Optional, Sequence, cast

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

import torch_pointcloud.transforms as T
import torch_pointcloud.transforms.functional as F
from torch_pointcloud.config import DATA_DIR
from torch_pointcloud.datasets import NuScenes, NuScenesMini
from torch_pointcloud.datasets.nuscenes import NUSCENES_DETECTION_CLASSES, NuScenesSplit, velocity_attributes
from torch_pointcloud.losses import TransFusionHeadLoss
from torch_pointcloud.metrics import nuscenes_detection_metrics
from torch_pointcloud.models import create_model, model_info
from torch_pointcloud.models.lion import LIONDetection, TransFusionHeadOutput
from torch_pointcloud.ops.box3d import nms3d
from torch_pointcloud.utils.data import DataKeys, PointCloudDataLoader
from torch_pointcloud.utils.random import seed_everything
from torch_pointcloud.utils.types import Boxes3D, Detection3D

CUDA_AVAILABLE = torch.cuda.is_available()
CPU_COUNT = os.cpu_count()
DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"
NUM_WORKERS = CPU_COUNT // 2 if CPU_COUNT is not None else 0
SEED = 42
MODEL = "lion-mamba.nuscenes.zhe-liu"
POINT_CLOUD_RANGE = (-54.0, -54.0, -5.0, 54.0, 54.0, 3.0)
# Objects pasted per scene and class, in the order of NUSCENES_DETECTION_CLASSES, counting the ones already there.
SAMPLES_PER_CLASS = dict(enumerate((2, 3, 7, 4, 6, 2, 6, 6, 2, 2)))
MIN_OBJECT_POINTS = 5
# The pasting stops for the last epochs (`DisableAugmentationHook`).
EPOCHS_WITHOUT_SAMPLING = 4
ROTATION_RANGE = (-45.0, 45.0)
SCALE_RANGE = (0.9, 1.1)
# Uniform offsets with the standard deviation of the reference's Gaussian ones (0.5 per axis).
TRANSLATION_RANGE = (-0.5 * math.sqrt(3), 0.5 * math.sqrt(3))
GRAD_CLIP_NORM = 35.0


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    train_loader, val_loader = configure_dataloaders(args)
    model = configure_model(args)
    optimizer, scheduler = configure_optimizers(args, model, len(train_loader))
    criterion = configure_criterion(args)

    num_train, num_val = len(train_loader.dataset), len(val_loader.dataset)  # type: ignore[arg-type]
    print(f"Training {args.model!r} on {num_train} keyframes, validating on {num_val}.")
    output: Optional[Path] = Path(args.output) if args.output else None
    best_loss = float("inf")
    for epoch in range(args.epochs):
        if epoch == args.epochs - EPOCHS_WITHOUT_SAMPLING:
            disable_sampling(train_loader.dataset)

        loss = train_one_epoch(model, criterion, optimizer, scheduler, train_loader, args.device)
        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch + 1}/{args.epochs}  lr={lr:.2e}  train/loss {loss:.4f}")

        if (epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs:
            val = evaluate(model, val_loader, args.device)
            print(f"  val/mAP {val['mAP']:.4f} | val/NDS {val['NDS']:.4f}")

        if output is not None:
            output.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), output / "last.pt")
            if loss < best_loss:
                best_loss = loss
                torch.save(model.state_dict(), output / "best.pt")


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Train LION on nuScenes.")
    parser.add_argument("--model", default=MODEL, choices=[MODEL])
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--root", default=DATA_DIR, help="Dataset root directory.")
    parser.add_argument(
        "--split",
        default="trainval",
        choices=("trainval", "mini"),
        help="nuScenes `train` / `val`, or the mini release for both.",
    )
    parser.add_argument("--max-sweeps", default=10, type=int, help="LiDAR sweeps per keyframe.")
    parser.add_argument("--output", default=None, help="Directory for the checkpoints (default: none saved).")
    parser.add_argument("--seed", default=SEED, type=int)
    parser.add_argument("--epochs", default=36, type=int)
    # The reference trains on 8 devices with 2 keyframes each; `--lr` is its learning rate for that total batch.
    parser.add_argument("--batch-size", default=2, type=int)
    parser.add_argument("--lr", default=3e-3, type=float)
    parser.add_argument("--weight-decay", default=0.05, type=float)
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
    train_dataset = nuscenes_dataset(args, split="train", transform=train_transform)
    val_transform = configure_val_transform(args)
    val_dataset = nuscenes_dataset(args, split="val", transform=val_transform)

    if args.limit_train_batches is not None:
        train_dataset = Subset(
            train_dataset, range(min(args.limit_train_batches * args.batch_size, len(train_dataset)))
        )
    if args.limit_val_batches is not None:
        val_dataset = Subset(val_dataset, range(min(args.limit_val_batches * args.batch_size, len(val_dataset))))

    return train_dataset, val_dataset


def configure_objects(args: Namespace) -> list[dict[str, Tensor]]:
    """The training objects with their velocities, cut out once and cached next to the processed split."""
    dataset = nuscenes_dataset(args, split="train", transform=None, force_process=args.force_process)
    path = Path(dataset.processed_dir).parent / f"objects_{Path(dataset.processed_dir).name}.pt"
    if path.exists() and not args.force_process:
        objects = torch.load(path, weights_only=True)
    else:
        objects = []
        for data in tqdm(dataset, desc="Cutting out the objects"):
            objects.extend(
                F.cut_boxes(
                    data,
                    keys=[DataKeys.POS, DataKeys.INTENSITY, DataKeys.TIMESTAMP],
                    attribute_keys=DataKeys.VELOCITY,
                )
            )
        torch.save(objects, path)

    return [obj for obj in objects if len(obj[DataKeys.POS]) >= MIN_OBJECT_POINTS]


def configure_dataloaders(args: Namespace) -> tuple[DataLoader, DataLoader]:
    train_dataset, val_dataset = configure_datasets(args)

    train_loader = PointCloudDataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        cat_keys=[DataKeys.POS, DataKeys.X, DataKeys.BOX],
    )
    val_loader = PointCloudDataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        cat_keys=[DataKeys.POS, DataKeys.X, DataKeys.BOX],
    )

    return train_loader, val_loader


def configure_model(args: Namespace) -> LIONDetection:
    model = create_model(args.model, task="detection", pretrained=False)
    assert isinstance(model, LIONDetection)
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


def configure_criterion(args: Namespace) -> TransFusionHeadLoss:
    """The TransFusion head loss on the grid of the registered model, velocity codes included."""
    hparams = model_info(args.model, task="detection")["hparams"]
    return TransFusionHeadLoss(
        hparams["num_classes"],
        point_cloud_range=hparams["point_cloud_range"],
        voxel_size=hparams["voxel_size"],
        feature_map_stride=hparams["feature_map_stride"],
        num_proposals=200,
        gaussian_overlap=0.1,
        min_radius=2,
        matcher_cls_cost=0.15,
        matcher_reg_cost=0.25,
        matcher_iou_cost=0.25,
        code_weights=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2),
        cls_weight=1.0,
        loc_weight=0.25,
        heatmap_weight=1.0,
        iou_weight=0.5,
    ).to(args.device)


def configure_train_transform(args: Namespace, objects: list[dict[str, Tensor]]) -> T.Compose:
    point_keys = [DataKeys.POS, DataKeys.INTENSITY, DataKeys.TIMESTAMP]
    return T.Compose(
        [
            # The boxes without a LiDAR point stay as targets, where the reference drops them.
            T.KeepItems(keys=[*point_keys, DataKeys.BOX, DataKeys.LABEL, DataKeys.VELOCITY]),
            T.PasteBoxes(objects, num_samples=SAMPLES_PER_CLASS, keys=point_keys, attribute_keys=DataKeys.VELOCITY),
            # The velocity rides in the last two box columns, where the augmentations and the loss read it.
            T.Cat(keys=[DataKeys.BOX, DataKeys.VELOCITY], dst_key=DataKeys.BOX, dim=1),
            T.RandomFlip(keys=DataKeys.POS, box_key=DataKeys.BOX, axes=(0, 1)),
            T.RandomRotate(keys=DataKeys.POS, box_key=DataKeys.BOX, angle_range=ROTATION_RANGE),
            T.RandomScale(keys=DataKeys.POS, box_key=DataKeys.BOX, scale_range=SCALE_RANGE),
            T.RandomTranslate(keys=DataKeys.POS, box_key=DataKeys.BOX, translation_range=TRANSLATION_RANGE),
            T.BoxMask(keys=[DataKeys.POS, DataKeys.BOX], bbox=POINT_CLOUD_RANGE, dst_keys=["point_mask", "box_mask"]),
            T.ApplyMask(keys=point_keys, mask_key="point_mask"),
            T.ApplyMask(keys=[DataKeys.BOX, DataKeys.LABEL], mask_key="box_mask"),
            T.ShufflePoint(keys=point_keys),
            T.Cat(keys=[DataKeys.INTENSITY, DataKeys.TIMESTAMP], dst_key=DataKeys.X, dim=1),
        ]
    )


def configure_val_transform(args: Namespace) -> T.Transform:
    """The registered inference transform of the model."""
    transform = model_info(args.model, task="detection")["transform"]
    assert transform is not None
    return transform


def nuscenes_dataset(
    args: Namespace,
    split: NuScenesSplit,
    transform: Optional[T.Transform],
    force_process: bool = False,
) -> NuScenes:
    """The `train` or `val` keyframes of the full release, or every keyframe of the mini release."""
    if args.split == "mini":
        return NuScenesMini(
            root=args.root, max_sweeps=args.max_sweeps, transform=transform, force_process=force_process
        )
    return NuScenes(
        root=args.root, split=split, max_sweeps=args.max_sweeps, transform=transform, force_process=force_process
    )


def disable_sampling(dataset: Dataset) -> None:
    """Drop the object pasting from the training transform."""
    if isinstance(dataset, Subset):
        dataset = dataset.dataset
    dataset = cast(NuScenes, dataset)
    transform = cast(T.Compose, dataset.transform)

    steps = [step for step in transform.transforms if not isinstance(step, T.PasteBoxes)]
    dataset.transform = T.Compose(steps)


def circular_nms(det: Detection3D, local_max_classes: Sequence[int], radius: float) -> Detection3D:
    """LION's per-task BEV filter: circular NMS on the crowded `local_max_classes`, keep all other classes."""
    boxes, scores, labels, batch = det["boxes"], det["scores"], det["labels"], det["batch"]
    keep_parts = []
    for b in torch.unique(batch):
        scene = (batch == b).nonzero(as_tuple=False).squeeze(-1)
        keep_mask = torch.ones(scene.numel(), dtype=torch.bool, device=scene.device)
        for cls in local_max_classes:
            local = (labels[scene] == cls).nonzero(as_tuple=False).squeeze(-1)
            if local.numel() == 0:
                continue

            cls_keep = torch.zeros(local.numel(), dtype=torch.bool, device=scene.device)
            cls_keep[nms3d(boxes[scene][local][:, :7], scores[scene][local], radius)] = True
            keep_mask[local] = cls_keep

        keep_parts.append(scene[keep_mask])

    idx = torch.cat(keep_parts)

    return {
        "boxes": boxes[idx],
        "scores": scores[idx],
        "labels": labels[idx],
        "batch": batch[idx],
        "velocity": det["velocity"][idx],
    }


def forward(model: LIONDetection, data: dict[str, Tensor]) -> TransFusionHeadOutput:
    return model(data[DataKeys.X], data[DataKeys.POS], data[f"batch_{DataKeys.POS}"])


def train_one_epoch(
    model: LIONDetection,
    criterion: TransFusionHeadLoss,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    loader: DataLoader,
    device: str,
) -> float:
    model.train()
    total_loss = 0.0
    for data in tqdm(loader, desc="Training", leave=False):
        data = {key: value.to(device) if isinstance(value, Tensor) else value for key, value in data.items()}
        loss = criterion(forward(model, data), data)["loss"]

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model: LIONDetection, loader: DataLoader, device: str) -> dict[str, float]:
    model.eval()
    preds: list[Detection3D] = []
    targets: list[Boxes3D] = []
    gt_velocities: list[Tensor] = []
    gt_num_points: list[Tensor] = []
    gt_attributes: list[Tensor] = []
    offsets: list[int] = [0]

    for data in tqdm(loader, desc="Evaluating", leave=False):
        data = {key: value.to(device) if isinstance(value, Tensor) else value for key, value in data.items()}
        det = circular_nms(model.decode(forward(model, data)), model.head.local_max_classes, model.head.nms_radius)
        preds.append(
            {
                "boxes": det["boxes"].cpu(),
                "scores": det["scores"].cpu(),
                "labels": det["labels"].cpu(),
                "batch": det["batch"].cpu(),
                "velocity": det["velocity"].cpu(),
            }
        )
        targets.append(
            {
                "boxes": data[DataKeys.BOX].cpu(),
                "labels": data[DataKeys.LABEL].cpu(),
                "batch": data[DataKeys.BATCH_BOX].cpu(),
            }
        )
        gt_velocities.append(data[DataKeys.VELOCITY].cpu())
        gt_num_points.append(data[DataKeys.NUM_POINTS].cpu())
        gt_attributes.append(data[DataKeys.ATTRIBUTE].cpu())
        offsets.append(offsets[-1] + len(data[DataKeys.TOKEN]))

    pred_labels = torch.cat([p["labels"] for p in preds])
    pred_velocity = torch.cat([p["velocity"] for p in preds])

    return nuscenes_detection_metrics(
        {
            "boxes": torch.cat([torch.cat([p["boxes"], p["velocity"]], dim=1) for p in preds]),
            "scores": torch.cat([p["scores"] for p in preds]),
            "labels": pred_labels,
            "batch": torch.cat([p["batch"] + offset for p, offset in zip(preds, offsets)]),
        },
        {
            "boxes": torch.cat([torch.cat([t["boxes"], v], dim=1) for t, v in zip(targets, gt_velocities)]),
            "labels": torch.cat([t["labels"] for t in targets]),
            "batch": torch.cat([t["batch"] + offset for t, offset in zip(targets, offsets)]),
        },
        class_names=NUSCENES_DETECTION_CLASSES,
        gt_num_points=torch.cat(gt_num_points),
        pred_attributes=velocity_attributes(pred_labels, pred_velocity),
        gt_attributes=torch.cat(gt_attributes),
    )


if __name__ == "__main__":
    main()
