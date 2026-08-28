"""Benchmark the SECOND detectors on KITTI and nuScenes with the reference evaluation protocols.

NOTE: nuScenes defaults to the `v1.0-mini` split, whose numbers are smoke checks; only the full `val` split is comparable.

Results:

    | Variant                             | metric                      | reference                     | torch-pointcloud |
    | ----------------------------------- | --------------------------- | ----------------------------- | ---------------- |
    | second.kitti.openpcdet              | Car / Ped / Cyc / mAP (R11) | 78.62 / 52.98 / 67.15 / 66.25 |                  |
    | second-multihead.nuscenes.openpcdet | mAP / NDS                   | 50.59 / 62.29                 |                  |

Usage:
    uv run --no-sync python examples/second_benchmark_detection.py --model second.kitti.openpcdet --split-file /path/to/ImageSets/val.txt
    uv run --no-sync python examples/second_benchmark_detection.py --model second-multihead.nuscenes.openpcdet
"""

import argparse
import os
from typing import Dict, List

import torch
from torch import Tensor
from torch.utils.data import Dataset, Subset
from tqdm import tqdm

import torch_pointcloud.transforms as T
from torch_pointcloud.config import DATA_DIR
from torch_pointcloud.datasets import KITTI, NuScenesMini
from torch_pointcloud.datasets.kitti import KITTI_CLASSES
from torch_pointcloud.datasets.nuscenes import NUSCENES_DETECTION_CLASSES, velocity_attributes
from torch_pointcloud.models import DetectionModel, create_model
from torch_pointcloud.utils.box3d import nms3d, projected_ignore_mask
from torch_pointcloud.utils.data import DataKeys, PointCloudDataLoader
from torch_pointcloud.utils.metrics import average_precision3d, nuscenes_detection_metrics
from torch_pointcloud.utils.random import seed_everything, set_determinism
from torch_pointcloud.utils.types import Boxes3D, Detection3D

CUDA_AVAILABLE = torch.cuda.is_available()
CPU_COUNT = os.cpu_count()
DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"
NUM_WORKERS = CPU_COUNT // 2 if CPU_COUNT is not None else 0
SEED = 42

KITTI_SCORE_THRESHOLD = 0.1
KITTI_NMS_IOU = 0.01
KITTI_DETECTION_CLASSES = ("Car", "Pedestrian", "Cyclist")
KITTI_IOU = {0: 0.7, 1: 0.5, 2: 0.5}
# Van / Person_sitting and harder-than-moderate boxes (occlusion > 1, truncation > 0.3, 2D height < 25 px) are ignored.
KITTI_TRANSFORM = T.RelabelBoxes(
    keys=(DataKeys.BOX, DataKeys.LABEL, DataKeys.TRUNCATION, DataKeys.OCCLUSION, DataKeys.BBOX_HEIGHT),
    mapping={KITTI_CLASSES.index(name): index for index, name in enumerate(KITTI_DETECTION_CLASSES)},
    ignore_mapping={KITTI_CLASSES.index("Van"): 0, KITTI_CLASSES.index("Person_sitting"): 1},
    ignore_fields={DataKeys.OCCLUSION: (None, 1), DataKeys.TRUNCATION: (None, 0.3), DataKeys.BBOX_HEIGHT: (25, None)},
)

NUSCENES_SCORE_THRESHOLD = 0.1
NUSCENES_NMS_IOU = 0.2


def forward(model: DetectionModel, data: Dict[str, Tensor]) -> Detection3D:
    output = model(
        data[DataKeys.VOXEL],
        data[DataKeys.POS_VOXEL],
        data[DataKeys.VOXEL_NUM_POINTS],
        data[f"batch_{DataKeys.POS_VOXEL}"],
    )
    return model.decode(output)


@torch.no_grad()
def evaluate_kitti(model: DetectionModel, dataloader: PointCloudDataLoader, device: str) -> Dict[str, float]:
    model.to(device).eval()
    preds: List[Detection3D] = []
    targets: List[Boxes3D] = []

    for data in tqdm(dataloader, total=len(dataloader), desc="Testing"):
        data = {key: value.to(device) if torch.is_tensor(value) else value for key, value in data.items()}
        det = forward(model, data)
        keep = det["scores"] > KITTI_SCORE_THRESHOLD
        boxes, scores, labels, batch = det["boxes"][keep], det["scores"][keep], det["labels"][keep], det["batch"][keep]
        idx = nms3d(boxes, scores, KITTI_NMS_IOU, batch=batch, rotated=True)
        boxes, scores, labels, batch = boxes[idx], scores[idx], labels[idx], batch[idx]
        ignore_mask = projected_ignore_mask(boxes, data[DataKeys.CALIB][batch], data[DataKeys.IMAGE_SHAPE][batch])
        preds.append(
            {
                "boxes": boxes.cpu(),
                "scores": scores.cpu(),
                "labels": labels.cpu(),
                "batch": batch.cpu(),
                "ignore_mask": ignore_mask.cpu(),
            }
        )
        targets.append(
            {
                "boxes": data[DataKeys.BOX].cpu(),
                "labels": data[DataKeys.LABEL].cpu(),
                "batch": data[DataKeys.BATCH_BOX].cpu(),
                "ignore_mask": data["ignore_mask"].cpu(),
            }
        )

    return average_precision3d(
        preds,
        targets,
        iou_per_class=KITTI_IOU,
        class_names=KITTI_DETECTION_CLASSES,
        interpolation="r11",
    )


@torch.no_grad()
def evaluate_nuscenes(model: DetectionModel, dataloader: PointCloudDataLoader, device: str) -> Dict[str, float]:
    model.to(device).eval()
    preds: List[Detection3D] = []
    targets: List[Boxes3D] = []
    gt_velocities: List[Tensor] = []
    gt_num_points: List[Tensor] = []
    gt_attributes: List[Tensor] = []
    offsets: List[int] = [0]

    for data in tqdm(dataloader, total=len(dataloader), desc="Testing"):
        data = {key: value.to(device) if torch.is_tensor(value) else value for key, value in data.items()}
        det = forward(model, data)
        keep = det["scores"] > NUSCENES_SCORE_THRESHOLD
        boxes, scores, labels, batch = det["boxes"][keep], det["scores"][keep], det["labels"][keep], det["batch"][keep]
        idx = nms3d(boxes, scores, NUSCENES_NMS_IOU, labels=labels, batch=batch)
        preds.append(
            {
                "boxes": boxes[idx].cpu(),
                "scores": scores[idx].cpu(),
                "labels": labels[idx].cpu(),
                "batch": batch[idx].cpu(),
                "velocity": det["velocity"][keep][idx].cpu(),
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
        torch.cat([torch.cat([p["boxes"], p["velocity"]], dim=1) for p in preds]),
        torch.cat([p["scores"] for p in preds]),
        pred_labels,
        torch.cat([p["batch"] + offset for p, offset in zip(preds, offsets)]),
        torch.cat([torch.cat([t["boxes"], v], dim=1) for t, v in zip(targets, gt_velocities)]),
        torch.cat([t["labels"] for t in targets]),
        torch.cat([t["batch"] + offset for t, offset in zip(targets, offsets)]),
        class_names=NUSCENES_DETECTION_CLASSES,
        gt_num_points=torch.cat(gt_num_points),
        pred_attributes=velocity_attributes(pred_labels, pred_velocity),
        gt_attributes=torch.cat(gt_attributes),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark SECOND 3D detection on KITTI or nuScenes.")
    parser.add_argument(
        "--model",
        default="second.kitti.openpcdet",
        choices=["second.kitti.openpcdet", "second-multihead.nuscenes.openpcdet"],
    )
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--root", default=DATA_DIR, help="Dataset root directory.")
    parser.add_argument("--split-file", default=None, help="KITTI frame-id list (e.g. ImageSets/val.txt).")
    parser.add_argument("--version", default="v1.0-mini", help="nuScenes version.")
    parser.add_argument("--max-sweeps", default=10, type=int, help="nuScenes LiDAR sweeps per keyframe.")
    parser.add_argument("--seed", default=SEED, type=int)
    parser.add_argument("--batch-size", default=4, type=int)
    parser.add_argument("--num-workers", default=NUM_WORKERS, type=int)
    parser.add_argument("--limit", default=None, type=int, help="Evaluate at most this many frames.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    set_determinism(tf32=False)

    model, model_info = create_model(args.model, task="detection", pretrained=True, return_info=True)
    assert isinstance(model, DetectionModel)

    dataset: Dataset
    if "kitti" in args.model:
        print(f"Benchmarking model {args.model!r} on KITTI!")
        dataset = KITTI(
            root=args.root,
            train=True,
            split_file=args.split_file,
            fov=True,
            return_calib=True,
            transform=T.Compose([KITTI_TRANSFORM, model_info["transform"]]),
        )
        evaluate, stack_keys = evaluate_kitti, [DataKeys.CALIB, DataKeys.IMAGE_SHAPE]
    else:
        print(f"Benchmarking model {args.model!r} on nuScenes ({args.version})!")
        dataset = NuScenesMini(
            root=args.root,
            version=args.version,
            max_sweeps=args.max_sweeps,
            transform=model_info["transform"],
        )
        evaluate, stack_keys = evaluate_nuscenes, []
    if args.limit is not None:
        n = min(int(args.limit), len(dataset))
        dataset = Subset(dataset, range(n))
        print(f"Evaluating on a subset of the first {n} frames.")

    dataloader = PointCloudDataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        stack_keys=stack_keys,
        cat_keys=[DataKeys.BOX, DataKeys.POS_VOXEL],
    )

    print(f"Test set: {len(dataset)} frames")
    metrics = evaluate(model, dataloader, args.device)
    print("\nResults:")
    for key, value in metrics.items():
        print(f"  {key:<24} {value:.4f}")


if __name__ == "__main__":
    main()
