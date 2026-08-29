"""Benchmark VoxelNeXt on nuScenes with the official detection metrics.

Results (nuScenes val, mAP / NDS):

    | Variant                      | reference   | torch-pointcloud |
    | ---------------------------- | ----------- | ---------------- |
    | voxelnext.nuscenes.openpcdet | 60.5 / 66.6 |                  |

Usage:
    uv run --no-sync python examples/voxelnext_benchmark_detection.py --root /path/to/data
    uv run --no-sync python examples/voxelnext_benchmark_detection.py --root /path/to/data --split mini
"""

import argparse
import os
from typing import Dict, List

import torch
from torch import Tensor
from torch.utils.data import Dataset, Subset
from tqdm import tqdm

from torch_pointcloud.config import DATA_DIR
from torch_pointcloud.datasets import NuScenes, NuScenesMini
from torch_pointcloud.datasets.nuscenes import NUSCENES_DETECTION_CLASSES, velocity_attributes
from torch_pointcloud.models import DetectionModel, create_model
from torch_pointcloud.utils.box3d import nms3d
from torch_pointcloud.utils.data import DataKeys, PointCloudDataLoader
from torch_pointcloud.utils.metrics import nuscenes_detection_metrics
from torch_pointcloud.utils.random import seed_everything, set_determinism
from torch_pointcloud.utils.types import Boxes3D, Detection3D

CUDA_AVAILABLE = torch.cuda.is_available()
CPU_COUNT = os.cpu_count()
DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"
NUM_WORKERS = CPU_COUNT // 2 if CPU_COUNT is not None else 0
SEED = 42
SCORE_THRESHOLD = 0.1
NMS_IOU = 0.2


@torch.no_grad()
def evaluate(model: DetectionModel, dataloader: PointCloudDataLoader, device: str) -> Dict[str, float]:
    model.to(device).eval()
    preds: List[Detection3D] = []
    targets: List[Boxes3D] = []
    gt_velocities: List[Tensor] = []
    gt_num_points: List[Tensor] = []
    gt_attributes: List[Tensor] = []
    offsets: List[int] = [0]

    for data in tqdm(dataloader, total=len(dataloader), desc="Testing"):
        data = {key: value.to(device) if torch.is_tensor(value) else value for key, value in data.items()}
        output = model(
            data[DataKeys.VOXEL],
            data[DataKeys.POS_VOXEL],
            data[DataKeys.VOXEL_NUM_POINTS],
            data[f"batch_{DataKeys.POS_VOXEL}"],
        )
        det = model.decode(output)
        keep = det["scores"] > SCORE_THRESHOLD
        boxes, scores, labels, batch = det["boxes"][keep], det["scores"][keep], det["labels"][keep], det["batch"][keep]
        idx = nms3d(boxes, scores, NMS_IOU, labels=labels, batch=batch)
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
    parser = argparse.ArgumentParser(description="Benchmark VoxelNeXt 3D detection on nuScenes.")
    parser.add_argument("--model", default="voxelnext.nuscenes.openpcdet", help="Registered detection model name")
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--root", default=DATA_DIR, help="Dataset root directory.")
    parser.add_argument("--split", default="val", choices=("val", "mini"), help="nuScenes `val`, or the mini release.")
    parser.add_argument("--max-sweeps", default=10, type=int, help="LiDAR sweeps per keyframe.")
    parser.add_argument("--seed", default=SEED, type=int)
    parser.add_argument("--batch-size", default=2, type=int)
    parser.add_argument("--num-workers", default=NUM_WORKERS, type=int)
    parser.add_argument("--limit", default=None, type=int, help="Evaluate at most this many keyframes.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    set_determinism(tf32=False)

    print(f"Benchmarking model {args.model!r} on nuScenes ({args.split})!")
    model, model_info = create_model(args.model, task="detection", pretrained=True, return_info=True)
    assert isinstance(model, DetectionModel)

    dataset: Dataset
    if args.split == "mini":
        dataset = NuScenesMini(
            root=args.root,
            max_sweeps=args.max_sweeps,
            transform=model_info["transform"],
        )
    else:
        dataset = NuScenes(
            root=args.root,
            split="val",
            max_sweeps=args.max_sweeps,
            transform=model_info["transform"],
        )

    if args.limit is not None:
        n = min(int(args.limit), len(dataset))
        dataset = Subset(dataset, range(n))
        print(f"Evaluating on a subset of the first {n} keyframes.")

    dataloader = PointCloudDataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        cat_keys=[DataKeys.BOX, DataKeys.POS_VOXEL],
    )

    print(f"Test set: {len(dataset)} keyframes")
    metrics = evaluate(model, dataloader, args.device)
    print("\nResults:")
    for key, value in metrics.items():
        print(f"  {key:<24} {value:.4f}")


if __name__ == "__main__":
    main()
