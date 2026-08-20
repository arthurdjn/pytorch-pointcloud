"""Evaluate `second-multihead.nuscenes.openpcdet` on nuScenes with the official detection metrics.

NOTE: defaults to the `v1.0-mini` split (404 keyframes); mini-split numbers are smoke checks, the full
`val` split is the run comparable to published results.

Results vs reference (official protocol, nuScenes val):

    | Source                             | mAP   | NDS   |
    | ---------------------------------- | ----- | ----- |
    | reference implementation model zoo | 50.59 | 62.29 |

Usage:
    uv run --no-sync python examples/second_benchmark_nuscenes.py --root "/path/to/parent"
"""

import os
from argparse import ArgumentParser, Namespace
from typing import Dict, List, Sequence

import torch
from torch import Tensor
from tqdm import tqdm

from torch_pointcloud.config import DATA_DIR
from torch_pointcloud.datasets import NuScenesMini
from torch_pointcloud.datasets.nuscenes import NUSCENES_ATTRIBUTES, NUSCENES_DETECTION_CLASSES
from torch_pointcloud.models import DetectionModel, create_model
from torch_pointcloud.utils.box3d import nms3d
from torch_pointcloud.utils.data import DataKeys, PointCloudDataLoader
from torch_pointcloud.utils.metrics import mean_average_precision3d, nuscenes_detection_metrics
from torch_pointcloud.utils.random import seed_everything, set_determinism
from torch_pointcloud.utils.types import Boxes3D, Detection3D

CPU_COUNT = os.cpu_count()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = CPU_COUNT // 2 if CPU_COUNT is not None else 0
SCORE_THRESHOLD = 0.1
NMS_IOU = 0.2
SPEED_THRESHOLD = 1.0  # m/s: faster boxes get their class's moving attribute, slower the parked / stopped default
MOVING_ATTRIBUTE = {
    "car": "vehicle.moving",
    "truck": "vehicle.moving",
    "construction_vehicle": "vehicle.moving",
    "bus": "vehicle.moving",
    "trailer": "vehicle.moving",
    "motorcycle": "cycle.with_rider",
    "bicycle": "cycle.with_rider",
    "pedestrian": "pedestrian.moving",
}
STOPPED_ATTRIBUTE = {
    "car": "vehicle.parked",
    "truck": "vehicle.parked",
    "construction_vehicle": "vehicle.parked",
    "bus": "vehicle.stopped",
    "trailer": "vehicle.parked",
    "motorcycle": "cycle.without_rider",
    "bicycle": "cycle.without_rider",
    "pedestrian": "pedestrian.standing",
}


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    set_determinism(tf32=False)

    model, info = create_model(args.model, task="detection", pretrained=True, return_info=True)
    assert isinstance(model, DetectionModel)
    model.to(args.device).eval()

    dataset = NuScenesMini(
        root=args.root,
        version=args.version,
        max_sweeps=args.max_sweeps,
        transform=info["transform"],
    )
    loader = PointCloudDataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        cat_keys=[DataKeys.BOX, DataKeys.POS_VOXEL],
    )

    print(f"Benchmarking {args.model!r} on nuScenes ({len(dataset)} keyframes)!")
    metrics = evaluate(model, loader, args.device, legacy_map=args.legacy_map, iou_thresholds=args.ap_iou)
    if args.legacy_map:
        print("\nResults (generic oriented-3D IoU mAP, non-reference):")
        for name, value in metrics.items():
            print(f"  {name:<10} {value * 100:.2f}")
    else:
        print("\nResults (official nuScenes detection metrics):")
        for name, value in metrics.items():
            print(f"  {name:<24} {value:.4f}")


def velocity_attributes(labels: Tensor, velocity: Tensor) -> Tensor:
    """Attribute id per box: the class's moving attribute above `SPEED_THRESHOLD` BEV speed, its parked /
    stopped / standing default below; `barrier` and `traffic_cone` carry no attribute (id -1)."""
    moving = torch.linalg.norm(velocity, dim=1) > SPEED_THRESHOLD
    attributes = torch.full_like(labels, -1)
    for index, name in enumerate(NUSCENES_DETECTION_CLASSES):
        if name not in MOVING_ATTRIBUTE:
            continue
        mask = labels == index
        attributes[mask & moving] = NUSCENES_ATTRIBUTES.index(MOVING_ATTRIBUTE[name])
        attributes[mask & ~moving] = NUSCENES_ATTRIBUTES.index(STOPPED_ATTRIBUTE[name])
    return attributes


@torch.no_grad()
def evaluate(
    model: DetectionModel,
    loader: PointCloudDataLoader,
    device: str,
    *,
    legacy_map: bool,
    iou_thresholds: Sequence[float],
) -> Dict[str, float]:
    preds: List[Detection3D] = []
    targets: List[Boxes3D] = []
    gt_velocities: List[Tensor] = []
    gt_num_points: List[Tensor] = []
    gt_attributes: List[Tensor] = []
    sizes: List[int] = []
    for data in tqdm(loader, desc="nuScenes"):
        out = model(
            data[DataKeys.VOXEL].to(device),
            data[DataKeys.POS_VOXEL].to(device),
            data[DataKeys.VOXEL_NUM_POINTS].to(device),
            data[f"batch_{DataKeys.POS_VOXEL}"].to(device),
        )
        det = model.decode(out)
        boxes, scores, labels, batch = det["boxes"], det["scores"], det["labels"], det["batch"]
        velocity = det["velocity"]
        keep = scores > SCORE_THRESHOLD
        boxes, scores, labels, batch, velocity = boxes[keep], scores[keep], labels[keep], batch[keep], velocity[keep]
        idx = nms3d(boxes, scores, NMS_IOU, labels=labels, batch=batch)
        preds.append(
            {
                "boxes": boxes[idx].cpu(),
                "scores": scores[idx].cpu(),
                "labels": labels[idx].cpu(),
                "batch": batch[idx].cpu(),
                "velocity": velocity[idx].cpu(),
            }
        )
        targets.append({"boxes": data[DataKeys.BOX], "labels": data[DataKeys.LABEL], "batch": data[DataKeys.BATCH_BOX]})
        gt_velocities.append(data[DataKeys.VELOCITY])
        gt_num_points.append(data[DataKeys.NUM_POINTS])
        gt_attributes.append(data[DataKeys.ATTRIBUTE])
        sizes.append(len(data[DataKeys.TOKEN]))
    if legacy_map:
        return mean_average_precision3d(preds, targets, iou_thresholds=iou_thresholds)

    offsets = [0]
    for size in sizes[:-1]:
        offsets.append(offsets[-1] + size)
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


def parse_args() -> Namespace:
    parser = ArgumentParser(description="SECOND-MultiHead nuScenes detection benchmark (official metrics).")
    parser.add_argument("--model", type=str, default="second-multihead.nuscenes.openpcdet")
    parser.add_argument("--root", type=str, default=DATA_DIR, help="Parent directory containing NuScenesMini/.")
    parser.add_argument("--version", type=str, default="v1.0-mini")
    parser.add_argument("--max-sweeps", type=int, default=10)
    parser.add_argument(
        "--legacy-map",
        action="store_true",
        help="Score with the generic oriented-3D IoU mAP (the non-reference legacy protocol) instead.",
    )
    parser.add_argument("--ap-iou", type=float, nargs="+", default=[0.25, 0.5], help="--legacy-map IoU thresholds.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--device", type=str, default=DEVICE)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    main()
