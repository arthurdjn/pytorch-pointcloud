"""Benchmark LION on nuScenes with the official detection metrics.

NOTE: defaults to the `v1.0-mini` split, whose numbers are smoke checks; only the full `val` split is comparable.

Results (nuScenes val, mAP / NDS):

    | Variant                     | reference   | torch-pointcloud |
    | --------------------------- | ----------- | ---------------- |
    | lion-mamba.nuscenes.zhe-liu | 68.0 / 72.1 |                  |

Usage:
    uv run --no-sync python examples/lion_benchmark_detection.py --root /path/to/data
"""

import argparse
import os
from typing import Dict, List, Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset, Subset
from tqdm import tqdm

from torch_pointcloud.config import DATA_DIR
from torch_pointcloud.datasets import NuScenesMini
from torch_pointcloud.datasets.nuscenes import NUSCENES_DETECTION_CLASSES, velocity_attributes
from torch_pointcloud.models import create_model
from torch_pointcloud.models.lion import LIONDetection
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


@torch.no_grad()
def evaluate(model: LIONDetection, dataloader: PointCloudDataLoader, device: str) -> Dict[str, float]:
    model.to(device).eval()
    preds: List[Detection3D] = []
    targets: List[Boxes3D] = []
    gt_velocities: List[Tensor] = []
    gt_num_points: List[Tensor] = []
    gt_attributes: List[Tensor] = []
    offsets: List[int] = [0]

    for data in tqdm(dataloader, total=len(dataloader), desc="Testing"):
        data = {key: value.to(device) if torch.is_tensor(value) else value for key, value in data.items()}
        output = model(data[DataKeys.X], data[DataKeys.POS], data[f"batch_{DataKeys.POS}"])
        det = circular_nms(model.decode(output), model.head.local_max_classes, model.head.nms_radius)
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
    parser = argparse.ArgumentParser(description="Benchmark LION 3D detection on nuScenes.")
    parser.add_argument("--model", default="lion-mamba.nuscenes.zhe-liu", help="Registered detection model name")
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--root", default=DATA_DIR, help="Dataset root directory.")
    parser.add_argument("--version", default="v1.0-mini", help="nuScenes version.")
    parser.add_argument("--max-sweeps", default=10, type=int, help="LiDAR sweeps per keyframe.")
    parser.add_argument("--seed", default=SEED, type=int)
    parser.add_argument("--batch-size", default=1, type=int)
    parser.add_argument("--num-workers", default=NUM_WORKERS, type=int)
    parser.add_argument("--limit", default=None, type=int, help="Evaluate at most this many keyframes.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    set_determinism(tf32=False)

    print(f"Benchmarking model {args.model!r} on nuScenes ({args.version})!")
    model, model_info = create_model(args.model, task="detection", pretrained=True, return_info=True)
    assert isinstance(model, LIONDetection)

    dataset: Dataset = NuScenesMini(
        root=args.root,
        version=args.version,
        max_sweeps=args.max_sweeps,
        transform=model_info["transform"],
    )
    if args.limit is not None:
        n = min(int(args.limit), len(dataset))  # type: ignore[arg-type]
        dataset = Subset(dataset, range(n))
        print(f"Evaluating on a subset of the first {n} keyframes.")

    dataloader = PointCloudDataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        cat_keys=[DataKeys.POS, DataKeys.X, DataKeys.BOX],
    )

    print(f"Test set: {len(dataset)} keyframes")  # type: ignore[arg-type]
    metrics = evaluate(model, dataloader, args.device)
    print("\nResults:")
    for key, value in metrics.items():
        print(f"  {key:<24} {value:.4f}")


if __name__ == "__main__":
    main()
