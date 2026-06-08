"""Evaluate `second-openpcdet-multihead.nuscenes` on nuScenes mini with 3D mAP.

`NuScenesMini` -> `PointCloudDataLoader` -> model -> `model.decode` -> `mean_average_precision3d`.

The dataset aggregates `max_sweeps` LiDAR sweeps per keyframe and converts the global-frame
annotations to LiDAR boxes; results are scored with the generic oriented-3D mAP (not the official
nuScenes NDS). Defaults to the `v1.0-mini` split (404 keyframes).

Results (nuScenes v1.0-mini, 404 keyframes, generic oriented-3D mAP, not official NDS):

    mAP@0.25 = 52.99    mAP@0.5 = 38.71

Usage:
    uv run --no-sync python examples/second_benchmark_nuscenes.py --root "/path/to/parent"
"""

import os
from argparse import ArgumentParser, Namespace
from typing import Dict, List, Sequence

import torch
from tqdm import tqdm

from torch_pointcloud.config import DATA_DIR
from torch_pointcloud.datasets import NuScenesMini
from torch_pointcloud.models import create_model
from torch_pointcloud.models._base import DetectionModel
from torch_pointcloud.utils.data import DataKeys, PointCloudDataLoader
from torch_pointcloud.utils.metrics import mean_average_precision3d
from torch_pointcloud.utils.random import seed_everything
from torch_pointcloud.utils.types import Boxes3D, Detection3D

CPU_COUNT = os.cpu_count()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = CPU_COUNT // 2 if CPU_COUNT is not None else 0


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    model, info = create_model(args.model, task="detection", pretrained=True, return_info=True)
    assert isinstance(model, DetectionModel)
    model.to(args.device).eval()

    dataset = NuScenesMini(
        root=args.root,
        version=args.version,
        max_sweeps=args.max_sweeps,
        transform=info["transforms"],
    )
    loader = PointCloudDataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        cat_keys=[DataKeys.BOX, DataKeys.POS_VOXEL],
    )

    print(f"Benchmarking {args.model!r} on nuScenes ({len(dataset)} keyframes)!")
    metrics = evaluate(model, loader, args.device, iou_thresholds=args.ap_iou)
    print("\nResults:")
    for name, value in metrics.items():
        print(f"  {name:<10} {value * 100:.2f}")


@torch.no_grad()
def evaluate(
    model: DetectionModel, loader: PointCloudDataLoader, device: str, *, iou_thresholds: Sequence[float]
) -> Dict[str, float]:
    preds: List[Detection3D] = []
    targets: List[Boxes3D] = []
    for data in tqdm(loader, desc="nuScenes"):
        out = model(
            data[DataKeys.VOXEL].to(device),
            data[DataKeys.POS_VOXEL].to(device),
            data[DataKeys.VOXEL_NUM_POINTS].to(device),
            data[f"batch_{DataKeys.POS_VOXEL}"].to(device),
        )
        pred = model.decode(out)
        preds.append(
            {
                "boxes": pred["boxes"].cpu(),
                "scores": pred["scores"].cpu(),
                "labels": pred["labels"].cpu(),
                "batch": pred["batch"].cpu(),
            }
        )
        targets.append({"boxes": data[DataKeys.BOX], "labels": data[DataKeys.LABEL], "batch": data[DataKeys.BATCH_BOX]})
    return mean_average_precision3d(preds, targets, iou_thresholds=iou_thresholds)


def parse_args() -> Namespace:
    parser = ArgumentParser(description="SECOND-MultiHead nuScenes 3D detection mAP benchmark.")
    parser.add_argument("--model", type=str, default="second-openpcdet-multihead.nuscenes")
    parser.add_argument("--root", type=str, default=DATA_DIR, help="Parent directory containing NuScenesMini/.")
    parser.add_argument("--version", type=str, default="v1.0-mini")
    parser.add_argument("--max-sweeps", type=int, default=10)
    parser.add_argument("--ap-iou", type=float, nargs="+", default=[0.25, 0.5])
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--device", type=str, default=DEVICE)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    main()
