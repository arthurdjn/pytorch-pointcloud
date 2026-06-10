"""Evaluate `voxel-mamba-gwenzhang.waymo` on the Waymo Open Dataset with 3D mAP.

`Waymo` -> `PointCloudDataLoader` -> model -> `model.decode` -> `mean_average_precision3d`.

Voxel Mamba serializes voxels into a single Hilbert-curve sequence and runs bidirectional Mamba
(state-space) blocks as a group-free backbone, then a BEV residual backbone and a center-based head.
The model takes packed points `(pos, x, batch)` and voxelizes internally (dynamic mean VFE); the
registered transform builds the 2-channel point feature `x` from intensity and elongation.

!!! warning "Data and weights"
    The Waymo Open Dataset is license-gated and is not bundled with this repo, so this benchmark
    cannot be run here; it is provided ready to run once a `Waymo` detection dataset (returning
    `DataKeys.POS`, `DataKeys.INTENSITY`, `elongation`, `DataKeys.BOX`, `DataKeys.LABEL`) is
    available. The Voxel Mamba Waymo checkpoint is also license-gated (no redistributable weights);
    the registered checkpoint is a verified architecture-equivalent random initialization, so the
    numbers below are placeholders until official weights can be redistributed.

Results vs reference:

    | Source              | Waymo val (L2 mAPH)                                                |
    | ------------------- | ----------------------------------------------------------------- |
    | Voxel Mamba (paper) | 73.6 (Veh 72.2 / Ped 73.6 / Cyc 74.8)                             |
    | torch-pointcloud    | not run (Waymo data + weights license-gated); arch-equiv init (2.4e-7) |

Usage:
    uv run --no-sync python examples/voxel_mamba_benchmark_waymo.py --root "/path/to/parent"
"""

import os
from argparse import ArgumentParser, Namespace
from typing import Dict, List, Sequence

import torch
from tqdm import tqdm

from torch_pointcloud.config import DATA_DIR
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

    from torch_pointcloud.datasets import Waymo

    dataset = Waymo(root=args.root, split=args.split, transform=info["transforms"])
    loader = PointCloudDataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        cat_keys=[DataKeys.POS, DataKeys.X, DataKeys.BOX],
    )

    print(f"Benchmarking {args.model!r} on Waymo ({len(dataset)} frames)!")
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
    for data in tqdm(loader, desc="Waymo"):
        out = model(
            data[DataKeys.X].to(device),
            data[DataKeys.POS].to(device),
            data[f"batch_{DataKeys.POS}"].to(device),
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
    parser = ArgumentParser(description="Voxel Mamba Waymo 3D detection mAP benchmark.")
    parser.add_argument("--model", type=str, default="voxel-mamba-gwenzhang.waymo")
    parser.add_argument("--root", type=str, default=DATA_DIR, help="Parent directory containing Waymo/.")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--ap-iou", type=float, nargs="+", default=[0.25, 0.5])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--device", type=str, default=DEVICE)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    main()
