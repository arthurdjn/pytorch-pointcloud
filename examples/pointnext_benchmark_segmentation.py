"""Benchmark PointNeXt semantic segmentation on S3DIS with the voxel-partition test protocol.

Results (mIoU / OA; the 6-fold rows average the per-area folds, the reference pools their confusion matrices):

    | Variant                               | reference   | torch-pointcloud        |
    | ------------------------------------- | ----------- | ----------------------- |
    | pointnext-sm.s3dis-area1.openpoints   |             | 74.46 / 89.37           |
    | pointnext-sm.s3dis-area2.openpoints   |             | 47.63 / 78.69           |
    | pointnext-sm.s3dis-area3.openpoints   |             | 75.94 / 91.11           |
    | pointnext-sm.s3dis-area4.openpoints   |             | 59.94 / 86.46           |
    | pointnext-sm.s3dis-area5.openpoints   | 64.2 / 88.2 | 64.30 / 88.26           |
    | pointnext-sm.s3dis-area6.openpoints   |             | 83.23 / 93.17           |
    | pointnext-sm 6-fold mean              | 68.0 / 87.4 | 67.58 / 87.84           |
    | pointnext-base.s3dis-area1.openpoints |             | 77.73 / 90.46           |
    | pointnext-base.s3dis-area2.openpoints |             | 58.60 / 82.00           |
    | pointnext-base.s3dis-area3.openpoints |             | 84.00 / 93.26           |
    | pointnext-base.s3dis-area4.openpoints |             | 62.63 / 87.03           |
    | pointnext-base.s3dis-area5.openpoints | 67.5 / 89.4 | 67.51 / 89.42           |
    | pointnext-base.s3dis-area6.openpoints |             | 84.62 / 93.66           |
    | pointnext-base 6-fold mean            | 71.5 / 88.8 | 72.52 / 89.31           |
    | pointnext-lg.s3dis-area1.openpoints   |             | 78.98 / 91.13           |
    | pointnext-lg.s3dis-area2.openpoints   |             | 61.81 / 84.80           |
    | pointnext-lg.s3dis-area3.openpoints   |             | 84.06 / 93.39           |
    | pointnext-lg.s3dis-area4.openpoints   |             | 65.09 / 88.09           |
    | pointnext-lg.s3dis-area5.openpoints   | 69.3 / 90.1 | 69.29 / 90.03           |
    | pointnext-lg.s3dis-area6.openpoints   |             | 85.90 / 93.99           |
    | pointnext-lg 6-fold mean              | 73.9 / 89.8 | 74.19 / 90.24           |
    | pointnext-xl.s3dis-area1.openpoints   |             | 79.59 / 91.22           |
    | pointnext-xl.s3dis-area2.openpoints   |             | 63.16 / 85.55           |
    | pointnext-xl.s3dis-area3.openpoints   |             | 84.95 / 93.72           |
    | pointnext-xl.s3dis-area4.openpoints   |             | 64.78 / 88.61           |
    | pointnext-xl.s3dis-area5.openpoints   | 71.1 / 91.0 | 71.24 / 90.95           |
    | pointnext-xl 6-fold mean              | 74.9 / 90.3 | 72.74 / 90.01 (5 folds) |

Usage:
    uv run --no-sync python examples/pointnext_benchmark_segmentation.py --model pointnext-xl.s3dis-area5.openpoints
    uv run --no-sync python examples/pointnext_benchmark_segmentation.py --model pointnext-sm.s3dis-area1.openpoints --areas Area_1
"""

import argparse
import os
from typing import Any, Dict

import torch
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

import torch_pointcloud.transforms as T
from torch_pointcloud.config import DATA_DIR
from torch_pointcloud.datasets import S3DIS
from torch_pointcloud.inferers import Inferer, VoxelPartitionInferer
from torch_pointcloud.metrics import accuracy, confusion_matrix, intersection_over_union
from torch_pointcloud.models import create_model
from torch_pointcloud.utils.data import DataKeys, PointCloudDataLoader
from torch_pointcloud.utils.random import seed_everything, set_determinism

CUDA_AVAILABLE = torch.cuda.is_available()
CPU_COUNT = os.cpu_count()
DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"
NUM_WORKERS = CPU_COUNT // 2 if CPU_COUNT is not None else 0
SEED = 42
VOXEL_SIZE = 0.04
SW_BATCH_SIZE = 8

TRANSFORM = T.Compose(
    [
        T.Relabel(keys=DataKeys.SEGMENT, labels=[0, 1, 2, 3, 4, 5, 6, 8, 7, 10, 9, 11, 12]),
        T.Shift(keys=DataKeys.POS, method="min"),
    ]
)
INFERER_TRANSFORM = T.Compose(
    [
        T.AxisMinOffset(keys=DataKeys.POS, axis=2, dst_keys="height"),
        T.Shift(keys=DataKeys.POS, method="centroid"),
        T.Shift(keys=DataKeys.POS, method="min", axes=[2]),
        T.Divide(keys=DataKeys.COLOR, divisor=255.0),
        T.Normalize(
            keys=DataKeys.COLOR, mean=[0.5136457, 0.49523646, 0.44921124], std=[0.18308958, 0.18415008, 0.19252081]
        ),
        T.Cat(keys=[DataKeys.COLOR, "height"], dst_key=DataKeys.X),
    ]
)


def build_inferer(sw_batch_size: int, seed: int) -> Inferer:
    return VoxelPartitionInferer(
        voxel_size=VOXEL_SIZE,
        transform=INFERER_TRANSFORM,
        sw_batch_size=sw_batch_size,
        seed=seed,
    )


@torch.no_grad()
def evaluate(model: Module, dataloader: DataLoader, inferer: Inferer, device: str, num_classes: int) -> Dict[str, Any]:
    model.to(device).eval()
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)

    pbar = tqdm(dataloader, total=len(dataloader), desc="Testing")
    for data in pbar:
        data = {key: value.to(device) if torch.is_tensor(value) else value for key, value in data.items()}
        logits = inferer(data, predictor=lambda d: model(d[DataKeys.X], d[DataKeys.POS], d[DataKeys.BATCH]))
        preds = logits.argmax(dim=1)
        cm += confusion_matrix(preds.cpu(), data[DataKeys.SEGMENT].cpu(), num_classes, ignore_index=-1)
        oa = accuracy(cm)
        pbar.set_postfix({"oa": f"{oa:.4f}"})

    return {
        "test/mIoU": intersection_over_union(cm),
        "test/oa": accuracy(cm),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark PointNeXt semantic segmentation on S3DIS.")
    parser.add_argument(
        "--model", default="pointnext-sm.s3dis-area5.openpoints", help="Registered segmentation model name"
    )
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--root", default=DATA_DIR, help="Dataset root directory.")
    parser.add_argument("--areas", nargs="+", default=["Area_5"])
    parser.add_argument("--seed", default=SEED, type=int)
    parser.add_argument("--sw-batch-size", default=SW_BATCH_SIZE, type=int, help="Voxel fragments per forward.")
    parser.add_argument("--num-workers", default=NUM_WORKERS, type=int)
    parser.add_argument("--limit", default=None, type=int, help="Evaluate at most this many rooms.")
    parser.add_argument("--download", action="store_true", help="Download S3DIS if missing.")
    parser.add_argument("--force-process", action="store_true", help="Force re-processing the dataset.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    set_determinism(tf32=False)

    print(f"Benchmarking model {args.model!r} on S3DIS (areas={args.areas})!")
    model = create_model(args.model, task="semantic-segmentation", pretrained=True)
    num_classes = int(model.num_classes)
    inferer = build_inferer(args.sw_batch_size, args.seed)

    dataset: Dataset = S3DIS(
        root=args.root,
        areas=args.areas,
        transform=TRANSFORM,
        download=args.download,
        force_process=args.force_process,
        num_workers=args.num_workers,
    )
    if args.limit is not None:
        n = min(int(args.limit), len(dataset))  # type: ignore[arg-type]
        dataset = Subset(dataset, range(n))
        print(f"Evaluating on a subset of the first {n} rooms.")

    dataloader = PointCloudDataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    print(f"Test set: {len(dataset)} rooms  (voxel partition at {VOXEL_SIZE} m, scored at full resolution)")  # type: ignore[arg-type]
    metrics = evaluate(model, dataloader, inferer, args.device, num_classes)
    print("\nResults:")
    for key, value in metrics.items():
        print(f"  {key:<24} {value:.4f}")


if __name__ == "__main__":
    main()
