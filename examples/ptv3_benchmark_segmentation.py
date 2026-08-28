"""Benchmark Point Transformer V3 semantic segmentation with the indoor precise-evaluation protocols.

NOTE: the S3DIS checkpoint was trained with mesh normals the public download lacks; normals are estimated here, so that
row cannot reach the reference.

NOTE: the ScanNet references are the checkpoints' training-log figures; the reference tester has not been re-run on this
machine for PTv3 (the SpUNet script shows the same kind of gap between a 2023 log and today's data).

Results (val mIoU):

    | Variant                         | reference | torch-pointcloud |
    | ------------------------------- | --------- | ---------------- |
    | ptv3-base.scannet20.pointcept   | 77.6      | 76.29 / 91.39    |
    | ptv3-base.scannet200.pointcept  | 35.3      | 33.42 / 82.97    |
    | ptv3-base.s3dis-area5.pointcept | 73.6      | 32.06 / 69.93    |

Usage:
    uv run --no-sync python examples/ptv3_benchmark_segmentation.py --model ptv3-base.scannet20.pointcept --limit 5
    uv run --no-sync python examples/ptv3_benchmark_segmentation.py --model ptv3-base.s3dis-area5.pointcept
"""

import argparse
import os
from typing import Any, Dict, List

import torch
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

import torch_pointcloud.transforms as T
from torch_pointcloud.config import DATA_DIR
from torch_pointcloud.datasets import S3DIS, ScanNet20, ScanNet200
from torch_pointcloud.inferers import Inferer, TTAInferer, VoxelPartitionInferer, simple_tta_transforms
from torch_pointcloud.models import create_model
from torch_pointcloud.utils.data import DataKeys, PointCloudDataLoader
from torch_pointcloud.utils.metrics import confusion_matrix
from torch_pointcloud.utils.random import seed_everything, set_determinism

CUDA_AVAILABLE = torch.cuda.is_available()
CPU_COUNT = os.cpu_count()
DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"
NUM_WORKERS = CPU_COUNT // 2 if CPU_COUNT is not None else 0
SEED = 42
VOXEL_SIZE = 0.02

INFERER_TRANSFORM = T.Compose(
    [
        T.Cat(keys=[DataKeys.COLOR, DataKeys.NORMAL], dst_key=DataKeys.X, dim=1),
        T.Quantize(keys=DataKeys.POS, size=VOXEL_SIZE, dst_keys=DataKeys.POS_GRID),
    ]
)
S3DIS_TRANSFORM = T.Compose(
    [
        T.Shift(keys=DataKeys.POS, method="bbox", axes=[0, 1]),
        T.Shift(keys=DataKeys.POS, method="min", axes=[2]),
        T.Divide(keys=DataKeys.COLOR, divisor=255),
        T.EstimateNormals(keys=DataKeys.POS, normal_key=DataKeys.NORMAL, orient_to_centroid=True),
    ]
)
S3DIS_VIEWS: List[T.Compose] = [
    T.Compose([T.RandomScale(keys=DataKeys.POS, scale_range=(scale, scale), p=1.0), *flip])
    for scale in (0.9, 0.95, 1.0, 1.05, 1.1)
    for flip in ([], [T.RandomFlip(keys=[DataKeys.POS, DataKeys.NORMAL], axes=(0, 1), p=1.0)])
]


def scannet_transform(num_classes: int) -> T.Compose:
    return T.Compose(
        [
            T.Shift(keys=DataKeys.POS, method="bbox", axes=[0, 1]),
            T.Shift(keys=DataKeys.POS, method="min", axes=[2]),
            T.Divide(keys=DataKeys.COLOR, divisor=255),
            T.Relabel(keys=DataKeys.SEGMENT, labels=range(1, num_classes + 1), default=-1),
        ]
    )


def build_inferer(views: List[T.Compose], seed: int) -> Inferer:
    base = VoxelPartitionInferer(
        voxel_size=VOXEL_SIZE,
        transform=INFERER_TRANSFORM,
        softmax=True,
        reduce="sum",
        seed=seed,
    )
    return TTAInferer(base=base, transforms=views, aggregate="mean")


@torch.no_grad()
def evaluate(model: Module, dataloader: DataLoader, inferer: Inferer, device: str, num_classes: int) -> Dict[str, Any]:
    model.to(device).eval()
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    skipped = 0

    pbar = tqdm(dataloader, total=len(dataloader), desc="Testing")
    for data in pbar:
        data = {key: value.to(device) if torch.is_tensor(value) else value for key, value in data.items()}
        try:
            scores = inferer(data, predictor=lambda d: model(d[DataKeys.X], d[DataKeys.POS_GRID], d[DataKeys.BATCH]))
        except torch.cuda.OutOfMemoryError:
            # RPE attention materialises the full per-patch score matrix, so the largest rooms can exceed GPU memory.
            torch.cuda.empty_cache()
            skipped += 1
            continue

        preds = scores.argmax(dim=1)
        cm += confusion_matrix(preds.cpu(), data[DataKeys.SEGMENT].cpu(), num_classes, ignore_index=-1)
        oa = cm.diag().sum().float() / cm.sum().float().clamp_min(1)
        pbar.set_postfix({"oa": f"{oa.item():.4f}", "skipped": skipped})

    intersection = cm.diag().float()
    union = cm.sum(dim=1).float() + cm.sum(dim=0).float() - intersection
    iou_per_class = intersection / union.clamp_min(1e-10)

    return {
        "test/mIoU": iou_per_class.mean().item(),
        "test/oa": (cm.diag().sum().float() / cm.sum().float().clamp_min(1)).item(),
        "test/skipped": float(skipped),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Point Transformer V3 semantic segmentation.")
    parser.add_argument(
        "--model",
        default="ptv3-base.scannet20.pointcept",
        choices=["ptv3-base.scannet20.pointcept", "ptv3-base.scannet200.pointcept", "ptv3-base.s3dis-area5.pointcept"],
    )
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--root", default=DATA_DIR, help="Dataset root directory.")
    parser.add_argument("--seed", default=SEED, type=int)
    parser.add_argument("--num-workers", default=NUM_WORKERS, type=int)
    parser.add_argument("--limit", default=None, type=int, help="Evaluate at most this many scenes.")
    parser.add_argument("--download", action="store_true", help="Download the dataset if missing.")
    parser.add_argument("--force-process", action="store_true", help="Force re-processing the dataset.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    set_determinism(tf32=False)

    model = create_model(args.model, task="segmentation", pretrained=True)
    num_classes = int(model.num_classes)

    dataset: Dataset
    if args.model == "ptv3-base.s3dis-area5.pointcept":
        print(f"Benchmarking model {args.model!r} on S3DIS Area 5!")
        dataset = S3DIS(
            root=args.root,
            areas=["Area_5"],
            transform=S3DIS_TRANSFORM,
            download=args.download,
            force_process=args.force_process,
            num_workers=args.num_workers,
        )
        inferer = build_inferer(S3DIS_VIEWS, args.seed)
    else:
        print(f"Benchmarking model {args.model!r} on ScanNet!")
        scannet = ScanNet200 if num_classes == 200 else ScanNet20
        dataset = scannet(
            root=args.root,
            split="val",
            transform=scannet_transform(num_classes),
            download=args.download,
            force_process=args.force_process,
            num_workers=args.num_workers,
            use_axis_alignment=False,
        )
        inferer = build_inferer(simple_tta_transforms(), args.seed)

    if args.limit is not None:
        n = min(int(args.limit), len(dataset))
        dataset = Subset(dataset, range(n))
        print(f"Evaluating on a subset of the first {n} scenes.")

    dataloader = PointCloudDataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    print(f"Test set: {len(dataset)} scenes  (test-time views x voxel-partition fragments, scored at full resolution)")
    metrics = evaluate(model, dataloader, inferer, args.device, num_classes)
    print("\nResults:")
    for key, value in metrics.items():
        print(f"  {key:<24} {value:.4f}")


if __name__ == "__main__":
    main()
