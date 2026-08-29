"""Benchmark the Concerto linear-probe head on ScanNet with the indoor precise-evaluation protocol.

Results (ScanNet val mIoU):

    | Variant                               | reference | torch-pointcloud |
    | ------------------------------------- | --------- | ---------------- |
    | concerto-large-lp.scannet20.pointcept | 77.5      | 78.59 / 92.29    |

Usage:
    uv run --no-sync python examples/concerto_benchmark_segmentation.py --limit 5
    uv run --no-sync python examples/concerto_benchmark_segmentation.py --download
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
from torch_pointcloud.datasets import ScanNet20
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

TRANSFORM = T.Compose(
    [
        T.Shift(keys=DataKeys.POS, method="bbox", axes=[0, 1]),
        T.Shift(keys=DataKeys.POS, method="min", axes=[2]),
        T.Divide(keys=DataKeys.COLOR, divisor=255),
        T.Relabel(keys=DataKeys.SEGMENT, labels=range(1, 21), default=-1),
    ]
)
INFERER_TRANSFORM = T.Compose(
    [
        T.Quantize(keys=DataKeys.POS, size=VOXEL_SIZE, dst_keys=DataKeys.POS_GRID),
        T.Shift(keys=DataKeys.POS, method="bbox", axes=[0, 1]),
        T.Cat(keys=[DataKeys.POS, DataKeys.COLOR, DataKeys.NORMAL], dst_key=DataKeys.X, dim=1),
    ]
)


def build_inferer(seed: int) -> Inferer:
    base = VoxelPartitionInferer(
        voxel_size=VOXEL_SIZE,
        transform=INFERER_TRANSFORM,
        softmax=True,
        reduce="sum",
        seed=seed,
    )
    return TTAInferer(base=base, transforms=simple_tta_transforms(), aggregate="mean")


@torch.no_grad()
def evaluate(model: Module, dataloader: DataLoader, inferer: Inferer, device: str, num_classes: int) -> Dict[str, Any]:
    model.to(device).eval()
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)

    pbar = tqdm(dataloader, total=len(dataloader), desc="Testing")
    for data in pbar:
        data = {key: value.to(device) if torch.is_tensor(value) else value for key, value in data.items()}
        scores = inferer(data, predictor=lambda d: model(d[DataKeys.X], d[DataKeys.POS_GRID], d[DataKeys.BATCH]))
        preds = scores.argmax(dim=1)
        cm += confusion_matrix(preds.cpu(), data[DataKeys.SEGMENT].cpu(), num_classes, ignore_index=-1)
        oa = cm.diag().sum().float() / cm.sum().float().clamp_min(1)
        pbar.set_postfix({"oa": f"{oa.item():.4f}"})

    intersection = cm.diag().float()
    union = cm.sum(dim=1).float() + cm.sum(dim=0).float() - intersection
    iou_per_class = intersection / union.clamp_min(1e-10)
    return {
        "test/mIoU": iou_per_class.mean().item(),
        "test/oa": (cm.diag().sum().float() / cm.sum().float().clamp_min(1)).item(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the Concerto linear-probe head on ScanNet with the indoor precise-evaluation protocol."
    )
    parser.add_argument(
        "--model", default="concerto-large-lp.scannet20.pointcept", help="Registered segmentation model name"
    )
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--root", default=DATA_DIR, help="Dataset root directory.")
    parser.add_argument("--seed", default=SEED, type=int)
    parser.add_argument("--num-workers", default=NUM_WORKERS, type=int)
    parser.add_argument("--limit", default=None, type=int, help="Evaluate at most this many scenes.")
    parser.add_argument("--download", action="store_true", help="Download ScanNet if missing.")
    parser.add_argument("--force-process", action="store_true", help="Force re-processing the dataset.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    set_determinism(tf32=False)

    print(f"Benchmarking model {args.model!r} on ScanNet!")
    model = create_model(args.model, task="segmentation", pretrained=True)
    num_classes = int(model.num_classes)
    inferer = build_inferer(args.seed)

    dataset: Dataset = ScanNet20(
        root=args.root,
        split="val",
        transform=TRANSFORM,
        download=args.download,
        force_process=args.force_process,
        num_workers=args.num_workers,
        use_axis_alignment=False,
    )
    if args.limit is not None:
        n = min(int(args.limit), len(dataset))  # type: ignore[arg-type]
        dataset = Subset(dataset, range(n))
        print(f"Evaluating on a subset of the first {n} scenes.")

    dataloader = PointCloudDataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    print(f"Test set: {len(dataset)} scenes  (13 views x voxel-partition fragments, scored at full resolution)")  # type: ignore[arg-type]
    metrics = evaluate(model, dataloader, inferer, args.device, num_classes)
    print("\nResults:")
    for key, value in metrics.items():
        print(f"  {key:<24} {value:.4f}")


if __name__ == "__main__":
    main()
