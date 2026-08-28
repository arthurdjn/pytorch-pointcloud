"""Benchmark PVCNN semantic segmentation on S3DIS Area 5 with the reference sliding-window protocol.

Results (Area-5 mIoU / OA):

    | Variant                       | reference | torch-pointcloud |
    | ----------------------------- | --------- | ---------------- |
    | pvcnn.s3dis-area5.mit-han-lab | 56.64     | 57.54 / 86.57    |

Usage:
    uv run --no-sync python examples/pvcnn_benchmark_segmentation.py
    uv run --no-sync python examples/pvcnn_benchmark_segmentation.py --limit 5
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
from torch_pointcloud.inferers import Inferer, SlidingWindowInferer
from torch_pointcloud.models import create_model
from torch_pointcloud.utils.data import DataKeys, PointCloudDataLoader
from torch_pointcloud.utils.metrics import confusion_matrix
from torch_pointcloud.utils.random import seed_everything, set_determinism

CUDA_AVAILABLE = torch.cuda.is_available()
CPU_COUNT = os.cpu_count()
DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"
NUM_WORKERS = CPU_COUNT // 2 if CPU_COUNT is not None else 0
SEED = 42
BLOCK_SIZE = 1.5
BLOCK_NUM_POINTS = 4096

TRANSFORM = T.Compose(
    [
        T.Shift(keys=DataKeys.POS, method="min"),
        T.Reduce(keys=DataKeys.POS, op="max", dst_keys="coord_max"),
    ]
)
INFERER_TRANSFORM = T.Compose(
    [
        T.BBoxCenter(keys="block_bbox", dst_keys=DataKeys.BLOCK_CENTER),
        T.CopyItems(keys=DataKeys.POS, names=DataKeys.NORM_POS),
        T.DivideKey(keys=DataKeys.NORM_POS, div_keys="coord_max"),
        T.SubtractKey(keys=DataKeys.POS, sub_keys=DataKeys.BLOCK_CENTER, axes=[0, 1]),
        T.ToFloat(keys=DataKeys.COLOR),
        T.Divide(keys=DataKeys.COLOR, divisor=255.0),
        T.Cat(keys=[DataKeys.POS, DataKeys.COLOR, DataKeys.NORM_POS], dst_key=DataKeys.X, dim=1),
        T.DivisiblePad(num_samples=BLOCK_NUM_POINTS, pad_fill="random", dst_inverse_key=DataKeys.INVERSE),
    ]
)


def build_inferer(seed: int) -> Inferer:
    return SlidingWindowInferer(
        block_size=BLOCK_SIZE,
        overlap=0.5,
        dims=(0, 1),
        roi_num_points=BLOCK_NUM_POINTS,
        softmax=True,
        aggregate="max",
        transform=INFERER_TRANSFORM,
        inverse_key=DataKeys.INVERSE,
        seed=seed,
    )


@torch.no_grad()
def evaluate(model: Module, dataloader: DataLoader, inferer: Inferer, device: str, num_classes: int) -> Dict[str, Any]:
    model.to(device).eval()
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)

    pbar = tqdm(dataloader, total=len(dataloader), desc="Testing")
    for data in pbar:
        data = {key: value.to(device) if torch.is_tensor(value) else value for key, value in data.items()}
        probs = inferer(data, predictor=lambda d: model(d[DataKeys.X], d[DataKeys.POS], d[DataKeys.BATCH]))
        preds = probs.argmax(dim=1)
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
    parser = argparse.ArgumentParser(description="Benchmark PVCNN semantic segmentation on S3DIS Area 5.")
    parser.add_argument("--model", default="pvcnn.s3dis-area5.mit-han-lab", help="Registered segmentation model name")
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--root", default=DATA_DIR, help="Dataset root directory.")
    parser.add_argument("--areas", nargs="+", default=["Area_5"])
    parser.add_argument("--seed", default=SEED, type=int)
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
    model = create_model(args.model, task="segmentation", pretrained=True)
    num_classes = int(model.num_classes)
    inferer = build_inferer(args.seed)

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

    print(f"Test set: {len(dataset)} rooms  ({BLOCK_SIZE} m blocks, 50% overlap, scored at full resolution)")  # type: ignore[arg-type]
    metrics = evaluate(model, dataloader, inferer, args.device, num_classes)
    print("\nResults:")
    for key, value in metrics.items():
        print(f"  {key:<24} {value:.4f}")


if __name__ == "__main__":
    main()
