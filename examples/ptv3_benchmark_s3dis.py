"""Benchmark Point Transformer V3 semantic segmentation on S3DIS Area-5.

The released `s3dis-semseg-pt-v3m1-0-rpe` weights (`legacy=True`, rpe attention) were trained on mesh-derived
normals. The repo's `S3DIS` dataset ships without normals, so the registered transform estimates them from the
coordinates by local PCA (`EstimateNormals`). These only approximate the training normals, so the mIoU here is
below the published 73.6 (which also uses test-time augmentation); the script is meant to be runnable, not exact.

Usage:
    uv run --no-sync python examples/ptv3_benchmark_s3dis.py --limit 5
"""

import argparse
import os
import time
from typing import Any, Dict

import torch
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from torch_pointcloud.config import DATA_DIR
from torch_pointcloud.datasets import S3DIS
from torch_pointcloud.models import create_model
from torch_pointcloud.utils.data import DataKeys, collate
from torch_pointcloud.utils.metrics import confusion_matrix
from torch_pointcloud.utils.random import seed_everything

CUDA_AVAILABLE = torch.cuda.is_available()
CPU_COUNT = os.cpu_count()
DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"
NUM_WORKERS = CPU_COUNT // 2 if CPU_COUNT is not None else 0
SEED = 42


@torch.inference_mode()
def predict(
    model: torch.nn.Module,
    feat: torch.Tensor,
    grid_coord: torch.Tensor,
    batch: torch.Tensor,
    device: str,
) -> tuple[torch.Tensor, float]:
    if device.startswith("cuda"):
        torch.cuda.synchronize()

    start = time.perf_counter()
    logits = model(feat, grid_coord, batch)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - start) * 1000.0
    return logits, latency_ms


@torch.no_grad()
def evaluate(model: Module, dataloader: DataLoader, device: str, num_classes: int) -> Dict[str, Any]:
    model.to(device).eval()
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    total_latency_ms = 0.0
    total_points = 0

    pbar = tqdm(dataloader, total=len(dataloader), desc="Testing")
    for data in pbar:
        x = data[DataKeys.X].to(device)
        pos_grid = data[DataKeys.POS_GRID].to(device)
        batch = data[DataKeys.BATCH].to(device)
        target = data[DataKeys.SEGMENT].to(device)

        logits, latency_ms = predict(model, x, pos_grid, batch, device)
        preds = logits.argmax(dim=1)

        cm += confusion_matrix(preds.cpu(), target.cpu(), num_classes, ignore_index=-1)
        total_latency_ms += latency_ms
        total_points += int(x.shape[0])
        oa = cm.diag().sum().float() / cm.sum().float().clamp_min(1)
        pbar.set_postfix({"oa": f"{oa.item():.4f}"})

    intersection = cm.diag().float()
    union = cm.sum(dim=1).float() + cm.sum(dim=0).float() - intersection
    iou_per_class = intersection / union.clamp_min(1e-10)
    return {
        "test/mIoU": iou_per_class.mean().item(),
        "test/oa": (cm.diag().sum().float() / cm.sum().float().clamp_min(1)).item(),
        "test/latency_ms": total_latency_ms / max(len(dataloader), 1),
        "test/points_per_second": total_points / max(total_latency_ms / 1000.0, 1e-12),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark PT-v3 semantic segmentation on S3DIS Area-5.")
    parser.add_argument("--model", default="ptv3-base.s3dis-area5", help="Registered segmentation model name")
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

    print(f"Benchmarking model {args.model!r} on S3DIS (areas={args.areas})!")
    model, model_info = create_model(args.model, task="segmentation", pretrained=True, return_info=True)
    num_classes = int(model.num_classes)
    transform = model_info.get("transforms")

    dataset: Dataset = S3DIS(
        root=args.root,
        areas=args.areas,
        transform=transform,
        download=args.download,
        force_process=args.force_process,
        num_workers=args.num_workers,
    )
    if args.limit is not None:
        n = min(int(args.limit), len(dataset))  # type: ignore[arg-type]
        dataset = Subset(dataset, range(n))
        print(f"Evaluating on a subset of the first {n} rooms.")

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=collate)

    print(f"Test set: {len(dataset)} rooms")  # type: ignore[arg-type]
    metrics = evaluate(model, dataloader, args.device, num_classes)
    print("\nResults:")
    for key, value in metrics.items():
        print(f"  {key:<24} {value:.4f}")


if __name__ == "__main__":
    main()
