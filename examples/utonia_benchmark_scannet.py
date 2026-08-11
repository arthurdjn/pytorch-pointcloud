"""Benchmark Utonia on ScanNet.

For the linear-probe segmentation variant (`utonia-lp.scannet20.pointcept`), reports mIoU
+ OA + latency on ScanNet20 val. For the encoder-only variant (`utonia.pretrain.pointcept`), only
latency / throughput is reported (no head, no labels). Utonia's forward takes
real-valued positions (`pos`) for 3D RoPE in addition to integer grid coords.

Single-forward, voxel-level mIoU on ScanNet20 val (published numbers add test-time augmentation and
full-resolution per-point evaluation):

| Model               | Here                  |
| ------------------- | --------------------- |
| utonia-lp.scannet20.pointcept | 71.12 mIoU / 89.05 OA |

Usage:
    uv run --no-sync python examples/utonia_benchmark_scannet.py --model utonia-lp.scannet20.pointcept --limit 5
    uv run --no-sync python examples/utonia_benchmark_scannet.py --model utonia.pretrain.pointcept --limit 5
"""

import argparse
import os
import time
from typing import Any, Dict, Tuple

import torch
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from torch_pointcloud.config import DATA_DIR
from torch_pointcloud.datasets import ScanNet20
from torch_pointcloud.models import create_model
from torch_pointcloud.utils.data import DataKeys, collate
from torch_pointcloud.utils.metrics import confusion_matrix
from torch_pointcloud.utils.random import seed_everything, set_determinism

CUDA_AVAILABLE = torch.cuda.is_available()
CPU_COUNT = os.cpu_count()
DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"
NUM_WORKERS = CPU_COUNT // 2 if CPU_COUNT is not None else 0
BATCH_SIZE = 1
SEED = 42

ENCODER_MODELS = ("utonia.pretrain.pointcept",)
SEG_MODELS = ("utonia-lp.scannet20.pointcept",)


def _resolve_task(model_name: str) -> str:
    if model_name in SEG_MODELS:
        return "segmentation"
    if model_name in ENCODER_MODELS:
        return "base"
    raise ValueError(f"Unknown Utonia model {model_name!r}.")


@torch.inference_mode()
def _forward_seg(
    model: Module,
    feat: torch.Tensor,
    pos: torch.Tensor,
    pos_grid: torch.Tensor,
    batch: torch.Tensor,
    device: str,
) -> Tuple[torch.Tensor, float]:
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    start = time.perf_counter()
    out = model(feat, pos, pos_grid, batch)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    return out, (time.perf_counter() - start) * 1000.0


@torch.inference_mode()
def _forward_encoder(
    model: Module,
    feat: torch.Tensor,
    pos: torch.Tensor,
    pos_grid: torch.Tensor,
    batch: torch.Tensor,
    device: str,
) -> float:
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    start = time.perf_counter()
    _ = model(feat, pos_grid, batch, pos=pos)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0


@torch.no_grad()
def _evaluate_segmentation(model: Module, dataloader: DataLoader, device: str, num_classes: int) -> Dict[str, Any]:
    model.to(device).eval()
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    total_latency_ms = 0.0
    total_points = 0

    pbar = tqdm(dataloader, total=len(dataloader), desc="Testing")
    for data in pbar:
        x = data[DataKeys.X].to(device)
        pos = data[DataKeys.POS].to(device)
        pos_grid = data[DataKeys.POS_GRID].to(device)
        batch = data[DataKeys.BATCH].to(device)
        target = data[DataKeys.SEGMENT].to(device)

        logits, latency_ms = _forward_seg(model, x, pos, pos_grid, batch, device)
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


@torch.no_grad()
def _benchmark_encoder(model: Module, dataloader: DataLoader, device: str) -> Dict[str, Any]:
    model.to(device).eval()
    total_latency_ms = 0.0
    total_points = 0

    pbar = tqdm(dataloader, total=len(dataloader), desc="Encoding")
    for data in pbar:
        x = data[DataKeys.X].to(device)
        pos = data[DataKeys.POS].to(device)
        pos_grid = data[DataKeys.POS_GRID].to(device)
        batch = data[DataKeys.BATCH].to(device)

        latency_ms = _forward_encoder(model, x, pos, pos_grid, batch, device)
        total_latency_ms += latency_ms
        total_points += int(x.shape[0])

    return {
        "test/latency_ms": total_latency_ms / max(len(dataloader), 1),
        "test/points_per_second": total_points / max(total_latency_ms / 1000.0, 1e-12),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Utonia on ScanNet.")
    parser.add_argument(
        "--model",
        default="utonia-lp.scannet20.pointcept",
        choices=[*ENCODER_MODELS, *SEG_MODELS],
        help="Registered Utonia model name.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--root", default=DATA_DIR, help="Dataset root directory.")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--seed", default=SEED, type=int)
    parser.add_argument("--batch-size", default=BATCH_SIZE, type=int)
    parser.add_argument("--num-workers", default=NUM_WORKERS, type=int)
    parser.add_argument("--limit", default=None, type=int, help="Evaluate at most this many scenes.")
    parser.add_argument("--download", action="store_true", help="Download ScanNet if missing.")
    parser.add_argument("--force-process", action="store_true", help="Force re-processing the dataset.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    set_determinism(tf32=False)

    task = _resolve_task(args.model)
    print(f"Benchmarking model {args.model!r} on ScanNet (split={args.split!r}, task={task!r})!")

    model, model_info = create_model(args.model, task=task, pretrained=True, return_info=True)  # type: ignore[call-overload]
    transform = model_info.get("transform")

    dataset: Dataset = ScanNet20(
        root=args.root,
        split=args.split,
        transform=transform,
        download=args.download,
        force_process=args.force_process,
        num_workers=args.num_workers,
    )
    if args.limit is not None:
        n = min(int(args.limit), len(dataset))  # type: ignore[arg-type]
        dataset = Subset(dataset, range(n))
        print(f"Evaluating on a subset of the first {n} scenes.")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
    )
    print(f"Test set: {len(dataset)} scenes")  # type: ignore[arg-type]

    if task == "segmentation":
        metrics = _evaluate_segmentation(model, dataloader, args.device, int(model.num_classes))
    else:
        metrics = _benchmark_encoder(model, dataloader, args.device)

    print("\nResults:")
    for key, value in metrics.items():
        print(f"  {key:<24} {value:.4f}")


if __name__ == "__main__":
    main()
