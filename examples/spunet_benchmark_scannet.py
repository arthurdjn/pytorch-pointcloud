"""Benchmark SpUNet (SparseUNet) semantic segmentation on ScanNet.

By default, evaluates at full point resolution: voxel-level logits are
broadcast back to all raw points via the inverse cluster mapping (matches
Pointcept's val pipeline). Pass `--voxel-eval` to evaluate at voxel resolution
(faster, but biased — small voxels weighted same as large ones).

Usage:
    uv run --no-sync python examples/spunet_benchmark_scannet.py --limit 5
"""

import argparse
import os
import time
from typing import Any, Callable, Dict

import torch
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

import torch_pointcloud.transforms as T
from torch_pointcloud.config import DATA_DIR
from torch_pointcloud.datasets import ScanNet20
from torch_pointcloud.models import create_model
from torch_pointcloud.utils.data import DataKeys, collate
from torch_pointcloud.utils.metrics import confusion_matrix
from torch_pointcloud.utils.random import seed_everything

ORIGIN_SEGMENT_KEY = "origin_segment"
CLUSTER_KEY = "cluster"

CUDA_AVAILABLE = torch.cuda.is_available()
CPU_COUNT = os.cpu_count()
DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"
NUM_WORKERS = CPU_COUNT // 2 if CPU_COUNT is not None else 0
BATCH_SIZE = 1
SEED = 42


@torch.inference_mode()
def forward_once(
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


def _full_resolution_transforms() -> Callable:
    """Like the registered transforms but also stores the per-raw-point cluster
    index and the pre-voxelization segment, so we can broadcast voxel logits
    back to per-point predictions (matches Pointcept's val evaluation)."""
    return T.Compose(
        [
            T.CenterShift(keys=DataKeys.POS, apply_z=True),
            T.Divide(keys=DataKeys.COLOR, divisor=255),
            T.Cat(keys=[DataKeys.COLOR, DataKeys.NORMAL], dst_key=DataKeys.X, dim=1),
            T.Relabel(keys=DataKeys.SEGMENT, labels=range(1, 21), default=-1),
            T.CopyItems(keys=DataKeys.SEGMENT, names=ORIGIN_SEGMENT_KEY),
            T.VoxelGrid(
                pos_key=DataKeys.POS,
                pos_reduce="grid",
                keys=[DataKeys.X, DataKeys.SEGMENT],
                reduce=["first", "first"],
                size=0.02,
                method="fnv",
                cluster_key=CLUSTER_KEY,
            ),
            T.CopyItems(keys=DataKeys.POS, names=DataKeys.POS_GRID),
        ]
    )


@torch.no_grad()
def evaluate(
    model: Module,
    dataloader: DataLoader,
    device: str,
    num_classes: int,
    full_resolution: bool = True,
) -> Dict[str, Any]:
    model.to(device).eval()
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    total_latency_ms = 0.0
    total_points = 0

    pbar = tqdm(dataloader, total=len(dataloader), desc="Testing")
    for data in pbar:
        x = data[DataKeys.X].to(device)
        pos = data[DataKeys.POS].to(device)
        batch = data[DataKeys.BATCH].to(device)

        logits, latency_ms = forward_once(model, x, pos, batch, device)

        if full_resolution:
            cluster = data[CLUSTER_KEY].to(device)  # (N_raw,) → voxel idx
            target = data[ORIGIN_SEGMENT_KEY].to(device)  # (N_raw,) raw labels (already 0-19)
            preds = logits[cluster].argmax(dim=1)  # broadcast voxel preds to raw points
        else:
            target = data[DataKeys.SEGMENT].to(device)
            preds = logits.argmax(dim=1)

        cm += confusion_matrix(preds.cpu(), target.cpu(), num_classes, ignore_index=-1)
        total_latency_ms += latency_ms
        total_points += int(target.shape[0])
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
    parser = argparse.ArgumentParser(description="Benchmark SpUNet semantic segmentation on ScanNet.")
    parser.add_argument("--model", default="spunet-v1m1.scannet20", help="Registered segmentation model name")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--root", default=DATA_DIR, help="Dataset root directory.")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--seed", default=SEED, type=int)
    parser.add_argument("--batch-size", default=BATCH_SIZE, type=int)
    parser.add_argument("--num-workers", default=NUM_WORKERS, type=int)
    parser.add_argument("--limit", default=None, type=int, help="Evaluate at most this many scenes.")
    parser.add_argument("--download", action="store_true", help="Download ScanNet if missing.")
    parser.add_argument("--force-process", action="store_true", help="Force re-processing the dataset.")
    parser.add_argument(
        "--voxel-eval",
        action="store_true",
        help="Evaluate at voxel resolution (faster, biased). Default: full point resolution.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    print(f"Benchmarking model {args.model!r} on ScanNet (split={args.split!r})!")
    model, model_info = create_model(args.model, task="segmentation", pretrained=True, return_info=True)
    num_classes = int(model.num_classes)
    transform = model_info.get("transforms") if args.voxel_eval else _full_resolution_transforms()

    dataset: Dataset = ScanNet20(
        root=args.root,
        split=args.split,
        transform=transform,
        download=args.download,
        force_process=args.force_process,
        num_workers=args.num_workers,
        use_axis_alignment=False,
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

    print(f"Test set: {len(dataset)} scenes  (eval @ {'voxel' if args.voxel_eval else 'full point'} resolution)")  # type: ignore[arg-type]
    metrics = evaluate(model, dataloader, args.device, num_classes, full_resolution=not args.voxel_eval)
    print("\nResults:")
    for key, value in metrics.items():
        print(f"  {key:<24} {value:.4f}")


if __name__ == "__main__":
    main()
