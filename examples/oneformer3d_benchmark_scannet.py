"""Benchmark OneFormer3D semantic segmentation on ScanNet.

The released OneFormer3D model expects precomputed superpoints (Felzenszwalb
mesh segmentation, as in the official repo's ScanNet preprocessing). When
superpoints are present in the loaded sample under the `superpoint` key, the
benchmark uses them as-is. Otherwise, each voxel is treated as its own
superpoint; the model still runs and produces structurally correct outputs,
but quality is below the paper's reported numbers.

To reproduce the paper metrics on ScanNet val (mIoU 76.4, mAP50 78.8, PQ 70.7),
process the dataset with the upstream preprocessing pipeline and store the
per-point superpoint indices under `super_points/` (matching the official
`scannet_oneformer3d_infos_val.pkl` layout), then expose them via a custom
dataset that loads them under `DataKeys.SEGMENT` or a `superpoint` extra key.

Usage:
    uv run --no-sync python examples/oneformer3d_benchmark_scannet.py --limit 5
    uv run --no-sync python examples/oneformer3d_benchmark_scannet.py --variant scannet200
"""

import argparse
import os
import time
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from torch_pointcloud.config import DATA_DIR
from torch_pointcloud.datasets import ScanNet20
from torch_pointcloud.models import create_model
from torch_pointcloud.models.oneformer3d import OneFormer3DSegmentation
from torch_pointcloud.utils.data import DataKeys, collate
from torch_pointcloud.utils.metrics import confusion_matrix
from torch_pointcloud.utils.random import seed_everything

CUDA_AVAILABLE = torch.cuda.is_available()
CPU_COUNT = os.cpu_count()
DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"
NUM_WORKERS = CPU_COUNT // 2 if CPU_COUNT is not None else 0
BATCH_SIZE = 1
SEED = 42

VARIANT_TO_MODEL = {
    "scannet20": "oneformer3d-base.scannet20",
    "scannet200": "oneformer3d-base.scannet200",
}


def _per_voxel_superpoints(batch: torch.Tensor, inverse: torch.Tensor) -> torch.Tensor:
    """Fallback: treat each voxel as its own superpoint, with batch offsets.

    The voxel that each point maps into is given by `inverse`; we use that as
    the superpoint id and shift per-scene so ids are globally contiguous.
    """
    voxel_batch = batch[inverse]
    super_ids = inverse.clone()
    batch_size = int(voxel_batch.max().item()) + 1 if voxel_batch.numel() > 0 else 0
    running = 0
    for b in range(batch_size):
        mask = voxel_batch == b
        if mask.any():
            local = super_ids[mask]
            super_ids[mask] = local + running - int(local.min().item())
            running = int(super_ids[mask].max().item()) + 1
    return super_ids


@torch.inference_mode()
def forward_once(
    model: OneFormer3DSegmentation,
    feat: torch.Tensor,
    pos_grid: torch.Tensor,
    batch: torch.Tensor,
    superpoint: torch.Tensor,
    inverse: torch.Tensor,
    device: str,
) -> tuple[Dict[str, Any], float]:
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    start = time.perf_counter()
    out = model(feat, pos_grid, batch, superpoint, inverse)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    return out, (time.perf_counter() - start) * 1000.0


@torch.no_grad()
def evaluate(
    model: OneFormer3DSegmentation,
    dataloader: DataLoader,
    device: str,
    num_classes: int,
) -> Dict[str, Any]:
    model.to(device).eval()
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    total_latency_ms = 0.0
    total_points = 0

    pbar = tqdm(dataloader, total=len(dataloader), desc="Testing")
    for data in pbar:
        x = data[DataKeys.X].to(device)
        pos_grid = data[DataKeys.POS_GRID].to(device).long()
        batch = data[DataKeys.BATCH].to(device)
        inverse = data[DataKeys.INVERSE].to(device)
        target = data[DataKeys.SEGMENT].to(device)

        if "superpoint" in data:
            superpoint = data["superpoint"].to(device)
        else:
            superpoint = _per_voxel_superpoints(batch, inverse)

        out, latency_ms = forward_once(model, x, pos_grid, batch, superpoint, inverse, device)

        # `target` (the voxelized SEGMENT) is at voxel resolution after the model's
        # voxelization transform. With the per-voxel-superpoint fallback there is
        # one superpoint per voxel, so we read predictions directly from `sem_preds`.
        per_scene_preds = []
        for sem_pred_scene in out["sem_preds"]:
            per_scene_preds.append(sem_pred_scene[:, :num_classes].argmax(dim=1))
        preds = torch.cat(per_scene_preds, dim=0)

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
    parser = argparse.ArgumentParser(description="Benchmark OneFormer3D semantic segmentation on ScanNet.")
    parser.add_argument(
        "--variant",
        default="scannet20",
        choices=list(VARIANT_TO_MODEL.keys()),
        help="ScanNet variant to evaluate.",
    )
    parser.add_argument("--device", default=DEVICE)
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

    model_name = VARIANT_TO_MODEL[args.variant]
    print(f"Benchmarking model {model_name!r} on ScanNet (split={args.split!r})")

    model, model_info = create_model(model_name, task="segmentation", pretrained=True, return_info=True)
    assert isinstance(model, OneFormer3DSegmentation)
    num_classes = int(model.num_semantic_classes)
    transform = model_info.get("transforms")

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

    print(f"Test set: {len(dataset)} scenes")  # type: ignore[arg-type]
    metrics = evaluate(model, dataloader, args.device, num_classes)
    print("\nResults:")
    for key, value in metrics.items():
        print(f"  {key:<24} {value:.4f}")


if __name__ == "__main__":
    main()
