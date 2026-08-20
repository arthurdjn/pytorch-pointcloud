"""Benchmark SpUNet on ScanNet: whole-scene forward vs. grid sliding-window.

NOTE: a non-overlapping sliding window exists for memory, not accuracy: grid-SW should land at parity with
the whole-scene baseline (a small dip is normal; a large divergence means the tiling/scatter-back is buggy).

Usage:
    uv run --no-sync python examples/spunet_benchmark_scannet_sw.py --limit 5
"""

import argparse
import os
import time
from typing import Any, Callable, Dict, Optional

import torch
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from torch_pointcloud.config import DATA_DIR
from torch_pointcloud.datasets import ScanNet20
from torch_pointcloud.inferers import SlidingWindowInferer
from torch_pointcloud.models import create_model
from torch_pointcloud.utils.data import DataKeys, collate
from torch_pointcloud.utils.metrics import confusion_matrix
from torch_pointcloud.utils.random import seed_everything, set_determinism

CUDA_AVAILABLE = torch.cuda.is_available()
CPU_COUNT = os.cpu_count()
DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"
NUM_WORKERS = CPU_COUNT // 2 if CPU_COUNT is not None else 0
SEED = 42


def make_predictor(model: Module) -> Callable[[Dict[str, Any]], torch.Tensor]:
    @torch.no_grad()
    def predictor(window: Dict[str, Any]) -> torch.Tensor:
        return model(window[DataKeys.X], window[DataKeys.POS], window[DataKeys.BATCH])

    return predictor


@torch.no_grad()
def evaluate(
    model: Module,
    dataloader: DataLoader,
    device: str,
    num_classes: int,
    inferer: Optional[SlidingWindowInferer],
) -> Dict[str, float]:
    """Score one full pass. `inferer=None` -> whole-scene forward; else grid-SW.

    Both paths produce voxel-resolution predictions, then broadcast to raw
    points via the CLUSTER inverse map and score against `origin_segment`, so
    the two numbers are directly comparable.
    """
    model.to(device).eval()
    predictor = make_predictor(model)
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    total_latency_ms = 0.0

    mode = "whole-scene" if inferer is None else "grid-sw"
    pbar = tqdm(dataloader, total=len(dataloader), desc=f"Testing ({mode})")
    for data in pbar:
        data = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in data.items()}

        if device.startswith("cuda"):
            torch.cuda.synchronize()
        start = time.perf_counter()
        if inferer is None:
            voxel_pred = model(data[DataKeys.X], data[DataKeys.POS], data[DataKeys.BATCH])
        else:
            voxel_pred = inferer(data, predictor=predictor)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        total_latency_ms += (time.perf_counter() - start) * 1000.0

        inverse = data[DataKeys.INVERSE]
        target = data["origin_segment"]
        preds = voxel_pred[inverse].argmax(dim=1)
        cm += confusion_matrix(preds.cpu(), target.cpu(), num_classes, ignore_index=-1)
        oa = cm.diag().sum().float() / cm.sum().float().clamp_min(1)
        pbar.set_postfix({"oa": f"{oa.item():.4f}"})

    intersection = cm.diag().float()
    union = cm.sum(dim=1).float() + cm.sum(dim=0).float() - intersection
    iou = intersection / union.clamp_min(1e-10)
    return {
        "mIoU": iou.mean().item(),
        "oA": (cm.diag().sum().float() / cm.sum().float().clamp_min(1)).item(),
        "latency_ms": total_latency_ms / max(len(dataloader), 1),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SpUNet ScanNet: whole-scene vs grid sliding-window.")
    parser.add_argument(
        "--model", default="spunet-v1m1.scannet20.pointcept", help="Registered segmentation model name."
    )
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--root", default=DATA_DIR, help="Dataset root directory.")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--seed", default=SEED, type=int)
    parser.add_argument("--num-workers", default=NUM_WORKERS, type=int)
    parser.add_argument("--limit", default=None, type=int, help="Evaluate at most this many scenes.")
    parser.add_argument(
        "--block-size",
        default=200,
        type=int,
        help="Grid block side in VOXEL units (eval transform voxelizes at 0.02 m, so 200 ~= 4 m).",
    )
    parser.add_argument(
        "--roi-num-points",
        default=100_000,
        type=int,
        help="Cap points per predictor call; oversized blocks split into sub-fragments (no point loss).",
    )
    parser.add_argument("--download", action="store_true", help="Download ScanNet if missing.")
    parser.add_argument("--force-process", action="store_true", help="Force re-processing the dataset.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    set_determinism(tf32=False)

    print(f"Benchmarking {args.model!r} on ScanNet (split={args.split!r}): whole-scene vs grid-SW")
    model, model_info = create_model(args.model, task="segmentation", pretrained=True, return_info=True)
    num_classes = int(model.num_classes)
    transform = model_info.get("transform")

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

    def new_loader() -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate,
        )

    print(f"Test set: {len(dataset)} scenes  (eval @ full point resolution)")  # type: ignore[arg-type]

    base = evaluate(model, new_loader(), args.device, num_classes, inferer=None)
    sw_inferer = SlidingWindowInferer(
        block_size=args.block_size,
        roi_num_points=args.roi_num_points,
        softmax=True,
    )
    sw = evaluate(model, new_loader(), args.device, num_classes, inferer=sw_inferer)

    print("\nResults (no TTA):")
    print(f"  {'metric':<14}{'whole-scene':>14}{'grid-sw':>14}{'delta':>12}")
    for key in ("mIoU", "oA", "latency_ms"):
        b, s = base[key], sw[key]
        print(f"  {key:<14}{b:>14.4f}{s:>14.4f}{s - b:>12.4f}")
    print(
        "\nNo-TTA expectation: the mIoU gap is inherent block-seam context loss, not a bug. "
        "It scales inversely with --block-size (large blocks -> 0, exact once one block "
        "covers the scene) and is recovered by wrapping the inferer in TTAInferer "
        "(multi-offset voting)."
    )


if __name__ == "__main__":
    main()
