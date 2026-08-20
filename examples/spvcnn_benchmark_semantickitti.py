"""Benchmark SPVCNN on the 19-class SemanticKITTI val split.

Reference (full-resolution val, sequence 08, SPVNAS model zoo):
    | Model                                     | mIoU |
    | ----------------------------------------- | ---- |
    | spvcnn-119gmacs.semantickitti.mit-han-lab | 63.8 |
    | spvcnn-47gmacs.semantickitti.mit-han-lab  | 61.4 |
    | spvcnn-30gmacs.semantickitti.mit-han-lab  | 60.7 |

Usage:
    uv run --no-sync python examples/spvcnn_benchmark_semantickitti.py --limit 5
"""

import argparse
import os
import time
from typing import Any, Dict, Optional

import torch
import torchsparse
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from torch_pointcloud.config import DATA_DIR
from torch_pointcloud.datasets import SemanticKITTI
from torch_pointcloud.models import create_model
from torch_pointcloud.utils.data import DataKeys, collate
from torch_pointcloud.utils.metrics import confusion_matrix
from torch_pointcloud.utils.random import seed_everything, set_determinism

torchsparse.nn.functional.set_conv_mode(2)

# Set torchsparse configurations, see:
# https://github.com/mit-han-lab/torchsparse/issues/347#issuecomment-2920272471
torchsparse.nn.functional.set_kmap_mode("hashmap")
ts_config = torchsparse.nn.functional.conv_config.get_default_conv_config()
ts_config.kmap_mode = "hashmap"
torchsparse.nn.functional.conv_config.set_global_conv_config(ts_config)


CUDA_AVAILABLE = torch.cuda.is_available()
CPU_COUNT = os.cpu_count()
DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"
NUM_WORKERS = CPU_COUNT // 2 if CPU_COUNT is not None else 0
BATCH_SIZE = 1
SEED = 42
IGNORE_INDEX = 255


@torch.inference_mode()
def predict(
    model: torch.nn.Module,
    x: torch.Tensor,
    pos: torch.Tensor,
    batch: torch.Tensor,
    device: str,
) -> tuple[torch.Tensor, float]:
    if device.startswith("cuda"):
        torch.cuda.synchronize()

    start = time.perf_counter()
    logits = model(x, pos, batch)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - start) * 1000.0
    return logits, latency_ms


@torch.no_grad()
def evaluate(
    model: Module,
    dataloader: DataLoader,
    device: str,
    num_classes: int,
    max_iters: Optional[int] = None,
) -> Dict[str, Any]:
    model.to(device).eval()
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    total_latency_ms = 0.0
    total_points = 0
    n_iters = 0

    total = len(dataloader) if max_iters is None else min(max_iters, len(dataloader))
    pbar = tqdm(dataloader, total=total, desc="Testing")
    for data in pbar:
        x = data[DataKeys.X].to(device)
        pos = data[DataKeys.POS].to(device)
        batch = data[DataKeys.BATCH].to(device)
        target_full = data["origin_segment"].to(device)
        inverse_full = data[DataKeys.INVERSE].to(device)

        logits, latency_ms = predict(model, x, pos, batch, device)
        preds_sub = logits.argmax(dim=1)
        # Every raw point takes its enclosing voxel's prediction (full-resolution scoring).
        preds_full = preds_sub[inverse_full]

        cm += confusion_matrix(preds_full.cpu(), target_full.cpu(), num_classes, ignore_index=IGNORE_INDEX)
        total_latency_ms += latency_ms
        total_points += int(target_full.shape[0])
        oa = cm.diag().sum().float() / cm.sum().float().clamp_min(1)
        pbar.set_postfix({"oa": f"{oa.item():.4f}"})

        n_iters += 1
        if max_iters is not None and n_iters >= max_iters:
            pbar.close()
            break

    intersection = cm.diag().float()
    union = cm.sum(dim=1).float() + cm.sum(dim=0).float() - intersection
    iou_per_class = intersection / union.clamp_min(1e-10)
    return {
        "test/mIoU": iou_per_class.mean().item(),
        "test/oa": (cm.diag().sum().float() / cm.sum().float().clamp_min(1)).item(),
        "test/latency_ms": total_latency_ms / max(n_iters, 1),
        "test/points_per_second": total_points / max(total_latency_ms / 1000.0, 1e-12),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark SPVCNN semantic segmentation on SemanticKITTI.")
    parser.add_argument(
        "--model", default="spvcnn-119gmacs.semantickitti.mit-han-lab", help="Registered segmentation model name."
    )
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--root", default=DATA_DIR, help="Dataset root directory.")
    parser.add_argument("--split", default="val", choices=["train", "val", "trainval", "test"])
    parser.add_argument("--seed", default=SEED, type=int)
    parser.add_argument("--batch-size", default=BATCH_SIZE, type=int)
    parser.add_argument("--num-workers", default=NUM_WORKERS, type=int)
    parser.add_argument("--limit", default=None, type=int, help="Evaluate at most this many scans.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    set_determinism(tf32=False)

    print(f"Benchmarking model {args.model!r} on SemanticKITTI (split={args.split!r})!")
    model, model_info = create_model(args.model, task="segmentation", pretrained=True, return_info=True)
    num_classes = int(model.num_classes)
    transform = model_info.get("transform")

    dataset: Dataset = SemanticKITTI(
        root=args.root,
        split=args.split,
        transform=transform,
    )
    if args.limit is not None:
        n = min(int(args.limit), len(dataset))  # type: ignore[arg-type]
        dataset = Subset(dataset, range(n))
        print(f"Evaluating on a subset of the first {n} scans.")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
    )

    print(f"Test set: {len(dataset)} scans")  # type: ignore[arg-type]
    metrics = evaluate(model, dataloader, args.device, num_classes)
    print("\nResults:")
    for key, value in metrics.items():
        print(f"  {key:<24} {value:.4f}")


if __name__ == "__main__":
    main()
