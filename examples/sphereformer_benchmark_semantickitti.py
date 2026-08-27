"""Benchmark SphereFormer on the 19-class SemanticKITTI val split.

NOTE: the released weights are no longer downloadable (dvlab issue #78), so no measured number is reported;
the ported architecture is bit-exact to the reference sptr build (2.4e-7).

Results vs reference:

    | Source               | SemanticKITTI val mIoU |
    | -------------------- | ---------------------- |
    | SphereFormer (paper) | 67.8                   |

Usage:
    CUDA_VISIBLE_DEVICES=0 uv run --no-sync python examples/sphereformer_benchmark_semantickitti.py --limit 5
"""

import argparse
import os
import time
from typing import Any, Dict, Optional

import torch
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from torch_pointcloud.config import DATA_DIR
from torch_pointcloud.datasets import SemanticKITTI
from torch_pointcloud.models import create_model
from torch_pointcloud.utils.data import DataKeys, collate
from torch_pointcloud.utils.metrics import confusion_matrix
from torch_pointcloud.utils.random import seed_everything

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

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
    pos_grid: torch.Tensor,
    batch: torch.Tensor,
    pos: torch.Tensor,
    device: str,
) -> tuple[torch.Tensor, float]:
    if device.startswith("cuda"):
        torch.cuda.synchronize()

    start = time.perf_counter()
    logits = model(x, pos, pos_grid, batch)
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
        data = {k: v.to(device) if torch.is_tensor(v) else v for k, v in data.items()}
        # The dataset transform voxelized `pos`/`pos_grid`/`x`/`segment` and kept the source labels under
        # `origin_segment` with the source-to-voxel inverse map, for back-projected scoring.
        target_full = data[DataKeys.ORIGIN_SEGMENT]
        inverse_full = data[DataKeys.INVERSE]
        n_full = int(target_full.shape[0])

        logits_sub, latency_ms = predict(
            model,
            data[DataKeys.X],
            data[DataKeys.POS_GRID],
            data[DataKeys.BATCH],
            data[DataKeys.POS],
            device,
        )
        preds_sub = logits_sub.argmax(dim=1)
        preds_full = preds_sub[inverse_full]

        cm += confusion_matrix(preds_full.cpu(), target_full.cpu(), num_classes, ignore_index=IGNORE_INDEX)
        total_latency_ms += latency_ms
        total_points += n_full
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
    parser = argparse.ArgumentParser(description="Benchmark SphereFormer semantic segmentation on SemanticKITTI.")
    parser.add_argument("--model", default="sphereformer.semantickitti", help="Registered segmentation model name.")
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

    print(f"Benchmarking model {args.model!r} on SemanticKITTI (split={args.split!r})!")
    model, model_info = create_model(args.model, task="segmentation", pretrained=True, return_info=True)
    num_classes = int(model.num_classes)
    transform = model_info.get("transform")

    dataset: Dataset = SemanticKITTI(root=args.root, split=args.split, transform=transform)
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
