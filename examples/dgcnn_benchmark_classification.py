"""Benchmark the DGCNN ModelNet40 classifiers (single pass, no voting).

Results (ModelNet40 overall accuracy):

    | Variant                      | reference | torch-pointcloud |
    | ---------------------------- | --------- | ---------------- |
    | dgcnn.modelnet40-1024.an-tao | 93.3      | 93.27            |
    | dgcnn.modelnet40-2048.an-tao | 93.6      | 93.60            |

Usage:
    uv run --no-sync python examples/dgcnn_benchmark_classification.py --model dgcnn.modelnet40-1024.an-tao
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
from torch_pointcloud.datasets import ModelNet40Hdf5
from torch_pointcloud.models import create_model
from torch_pointcloud.utils.data import DataKeys, PointCloudDataLoader
from torch_pointcloud.utils.metrics import confusion_matrix
from torch_pointcloud.utils.random import seed_everything, set_determinism

CUDA_AVAILABLE = torch.cuda.is_available()
CPU_COUNT = os.cpu_count()
DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"
NUM_WORKERS = CPU_COUNT // 2 if CPU_COUNT is not None else 0
BATCH_SIZE = 16
SEED = 42
NUM_POINTS = {
    "dgcnn.modelnet40-1024.an-tao": 1024,
    "dgcnn.modelnet40-2048.an-tao": 2048,
}


@torch.no_grad()
def evaluate(model: Module, dataloader: DataLoader, device: str, num_classes: int) -> Dict[str, Any]:
    model.to(device).eval()
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)

    pbar = tqdm(dataloader, total=len(dataloader), desc="Testing")
    for data in pbar:
        data = {key: value.to(device) if torch.is_tensor(value) else value for key, value in data.items()}
        logits = model(None, data[DataKeys.POS], data[DataKeys.BATCH])
        preds = logits.argmax(dim=1)
        cm += confusion_matrix(preds.cpu(), data[DataKeys.LABEL].cpu(), num_classes)
        oa = cm.diag().sum().float() / cm.sum().float().clamp_min(1)
        pbar.set_postfix({"oa": f"{oa.item():.4f}"})

    per_class_acc = cm.diag().float() / cm.sum(dim=1).float().clamp_min(1)
    return {
        "test/overall_acc": (cm.diag().sum().float() / cm.sum().float().clamp_min(1)).item(),
        "test/mean_class_acc": per_class_acc.mean().item(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark DGCNN classification on ModelNet40 (HDF5).")
    parser.add_argument("--model", default="dgcnn.modelnet40-1024.an-tao", choices=sorted(NUM_POINTS))
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--root", default=DATA_DIR, help="Dataset root directory.")
    parser.add_argument("--seed", default=SEED, type=int)
    parser.add_argument("--batch-size", default=BATCH_SIZE, type=int)
    parser.add_argument("--num-workers", default=NUM_WORKERS, type=int)
    parser.add_argument("--limit", default=None, type=int, help="Evaluate at most this many shapes.")
    parser.add_argument("--download", action="store_true", help="Download ModelNet40 if missing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    set_determinism(tf32=False)

    print(f"Benchmarking model {args.model!r} on ModelNet40!")
    model = create_model(args.model, task="classification", pretrained=True)
    num_classes = int(model.num_classes)

    transform = T.Slice(keys=[DataKeys.POS, DataKeys.NORMAL], stop=NUM_POINTS[args.model])
    dataset: Dataset = ModelNet40Hdf5(root=args.root, train=False, download=args.download, transform=transform)
    if args.limit is not None:
        n = min(int(args.limit), len(dataset))  # type: ignore[arg-type]
        dataset = Subset(dataset, range(n))
        print(f"Evaluating on a subset of the first {n} shapes.")

    dataloader = PointCloudDataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print(f"Test set: {len(dataset)} shapes")  # type: ignore[arg-type]
    metrics = evaluate(model, dataloader, args.device, num_classes)
    print("\nResults:")
    for key, value in metrics.items():
        print(f"  {key:<24} {value:.4f}")


if __name__ == "__main__":
    main()
