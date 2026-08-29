"""Benchmark the PointGPT classifiers on ModelNet40 and ScanObjectNN (single pass, no voting).

NOTE: the ModelNet40 references are the paper's voted numbers; this script runs a single pass.

Results (overall accuracy):

    | Variant                                       | reference | torch-pointcloud |
    | --------------------------------------------- | --------- | ---------------- |
    | pointgpt-s.modelnet40.guangyan-chen           | 94.0      | 93.31            |
    | pointgpt-b.modelnet40.guangyan-chen           | 94.4      | 94.37            |
    | pointgpt-l.modelnet40.guangyan-chen           | 94.7      | 93.88            |
    | pointgpt-s.modelnet40-8k.guangyan-chen        | 94.2      | 93.76            |
    | pointgpt-b.modelnet40-8k.guangyan-chen        |           | 94.25            |
    | pointgpt-l.modelnet40-8k.guangyan-chen        |           | 93.92            |
    | pointgpt-s.scanobjectnn-objbg.guangyan-chen   | 91.6      | 91.57            |
    | pointgpt-b.scanobjectnn-objbg.guangyan-chen   | 95.8      | 97.07            |
    | pointgpt-l.scanobjectnn-objbg.guangyan-chen   | 97.2      | 98.45            |
    | pointgpt-s.scanobjectnn-objonly.guangyan-chen | 90.0      | 90.71            |
    | pointgpt-b.scanobjectnn-objonly.guangyan-chen | 95.2      | 95.18            |
    | pointgpt-l.scanobjectnn-objonly.guangyan-chen | 96.6      | 96.90            |
    | pointgpt-s.scanobjectnn-hardest.guangyan-chen | 86.9      | 86.95            |
    | pointgpt-b.scanobjectnn-hardest.guangyan-chen | 91.9      | 91.92            |
    | pointgpt-l.scanobjectnn-hardest.guangyan-chen | 93.4      | 93.75            |

Usage:
    uv run --no-sync python examples/pointgpt_benchmark_classification.py --model pointgpt-s.modelnet40.guangyan-chen
"""

import argparse
import os
from typing import Any, Dict, Union

import torch
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from torch_pointcloud.config import DATA_DIR
from torch_pointcloud.datasets import ModelNetNormalResampled, ScanObjectNN
from torch_pointcloud.models import create_model
from torch_pointcloud.utils.data import DataKeys, PointCloudDataLoader
from torch_pointcloud.utils.metrics import confusion_matrix
from torch_pointcloud.utils.random import seed_everything, set_determinism

CUDA_AVAILABLE = torch.cuda.is_available()
CPU_COUNT = os.cpu_count()
DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"
NUM_WORKERS = CPU_COUNT // 2 if CPU_COUNT is not None else 0
BATCH_SIZE = 32
SEED = 42

MODELS: Dict[str, Dict[str, Any]] = {
    "pointgpt-s.modelnet40.guangyan-chen": {"dataset": "modelnet40"},
    "pointgpt-b.modelnet40.guangyan-chen": {"dataset": "modelnet40"},
    "pointgpt-l.modelnet40.guangyan-chen": {"dataset": "modelnet40"},
    "pointgpt-s.modelnet40-8k.guangyan-chen": {"dataset": "modelnet40"},
    "pointgpt-b.modelnet40-8k.guangyan-chen": {"dataset": "modelnet40"},
    "pointgpt-l.modelnet40-8k.guangyan-chen": {"dataset": "modelnet40"},
    "pointgpt-s.scanobjectnn-objbg.guangyan-chen": {"dataset": "scanobjectnn", "background": True, "variant": None},
    "pointgpt-b.scanobjectnn-objbg.guangyan-chen": {"dataset": "scanobjectnn", "background": True, "variant": None},
    "pointgpt-l.scanobjectnn-objbg.guangyan-chen": {"dataset": "scanobjectnn", "background": True, "variant": None},
    "pointgpt-s.scanobjectnn-objonly.guangyan-chen": {"dataset": "scanobjectnn", "background": False, "variant": None},
    "pointgpt-b.scanobjectnn-objonly.guangyan-chen": {"dataset": "scanobjectnn", "background": False, "variant": None},
    "pointgpt-l.scanobjectnn-objonly.guangyan-chen": {"dataset": "scanobjectnn", "background": False, "variant": None},
    "pointgpt-s.scanobjectnn-hardest.guangyan-chen": {
        "dataset": "scanobjectnn",
        "background": True,
        "variant": "augmentedrot_scale75",
    },
    "pointgpt-b.scanobjectnn-hardest.guangyan-chen": {
        "dataset": "scanobjectnn",
        "background": True,
        "variant": "augmentedrot_scale75",
    },
    "pointgpt-l.scanobjectnn-hardest.guangyan-chen": {
        "dataset": "scanobjectnn",
        "background": True,
        "variant": "augmentedrot_scale75",
    },
}


def build_dataset(args: argparse.Namespace, transform: Any) -> Union[ModelNetNormalResampled, ScanObjectNN]:
    spec = MODELS[args.model]
    if spec["dataset"] == "modelnet40":
        return ModelNetNormalResampled(
            root=args.root,
            variant="40",
            train=False,
            transform=transform,
            download=args.download,
        )
    return ScanObjectNN(
        root=args.root,
        train=False,
        partition="main",
        background=spec["background"],
        variant=spec["variant"],
        transform=transform,
        download=args.download,
    )


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
    parser = argparse.ArgumentParser(description="Benchmark the PointGPT classifiers on ModelNet40 and ScanObjectNN.")
    parser.add_argument("--model", default="pointgpt-s.modelnet40.guangyan-chen", choices=sorted(MODELS))
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--root", default=DATA_DIR, help="Dataset root directory.")
    parser.add_argument("--seed", default=SEED, type=int)
    parser.add_argument("--batch-size", default=BATCH_SIZE, type=int)
    parser.add_argument("--num-workers", default=NUM_WORKERS, type=int)
    parser.add_argument("--limit", default=None, type=int, help="Evaluate at most this many shapes.")
    parser.add_argument("--download", action="store_true", help="Download the dataset if missing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    set_determinism(tf32=False)

    print(f"Benchmarking model {args.model!r} on {MODELS[args.model]['dataset']}!")
    model, model_info = create_model(args.model, task="classification", pretrained=True, return_info=True)

    full_dataset = build_dataset(args, model_info["transform"])
    # Some released heads are wider than the benchmark's label space; score the dataset's classes only.
    num_classes = len(full_dataset.classes)
    dataset: Dataset = full_dataset
    if args.limit is not None:
        n = min(int(args.limit), len(full_dataset))
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
