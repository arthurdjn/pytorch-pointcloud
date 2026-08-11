"""Evaluate the PointMLP ModelNet40 classifiers (single-pass, no voting).

`ModelNet40Hdf5` -> `DataLoader` -> model -> argmax -> overall accuracy, following the reference
protocol: take the first 1024 points of each pre-shuffled 2048-point HDF5 cloud (no FPS).

Results vs reference (ModelNet40 overall accuracy; reference is the paper's best-seed number without
voting, with 94.5 / 94.0 reported with voting):

    | Variant                         | reference | torch-pointcloud            |
    | ------------------------------- | --------- | --------------------------- |
    | pointmlp-base.modelnet40.xu-ma  | 94.1      | 93.52 (pre-fix, re-measure) |
    | pointmlp-elite.modelnet40.xu-ma | 93.6      | 92.46 (pre-fix, re-measure) |

The torch-pointcloud numbers were measured before this script switched to the reference protocol
(FPS on the normal-resampled dataset); re-measurement on the HDF5 first-1024 protocol is pending.

Usage:
    uv run --no-sync python examples/pointmlp_benchmark_modelnet.py --download
    uv run --no-sync python examples/pointmlp_benchmark_modelnet.py --model pointmlp-elite.modelnet40.xu-ma
"""

import os
from argparse import ArgumentParser, Namespace
from typing import Dict

import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm

import torch_pointcloud.transforms as T
from torch_pointcloud.config import DATA_DIR
from torch_pointcloud.datasets import ModelNet40Hdf5
from torch_pointcloud.models import create_model
from torch_pointcloud.utils.data import DataKeys, collate
from torch_pointcloud.utils.metrics import confusion_matrix
from torch_pointcloud.utils.random import seed_everything, set_determinism

CUDA_AVAILABLE = torch.cuda.is_available()
CPU_COUNT = os.cpu_count()
DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"
NUM_WORKERS = CPU_COUNT // 2 if CPU_COUNT is not None else 0
BATCH_SIZE = 16
SEED = 42
NUM_POINTS = 1024

MODEL_CHOICES = [
    "pointmlp-base.modelnet40.xu-ma",
    "pointmlp-elite.modelnet40.xu-ma",
]


def main() -> None:
    args = parse_args()

    print(f"Seeding everything to {args.seed}!")
    seed_everything(args.seed)
    set_determinism(tf32=False)

    print(f"Loading model {args.model!r}!")
    model = create_model(
        args.model,
        task="classification",
        pretrained=True,
    )

    num_classes: int = int(model.num_classes)
    transform = T.Slice(keys=[DataKeys.POS, DataKeys.NORMAL], stop=NUM_POINTS)

    print("Loading ModelNet40 (HDF5) test dataset!")
    test_dataset = ModelNet40Hdf5(
        root=args.root,
        split="test",
        download=args.download,
        transform=transform,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
    )

    print(f"Test set: {len(test_dataset)} samples")
    print("Evaluating...")
    metrics = evaluate(model, test_dataloader, args.device, num_classes=num_classes)

    print("\nResults:")
    for k, v in metrics.items():
        print(f"  {k:<20} {v:.4f}")


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Benchmark PointMLP classification on ModelNet40 (HDF5).")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--root", type=str, default=DATA_DIR, help="Dataset root directory.")
    parser.add_argument(
        "--model",
        type=str,
        default="pointmlp-base.modelnet40.xu-ma",
        choices=MODEL_CHOICES,
    )
    parser.add_argument("--download", action="store_true", help="Download ModelNet if missing.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--device", type=str, default=DEVICE)
    return parser.parse_args()


@torch.no_grad()
def evaluate(model: Module, dataloader: DataLoader, device: str, *, num_classes: int) -> Dict[str, float]:
    model.to(device).eval()
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)

    pbar = tqdm(dataloader, total=len(dataloader), desc="Testing")
    for data in pbar:
        pos = data[DataKeys.POS].to(device)
        label = data[DataKeys.LABEL].to(device)
        batch = data[DataKeys.BATCH].to(device)

        # PointMLP uses in_channels=3 (XYZ-as-features); pass `None` to let the
        # model fall back to using positions as features.
        logits = model(None, pos, batch)
        preds = logits.argmax(dim=1)

        cm += confusion_matrix(preds.cpu(), label.cpu(), num_classes)
        oa = cm.diag().sum().float() / cm.sum().float().clamp_min(1)
        pbar.set_postfix({"oa": f"{oa.item():.4f}"})

    oa = cm.diag().sum().float() / cm.sum().float()
    per_class_acc = cm.diag().float() / cm.sum(dim=1).float().clamp_min(1)
    mean_class_acc = per_class_acc.mean()

    return {
        "test/overall_acc": oa.item(),
        "test/mean_class_acc": mean_class_acc.item(),
    }


if __name__ == "__main__":
    main()
