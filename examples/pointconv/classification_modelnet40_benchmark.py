"""Benchmark the PointConv classifier on ModelNet40 (single pass, no voting)."""

import os
from argparse import ArgumentParser, Namespace
from typing import Any, Dict

import torch
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from torch_pointcloud.config import DATA_DIR
from torch_pointcloud.datasets import ModelNetNormalResampled
from torch_pointcloud.metrics import accuracy, confusion_matrix
from torch_pointcloud.models import create_model, model_info
from torch_pointcloud.utils.data import DataKeys, PointCloudDataLoader
from torch_pointcloud.utils.random import seed_everything, set_determinism

CUDA_AVAILABLE = torch.cuda.is_available()
CPU_COUNT = os.cpu_count()
DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"
NUM_WORKERS = CPU_COUNT // 2 if CPU_COUNT is not None else 0
BATCH_SIZE = 32
SEED = 42

MODELS = ("pointconv-density-base.modelnet40.wenxuan-wu",)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    set_determinism(tf32=False)

    print(f"Benchmarking model {args.model!r} on ModelNet40!")
    model = configure_model(args)
    dataloader = configure_dataloader(args)

    print(f"Test set: {len(dataloader.dataset)} shapes")  # type: ignore[arg-type]
    metrics = evaluate(model, dataloader, args.device, num_classes(dataloader.dataset))
    print("\nResults:")
    for key, value in metrics.items():
        print(f"  {key:<24} {value:.4f}")


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Benchmark the PointConv classifier on ModelNet40.")
    parser.add_argument("--model", default="pointconv-density-base.modelnet40.wenxuan-wu", choices=sorted(MODELS))
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--root", default=DATA_DIR, help="Dataset root directory.")
    parser.add_argument("--seed", default=SEED, type=int)
    parser.add_argument("--batch-size", default=BATCH_SIZE, type=int)
    parser.add_argument("--num-workers", default=NUM_WORKERS, type=int)
    parser.add_argument("--limit", default=None, type=int, help="Evaluate at most this many shapes.")
    parser.add_argument("--download", action="store_true", help="Download the dataset if missing.")
    return parser.parse_args()


def configure_dataset(args: Namespace) -> Dataset:
    entry = model_info(args.model)

    dataset: Dataset
    dataset = ModelNetNormalResampled(
        root=args.root,
        variant="40",
        train=False,
        transform=entry["transform"],
        download=args.download,
    )

    if args.limit is not None:
        n = min(int(args.limit), len(dataset))
        dataset = Subset(dataset, range(n))
        print(f"Evaluating on a subset of the first {n} shapes.")

    return dataset


def configure_dataloader(args: Namespace) -> DataLoader:
    dataset = configure_dataset(args)
    return PointCloudDataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)


def configure_model(args: Namespace) -> Module:
    return create_model(args.model, task="classification", pretrained=True)


def num_classes(dataset: Dataset) -> int:
    """Some released heads are wider than the benchmark's label space: score the dataset's classes only."""
    while isinstance(dataset, Subset):
        dataset = dataset.dataset
    return len(dataset.classes)  # type: ignore[attr-defined]


@torch.no_grad()
def evaluate(model: Module, dataloader: DataLoader, device: str, num_classes: int) -> Dict[str, Any]:
    model.to(device).eval()
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)

    pbar = tqdm(dataloader, total=len(dataloader), desc="Testing")
    for data in pbar:
        data = {key: value.to(device) if torch.is_tensor(value) else value for key, value in data.items()}
        logits = model(data[DataKeys.NORMAL], data[DataKeys.POS], data[DataKeys.BATCH])
        preds = logits.argmax(dim=1)
        cm += confusion_matrix(preds.cpu(), data[DataKeys.LABEL].cpu(), num_classes)
        oa = accuracy(cm)
        pbar.set_postfix({"oa": f"{oa:.4f}"})

    return {
        "test/overall_acc": accuracy(cm),
        "test/mean_class_acc": accuracy(cm, average="macro"),
    }


if __name__ == "__main__":
    main()
