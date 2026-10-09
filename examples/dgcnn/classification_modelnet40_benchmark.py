"""Benchmark the DGCNN ModelNet40 classifiers (single pass, no voting)."""

import os
from argparse import ArgumentParser, Namespace
from typing import Any, Dict

import torch
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

import torch_pointcloud.transforms as T
from torch_pointcloud.config import DATA_DIR
from torch_pointcloud.datasets import ModelNet40Hdf5
from torch_pointcloud.metrics import accuracy, confusion_matrix
from torch_pointcloud.models import ClassificationModel, create_model
from torch_pointcloud.utils.data import DataKeys, PointCloudDataLoader
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


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    set_determinism(tf32=False)

    print(f"Benchmarking model {args.model!r} on ModelNet40!")
    model = configure_model(args)
    dataloader = configure_dataloader(args)

    num_shapes = len(dataloader.dataset)  # type: ignore[arg-type]
    print(f"Test set: {num_shapes} shapes")
    metrics = evaluate(model, dataloader, args.device, int(model.num_classes))
    print("\nResults:")
    for key, value in metrics.items():
        print(f"  {key:<24} {value:.4f}")


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Benchmark DGCNN classification on ModelNet40 (HDF5).")
    parser.add_argument("--model", default="dgcnn.modelnet40-1024.an-tao", choices=sorted(NUM_POINTS))
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--root", default=DATA_DIR, help="Dataset root directory.")
    parser.add_argument("--seed", default=SEED, type=int)
    parser.add_argument("--batch-size", default=BATCH_SIZE, type=int)
    parser.add_argument("--num-workers", default=NUM_WORKERS, type=int)
    parser.add_argument("--limit", default=None, type=int, help="Evaluate at most this many shapes.")
    parser.add_argument("--download", action="store_true", help="Download ModelNet40 if missing.")
    return parser.parse_args()


def configure_transform(args: Namespace) -> T.Transform:
    return T.Slice(keys=[DataKeys.POS, DataKeys.NORMAL], stop=NUM_POINTS[args.model])


def configure_dataset(args: Namespace) -> Dataset:
    dataset: Dataset
    transform = configure_transform(args)
    dataset = ModelNet40Hdf5(root=args.root, train=False, download=args.download, transform=transform)
    if args.limit is not None:
        n = min(int(args.limit), len(dataset))
        dataset = Subset(dataset, range(n))
        print(f"Evaluating on a subset of the first {n} shapes.")

    return dataset


def configure_dataloader(args: Namespace) -> DataLoader:
    dataset = configure_dataset(args)
    return PointCloudDataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)


def configure_model(args: Namespace) -> ClassificationModel:
    return create_model(args.model, task="classification", pretrained=True)


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
        oa = accuracy(cm)
        pbar.set_postfix({"oa": f"{oa:.4f}"})

    return {
        "test/overall_acc": accuracy(cm),
        "test/mean_class_acc": accuracy(cm, average="macro"),
    }


if __name__ == "__main__":
    main()
