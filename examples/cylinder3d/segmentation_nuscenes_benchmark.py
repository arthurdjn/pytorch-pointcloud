"""Benchmark Cylinder3D semantic segmentation on nuScenes-lidarseg in a single pass, scored on every point."""

import os
from argparse import ArgumentParser, Namespace
from typing import Any, Dict

import torch
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from torch_pointcloud.config import DATA_DIR
from torch_pointcloud.datasets import NuScenes, NuScenesMini
from torch_pointcloud.inferers import Inferer, SimpleInferer
from torch_pointcloud.metrics import accuracy, confusion_matrix, intersection_over_union
from torch_pointcloud.models import create_model, model_info
from torch_pointcloud.utils.data import DataKeys, PointCloudDataLoader
from torch_pointcloud.utils.random import seed_everything, set_determinism

CUDA_AVAILABLE = torch.cuda.is_available()
CPU_COUNT = os.cpu_count()
DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"
NUM_WORKERS = CPU_COUNT // 2 if CPU_COUNT is not None else 0
SEED = 42
MODELS = ("cylinder3d-0p5x.nuscenes.yuenan-hou",)
NUM_CLASSES = 16


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    set_determinism(tf32=False)

    print(f"Benchmarking model {args.model!r} on nuScenes-lidarseg!")
    dataloader = configure_dataloader(args)
    model = configure_model(args)
    inferer = configure_inferer(args)

    num_scans = len(dataloader.dataset)  # type: ignore[arg-type]
    print(f"Test set: {num_scans} scans  (single pass, scored on every point)")
    metrics = evaluate(model, dataloader, inferer, args.device, NUM_CLASSES)
    print("\nResults:")
    for key, value in metrics.items():
        print(f"  {key:<24} {value:.4f}")


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Benchmark Cylinder3D semantic segmentation on nuScenes-lidarseg.")
    parser.add_argument("--model", default=MODELS[0], choices=MODELS)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--root", default=DATA_DIR, help="Dataset root directory.")
    parser.add_argument("--split", default="val", choices=("val", "mini"), help="nuScenes `val`, or the mini release.")
    parser.add_argument("--seed", default=SEED, type=int)
    parser.add_argument("--num-workers", default=NUM_WORKERS, type=int)
    parser.add_argument("--limit", default=None, type=int, help="Evaluate at most this many scans.")
    return parser.parse_args()


def configure_dataset(args: Namespace) -> Dataset:
    transform = model_info(args.model, task="semantic-segmentation")["transform"]
    dataset: Dataset
    if args.split == "mini":
        dataset = NuScenesMini(
            root=args.root,
            max_sweeps=1,
            return_boxes=False,
            return_segment=True,
            transform=transform,
        )
    else:
        dataset = NuScenes(
            root=args.root,
            split="val",
            max_sweeps=1,
            return_boxes=False,
            return_segment=True,
            transform=transform,
        )

    if args.limit is not None:
        n = min(int(args.limit), len(dataset))
        dataset = Subset(dataset, range(n))
        print(f"Evaluating on a subset of the first {n} scans.")
    return dataset


def configure_dataloader(args: Namespace) -> DataLoader:
    dataset = configure_dataset(args)
    return PointCloudDataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)


def configure_model(args: Namespace) -> Module:
    return create_model(args.model, task="semantic-segmentation", pretrained=True)


def configure_inferer(args: Namespace) -> Inferer:
    return SimpleInferer()


@torch.no_grad()
def evaluate(model: Module, dataloader: DataLoader, inferer: Inferer, device: str, num_classes: int) -> Dict[str, Any]:
    model.to(device).eval()
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)

    pbar = tqdm(dataloader, total=len(dataloader), desc="Testing")
    for data in pbar:
        data = {key: value.to(device) if torch.is_tensor(value) else value for key, value in data.items()}
        scores = inferer(data, predictor=lambda d: model(d[DataKeys.X], d[DataKeys.POS], d[DataKeys.BATCH]))
        preds = scores.argmax(dim=1)
        cm += confusion_matrix(preds.cpu(), data[DataKeys.SEGMENT].cpu(), num_classes, ignore_index=-1)
        oa = accuracy(cm)
        pbar.set_postfix({"oa": f"{oa:.4f}"})

    return {
        "test/mIoU": intersection_over_union(cm),
        "test/oa": accuracy(cm),
    }


if __name__ == "__main__":
    main()
