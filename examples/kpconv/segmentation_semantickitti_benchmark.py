"""Benchmark the KP-FCNN SemanticKITTI model on sequence 08 with Open3D-ML's sphere-voting protocol."""

import os
from argparse import ArgumentParser, Namespace
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from torch_pointcloud.config import DATA_DIR
from torch_pointcloud.datasets import SemanticKITTI
from torch_pointcloud.inferers import Inferer, PotentialSphereInferer
from torch_pointcloud.metrics import accuracy, confusion_matrix, intersection_over_union
from torch_pointcloud.models import SemanticSegmentationModel, create_model, model_info
from torch_pointcloud.utils.data import DataKeys, PointCloudDataLoader
from torch_pointcloud.utils.random import seed_everything, set_determinism

CUDA_AVAILABLE = torch.cuda.is_available()
CPU_COUNT = os.cpu_count()
DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"
NUM_WORKERS = CPU_COUNT // 2 if CPU_COUNT is not None else 0
SEED = 42
SPHERE_RADIUS = 4.0
NUM_VOTES = 3.0
SW_BATCH_SIZE = 8
IGNORE_INDEX = -1


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    set_determinism(tf32=False)

    print(f"Benchmarking model {args.model!r} on SemanticKITTI (split={args.split!r})!")
    dataloader = configure_dataloader(args)
    model = configure_model(args)
    inferer = configure_inferer(args)

    num_scans = len(dataloader.dataset)  # type: ignore[arg-type]
    print(f"Test set: {num_scans} scans  (sphere voting, radius {SPHERE_RADIUS}, scored at full resolution)")
    metrics = evaluate(model, dataloader, inferer, args.device, int(model.num_classes))
    print("\nResults:")
    for key, value in metrics.items():
        print(f"  {key:<24} {value:.4f}")


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Benchmark KP-FCNN semantic segmentation on SemanticKITTI.")
    parser.add_argument("--model", default="kpfcnn.semantickitti.open3d", help="Registered segmentation model name")
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--root", default=DATA_DIR, help="Dataset root directory.")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--seed", default=SEED, type=int)
    parser.add_argument("--sw-batch-size", default=SW_BATCH_SIZE, type=int, help="Spheres per forward.")
    parser.add_argument("--num-workers", default=NUM_WORKERS, type=int)
    parser.add_argument("--limit", default=None, type=int, help="Evaluate at most this many scans.")
    return parser.parse_args()


def configure_dataset(args: Namespace) -> Dataset:
    transform = model_info(args.model, task="semantic-segmentation")["transform"]

    dataset: Dataset
    dataset = SemanticKITTI(root=args.root, split=args.split, transform=transform)
    if args.limit is not None:
        n = min(int(args.limit), len(dataset))
        dataset = Subset(dataset, range(n))
        print(f"Evaluating on a subset of the first {n} scans.")

    return dataset


def configure_dataloader(args: Namespace) -> DataLoader:
    dataset = configure_dataset(args)
    return PointCloudDataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)


def configure_model(args: Namespace) -> SemanticSegmentationModel:
    return create_model(args.model, task="semantic-segmentation", pretrained=True)


def configure_inferer(args: Namespace) -> Inferer:
    return PotentialSphereInferer(
        radius=SPHERE_RADIUS,
        num_votes=NUM_VOTES,
        inner_ratio=1.0,
        ema_smoothing=0.95,
        sw_batch_size=args.sw_batch_size,
        seed=args.seed,
    )


@torch.no_grad()
def evaluate(
    model: SemanticSegmentationModel, dataloader: DataLoader, inferer: Inferer, device: str, num_classes: int
) -> Dict[str, Any]:
    model.to(device).eval()
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)

    pbar = tqdm(dataloader, total=len(dataloader), desc="Testing")
    for data in pbar:
        data = {key: value.to(device) if torch.is_tensor(value) else value for key, value in data.items()}
        probs = inferer(data, predictor=lambda d: model(d[DataKeys.X], d[DataKeys.POS], d[DataKeys.BATCH]))
        preds = probs.argmax(dim=1)[data[DataKeys.INVERSE]]
        cm += confusion_matrix(preds.cpu(), data[DataKeys.ORIGIN_SEGMENT].cpu(), num_classes, ignore_index=IGNORE_INDEX)
        oa = accuracy(cm)
        pbar.set_postfix({"oa": f"{oa:.4f}"})

    return {
        "test/mIoU": intersection_over_union(cm),
        "test/oa": accuracy(cm),
    }


if __name__ == "__main__":
    main()
