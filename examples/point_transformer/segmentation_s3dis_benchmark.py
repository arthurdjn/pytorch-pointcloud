"""Benchmark the Point Transformer S3DIS model on its held-out area, one pass over each whole room."""

import os
from argparse import ArgumentParser, Namespace
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from torch_pointcloud.config import DATA_DIR
from torch_pointcloud.datasets import S3DIS
from torch_pointcloud.datasets.s3dis import S3DISArea
from torch_pointcloud.inferers import Inferer, SimpleInferer
from torch_pointcloud.metrics import accuracy, confusion_matrix, intersection_over_union
from torch_pointcloud.models import SemanticSegmentationModel, create_model, model_info
from torch_pointcloud.utils.data import DataKeys, PointCloudDataLoader
from torch_pointcloud.utils.random import seed_everything, set_determinism

CUDA_AVAILABLE = torch.cuda.is_available()
CPU_COUNT = os.cpu_count()
DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"
NUM_WORKERS = CPU_COUNT // 2 if CPU_COUNT is not None else 0
SEED = 42
IGNORE_INDEX = -1
# The held-out area of every entry; the model trains on the five other areas.
MODELS: Dict[str, S3DISArea] = {
    "point-transformer.s3dis-area5.open3d": "Area_5",
}


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    set_determinism(tf32=False)

    print(f"Benchmarking model {args.model!r} on S3DIS {MODELS[args.model]}!")
    dataloader = configure_dataloader(args)
    model = configure_model(args)
    inferer = configure_inferer(args)

    num_rooms = len(dataloader.dataset)  # type: ignore[arg-type]
    print(f"Test set: {num_rooms} rooms  (whole rooms, scored at full resolution)")
    metrics = evaluate(model, dataloader, inferer, args.device, int(model.num_classes))
    print("\nResults:")
    for key, value in metrics.items():
        print(f"  {key:<24} {value:.4f}")


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Benchmark Point Transformer semantic segmentation on S3DIS.")
    parser.add_argument("--model", default="point-transformer.s3dis-area5.open3d", choices=list(MODELS))
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--root", default=DATA_DIR, help="Dataset root directory.")
    parser.add_argument("--seed", default=SEED, type=int)
    parser.add_argument("--num-workers", default=NUM_WORKERS, type=int)
    parser.add_argument("--limit", default=None, type=int, help="Evaluate at most this many rooms.")
    parser.add_argument("--download", action="store_true", help="Download S3DIS if missing.")
    parser.add_argument("--force-process", action="store_true", help="Force re-processing the dataset.")
    return parser.parse_args()


def configure_dataset(args: Namespace) -> Dataset:
    transform = model_info(args.model, task="semantic-segmentation")["transform"]

    dataset: Dataset
    dataset = S3DIS(
        root=args.root,
        areas=[MODELS[args.model]],
        transform=transform,
        download=args.download,
        force_process=args.force_process,
        num_workers=args.num_workers,
    )
    if args.limit is not None:
        n = min(int(args.limit), len(dataset))
        dataset = Subset(dataset, range(n))
        print(f"Evaluating on a subset of the first {n} rooms.")

    return dataset


def configure_dataloader(args: Namespace) -> DataLoader:
    dataset = configure_dataset(args)
    return PointCloudDataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)


def configure_model(args: Namespace) -> SemanticSegmentationModel:
    return create_model(args.model, task="semantic-segmentation", pretrained=True)


def configure_inferer(args: Namespace) -> Inferer:
    return SimpleInferer()


@torch.no_grad()
def evaluate(
    model: SemanticSegmentationModel, dataloader: DataLoader, inferer: Inferer, device: str, num_classes: int
) -> Dict[str, Any]:
    model.to(device).eval()
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)

    pbar = tqdm(dataloader, total=len(dataloader), desc="Testing")
    for data in pbar:
        data = {key: value.to(device) if torch.is_tensor(value) else value for key, value in data.items()}
        logits = inferer(data, predictor=lambda d: model(d[DataKeys.X], d[DataKeys.POS], d[DataKeys.BATCH]))
        preds = logits.argmax(dim=1)[data[DataKeys.INVERSE]]
        cm += confusion_matrix(preds.cpu(), data[DataKeys.ORIGIN_SEGMENT].cpu(), num_classes, ignore_index=IGNORE_INDEX)
        oa = accuracy(cm)
        pbar.set_postfix({"oa": f"{oa:.4f}"})

    return {
        "test/mIoU": intersection_over_union(cm),
        "test/oa": accuracy(cm),
    }


if __name__ == "__main__":
    main()
