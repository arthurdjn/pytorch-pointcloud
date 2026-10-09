"""Benchmark the Point-MAE part segmentation model on ShapeNetPart, single pass without voting."""

import os
from argparse import ArgumentParser, Namespace
from typing import Any, Dict, List

import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from torch_pointcloud.config import DATA_DIR
from torch_pointcloud.datasets import ShapeNetPart
from torch_pointcloud.inferers import Inferer, SimpleInferer
from torch_pointcloud.metrics import part_intersection_over_union, part_mean_intersection_over_union
from torch_pointcloud.models import create_model, model_info
from torch_pointcloud.utils.data import DataKeys, PointCloudDataLoader
from torch_pointcloud.utils.random import seed_everything, set_determinism

CUDA_AVAILABLE = torch.cuda.is_available()
CPU_COUNT = os.cpu_count()
DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"
NUM_WORKERS = CPU_COUNT // 2 if CPU_COUNT is not None else 0
BATCH_SIZE = 16
SEED = 42


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    set_determinism(tf32=False)

    print(f"Benchmarking model {args.model!r} on ShapeNetPart!")
    dataloader = configure_dataloader(args)
    model = configure_model(args)
    inferer = configure_inferer(args)

    print(f"Test set: {len(dataloader.dataset)} shapes")  # type: ignore[arg-type]
    metrics = evaluate(model, dataloader, inferer, args.device)
    print("\nResults:")
    for key, value in metrics.items():
        print(f"  {key:<24} {value:.4f}")


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Benchmark Point-MAE part segmentation on ShapeNetPart.")
    parser.add_argument(
        "--model", default="point-mae-base.shapenetpart.yatian-pang", help="Registered segmentation model name"
    )
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--root", default=DATA_DIR, help="Dataset root directory.")
    parser.add_argument("--seed", default=SEED, type=int)
    parser.add_argument("--batch-size", default=BATCH_SIZE, type=int)
    parser.add_argument("--num-workers", default=NUM_WORKERS, type=int)
    parser.add_argument("--limit", default=None, type=int, help="Evaluate at most this many shapes.")
    return parser.parse_args()


def configure_dataset(args: Namespace) -> Dataset:
    transform = model_info(args.model, task="part-segmentation")["transform"]
    dataset: Dataset
    dataset = ShapeNetPart(root=args.root, split="test", transform=transform)
    if args.limit is not None:
        n = min(int(args.limit), len(dataset))
        dataset = Subset(dataset, range(n))
        print(f"Evaluating on a subset of the first {n} shapes.")
    return dataset


def configure_dataloader(args: Namespace) -> DataLoader:
    dataset = configure_dataset(args)
    return PointCloudDataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)


def configure_model(args: Namespace) -> Module:
    return create_model(args.model, task="part-segmentation", pretrained=True)


def configure_inferer(args: Namespace) -> Inferer:
    return SimpleInferer()


@torch.no_grad()
def evaluate(model: Module, dataloader: DataLoader, inferer: Inferer, device: str) -> Dict[str, Any]:
    model.to(device).eval()
    part_ids = list(ShapeNetPart.seg_ids.values())
    ious: List[Tensor] = []
    categories: List[Tensor] = []

    for data in tqdm(dataloader, total=len(dataloader), desc="Testing"):
        data = {key: value.to(device) if torch.is_tensor(value) else value for key, value in data.items()}
        scores = inferer(
            data, predictor=lambda d: model(None, d[DataKeys.POS], d[DataKeys.BATCH], d[DataKeys.CATEGORY])
        )
        preds = scores.argmax(dim=1).cpu()
        category = data[DataKeys.CATEGORY].argmax(dim=1).cpu()
        shape_ious = part_intersection_over_union(
            preds, data[DataKeys.SEGMENT].cpu(), part_ids, category, data[DataKeys.BATCH].cpu()
        )
        ious.append(shape_ious)
        categories.append(category)

    shape_ious, category = torch.cat(ious), torch.cat(categories)
    return {
        "test/ins_mIoU": part_mean_intersection_over_union(shape_ious, category),
        "test/cls_mIoU": part_mean_intersection_over_union(shape_ious, category, average="macro"),
    }


if __name__ == "__main__":
    main()
