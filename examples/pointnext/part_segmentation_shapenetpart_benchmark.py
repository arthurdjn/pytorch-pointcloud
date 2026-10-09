"""Benchmark the PointNeXt ShapeNetPart part-segmentation models with the voting and refinement test protocol."""

import os
from argparse import ArgumentParser, Namespace
from typing import Any, Dict, List

import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

import torch_pointcloud.transforms as T
from torch_pointcloud.config import DATA_DIR
from torch_pointcloud.datasets import ShapeNetPart
from torch_pointcloud.inferers import Inferer, PartRefinementInferer, SimpleInferer, TTAInferer
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
NUM_VOTES = 10

INFERER_TRANSFORM = T.Compose(
    [
        T.RandomScale(keys=DataKeys.POS, scale_range=(0.8, 1.2), anisotropic=True, p=1.0),
        T.Cat(keys=[DataKeys.POS, DataKeys.NORMAL, "height"], dst_key=DataKeys.X),
    ]
)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    set_determinism(tf32=False)

    print(f"Benchmarking model {args.model!r} on ShapeNetPart!")
    model = configure_model(args)
    dataloader = configure_dataloader(args)
    inferer = configure_inferer(args)

    num_shapes = len(dataloader.dataset)  # type: ignore[arg-type]
    print(f"Test set: {num_shapes} shapes  ({NUM_VOTES} scale votes + identity, nearest-neighbor part refinement)")
    metrics = evaluate(model, dataloader, inferer, args.device)
    print("\nResults:")
    for key, value in metrics.items():
        print(f"  {key:<24} {value:.4f}")


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Benchmark PointNeXt part segmentation on ShapeNetPart.")
    parser.add_argument(
        "--model", default="pointnext-sm.shapenetpart.openpoints", help="Registered segmentation model name"
    )
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--root", default=DATA_DIR, help="Dataset root directory.")
    parser.add_argument("--seed", default=SEED, type=int)
    parser.add_argument("--batch-size", default=BATCH_SIZE, type=int)
    parser.add_argument("--num-workers", default=NUM_WORKERS, type=int)
    parser.add_argument("--limit", default=None, type=int, help="Evaluate at most this many shapes.")
    return parser.parse_args()


def configure_dataset(args: Namespace) -> Dataset:
    dataset: Dataset = ShapeNetPart(root=args.root, split="test", transform=model_info(args.model)["transform"])
    if args.limit is not None:
        n = min(int(args.limit), len(dataset))  # type: ignore[arg-type]
        dataset = Subset(dataset, range(n))
        print(f"Evaluating on a subset of the first {n} shapes.")

    return dataset


def configure_dataloader(args: Namespace) -> DataLoader:
    dataset = configure_dataset(args)
    return PointCloudDataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)


def configure_model(args: Namespace) -> Module:
    return create_model(args.model, task="part-segmentation", pretrained=True)


def configure_inferer(args: Namespace) -> Inferer:
    votes = TTAInferer(base=SimpleInferer(), transforms=INFERER_TRANSFORM, num_passes=NUM_VOTES, include_identity=True)
    return PartRefinementInferer(base=votes, min_count=10, num_neighbors=11)


@torch.no_grad()
def evaluate(model: Module, dataloader: DataLoader, inferer: Inferer, device: str) -> Dict[str, Any]:
    model.to(device).eval()
    part_ids = list(ShapeNetPart.seg_ids.values())
    ious: List[Tensor] = []
    categories: List[Tensor] = []

    for data in tqdm(dataloader, total=len(dataloader), desc="Testing"):
        data = {key: value.to(device) if torch.is_tensor(value) else value for key, value in data.items()}
        scores = inferer(
            data,
            predictor=lambda d: model(d[DataKeys.X], d[DataKeys.POS], d[DataKeys.BATCH], d[DataKeys.CATEGORY]),
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
