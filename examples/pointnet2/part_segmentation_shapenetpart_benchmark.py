"""Benchmark PointNet++ part segmentation on ShapeNetPart with the full-shape chunk voting protocol."""

import os
from argparse import ArgumentParser, Namespace
from typing import Any, Dict, List

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

import torch_pointcloud.transforms as T
from torch_pointcloud.config import DATA_DIR
from torch_pointcloud.datasets import ShapeNetPart
from torch_pointcloud.inferers import Inferer, SlidingWindowInferer, TTAInferer
from torch_pointcloud.metrics import (
    part_intersection_over_union,
    part_mean_intersection_over_union,
    part_predictions,
)
from torch_pointcloud.models import PartSegmentationModel, create_model, model_info
from torch_pointcloud.utils.data import DataKeys, PointCloudDataLoader
from torch_pointcloud.utils.random import seed_everything, set_determinism

CUDA_AVAILABLE = torch.cuda.is_available()
CPU_COUNT = os.cpu_count()
DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"
NUM_WORKERS = CPU_COUNT // 2 if CPU_COUNT is not None else 0
SEED = 42
NUM_POINTS = 2048
NUM_VOTES = 10
SW_BATCH_SIZE = 8


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    set_determinism(tf32=False)

    print(f"Benchmarking model {args.model!r} on ShapeNetPart!")
    dataloader = configure_dataloader(args)
    model = configure_model(args)
    inferer = configure_inferer(args)

    num_shapes = len(dataloader.dataset)  # type: ignore[arg-type]
    print(f"Test set: {num_shapes} shapes  ({args.num_votes} votes of random {args.num_points}-point chunks)")
    metrics = evaluate(model, dataloader, inferer, args.device)
    print("\nResults:")
    for key, value in metrics.items():
        print(f"  {key:<24} {value:.4f}")


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Benchmark PointNet++ part segmentation on ShapeNetPart.")
    parser.add_argument(
        "--model", default="pointnet2-ssg.shapenetpart.mit-han-lab", help="Registered segmentation model name"
    )
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--root", default=DATA_DIR, help="Dataset root directory.")
    parser.add_argument("--seed", default=SEED, type=int)
    parser.add_argument("--num-points", default=NUM_POINTS, type=int, help="Points per chunk.")
    parser.add_argument("--num-votes", default=NUM_VOTES, type=int, help="Random chunkings of each shape.")
    parser.add_argument("--sw-batch-size", default=SW_BATCH_SIZE, type=int, help="Chunks per forward.")
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
    return PointCloudDataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)


def configure_model(args: Namespace) -> PartSegmentationModel:
    return create_model(args.model, task="part-segmentation", pretrained=True)


def configure_inferer_transform(args: Namespace) -> T.Compose:
    return T.Compose(
        [
            T.DivisiblePad(num_samples=args.num_points, pad_fill="random", dst_inverse_key=DataKeys.INVERSE),
        ]
    )


def configure_inferer(args: Namespace) -> Inferer:
    chunks = SlidingWindowInferer(
        block_size=None,
        roi_num_points=args.num_points,
        sw_batch_size=args.sw_batch_size,
        softmax=True,
        transform=configure_inferer_transform(args),
        inverse_key=DataKeys.INVERSE,
    )
    return TTAInferer(base=chunks, num_passes=args.num_votes, aggregate="max")


@torch.no_grad()
def evaluate(model: PartSegmentationModel, dataloader: DataLoader, inferer: Inferer, device: str) -> Dict[str, Any]:
    """Instance and class mIoU, each shape predicted among the parts of its own category as the reference does."""
    model.to(device).eval()
    part_ids = list(ShapeNetPart.seg_ids.values())
    ious: List[Tensor] = []
    categories: List[Tensor] = []
    for data in tqdm(dataloader, total=len(dataloader), desc="Testing"):
        data = {key: value.to(device) if torch.is_tensor(value) else value for key, value in data.items()}
        probs = inferer(
            data,
            predictor=lambda d: model(d[DataKeys.X], d[DataKeys.POS], d[DataKeys.BATCH], d[DataKeys.CATEGORY]),
        )
        category = data[DataKeys.CATEGORY].argmax(dim=1)
        preds = part_predictions(probs, part_ids, category, data[DataKeys.BATCH]).cpu()
        shape_ious = part_intersection_over_union(
            preds, data[DataKeys.SEGMENT].cpu(), part_ids, category.cpu(), data[DataKeys.BATCH].cpu()
        )
        ious.append(shape_ious)
        categories.append(category.cpu())

    shape_ious, category = torch.cat(ious), torch.cat(categories)
    return {
        "test/ins_mIoU": part_mean_intersection_over_union(shape_ious, category),
        "test/cls_mIoU": part_mean_intersection_over_union(shape_ious, category, average="macro"),
    }


if __name__ == "__main__":
    main()
