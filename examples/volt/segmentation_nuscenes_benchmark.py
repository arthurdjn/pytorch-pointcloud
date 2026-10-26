"""Benchmark Volt semantic segmentation on nuScenes-lidarseg at full resolution."""

import math
import os
from argparse import ArgumentParser, Namespace
from typing import Any, Dict

import torch
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

import torch_pointcloud.transforms as T
from torch_pointcloud.config import DATA_DIR
from torch_pointcloud.datasets import NuScenes, NuScenesMini
from torch_pointcloud.datasets.nuscenes import NUSCENES_LIDARSEG_LEARNING_MAP
from torch_pointcloud.inferers import Inferer, SimpleInferer, TTAInferer, VoxelPartitionInferer
from torch_pointcloud.metrics import accuracy, confusion_matrix, intersection_over_union
from torch_pointcloud.models import create_model, model_info
from torch_pointcloud.utils.data import DataKeys, PointCloudDataLoader
from torch_pointcloud.utils.random import seed_everything, set_determinism

CUDA_AVAILABLE = torch.cuda.is_available()
CPU_COUNT = os.cpu_count()
DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"
NUM_WORKERS = CPU_COUNT // 2 if CPU_COUNT is not None else 0
SEED = 42
VOXEL_SIZE = 0.05
SW_BATCH_SIZE = 8
MODELS = (
    "volt-s.nuscenes.kadir-yilmaz",
    "volt-s-joint.nuscenes.kadir-yilmaz",
    "volt-b-joint.nuscenes.kadir-yilmaz",
)
NUM_CLASSES = 16

INFERER_TRANSFORM = T.Compose(
    [
        T.Cat(keys=[DataKeys.POS, DataKeys.INTENSITY], dst_key=DataKeys.X, dim=1),
        T.Quantize(keys=DataKeys.POS, size=VOXEL_SIZE, dst_keys=DataKeys.POS_GRID),
    ]
)
TRANSFORM = T.Compose(
    [
        T.Relabel(keys=DataKeys.SEGMENT, labels=NUSCENES_LIDARSEG_LEARNING_MAP, default=-1),
        # The released weights were trained on intensities in $[0, 1]$.
        T.Normalize(keys=DataKeys.INTENSITY, mean=[0.0], std=[255.0]),
        T.CopyItems(keys=DataKeys.SEGMENT, dst_keys=DataKeys.ORIGIN_SEGMENT),
        # Points beyond 70 m are pulled back onto that range and heights clamped, as in training.
        T.ClampNorm(keys=DataKeys.POS, max_norm=70.0),
        T.Clamp(keys=DataKeys.POS, min=[-math.inf, -math.inf, -4.0], max=[math.inf, math.inf, 2.0]),
        # The authors' test pipeline first samples one point per 2.5 cm voxel, then scores every raw point.
        T.Voxelize(
            pos_key=DataKeys.POS,
            pos_reduce="first",
            keys=[DataKeys.INTENSITY, DataKeys.SEGMENT],
            reduce="first",
            size=0.025,
            method="fnv",
            dst_inverse_key=DataKeys.INVERSE,
        ),
    ]
)
# The authors' precise evaluation: five scales, each with and without an x/y flip.
TTA_TRANSFORMS = [
    T.Compose(
        [
            T.RandomScale(keys=DataKeys.POS, scale_range=(scale, scale), p=1.0),
            *([T.RandomFlip(keys=DataKeys.POS, axes=(0, 1), p=1.0)] if flip else []),
        ]
    )
    for flip in (False, True)
    for scale in (0.9, 0.95, 1.0, 1.05, 1.1)
]


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    set_determinism(tf32=False)

    print(f"Benchmarking model {args.model!r} on nuScenes-lidarseg!")
    dataloader = configure_dataloader(args)
    model = configure_model(args)
    inferer = configure_inferer(args)

    num_scans = len(dataloader.dataset)  # type: ignore[arg-type]
    protocol = "single pass" if args.simple else "test-time views x voxel-partition fragments"
    print(f"Test set: {num_scans} scans  ({protocol}, scored at full resolution)")
    metrics = evaluate(model, dataloader, inferer, args.device, NUM_CLASSES)
    print("\nResults:")
    for key, value in metrics.items():
        print(f"  {key:<24} {value:.4f}")


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Benchmark Volt semantic segmentation on nuScenes-lidarseg.")
    parser.add_argument("--model", default=MODELS[0], choices=MODELS)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--root", default=DATA_DIR, help="Dataset root directory.")
    parser.add_argument("--split", default="val", choices=("val", "mini"), help="nuScenes `val`, or the mini release.")
    parser.add_argument("--seed", default=SEED, type=int)
    parser.add_argument("--sw-batch-size", default=SW_BATCH_SIZE, type=int, help="Voxel fragments per forward.")
    parser.add_argument("--num-workers", default=NUM_WORKERS, type=int)
    parser.add_argument("--limit", default=None, type=int, help="Evaluate at most this many scans.")
    parser.add_argument(
        "--simple",
        action="store_true",
        help="Predict each point in one pass, without voting or test-time augmentation.",
    )
    return parser.parse_args()


def configure_dataset(args: Namespace) -> Dataset:
    transform = model_info(args.model, task="semantic-segmentation")["transform"] if args.simple else TRANSFORM
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
    if args.simple:
        return SimpleInferer()

    base = VoxelPartitionInferer(
        voxel_size=VOXEL_SIZE,
        transform=INFERER_TRANSFORM,
        softmax=True,
        aggregate="sum",
        sw_batch_size=args.sw_batch_size,
        seed=args.seed,
    )
    return TTAInferer(base=base, transforms=TTA_TRANSFORMS, aggregate="mean")


@torch.no_grad()
def evaluate(model: Module, dataloader: DataLoader, inferer: Inferer, device: str, num_classes: int) -> Dict[str, Any]:
    model.to(device).eval()
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)

    pbar = tqdm(dataloader, total=len(dataloader), desc="Testing")
    for data in pbar:
        data = {key: value.to(device) if torch.is_tensor(value) else value for key, value in data.items()}
        scores = inferer(data, predictor=lambda d: model(d[DataKeys.X], d[DataKeys.POS_GRID], d[DataKeys.BATCH]))
        preds = scores.argmax(dim=1)[data[DataKeys.INVERSE]]
        cm += confusion_matrix(preds.cpu(), data[DataKeys.ORIGIN_SEGMENT].cpu(), num_classes, ignore_index=-1)
        oa = accuracy(cm)
        pbar.set_postfix({"oa": f"{oa:.4f}"})

    return {
        "test/mIoU": intersection_over_union(cm),
        "test/oa": accuracy(cm),
    }


if __name__ == "__main__":
    main()
