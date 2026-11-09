"""Benchmark Volt semantic segmentation on ScanNet and ScanNet200 at full resolution."""

import os
from argparse import ArgumentParser, Namespace
from typing import Any, Dict

import torch
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

import torch_pointcloud.transforms as T
from torch_pointcloud.config import DATA_DIR
from torch_pointcloud.datasets import ScanNet20, ScanNet200
from torch_pointcloud.inferers import Inferer, SimpleInferer, TTAInferer, VoxelPartitionInferer, simple_tta_transforms
from torch_pointcloud.metrics import accuracy, confusion_matrix, intersection_over_union
from torch_pointcloud.models import create_model, model_info
from torch_pointcloud.utils.data import DataKeys, PointCloudDataLoader
from torch_pointcloud.utils.random import seed_everything, set_determinism

CUDA_AVAILABLE = torch.cuda.is_available()
CPU_COUNT = os.cpu_count()
DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"
NUM_WORKERS = CPU_COUNT // 2 if CPU_COUNT is not None else 0
SEED = 42
VOXEL_SIZE = 0.02
SW_BATCH_SIZE = 4
MODELS = (
    "volt-s.scannet20.kadir-yilmaz",
    "volt-s-joint.scannet20.kadir-yilmaz",
    "volt-b-joint.scannet20.kadir-yilmaz",
    "volt-s.scannet200.kadir-yilmaz",
    "volt-s-joint.scannet200.kadir-yilmaz",
    "volt-b-joint.scannet200.kadir-yilmaz",
)
DATASETS = {"scannet20": ScanNet20, "scannet200": ScanNet200}

INFERER_TRANSFORM = T.Compose(
    [
        T.Cat(keys=[DataKeys.COLOR, DataKeys.NORMAL], dst_key=DataKeys.X, dim=1),
        T.Quantize(keys=DataKeys.POS, size=VOXEL_SIZE, dst_keys=DataKeys.POS_GRID),
    ]
)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    set_determinism(tf32=False)

    weights = model_info(args.model, task="semantic-segmentation")["weights"]
    assert weights is not None
    num_classes = len(weights["classes"])
    print(f"Benchmarking model {args.model!r} on {weights['dataset']}!")
    dataloader = configure_dataloader(args)
    model = configure_model(args)
    inferer = configure_inferer(args)

    num_scenes = len(dataloader.dataset)  # type: ignore[arg-type]
    protocol = "single pass" if args.simple else "test-time views x voxel-partition fragments"
    print(f"Test set: {num_scenes} scenes  ({protocol}, scored at full resolution)")
    metrics = evaluate(model, dataloader, inferer, args.device, num_classes, args.simple)
    print("\nResults:")
    for key, value in metrics.items():
        print(f"  {key:<24} {value:.4f}")


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Benchmark Volt semantic segmentation on ScanNet and ScanNet200.")
    parser.add_argument("--model", default=MODELS[0], choices=MODELS)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--root", default=DATA_DIR, help="Dataset root directory.")
    parser.add_argument("--seed", default=SEED, type=int)
    parser.add_argument("--sw-batch-size", default=SW_BATCH_SIZE, type=int, help="Voxel fragments per forward.")
    parser.add_argument("--num-workers", default=NUM_WORKERS, type=int)
    parser.add_argument("--limit", default=None, type=int, help="Evaluate at most this many scenes.")
    parser.add_argument("--download", action="store_true", help="Download the dataset if missing.")
    parser.add_argument("--force-process", action="store_true", help="Force re-processing the dataset.")
    parser.add_argument(
        "--simple",
        action="store_true",
        help="Predict each point in one pass, without voting or test-time augmentation.",
    )
    return parser.parse_args()


def configure_transform(num_classes: int) -> T.Compose:
    return T.Compose(
        [
            T.Shift(keys=DataKeys.POS, method="bbox", axes=[0, 1]),
            T.Shift(keys=DataKeys.POS, method="min", axes=[2]),
            # The released weights were trained on colors in $[0, 1]$.
            T.Normalize(keys=DataKeys.COLOR, mean=[0.0, 0.0, 0.0], std=[255.0, 255.0, 255.0]),
            T.Relabel(keys=DataKeys.SEGMENT, labels=range(1, num_classes + 1), default=-1),
        ]
    )


def configure_dataset(args: Namespace) -> Dataset:
    info = model_info(args.model, task="semantic-segmentation")
    weights = info["weights"]
    assert weights is not None
    transform = info["transform"] if args.simple else configure_transform(len(weights["classes"]))
    dataset: Dataset
    dataset = DATASETS[weights["dataset"]](
        root=args.root,
        split="val",
        transform=transform,
        download=args.download,
        force_process=args.force_process,
        num_workers=args.num_workers,
        use_axis_alignment=False,
    )

    if args.limit is not None:
        n = min(int(args.limit), len(dataset))
        dataset = Subset(dataset, range(n))
        print(f"Evaluating on a subset of the first {n} scenes.")
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
    return TTAInferer(base=base, transforms=simple_tta_transforms(), aggregate="mean")


@torch.no_grad()
def evaluate(
    model: Module, dataloader: DataLoader, inferer: Inferer, device: str, num_classes: int, simple: bool
) -> Dict[str, Any]:
    model.to(device).eval()
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)

    pbar = tqdm(dataloader, total=len(dataloader), desc="Testing")
    for data in pbar:
        data = {key: value.to(device) if torch.is_tensor(value) else value for key, value in data.items()}
        scores = inferer(data, predictor=lambda d: model(d[DataKeys.X], d[DataKeys.POS_GRID], d[DataKeys.BATCH]))
        preds = scores.argmax(dim=1)
        if simple:
            preds = preds[data[DataKeys.INVERSE]]
        target = data[DataKeys.ORIGIN_SEGMENT if simple else DataKeys.SEGMENT]
        cm += confusion_matrix(preds.cpu(), target.cpu(), num_classes, ignore_index=-1)
        oa = accuracy(cm)
        pbar.set_postfix({"oa": f"{oa:.4f}"})

    return {
        "test/mIoU": intersection_over_union(cm),
        "test/oa": accuracy(cm),
    }


if __name__ == "__main__":
    main()
