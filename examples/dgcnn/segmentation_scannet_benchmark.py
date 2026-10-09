"""Benchmark the DGCNN ScanNet semantic-segmentation model with its sliding-block protocol."""

import os
from argparse import ArgumentParser, Namespace
from typing import Any, Dict

import torch
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

import torch_pointcloud.transforms as T
from torch_pointcloud.config import DATA_DIR
from torch_pointcloud.datasets import ScanNet20
from torch_pointcloud.inferers import Inferer, SlidingWindowInferer
from torch_pointcloud.metrics import accuracy, confusion_matrix, intersection_over_union
from torch_pointcloud.models import SemanticSegmentationModel, create_model
from torch_pointcloud.utils.data import DataKeys, PointCloudDataLoader
from torch_pointcloud.utils.random import seed_everything, set_determinism

CUDA_AVAILABLE = torch.cuda.is_available()
CPU_COUNT = os.cpu_count()
DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"
NUM_WORKERS = CPU_COUNT // 2 if CPU_COUNT is not None else 0
SEED = 42
IGNORE_INDEX = 255
BLOCK_SIZE = 1.5
SW_BATCH_SIZE = 8
BLOCK_NUM_POINTS = 8192
MODELS = ["dgcnn.scannet20.an-tao"]

TRANSFORM = T.Compose(
    [
        T.Relabel(keys=DataKeys.SEGMENT, labels=range(1, 21), default=IGNORE_INDEX),
        T.Reduce(keys=DataKeys.POS, op="max", dst_keys=DataKeys.SCENE_MAX),
    ]
)
INFERER_TRANSFORM = T.Compose(
    [
        T.BBoxCenter(keys=DataKeys.BLOCK_BBOX, dst_keys=DataKeys.BLOCK_CENTER),
        T.CopyItems(keys=DataKeys.POS, dst_keys=DataKeys.NORM_POS),
        T.DivideItems(keys=DataKeys.NORM_POS, div_keys=DataKeys.SCENE_MAX),
        T.SubtractItems(keys=DataKeys.POS, sub_keys=DataKeys.BLOCK_CENTER, axes=[0, 1]),
        T.Divide(keys=DataKeys.COLOR, divisor=255),
        T.Cat(keys=[DataKeys.POS, DataKeys.COLOR], dst_key=DataKeys.X, dim=1),
        T.DivisiblePad(num_samples=BLOCK_NUM_POINTS, pad_fill="random", dst_inverse_key=DataKeys.INVERSE),
    ]
)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    set_determinism(tf32=False)

    print(f"Benchmarking model {args.model!r} on ScanNet!")
    dataloader = configure_dataloader(args)
    model = configure_model(args)
    inferer = configure_inferer(args)

    num_samples = len(dataloader.dataset)  # type: ignore[arg-type]
    print(f"Test set: {num_samples} samples")
    metrics = evaluate(model, dataloader, inferer, args.device, int(model.num_classes))
    print("\nResults:")
    for key, value in metrics.items():
        print(f"  {key:<24} {value:.4f}")


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Benchmark DGCNN semantic segmentation on ScanNet.")
    parser.add_argument("--model", default=MODELS[0], choices=MODELS, help="Registered segmentation model name")
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--root", default=DATA_DIR, help="Dataset root directory.")
    parser.add_argument("--seed", default=SEED, type=int)
    parser.add_argument("--sw-batch-size", default=SW_BATCH_SIZE, type=int, help="Blocks per forward.")
    parser.add_argument("--num-workers", default=NUM_WORKERS, type=int)
    parser.add_argument("--limit", default=None, type=int, help="Evaluate at most this many scenes.")
    parser.add_argument("--download", action="store_true", help="Download the dataset if missing.")
    parser.add_argument("--force-process", action="store_true", help="Force re-processing the dataset.")
    return parser.parse_args()


def configure_dataset(args: Namespace) -> Dataset:
    dataset: Dataset
    dataset = ScanNet20(
        root=args.root,
        split="val",
        transform=TRANSFORM,
        download=args.download,
        force_process=args.force_process,
        num_workers=args.num_workers,
        use_axis_alignment=False,
    )
    if args.limit is not None:
        n = min(int(args.limit), len(dataset))
        dataset = Subset(dataset, range(n))
        print(f"Evaluating on a subset of the first {n} samples.")

    return dataset


def configure_dataloader(args: Namespace) -> DataLoader:
    # Scenes go through the sliding window one at a time.
    dataset = configure_dataset(args)
    return PointCloudDataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)


def configure_model(args: Namespace) -> SemanticSegmentationModel:
    return create_model(args.model, task="semantic-segmentation", pretrained=True)


def configure_inferer(args: Namespace) -> Inferer:
    return SlidingWindowInferer(
        block_size=BLOCK_SIZE,
        overlap=0.5,
        dims=(0, 1),
        padding=1e-8,
        roi_num_points=BLOCK_NUM_POINTS,
        sw_batch_size=args.sw_batch_size,
        softmax=False,
        aggregate="mean",
        transform=INFERER_TRANSFORM,
        inverse_key=DataKeys.INVERSE,
        seed=args.seed,
    )


@torch.no_grad()
def evaluate(model: Module, dataloader: DataLoader, inferer: Inferer, device: str, num_classes: int) -> Dict[str, Any]:
    model.to(device).eval()
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)

    pbar = tqdm(dataloader, total=len(dataloader), desc="Testing")
    for data in pbar:
        data = {key: value.to(device) if torch.is_tensor(value) else value for key, value in data.items()}
        logits = inferer(data, predictor=lambda d: model(d[DataKeys.X], d[DataKeys.NORM_POS], d[DataKeys.BATCH]))
        preds = logits.argmax(dim=1)
        cm += confusion_matrix(preds.cpu(), data[DataKeys.SEGMENT].cpu(), num_classes, ignore_index=IGNORE_INDEX)
        oa = accuracy(cm)
        pbar.set_postfix({"oa": f"{oa:.4f}"})

    return {
        "test/mIoU": intersection_over_union(cm),
        "test/oa": accuracy(cm),
    }


if __name__ == "__main__":
    main()
