"""Benchmark Point Transformer V3 semantic segmentation on DALES, 50 m blocks scored at full resolution.

NOTE: the reference 69.29 mIoU (opengeos/geoai) is scored on one random draw of 24 000 grid-sampled points per block;
this script predicts every voxel of a block, dealt into random groups of at most 24 000 as the model was trained, and
scores every point of the tile.
"""

import os
from argparse import ArgumentParser, Namespace
from typing import Any, Dict

import torch
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

import torch_pointcloud.transforms as T
from torch_pointcloud.config import DATA_DIR
from torch_pointcloud.datasets import DALES
from torch_pointcloud.datasets.dales import DALES_CLASSES, DALES_IGNORE_IDX
from torch_pointcloud.inferers import Inferer, SlidingWindowInferer
from torch_pointcloud.metrics import accuracy, confusion_matrix, intersection_over_union
from torch_pointcloud.models import create_model
from torch_pointcloud.utils.data import DataKeys, PointCloudDataLoader
from torch_pointcloud.utils.random import seed_everything, set_determinism

CUDA_AVAILABLE = torch.cuda.is_available()
CPU_COUNT = os.cpu_count()
DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"
NUM_WORKERS = CPU_COUNT // 2 if CPU_COUNT is not None else 0
SEED = 42
MODEL = "ptv3.dales.jayakumar-pujar"
NUM_CLASSES = len(DALES_CLASSES)
BLOCK_SIZE = 50.0
VOXEL_SIZE = 0.15
# The checkpoint was trained on random subsets of at most 40 000 grid-sampled points per block and evaluated on 24 000:
# a whole 50 m block (60 000 to 70 000 voxels) is far denser than it has seen, and loses 6 mIoU.
ROI_NUM_POINTS = 24_000
SW_BATCH_SIZE = 2

# The return number is divided by the highest return number of the tile.
TRANSFORM = T.Compose([T.MaxAbsScale(keys=DataKeys.RETURN_NUMBER)])
# Each block is grid-sampled, centred on its mean, and fed `[x, y, z, return number]`.
INFERER_TRANSFORM = T.Compose(
    [
        T.Voxelize(
            pos_key=DataKeys.POS,
            pos_reduce="first",
            dst_pos_grid_key=DataKeys.POS_GRID,
            keys=[DataKeys.RETURN_NUMBER],
            reduce="first",
            size=VOXEL_SIZE,
            dst_inverse_key=DataKeys.INVERSE,
        ),
        T.Shift(keys=DataKeys.POS, method="centroid"),
        T.Cat(keys=[DataKeys.POS, DataKeys.RETURN_NUMBER], dst_key=DataKeys.X, dim=1),
    ]
)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    set_determinism(tf32=False)

    print(f"Benchmarking model {args.model!r} on DALES!")
    dataloader = configure_dataloader(args)
    model = configure_model(args)
    inferer = configure_inferer(args)

    num_tiles = len(dataloader.dataset)  # type: ignore[arg-type]
    print(f"Test set: {num_tiles} tiles  (non-overlapping {BLOCK_SIZE:g} m blocks, scored at full resolution)")
    metrics = evaluate(model, dataloader, inferer, args.device, NUM_CLASSES)
    print("\nResults:")
    for key, value in metrics.items():
        print(f"  {key:<24} {value:.4f}")


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Benchmark Point Transformer V3 semantic segmentation on DALES.")
    parser.add_argument("--model", default=MODEL, choices=[MODEL])
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--root", default=DATA_DIR, help="Dataset root directory.")
    parser.add_argument("--seed", default=SEED, type=int)
    parser.add_argument("--sw-batch-size", default=SW_BATCH_SIZE, type=int, help="Blocks per forward.")
    parser.add_argument("--num-workers", default=NUM_WORKERS, type=int)
    parser.add_argument("--limit", default=None, type=int, help="Evaluate at most this many tiles.")
    return parser.parse_args()


def configure_dataset(args: Namespace) -> Dataset:
    dataset: Dataset = DALES(root=args.root, train=False, transform=TRANSFORM)
    if args.limit is not None:
        n = min(int(args.limit), len(dataset))  # type: ignore[arg-type]
        dataset = Subset(dataset, range(n))
        print(f"Evaluating on a subset of the first {n} tiles.")

    return dataset


def configure_dataloader(args: Namespace) -> DataLoader:
    dataset = configure_dataset(args)
    return PointCloudDataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)


def configure_model(args: Namespace) -> Module:
    return create_model(args.model, task="semantic-segmentation", pretrained=True)


def configure_inferer(args: Namespace) -> Inferer:
    return SlidingWindowInferer(
        block_size=BLOCK_SIZE,
        overlap=0.0,
        dims=(0, 1),
        roi_num_points=ROI_NUM_POINTS,
        sw_batch_size=args.sw_batch_size,
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
        scores = inferer(
            data,
            predictor=lambda d: model(d[DataKeys.X], d[DataKeys.POS_GRID], d[DataKeys.BATCH], pos=d[DataKeys.POS]),
        )
        preds = scores.argmax(dim=1)
        cm += confusion_matrix(preds.cpu(), data[DataKeys.SEGMENT].cpu(), num_classes, ignore_index=DALES_IGNORE_IDX)
        oa = accuracy(cm, ignore_index=DALES_IGNORE_IDX)
        pbar.set_postfix({"oa": f"{oa:.4f}"})

    per_class = intersection_over_union(cm, average="none", ignore_index=DALES_IGNORE_IDX, class_names=DALES_CLASSES)
    return {
        "test/mIoU": intersection_over_union(cm, ignore_index=DALES_IGNORE_IDX),
        "test/oa": accuracy(cm, ignore_index=DALES_IGNORE_IDX),
        **{f"test/IoU/{name}": iou for name, iou in per_class.items() if name != DALES_CLASSES[DALES_IGNORE_IDX]},
    }


if __name__ == "__main__":
    main()
