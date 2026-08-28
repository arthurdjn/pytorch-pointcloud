"""Benchmark the DGCNN semantic-segmentation models on ScanNet and S3DIS with their reference block protocols.

NOTE: the S3DIS reference publishes only the pooled 6-fold result (59.2 mIoU / 85.0 OA); the mean below averages per-area
metrics instead.

Results (mIoU / OA):

    | Variant                  | reference | torch-pointcloud |
    | ------------------------ | --------- | ---------------- |
    | dgcnn.scannet20.an-tao   | 49.6      | 53.06 / 81.83    |
    | dgcnn.s3dis-area1.an-tao |           | 69.19 / 89.69    |
    | dgcnn.s3dis-area2.an-tao |           | 43.50 / 81.69    |
    | dgcnn.s3dis-area3.an-tao |           | 68.73 / 90.86    |
    | dgcnn.s3dis-area4.an-tao |           | 50.68 / 85.06    |
    | dgcnn.s3dis-area5.an-tao |           | 50.30 / 84.92    |
    | dgcnn.s3dis-area6.an-tao |           | 75.60 / 92.10    |
    | S3DIS 6-fold mean        | 59.2      | 59.67 / 87.39    |

Usage:
    uv run --no-sync python examples/dgcnn_benchmark_segmentation.py --dataset scannet --limit 5
    uv run --no-sync python examples/dgcnn_benchmark_segmentation.py --dataset s3dis --area 5
"""

import argparse
import os
from typing import Any, Dict

import torch
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

import torch_pointcloud.transforms as T
from torch_pointcloud.config import DATA_DIR
from torch_pointcloud.datasets import S3DISHdf5, ScanNet20
from torch_pointcloud.datasets.s3dis import S3DIS_AREAS
from torch_pointcloud.inferers import Inferer, SimpleInferer, SlidingWindowInferer
from torch_pointcloud.models import create_model
from torch_pointcloud.utils.data import DataKeys, PointCloudDataLoader
from torch_pointcloud.utils.metrics import confusion_matrix
from torch_pointcloud.utils.random import seed_everything, set_determinism

CUDA_AVAILABLE = torch.cuda.is_available()
CPU_COUNT = os.cpu_count()
DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"
NUM_WORKERS = CPU_COUNT // 2 if CPU_COUNT is not None else 0
SEED = 42
IGNORE_INDEX = 255
BLOCK_SIZE = 1.5
BLOCK_NUM_POINTS = 8192

SCANNET_TRANSFORM = T.Compose(
    [
        T.Relabel(keys=DataKeys.SEGMENT, labels=range(1, 21), default=IGNORE_INDEX),
        T.Reduce(keys=DataKeys.POS, op="max", dst_keys=DataKeys.SCENE_MAX),
    ]
)
SCANNET_INFERER_TRANSFORM = T.Compose(
    [
        T.BBoxCenter(keys="block_bbox", dst_keys=DataKeys.BLOCK_CENTER),
        T.CopyItems(keys=DataKeys.POS, names=DataKeys.NORM_POS),
        T.DivideKey(keys=DataKeys.NORM_POS, div_keys=DataKeys.SCENE_MAX),
        T.SubtractKey(keys=DataKeys.POS, sub_keys=DataKeys.BLOCK_CENTER, axes=[0, 1]),
        T.Divide(keys=DataKeys.COLOR, divisor=255),
        T.Cat(keys=[DataKeys.POS, DataKeys.COLOR], dst_key=DataKeys.X, dim=1),
        T.DivisiblePad(num_samples=BLOCK_NUM_POINTS, pad_fill="random", dst_inverse_key=DataKeys.INVERSE),
    ]
)


def build_scannet_inferer(seed: int) -> Inferer:
    return SlidingWindowInferer(
        block_size=BLOCK_SIZE,
        overlap=0.5,
        dims=(0, 1),
        padding=1e-8,
        roi_num_points=BLOCK_NUM_POINTS,
        softmax=False,
        aggregate="mean",
        transform=SCANNET_INFERER_TRANSFORM,
        inverse_key=DataKeys.INVERSE,
        seed=seed,
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
        oa = cm.diag().sum().float() / cm.sum().float().clamp_min(1)
        pbar.set_postfix({"oa": f"{oa.item():.4f}"})

    intersection = cm.diag().float()
    union = cm.sum(dim=1).float() + cm.sum(dim=0).float() - intersection
    iou_per_class = intersection / union.clamp_min(1e-10)
    return {
        "test/mIoU": iou_per_class.mean().item(),
        "test/oa": (cm.diag().sum().float() / cm.sum().float().clamp_min(1)).item(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark DGCNN semantic segmentation on ScanNet or S3DIS.")
    parser.add_argument("--dataset", default="scannet", choices=["scannet", "s3dis"])
    parser.add_argument("--area", default=5, type=int, choices=range(1, 7), help="Held-out S3DIS area.")
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--root", default=DATA_DIR, help="Dataset root directory.")
    parser.add_argument("--seed", default=SEED, type=int)
    parser.add_argument("--batch-size", default=16, type=int, help="Blocks per forward on S3DIS.")
    parser.add_argument("--num-workers", default=NUM_WORKERS, type=int)
    parser.add_argument("--limit", default=None, type=int, help="Evaluate at most this many scenes or blocks.")
    parser.add_argument("--download", action="store_true", help="Download the dataset if missing.")
    parser.add_argument("--force-process", action="store_true", help="Force re-processing the dataset.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    set_determinism(tf32=False)

    dataset: Dataset
    if args.dataset == "scannet":
        name = "dgcnn.scannet20.an-tao"
        print(f"Benchmarking model {name!r} on ScanNet!")
        model = create_model(name, task="segmentation", pretrained=True)
        dataset = ScanNet20(
            root=args.root,
            split="val",
            transform=SCANNET_TRANSFORM,
            download=args.download,
            force_process=args.force_process,
            num_workers=args.num_workers,
            use_axis_alignment=False,
        )
        inferer = build_scannet_inferer(args.seed)
        batch_size = 1
    else:
        name = f"dgcnn.s3dis-area{args.area}.an-tao"
        print(f"Benchmarking model {name!r} on S3DIS Area {args.area} (pre-tiled blocks)!")
        model, model_info = create_model(name, task="segmentation", pretrained=True, return_info=True)
        dataset = S3DISHdf5(
            root=args.root,
            areas=[S3DIS_AREAS[args.area - 1]],
            transform=model_info["transform"],
            download=args.download,
            force_process=args.force_process,
        )
        inferer = SimpleInferer()
        batch_size = args.batch_size
    num_classes = int(model.num_classes)
    if args.limit is not None:
        n = min(int(args.limit), len(dataset))
        dataset = Subset(dataset, range(n))
        print(f"Evaluating on a subset of the first {n} samples.")

    dataloader = PointCloudDataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)

    print(f"Test set: {len(dataset)} samples")
    metrics = evaluate(model, dataloader, inferer, args.device, num_classes)
    print("\nResults:")
    for key, value in metrics.items():
        print(f"  {key:<24} {value:.4f}")


if __name__ == "__main__":
    main()
