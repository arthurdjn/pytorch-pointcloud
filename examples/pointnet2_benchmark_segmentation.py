"""Benchmark the PointNet++ S3DIS semantic-segmentation checkpoints on Area 5, each with its reference protocol.

Results (mIoU / OA; the 6-fold rows average the per-area folds, the reference pools their confusion matrices):

    | Variant                          | reference   | torch-pointcloud |
    | -------------------------------- | ----------- | ---------------- |
    | pointnet2.s3dis-area5.xu-yan     | 53.5        | 54.28 / 83.54    |
    | pointnet2.s3dis-area1.openpoints |             | 74.96 / 89.77    |
    | pointnet2.s3dis-area2.openpoints |             | 48.22 / 80.08    |
    | pointnet2.s3dis-area3.openpoints |             | 76.31 / 90.89    |
    | pointnet2.s3dis-area4.openpoints |             | 59.96 / 85.67    |
    | pointnet2.s3dis-area5.openpoints | 63.6        | 63.66 / 88.23    |
    | pointnet2.s3dis-area6.openpoints |             | 82.45 / 92.99    |
    | pointnet2 openpoints 6-fold mean | 68.1 / 87.6 | 67.59 / 87.94    |

Usage:
    uv run --no-sync python examples/pointnet2_benchmark_segmentation.py --model pointnet2.s3dis-area5.xu-yan
    uv run --no-sync python examples/pointnet2_benchmark_segmentation.py --model pointnet2.s3dis-area5.openpoints
"""

import argparse
import os
from typing import Any, Callable, Dict, Optional, Tuple

import torch
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

import torch_pointcloud.transforms as T
from torch_pointcloud.config import DATA_DIR
from torch_pointcloud.datasets import S3DIS
from torch_pointcloud.inferers import Inferer, SlidingWindowInferer, TTAInferer, VoxelPartitionInferer
from torch_pointcloud.models import create_model
from torch_pointcloud.utils.data import DataKeys, PointCloudDataLoader
from torch_pointcloud.utils.metrics import confusion_matrix
from torch_pointcloud.utils.random import seed_everything, set_determinism

CUDA_AVAILABLE = torch.cuda.is_available()
CPU_COUNT = os.cpu_count()
DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"
NUM_WORKERS = CPU_COUNT // 2 if CPU_COUNT is not None else 0
SEED = 42

BLOCK_SIZE = 1.0
BLOCK_NUM_POINTS = 4096
NUM_VOTES = 3
VOXEL_SIZE = 0.04
SUB_BATCH_SIZE = 4

XU_YAN_TRANSFORM = T.Compose(
    [
        T.Shift(keys=DataKeys.POS, method="min"),
        T.Reduce(keys=DataKeys.POS, op="max", dst_keys="coord_max"),
    ]
)
XU_YAN_INFERER_TRANSFORM = T.Compose(
    [
        T.BBoxCenter(keys="block_bbox", dst_keys=DataKeys.BLOCK_CENTER),
        T.CopyItems(keys=DataKeys.POS, names=DataKeys.NORM_POS),
        T.DivideKey(keys=DataKeys.NORM_POS, div_keys="coord_max"),
        T.SubtractKey(keys=DataKeys.POS, sub_keys=DataKeys.BLOCK_CENTER, axes=[0, 1]),
        T.ToFloat(keys=DataKeys.COLOR),
        T.Divide(keys=DataKeys.COLOR, divisor=255.0),
        T.Cat(keys=[DataKeys.POS, DataKeys.COLOR, DataKeys.NORM_POS], dst_key=DataKeys.X, dim=1),
        T.DivisiblePad(num_samples=BLOCK_NUM_POINTS, pad_fill="random", dst_inverse_key=DataKeys.INVERSE),
    ]
)
OPENPOINTS_TRANSFORM = T.Compose([T.Shift(keys=DataKeys.POS, method="min")])


def build_xu_yan_inferer(inferer_transform: T.Transform, seed: int) -> Inferer:
    blocks = SlidingWindowInferer(
        block_size=BLOCK_SIZE,
        overlap=0.5,
        dims=(0, 1),
        padding=0.001,
        roi_num_points=BLOCK_NUM_POINTS,
        softmax=True,
        aggregate="vote",
        transform=inferer_transform,
        inverse_key=DataKeys.INVERSE,
        seed=seed,
    )
    return TTAInferer(base=blocks, transforms=T.Compose([]), num_passes=NUM_VOTES)


def build_openpoints_inferer(inferer_transform: T.Transform, seed: int, sub_batch_size: int) -> Inferer:
    return VoxelPartitionInferer(
        voxel_size=VOXEL_SIZE,
        transform=inferer_transform,
        sub_batch_size=sub_batch_size,
        seed=seed,
    )


PROTOCOLS: Dict[str, Tuple[T.Transform, Optional[T.Transform], Callable[..., Inferer]]] = {
    "pointnet2.s3dis-area5.xu-yan": (XU_YAN_TRANSFORM, XU_YAN_INFERER_TRANSFORM, build_xu_yan_inferer),
    "pointnet2.s3dis-area1.openpoints": (OPENPOINTS_TRANSFORM, None, build_openpoints_inferer),
    "pointnet2.s3dis-area2.openpoints": (OPENPOINTS_TRANSFORM, None, build_openpoints_inferer),
    "pointnet2.s3dis-area3.openpoints": (OPENPOINTS_TRANSFORM, None, build_openpoints_inferer),
    "pointnet2.s3dis-area4.openpoints": (OPENPOINTS_TRANSFORM, None, build_openpoints_inferer),
    "pointnet2.s3dis-area5.openpoints": (OPENPOINTS_TRANSFORM, None, build_openpoints_inferer),
    "pointnet2.s3dis-area6.openpoints": (OPENPOINTS_TRANSFORM, None, build_openpoints_inferer),
}


@torch.no_grad()
def evaluate(model: Module, dataloader: DataLoader, inferer: Inferer, device: str, num_classes: int) -> Dict[str, Any]:
    model.to(device).eval()
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)

    pbar = tqdm(dataloader, total=len(dataloader), desc="Testing")
    for data in pbar:
        data = {key: value.to(device) if torch.is_tensor(value) else value for key, value in data.items()}
        scores = inferer(data, predictor=lambda d: model(d[DataKeys.X], d[DataKeys.POS], d[DataKeys.BATCH]))
        preds = scores.argmax(dim=1)
        cm += confusion_matrix(preds.cpu(), data[DataKeys.SEGMENT].cpu(), num_classes, ignore_index=-1)
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
    parser = argparse.ArgumentParser(description="Benchmark PointNet++ semantic segmentation on S3DIS Area 5.")
    parser.add_argument("--model", default="pointnet2.s3dis-area5.xu-yan", choices=sorted(PROTOCOLS))
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--root", default=DATA_DIR, help="Dataset root directory.")
    parser.add_argument("--areas", nargs="+", default=["Area_5"])
    parser.add_argument("--seed", default=SEED, type=int)
    parser.add_argument("--sub-batch-size", default=SUB_BATCH_SIZE, type=int, help="Voxel fragments per forward.")
    parser.add_argument("--num-workers", default=NUM_WORKERS, type=int)
    parser.add_argument("--limit", default=None, type=int, help="Evaluate at most this many rooms.")
    parser.add_argument("--download", action="store_true", help="Download S3DIS if missing.")
    parser.add_argument("--force-process", action="store_true", help="Force re-processing the dataset.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    set_determinism(tf32=False)

    print(f"Benchmarking model {args.model!r} on S3DIS (areas={args.areas})!")
    model, model_info = create_model(args.model, task="segmentation", pretrained=True, return_info=True)
    num_classes = int(model.num_classes)
    transform, inferer_transform, build_inferer = PROTOCOLS[args.model]
    inferer_transform = inferer_transform or model_info["transform"]
    if build_inferer is build_openpoints_inferer:
        inferer = build_openpoints_inferer(inferer_transform, args.seed, args.sub_batch_size)
    else:
        inferer = build_inferer(inferer_transform, args.seed)

    dataset: Dataset = S3DIS(
        root=args.root,
        areas=args.areas,
        transform=transform,
        download=args.download,
        force_process=args.force_process,
        num_workers=args.num_workers,
    )
    if args.limit is not None:
        n = min(int(args.limit), len(dataset))  # type: ignore[arg-type]
        dataset = Subset(dataset, range(n))
        print(f"Evaluating on a subset of the first {n} rooms.")

    dataloader = PointCloudDataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    print(f"Test set: {len(dataset)} rooms  (scored at full resolution)")  # type: ignore[arg-type]
    metrics = evaluate(model, dataloader, inferer, args.device, num_classes)
    print("\nResults:")
    for key, value in metrics.items():
        print(f"  {key:<24} {value:.4f}")


if __name__ == "__main__":
    main()
