"""Benchmark the PointNeXt ShapeNetPart part-segmentation models with the reference voting protocol.

Results (instance mIoU / class mIoU):

    | Variant                                   | reference   | torch-pointcloud |
    | ----------------------------------------- | ----------- | ---------------- |
    | pointnext-sm.shapenetpart.openpoints      | 86.7 / 84.2 | 86.88 / 84.48    |
    | pointnext-sm-c64.shapenetpart.openpoints  | 86.9 / 85.2 | 86.93 / 85.24    |
    | pointnext-sm-c160.shapenetpart.openpoints | 87.1 / 85.4 | 87.11 / 85.25    |

Usage:
    uv run --no-sync python examples/pointnext_benchmark_part_segmentation.py --model pointnext-sm.shapenetpart.openpoints
"""

import argparse
import os
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
from torch_pointcloud.models import create_model
from torch_pointcloud.utils.data import DataKeys, PointCloudDataLoader
from torch_pointcloud.utils.metrics import part_mean_iou
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


def build_inferer() -> Inferer:
    votes = TTAInferer(base=SimpleInferer(), transforms=INFERER_TRANSFORM, num_passes=NUM_VOTES, include_identity=True)
    return PartRefinementInferer(base=votes, min_count=10, num_neighbors=11)


@torch.no_grad()
def evaluate(model: Module, dataloader: DataLoader, inferer: Inferer, device: str) -> Dict[str, Any]:
    model.to(device).eval()
    preds: List[Tensor] = []
    targets: List[Tensor] = []
    categories: List[Tensor] = []
    batches: List[Tensor] = []
    num_shapes = 0

    for data in tqdm(dataloader, total=len(dataloader), desc="Testing"):
        data = {key: value.to(device) if torch.is_tensor(value) else value for key, value in data.items()}
        scores = inferer(
            data,
            predictor=lambda d: model(d[DataKeys.X], d[DataKeys.POS], d[DataKeys.BATCH], d[DataKeys.CATEGORY]),
        )
        preds.append(scores.argmax(dim=1).cpu())
        targets.append(data[DataKeys.SEGMENT].cpu())
        categories.append(data[DataKeys.CATEGORY].argmax(dim=1).cpu())
        batches.append(data[DataKeys.BATCH].cpu() + num_shapes)
        num_shapes += int(data[DataKeys.CATEGORY].shape[0])

    metrics = part_mean_iou(
        torch.cat(preds),
        torch.cat(targets),
        list(ShapeNetPart.seg_ids.values()),
        torch.cat(categories),
        torch.cat(batches),
    )
    return {f"test/{key}": value for key, value in metrics.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark PointNeXt part segmentation on ShapeNetPart.")
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


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    set_determinism(tf32=False)

    print(f"Benchmarking model {args.model!r} on ShapeNetPart!")
    model, model_info = create_model(args.model, task="segmentation", pretrained=True, return_info=True)
    inferer = build_inferer()

    dataset: Dataset = ShapeNetPart(root=args.root, split="test", transform=model_info["transform"])
    if args.limit is not None:
        n = min(int(args.limit), len(dataset))  # type: ignore[arg-type]
        dataset = Subset(dataset, range(n))
        print(f"Evaluating on a subset of the first {n} shapes.")

    dataloader = PointCloudDataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print(f"Test set: {len(dataset)} shapes  ({NUM_VOTES} scale votes + identity, nearest-neighbor part refinement)")  # type: ignore[arg-type]
    metrics = evaluate(model, dataloader, inferer, args.device)
    print("\nResults:")
    for key, value in metrics.items():
        print(f"  {key:<24} {value:.4f}")


if __name__ == "__main__":
    main()
