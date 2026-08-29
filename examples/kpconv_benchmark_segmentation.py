"""Benchmark the KP-FCNN S3DIS semantic-segmentation models on Area 5 with the reference voting protocol.

Results (Area-5 mIoU):

    | Variant                                   | radius | reference | torch-pointcloud |
    | ----------------------------------------- | ------ | --------- | ---------------- |
    | kpfcnn-base.s3dis.hugues-thomas           | 1.8    | 66.4      | 66.44            |
    | kpfcnn-base-sm.s3dis.hugues-thomas        | 1.2    | 65.4      | 65.39            |
    | kpfcnn-base-deform.s3dis.hugues-thomas    | 1.5    | 67.3      | 67.02            |
    | kpfcnn-base-sm-deform.s3dis.hugues-thomas | 1.2    | 66.7      | 66.12            |

Usage:
    uv run --no-sync python examples/kpconv_benchmark_segmentation.py --model kpfcnn-base.s3dis.hugues-thomas
    uv run --no-sync python examples/kpconv_benchmark_segmentation.py --limit 2
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
from torch_pointcloud.datasets import S3DIS
from torch_pointcloud.inferers import Inferer, PotentialSphereInferer
from torch_pointcloud.models import create_model
from torch_pointcloud.utils.data import DataKeys, PointCloudDataLoader
from torch_pointcloud.utils.metrics import confusion_matrix
from torch_pointcloud.utils.random import seed_everything, set_determinism

CUDA_AVAILABLE = torch.cuda.is_available()
CPU_COUNT = os.cpu_count()
DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"
NUM_WORKERS = CPU_COUNT // 2 if CPU_COUNT is not None else 0
SEED = 42
SPHERE_RADIUS = {
    "kpfcnn-base.s3dis.hugues-thomas": 1.8,
    "kpfcnn-base-sm.s3dis.hugues-thomas": 1.2,
    "kpfcnn-base-deform.s3dis.hugues-thomas": 1.5,
    "kpfcnn-base-sm-deform.s3dis.hugues-thomas": 1.2,
}

INFERER_TRANSFORM = T.Compose(
    [
        T.RandomRotate(keys=DataKeys.POS, angle_range=(-180.0, 180.0), axis=2, p=1.0),
        T.RandomScale(keys=DataKeys.POS, scale_range=(0.9, 1.1), anisotropic=True, p=1.0),
        T.RandomFlip(keys=DataKeys.POS, axes=[0], p=0.5),
        T.RandomJitter(keys=DataKeys.POS, sigma=0.001, clip=0.005),
    ]
)


def build_inferer(radius: float, seed: int) -> Inferer:
    return PotentialSphereInferer(
        radius=radius,
        num_votes=10.0,
        inner_ratio=0.7,
        ema_smoothing=0.95,
        transform=INFERER_TRANSFORM,
        seed=seed,
    )


@torch.no_grad()
def evaluate(model: Module, dataloader: DataLoader, inferer: Inferer, device: str, num_classes: int) -> Dict[str, Any]:
    model.to(device).eval()
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)

    pbar = tqdm(dataloader, total=len(dataloader), desc="Testing")
    for data in pbar:
        data = {key: value.to(device) if torch.is_tensor(value) else value for key, value in data.items()}
        probs = inferer(data, predictor=lambda d: model(d[DataKeys.X], d[DataKeys.POS], d[DataKeys.BATCH]))
        preds = probs.argmax(dim=1)[data[DataKeys.INVERSE]]
        cm += confusion_matrix(preds.cpu(), data[DataKeys.ORIGIN_SEGMENT].cpu(), num_classes, ignore_index=-1)
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
    parser = argparse.ArgumentParser(description="Benchmark KP-FCNN semantic segmentation on S3DIS Area 5.")
    parser.add_argument("--model", default="kpfcnn-base.s3dis.hugues-thomas", help="Registered segmentation model name")
    parser.add_argument("--radius", default=None, type=float, help="Sphere radius (default: the checkpoint's).")
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--root", default=DATA_DIR, help="Dataset root directory.")
    parser.add_argument("--areas", nargs="+", default=["Area_5"])
    parser.add_argument("--seed", default=SEED, type=int)
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
    radius = args.radius if args.radius is not None else SPHERE_RADIUS[args.model]
    inferer = build_inferer(radius, args.seed)

    dataset: Dataset = S3DIS(
        root=args.root,
        areas=args.areas,
        transform=model_info["transform"],
        download=args.download,
        force_process=args.force_process,
        num_workers=args.num_workers,
    )
    if args.limit is not None:
        n = min(int(args.limit), len(dataset))  # type: ignore[arg-type]
        dataset = Subset(dataset, range(n))
        print(f"Evaluating on a subset of the first {n} rooms.")

    dataloader = PointCloudDataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    print(f"Test set: {len(dataset)} rooms  (sphere voting, radius {radius}, scored at full resolution)")  # type: ignore[arg-type]
    metrics = evaluate(model, dataloader, inferer, args.device, num_classes)
    print("\nResults:")
    for key, value in metrics.items():
        print(f"  {key:<24} {value:.4f}")


if __name__ == "__main__":
    main()
