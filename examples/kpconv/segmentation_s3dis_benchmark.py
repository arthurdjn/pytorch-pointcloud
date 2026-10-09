"""Benchmark the KP-FCNN S3DIS semantic-segmentation models on Area 5 with the sphere-voting protocol."""

import os
from argparse import ArgumentParser, Namespace
from typing import Any, Dict

import torch
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

import torch_pointcloud.transforms as T
from torch_pointcloud.config import DATA_DIR
from torch_pointcloud.datasets import S3DIS
from torch_pointcloud.inferers import Inferer, PotentialSphereInferer
from torch_pointcloud.metrics import accuracy, confusion_matrix, intersection_over_union
from torch_pointcloud.models import SemanticSegmentationModel, create_model, model_info
from torch_pointcloud.utils.data import DataKeys, PointCloudDataLoader
from torch_pointcloud.utils.random import seed_everything, set_determinism

CUDA_AVAILABLE = torch.cuda.is_available()
CPU_COUNT = os.cpu_count()
DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"
NUM_WORKERS = CPU_COUNT // 2 if CPU_COUNT is not None else 0
SEED = 42
SW_BATCH_SIZE = 8
SPHERE_RADIUS = {
    "kpfcnn-base.s3dis-area5.hugues-thomas": 1.8,
    "kpfcnn-base-sm.s3dis-area5.hugues-thomas": 1.2,
    "kpfcnn-base-deform.s3dis-area5.hugues-thomas": 1.5,
    "kpfcnn-base-sm-deform.s3dis-area5.hugues-thomas": 1.2,
}

INFERER_TRANSFORM = T.Compose(
    [
        T.RandomRotate(keys=DataKeys.POS, angle_range=(-180.0, 180.0), axis=2, p=1.0),
        T.RandomScale(keys=DataKeys.POS, scale_range=(0.9, 1.1), anisotropic=True, p=1.0),
        T.RandomFlip(keys=DataKeys.POS, axes=[0], p=0.5),
        T.RandomJitter(keys=DataKeys.POS, sigma=0.001, clip=0.005),
    ]
)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    set_determinism(tf32=False)

    print(f"Benchmarking model {args.model!r} on S3DIS (areas={args.areas})!")
    dataloader = configure_dataloader(args)
    model = configure_model(args)
    inferer = configure_inferer(args)

    num_rooms = len(dataloader.dataset)  # type: ignore[arg-type]
    print(f"Test set: {num_rooms} rooms  (sphere voting, radius {args.radius}, scored at full resolution)")
    metrics = evaluate(model, dataloader, inferer, args.device, int(model.num_classes))
    print("\nResults:")
    for key, value in metrics.items():
        print(f"  {key:<24} {value:.4f}")


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Benchmark KP-FCNN semantic segmentation on S3DIS Area 5.")
    parser.add_argument(
        "--model", default="kpfcnn-base.s3dis-area5.hugues-thomas", help="Registered segmentation model name"
    )
    parser.add_argument("--radius", default=None, type=float, help="Sphere radius (default: the checkpoint's).")
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--root", default=DATA_DIR, help="Dataset root directory.")
    parser.add_argument("--areas", nargs="+", default=["Area_5"])
    parser.add_argument("--seed", default=SEED, type=int)
    parser.add_argument("--sw-batch-size", default=SW_BATCH_SIZE, type=int, help="Spheres per forward.")
    parser.add_argument("--num-workers", default=NUM_WORKERS, type=int)
    parser.add_argument("--limit", default=None, type=int, help="Evaluate at most this many rooms.")
    parser.add_argument("--download", action="store_true", help="Download S3DIS if missing.")
    parser.add_argument("--force-process", action="store_true", help="Force re-processing the dataset.")
    args = parser.parse_args()
    if args.radius is None:
        args.radius = SPHERE_RADIUS[args.model]
    return args


def configure_dataset(args: Namespace) -> Dataset:
    transform = model_info(args.model, task="semantic-segmentation")["transform"]

    dataset: Dataset
    dataset = S3DIS(
        root=args.root,
        areas=args.areas,
        transform=transform,
        download=args.download,
        force_process=args.force_process,
        num_workers=args.num_workers,
    )
    if args.limit is not None:
        n = min(int(args.limit), len(dataset))
        dataset = Subset(dataset, range(n))
        print(f"Evaluating on a subset of the first {n} rooms.")

    return dataset


def configure_dataloader(args: Namespace) -> DataLoader:
    dataset = configure_dataset(args)
    return PointCloudDataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)


def configure_model(args: Namespace) -> SemanticSegmentationModel:
    return create_model(args.model, task="semantic-segmentation", pretrained=True)


def configure_inferer(args: Namespace) -> Inferer:
    return PotentialSphereInferer(
        radius=args.radius,
        num_votes=10.0,
        inner_ratio=0.7,
        ema_smoothing=0.95,
        transform=INFERER_TRANSFORM,
        sw_batch_size=args.sw_batch_size,
        seed=args.seed,
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
        oa = accuracy(cm)
        pbar.set_postfix({"oa": f"{oa:.4f}"})

    return {
        "test/mIoU": intersection_over_union(cm),
        "test/oa": accuracy(cm),
    }


if __name__ == "__main__":
    main()
