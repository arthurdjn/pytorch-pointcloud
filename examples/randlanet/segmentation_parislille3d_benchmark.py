"""Benchmark the RandLA-Net Paris-Lille-3D model on the held-out tile with Open3D-ML's kNN-window voting protocol."""

import os
from argparse import ArgumentParser, Namespace
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import torch_pointcloud.transforms as T
from torch_pointcloud.config import DATA_DIR
from torch_pointcloud.datasets import ParisLille3D
from torch_pointcloud.inferers import Inferer, KNNWindowInferer
from torch_pointcloud.metrics import accuracy, confusion_matrix, intersection_over_union
from torch_pointcloud.models import SemanticSegmentationModel, create_model, model_info
from torch_pointcloud.utils.data import DataKeys, PointCloudDataLoader
from torch_pointcloud.utils.random import seed_everything, set_determinism

CUDA_AVAILABLE = torch.cuda.is_available()
CPU_COUNT = os.cpu_count()
DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"
NUM_WORKERS = CPU_COUNT // 2 if CPU_COUNT is not None else 0
SEED = 42
WINDOW_NUM_POINTS = 65536
IGNORE_INDEX = -1
# Every window is recentred on its x/y centroid; the coordinates are the only input feature.
WINDOW_TRANSFORM = T.Compose(
    [
        T.Shift(keys=DataKeys.POS, method="centroid", axes=[0, 1]),
        T.CopyItems(keys=DataKeys.POS, dst_keys=DataKeys.X),
    ]
)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    set_determinism(tf32=False)

    print(f"Benchmarking model {args.model!r} on Paris-Lille-3D (split={args.split!r})!")
    dataloader = configure_dataloader(args)
    model = configure_model(args)
    inferer = configure_inferer(args)

    num_tiles = len(dataloader.dataset)  # type: ignore[arg-type]
    print(f"Test set: {num_tiles} tiles  (KNN windows with EMA voting, scored at full resolution)")
    metrics = evaluate(model, dataloader, inferer, args.device, int(model.num_classes))
    print("\nResults:")
    for key, value in metrics.items():
        print(f"  {key:<24} {value:.4f}")


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Benchmark RandLA-Net semantic segmentation on Paris-Lille-3D.")
    parser.add_argument("--model", default="randlanet.parislille3d.open3d", help="Registered segmentation model name")
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--root", default=DATA_DIR, help="Dataset root directory.")
    parser.add_argument("--split", default="val", choices=["train", "val", "trainval", "all"])
    parser.add_argument("--seed", default=SEED, type=int)
    parser.add_argument("--num-workers", default=NUM_WORKERS, type=int)
    return parser.parse_args()


def configure_dataset(args: Namespace) -> Dataset:
    transform = model_info(args.model, task="semantic-segmentation")["transform"]
    return ParisLille3D(root=args.root, split=args.split, transform=transform)


def configure_dataloader(args: Namespace) -> DataLoader:
    dataset = configure_dataset(args)
    return PointCloudDataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)


def configure_model(args: Namespace) -> SemanticSegmentationModel:
    return create_model(args.model, task="semantic-segmentation", pretrained=True)


def configure_inferer(args: Namespace) -> Inferer:
    return KNNWindowInferer(
        roi_num_points=WINDOW_NUM_POINTS,
        overlap=0.5,
        aggregate="ema",
        ema_smoothing=0.95,
        transform=WINDOW_TRANSFORM,
        seed=args.seed,
    )


@torch.no_grad()
def evaluate(
    model: SemanticSegmentationModel, dataloader: DataLoader, inferer: Inferer, device: str, num_classes: int
) -> Dict[str, Any]:
    model.to(device).eval()
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)

    pbar = tqdm(dataloader, total=len(dataloader), desc="Testing")
    for data in pbar:
        data = {key: value.to(device) if torch.is_tensor(value) else value for key, value in data.items()}
        scores = inferer(data, predictor=lambda d: model(d[DataKeys.X], d[DataKeys.POS], d[DataKeys.BATCH]))
        preds = scores.argmax(dim=1)[data[DataKeys.INVERSE]]
        cm += confusion_matrix(preds.cpu(), data[DataKeys.ORIGIN_SEGMENT].cpu(), num_classes, ignore_index=IGNORE_INDEX)
        oa = accuracy(cm)
        pbar.set_postfix({"oa": f"{oa:.4f}"})

    return {
        "test/mIoU": intersection_over_union(cm),
        "test/oa": accuracy(cm),
    }


if __name__ == "__main__":
    main()
