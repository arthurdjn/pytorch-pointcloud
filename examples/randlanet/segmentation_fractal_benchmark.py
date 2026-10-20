"""Benchmark RandLA-Net semantic segmentation on FRACTAL, one pass per patch, scored at full resolution."""

import os
from argparse import ArgumentParser, Namespace
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torch_geometric.utils import scatter
from tqdm import tqdm

import torch_pointcloud.transforms as T
from torch_pointcloud.config import DATA_DIR
from torch_pointcloud.datasets import FRACTAL
from torch_pointcloud.datasets.fractal import FRACTAL_CLASSES
from torch_pointcloud.inferers import Inferer, SimpleInferer
from torch_pointcloud.metrics import accuracy, confusion_matrix, intersection_over_union
from torch_pointcloud.models import SemanticSegmentationModel, create_model, model_info
from torch_pointcloud.ops.cluster import knn_interpolate
from torch_pointcloud.utils.data import DataKeys, PointCloudDataLoader
from torch_pointcloud.utils.random import seed_everything, set_determinism

CUDA_AVAILABLE = torch.cuda.is_available()
CPU_COUNT = os.cpu_count()
DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"
NUM_WORKERS = CPU_COUNT // 2 if CPU_COUNT is not None else 0
SEED = 42
IGNORE_INDEX = -1
# Myria3D scores every point by inverse squared distance interpolation from its 10 nearest predicted points.
INTERPOLATION_K = 10


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    set_determinism(tf32=False)

    print(f"Benchmarking model {args.model!r} on FRACTAL!")
    model = configure_model(args)
    num_classes = int(model.num_classes)
    inferer = configure_inferer(args)
    dataloader = configure_dataloader(args)

    num_patches = len(dataloader.dataset)  # type: ignore[arg-type]
    print(f"{args.split.capitalize()} set: {num_patches} patches  (single pass, scored at full resolution)")
    metrics = evaluate(model, dataloader, inferer, args.device, num_classes)
    print("\nResults:")
    for key, value in metrics.items():
        print(f"  {key:<24} {value:.4f}")


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Benchmark RandLA-Net semantic segmentation on FRACTAL.")
    parser.add_argument("--model", default="randlanet.fractal.ignf", help="Registered segmentation model name")
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--root", default=DATA_DIR, help="Dataset root directory.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--download", action="store_true", help="Download the archives of the split (9 GB).")
    parser.add_argument("--seed", default=SEED, type=int)
    parser.add_argument("--num-workers", default=NUM_WORKERS, type=int)
    parser.add_argument("--limit", default=None, type=int, help="Evaluate at most this many patches.")
    return parser.parse_args()


def configure_transform(args: Namespace) -> T.Transform:
    """The registered inference transform of the model."""
    transform = model_info(args.model, task="semantic-segmentation")["transform"]
    if transform is None:
        raise ValueError(f"{args.model!r} registers no transform.")

    return transform


def configure_dataset(args: Namespace) -> Dataset:
    dataset: Dataset
    transform = configure_transform(args)
    dataset = FRACTAL(root=args.root, split=args.split, transform=transform, download=args.download)
    if args.limit is not None:
        n = min(int(args.limit), len(dataset))
        dataset = Subset(dataset, range(n))
        print(f"Evaluating on a subset of the first {n} patches.")

    return dataset


def configure_dataloader(args: Namespace) -> DataLoader:
    dataset = configure_dataset(args)
    return PointCloudDataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)


def configure_model(args: Namespace) -> SemanticSegmentationModel:
    return create_model(args.model, task="semantic-segmentation", pretrained=True)


def configure_inferer(args: Namespace) -> Inferer:
    return SimpleInferer()


@torch.no_grad()
def evaluate(
    model: SemanticSegmentationModel, dataloader: DataLoader, inferer: Inferer, device: str, num_classes: int
) -> dict[str, Any]:
    model.to(device).eval()
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)

    pbar = tqdm(dataloader, total=len(dataloader), desc="Testing")
    for data in pbar:
        data = {key: value.to(device) if torch.is_tensor(value) else value for key, value in data.items()}
        scores = inferer(data, predictor=lambda d: model(d[DataKeys.X], d[DataKeys.POS], d[DataKeys.BATCH]))

        # The predicted points are the sampled voxels, at the mean position of their source points.
        voxel_pos = scatter(data[DataKeys.ORIGIN_POS], data[DataKeys.INVERSE], dim=0, reduce="mean")
        scores = knn_interpolate(scores, voxel_pos[data[DataKeys.INDEX]], data[DataKeys.ORIGIN_POS], k=INTERPOLATION_K)
        preds = scores.argmax(dim=1)
        cm += confusion_matrix(preds.cpu(), data[DataKeys.ORIGIN_SEGMENT].cpu(), num_classes, ignore_index=IGNORE_INDEX)
        pbar.set_postfix({"oa": f"{accuracy(cm):.4f}"})

    per_class = intersection_over_union(cm, average="none", class_names=FRACTAL_CLASSES)
    return {
        "test/mIoU": intersection_over_union(cm),
        "test/oa": accuracy(cm),
        **{f"test/IoU/{name}": iou for name, iou in per_class.items()},
    }


if __name__ == "__main__":
    main()
