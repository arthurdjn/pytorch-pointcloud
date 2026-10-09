"""Benchmark the OctFormer ScanNet semantic-segmentation model on the val split (single pass)."""

import os
from argparse import ArgumentParser, Namespace
from typing import TYPE_CHECKING, Any, Dict

import torch
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from torch_pointcloud.config import DATA_DIR
from torch_pointcloud.datasets import ScanNet20
from torch_pointcloud.metrics import accuracy, confusion_matrix, intersection_over_union
from torch_pointcloud.models import SemanticSegmentationModel, create_model, model_info
from torch_pointcloud.utils.data import DataKeys, PointCloudDataLoader
from torch_pointcloud.utils.imports import _OCNN_GITHUB_URL, optional_import
from torch_pointcloud.utils.random import seed_everything, set_determinism

if TYPE_CHECKING:
    from ocnn.octree import Octree, Points

Octree, _ = optional_import("ocnn.octree", "Octree", url=_OCNN_GITHUB_URL)
Points, _ = optional_import("ocnn.octree", "Points", url=_OCNN_GITHUB_URL)

CUDA_AVAILABLE = torch.cuda.is_available()
CPU_COUNT = os.cpu_count()
DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"
NUM_WORKERS = CPU_COUNT // 2 if CPU_COUNT is not None else 0
BATCH_SIZE = 1
SEED = 42
MODELS = ["octformer-base.scannet20.octree-nn"]


def main() -> None:
    args = parse_args()

    print(f"Seeding everything to {args.seed}!")
    seed_everything(args.seed)
    set_determinism(tf32=False)
    print(f"Benchmarking model {args.model!r} on ScanNet (split={args.split!r})!")

    test_dataloader = configure_dataloader(args)
    model = configure_model(args)

    num_scenes = len(test_dataloader.dataset)  # type: ignore[arg-type]
    print(f"Test set: {num_scenes} scenes")
    print("Evaluating...")
    metrics = evaluate(model, test_dataloader, args.device, num_classes=int(model.num_classes))

    print("\nResults:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<20} {v:.4f}")
        else:
            print(f"  {k:<20} {v}")


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Benchmark OctFormer segmentation on ScanNet.")
    parser.add_argument("--force-process", action="store_true", help="Force re-processing the dataset.")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--root", type=str, default=DATA_DIR, help="Dataset root (contains ScanNet/).")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--model", type=str, default=MODELS[0], choices=MODELS)
    parser.add_argument("--download", action="store_true", help="Download ScanNet if missing.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--device", type=str, default=DEVICE)
    parser.add_argument("--limit", type=int, default=None, help="Evaluate at most this many scenes (debug).")
    return parser.parse_args()


def configure_dataset(args: Namespace) -> Dataset:
    transform = model_info(args.model, task="semantic-segmentation")["transform"]
    if transform is None:
        raise ValueError(
            f"Model {args.model!r} has no registered `transforms`. "
            "Use a registered ScanNet OctFormer checkpoint (e.g. 'octformer-base.scannet20.octree-nn')."
        )

    test_dataset: Dataset
    test_dataset = ScanNet20(
        root=args.root,
        split=args.split,
        download=args.download,
        transform=transform,
        force_process=args.force_process,
        num_workers=args.num_workers,
        use_axis_alignment=False,
    )

    if args.limit is not None:
        n = min(int(args.limit), len(test_dataset))
        test_dataset = Subset(test_dataset, range(n))
        print(f"Evaluating on a subset of the first {n} scenes.")

    return test_dataset


def configure_dataloader(args: Namespace) -> DataLoader:
    dataset = configure_dataset(args)
    return PointCloudDataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )


def configure_model(args: Namespace) -> SemanticSegmentationModel:
    return create_model(args.model, task="semantic-segmentation", pretrained=True)


@torch.no_grad()
def evaluate(
    model: Module,
    dataloader: DataLoader,
    device: str,
    *,
    num_classes: int,
    ignore_index: int = 0,
) -> Dict[str, Any]:
    model.to(device).eval()
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)

    pbar = tqdm(dataloader, total=len(dataloader), desc="Testing")
    for data in pbar:
        octree = data[DataKeys.OCTREE].to(device)
        pos = data[DataKeys.POS].to(device)
        batch = data[DataKeys.BATCH].to(device)
        x = data[DataKeys.X].to(device)
        target = data[DataKeys.SEGMENT].to(device)

        logits = model(x, octree, octree.depth, pos, batch)
        preds = logits.argmax(dim=1)

        cm += confusion_matrix(preds.cpu(), target.cpu(), num_classes, ignore_index=ignore_index)
        oa = accuracy(cm)
        pbar.set_postfix({"oa": f"{oa:.4f}"})

    return {
        "test/mIoU": intersection_over_union(cm, ignore_index=ignore_index),
        "test/oa": accuracy(cm),
        "test/iou_per_class": intersection_over_union(cm, average="none", ignore_index=ignore_index),
    }


if __name__ == "__main__":
    main()
