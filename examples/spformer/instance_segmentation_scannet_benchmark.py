"""Benchmark SPFormer on the ScanNet validation scenes with the ScanNet instance segmentation protocol."""

import os
from argparse import ArgumentParser, Namespace
from typing import Dict, List

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from torch_pointcloud.config import DATA_DIR
from torch_pointcloud.datasets import ScanNet20
from torch_pointcloud.metrics import instance_average_precision, instance_matches
from torch_pointcloud.metrics.instance_segmentation import InstanceMatches
from torch_pointcloud.models import SPFormerInstanceSegmentation, create_model, model_info
from torch_pointcloud.utils.data import DataKeys, PointCloudDataLoader
from torch_pointcloud.utils.random import seed_everything, set_determinism

CUDA_AVAILABLE = torch.cuda.is_available()
CPU_COUNT = os.cpu_count()
DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"
NUM_WORKERS = CPU_COUNT // 2 if CPU_COUNT is not None else 0
SEED = 42
SCORE_THRESHOLD = 0.0
MIN_POINTS = 100


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    set_determinism(tf32=False)

    print(f"Benchmarking model {args.model!r} on ScanNet (split={args.split!r})!")
    dataloader = configure_dataloader(args)
    model = configure_model(args)

    print(f"Test set: {len(dataloader.dataset)} scenes")  # type: ignore[arg-type]
    metrics = evaluate(model, dataloader, args.device)
    print("\nResults:")
    for key, value in metrics.items():
        print(f"  {key:<24} {value * 100:.2f}")


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Benchmark SPFormer instance segmentation on ScanNet.")
    parser.add_argument("--model", default="spformer.scannet.jiahao-sun", help="Registered model name")
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--root", default=DATA_DIR, help="Dataset root directory.")
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--seed", default=SEED, type=int)
    parser.add_argument("--num-workers", default=NUM_WORKERS, type=int)
    parser.add_argument("--limit", default=None, type=int, help="Evaluate at most this many scenes.")
    return parser.parse_args()


def configure_dataset(args: Namespace) -> Dataset:
    transform = model_info(args.model, task="instance-segmentation")["transform"]

    dataset: Dataset
    dataset = ScanNet20(
        root=args.root,
        split=args.split,
        use_axis_alignment=False,
        return_superpoint=True,
        transform=transform,
        num_workers=args.num_workers,
    )
    if args.limit is not None:
        n = min(int(args.limit), len(dataset))
        dataset = Subset(dataset, range(n))
        print(f"Evaluating on a subset of the first {n} scenes.")

    return dataset


def configure_dataloader(args: Namespace) -> DataLoader:
    dataset = configure_dataset(args)
    return PointCloudDataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        cat_keys=[DataKeys.INVERSE],
    )


def configure_model(args: Namespace) -> SPFormerInstanceSegmentation:
    model = create_model(args.model, task="instance-segmentation", pretrained=True)
    assert isinstance(model, SPFormerInstanceSegmentation)
    return model


@torch.no_grad()
def evaluate(model: SPFormerInstanceSegmentation, dataloader: DataLoader, device: str) -> Dict[str, float]:
    model.to(device).eval()
    matches: List[InstanceMatches] = []

    for data in tqdm(dataloader, total=len(dataloader), desc="Testing"):
        data = {key: value.to(device) if torch.is_tensor(value) else value for key, value in data.items()}
        output = model(
            data[DataKeys.X],
            data[DataKeys.POS_GRID],
            data[DataKeys.BATCH],
            data[DataKeys.SUPERPOINT],
            data[DataKeys.INVERSE],
            data[DataKeys.BATCH_INVERSE],
        )
        instances = model.decode(output)
        sizes = torch.bincount(instances["mask_index"][0], minlength=instances["labels"].shape[0])
        # SPFormer keeps masks of more than MIN_POINTS points, the protocol scores instances of at least MIN_POINTS.
        instances["ignore_mask"] = (instances["scores"] <= SCORE_THRESHOLD) | (sizes <= MIN_POINTS)
        match = instance_matches(
            instances,
            data[DataKeys.INSTANCE],
            data[DataKeys.SEGMENT],
            batch=data[DataKeys.BATCH_INVERSE],
        )
        matches.append(match)

    num_classes = model.num_classes
    return {
        "mAP": instance_average_precision(matches, num_classes=num_classes, min_points=MIN_POINTS),
        "mAP@0.5": instance_average_precision(
            matches,
            iou_threshold=0.5,
            num_classes=num_classes,
            min_points=MIN_POINTS,
        ),
        "mAP@0.25": instance_average_precision(
            matches,
            iou_threshold=0.25,
            num_classes=num_classes,
            min_points=MIN_POINTS,
        ),
    }


if __name__ == "__main__":
    main()
