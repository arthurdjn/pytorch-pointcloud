"""Benchmark OpenScene zero-shot segmentation on ScanNet: one pass per scene, each point labelled by its nearest prompt."""

import os
from argparse import ArgumentParser, Namespace
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from torch_pointcloud.config import DATA_DIR
from torch_pointcloud.datasets import ScanNet20
from torch_pointcloud.layers import CLIPTextEncoder
from torch_pointcloud.metrics import accuracy, confusion_matrix, intersection_over_union
from torch_pointcloud.models import OpenVocabularySegmentationModel, create_model
from torch_pointcloud.utils.data import DataKeys, PointCloudDataLoader
from torch_pointcloud.utils.random import seed_everything, set_determinism

CUDA_AVAILABLE = torch.cuda.is_available()
CPU_COUNT = os.cpu_count()
DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"
NUM_WORKERS = CPU_COUNT // 2 if CPU_COUNT is not None else 0
SEED = 42
# The benchmark classes as OpenScene prompts them: "a {class} in a scene", and the bare "other" for otherfurniture.
CLASSES = (
    "wall",
    "floor",
    "cabinet",
    "bed",
    "chair",
    "sofa",
    "table",
    "door",
    "window",
    "bookshelf",
    "picture",
    "counter",
    "desk",
    "curtain",
    "refrigerator",
    "shower curtain",
    "toilet",
    "sink",
    "bathtub",
)
PROMPTS = [f"a {name} in a scene" for name in CLASSES] + ["other"]


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    set_determinism(tf32=False)

    print(f"Benchmarking model {args.model!r} on ScanNet (split={args.split!r})!")
    model, info = create_model(args.model, task="open-vocabulary-segmentation", pretrained=True, return_info=True)
    text_encoder = CLIPTextEncoder(model.text_model)
    dataloader = configure_dataloader(args, info["transform"])

    num_scenes = len(dataloader.dataset)  # type: ignore[arg-type]
    print(f"Test set: {num_scenes} scenes  (one pass per scene, scored at full resolution)")
    metrics = evaluate(model, text_encoder, dataloader, args.device)
    print("\nResults:")
    for key, value in metrics.items():
        print(f"  {key:<24} {value:.4f}")


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Benchmark OpenScene zero-shot segmentation on ScanNet.")
    parser.add_argument("--model", default="openscene-openseg.scannet20.songyou-peng", help="Registered model name")
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--root", default=DATA_DIR, help="Dataset root directory.")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--seed", default=SEED, type=int)
    parser.add_argument("--num-workers", default=NUM_WORKERS, type=int)
    parser.add_argument("--limit", default=None, type=int, help="Evaluate at most this many scenes.")
    parser.add_argument("--download", action="store_true", help="Download ScanNet if missing.")
    parser.add_argument("--force-process", action="store_true", help="Force re-processing the dataset.")
    return parser.parse_args()


def configure_dataset(args: Namespace, transform: Any) -> Dataset:
    dataset: Dataset
    dataset = ScanNet20(
        root=args.root,
        split=args.split,
        transform=transform,
        download=args.download,
        force_process=args.force_process,
        num_workers=args.num_workers,
        use_axis_alignment=False,
    )
    if args.limit is not None:
        n = min(int(args.limit), len(dataset))
        dataset = Subset(dataset, range(n))
        print(f"Evaluating on a subset of the first {n} scenes.")

    return dataset


def configure_dataloader(args: Namespace, transform: Any) -> DataLoader:
    dataset = configure_dataset(args, transform)
    return PointCloudDataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)


@torch.no_grad()
def evaluate(
    model: OpenVocabularySegmentationModel,
    text_encoder: CLIPTextEncoder,
    dataloader: DataLoader,
    device: str,
) -> dict[str, Any]:
    model.to(device).eval()
    text = text_encoder.to(device)(PROMPTS)
    num_classes = len(PROMPTS)
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)

    pbar = tqdm(dataloader, total=len(dataloader), desc="Testing")
    for data in pbar:
        data = {key: value.to(device) if torch.is_tensor(value) else value for key, value in data.items()}
        embeddings = model(data.get(DataKeys.X), data[DataKeys.POS_GRID], data[DataKeys.BATCH])
        logits = F.normalize(embeddings, dim=1) @ text.t()
        preds = logits.argmax(dim=1)[data[DataKeys.INVERSE]]
        cm += confusion_matrix(preds.cpu(), data[DataKeys.ORIGIN_SEGMENT].cpu(), num_classes, ignore_index=-1)
        pbar.set_postfix({"mIoU": f"{intersection_over_union(cm):.4f}"})

    return {
        "test/mIoU": intersection_over_union(cm),
        "test/mAcc": accuracy(cm, average="macro"),
        "test/oa": accuracy(cm),
    }


if __name__ == "__main__":
    main()
