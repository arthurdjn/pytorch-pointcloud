"""Benchmark OpenScene zero-shot segmentation on ScanNet with the authors' five randomly augmented passes."""

import os
from argparse import ArgumentParser, Namespace
from typing import Any, Dict

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

import torch_pointcloud.transforms as T
from torch_pointcloud.config import DATA_DIR
from torch_pointcloud.datasets import ScanNet20
from torch_pointcloud.inferers import Inferer, SimpleInferer, TTAInferer
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
VOXEL_SIZE = 0.02
NUM_PASSES = 5
# The ScanNet20 classes as the authors prompt them: "shower curtain" with a space and the bare "other" for
# otherfurniture, so `SCANNET20_CLASSES` is not reused.
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

TRANSFORM = T.Compose([T.Relabel(keys=DataKeys.SEGMENT, labels=range(1, 21), default=-1)])
INFERER_TRANSFORM = T.Voxelize(
    pos_key=DataKeys.POS,
    pos_reduce="first",
    size=VOXEL_SIZE,
    dst_pos_grid_key=DataKeys.POS_GRID,
    dst_inverse_key=DataKeys.INVERSE,
)
# The authors' evaluation voxelizes each pass after a random rotation and scale of the scene.
TTA_TRANSFORM = T.Compose(
    [
        T.RandomRotate(keys=DataKeys.POS, angle_range=(-180.0 / 64, 180.0 / 64), axis=0),
        T.RandomRotate(keys=DataKeys.POS, angle_range=(-180.0 / 64, 180.0 / 64), axis=1),
        T.RandomRotate(keys=DataKeys.POS, angle_range=(-180.0, 180.0), axis=2),
        T.RandomScale(keys=DataKeys.POS, scale_range=(0.9, 1.1)),
    ]
)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    set_determinism(tf32=False)

    print(f"Benchmarking model {args.model!r} on ScanNet (split={args.split!r})!")
    model, info = create_model(args.model, task="open-vocabulary-segmentation", pretrained=True, return_info=True)
    weights = info["weights"]
    assert weights is not None
    text_encoder = CLIPTextEncoder(weights["text_model"], revision=weights["text_revision"])
    prompts = [weights["prompt_template"].format(name) for name in CLASSES] + ["other"]
    inferer = configure_inferer(args)
    dataloader = configure_dataloader(args)

    num_scenes = len(dataloader.dataset)  # type: ignore[arg-type]
    protocol = "single pass" if args.simple else f"{NUM_PASSES} randomly rotated and scaled passes"
    print(f"Test set: {num_scenes} scenes  ({protocol}, scored at full resolution)")
    metrics = evaluate(model, text_encoder, prompts, dataloader, inferer, args.device)
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
    parser.add_argument(
        "--simple",
        action="store_true",
        help="Predict each point in one pass, without test-time augmentation.",
    )
    return parser.parse_args()


def configure_dataset(args: Namespace) -> Dataset:
    dataset: Dataset
    dataset = ScanNet20(
        root=args.root,
        split=args.split,
        transform=TRANSFORM,
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


def configure_dataloader(args: Namespace) -> DataLoader:
    dataset = configure_dataset(args)
    return PointCloudDataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)


def configure_inferer(args: Namespace) -> Inferer:
    base = SimpleInferer(transform=INFERER_TRANSFORM, inverse_key=DataKeys.INVERSE)
    if args.simple:
        return base
    return TTAInferer(base=base, transforms=TTA_TRANSFORM, num_passes=NUM_PASSES, aggregate="mean")


@torch.no_grad()
def evaluate(
    model: OpenVocabularySegmentationModel,
    text_encoder: CLIPTextEncoder,
    prompts: list[str],
    dataloader: DataLoader,
    inferer: Inferer,
    device: str,
) -> Dict[str, Any]:
    model.to(device).eval()
    text = text_encoder.to(device)(prompts)
    num_classes = len(prompts)
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)

    def predictor(d: Dict[str, Any]) -> Tensor:
        embeddings = model(None, d[DataKeys.POS_GRID], d[DataKeys.BATCH])
        return embeddings @ text.t()

    pbar = tqdm(dataloader, total=len(dataloader), desc="Testing")
    for data in pbar:
        data = {key: value.to(device) if torch.is_tensor(value) else value for key, value in data.items()}
        logits = inferer(data, predictor=predictor)
        preds = logits.argmax(dim=1)
        cm += confusion_matrix(preds.cpu(), data[DataKeys.SEGMENT].cpu(), num_classes, ignore_index=-1)
        pbar.set_postfix({"mIoU": f"{intersection_over_union(cm):.4f}"})

    return {
        "test/mIoU": intersection_over_union(cm),
        "test/mAcc": accuracy(cm, average="macro"),
        "test/oa": accuracy(cm),
    }


if __name__ == "__main__":
    main()
