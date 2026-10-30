"""Benchmark OpenScene zero-shot segmentation on nuScenes-lidarseg with the authors' five randomly augmented passes."""

import os
from argparse import ArgumentParser, Namespace
from typing import Any, Dict

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

import torch_pointcloud.transforms as T
from torch_pointcloud.config import DATA_DIR
from torch_pointcloud.datasets import NuScenes, NuScenesMini
from torch_pointcloud.datasets.nuscenes import NUSCENES_LIDARSEG_CLASSES, NUSCENES_LIDARSEG_LEARNING_MAP
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
VOXEL_SIZE = 0.05
NUM_PASSES = 5
# The authors prompt 43 detailed names and read each one as its lidarseg class, `NUSCENES_LIDARSEG_CLASSES[i]`.
CLASSES = {
    "barrier": 0,
    "barricade": 0,
    "bicycle": 1,
    "bus": 2,
    "car": 3,
    "bulldozer": 4,
    "excavator": 4,
    "concrete mixer": 4,
    "crane": 4,
    "dump truck": 4,
    "motorcycle": 5,
    "person": 6,
    "pedestrian": 6,
    "traffic cone": 7,
    "trailer": 8,
    "semi trailer": 8,
    "cargo container": 8,
    "shipping container": 8,
    "freight container": 8,
    "truck": 9,
    "road": 10,
    "curb": 11,
    "traffic island": 11,
    "traffic median": 11,
    "sidewalk": 12,
    "grass": 13,
    "grassland": 13,
    "lawn": 13,
    "meadow": 13,
    "turf": 13,
    "sod": 13,
    "building": 14,
    "wall": 14,
    "pole": 14,
    "awning": 14,
    "tree": 15,
    "trunk": 15,
    "tree trunk": 15,
    "bush": 15,
    "shrub": 15,
    "plant": 15,
    "flower": 15,
    "woods": 15,
}

TRANSFORM = T.Compose([T.Relabel(keys=DataKeys.SEGMENT, labels=NUSCENES_LIDARSEG_LEARNING_MAP, default=-1)])
INFERER_TRANSFORM = T.Voxelize(
    pos_key=DataKeys.POS,
    pos_reduce="first",
    size=VOXEL_SIZE,
    dst_pos_grid_key=DataKeys.POS_GRID,
    dst_inverse_key=DataKeys.INVERSE,
)
# The authors' evaluation voxelizes each pass after a random rotation and scale of the scan.
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

    print(f"Benchmarking model {args.model!r} on nuScenes-lidarseg!")
    model, info = create_model(args.model, task="open-vocabulary-segmentation", pretrained=True, return_info=True)
    weights = info["weights"]
    assert weights is not None
    text_encoder = CLIPTextEncoder(weights["text_model"])
    prompts = [weights["prompt_template"].format(name) for name in CLASSES]
    inferer = configure_inferer(args)
    dataloader = configure_dataloader(args)

    num_scans = len(dataloader.dataset)  # type: ignore[arg-type]
    protocol = "single pass" if args.simple else f"{NUM_PASSES} randomly rotated and scaled passes"
    print(f"Test set: {num_scans} scans of {args.max_sweeps} sweeps  ({protocol}, scored on the keyframe points)")
    metrics = evaluate(model, text_encoder, prompts, dataloader, inferer, args.device)
    print("\nResults:")
    for key, value in metrics.items():
        print(f"  {key:<24} {value:.4f}")


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Benchmark OpenScene zero-shot segmentation on nuScenes-lidarseg.")
    parser.add_argument("--model", default="openscene-openseg.nuscenes.songyou-peng", help="Registered model name")
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--root", default=DATA_DIR, help="Dataset root directory.")
    parser.add_argument("--split", default="val", choices=("val", "mini"), help="nuScenes `val`, or the mini release.")
    parser.add_argument("--max-sweeps", default=10, type=int, help="LiDAR sweeps per scan, the authors' 0.5 s.")
    parser.add_argument("--seed", default=SEED, type=int)
    parser.add_argument("--num-workers", default=NUM_WORKERS, type=int)
    parser.add_argument("--limit", default=None, type=int, help="Evaluate at most this many scans.")
    parser.add_argument(
        "--simple",
        action="store_true",
        help="Predict each point in one pass, without test-time augmentation.",
    )
    return parser.parse_args()


def configure_dataset(args: Namespace) -> Dataset:
    dataset: Dataset
    if args.split == "mini":
        dataset = NuScenesMini(
            root=args.root,
            max_sweeps=args.max_sweeps,
            return_boxes=False,
            return_segment=True,
            transform=TRANSFORM,
        )
    else:
        dataset = NuScenes(
            root=args.root,
            split="val",
            max_sweeps=args.max_sweeps,
            return_boxes=False,
            return_segment=True,
            transform=TRANSFORM,
        )

    if args.limit is not None:
        n = min(int(args.limit), len(dataset))
        dataset = Subset(dataset, range(n))
        print(f"Evaluating on a subset of the first {n} scans.")
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
    prompt_to_class = torch.tensor(list(CLASSES.values()), device=device)
    num_classes = len(NUSCENES_LIDARSEG_CLASSES)
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)

    def predictor(d: Dict[str, Any]) -> Tensor:
        embeddings = model(None, d[DataKeys.POS_GRID], d[DataKeys.BATCH])
        return embeddings @ text.t()

    pbar = tqdm(dataloader, total=len(dataloader), desc="Testing")
    for data in pbar:
        data = {key: value.to(device) if torch.is_tensor(value) else value for key, value in data.items()}
        logits = inferer(data, predictor=predictor)
        preds = prompt_to_class[logits.argmax(dim=1)]
        cm += confusion_matrix(preds.cpu(), data[DataKeys.SEGMENT].cpu(), num_classes, ignore_index=-1)
        pbar.set_postfix({"mIoU": f"{intersection_over_union(cm):.4f}"})

    return {
        "test/mIoU": intersection_over_union(cm),
        "test/mAcc": accuracy(cm, average="macro"),
        "test/oa": accuracy(cm),
    }


if __name__ == "__main__":
    main()
