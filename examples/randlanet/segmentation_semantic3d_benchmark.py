"""Benchmark the RandLA-Net Semantic3D model on Open3D-ML's two held-out scenes with its kNN-window voting protocol."""

import os
from argparse import ArgumentParser, Namespace
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import torch_pointcloud.transforms as T
from torch_pointcloud.config import DATA_DIR
from torch_pointcloud.datasets import Semantic3D
from torch_pointcloud.inferers import Inferer, KNNWindowSegmentationInferer
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
# The benchmark test labels are not public; these two training scenes are the validation split the checkpoint was
# trained against.
VAL_SCENES = ["bildstein_station1_xyz_intensity_rgb", "domfountain_station1_xyz_intensity_rgb"]
# Every window is recentred on its x/y centroid before the colors are appended.
WINDOW_TRANSFORM = T.Compose(
    [
        T.Shift(keys=DataKeys.POS, method="centroid", axes=[0, 1]),
        T.Cat(keys=[DataKeys.POS, DataKeys.COLOR], dst_key=DataKeys.X, dim=1),
    ]
)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    set_determinism(tf32=False)

    print(f"Benchmarking model {args.model!r} on Semantic3D (scenes={args.scenes})!")
    dataloader = configure_dataloader(args)
    model = configure_model(args)
    inferer = configure_inferer(args)

    num_scenes = len(dataloader.dataset)  # type: ignore[arg-type]
    print(f"Test set: {num_scenes} scenes  (KNN windows with EMA voting, scored at full resolution)")
    metrics = evaluate(model, dataloader, inferer, args.device, int(model.num_classes))
    print("\nResults:")
    for key, value in metrics.items():
        print(f"  {key:<24} {value:.4f}")


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Benchmark RandLA-Net semantic segmentation on Semantic3D.")
    parser.add_argument("--model", default="randlanet.semantic3d.open3d", help="Registered segmentation model name")
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--root", default=DATA_DIR, help="Dataset root directory.")
    parser.add_argument("--scenes", nargs="+", default=VAL_SCENES, help="Labelled scenes to score.")
    parser.add_argument("--seed", default=SEED, type=int)
    parser.add_argument("--num-workers", default=NUM_WORKERS, type=int)
    parser.add_argument(
        "--download", action="store_true", help="Download the scenes if missing (asks to accept the licence)."
    )
    return parser.parse_args()


def configure_dataset(args: Namespace) -> Dataset:
    transform = model_info(args.model, task="semantic-segmentation")["transform"]
    return Semantic3D(root=args.root, scenes=args.scenes, transform=transform, download=args.download)


def configure_dataloader(args: Namespace) -> DataLoader:
    dataset = configure_dataset(args)
    return PointCloudDataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)


def configure_model(args: Namespace) -> SemanticSegmentationModel:
    return create_model(args.model, task="semantic-segmentation", pretrained=True)


def configure_inferer(args: Namespace) -> Inferer:
    return KNNWindowSegmentationInferer(
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
