"""Benchmark Point Transformer V3 semantic segmentation on GridNet-HD, 20 m chunks of the zones scored at full
resolution.

NOTE: the paper's 69.32 mIoU is scored on the test zones, whose labels the leaderboard keeps, over ten views (66.86 in
one view); this script scores the six validation zones in one view: each zone is sampled to one point per 2 cm voxel,
cut into 20 m chunks every 10 m, each chunk sampled to one point per 2.5 cm voxel and predicted by 5 cm
voxel-partition fragments, and every original point takes the prediction of its 2 cm voxel, averaged over the chunks
that cover it.
"""

from argparse import ArgumentParser, Namespace
from typing import Any, Dict

import torch
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

import torch_pointcloud.transforms as T
from torch_pointcloud.config import DATA_DIR
from torch_pointcloud.datasets import GridNetHD
from torch_pointcloud.datasets.gridnethd import GRIDNETHD_CLASSES, GRIDNETHD_IGNORE_IDX
from torch_pointcloud.inferers import Inferer, SlidingWindowInferer, VoxelPartitionInferer
from torch_pointcloud.metrics import accuracy, confusion_matrix, intersection_over_union
from torch_pointcloud.models import create_model
from torch_pointcloud.utils.data import DataKeys, PointCloudDataLoader
from torch_pointcloud.utils.random import seed_everything, set_determinism

CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"
# A zone holds 40 to 110 million points: reading it takes a minute, predicting it an hour, so no worker prefetches.
NUM_WORKERS = 0
SEED = 42
MODEL = "ptv3.gridnet-hd.heig-vd-geo"
NUM_CLASSES = len(GRIDNETHD_CLASSES)
ZONE_VOXEL_SIZE = 0.02
CHUNK_SIZE = 20.0
CHUNK_OVERLAP = 0.5
CHUNK_VOXEL_SIZE = 0.025
VOXEL_SIZE = 0.05
SW_BATCH_SIZE = 1

# The zone is sampled to one point per 2 cm voxel, and scored back on every point through `inverse`; nothing else
# leaves the transform.
TRANSFORM = T.Compose(
    [
        T.Normalize(keys=DataKeys.COLOR, mean=[0.0, 0.0, 0.0], std=[255.0, 255.0, 255.0]),
        T.CopyItems(keys=[DataKeys.SEGMENT], dst_keys=[DataKeys.ORIGIN_SEGMENT]),
        T.Voxelize(
            pos_key=DataKeys.POS,
            pos_reduce="first",
            keys=[DataKeys.COLOR],
            reduce="first",
            size=ZONE_VOXEL_SIZE,
            method="fnv",
            dst_inverse_key=DataKeys.INVERSE,
        ),
        T.KeepItems(keys=[DataKeys.POS, DataKeys.COLOR, DataKeys.ORIGIN_SEGMENT, DataKeys.INVERSE]),
    ]
)
# Each chunk is sampled to one point per 2.5 cm voxel, so a 5 cm voxel-partition fragment holds up to eight of them.
CHUNK_TRANSFORM = T.Voxelize(
    pos_key=DataKeys.POS,
    pos_reduce="first",
    keys=[DataKeys.COLOR],
    reduce="first",
    size=CHUNK_VOXEL_SIZE,
    method="fnv",
    dst_inverse_key=DataKeys.INVERSE,
)
# Each fragment is fed the colour, with its grid coordinates.
FRAGMENT_TRANSFORM = T.Compose(
    [
        T.Cat(keys=[DataKeys.COLOR], dst_key=DataKeys.X, dim=1),
        T.Voxelize(
            pos_key=DataKeys.POS,
            pos_reduce="first",
            dst_pos_grid_key=DataKeys.POS_GRID,
            keys=[DataKeys.X],
            reduce="first",
            size=VOXEL_SIZE,
            method="fnv",
            dst_inverse_key=DataKeys.INVERSE,
        ),
    ]
)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    set_determinism(tf32=False)

    print(f"Benchmarking model {args.model!r} on GridNet-HD!")
    dataloader = configure_dataloader(args)
    model = configure_model(args)
    inferer = configure_inferer(args)
    chunk_inferer = configure_chunk_inferer(args)

    num_zones = len(dataloader.dataset)  # type: ignore[arg-type]
    print(f"Validation set: {num_zones} zones  (overlapping {CHUNK_SIZE:g} m chunks, scored at full resolution)")
    metrics = evaluate(model, dataloader, inferer, chunk_inferer, args.device, NUM_CLASSES)
    print("\nResults:")
    for key, value in metrics.items():
        print(f"  {key:<32} {value:.4f}")


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Benchmark Point Transformer V3 semantic segmentation on GridNet-HD.")
    parser.add_argument("--model", default=MODEL, choices=[MODEL])
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--root", default=DATA_DIR, help="Dataset root directory.")
    parser.add_argument("--seed", default=SEED, type=int)
    parser.add_argument("--sw-batch-size", default=SW_BATCH_SIZE, type=int, help="Voxel fragments per forward.")
    parser.add_argument("--num-workers", default=NUM_WORKERS, type=int)
    parser.add_argument("--limit", default=None, type=int, help="Evaluate at most this many zones.")
    parser.add_argument("--download", action="store_true", help="Download the dataset if missing.")
    return parser.parse_args()


def configure_dataset(args: Namespace) -> Dataset:
    dataset: Dataset = GridNetHD(root=args.root, split="val", transform=TRANSFORM, download=args.download)
    if args.limit is not None:
        n = min(int(args.limit), len(dataset))  # type: ignore[arg-type]
        dataset = Subset(dataset, range(n))
        print(f"Evaluating on a subset of the first {n} zones.")

    return dataset


def configure_dataloader(args: Namespace) -> DataLoader:
    dataset = configure_dataset(args)
    return PointCloudDataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)


def configure_model(args: Namespace) -> Module:
    return create_model(args.model, task="semantic-segmentation", pretrained=True)


def configure_inferer(args: Namespace) -> Inferer:
    return SlidingWindowInferer(
        block_size=CHUNK_SIZE,
        overlap=CHUNK_OVERLAP,
        dims=(0, 1),
        transform=CHUNK_TRANSFORM,
        inverse_key=DataKeys.INVERSE,
        seed=args.seed,
        progress=True,
    )


def configure_chunk_inferer(args: Namespace) -> Inferer:
    return VoxelPartitionInferer(
        voxel_size=VOXEL_SIZE,
        transform=FRAGMENT_TRANSFORM,
        softmax=True,
        sw_batch_size=args.sw_batch_size,
        inverse_key=DataKeys.INVERSE,
        seed=args.seed,
    )


@torch.no_grad()
def evaluate(
    model: Module,
    dataloader: DataLoader,
    inferer: Inferer,
    chunk_inferer: Inferer,
    device: str,
    num_classes: int,
) -> Dict[str, Any]:
    model.to(device).eval()
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)

    def predict_fragment(fragment: Dict[str, Any]) -> torch.Tensor:
        logits = model(
            fragment[DataKeys.X].to(device), fragment[DataKeys.POS_GRID].to(device), fragment[DataKeys.BATCH].to(device)
        )
        return logits.cpu()

    pbar = tqdm(dataloader, total=len(dataloader), desc="Testing")
    for data in pbar:
        # Only the model inputs travel through the chunks: the labels and the full-resolution keys stay behind for the
        # scoring. The zone stays on the host: its chunks are predicted fragment by fragment on the device.
        inputs: Dict[str, Any] = {key: data[key] for key in (DataKeys.POS, DataKeys.COLOR, DataKeys.BATCH)}
        probs = inferer(inputs, predictor=lambda chunk: chunk_inferer(chunk, predictor=predict_fragment))
        preds = probs.argmax(dim=1)[data[DataKeys.INVERSE]]
        cm += confusion_matrix(preds, data[DataKeys.ORIGIN_SEGMENT], num_classes, ignore_index=GRIDNETHD_IGNORE_IDX)
        oa = accuracy(cm)
        pbar.set_postfix({"oa": f"{oa:.4f}"})

    per_class = intersection_over_union(cm, average="none", class_names=GRIDNETHD_CLASSES)
    return {
        "val/mIoU": intersection_over_union(cm),
        "val/oa": accuracy(cm),
        **{f"val/IoU/{name}": iou for name, iou in per_class.items()},
    }


if __name__ == "__main__":
    main()
