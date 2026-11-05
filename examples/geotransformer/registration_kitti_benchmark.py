"""Benchmark GeoTransformer on the KITTI odometry pairs with the authors' protocol."""

import os
from argparse import ArgumentParser, Namespace
from typing import Dict, List

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

import torch_pointcloud.transforms as T
from torch_pointcloud.config import DATA_DIR
from torch_pointcloud.datasets import KITTIOdometry
from torch_pointcloud.metrics import rotation_error, translation_error
from torch_pointcloud.models import GeoTransformerRegistration, create_model
from torch_pointcloud.utils.data import DataKeys, PointCloudDataLoader
from torch_pointcloud.utils.random import seed_everything, set_determinism

CUDA_AVAILABLE = torch.cuda.is_available()
CPU_COUNT = os.cpu_count()
DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"
NUM_WORKERS = CPU_COUNT // 2 if CPU_COUNT is not None else 0
SEED = 42
VOXEL_SIZE = 0.3
RRE_THRESHOLD = 5.0
RTE_THRESHOLD = 2.0

# Both scans are averaged in 30 cm voxels on a grid anchored at the cloud minimum; the authors' grid starts half a
# voxel lower.
TRANSFORM = T.Compose(
    [
        T.Voxelize(pos_key=DataKeys.POS, pos_reduce="mean", size=VOXEL_SIZE, origin="min"),
        T.Voxelize(pos_key=DataKeys.REF_POS, pos_reduce="mean", size=VOXEL_SIZE, origin="min"),
    ]
)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    set_determinism(tf32=False)

    print(f"Benchmarking model {args.model!r} on KITTI odometry!")
    model = configure_model(args)
    dataloader = configure_dataloader(args)

    print(f"Test set: {len(dataloader.dataset)} pairs")  # type: ignore[arg-type]
    metrics = evaluate(model, dataloader, args.device)
    print("\nResults:")
    for key, value in metrics.items():
        print(f"  {key:<8} {value:.3f}")


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Benchmark GeoTransformer registration on KITTI odometry.")
    parser.add_argument("--model", default="geotransformer.kitti.zheng-qin", help="Registered registration model")
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--root", default=DATA_DIR, help="Dataset root directory.")
    parser.add_argument("--seed", default=SEED, type=int)
    parser.add_argument("--num-workers", default=NUM_WORKERS, type=int)
    parser.add_argument("--limit", default=None, type=int, help="Evaluate at most this many pairs.")
    parser.add_argument("--download", action="store_true", help="Download the pair lists if missing.")
    parser.add_argument("--force-process", action="store_true", help="Force re-processing the pair lists.")
    return parser.parse_args()


def configure_dataset(args: Namespace) -> Dataset:
    dataset: Dataset
    dataset = KITTIOdometry(
        root=args.root,
        split="test",
        transform=TRANSFORM,
        download=args.download,
        force_process=args.force_process,
    )
    if args.limit is not None:
        n = min(int(args.limit), len(dataset))
        dataset = Subset(dataset, range(n))
        print(f"Evaluating on a subset of the first {n} pairs.")

    return dataset


def configure_dataloader(args: Namespace) -> DataLoader:
    dataset = configure_dataset(args)
    return PointCloudDataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        cat_keys=[DataKeys.REF_POS],
        stack_keys=[DataKeys.TRANSFORM],
    )


def configure_model(args: Namespace) -> GeoTransformerRegistration:
    model = create_model(args.model, task="registration", pretrained=True)
    assert isinstance(model, GeoTransformerRegistration)
    return model


@torch.no_grad()
def evaluate(model: GeoTransformerRegistration, dataloader: DataLoader, device: str) -> Dict[str, float]:
    """Registration recall and the errors of the registered pairs.

    A pair is registered when its rotation error is under 5 degrees and its translation error under 2 m; RRE and
    RTE average over those pairs, as the KITTI evaluation does.
    """
    model.to(device).eval()
    rotation_errors: List[torch.Tensor] = []
    translation_errors: List[torch.Tensor] = []

    for data in tqdm(dataloader, total=len(dataloader), desc="Testing"):
        data = {key: value.to(device) if torch.is_tensor(value) else value for key, value in data.items()}
        output = model(
            None,
            data[DataKeys.POS],
            data[DataKeys.BATCH],
            None,
            data[DataKeys.REF_POS],
            data[DataKeys.BATCH_REF_POS],
        )
        transform, target = model.decode(output)["transform"], data[DataKeys.TRANSFORM]
        rotation_errors.append(rotation_error(transform, target))
        translation_errors.append(translation_error(transform, target))

    rre, rte = torch.cat(rotation_errors), torch.cat(translation_errors)
    registered = (rre < RRE_THRESHOLD) & (rte < RTE_THRESHOLD)
    return {
        "RR": 100 * registered.float().mean().item(),
        "RRE": rre[registered].mean().item(),
        "RTE": 100 * rte[registered].mean().item(),
    }


if __name__ == "__main__":
    main()
