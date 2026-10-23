"""Benchmark GeoTransformer on the ModelNet40 registration pairs with the authors' protocol."""

import os
from argparse import ArgumentParser, Namespace
from typing import Dict, List

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from torch_pointcloud.config import DATA_DIR
from torch_pointcloud.datasets import ModelNet40Registration
from torch_pointcloud.metrics import alignment_error, rotation_error, translation_error
from torch_pointcloud.models import GeoTransformerRegistration, create_model
from torch_pointcloud.utils.data import DataKeys, PointCloudDataLoader
from torch_pointcloud.utils.random import seed_everything, set_determinism

CUDA_AVAILABLE = torch.cuda.is_available()
CPU_COUNT = os.cpu_count()
DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"
NUM_WORKERS = CPU_COUNT // 2 if CPU_COUNT is not None else 0
SEED = 42


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    set_determinism(tf32=False)

    print(f"Benchmarking model {args.model!r} on ModelNet40 registration pairs!")
    model = configure_model(args)
    dataloader = configure_dataloader(args)

    print(f"Test set: {len(dataloader.dataset)} pairs")  # type: ignore[arg-type]
    metrics = evaluate(model, dataloader, args.device)
    print("\nResults:")
    for key, value in metrics.items():
        print(f"  {key:<8} {value:.3f}")


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Benchmark GeoTransformer registration on ModelNet40 pairs.")
    parser.add_argument("--model", default="geotransformer.modelnet40.zheng-qin", help="Registered registration model")
    parser.add_argument("--classes", default="all", choices=["all", "seen", "unseen"])
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--root", default=DATA_DIR, help="Dataset root directory.")
    parser.add_argument("--seed", default=SEED, type=int)
    parser.add_argument("--batch-size", default=4, type=int)
    parser.add_argument("--num-workers", default=NUM_WORKERS, type=int)
    parser.add_argument("--limit", default=None, type=int, help="Evaluate at most this many pairs.")
    parser.add_argument("--download", action="store_true", help="Download the dataset if missing.")
    return parser.parse_args()


def configure_dataset(args: Namespace) -> Dataset:
    dataset: Dataset
    dataset = ModelNet40Registration(root=args.root, train=False, classes=args.classes, download=args.download)
    if args.limit is not None:
        n = min(int(args.limit), len(dataset))
        dataset = Subset(dataset, range(n))
        print(f"Evaluating on a subset of the first {n} pairs.")

    return dataset


def configure_dataloader(args: Namespace) -> DataLoader:
    dataset = configure_dataset(args)
    return PointCloudDataLoader(
        dataset,
        batch_size=args.batch_size,
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
    """Mean rotation error, translation error and alignment error (the RMSE of the paper) over the pairs."""
    model.to(device).eval()
    rotation_errors: List[torch.Tensor] = []
    translation_errors: List[torch.Tensor] = []
    alignment_errors: List[torch.Tensor] = []

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
        alignment_errors.append(alignment_error(data[DataKeys.POS], transform, target, batch=data[DataKeys.BATCH]))

    return {
        "RRE": torch.cat(rotation_errors).mean().item(),
        "RTE": torch.cat(translation_errors).mean().item(),
        "RMSE": torch.cat(alignment_errors).mean().item(),
    }


if __name__ == "__main__":
    main()
