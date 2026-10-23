"""Benchmark GeoTransformer on 3DMatch / 3DLoMatch with the authors' protocol."""

import os
from argparse import ArgumentParser, Namespace
from collections import defaultdict
from typing import Dict, List

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from torch_pointcloud.config import DATA_DIR
from torch_pointcloud.datasets import ThreeDMatch
from torch_pointcloud.metrics import inlier_ratio, registration_rmse, rotation_error, translation_error
from torch_pointcloud.models import GeoTransformerRegistration, create_model
from torch_pointcloud.utils.data import DataKeys, PointCloudDataLoader
from torch_pointcloud.utils.random import seed_everything, set_determinism

CUDA_AVAILABLE = torch.cuda.is_available()
CPU_COUNT = os.cpu_count()
DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"
NUM_WORKERS = CPU_COUNT // 2 if CPU_COUNT is not None else 0
SEED = 42
INLIER_RADIUS = 0.1
INLIER_RATIO_THRESHOLD = 0.05
RMSE_THRESHOLD = 0.2


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    set_determinism(tf32=False)

    print(f"Benchmarking model {args.model!r} on {args.benchmark}!")
    model = configure_model(args)
    dataloader = configure_dataloader(args)

    print(f"Test set: {len(dataloader.dataset)} pairs")  # type: ignore[arg-type]
    metrics = evaluate(model, dataloader, args.device)
    print("\nResults:")
    for key, value in metrics.items():
        print(f"  {key:<8} {value:.2f}")


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Benchmark GeoTransformer registration on 3DMatch / 3DLoMatch.")
    parser.add_argument("--model", default="geotransformer.3dmatch.zheng-qin", help="Registered registration model")
    parser.add_argument("--benchmark", default="3DMatch", choices=["3DMatch", "3DLoMatch"])
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--root", default=DATA_DIR, help="Dataset root directory.")
    parser.add_argument("--seed", default=SEED, type=int)
    parser.add_argument("--num-workers", default=NUM_WORKERS, type=int)
    parser.add_argument("--limit", default=None, type=int, help="Evaluate at most this many pairs.")
    parser.add_argument("--download", action="store_true", help="Download the dataset if missing.")
    parser.add_argument("--force-process", action="store_true", help="Force re-processing the dataset.")
    return parser.parse_args()


def configure_dataset(args: Namespace) -> Dataset:
    dataset: Dataset
    dataset = ThreeDMatch(
        root=args.root,
        split="test",
        benchmark=args.benchmark,
        download=args.download,
        force_process=args.force_process,
        num_workers=args.num_workers,
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
        stack_keys=[DataKeys.TRANSFORM, DataKeys.INFORMATION, DataKeys.FRAME],
    )


def configure_model(args: Namespace) -> GeoTransformerRegistration:
    model = create_model(args.model, task="registration", pretrained=True)
    assert isinstance(model, GeoTransformerRegistration)
    return model


@torch.no_grad()
def evaluate(model: GeoTransformerRegistration, dataloader: DataLoader, device: str) -> Dict[str, float]:
    """Inlier ratio, feature matching recall, registration recall and the errors of the registered pairs.

    Every number is first averaged within a scene, then over the 8 scenes; the registration recall counts the
    non-consecutive fragment pairs of the benchmark log files, as the 3DMatch evaluation does.
    """
    model.to(device).eval()
    scene_inlier_ratios: Dict[str, List[float]] = defaultdict(list)
    scene_matching: Dict[str, List[float]] = defaultdict(list)
    scene_registered: Dict[str, List[float]] = defaultdict(list)
    scene_rre: Dict[str, List[float]] = defaultdict(list)
    scene_rte: Dict[str, List[float]] = defaultdict(list)

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
        registration = model.decode(output)
        transform, target = registration["transform"], data[DataKeys.TRANSFORM]
        scene = data[DataKeys.SCENE][0]

        ratio = inlier_ratio(registration["src_points"], registration["ref_points"], target, INLIER_RADIUS).item()
        scene_inlier_ratios[scene].append(ratio)
        scene_matching[scene].append(float(ratio >= INLIER_RATIO_THRESHOLD))

        ref_frame, src_frame = data[DataKeys.FRAME][0].tolist()
        if src_frame > ref_frame + 1:
            rmse = registration_rmse(transform, target, data[DataKeys.INFORMATION]).item()
            registered = rmse < RMSE_THRESHOLD
            scene_registered[scene].append(float(registered))
            if registered:
                scene_rre[scene].append(rotation_error(transform, target).item())
                scene_rte[scene].append(translation_error(transform, target).item())

    def scene_mean(values: Dict[str, List[float]]) -> float:
        return sum(sum(v) / len(v) for v in values.values() if v) / max(len(values), 1)

    return {
        "RR": 100 * scene_mean(scene_registered),
        "FMR": 100 * scene_mean(scene_matching),
        "IR": 100 * scene_mean(scene_inlier_ratios),
        "RRE": scene_mean(scene_rre),
        "RTE": 100 * scene_mean(scene_rte),
    }


if __name__ == "__main__":
    main()
