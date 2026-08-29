"""Benchmark OneFormer3D semantic and instance segmentation on ScanNet and S3DIS with the reference protocol.

NOTE: the S3DIS reference's `semantic * 1000 + instance` id encoding drops unmatched ceiling predictions instead of
counting false positives; this script counts them, so its ceiling AP can only read lower.

Results (mIoU / mAP / mAP@0.5 / mAP@0.25):

    | Variant                                       | reference                 | torch-pointcloud |
    | --------------------------------------------- | ------------------------- | ---------------- |
    | oneformer3d-base.scannet20.danila-rukhovich   | 76.4 / 59.3 / 78.8 / 86.7 | 76.51 / 59.54 / 78.57 / 86.65 |
    | oneformer3d-base.s3dis-area5.danila-rukhovich | 71.9 / 58.0 / 72.7 / 80.6 | 71.95 / 58.24 / 72.47 / 80.05 |

Usage:
    uv run --no-sync python examples/oneformer3d_benchmark_segmentation.py --model oneformer3d-base.scannet20.danila-rukhovich --limit 20
    uv run --no-sync python examples/oneformer3d_benchmark_segmentation.py --model oneformer3d-base.s3dis-area5.danila-rukhovich
"""

import argparse
import os
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

import torch_pointcloud.transforms as T
from torch_pointcloud.config import DATA_DIR
from torch_pointcloud.datasets import S3DIS, ScanNet20
from torch_pointcloud.datasets.s3dis import S3DIS_CLASS_TO_IDX, S3DIS_CLASSES
from torch_pointcloud.datasets.scannet import SCANNET20_CLASSES
from torch_pointcloud.models import create_model
from torch_pointcloud.models.oneformer3d import OneFormer3DSegmentation, _shift_superpoints
from torch_pointcloud.utils.data import DataKeys, PointCloudDataLoader
from torch_pointcloud.utils.metrics import confusion_matrix, instance_average_precision, instance_matches
from torch_pointcloud.utils.random import seed_everything, set_determinism

CUDA_AVAILABLE = torch.cuda.is_available()
CPU_COUNT = os.cpu_count()
DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"
NUM_WORKERS = CPU_COUNT // 2 if CPU_COUNT is not None else 0
SEED = 42
MODELS = ["oneformer3d-base.scannet20.danila-rukhovich", "oneformer3d-base.s3dis-area5.danila-rukhovich"]


def summarize(
    cm: torch.Tensor, records: List[Dict[str, Any]], num_classes: int, class_names: List[str]
) -> Dict[str, Any]:
    intersection = cm.diag().float()
    union = cm.sum(dim=1).float() + cm.sum(dim=0).float() - intersection
    valid = union > 0
    instance_ap = instance_average_precision(records, num_classes=num_classes, class_names=class_names)
    return {
        "test/mIoU": (intersection[valid] / union[valid]).mean().item(),
        "test/oa": (cm.diag().sum().float() / cm.sum().float().clamp_min(1)).item(),
        "test/mAP": instance_ap["mAP"],
        "test/mAP@0.5": instance_ap["mAP@0.5"],
        "test/mAP@0.25": instance_ap["mAP@0.25"],
    }


@torch.no_grad()
def evaluate_scannet(model: OneFormer3DSegmentation, dataloader: DataLoader, device: str) -> Dict[str, Any]:
    model.to(device).eval()
    num_classes = model.num_semantic_classes
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    records = []

    pbar = tqdm(dataloader, total=len(dataloader), desc="Testing")
    for data in pbar:
        data = {key: value.to(device) if torch.is_tensor(value) else value for key, value in data.items()}
        batch, inverse, superpoint = data[DataKeys.BATCH], data[DataKeys.INVERSE], data[DataKeys.SUPERPOINT]
        target = data[DataKeys.ORIGIN_SEGMENT].long()
        output = model(data[DataKeys.X], data[DataKeys.POS_GRID].long(), batch, superpoint, inverse)
        superpoint_shift, _ = _shift_superpoints(superpoint, inverse, batch)
        preds = model.predict_semantic(output, superpoint_shift)
        cm += confusion_matrix(preds.cpu(), target.cpu(), num_classes, ignore_index=-1)
        masks, labels, scores = model.predict_instance(output, superpoint_shift)
        # Instance classes are the semantic classes minus the two stuff classes (wall, floor).
        instance = data[DataKeys.ORIGIN_INSTANCE].long()
        gt_labels = torch.where((target >= 2) & (instance >= 0), target - 2, torch.full_like(target, -1))
        records.append(instance_matches(masks, labels, scores, instance, gt_labels))
        pbar.set_postfix({"oa": f"{(cm.diag().sum().float() / cm.sum().float().clamp_min(1)).item():.4f}"})

    return summarize(cm, records, model.num_instance_classes, list(SCANNET20_CLASSES[2:]))


@torch.no_grad()
def evaluate_s3dis(
    model: OneFormer3DSegmentation, dataloader: DataLoader, device: str, classes: List[str]
) -> Dict[str, Any]:
    model.to(device).eval()
    num_classes = model.num_semantic_classes
    # The checkpoint's channel order differs from the dataset's label order.
    remap = torch.tensor([S3DIS_CLASS_TO_IDX[name] for name in classes], device=device)
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    records = []

    pbar = tqdm(dataloader, total=len(dataloader), desc="Testing")
    for data in pbar:
        data = {key: value.to(device) if torch.is_tensor(value) else value for key, value in data.items()}
        inverse = data[DataKeys.INVERSE]
        target = data[DataKeys.ORIGIN_SEGMENT].long()
        output = model(data[DataKeys.X], data[DataKeys.POS_GRID].long(), data[DataKeys.BATCH])
        preds = remap[model.predict_semantic(output, inverse)]
        cm += confusion_matrix(preds.cpu(), target.cpu(), num_classes, ignore_index=-1)
        masks, labels, scores = model.predict_instance(
            output,
            inverse,
            topk=450,
            sp_score_threshold=0.15,
            npoint_threshold=300,
            obj_normalization_threshold=0.01,
        )
        records.append(instance_matches(masks, remap[labels], scores, data[DataKeys.ORIGIN_INSTANCE].long(), target))
        pbar.set_postfix({"oa": f"{(cm.diag().sum().float() / cm.sum().float().clamp_min(1)).item():.4f}"})

    return summarize(cm, records, num_classes, list(S3DIS_CLASSES))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark OneFormer3D semantic and instance segmentation.")
    parser.add_argument("--model", default=MODELS[0], choices=MODELS)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--root", default=DATA_DIR, help="Dataset root directory.")
    parser.add_argument("--seed", default=SEED, type=int)
    parser.add_argument("--num-workers", default=NUM_WORKERS, type=int)
    parser.add_argument("--limit", default=None, type=int, help="Evaluate at most this many scenes.")
    parser.add_argument("--download", action="store_true", help="Download the dataset if missing.")
    parser.add_argument("--force-process", action="store_true", help="Force re-processing the dataset.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    set_determinism(tf32=False)

    model, model_info = create_model(args.model, task="segmentation", pretrained=True, return_info=True)
    assert isinstance(model, OneFormer3DSegmentation)
    # The registered transform voxelizes the instance ids; the raw ones score the instance masks.
    transform = T.Compose(
        [T.CopyItems(keys=DataKeys.INSTANCE, names=DataKeys.ORIGIN_INSTANCE), model_info["transform"]]
    )

    dataset: Dataset
    if "scannet" in args.model:
        print(f"Benchmarking model {args.model!r} on ScanNet!")
        dataset = ScanNet20(
            root=args.root,
            split="val",
            return_superpoint=True,
            transform=transform,
            download=args.download,
            force_process=args.force_process,
            num_workers=args.num_workers,
            use_axis_alignment=False,
        )
    else:
        print(f"Benchmarking model {args.model!r} on S3DIS Area 5!")
        dataset = S3DIS(
            root=args.root,
            areas=["Area_5"],
            aligned=True,
            transform=transform,
            download=args.download,
            force_process=args.force_process,
            num_workers=args.num_workers,
        )
    if args.limit is not None:
        n = min(int(args.limit), len(dataset))
        dataset = Subset(dataset, range(n))
        print(f"Evaluating on a subset of the first {n} scenes.")

    dataloader = PointCloudDataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    print(f"Test set: {len(dataset)} scenes")
    if "scannet" in args.model:
        metrics = evaluate_scannet(model, dataloader, args.device)
    else:
        metrics = evaluate_s3dis(model, dataloader, args.device, list(model_info["weights"]["classes"]))
    print("\nResults:")
    for key, value in metrics.items():
        print(f"  {key:<24} {value:.4f}")


if __name__ == "__main__":
    main()
