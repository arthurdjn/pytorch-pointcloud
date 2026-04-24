import os
from argparse import ArgumentParser, Namespace
from typing import Dict

import numpy as np
import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm

from torch_pointcloud.config import DATA_DIR
from torch_pointcloud.datasets.s3dis import S3DIS_CLASSES, S3DISHdf5
from torch_pointcloud.models._registry import create_model
from torch_pointcloud.utils.data import DataKeys, collate
from torch_pointcloud.utils.random import seed_everything

CUDA_AVAILABLE = torch.cuda.is_available()
CPU_COUNT = os.cpu_count()
DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"
NUM_WORKERS = CPU_COUNT // 2 if CPU_COUNT is not None else 0
BATCH_SIZE = 16
SEED = 42


def main() -> None:
    args = parse_args()

    print(f"Seeding everything to {args.seed}!")
    seed_everything(args.seed)

    model_name = f"dgcnn-antao.s3dis.area{args.area}"
    print(f"Loading model {model_name!r}!")
    model, model_info = create_model(model_name, task="segmentation", pretrained=True, return_info=True)

    area = f"Area_{args.area}"
    print(f"Loading test dataset for area {area}!")
    test_dataset = S3DISHdf5(
        root=args.root,
        areas=area,  # type: ignore[arg-type]
        transform=model_info.get("transforms"),
        download=False,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
    )

    preds, targets = predict(model, test_dataloader, args.device)
    metrics = compute_metrics(preds, targets, num_classes=13)

    print("\nResults:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Benchmark DGCNN semantic segmentation on S3DIS.")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--root", type=str, default=DATA_DIR, help="Dataset root directory.")
    parser.add_argument("--area", type=int, default=5, choices=list(range(1, 7)))
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--device", type=str, default=DEVICE)
    return parser.parse_args()


@torch.no_grad()
def predict(model: Module, dataloader: DataLoader, device: str) -> tuple:
    model.to(device).eval()
    all_preds = []
    all_targets = []

    for data in tqdm(dataloader, desc="Predicting"):
        pos = data[DataKeys.POS].to(device)
        color = data[DataKeys.COLOR].to(device)
        norm_pos = data["norm_pos"].to(device)
        segment = data[DataKeys.SEGMENT]
        batch = data[DataKeys.BATCH].to(device)
        x = torch.cat([pos, color], dim=1)

        logits = model(x, norm_pos, batch)
        preds = logits.argmax(dim=1).cpu().numpy()

        all_preds.append(preds)
        all_targets.append(segment.numpy())

    return np.concatenate(all_preds), np.concatenate(all_targets)


def compute_metrics(preds: np.ndarray, targets: np.ndarray, num_classes: int) -> Dict[str, float]:
    overall_acc = (preds == targets).mean()

    ious = []
    for c in range(num_classes):
        gt_mask = targets == c
        pred_mask = preds == c
        intersection = np.logical_and(gt_mask, pred_mask).sum()
        union = np.logical_or(gt_mask, pred_mask).sum()
        if union > 0:
            ious.append(intersection / union)

    miou = np.mean(ious) if ious else 0.0

    results: Dict[str, float] = {
        "test/overall_acc": overall_acc,
        "test/mIoU": miou,
    }
    for c, name in enumerate(S3DIS_CLASSES):
        gt_mask = targets == c
        pred_mask = preds == c
        intersection = np.logical_and(gt_mask, pred_mask).sum()
        union = np.logical_or(gt_mask, pred_mask).sum()
        results[f"test/iou_{name}"] = (intersection / union) if union > 0 else float("nan")

    return results


if __name__ == "__main__":
    main()
