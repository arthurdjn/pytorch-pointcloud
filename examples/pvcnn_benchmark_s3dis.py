"""Evaluate `pvcnn-mit-han-lab.s3dis-area5` on S3DIS Area-5 HDF5 blocks (per-block protocol).

Same per-block evaluation as `pointnet2_benchmark_s3dis.py`: load the pre-tiled
$4096$-point HDF5 blocks from `indoor3d_sem_seg_hdf5_data`, run the model once
per block, and aggregate IoU without cross-block voting. The per-block
protocol is a lower bound because each block is classified in isolation;
see `pvcnn_benchmark_s3dis_sw.py` for the upstream-style $1.5\\,\\text{m}$
sliding-window protocol that recovers the missing room-level context.

Reproduced performance on S3DIS Area-5 (seed=42, batch_size=16):

| Setting                             | Upstream paper | per-block (this script)   | sliding-window |
| ----------------------------------- | -------------- | ------------------------- | -------------- |
| `pvcnn-mit-han-lab.s3dis-area5`     | 56.64 % mIoU   | 35.93 % mIoU / 76.54 % OA | 57.71 % mIoU   |

Usage:

    uv run --no-sync python examples/pvcnn_benchmark_s3dis.py
"""

import os
from argparse import ArgumentParser, Namespace
from typing import Dict

import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm

from torch_pointcloud.config import DATA_DIR
from torch_pointcloud.datasets import S3DISHdf5
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

    print(f"Loading model {args.model!r}!")
    model, model_info = create_model(
        args.model,
        task="segmentation",
        pretrained=True,
        return_info=True,
    )

    num_classes: int = int(model.num_classes)
    transform = model_info.get("transforms")

    print(f"Loading S3DIS HDF5 test areas {args.areas}!")
    test_dataset = S3DISHdf5(
        root=args.root,
        areas=args.areas,
        transform=transform,
        download=args.download,
        show_progress=False,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
    )

    print(f"Test set: {len(test_dataset)} samples")
    print("Evaluating...")
    metrics = evaluate(model, test_dataloader, args.device, num_classes=num_classes)

    print("\nResults:")
    for k, v in metrics.items():
        print(f"  {k:<20} {v:.4f}")


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Benchmark PVCNN semantic segmentation on S3DIS.")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--root", type=str, default=DATA_DIR, help="Dataset root directory.")
    parser.add_argument("--model", type=str, default="pvcnn-mit-han-lab.s3dis-area5")
    parser.add_argument("--areas", nargs="+", default=["Area_5"])
    parser.add_argument("--download", action="store_true", help="Download S3DIS-HDF5 if missing.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--device", type=str, default=DEVICE)
    return parser.parse_args()


@torch.no_grad()
def evaluate(
    model: Module,
    dataloader: DataLoader,
    device: str,
    *,
    num_classes: int,
) -> Dict[str, float]:
    model.to(device).eval()

    intersection_total = torch.zeros(num_classes, dtype=torch.float64)
    union_total = torch.zeros(num_classes, dtype=torch.float64)
    target_total = torch.zeros(num_classes, dtype=torch.float64)
    correct_total = 0
    seen_total = 0

    pbar = tqdm(dataloader, total=len(dataloader), desc="Testing")
    for data in pbar:
        pos = data[DataKeys.POS].to(device)
        x = data[DataKeys.X].to(device)
        target = data[DataKeys.SEGMENT].to(device)
        batch = data[DataKeys.BATCH].to(device)

        logits = model(x, pos, batch)
        preds = logits.argmax(dim=1)

        inter, union, gt = _intersection_union_target(preds, target, num_classes)
        intersection_total += inter.cpu().double()
        union_total += union.cpu().double()
        target_total += gt.cpu().double()
        correct_total += int(preds.eq(target).sum().item())
        seen_total += int(target.numel())

        running_iou = (intersection_total / union_total.clamp_min(1)).mean().item()
        pbar.set_postfix({"mIoU": f"{running_iou:.4f}", "oa": f"{correct_total / seen_total:.4f}"})

    iou_per_class = (intersection_total / union_total.clamp_min(1)).float()
    acc_per_class = (intersection_total / target_total.clamp_min(1)).float()
    return {
        "test/mIoU": iou_per_class.mean().item(),
        "test/mean_class_acc": acc_per_class.mean().item(),
        "test/overall_acc": correct_total / max(seen_total, 1),
    }


def _intersection_union_target(preds: Tensor, target: Tensor, num_classes: int) -> tuple[Tensor, Tensor, Tensor]:
    valid = target >= 0
    preds, target = preds[valid], target[valid]
    bins = num_classes
    inter = torch.zeros(bins, dtype=torch.long, device=preds.device)
    union = torch.zeros(bins, dtype=torch.long, device=preds.device)
    gt = torch.zeros(bins, dtype=torch.long, device=preds.device)
    for c in range(bins):
        pred_c = preds == c
        target_c = target == c
        inter[c] = int((pred_c & target_c).sum().item())
        union[c] = int((pred_c | target_c).sum().item())
        gt[c] = int(target_c.sum().item())
    return inter, union, gt


if __name__ == "__main__":
    main()
