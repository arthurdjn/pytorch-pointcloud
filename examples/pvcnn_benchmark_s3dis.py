"""Evaluate `pvcnn.s3dis-area5.mit-han-lab` on S3DIS Area-5, per-block protocol.

Each pre-tiled $4096$-point block is classified in isolation, so the result is
a lower bound on what the model can do. For the upstream sliding-window
protocol that recovers room-level context, see `pvcnn_benchmark_s3dis_sw.py`.

| Model                             | Paper        | Per-block (here)          | Sliding-window |
| --------------------------------- | ------------ | ------------------------- | -------------- |
| `pvcnn.s3dis-area5.mit-han-lab`   | 56.64 % mIoU | 35.93 % mIoU / 76.54 % OA | 57.51 % mIoU   |

Usage:

    uv run --no-sync python examples/pvcnn_benchmark_s3dis.py
"""

import os
from argparse import ArgumentParser, Namespace
from typing import Dict

import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm

from torch_pointcloud.config import DATA_DIR
from torch_pointcloud.datasets import S3DISHdf5
from torch_pointcloud.models._registry import create_model
from torch_pointcloud.utils.data import DataKeys, collate
from torch_pointcloud.utils.metrics import confusion_matrix
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
    parser.add_argument("--model", type=str, default="pvcnn.s3dis-area5.mit-han-lab")
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
    cm = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    pbar = tqdm(dataloader, total=len(dataloader), desc="Testing")
    for data in pbar:
        pos = data[DataKeys.POS].to(device)
        x = data[DataKeys.X].to(device)
        target = data[DataKeys.SEGMENT].to(device)
        batch = data[DataKeys.BATCH].to(device)

        logits = model(x, pos, batch)
        preds = logits.argmax(dim=1)
        cm += confusion_matrix(preds.cpu(), target.cpu(), num_classes, ignore_index=-1)

        diag = cm.diag().float()
        iou = diag / (cm.sum(0) + cm.sum(1) - cm.diag()).clamp_min(1).float()
        acc = diag.sum() / cm.sum().clamp_min(1).float()
        pbar.set_postfix({"mIoU": f"{iou.mean().item():.4f}", "oa": f"{acc.item():.4f}"})

    diag = cm.diag().float()
    iou = diag / (cm.sum(0) + cm.sum(1) - cm.diag()).clamp_min(1).float()
    per_class_acc = diag / cm.sum(1).clamp_min(1).float()
    return {
        "test/mIoU": iou.mean().item(),
        "test/mean_class_acc": per_class_acc.mean().item(),
        "test/overall_acc": diag.sum().item() / max(int(cm.sum()), 1),
    }


if __name__ == "__main__":
    main()
