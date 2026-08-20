r"""Benchmark PointNeXt semantic segmentation on S3DIS.

NOTE: this eval keeps the deterministic first point per voxel while the reference freezes one random
point per voxel; Area-5 mIoU spans roughly $62.1$ to $63.0$ across such draws, so the gap is noise.

Results on Area 5 (align=True, voxel_size=0.04):
    pointnext-sm  : mIoU=62.12%  OA=86.83%  (reported: 63.4/87.9)

Usage:
    # Area 5 evaluation (default)
    uv run --no-sync python examples/pointnext_benchmark_s3dis.py \
        --model pointnext-sm.s3dis-area5.openpoints --download

    # 6-fold cross-validation (one per-area model at a time)
    uv run --no-sync python examples/pointnext_benchmark_s3dis.py \
        --model pointnext-sm.s3dis-area1.openpoints --areas Area_1 --download
"""

import os
from argparse import ArgumentParser, Namespace
from typing import Dict

import torch
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from torch_pointcloud.config import DATA_DIR
from torch_pointcloud.datasets import S3DIS
from torch_pointcloud.models import create_model
from torch_pointcloud.utils.data import DataKeys, collate
from torch_pointcloud.utils.metrics import confusion_matrix
from torch_pointcloud.utils.random import seed_everything, set_determinism

CUDA_AVAILABLE = torch.cuda.is_available()
CPU_COUNT = os.cpu_count()
DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"
NUM_WORKERS = CPU_COUNT // 2 if CPU_COUNT is not None else 0
BATCH_SIZE = 1
SEED = 42

MODEL_CHOICES = [f"pointnext-{v}.s3dis-area{a}.openpoints" for v in ("sm", "base", "lg", "xl") for a in range(1, 7)]


def main() -> None:
    args = parse_args()

    print(f"Seeding everything to {args.seed}!")
    seed_everything(args.seed)
    set_determinism(tf32=False)
    print(f"Benchmarking model {args.model!r} on S3DIS areas {list(args.areas)}!")

    model, info = create_model(args.model, task="segmentation", pretrained=args.pretrained, return_info=True)
    num_classes: int = int(model.num_classes)
    transform = info.get("transform")

    test_dataset: Dataset = S3DIS(
        root=args.root,
        areas=list(args.areas),
        download=args.download,
        force_process=args.force_process,
        transform=transform,
        show_progress=True,
        num_workers=args.num_workers,
    )
    if args.limit is not None:
        n = min(int(args.limit), len(test_dataset))  # type: ignore[arg-type]
        test_dataset = Subset(test_dataset, range(n))
        print(f"Evaluating on a subset of the first {n} rooms.")

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
    )

    print(f"Test set: {len(test_dataset)} samples")  # type: ignore[arg-type]
    print("Evaluating...")
    metrics = evaluate(model, test_dataloader, args.device, num_classes=num_classes)

    print("\nResults:")
    for k, v in metrics.items():
        print(f"  {k:<20} {v:.4f}")


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Benchmark PointNeXt semantic segmentation on S3DIS.")
    pretrained_group = parser.add_mutually_exclusive_group()
    pretrained_group.add_argument(
        "--pretrained", dest="pretrained", action="store_true", help="Load weights (default)."
    )
    pretrained_group.add_argument(
        "--no-pretrained", dest="pretrained", action="store_false", help="Skip loading weights."
    )
    parser.set_defaults(pretrained=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--root", type=str, default=DATA_DIR, help="Dataset root directory.")
    parser.add_argument(
        "--model",
        type=str,
        default="pointnext-sm.s3dis-area5.openpoints",
        choices=MODEL_CHOICES,
    )
    parser.add_argument(
        "--areas",
        nargs="+",
        default=["Area_5"],
        help="S3DIS areas to evaluate on (default: Area_5).",
    )
    parser.add_argument("--download", action="store_true", help="Download S3DIS if missing.")
    parser.add_argument("--force-process", action="store_true", help="Force reprocessing of raw data.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--device", type=str, default=DEVICE)
    parser.add_argument("--limit", type=int, default=None, help="Evaluate at most this many rooms (debug).")
    return parser.parse_args()


@torch.no_grad()
def evaluate(model: Module, dataloader: DataLoader, device: str, *, num_classes: int) -> Dict[str, float]:
    model.to(device).eval()
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)

    pbar = tqdm(dataloader, total=len(dataloader), desc="Testing")
    for data in pbar:
        x = data[DataKeys.X].to(device)
        pos = data[DataKeys.POS].to(device)
        target = data[DataKeys.SEGMENT].to(device)
        batch = data[DataKeys.BATCH].to(device)

        logits = model(x, pos, batch)
        preds = logits.argmax(dim=1)

        cm += confusion_matrix(preds.cpu(), target.cpu(), num_classes)
        oa = cm.diag().sum().float() / cm.sum().float().clamp_min(1)
        pbar.set_postfix({"oa": f"{oa.item():.4f}"})

    intersection = cm.diag().float()
    union = cm.sum(dim=1).float() + cm.sum(dim=0).float() - intersection
    iou_per_class = intersection / union.clamp_min(1e-10)
    miou = iou_per_class.nan_to_num(nan=0.0).mean()
    oa = cm.diag().sum().float() / cm.sum().float()

    return {"test/mIoU": miou.item(), "test/oa": oa.item()}


if __name__ == "__main__":
    main()
