"""Evaluate the Point-M2AE ShapeNetPart part-segmentation model (single-pass, no voting).

`ShapeNetPart` -> `DataLoader` -> model -> argmax -> per-shape IoU averaged into instance / class mIoU.
The registered transform normalizes each cloud to the unit sphere and one-hot encodes the category.

Results vs reference (instance mIoU / class mIoU; paper Tab. 5):

    | Variant                      | reference     | torch-pointcloud |
    | ---------------------------- | ------------- | ---------------- |
    | point-m2ae-base.shapenetpart | 86.51 / 84.86 | 86.17 / 84.60    |

Usage:
    uv run --no-sync python examples/point_m2ae_benchmark_shapenetpart.py
"""

import os
from argparse import ArgumentParser, Namespace
from collections import defaultdict

import numpy as np
import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm

import torch_pointcloud.models.point_m2ae  # noqa: F401
from torch_pointcloud.config import DATA_DIR
from torch_pointcloud.datasets import ShapeNetPart
from torch_pointcloud.models._registry import create_model
from torch_pointcloud.utils.data import DataKeys, collate
from torch_pointcloud.utils.metrics import compute_intersection_union
from torch_pointcloud.utils.ops import safe_divide
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

    print("Loading ShapeNetPart test dataset!")
    test_dataset = ShapeNetPart(
        root=args.root,
        split="test",
        transform=model_info.get("transforms"),
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
    )

    seg_ids = ShapeNetPart.seg_ids
    num_categories = len(seg_ids)

    print(f"Test set: {len(test_dataset)} samples")
    print("Evaluating...")
    metrics = evaluate(model, test_dataloader, args.device, seg_ids, num_categories)  # type: ignore[arg-type]

    print("\nResults:")
    for k, v in metrics.items():
        print(f"  {k:<20} {v:.4f}")


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Benchmark Point-M2AE part segmentation on ShapeNet.")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--root", type=str, default=DATA_DIR, help="Dataset root directory.")
    parser.add_argument(
        "--model",
        type=str,
        default="point-m2ae-base.shapenetpart",
        choices=["point-m2ae-base.shapenetpart"],
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--device", type=str, default=DEVICE)
    return parser.parse_args()


@torch.no_grad()
def evaluate(
    model: Module,
    dataloader: DataLoader,
    device: str,
    seg_ids: dict[str, list[int]],
    num_categories: int,
) -> dict[str, float]:
    model.to(device).eval()

    category_names = list(seg_ids.keys())
    num_classes = max(max(parts) for parts in seg_ids.values()) + 1
    shape_ious: dict[str, list[float]] = defaultdict(list)

    pbar = tqdm(dataloader, total=len(dataloader), desc="Testing")
    for data in pbar:
        pos = data[DataKeys.POS].to(device)
        segment = data[DataKeys.SEGMENT].to(device)
        cat_onehot = data[DataKeys.CATEGORY].to(device)
        batch = data[DataKeys.BATCH].to(device)

        logits = model(None, pos, batch, cat_onehot)
        preds = logits.argmax(dim=1)

        inter, union = compute_intersection_union(preds, segment, num_classes, batch=batch)
        for b in range(inter.shape[0]):
            cat_name = category_names[int(cat_onehot[b].argmax().item())]
            parts = seg_ids[cat_name]
            iou_b = safe_divide(inter[b, parts], union[b, parts], default=1.0)
            shape_ious[cat_name].append(iou_b.mean().item())

    cat_ious = {
        cat_name: float(np.mean(shape_ious[cat_name]))
        for cat_name in category_names
        if cat_name in shape_ious  # fmt: skip
    }
    instance_iou = float(np.mean([iou for ious in shape_ious.values() for iou in ious]))
    class_iou = float(np.mean(list(cat_ious.values())))

    results: dict[str, float] = {
        "test/instance_mIoU": instance_iou,
        "test/class_mIoU": class_iou,
    }
    for cat_name, iou in cat_ious.items():
        results[f"test/iou_{cat_name}"] = iou

    return results


if __name__ == "__main__":
    main()
