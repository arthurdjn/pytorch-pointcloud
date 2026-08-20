r"""Evaluate 3DETR on SUN RGB-D val with mAP@0.25 / mAP@0.5.

Results vs reference (SUN RGB-D val):

| Model                | This script (mAP@0.25 / @0.50) | Reference (@0.25 / @0.50) |
| -------------------- | ------------------------------ | ------------------------- |
| `3detr.sunrgbd.fair` | 58.20 / 29.70                  | 58.0 / 30.3               |

Usage:
    uv run --no-sync python examples/3detr_benchmark_sunrgbd.py --root /path/to/data
"""

import os
from argparse import ArgumentParser, Namespace
from typing import Dict, List

import torch
from torch.utils.data import Dataset, Subset
from tqdm import tqdm

from torch_pointcloud.config import DATA_DIR
from torch_pointcloud.datasets import SunRGBD
from torch_pointcloud.models import create_model
from torch_pointcloud.models.detr3d import DETR3DDetection
from torch_pointcloud.utils.box3d import count_points_in_boxes, nms3d
from torch_pointcloud.utils.data import DataKeys, PointCloudDataLoader
from torch_pointcloud.utils.metrics import mean_average_precision3d
from torch_pointcloud.utils.random import seed_everything, set_determinism
from torch_pointcloud.utils.types import Boxes3D, Detection3D

CPU_COUNT = os.cpu_count()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = CPU_COUNT // 2 if CPU_COUNT is not None else 0
SEED = 42
SCORE_THRESHOLD = 0.05
NMS_IOU = 0.25
MIN_POINTS = 5


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    set_determinism(tf32=False)

    model, info = create_model(args.model, task="detection", pretrained=True, return_info=True)
    assert isinstance(model, DETR3DDetection)
    model.to(args.device).eval()

    dataset: Dataset = SunRGBD(root=args.root, train=False, transform=info["transform"], download=args.download)
    if args.limit is not None:
        dataset = Subset(dataset, range(args.limit))

    dataloader = PointCloudDataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        cat_keys=[DataKeys.BOX, DataKeys.LABEL],
    )
    print(f"Test set: {len(dataset)} scenes")  # type: ignore[arg-type]
    metrics = evaluate(model, dataloader, args.device, iou_thresholds=args.ap_iou)

    print("\nResults:")
    for name, value in metrics.items():
        print(f"  {name:<10} {value * 100:.2f}")


@torch.no_grad()
def evaluate(
    model: DETR3DDetection,
    dataloader: PointCloudDataLoader,
    device: str,
    *,
    iou_thresholds: List[float],
) -> Dict[str, float]:
    all_preds: List[Detection3D] = []
    all_targets: List[Boxes3D] = []

    for data in tqdm(dataloader, desc="SUN RGB-D val"):
        pos = data[DataKeys.POS].to(device)
        box = data[DataKeys.BOX].to(device)
        gt_labels = data[DataKeys.LABEL].to(device)
        batch = data[DataKeys.BATCH].to(device)
        batch_box = data[DataKeys.BATCH_BOX].to(device)

        out = model(None, pos, batch)
        det = model.decode(out)
        boxes, obj, labels, det_batch = det["boxes"], det["scores"], det["labels"], det["batch"]
        counts = count_points_in_boxes(pos, boxes, pos_batch=batch, box_batch=det_batch)
        cand = (counts >= MIN_POINTS).nonzero(as_tuple=False).squeeze(-1)
        keep = cand[nms3d(boxes[cand], obj[cand], NMS_IOU, labels=labels[cand], batch=det_batch[cand])]
        keep = keep[obj[keep] > SCORE_THRESHOLD]
        # Indoor AP convention: score every surviving box against each class by its class probability.
        class_probs = out["sem_cls_prob"].reshape(-1, model.num_classes)[keep]
        all_preds.append(
            {
                "boxes": boxes[keep].repeat_interleave(model.num_classes, dim=0),
                "scores": (class_probs * obj[keep, None]).reshape(-1),
                "labels": torch.arange(model.num_classes, device=boxes.device).repeat(keep.numel()),
                "batch": det_batch[keep].repeat_interleave(model.num_classes),
            }
        )
        all_targets.append({"boxes": box, "labels": gt_labels, "batch": batch_box})

    return mean_average_precision3d(all_preds, all_targets, iou_thresholds=iou_thresholds)


def parse_args() -> Namespace:
    parser = ArgumentParser(description="3DETR SUN RGB-D detection AP benchmark.")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--root", type=str, default=DATA_DIR, help="Dataset root directory.")
    parser.add_argument("--model", type=str, default="3detr.sunrgbd.fair")
    parser.add_argument("--download", action="store_true", help="Download SUN RGB-D if missing.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--ap-iou", type=float, nargs="+", default=[0.25, 0.5])
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N scenes.")
    parser.add_argument("--device", type=str, default=DEVICE)
    return parser.parse_args()


if __name__ == "__main__":
    main()
