r"""Evaluate `votenet.sunrgbd.fair` on SUN RGB-D val with mAP@0.25 / mAP@0.5.

The whole benchmark is `create_model` -> `model.predict` -> `mean_average_precision3d`: the `SunRGBD`
dataset reconstructs the upright cloud and oriented boxes, the registered transform adds the
floor-relative height feature and samples to 20k points, the model decodes oriented boxes (per-class
3D NMS, empty-box removal), and the AP is the dataset-agnostic 3D metric in `torch_pointcloud.utils.metrics`.

| Model                       | This script (mAP@0.25 / @0.50) | Reference (@0.25 / @0.50) |
| --------------------------- | ------------------------------ | ------------------------- |
| `votenet.sunrgbd.fair` | 59.04 / 34.27                  | 57.7 / 32.0               |

Usage:
    uv run --no-sync python examples/votenet_benchmark_sunrgbd.py --root /path/to/data
"""

import os
from argparse import ArgumentParser, Namespace
from typing import Dict, List

import torch
from torch.utils.data import Dataset, Subset
from tqdm import tqdm

from torch_pointcloud.config import DATA_DIR
from torch_pointcloud.datasets import SunRGBD
from torch_pointcloud.models import VoteNetDetection, create_model
from torch_pointcloud.utils.box3d import count_points_in_boxes, nms3d
from torch_pointcloud.utils.data import DataKeys, PointCloudDataLoader
from torch_pointcloud.utils.metrics import mean_average_precision3d
from torch_pointcloud.utils.random import seed_everything
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

    model, info = create_model(args.model, task="detection", pretrained=True, return_info=True)
    assert isinstance(model, VoteNetDetection)
    model.to(args.device).eval()

    dataset: Dataset = SunRGBD(root=args.root, split="val", transform=info["transform"], download=args.download)
    if args.limit is not None:
        dataset = Subset(dataset, range(args.limit))

    dataloader = PointCloudDataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        cat_keys=[DataKeys.BOX, DataKeys.CLASS],
    )
    print(f"Test set: {len(dataset)} scenes")  # type: ignore[arg-type]
    metrics = evaluate(model, dataloader, args.device, iou_thresholds=args.ap_iou)

    print("\nResults:")
    for name, value in metrics.items():
        print(f"  {name:<10} {value * 100:.2f}")


@torch.no_grad()
def evaluate(
    model: VoteNetDetection,
    dataloader: PointCloudDataLoader,
    device: str,
    *,
    iou_thresholds: List[float],
) -> Dict[str, float]:
    all_preds: List[Detection3D] = []
    all_targets: List[Boxes3D] = []

    for data in tqdm(dataloader, desc="SUN RGB-D val"):
        x = data[DataKeys.X].to(device)
        pos = data[DataKeys.POS].to(device)
        box = data[DataKeys.BOX].to(device)
        gt_labels = data[DataKeys.CLASS].to(device)
        batch = data[DataKeys.BATCH].to(device)
        batch_box = data[DataKeys.BATCH_BOX].to(device)

        out = model(x, pos, batch)
        det = model.decode(out)
        boxes, obj, labels, det_batch = det["boxes"], det["scores"], det["labels"], det["batch"]
        counts = count_points_in_boxes(pos, boxes, pos_batch=batch, box_batch=det_batch)
        cand = (counts >= MIN_POINTS).nonzero(as_tuple=False).squeeze(-1)
        keep = cand[nms3d(boxes[cand], obj[cand], NMS_IOU, labels=labels[cand], batch=det_batch[cand])]
        keep = keep[obj[keep] > SCORE_THRESHOLD]
        # Indoor AP convention: score every surviving box against each class by its class probability.
        class_probs = out["sem_cls_scores"].softmax(-1).reshape(-1, model.num_classes)[keep]
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
    parser = ArgumentParser(description="VoteNet SUN RGB-D detection AP benchmark.")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--root", type=str, default=DATA_DIR, help="Dataset root directory.")
    parser.add_argument("--model", type=str, default="votenet.sunrgbd.fair")
    parser.add_argument("--download", action="store_true", help="Download SUN RGB-D if missing.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--ap-iou", type=float, nargs="+", default=[0.25, 0.5])
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N scenes.")
    parser.add_argument("--device", type=str, default=DEVICE)
    return parser.parse_args()


if __name__ == "__main__":
    main()
