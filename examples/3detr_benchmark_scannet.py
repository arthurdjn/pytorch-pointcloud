r"""Evaluate 3DETR on ScanNet-V2 val with mAP@0.25 / mAP@0.5.

The whole benchmark is `create_model` -> `model` -> `model.decode` -> `mean_average_precision3d`: the
`ScanNet` dataset gives the upright cloud and per-point instance / segment labels, `Relabel` +
`InstanceToBox` derive the 18-class axis-aligned detection boxes (no detection-specific data export), the
registered transform samples to 40k points (3DETR's eval preprocessing), and the AP is the
dataset-agnostic 3D metric in `torch_pointcloud.utils.metrics`.

| Model                  | This script (mAP@0.25 / @0.50) | Reference (@0.25 / @0.50) |
| ---------------------- | ------------------------------ | ------------------------- |
| `3detr-m.scannet.fair` | 66.76 / 47.77                  | 65.0 / 47.0               |
| `3detr.scannet.fair`   | 62.17 / 38.60                  | 62.1 / 37.9               |

Usage:
    uv run --no-sync python examples/3detr_benchmark_scannet.py --root "/path/to/data"
"""

import os
from argparse import ArgumentParser, Namespace
from typing import Dict, List

import torch
from torch.utils.data import Dataset, Subset
from tqdm import tqdm

from torch_pointcloud import transforms as T
from torch_pointcloud.config import DATA_DIR
from torch_pointcloud.datasets import ScanNet
from torch_pointcloud.datasets.scannet import SCANNET_DETECTION_LABELS
from torch_pointcloud.models import create_model
from torch_pointcloud.models.detr3d import DETR3DDetection
from torch_pointcloud.utils.box3d import nms3d
from torch_pointcloud.utils.data import DataKeys, PointCloudDataLoader
from torch_pointcloud.utils.metrics import mean_average_precision3d
from torch_pointcloud.utils.random import seed_everything
from torch_pointcloud.utils.types import Boxes3D, Detection3D

CPU_COUNT = os.cpu_count()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = CPU_COUNT // 2 if CPU_COUNT is not None else 0
SEED = 42
NMS_IOU = 0.25


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    model, info = create_model(args.model, task="detection", pretrained=True, return_info=True)
    assert isinstance(model, DETR3DDetection)
    model.to(args.device).eval()

    # ScanNet ships no detection boxes: derive them from the per-instance labels, then sample as the model expects.
    transform = T.Compose(
        [
            T.Relabel(keys=DataKeys.SEGMENT, labels=SCANNET_DETECTION_LABELS, default=-1),
            T.InstanceToBox(ignore_index=-1),
            T.KeepItems(keys=[DataKeys.POS, DataKeys.BOX]),
            info["transform"],
        ]
    )
    dataset: Dataset = ScanNet(root=args.root, split="val", transform=transform)
    if args.limit is not None:
        dataset = Subset(dataset, range(args.limit))

    dataloader = PointCloudDataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, cat_keys=[DataKeys.BOX]
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

    for data in tqdm(dataloader, desc="ScanNet val"):
        pos = data[DataKeys.POS].to(device)
        box = data[DataKeys.BOX].to(device)
        batch = data[DataKeys.BATCH].to(device)
        batch_box = data[DataKeys.BATCH_BOX].to(device)

        out = model(None, pos, batch)
        det = model.decode(out)
        boxes, obj, labels, det_batch = det["boxes"], det["scores"], det["labels"], det["batch"]
        keep = nms3d(boxes, obj, NMS_IOU, labels=labels, batch=det_batch)
        class_probs = out["sem_cls_prob"].reshape(-1, model.num_classes)[keep]
        all_preds.append(
            {
                "boxes": boxes[keep].repeat_interleave(model.num_classes, dim=0),
                "scores": (class_probs * obj[keep, None]).reshape(-1),
                "labels": torch.arange(model.num_classes, device=boxes.device).repeat(keep.numel()),
                "batch": det_batch[keep].repeat_interleave(model.num_classes),
            }
        )
        full = torch.cat([box[:, :3], 2 * box[:, 3:6], box[:, 6:7]], dim=1)
        all_targets.append({"boxes": full, "labels": box[:, 7].long(), "batch": batch_box})

    return mean_average_precision3d(all_preds, all_targets, iou_thresholds=iou_thresholds)


def parse_args() -> Namespace:
    parser = ArgumentParser(description="3DETR ScanNet-V2 detection AP benchmark.")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--root", type=str, default=DATA_DIR, help="Dataset root directory.")
    parser.add_argument("--model", type=str, default="3detr-m.scannet.fair")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--ap-iou", type=float, nargs="+", default=[0.25, 0.5])
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N scenes.")
    parser.add_argument("--device", type=str, default=DEVICE)
    return parser.parse_args()


if __name__ == "__main__":
    main()
