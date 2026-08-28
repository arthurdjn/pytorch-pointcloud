"""Benchmark PointRCNN on KITTI with the reference evaluation protocol.

Results (Car / Pedestrian / Cyclist / mAP, R11):

    | Variant                   | reference                     | torch-pointcloud |
    | ------------------------- | ----------------------------- | ---------------- |
    | pointrcnn.kitti.openpcdet | 78.70 / 54.41 / 72.11 / 68.41 |                  |

Usage:
    uv run --no-sync python examples/pointrcnn_benchmark_detection.py --split-file /path/to/ImageSets/val.txt
"""

import argparse
import os
from typing import Dict, List

import torch
from torch.utils.data import Dataset, Subset
from tqdm import tqdm

import torch_pointcloud.transforms as T
from torch_pointcloud.config import DATA_DIR
from torch_pointcloud.datasets import KITTI
from torch_pointcloud.datasets.kitti import KITTI_CLASSES
from torch_pointcloud.models import DetectionModel, create_model
from torch_pointcloud.utils.box3d import nms3d, projected_ignore_mask
from torch_pointcloud.utils.data import DataKeys, PointCloudDataLoader
from torch_pointcloud.utils.metrics import average_precision3d
from torch_pointcloud.utils.random import seed_everything, set_determinism
from torch_pointcloud.utils.types import Boxes3D, Detection3D

CUDA_AVAILABLE = torch.cuda.is_available()
CPU_COUNT = os.cpu_count()
DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"
NUM_WORKERS = CPU_COUNT // 2 if CPU_COUNT is not None else 0
SEED = 42
SCORE_THRESHOLD = 0.1
NMS_IOU = 0.1
KITTI_DETECTION_CLASSES = ("Car", "Pedestrian", "Cyclist")
KITTI_IOU = {0: 0.7, 1: 0.5, 2: 0.5}
# Van / Person_sitting and harder-than-moderate boxes (occlusion > 1, truncation > 0.3, 2D height < 25 px) are ignored.
KITTI_TRANSFORM = T.RelabelBoxes(
    keys=(DataKeys.BOX, DataKeys.LABEL, DataKeys.TRUNCATION, DataKeys.OCCLUSION, DataKeys.BBOX_HEIGHT),
    mapping={KITTI_CLASSES.index(name): index for index, name in enumerate(KITTI_DETECTION_CLASSES)},
    ignore_mapping={KITTI_CLASSES.index("Van"): 0, KITTI_CLASSES.index("Person_sitting"): 1},
    ignore_fields={DataKeys.OCCLUSION: (None, 1), DataKeys.TRUNCATION: (None, 0.3), DataKeys.BBOX_HEIGHT: (25, None)},
)


@torch.no_grad()
def evaluate(model: DetectionModel, dataloader: PointCloudDataLoader, device: str) -> Dict[str, float]:
    model.to(device).eval()
    preds: List[Detection3D] = []
    targets: List[Boxes3D] = []

    for data in tqdm(dataloader, total=len(dataloader), desc="Testing"):
        data = {key: value.to(device) if torch.is_tensor(value) else value for key, value in data.items()}
        det = model.decode(model(data[DataKeys.X], data[DataKeys.POS], data[DataKeys.BATCH]))
        keep = det["scores"] >= SCORE_THRESHOLD
        boxes, scores, labels, batch = det["boxes"][keep], det["scores"][keep], det["labels"][keep], det["batch"][keep]
        idx = nms3d(boxes, scores, NMS_IOU, batch=batch, rotated=True)
        boxes, scores, labels, batch = boxes[idx], scores[idx], labels[idx], batch[idx]
        ignore_mask = projected_ignore_mask(boxes, data[DataKeys.CALIB][batch], data[DataKeys.IMAGE_SHAPE][batch])
        preds.append(
            {
                "boxes": boxes.cpu(),
                "scores": scores.cpu(),
                "labels": labels.cpu(),
                "batch": batch.cpu(),
                "ignore_mask": ignore_mask.cpu(),
            }
        )
        targets.append(
            {
                "boxes": data[DataKeys.BOX].cpu(),
                "labels": data[DataKeys.LABEL].cpu(),
                "batch": data[DataKeys.BATCH_BOX].cpu(),
                "ignore_mask": data["ignore_mask"].cpu(),
            }
        )

    return average_precision3d(
        preds,
        targets,
        iou_per_class=KITTI_IOU,
        class_names=KITTI_DETECTION_CLASSES,
        interpolation="r11",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark PointRCNN 3D detection on KITTI.")
    parser.add_argument("--model", default="pointrcnn.kitti.openpcdet", help="Registered detection model name")
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--root", default=DATA_DIR, help="Dataset root directory.")
    parser.add_argument("--split-file", default=None, help="KITTI frame-id list (e.g. ImageSets/val.txt).")
    parser.add_argument("--seed", default=SEED, type=int)
    parser.add_argument("--batch-size", default=2, type=int)
    parser.add_argument("--num-workers", default=NUM_WORKERS, type=int)
    parser.add_argument("--limit", default=None, type=int, help="Evaluate at most this many frames.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    set_determinism(tf32=False)

    print(f"Benchmarking model {args.model!r} on KITTI!")
    model, model_info = create_model(args.model, task="detection", pretrained=True, return_info=True)
    assert isinstance(model, DetectionModel)

    dataset: Dataset = KITTI(
        root=args.root,
        train=True,
        split_file=args.split_file,
        fov=True,
        return_calib=True,
        transform=T.Compose([KITTI_TRANSFORM, model_info["transform"]]),
    )
    if args.limit is not None:
        n = min(int(args.limit), len(dataset))  # type: ignore[arg-type]
        dataset = Subset(dataset, range(n))
        print(f"Evaluating on a subset of the first {n} frames.")

    dataloader = PointCloudDataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        stack_keys=[DataKeys.CALIB, DataKeys.IMAGE_SHAPE],
        cat_keys=[DataKeys.BOX],
    )

    print(f"Test set: {len(dataset)} frames")  # type: ignore[arg-type]
    metrics = evaluate(model, dataloader, args.device)
    print("\nResults:")
    for key, value in metrics.items():
        print(f"  {key:<24} {value * 100:.2f}")


if __name__ == "__main__":
    main()
