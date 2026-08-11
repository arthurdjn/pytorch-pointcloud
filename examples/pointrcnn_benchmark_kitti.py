"""Evaluate `pointrcnn.kitti.openpcdet` on KITTI with per-class 3D AP.

`KITTI` -> `RelabelBoxes` -> `PointCloudDataLoader` -> model -> `model.decode` -> `nms3d` -> `average_precision3d`.

`KITTI` returns the raw annotated boxes; `RelabelBoxes` maps them to the 3 detection classes and flags
the `Van` / `Person_sitting` neighbours (ignore regions for `Car` / `Pedestrian`) plus harder-than-moderate
boxes (occlusion > 1, truncation > 0.3, 2D height < 25 px) as ignore regions that excuse only their own
class. Predictions whose 2D box (the 3D box projected with the per-frame calib, clipped to the image)
is under 25 px tall are likewise excluded from scoring, the reference `ignored_dt` rule. Scoring is the
in-repo per-class R11 AP at the official KITTI IoUs (Car@0.7, Pedestrian/Cyclist@0.5) after
class-agnostic `nms3d`, matching the reference protocol up to two residuals: matching is
detection-centric (each prediction greedily takes its best unused GT, not the reference GT-centric
assignment) and the NMS is axis-aligned rather than rotated-BEV; both only act on borderline overlaps.
For Easy / Hard, change the `RelabelBoxes` ignore rule and `MIN_BBOX_HEIGHT` (Easy: occlusion <= 0,
truncation <= 0.15, height >= 40 px; Hard: occlusion <= 2, truncation <= 0.5, height >= 25 px). The raw
split is read from `<root>/KITTI/raw/<split>/` and cached to `.npy` on first use. Pass `--split-file
ImageSets/val.txt` for the val split; `raw/image_2/` enables the front-camera FOV filter and the
min-height projection (else pass `--no-fov`).

PointRCNN is point-based (no voxelization): its registered transform crops the points to the point cloud
range and samples 16384 of them, then `forward(x, pos, batch)` runs the two-stage detector directly.

Results (KITTI val, FOV, moderate difficulty, per-class 3D AP):

    | Source                             | Car   | Ped   | Cyc   | mAP   |
    | ---------------------------------- | ----- | ----- | ----- | ----- |
    | OpenPCDet model zoo (val, mod R11) | 78.70 | 54.41 | 72.11 | 68.41 |
    | torch-pointcloud                   | 76.16 | 49.66 | 64.87 | 63.56 |

The reference row is the OpenPCDet zoo val moderate R11 entry (which matches the paper's val table); the
paper's KITTI test-split numbers are not like-for-like with this val protocol. The torch-pointcloud row
is the all-point-AP measurement predating the R11 / min-height protocol.

Usage:
    uv run --no-sync python examples/pointrcnn_benchmark_kitti.py \
        --root "/path/to/parent" --split-file /path/to/ImageSets/val.txt
"""

import os
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

import torch_pointcloud.transforms as T
from torch_pointcloud.config import DATA_DIR
from torch_pointcloud.datasets import KITTI
from torch_pointcloud.datasets.kitti import (
    KITTI_CLASSES,
    _read_image_shape,
    lidar_to_rect,
    load_kitti_calib,
    rect_to_img,
)
from torch_pointcloud.models import DetectionModel, create_model
from torch_pointcloud.utils.box3d import box_corners, nms3d
from torch_pointcloud.utils.data import DataKeys, PointCloudDataLoader
from torch_pointcloud.utils.metrics import average_precision3d
from torch_pointcloud.utils.random import seed_everything, set_determinism
from torch_pointcloud.utils.types import Boxes3D, Detection3D

CPU_COUNT = os.cpu_count()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = CPU_COUNT // 2 if CPU_COUNT is not None else 0
# Official KITTI 3D IoU thresholds: Car @ 0.7, Pedestrian / Cyclist @ 0.5.
KITTI_DETECTION_CLASSES = ("Car", "Pedestrian", "Cyclist")
KITTI_DETECTION_MAPPING = {KITTI_CLASSES.index(name): i for i, name in enumerate(KITTI_DETECTION_CLASSES)}
KITTI_IGNORE_MAPPING = {
    KITTI_CLASSES.index("Van"): KITTI_DETECTION_CLASSES.index("Car"),
    KITTI_CLASSES.index("Person_sitting"): KITTI_DETECTION_CLASSES.index("Pedestrian"),
}
KITTI_IOU = {0: 0.7, 1: 0.5, 2: 0.5}
SCORE_THRESHOLD = 0.1
NMS_IOU = 0.1
MIN_BBOX_HEIGHT = 25.0  # moderate difficulty: smaller projected predictions are excluded from scoring


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    set_determinism(tf32=False)

    model, info = create_model(args.model, task="detection", pretrained=True, return_info=True)
    assert isinstance(model, DetectionModel)
    model.to(args.device).eval()

    relabel = T.RelabelBoxes(
        keys=(DataKeys.BOX, DataKeys.LABEL, DataKeys.TRUNCATION, DataKeys.OCCLUSION, DataKeys.BBOX_HEIGHT),
        mapping=KITTI_DETECTION_MAPPING,
        ignore_mapping=KITTI_IGNORE_MAPPING,
        ignore_fields={
            DataKeys.OCCLUSION: (None, 1),
            DataKeys.TRUNCATION: (None, 0.3),
            DataKeys.BBOX_HEIGHT: (25, None),
        },
    )
    dataset = KITTI(
        root=args.root,
        split=args.split,
        split_file=args.split_file,
        fov=not args.no_fov,
        transform=T.Compose([relabel, info["transform"]]),
    )
    loader = PointCloudDataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        cat_keys=[DataKeys.BOX],
    )

    print(f"Benchmarking {args.model!r} on KITTI ({len(dataset)} frames)!")
    metrics = evaluate(model, loader, args.device, dataset.raw_split_dir)
    print("\nResults (per-class 3D AP):")
    for name, value in metrics.items():
        print(f"  {name:<12} {value * 100:.2f}")


def projected_ignore_mask(boxes: Tensor, batch: Tensor, frames: Sequence[str], raw_dir: Path) -> Tensor:
    mask = torch.zeros(boxes.shape[0], dtype=torch.bool)
    for scene, frame in enumerate(frames):
        rows = batch == scene
        image_path = raw_dir / "image_2" / f"{frame}.png"
        if not bool(rows.any()) or not image_path.exists():
            continue
        calib = load_kitti_calib(raw_dir / "calib" / f"{frame}.txt")
        image_height, _ = _read_image_shape(image_path)
        corners = box_corners(boxes[rows]).reshape(-1, 3).numpy()
        pixels, _ = rect_to_img(lidar_to_rect(corners, calib), calib)
        y = np.clip(pixels.reshape(-1, 8, 2)[..., 1], 0, image_height - 1)
        mask[rows] = torch.from_numpy((y.max(axis=1) - y.min(axis=1)) < MIN_BBOX_HEIGHT)
    return mask


@torch.no_grad()
def evaluate(model: DetectionModel, loader: PointCloudDataLoader, device: str, raw_dir: Path) -> Dict[str, float]:
    preds: List[Detection3D] = []
    targets: List[Boxes3D] = []
    for data in tqdm(loader, desc="KITTI"):
        out = model(
            data[DataKeys.X].to(device),
            data[DataKeys.POS].to(device),
            data[DataKeys.BATCH].to(device),
        )
        det = model.decode(out)
        boxes, scores, labels, batch = det["boxes"], det["scores"], det["labels"], det["batch"]
        idx = nms3d(boxes, scores, NMS_IOU, batch=batch)
        boxes, scores, labels, batch = boxes[idx], scores[idx], labels[idx], batch[idx]
        keep = scores >= SCORE_THRESHOLD
        boxes, scores, labels, batch = boxes[keep].cpu(), scores[keep].cpu(), labels[keep].cpu(), batch[keep].cpu()
        preds.append(
            {
                "boxes": boxes,
                "scores": scores,
                "labels": labels,
                "batch": batch,
                "ignore_mask": projected_ignore_mask(boxes, batch, data[DataKeys.FRAME], raw_dir),
            }
        )
        targets.append(
            {
                "boxes": data[DataKeys.BOX],
                "labels": data[DataKeys.LABEL],
                "batch": data[DataKeys.BATCH_BOX],
                "ignore_mask": data["ignore_mask"],
            }
        )
    return average_precision3d(
        preds, targets, iou_per_class=KITTI_IOU, class_names=KITTI_DETECTION_CLASSES, interpolation="r11"
    )


def parse_args() -> Namespace:
    parser = ArgumentParser(description="PointRCNN KITTI 3D detection AP benchmark.")
    parser.add_argument("--model", type=str, default="pointrcnn.kitti.openpcdet")
    parser.add_argument("--root", type=str, default=DATA_DIR, help="Parent directory containing KITTI/.")
    parser.add_argument("--split", type=str, default="training")
    parser.add_argument("--split-file", type=str, default=None, help="Frame-id list (e.g. ImageSets/val.txt).")
    parser.add_argument("--no-fov", action="store_true", help="Disable the front-camera FOV filter.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--device", type=str, default=DEVICE)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    main()
