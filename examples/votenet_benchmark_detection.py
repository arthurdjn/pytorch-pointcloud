"""Benchmark the VoteNet detectors on SUN RGB-D and ScanNet with the reference evaluation protocol.

NOTE: SUN RGB-D ground truth comes from the v1 labels of the `SunRGBD` dataset rather than the reference's exported arrays.

Results (mAP@0.25 / mAP@0.5):

    | Variant              | reference   | torch-pointcloud |
    | -------------------- | ----------- | ---------------- |
    | votenet.sunrgbd.fair | 57.7 / 32.0 | 58.81 / 34.15    |
    | votenet.scannet.fair | 58.6 / 33.5 | 57.65 / 34.10    |

Usage:
    uv run --no-sync python examples/votenet_benchmark_detection.py --model votenet.sunrgbd.fair
    uv run --no-sync python examples/votenet_benchmark_detection.py --model votenet.scannet.fair --limit 5
"""

import argparse
import os
from typing import Dict, List

import torch
from torch.utils.data import Dataset, Subset
from tqdm import tqdm

import torch_pointcloud.transforms as T
from torch_pointcloud.config import DATA_DIR
from torch_pointcloud.datasets import ScanNet, SunRGBD
from torch_pointcloud.datasets.scannet import SCANNET_DETECTION_LABELS
from torch_pointcloud.models import VoteNetDetection, create_model
from torch_pointcloud.utils.box3d import count_points_in_boxes, nms3d
from torch_pointcloud.utils.data import DataKeys, PointCloudDataLoader
from torch_pointcloud.utils.metrics import mean_average_precision3d
from torch_pointcloud.utils.random import seed_everything, set_determinism
from torch_pointcloud.utils.types import Boxes3D, Detection3D

CUDA_AVAILABLE = torch.cuda.is_available()
CPU_COUNT = os.cpu_count()
DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"
NUM_WORKERS = CPU_COUNT // 2 if CPU_COUNT is not None else 0
SEED = 42
SCORE_THRESHOLD = 0.05
NMS_IOU = 0.25
MIN_POINTS = 5
IOU_THRESHOLDS = [0.25, 0.5]


@torch.no_grad()
def evaluate(model: VoteNetDetection, dataloader: PointCloudDataLoader, device: str) -> Dict[str, float]:
    model.to(device).eval()
    num_classes = model.num_classes
    preds: List[Detection3D] = []
    targets: List[Boxes3D] = []

    for data in tqdm(dataloader, total=len(dataloader), desc="Testing"):
        data = {key: value.to(device) if torch.is_tensor(value) else value for key, value in data.items()}
        pos, batch = data[DataKeys.POS], data[DataKeys.BATCH]
        det = model.decode(model(data[DataKeys.X], pos, batch))
        boxes, scores, labels, det_batch = det["boxes"], det["scores"], det["labels"], det["batch"]
        counts = count_points_in_boxes(pos, boxes, pos_batch=batch, box_batch=det_batch)
        cand = (counts >= MIN_POINTS).nonzero(as_tuple=False).squeeze(-1)
        keep = cand[nms3d(boxes[cand], scores[cand], NMS_IOU, labels=labels[cand], batch=det_batch[cand])]
        keep = keep[scores[keep] > SCORE_THRESHOLD]
        # Indoor AP convention: score every surviving box against each class by its class probability.
        class_probs = det["class_probs"][keep]
        preds.append(
            {
                "boxes": boxes[keep].repeat_interleave(num_classes, dim=0),
                "scores": (class_probs * scores[keep, None]).reshape(-1),
                "labels": torch.arange(num_classes, device=device).repeat(keep.numel()),
                "batch": det_batch[keep].repeat_interleave(num_classes),
            }
        )
        targets.append({"boxes": data[DataKeys.BOX], "labels": data[DataKeys.LABEL], "batch": data[DataKeys.BATCH_BOX]})

    return mean_average_precision3d(preds, targets, iou_thresholds=IOU_THRESHOLDS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark VoteNet 3D detection on SUN RGB-D or ScanNet.")
    parser.add_argument(
        "--model", default="votenet.sunrgbd.fair", choices=["votenet.sunrgbd.fair", "votenet.scannet.fair"]
    )
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--root", default=DATA_DIR, help="Dataset root directory.")
    parser.add_argument("--seed", default=SEED, type=int)
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--num-workers", default=NUM_WORKERS, type=int)
    parser.add_argument("--limit", default=None, type=int, help="Evaluate at most this many scenes.")
    parser.add_argument("--download", action="store_true", help="Download SUN RGB-D if missing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    set_determinism(tf32=False)

    model, model_info = create_model(args.model, task="detection", pretrained=True, return_info=True)
    assert isinstance(model, VoteNetDetection)

    dataset: Dataset
    if "sunrgbd" in args.model:
        print(f"Benchmarking model {args.model!r} on SUN RGB-D!")
        dataset = SunRGBD(root=args.root, train=False, transform=model_info["transform"], download=args.download)
    else:
        print(f"Benchmarking model {args.model!r} on ScanNet!")
        transform = T.Compose(
            [
                T.Relabel(keys=DataKeys.SEGMENT, labels=SCANNET_DETECTION_LABELS, default=-1),
                T.InstanceToBox(ignore_index=-1),
                T.KeepItems(keys=[DataKeys.POS, DataKeys.BOX, DataKeys.LABEL]),
                model_info["transform"],
            ]
        )
        dataset = ScanNet(root=args.root, split="val", transform=transform)
    if args.limit is not None:
        n = min(int(args.limit), len(dataset))
        dataset = Subset(dataset, range(n))
        print(f"Evaluating on a subset of the first {n} scenes.")

    dataloader = PointCloudDataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        cat_keys=[DataKeys.BOX, DataKeys.LABEL],
    )

    print(f"Test set: {len(dataset)} scenes")
    metrics = evaluate(model, dataloader, args.device)
    print("\nResults:")
    for key, value in metrics.items():
        print(f"  {key:<24} {value * 100:.2f}")


if __name__ == "__main__":
    main()
