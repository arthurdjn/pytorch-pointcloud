"""Benchmark 3DETR on ScanNet with its evaluation protocol."""

import os
from argparse import ArgumentParser, Namespace
from typing import cast

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

import torch_pointcloud.transforms as T
from torch_pointcloud.config import DATA_DIR
from torch_pointcloud.datasets import ScanNet
from torch_pointcloud.datasets.scannet import SCANNET_DETECTION_LABELS
from torch_pointcloud.metrics import box_average_precision, box_matches
from torch_pointcloud.metrics.detection import BoxMatches
from torch_pointcloud.models import ThreeDETRDetection, create_model, model_info
from torch_pointcloud.ops.box3d import count_points_in_boxes, nms3d
from torch_pointcloud.utils.data import DataKeys, PointCloudDataLoader
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


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    set_determinism(tf32=False)

    print(f"Benchmarking model {args.model!r} on ScanNet!")
    model = configure_model(args)
    dataloader = configure_dataloader(args)

    print(f"Test set: {len(dataloader.dataset)} scenes")  # type: ignore[arg-type]
    metrics = evaluate(model, dataloader, args.device)
    print("\nResults:")
    for key, value in metrics.items():
        print(f"  {key:<24} {value * 100:.2f}")


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Benchmark 3DETR 3D detection on ScanNet.")
    parser.add_argument(
        "--model", default="3detr-m.scannet.fair", choices=["3detr.scannet.fair", "3detr-m.scannet.fair"]
    )
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--root", default=DATA_DIR, help="Dataset root directory.")
    parser.add_argument("--seed", default=SEED, type=int)
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--num-workers", default=NUM_WORKERS, type=int)
    parser.add_argument("--limit", default=None, type=int, help="Evaluate at most this many scenes.")
    return parser.parse_args()


def configure_transform(args: Namespace) -> T.Transform:
    """The registered inference transform of the model, after the ScanNet instances are turned into boxes."""
    transform = model_info(args.model, task="detection")["transform"]
    if transform is None:
        raise ValueError(f"{args.model!r} registers no transform.")

    return T.Compose(
        [
            T.Relabel(keys=DataKeys.SEGMENT, labels=SCANNET_DETECTION_LABELS, default=-1),
            T.InstanceToBox(ignore_index=-1),
            T.KeepItems(keys=[DataKeys.POS, DataKeys.BOX, DataKeys.LABEL]),
            transform,
        ]
    )


def configure_dataset(args: Namespace) -> Dataset:
    dataset: Dataset
    transform = configure_transform(args)
    dataset = ScanNet(root=args.root, split="val", transform=transform)

    if args.limit is not None:
        n = min(int(args.limit), len(dataset))
        dataset = Subset(dataset, range(n))
        print(f"Evaluating on a subset of the first {n} scenes.")

    return dataset


def configure_dataloader(args: Namespace) -> DataLoader:
    dataset = configure_dataset(args)
    return PointCloudDataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        cat_keys=[DataKeys.BOX, DataKeys.LABEL],
    )


def configure_model(args: Namespace) -> ThreeDETRDetection:
    model = cast(ThreeDETRDetection, create_model(args.model, task="detection", pretrained=True))
    return model


@torch.no_grad()
def evaluate(model: ThreeDETRDetection, dataloader: DataLoader, device: str) -> dict[str, float]:
    model.to(device).eval()
    num_classes = model.num_classes
    matches: list[BoxMatches] = []

    for data in tqdm(dataloader, total=len(dataloader), desc="Testing"):
        data = {key: value.to(device) if torch.is_tensor(value) else value for key, value in data.items()}
        pos, batch = data[DataKeys.POS], data[DataKeys.BATCH]
        det = model.decode(model(None, pos, batch))
        boxes, scores, labels, det_batch = det["boxes"], det["scores"], det["labels"], det["batch"]
        counts = count_points_in_boxes(pos, boxes, pos_batch=batch, box_batch=det_batch)
        cand = (counts >= MIN_POINTS).nonzero(as_tuple=False).squeeze(-1)
        keep = cand[nms3d(boxes[cand], scores[cand], NMS_IOU, labels=labels[cand], batch=det_batch[cand])]
        keep = keep[scores[keep] > SCORE_THRESHOLD]
        # Indoor AP convention: score every surviving box against each class by its class probability.
        class_probs = det["class_probs"][keep]
        preds: Detection3D = {
            "boxes": boxes[keep].repeat_interleave(num_classes, dim=0),
            "scores": (class_probs * scores[keep, None]).reshape(-1),
            "labels": torch.arange(num_classes, device=device).repeat(keep.numel()),
            "batch": det_batch[keep].repeat_interleave(num_classes),
        }
        target: Boxes3D = {
            "boxes": data[DataKeys.BOX],
            "labels": data[DataKeys.LABEL],
            "batch": data[DataKeys.BATCH_BOX],
        }
        matches.append(box_matches(preds, target))

    return {
        f"mAP@{threshold:g}": box_average_precision(matches, iou_threshold=threshold) for threshold in IOU_THRESHOLDS
    }


if __name__ == "__main__":
    main()
