"""Benchmark TR3D on ScanNet, SUN RGB-D or S3DIS Area 5 with its evaluation protocol.

NOTE: the S3DIS reference 72.25 / 51.66 mAP@0.25 / 0.5 is the last validation of the checkpoint's training log. We ran
the checkpoint through mmdet3d v1.4.0's own TR3D model, `predict_by_feat` and `indoor_eval` (MinkowskiEngine 0.5.4 on
CPU, mmcv's NMS on GPU) on the Area 5 rooms of its `data/s3dis/indoor3d_util.py` export, seed 42: 71.70 / 45.69.
"""

import os
from argparse import ArgumentParser, Namespace
from typing import cast

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torch_geometric.nn.pool.select.topk import topk
from tqdm import tqdm

import torch_pointcloud.transforms as T
from torch_pointcloud.config import DATA_DIR
from torch_pointcloud.datasets import S3DIS, ScanNet, SunRGBD
from torch_pointcloud.datasets.s3dis import S3DIS_DETECTION_LABELS
from torch_pointcloud.datasets.scannet import SCANNET_DETECTION_LABELS
from torch_pointcloud.metrics import box_average_precision, box_matches
from torch_pointcloud.metrics.detection import BoxMatches
from torch_pointcloud.models import TR3DDetection, create_model, model_info
from torch_pointcloud.ops.box3d import nms3d
from torch_pointcloud.utils.data import DataKeys, PointCloudDataLoader
from torch_pointcloud.utils.random import seed_everything, set_determinism
from torch_pointcloud.utils.types import Boxes3D, Detection3D

CUDA_AVAILABLE = torch.cuda.is_available()
CPU_COUNT = os.cpu_count()
DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"
NUM_WORKERS = CPU_COUNT // 2 if CPU_COUNT is not None else 0
SEED = 42
NMS_PRE = 1000
SCORE_THRESHOLD = 0.01
NMS_IOU = 0.5
IOU_THRESHOLDS = [0.25, 0.5]
MODELS = {"scannet": "tr3d.scannet.mmdet3d", "sunrgbd": "tr3d.sunrgbd.mmdet3d", "s3dis": "tr3d.s3dis-area5.mmdet3d"}


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    set_determinism(tf32=False)

    print(f"Benchmarking model {args.model!r} on {args.dataset}!")
    model = configure_model(args)
    dataloader = configure_dataloader(args)

    print(f"Test set: {len(dataloader.dataset)} scenes")  # type: ignore[arg-type]
    metrics = evaluate(model, dataloader, args.device)
    print("\nResults:")
    for key, value in metrics.items():
        print(f"  {key:<24} {value * 100:.2f}")


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Benchmark TR3D 3D detection on an indoor dataset.")
    parser.add_argument("--dataset", default="scannet", choices=list(MODELS))
    parser.add_argument("--model", default=None, help="Registered detection model name (default: the dataset's).")
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--root", default=DATA_DIR, help="Dataset root directory.")
    parser.add_argument("--seed", default=SEED, type=int)
    parser.add_argument("--batch-size", default=4, type=int)
    parser.add_argument("--num-workers", default=NUM_WORKERS, type=int)
    parser.add_argument("--limit", default=None, type=int, help="Evaluate at most this many scenes.")
    args = parser.parse_args()
    args.model = args.model or MODELS[args.dataset]
    return args


def configure_transform(args: Namespace) -> T.Transform:
    """The registered inference transform of the model, after the ground-truth boxes are derived."""
    transform = model_info(args.model, task="detection")["transform"]
    if transform is None:
        raise ValueError(f"{args.model!r} registers no transform.")

    if args.dataset == "scannet":
        steps = [
            T.Relabel(keys=DataKeys.SEGMENT, labels=SCANNET_DETECTION_LABELS, default=-1),
            T.InstanceToBox(ignore_index=-1),
        ]
    elif args.dataset == "s3dis":
        steps = [
            # The reference rooms start at their minimum corner.
            T.Shift(keys=DataKeys.POS, method="min"),
            T.Relabel(keys=DataKeys.SEGMENT, labels=S3DIS_DETECTION_LABELS, default=-1),
            T.InstanceToBox(ignore_index=-1),
        ]
    else:
        # The reference SUN RGB-D scenes keep 50000 random points.
        steps = [T.RandomSample(keys=[DataKeys.POS, DataKeys.COLOR], num_samples=50000)]

    return T.Compose(
        [*steps, T.KeepItems(keys=[DataKeys.POS, DataKeys.COLOR, DataKeys.BOX, DataKeys.LABEL]), transform]
    )


def configure_dataset(args: Namespace) -> Dataset:
    dataset: Dataset
    transform = configure_transform(args)
    if args.dataset == "scannet":
        dataset = ScanNet(root=args.root, split="val", transform=transform)
    elif args.dataset == "s3dis":
        dataset = S3DIS(root=args.root, areas=["Area_5"], transform=transform)
    else:
        dataset = SunRGBD(root=args.root, train=False, transform=transform)

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


def configure_model(args: Namespace) -> TR3DDetection:
    model = cast(TR3DDetection, create_model(args.model, task="detection", pretrained=True))
    return model


@torch.no_grad()
def evaluate(model: TR3DDetection, dataloader: DataLoader, device: str) -> dict[str, float]:
    model.to(device).eval()
    matches: list[BoxMatches] = []

    for data in tqdm(dataloader, total=len(dataloader), desc="Testing"):
        data = {key: value.to(device) if torch.is_tensor(value) else value for key, value in data.items()}
        out = model(data[DataKeys.X], data[DataKeys.POS], data[DataKeys.BATCH])
        det = model.decode(out)
        # Keep the best locations of every scene, then score each of them against every class.
        keep = topk(det["scores"], NMS_PRE, det["batch"])
        box, label = (det["class_probs"][keep] > SCORE_THRESHOLD).nonzero(as_tuple=True)
        boxes, batch = det["boxes"][keep][box], det["batch"][keep][box]
        scores = det["class_probs"][keep][box, label]
        kept = nms3d(boxes, scores, NMS_IOU, labels=label, batch=batch, rotated=True)
        preds: Detection3D = {"boxes": boxes[kept], "scores": scores[kept], "labels": label[kept], "batch": batch[kept]}
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
