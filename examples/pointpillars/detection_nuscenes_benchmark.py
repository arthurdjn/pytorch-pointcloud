"""Benchmark PointPillars on nuScenes with the official detection metrics."""

import os
from argparse import ArgumentParser, Namespace
from typing import Optional

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

import torch_pointcloud.transforms as T
from torch_pointcloud.config import DATA_DIR
from torch_pointcloud.datasets import NuScenes, NuScenesMini
from torch_pointcloud.datasets.nuscenes import NUSCENES_DETECTION_CLASSES, velocity_attributes
from torch_pointcloud.metrics import nuscenes_detection_metrics
from torch_pointcloud.models import DetectionModel, create_model, model_info
from torch_pointcloud.ops.box3d import nms3d
from torch_pointcloud.utils.data import DataKeys, PointCloudDataLoader
from torch_pointcloud.utils.random import seed_everything, set_determinism
from torch_pointcloud.utils.types import Boxes3D, Detection3D

CUDA_AVAILABLE = torch.cuda.is_available()
CPU_COUNT = os.cpu_count()
DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"
NUM_WORKERS = CPU_COUNT // 2 if CPU_COUNT is not None else 0
SEED = 42

SCORE_THRESHOLD = 0.1
NMS_IOU = 0.2


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    set_determinism(tf32=False)

    print(f"Benchmarking model {args.model!r} on nuScenes ({args.split})!")
    model = configure_model(args)
    dataloader = configure_dataloader(args)

    print(f"Test set: {len(dataloader.dataset)} frames")  # type: ignore[arg-type]
    metrics = evaluate(
        model,
        dataloader,
        args.device,
        score_threshold=args.score_threshold,
        nms_pre=args.nms_pre,
        max_detections=args.max_detections,
        multi_label=args.multi_label,
        rotated_nms=args.rotated_nms,
    )
    print("\nResults:")
    for key, value in metrics.items():
        print(f"  {key:<24} {value:.4f}")


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Benchmark PointPillars 3D detection on nuScenes.")
    parser.add_argument(
        "--model", default="pointpillars-multihead.nuscenes.openpcdet", help="Registered detection model name"
    )
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--root", default=DATA_DIR, help="Dataset root directory.")
    parser.add_argument("--split", default="val", choices=("val", "mini"), help="nuScenes `val`, or the mini release.")
    parser.add_argument("--max-sweeps", default=10, type=int, help="nuScenes LiDAR sweeps per keyframe.")
    parser.add_argument("--seed", default=SEED, type=int)
    parser.add_argument("--batch-size", default=4, type=int)
    parser.add_argument("--num-workers", default=NUM_WORKERS, type=int)
    parser.add_argument("--limit", default=None, type=int, help="Evaluate at most this many frames.")
    parser.add_argument("--score-threshold", default=SCORE_THRESHOLD, type=float, help="Lowest score kept.")
    parser.add_argument(
        "--nms-pre", default=None, type=int, help="Anchors kept per scene and head level before the NMS, by score."
    )
    parser.add_argument("--max-detections", default=None, type=int, help="Detections kept per scene after the NMS.")
    parser.add_argument(
        "--multi-label",
        action="store_true",
        help="Give every anchor to each class it scores above the threshold in, instead of its top class only.",
    )
    parser.add_argument("--rotated-nms", action="store_true", help="Suppress on the rotated BEV IoU.")
    return parser.parse_args()


def configure_dataset(args: Namespace) -> Dataset:
    dataset: Dataset
    if args.split == "mini":
        transform = configure_transform(args)
        dataset = NuScenesMini(root=args.root, max_sweeps=args.max_sweeps, transform=transform)
    else:
        transform = configure_transform(args)
        dataset = NuScenes(root=args.root, split="val", max_sweeps=args.max_sweeps, transform=transform)

    if args.limit is not None:
        n = min(int(args.limit), len(dataset))
        dataset = Subset(dataset, range(n))
        print(f"Evaluating on a subset of the first {n} frames.")

    return dataset


def configure_dataloader(args: Namespace) -> DataLoader:
    dataset = configure_dataset(args)
    return PointCloudDataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        cat_keys=[DataKeys.BOX, DataKeys.POS_VOXEL],
    )


def configure_model(args: Namespace) -> DetectionModel:
    model = create_model(args.model, task="detection", pretrained=True)
    assert isinstance(model, DetectionModel)
    return model


def configure_transform(args: Namespace) -> T.Transform:
    """The registered inference transform of the model."""
    transform = model_info(args.model, task="detection")["transform"]
    assert transform is not None
    return transform


def top_per_group(scores: Tensor, group: Tensor, k: int) -> Tensor:
    """Indices of the `k` best scores of every group, group by group and best first."""
    order = scores.argsort(descending=True, stable=True)
    order = order[group[order].argsort(stable=True)]
    sorted_group = group[order]
    rank = torch.arange(order.numel(), device=order.device) - torch.searchsorted(sorted_group, sorted_group)
    return order[rank < k]


def select_detections(
    det: Detection3D,
    score_threshold: float,
    nms_pre: Optional[int],
    max_detections: Optional[int],
    multi_label: bool,
    rotated_nms: bool,
) -> Detection3D:
    """The detections of the protocol: optional top anchors, score threshold, per-class NMS, optional cap."""
    index = torch.arange(det["scores"].numel(), device=det["scores"].device)
    if nms_pre is not None:
        # The best anchors of every scene and, for a multi-level head, of every pyramid level.
        group = det["batch"] if "level" not in det else det["batch"] * (int(det["level"].max()) + 1) + det["level"]
        index = top_per_group(det["scores"], group, nms_pre)
    if multi_label:
        anchor, labels = (det["class_probs"][index] > score_threshold).nonzero(as_tuple=True)
        index = index[anchor]
        scores = det["class_probs"][index, labels]
    else:
        index = index[det["scores"][index] > score_threshold]
        scores, labels = det["scores"][index], det["labels"][index]
    boxes, batch = det["boxes"][index], det["batch"][index]
    keep = nms3d(
        boxes,
        scores,
        NMS_IOU,
        labels=labels,
        batch=batch,
        rotated=rotated_nms,
        max_keep=max_detections,
    )
    return {
        "boxes": boxes[keep],
        "scores": scores[keep],
        "labels": labels[keep],
        "batch": batch[keep],
        "velocity": det["velocity"][index][keep],
    }


def forward(model: DetectionModel, data: dict[str, Tensor]) -> Detection3D:
    output = model(
        data[DataKeys.VOXEL],
        data[DataKeys.POS_VOXEL],
        data[DataKeys.VOXEL_NUM_POINTS],
        data[f"batch_{DataKeys.POS_VOXEL}"],
    )
    return model.decode(output)


@torch.no_grad()
def evaluate(
    model: DetectionModel,
    dataloader: DataLoader,
    device: str,
    *,
    score_threshold: float = SCORE_THRESHOLD,
    nms_pre: Optional[int] = None,
    max_detections: Optional[int] = None,
    multi_label: bool = False,
    rotated_nms: bool = False,
) -> dict[str, float]:
    model.to(device).eval()
    preds: list[Detection3D] = []
    targets: list[Boxes3D] = []
    gt_velocities: list[Tensor] = []
    gt_num_points: list[Tensor] = []
    gt_attributes: list[Tensor] = []
    offsets: list[int] = [0]

    for data in tqdm(dataloader, total=len(dataloader), desc="Testing"):
        data = {key: value.to(device) if torch.is_tensor(value) else value for key, value in data.items()}
        raw = forward(model, data)
        det = select_detections(
            raw,
            score_threshold,
            nms_pre,
            max_detections,
            multi_label,
            rotated_nms,
        )
        preds.append(
            {
                "boxes": det["boxes"].cpu(),
                "scores": det["scores"].cpu(),
                "labels": det["labels"].cpu(),
                "batch": det["batch"].cpu(),
                "velocity": det["velocity"].cpu(),
            }
        )
        targets.append(
            {
                "boxes": data[DataKeys.BOX].cpu(),
                "labels": data[DataKeys.LABEL].cpu(),
                "batch": data[DataKeys.BATCH_BOX].cpu(),
            }
        )
        gt_velocities.append(data[DataKeys.VELOCITY].cpu())
        gt_num_points.append(data[DataKeys.NUM_POINTS].cpu())
        gt_attributes.append(data[DataKeys.ATTRIBUTE].cpu())
        offsets.append(offsets[-1] + len(data[DataKeys.TOKEN]))

    pred_labels = torch.cat([p["labels"] for p in preds])
    pred_velocity = torch.cat([p["velocity"] for p in preds])

    return nuscenes_detection_metrics(
        {
            "boxes": torch.cat([torch.cat([p["boxes"], p["velocity"]], dim=1) for p in preds]),
            "scores": torch.cat([p["scores"] for p in preds]),
            "labels": pred_labels,
            "batch": torch.cat([p["batch"] + offset for p, offset in zip(preds, offsets)]),
        },
        {
            "boxes": torch.cat([torch.cat([t["boxes"], v], dim=1) for t, v in zip(targets, gt_velocities)]),
            "labels": torch.cat([t["labels"] for t in targets]),
            "batch": torch.cat([t["batch"] + offset for t, offset in zip(targets, offsets)]),
        },
        class_names=NUSCENES_DETECTION_CLASSES,
        gt_num_points=torch.cat(gt_num_points),
        pred_attributes=velocity_attributes(pred_labels, pred_velocity),
        gt_attributes=torch.cat(gt_attributes),
    )


if __name__ == "__main__":
    main()
