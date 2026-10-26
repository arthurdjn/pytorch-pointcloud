"""Benchmark PointPillars on KITTI with its evaluation protocol."""

import os
from argparse import ArgumentParser, Namespace
from typing import Optional

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

import torch_pointcloud.transforms as T
from torch_pointcloud.config import DATA_DIR
from torch_pointcloud.datasets import KITTI
from torch_pointcloud.datasets.kitti import KITTI_CLASSES
from torch_pointcloud.metrics import box_matches, kitti_average_precision
from torch_pointcloud.metrics.detection import BoxMatches
from torch_pointcloud.models import DetectionModel, create_model, model_info
from torch_pointcloud.ops.box3d import nms3d, projected_ignore_mask
from torch_pointcloud.utils.data import DataKeys, PointCloudDataLoader
from torch_pointcloud.utils.random import seed_everything, set_determinism
from torch_pointcloud.utils.types import Boxes3D, Detection3D

CUDA_AVAILABLE = torch.cuda.is_available()
CPU_COUNT = os.cpu_count()
DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"
NUM_WORKERS = CPU_COUNT // 2 if CPU_COUNT is not None else 0
SEED = 42

SCORE_THRESHOLD = 0.1
NMS_IOU = 0.01


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    set_determinism(tf32=False)

    print(f"Benchmarking model {args.model!r} on KITTI!")
    model = configure_model(args)
    dataloader = configure_dataloader(args)

    print(f"Test set: {len(dataloader.dataset)} frames")  # type: ignore[arg-type]
    metrics = evaluate(
        model,
        dataloader,
        args.device,
        configure_classes(args),
        nms_pre=args.nms_pre,
        max_detections=args.max_detections,
        multi_label=args.multi_label,
    )
    print("\nResults:")
    for key, value in metrics.items():
        print(f"  {key:<24} {value:.4f}")


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Benchmark PointPillars 3D detection on KITTI.")
    parser.add_argument("--model", default="pointpillars.kitti.openpcdet", help="Registered detection model name")
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--root", default=DATA_DIR, help="Dataset root directory.")
    parser.add_argument("--split-file", default=None, help="KITTI frame-id list (e.g. ImageSets/val.txt).")
    parser.add_argument("--seed", default=SEED, type=int)
    parser.add_argument("--batch-size", default=4, type=int)
    parser.add_argument("--num-workers", default=NUM_WORKERS, type=int)
    parser.add_argument("--limit", default=None, type=int, help="Evaluate at most this many frames.")
    parser.add_argument("--nms-pre", default=None, type=int, help="Anchors kept per scene before the NMS, by score.")
    parser.add_argument("--max-detections", default=None, type=int, help="Detections kept per scene after the NMS.")
    parser.add_argument(
        "--multi-label",
        action="store_true",
        help="Run one NMS per class over the anchors scoring above the threshold in that class.",
    )
    return parser.parse_args()


def configure_classes(args: Namespace) -> tuple[str, ...]:
    """The detection classes of the model, in its label order."""
    weights = model_info(args.model, task="detection")["weights"]
    assert weights is not None and weights["classes"] is not None
    return tuple(weights["classes"])


def configure_dataset(args: Namespace) -> Dataset:
    dataset: Dataset
    classes = configure_classes(args)
    # Van / Person_sitting and harder-than-moderate boxes (occlusion > 1, truncation > 0.3, 2D height < 25 px) are
    # ignored.
    relabel = T.RelabelBoxes(
        keys=(DataKeys.BOX, DataKeys.LABEL, DataKeys.TRUNCATION, DataKeys.OCCLUSION, DataKeys.BBOX_HEIGHT),
        mapping={KITTI_CLASSES.index(name): index for index, name in enumerate(classes)},
        ignore_mapping={
            KITTI_CLASSES.index(ignored): classes.index(name)
            for ignored, name in (("Van", "Car"), ("Person_sitting", "Pedestrian"))
            if name in classes
        },
        ignore_fields={
            DataKeys.OCCLUSION: (None, 1),
            DataKeys.TRUNCATION: (None, 0.3),
            DataKeys.BBOX_HEIGHT: (25, None),
        },
    )
    transform = T.Compose([relabel, configure_transform(args)])
    dataset = KITTI(
        root=args.root,
        train=True,
        split_file=args.split_file,
        fov=True,
        return_calib=True,
        transform=transform,
    )

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
        stack_keys=[DataKeys.CALIB, DataKeys.IMAGE_SHAPE],
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


def select_detections(
    det: Detection3D,
    nms_pre: Optional[int],
    max_detections: Optional[int],
    multi_label: bool,
) -> Detection3D:
    """The detections of the protocol: optional top anchors, score threshold, rotated NMS, optional cap."""
    index = torch.arange(det["scores"].numel(), device=det["scores"].device)
    if nms_pre is not None:
        # Every scene holds the same anchors, so the scores fold into one row per scene.
        scores = det["scores"].view(int(det["batch"][-1]) + 1, -1)
        top = scores.topk(min(nms_pre, scores.shape[1]), dim=1).indices
        index = (top + torch.arange(scores.shape[0], device=top.device)[:, None] * scores.shape[1]).view(-1)
    if multi_label:
        anchor, labels = (det["class_probs"][index] > SCORE_THRESHOLD).nonzero(as_tuple=True)
        index = index[anchor]
        scores = det["class_probs"][index, labels]
    else:
        index = index[det["scores"][index] > SCORE_THRESHOLD]
        scores, labels = det["scores"][index], det["labels"][index]
    boxes, batch = det["boxes"][index], det["batch"][index]
    keep = nms3d(
        boxes,
        scores,
        NMS_IOU,
        labels=labels if multi_label else None,
        batch=batch,
        rotated=True,
        max_keep=max_detections,
    )
    return {"boxes": boxes[keep], "scores": scores[keep], "labels": labels[keep], "batch": batch[keep]}


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
    classes: tuple[str, ...],
    *,
    nms_pre: Optional[int] = None,
    max_detections: Optional[int] = None,
    multi_label: bool = False,
) -> dict[str, float]:
    model.to(device).eval()
    matches: list[BoxMatches] = []

    for data in tqdm(dataloader, total=len(dataloader), desc="Testing"):
        data = {key: value.to(device) if torch.is_tensor(value) else value for key, value in data.items()}
        det = select_detections(forward(model, data), nms_pre, max_detections, multi_label)
        boxes, scores, labels, batch = det["boxes"], det["scores"], det["labels"], det["batch"]
        ignore_mask = projected_ignore_mask(boxes, data[DataKeys.CALIB][batch], data[DataKeys.IMAGE_SHAPE][batch])
        preds: Detection3D = {
            "boxes": boxes,
            "scores": scores,
            "labels": labels,
            "batch": batch,
            "ignore_mask": ignore_mask,
        }
        target: Boxes3D = {
            "boxes": data[DataKeys.BOX],
            "labels": data[DataKeys.LABEL],
            "batch": data[DataKeys.BATCH_BOX],
            "ignore_mask": data["ignore_mask"],
        }
        matches.append(box_matches(preds, target))

    return kitti_average_precision(matches, class_names=classes)


if __name__ == "__main__":
    main()
