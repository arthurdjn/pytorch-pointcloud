"""Evaluate `lion-mamba.nuscenes.zhe-liu` on nuScenes mini with 3D mAP.

| Source             | nuScenes val                                                   |
| ------------------ | -------------------------------------------------------------- |
| LION-Mamba (paper) | NDS 72.1, mAP 68.0                                             |
| torch-pointcloud   | mAP@0.25 59.64, @0.5 43.87 (v1.0-mini smoke, not official NDS) |

Usage:
    uv run --no-sync python examples/lion_benchmark_nuscenes.py --root "/path/to/parent"
"""

import os
from argparse import ArgumentParser, Namespace
from typing import Dict, List, Sequence

import torch
from tqdm import tqdm

import torch_pointcloud.models.lion  # noqa: F401
from torch_pointcloud.config import DATA_DIR
from torch_pointcloud.datasets import NuScenesMini
from torch_pointcloud.models import create_model
from torch_pointcloud.models._base import DetectionModel
from torch_pointcloud.models.lion import LIONDetection
from torch_pointcloud.utils.box3d import nms3d
from torch_pointcloud.utils.data import DataKeys, PointCloudDataLoader
from torch_pointcloud.utils.metrics import mean_average_precision3d
from torch_pointcloud.utils.random import seed_everything
from torch_pointcloud.utils.types import Boxes3D, Detection3D

CPU_COUNT = os.cpu_count()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = CPU_COUNT // 2 if CPU_COUNT is not None else 0


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    model, info = create_model(args.model, task="detection", pretrained=True, return_info=True)
    assert isinstance(model, DetectionModel)
    model.to(args.device).eval()

    dataset = NuScenesMini(
        root=args.root,
        version=args.version,
        max_sweeps=args.max_sweeps,
        transform=info["transform"],
    )
    loader = PointCloudDataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        cat_keys=[DataKeys.POS, DataKeys.X, DataKeys.BOX],
    )

    print(f"Benchmarking {args.model!r} on nuScenes ({len(dataset)} keyframes)!")
    metrics = evaluate(model, loader, args.device, iou_thresholds=args.ap_iou)
    print("\nResults:")
    for name, value in metrics.items():
        print(f"  {name:<10} {value * 100:.2f}")


def lion_circular_nms(det: Detection3D, *, local_max_classes: Sequence[int], radius: float) -> Detection3D:
    """LION's per-task BEV filter: circular NMS on the crowded `local_max_classes`, keep all other classes."""
    boxes, scores, labels, batch = det["boxes"], det["scores"], det["labels"], det["batch"]
    keep_parts = []
    for b in torch.unique(batch):
        scene = (batch == b).nonzero(as_tuple=False).squeeze(-1)
        keep_mask = torch.ones(scene.numel(), dtype=torch.bool, device=scene.device)
        for cls in local_max_classes:
            local = (labels[scene] == cls).nonzero(as_tuple=False).squeeze(-1)
            if local.numel() == 0:
                continue
            cls_keep = torch.zeros(local.numel(), dtype=torch.bool, device=scene.device)
            cls_keep[nms3d(boxes[scene][local][:, :7], scores[scene][local], radius)] = True
            keep_mask[local] = cls_keep
        keep_parts.append(scene[keep_mask])
    idx = torch.cat(keep_parts)
    return {"boxes": boxes[idx], "scores": scores[idx], "labels": labels[idx], "batch": batch[idx]}


@torch.no_grad()
def evaluate(
    model: DetectionModel, loader: PointCloudDataLoader, device: str, *, iou_thresholds: Sequence[float]
) -> Dict[str, float]:
    assert isinstance(model, LIONDetection)
    head = model.head
    preds: List[Detection3D] = []
    targets: List[Boxes3D] = []
    for data in tqdm(loader, desc="nuScenes"):
        out = model(
            data[DataKeys.X].to(device),
            data[DataKeys.POS].to(device),
            data[f"batch_{DataKeys.POS}"].to(device),
        )
        det = lion_circular_nms(model.decode(out), local_max_classes=head.local_max_classes, radius=head.nms_radius)
        preds.append(
            {
                "boxes": det["boxes"].cpu(),
                "scores": det["scores"].cpu(),
                "labels": det["labels"].cpu(),
                "batch": det["batch"].cpu(),
            }
        )
        targets.append({"boxes": data[DataKeys.BOX], "labels": data[DataKeys.LABEL], "batch": data[DataKeys.BATCH_BOX]})
    return mean_average_precision3d(preds, targets, iou_thresholds=iou_thresholds)


def parse_args() -> Namespace:
    parser = ArgumentParser(description="LION nuScenes 3D detection mAP benchmark.")
    parser.add_argument("--model", type=str, default="lion-mamba.nuscenes.zhe-liu")
    parser.add_argument("--root", type=str, default=DATA_DIR, help="Parent directory containing NuScenesMini/.")
    parser.add_argument("--version", type=str, default="v1.0-mini")
    parser.add_argument("--max-sweeps", type=int, default=10)
    parser.add_argument("--ap-iou", type=float, nargs="+", default=[0.25, 0.5])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--device", type=str, default=DEVICE)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    main()
