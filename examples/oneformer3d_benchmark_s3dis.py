"""Benchmark OneFormer3D semantic and instance segmentation on S3DIS Area 5.

The S3DIS variant has no superpoint pooling: the decoder runs on per-scene voxel
features with 400 learned instance + 13 learned semantic queries, and semantics
come from the 13 semantic-query masks (no `out_sem` head). Each room is voxelized
at 5 cm, run through the model once, and the per-voxel semantic argmax is mapped
back to points via the voxelization `inverse` for point-level mIoU.

The released model is trained with the upstream S3DIS class order
(`...door, table, chair, sofa, bookcase, board, clutter`); the repo's `S3DIS`
dataset uses a different order (`chair`/`table` and `sofa`/`bookcase` swapped), so
model predictions are remapped to the dataset's label space by class name.

Rooms are loaded with `aligned=True`: the reference trains and evaluates on the
`Stanford3dDataset_v1.2_Aligned_Version` release. The 68.2 below was measured on
unaligned rooms with the previous protocol and is pending re-measurement.

The instance path decodes the 400 instance queries per room with `predict_instance`
using the released S3DIS settings (top-450 query-class pairs, matrix NMS, voxel-score
threshold 0.15, point-count threshold 300, objectness normalization threshold 0.01),
maps the voxel masks to points via `inverse`, and scores mask mAP over all 13 classes
with `instance_matches` / `instance_average_precision`: greedy mask-IoU matching per
class, AP averaged over thresholds 0.5:0.05:0.9 plus AP@50 / AP@25, instances under
100 points ignored.

Deviation from the reference evaluation: the reference encodes ground-truth ids as
`semantic * 1000 + instance`, so class-0 (ceiling) instances get ids below 1000 and
trip its group-ignore branch: an unmatched ceiling prediction overlapping any ceiling
instance is silently dropped instead of counted as a false positive. This id-encoding
artifact is not reproduced; it can only make the reference's ceiling AP read higher
than ours, and touches no other class.

Results (S3DIS Area 5):

    | Source                        | mIoU | mAP  | mAP@50 | mAP@25 |
    | ----------------------------- | ---- | ---- | ------ | ------ |
    | OneFormer3D (paper)           | 71.9 | 58.0 | 72.7   | 80.6   |
    | torch-pointcloud (unaligned)  | 68.2 | TBD  | TBD    | TBD    |

The instance mAP columns are pending a GPU run on aligned rooms.

Usage:
    uv run --no-sync python examples/oneformer3d_benchmark_s3dis.py --limit 5
    uv run --no-sync python examples/oneformer3d_benchmark_s3dis.py
"""

import argparse
import time
from typing import Any, Dict

import torch
from tqdm import tqdm

from torch_pointcloud.config import DATA_DIR
from torch_pointcloud.datasets import S3DIS
from torch_pointcloud.datasets.s3dis import S3DIS_CLASS_TO_IDX, S3DIS_CLASSES
from torch_pointcloud.models import create_model
from torch_pointcloud.models.oneformer3d import OneFormer3DSegmentation
from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.metrics import confusion_matrix, instance_average_precision, instance_matches
from torch_pointcloud.utils.random import seed_everything, set_determinism

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

# Upstream OneFormer3D S3DIS class order = the model's output class order.
UPSTREAM_CLASSES = (
    "ceiling",
    "floor",
    "wall",
    "beam",
    "column",
    "window",
    "door",
    "table",
    "chair",
    "sofa",
    "bookcase",
    "board",
    "clutter",
)


@torch.inference_mode()
def evaluate(
    model: OneFormer3DSegmentation,
    dataset: S3DIS,
    transform: Any,
    device: str,
    num_classes: int,
    limit: int | None,
) -> Dict[str, Any]:
    model.to(device).eval()
    # model class i (upstream order) -> repo label index, so preds match dataset labels.
    remap = torch.tensor([S3DIS_CLASS_TO_IDX[c] for c in UPSTREAM_CLASSES], device=device)

    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    records = []
    n_scenes = 0
    total_points = 0
    total_latency_ms = 0.0
    n = len(dataset) if limit is None else min(limit, len(dataset))

    pbar = tqdm(range(n), desc="Testing")
    for i in pbar:
        room = dataset[i]
        target = room[DataKeys.SEGMENT].long()
        data = transform(
            {
                DataKeys.POS: room[DataKeys.POS].clone(),
                DataKeys.COLOR: room[DataKeys.COLOR].clone(),
                DataKeys.SEGMENT: room[DataKeys.SEGMENT].clone(),
            }
        )
        x = data[DataKeys.X].to(device)
        pos_grid = data[DataKeys.POS_GRID].to(device).long()
        inverse = data[DataKeys.INVERSE].to(device)
        batch = torch.zeros(pos_grid.shape[0], dtype=torch.long, device=device)

        if device.startswith("cuda"):
            torch.cuda.synchronize()
        start = time.perf_counter()
        out = model(x, pos_grid, batch)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        total_latency_ms += (time.perf_counter() - start) * 1000.0

        preds = remap[model.predict_semantic(out, inverse)]
        cm += confusion_matrix(preds.cpu(), target, num_classes, ignore_index=-1)

        masks, labels, scores = model.predict_instance(
            out,
            inverse,
            topk=450,
            sp_score_threshold=0.15,
            npoint_threshold=300,
            obj_normalization_threshold=0.01,
        )
        records.append(
            instance_matches(masks, remap[labels], scores, room[DataKeys.INSTANCE].to(device), target.to(device))
        )
        n_scenes += 1
        total_points += int(target.shape[0])

        inter = cm.diag().float()
        union = cm.sum(1).float() + cm.sum(0).float() - inter
        miou = (inter[union > 0] / union[union > 0]).mean()
        pbar.set_postfix({"mIoU": f"{miou.item():.4f}"})

    inter = cm.diag().float()
    union = cm.sum(1).float() + cm.sum(0).float() - inter
    valid = union > 0
    instance_ap = instance_average_precision(records, num_classes=num_classes, class_names=S3DIS_CLASSES)
    return {
        "test/mIoU": (inter[valid] / union[valid]).mean().item(),
        "test/oA": (cm.diag().sum().float() / cm.sum().float().clamp_min(1)).item(),
        "test/mAP": instance_ap["mAP"],
        "test/mAP@0.5": instance_ap["mAP@0.5"],
        "test/mAP@0.25": instance_ap["mAP@0.25"],
        "test/latency_ms": total_latency_ms / max(n_scenes, 1),
        "test/points_per_second": total_points / max(total_latency_ms / 1000.0, 1e-12),
        "test/scenes": n_scenes,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark OneFormer3D semantic segmentation on S3DIS Area 5.")
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--root", default=DATA_DIR, help="Dataset root directory.")
    parser.add_argument("--area", default="Area_5", help="S3DIS test area.")
    parser.add_argument("--seed", default=SEED, type=int)
    parser.add_argument("--limit", default=None, type=int, help="Evaluate at most this many rooms.")
    parser.add_argument("--download", action="store_true", help="Download S3DIS if missing.")
    parser.add_argument("--force-process", action="store_true", help="Force re-processing the dataset.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    set_determinism(tf32=False)

    model, model_info = create_model(
        "oneformer3d-base.s3dis-area5.danila-rukhovich", task="segmentation", pretrained=True, return_info=True
    )
    assert isinstance(model, OneFormer3DSegmentation)
    transform = model_info["transform"]
    num_classes = int(model.num_semantic_classes)

    dataset = S3DIS(
        root=args.root,
        areas=[args.area],
        aligned=True,
        transform=None,
        download=args.download,
        force_process=args.force_process,
    )

    print(f"Benchmarking 'oneformer3d-base.s3dis-area5.danila-rukhovich' on S3DIS {args.area} ({len(dataset)} rooms)")
    metrics = evaluate(model, dataset, transform, args.device, num_classes, args.limit)
    print("\nResults:")
    for key, value in metrics.items():
        print(f"  {key:<24} {value:.4f}")


if __name__ == "__main__":
    main()
