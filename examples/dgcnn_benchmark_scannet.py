"""Evaluate the DGCNN ScanNet-V2 semantic-segmentation model on the val split.

`ScanNet20` scenes -> sliding $1.5$ m blocks of $8192$ points -> model -> argmax -> confusion matrix over
all scenes (antao97's block-based eval protocol, rebuilt on the repo dataset).

Results vs reference (ScanNet val mIoU; reference is the antao97 repo's ScanNet section):

    | Variant               | reference | torch-pointcloud |
    | --------------------- | --------- | ---------------- |
    | dgcnn.scannet20.an-tao | 49.6      | 50.58            |

Usage:
    uv run --no-sync python examples/dgcnn_benchmark_scannet.py
"""

import os
from argparse import ArgumentParser, Namespace
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from torch_scatter import scatter_sum
from tqdm import tqdm

from torch_pointcloud.config import DATA_DIR
from torch_pointcloud.datasets.scannet import ScanNet20
from torch_pointcloud.models._registry import create_model
from torch_pointcloud.utils.data import DataKeys, collate
from torch_pointcloud.utils.random import seed_everything

CUDA_AVAILABLE = torch.cuda.is_available()
CPU_COUNT = os.cpu_count()
DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"
BATCH_SIZE = 6
SEED = 1
NUM_POINT = 8192
BLOCK_SIZE = 1.5
STRIDE_RATE = 0.5
NUM_CLASSES = 20
NUM_WORKERS = 8
SCANNET20_CLASSES = [
    "wall",
    "floor",
    "cabinet",
    "bed",
    "chair",
    "sofa",
    "table",
    "door",
    "window",
    "bookshelf",
    "picture",
    "counter",
    "desk",
    "curtain",
    "refrigerator",
    "showercurtain",
    "toilet",
    "sink",
    "bathtub",
    "otherfurniture",
]


def main() -> None:
    args = parse_args()

    print(f"Seeding everything to {args.seed}!")
    seed_everything(args.seed)

    print("Loading model 'dgcnn.scannet20.an-tao'!")
    model, model_info = create_model("dgcnn.scannet20.an-tao", task="segmentation", pretrained=True, return_info=True)

    print("Loading ScanNet20 val blocks (use_axis_alignment=False)!")
    dataset = ScanNet20(
        root=args.root,
        split="val",
        use_axis_alignment=False,
        block_size=args.block_size,
        block_stride=args.block_size * args.stride_rate,
        num_nodes=args.npoint,
        transform=model_info.get("transform"),
        show_progress=True,
        force_process=args.force_process,
        num_workers=args.num_workers,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=args.num_workers,
    )

    preds, targets = predict(model, dataloader, dataset.scene_boundaries, args)
    metrics = compute_metrics(preds, targets, num_classes=NUM_CLASSES)

    print("\nResults:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Benchmark DGCNN semantic segmentation on ScanNet.")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--root", type=str, default=DATA_DIR, help="Dataset root directory.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--device", type=str, default=DEVICE)
    parser.add_argument("--npoint", type=int, default=NUM_POINT)
    parser.add_argument("--force-process", action="store_true", help="Force re-processing the dataset.")
    parser.add_argument("--block-size", type=float, default=BLOCK_SIZE)
    parser.add_argument("--stride-rate", type=float, default=STRIDE_RATE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    return parser.parse_args()


@torch.no_grad()
def predict(
    model: Module,
    dataloader: DataLoader,
    scene_boundaries: List[int],
    args: Namespace,
) -> Tuple[np.ndarray, np.ndarray]:
    model.to(args.device).eval()

    all_logits: List[torch.Tensor] = []
    all_indices: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    all_scene_ids: List[torch.Tensor] = []
    all_num_scene_points: List[torch.Tensor] = []
    correct, total = 0, 0

    pbar_pred = tqdm(dataloader, desc="Predicting")
    for data in pbar_pred:
        pos = data[DataKeys.POS].to(args.device)
        color = data[DataKeys.COLOR].to(args.device)
        norm_pos = data["norm_pos"].to(args.device)
        batch = data[DataKeys.BATCH].to(args.device)

        x = torch.cat([pos, color], dim=1)
        logits = model(x, norm_pos, batch)

        preds_block = logits.argmax(dim=1).cpu()
        labels_block = data[DataKeys.SEGMENT]
        valid = labels_block != 255
        correct += int((preds_block[valid] == labels_block[valid]).sum())
        total += int(valid.sum())
        pbar_pred.set_postfix(block_acc=f"{correct / max(total, 1):.3f}")

        scene_ids_per_point = data["scene_index"].repeat_interleave(args.npoint)
        num_pts_per_point = data["num_scene_points"].repeat_interleave(args.npoint)

        all_logits.append(logits.cpu())
        all_indices.append(data["point_indices"])
        all_labels.append(labels_block)
        all_scene_ids.append(scene_ids_per_point)
        all_num_scene_points.append(num_pts_per_point)

    logits_cat = torch.cat(all_logits, dim=0)
    indices_cat = torch.cat(all_indices, dim=0)
    labels_cat = torch.cat(all_labels, dim=0)
    scene_ids_cat = torch.cat(all_scene_ids, dim=0)
    num_scene_points_list = torch.cat(all_num_scene_points, dim=0)

    num_scenes = len(scene_boundaries)
    pred_all = np.array([], dtype=np.int64)
    gt_all = np.array([], dtype=np.int64)

    pbar = tqdm(range(num_scenes), desc="Aggregating scenes")
    for scene_idx in pbar:
        mask = scene_ids_cat == scene_idx
        if not mask.any():
            continue

        scene_logits = logits_cat[mask]
        scene_indices = indices_cat[mask]
        scene_labels = labels_cat[mask]
        num_pts = int(num_scene_points_list[mask][0].item())

        outputs_agg = scatter_sum(scene_logits, scene_indices, dim=0, dim_size=num_pts)
        pred = torch.argmax(outputs_agg, dim=1).numpy()

        gt = _recover_scene_labels(scene_labels, scene_indices, num_pts)

        scene_i, scene_u, _ = intersection_and_union(pred, gt, NUM_CLASSES, ignore_index=255)
        present = scene_u > 0
        scene_miou = np.mean(scene_i[present] / (scene_u[present] + 1e-10)) if present.any() else 0.0

        pred_all = np.hstack([pred_all, pred]) if pred_all.size else pred
        gt_all = np.hstack([gt_all, gt]) if gt_all.size else gt

        run_i, run_u, _ = intersection_and_union(pred_all, gt_all, NUM_CLASSES, ignore_index=255)
        run_present = run_u > 0
        run_miou = np.mean(run_i[run_present] / (run_u[run_present] + 1e-10)) if run_present.any() else 0.0
        pbar.set_postfix(scene_mIoU=f"{scene_miou:.3f}", mIoU=f"{run_miou:.3f}")

    return pred_all, gt_all


def _recover_scene_labels(block_labels: torch.Tensor, block_indices: torch.Tensor, num_pts: int) -> np.ndarray:
    """Recover original per-point labels from overlapping block data."""
    gt = torch.full((num_pts,), 255, dtype=block_labels.dtype)
    gt[block_indices] = block_labels
    return gt.numpy()


def intersection_and_union(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int,
    ignore_index: int = 255,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred = pred.copy()
    target = target.copy()
    pred[target == ignore_index] = ignore_index

    intersection = pred[pred == target]
    area_intersection = np.bincount(intersection[intersection < num_classes], minlength=num_classes)
    area_pred = np.bincount(pred[pred < num_classes], minlength=num_classes)
    area_target = np.bincount(target[target < num_classes], minlength=num_classes)
    area_union = area_pred + area_target - area_intersection

    return area_intersection, area_union, area_target


def compute_metrics(preds: np.ndarray, targets: np.ndarray, num_classes: int) -> Dict[str, float]:
    intersection, union, target = intersection_and_union(preds, targets, num_classes, ignore_index=255)
    iou_class = intersection / (union + 1e-10)
    miou = np.mean(iou_class)

    valid = targets != 255
    overall_acc = float((preds[valid] == targets[valid]).sum() / valid.sum()) if valid.sum() > 0 else 0.0

    results: Dict[str, float] = {
        "test/overall_acc": overall_acc,
        "test/mIoU": float(miou),
    }
    for c, name in enumerate(SCANNET20_CLASSES):
        results[f"test/iou_{name}"] = float(iou_class[c])

    return results


if __name__ == "__main__":
    main()
