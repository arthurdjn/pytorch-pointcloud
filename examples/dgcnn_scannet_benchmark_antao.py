"""Evaluate the DGCNN ScanNet-V2 model with antao97's original block protocol.

Reads the raw ScanNet `.ply` scenes directly and reproduces the antao97 eval verbatim (per-scene sliding
$1.5$ m blocks, block-level voting, full-resolution scoring); use `dgcnn_benchmark_scannet.py` for the
equivalent benchmark built on the repo dataset (which scores slightly higher).

Results vs reference (ScanNet val mIoU; reference is the antao97 repo's ScanNet section):

    | Variant               | reference | torch-pointcloud |
    | --------------------- | --------- | ---------------- |
    | dgcnn.scannet20.an-tao | 49.6      | 49.17            |

Usage:
    uv run --no-sync python examples/dgcnn_scannet_benchmark_antao.py
"""

import os
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import plyfile
import torch
from torch.nn import Module
from torch_scatter import scatter_sum
from tqdm import tqdm

from torch_pointcloud.config import DATA_DIR
from torch_pointcloud.models._registry import create_model
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

# NYU40 label ids for the 20 ScanNet benchmark classes (+ 0 for unannotated).
# Used to build the same label_map as antao/dgcnn.pytorch prepare_data.
SCANNET20_NYU40IDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]


def gen_label_map() -> np.ndarray:
    """Build the NYU40 -> contiguous 0-20 label map (identical to antao reference)."""
    label_map = np.zeros(41, dtype=np.int64)
    for i in range(41):
        if i in SCANNET20_NYU40IDS:
            label_map[i] = SCANNET20_NYU40IDS.index(i)
    return label_map


def load_scene_ply(ply_path: str) -> np.ndarray:
    """Load xyzrgb from a ScanNet _vh_clean_2.ply file. No axis alignment applied."""
    plydata = plyfile.PlyData.read(ply_path)
    vertex = plydata["vertex"]
    xyz = np.stack((vertex["x"], vertex["y"], vertex["z"]), axis=-1).astype(np.float32)
    rgb = np.stack((vertex["red"], vertex["green"], vertex["blue"]), axis=-1).astype(np.float32)
    return np.concatenate([xyz, rgb], axis=1)


def load_scene_labels(labels_ply_path: str) -> np.ndarray:
    """Load per-vertex labels from a ScanNet _vh_clean_2.labels.ply file."""
    plydata = plyfile.PlyData.read(labels_ply_path)
    return np.array(plydata["vertex"]["label"], dtype=np.int64)


def load_val_scenes(
    scans_dir: str, metadata_dir: str, label_map: np.ndarray, limit: Optional[int] = None
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Load all val scenes using the same pipeline as the antao reference (keep_unanno=True)."""
    split_file = os.path.join(metadata_dir, "scannetv2_val.txt")
    with open(split_file) as f:
        scene_ids = [line.strip() for line in f if line.strip()]
    if limit is not None:
        scene_ids = scene_ids[:limit]

    xyz_all: List[np.ndarray] = []
    label_all: List[np.ndarray] = []

    for scene_id in tqdm(scene_ids, desc="Loading scenes"):
        ply_path = os.path.join(scans_dir, scene_id, f"{scene_id}_vh_clean_2.ply")
        labels_ply_path = os.path.join(scans_dir, scene_id, f"{scene_id}_vh_clean_2.labels.ply")
        points = load_scene_ply(ply_path)
        raw_labels = load_scene_labels(labels_ply_path)

        # Clamp out-of-range labels, then map through label_map (same as antao keep_unanno=True)
        raw_labels[raw_labels > 40] = 0
        mapped_labels = label_map[raw_labels]

        xyz_all.append(points)
        label_all.append(mapped_labels)

    return xyz_all, label_all


def main() -> None:
    args = parse_args()

    print(f"Seeding everything to {args.seed}!")
    seed_everything(args.seed)

    print("Loading model 'dgcnn.scannet20.an-tao'!")
    model = create_model("dgcnn.scannet20.an-tao", task="segmentation", pretrained=True)

    label_map = gen_label_map()
    scans_dir = os.path.join(args.root, "raw", "v2", "scans")
    metadata_dir = os.path.join(args.root, "raw", "metadata")

    print("Loading ScanNet val scenes!")
    xyz_all, label_all = load_val_scenes(scans_dir, metadata_dir, label_map, limit=args.limit_scenes)

    # Antao convention: shift labels by -1, class 0 (unannotated) becomes 255 (ignore)
    for i in range(len(label_all)):
        label = label_all[i] - 1
        label[label_all[i] == 0] = 255
        label_all[i] = label

    preds, targets = predict(model, xyz_all, label_all, args)
    metrics = compute_metrics(preds, targets, num_classes=NUM_CLASSES)
    print_metrics(metrics)


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Benchmark DGCNN semantic segmentation on ScanNet.")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--root",
        type=str,
        default=str(Path(DATA_DIR, "ScanNet")),
        help="Root directory of the ScanNet dataset (containing raw/ and metadata/).",
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--device", type=str, default=DEVICE)
    parser.add_argument("--npoint", type=int, default=NUM_POINT)
    parser.add_argument("--block-size", type=float, default=BLOCK_SIZE)
    parser.add_argument("--stride-rate", type=float, default=STRIDE_RATE)
    parser.add_argument(
        "--limit-scenes", type=int, default=None, help="WIP: only load and evaluate the first N scenes."
    )
    return parser.parse_args()


def data_prepare(
    points: np.ndarray,
    labels: np.ndarray,
    num_point: int,
    block_size: float = 1.5,
    stride_rate: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tile a full scene into overlapping fixed-size blocks.

    Replicates the `data_prepare` function from the antao/dgcnn.pytorch reference
    implementation exactly.

    Args:
        points: Scene points, shape (N, 6) with columns [x, y, z, r, g, b].
        labels: Per-point labels, shape (N,).
        num_point: Fixed number of points per block.
        block_size: Spatial size of each block in meters.
        stride_rate: Fraction of block_size used as stride.

    Returns:
        data_room: Tiled blocks, shape (M, 9) where M is a multiple of num_point.
        label_room: Per-point labels for all blocks, shape (M,).
        index_room: Original point indices for scatter aggregation, shape (M,).
    """
    coord_min = np.amin(points, axis=0)[:3]
    coord_max = np.amax(points, axis=0)[:3]
    stride = block_size * stride_rate
    grid_x = int(np.ceil(float(coord_max[0] - coord_min[0] - block_size) / stride) + 1)
    grid_y = int(np.ceil(float(coord_max[1] - coord_min[1] - block_size) / stride) + 1)

    data_room: np.ndarray = np.array([])
    label_room: np.ndarray = np.array([])
    index_room: np.ndarray = np.array([])

    for index_y in range(grid_y):
        for index_x in range(grid_x):
            s_x = coord_min[0] + index_x * stride
            e_x = min(s_x + block_size, coord_max[0])
            s_x = e_x - block_size
            s_y = coord_min[1] + index_y * stride
            e_y = min(s_y + block_size, coord_max[1])
            s_y = e_y - block_size
            point_idxs = np.where(
                (points[:, 0] >= s_x - 1e-8)
                & (points[:, 0] <= e_x + 1e-8)
                & (points[:, 1] >= s_y - 1e-8)
                & (points[:, 1] <= e_y + 1e-8)
            )[0]
            if point_idxs.size == 0:
                continue
            num_batch = int(np.ceil(point_idxs.size / num_point))
            point_size = int(num_batch * num_point)
            replace = point_size - point_idxs.size > point_idxs.size
            point_idxs_repeat = np.random.choice(point_idxs, point_size - point_idxs.size, replace=replace)
            point_idxs = np.concatenate((point_idxs, point_idxs_repeat))

            data_batch = points[point_idxs, :].copy()
            normlized_xyz = np.zeros((point_size, 3))
            normlized_xyz[:, 0] = data_batch[:, 0] / coord_max[0]
            normlized_xyz[:, 1] = data_batch[:, 1] / coord_max[1]
            normlized_xyz[:, 2] = data_batch[:, 2] / coord_max[2]
            data_batch[:, 0] = data_batch[:, 0] - (s_x + block_size / 2.0)
            data_batch[:, 1] = data_batch[:, 1] - (s_y + block_size / 2.0)
            data_batch[:, 3:6] = data_batch[:, 3:6] / 255.0
            data_batch = np.concatenate((data_batch, normlized_xyz), axis=1)

            label_batch = labels[point_idxs]
            data_room = np.vstack([data_room, data_batch]) if data_room.size else data_batch
            label_room = np.hstack([label_room, label_batch]) if label_room.size else label_batch
            index_room = np.hstack([index_room, point_idxs]) if index_room.size else point_idxs

    # if index_room.size == 0 or np.unique(index_room).size != labels.size:
    #     import warnings
    #     covered = np.unique(index_room).size if index_room.size else 0
    #     warnings.warn(f"data_prepare: only {covered}/{labels.size} points covered by tiling grid")

    return data_room, label_room, index_room


@torch.no_grad()
def predict(
    model: Module, xyz_all: List[np.ndarray], label_all: List[np.ndarray], args: Namespace
) -> Tuple[np.ndarray, np.ndarray]:
    model.to(args.device).eval()
    pred_all: np.ndarray = np.array([])
    gt_all: np.ndarray = np.array([])
    num_scenes = len(xyz_all)

    pbar = tqdm(range(num_scenes), desc="Evaluating scenes")
    for idx in pbar:
        points = xyz_all[idx]
        gt = label_all[idx].astype(np.int32)

        data_room, label_room, index_room = data_prepare(points, gt, args.npoint, args.block_size, args.stride_rate)

        if data_room.size == 0:
            gt_all = np.hstack([gt_all, gt]) if gt_all.size else gt
            pred_all = np.hstack([pred_all, np.full_like(gt, 255)]) if pred_all.size else np.full_like(gt, 255)
            continue

        data_room_t = torch.from_numpy(data_room).float()
        input_blocks = data_room_t.view(-1, args.npoint, data_room_t.shape[1])

        outputs = []
        eval_batch_size = args.batch_size
        num_blocks = input_blocks.shape[0]
        for b in range(0, num_blocks, eval_batch_size):
            input_b = input_blocks[b : b + eval_batch_size].to(args.device)
            if input_b.shape[0] == 0:
                break

            # Channels 0:3 = centered xyz, 3:6 = rgb/255 -> x (in_channels=6)
            # Channels 6:9 = normalized xyz -> pos (spatial_dim=3)
            x_feat = input_b[:, :, :6].reshape(-1, 6)
            pos_feat = input_b[:, :, 6:9].reshape(-1, 3)
            batch_idx = torch.arange(input_b.shape[0], device=args.device).repeat_interleave(args.npoint)

            logits = model(x_feat, pos_feat, batch_idx)
            outputs.append(logits.cpu())

        outputs_cat = torch.cat(outputs, dim=0)

        index_room_tensor = torch.from_numpy(index_room).long()
        num_original_points = points.shape[0]
        outputs_agg = torch.zeros(num_original_points, NUM_CLASSES)
        for c in range(NUM_CLASSES):
            outputs_agg[:, c] = scatter_sum(outputs_cat[:, c], index_room_tensor, dim_size=num_original_points)

        pred = torch.argmax(outputs_agg, dim=1).numpy()

        scene_i, scene_u, _ = intersection_and_union(pred, gt, NUM_CLASSES, ignore_index=255)
        present = scene_u > 0
        scene_miou = np.mean(scene_i[present] / (scene_u[present] + 1e-10)) if present.any() else 0.0

        pred_all = np.hstack([pred_all, pred]) if pred_all.size else pred
        gt_all = np.hstack([gt_all, gt]) if gt_all.size else gt

        # Running global mIoU over all points so far
        run_i, run_u, _ = intersection_and_union(pred_all, gt_all, NUM_CLASSES, ignore_index=255)
        run_present = run_u > 0
        run_miou = np.mean(run_i[run_present] / (run_u[run_present] + 1e-10)) if run_present.any() else 0.0
        pbar.set_postfix(scene_mIoU=f"{scene_miou:.3f}", mIoU=f"{run_miou:.3f}")

    return pred_all, gt_all


def intersection_and_union(
    pred: np.ndarray, target: np.ndarray, num_classes: int, ignore_index: int = 255
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


def print_metrics(metrics: Dict[str, float]) -> None:
    print("\nResults:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
