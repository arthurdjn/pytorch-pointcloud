"""Benchmark OneFormer3D semantic segmentation on ScanNet val.

OneFormer3D predicts over superpoints (offline Felzenszwalb mesh segmentation),
not over points or voxels directly. The released ScanNet model therefore needs
the per-point superpoint ids that ship with raw ScanNet as
`scans/<scene>/<scene>_vh_clean_2.0.010000.segs.json`. This script streams the
processed val scenes one at a time (constant memory), loads those superpoints,
runs one forward per scene, maps the per-superpoint semantic logits back to points
with `predict_semantic`, and scores point-level mIoU against the raw ScanNet20
labels (ignore index $-1$), matching the official protocol.

The processed ScanNet vertices keep the mesh vertex order, so `segIndices`
aligns 1:1 with the loaded points.

Results (ScanNet val, semantic mIoU):

    | Source              | mIoU |
    | ------------------- | ---- |
    | OneFormer3D (paper) | 76.4 |
    | torch-pointcloud    | 76.5 |

Usage:
    uv run --no-sync python examples/oneformer3d_benchmark_scannet.py --limit 20
    uv run --no-sync python examples/oneformer3d_benchmark_scannet.py
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict

import torch
from tqdm import tqdm

from torch_pointcloud.config import DATA_DIR
from torch_pointcloud.datasets.scannet import SCANNET20_LABELS
from torch_pointcloud.models import create_model
from torch_pointcloud.models.oneformer3d import OneFormer3DSegmentation, _shift_superpoints
from torch_pointcloud.transforms import Relabel
from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.io import load_safetensors
from torch_pointcloud.utils.metrics import confusion_matrix
from torch_pointcloud.utils.random import seed_everything

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
SEGS_SUFFIX = "_vh_clean_2.0.010000.segs.json"


def load_superpoints(scans_root: Path, scene_id: str) -> torch.Tensor:
    """Per-point superpoint ids from the raw ScanNet `.segs.json` mesh segmentation."""
    path = scans_root / scene_id / f"{scene_id}{SEGS_SUFFIX}"
    seg_indices = json.loads(path.read_text())["segIndices"]
    return torch.tensor(seg_indices, dtype=torch.long)


@torch.inference_mode()
def evaluate(
    model: OneFormer3DSegmentation,
    scene_files: list[Path],
    transform: Any,
    scans_root: Path,
    device: str,
    num_classes: int,
) -> Dict[str, Any]:
    model.to(device).eval()
    # Raw NYU40 ids -> 0..20 (as the ScanNet20 dataset does on load); the registered
    # transform's `Relabel` then maps 0..20 -> the 0..19 training space (ignore -> -1).
    relabel_raw = Relabel(keys=DataKeys.SEGMENT, labels=SCANNET20_LABELS)
    relabel_eval = next(t for t in transform.transforms if isinstance(t, Relabel))

    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    n_scenes = 0
    total_points = 0
    total_latency_ms = 0.0

    pbar = tqdm(scene_files, desc="Testing")
    for path in pbar:
        scene_id = path.stem
        scene = relabel_raw(load_safetensors(path))
        superpoint = load_superpoints(scans_root, scene_id)
        pos = scene[DataKeys.POS]
        if superpoint.shape[0] != pos.shape[0]:
            raise RuntimeError(
                f"{scene_id}: superpoint/point count mismatch ({superpoint.shape[0]} vs {pos.shape[0]})."
            )

        target = relabel_eval({DataKeys.SEGMENT: scene[DataKeys.SEGMENT].clone()})[DataKeys.SEGMENT].long()
        data = transform(
            {
                DataKeys.POS: pos.clone(),
                DataKeys.COLOR: scene[DataKeys.COLOR].clone(),
                DataKeys.SEGMENT: scene[DataKeys.SEGMENT].clone(),
                "superpoint": superpoint,
            }
        )
        x = data[DataKeys.X].to(device)
        pos_grid = data[DataKeys.POS_GRID].to(device).long()
        inverse = data[DataKeys.INVERSE].to(device)
        superpoint = data["superpoint"].to(device)
        batch = torch.zeros(pos_grid.shape[0], dtype=torch.long, device=device)

        if device.startswith("cuda"):
            torch.cuda.synchronize()
        start = time.perf_counter()
        out = model(x, pos_grid, batch, superpoint, inverse)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        total_latency_ms += (time.perf_counter() - start) * 1000.0

        sp_shift, _ = _shift_superpoints(superpoint, inverse, batch)
        preds = model.predict_semantic(out, sp_shift)
        cm += confusion_matrix(preds.cpu(), target, num_classes, ignore_index=-1)
        n_scenes += 1
        total_points += int(target.shape[0])

        inter = cm.diag().float()
        union = cm.sum(1).float() + cm.sum(0).float() - inter
        miou = (inter[union > 0] / union[union > 0]).mean()
        pbar.set_postfix({"mIoU": f"{miou.item():.4f}"})

    inter = cm.diag().float()
    union = cm.sum(1).float() + cm.sum(0).float() - inter
    valid = union > 0
    return {
        "test/mIoU": (inter[valid] / union[valid]).mean().item(),
        "test/oA": (cm.diag().sum().float() / cm.sum().float().clamp_min(1)).item(),
        "test/latency_ms": total_latency_ms / max(n_scenes, 1),
        "test/points_per_second": total_points / max(total_latency_ms / 1000.0, 1e-12),
        "test/scenes": n_scenes,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark OneFormer3D semantic segmentation on ScanNet val.")
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--root", default=DATA_DIR, help="Dataset root directory.")
    parser.add_argument(
        "--processed-dir",
        default=None,
        help="Processed ScanNet split dir with `<scene>.safetensors`. "
        "Defaults to `<root>/ScanNet/processed_noalign_20/<split>`.",
    )
    parser.add_argument(
        "--scans-root",
        default=None,
        help="Raw ScanNet `scans` dir with `<scene>/<scene>_vh_clean_2.0.010000.segs.json`. "
        "Defaults to `<root>/ScanNet/raw/v2/scans`.",
    )
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--seed", default=SEED, type=int)
    parser.add_argument("--limit", default=None, type=int, help="Evaluate at most this many scenes.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    root = Path(args.root) / "ScanNet"
    processed_dir = Path(args.processed_dir) if args.processed_dir else root / "processed_noalign_20" / args.split
    scans_root = Path(args.scans_root) if args.scans_root else root / "raw" / "v2" / "scans"

    model, model_info = create_model(
        "oneformer3d-base.scannet20", task="segmentation", pretrained=True, return_info=True
    )
    assert isinstance(model, OneFormer3DSegmentation)
    transform = model_info["transforms"]
    num_classes = int(model.num_semantic_classes)

    scene_files = sorted(processed_dir.glob("*.safetensors"))
    if args.limit is not None:
        scene_files = scene_files[: args.limit]

    print(f"Benchmarking 'oneformer3d-base.scannet20' on ScanNet {args.split} ({len(scene_files)} scenes)")
    print(f"Processed: {processed_dir}\nSuperpoints: {scans_root}")
    metrics = evaluate(model, scene_files, transform, scans_root, args.device, num_classes)
    print("\nResults:")
    for key, value in metrics.items():
        print(f"  {key:<24} {value:.4f}")


if __name__ == "__main__":
    main()
