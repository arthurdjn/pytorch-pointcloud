"""Reproduce Pointcept's SpUNet / ScanNet test recipe exactly (`SemSegTester`).

NOTE: 13 aug views x `count.max()` fragment forwards per scene (Pointcept's actual cost); the multi-fragment
voting is what a single `Voxelize(reduce="first")` + broadcast omits (~3 mIoU). Use --limit to sanity-check
before the full val run.

Usage:
    uv run --no-sync python examples/spunet_benchmark_scannet_tta.py --limit 5 --download
"""

import argparse
import copy
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from tqdm import tqdm

from torch_pointcloud.config import DATA_DIR
from torch_pointcloud.datasets import ScanNet20
from torch_pointcloud.models import create_model
from torch_pointcloud.transforms import Compose, RandomFlip, RandomRotate, RandomScale
from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.metrics import confusion_matrix
from torch_pointcloud.utils.random import seed_everything, set_determinism

CUDA_AVAILABLE = torch.cuda.is_available()
CPU_COUNT = os.cpu_count()
DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"
NUM_WORKERS = CPU_COUNT // 2 if CPU_COUNT is not None else 0
SEED = 42
GRID_SIZE = 0.02


def fnv_hash_vec(arr: np.ndarray) -> np.ndarray:
    """FNV64-1A hash of integer voxel coords, verbatim from Pointcept GridSample."""
    assert arr.ndim == 2
    arr = arr.astype(np.uint64, copy=True)
    hashed = np.uint64(14695981039346656037) * np.ones(arr.shape[0], dtype=np.uint64)
    for j in range(arr.shape[1]):
        hashed *= np.uint64(1099511628211)
        hashed = np.bitwise_xor(hashed, arr[:, j])
    return hashed


def grid_sample_test_fragments(coord: np.ndarray, grid_size: float) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Pointcept `GridSample(mode="test")`: list of (raw_index, grid_coord) fragments.

    Fragment `i` selects point `i % count` of every voxel; together the
    `count.max()` fragments cover every raw point exactly as Pointcept does.
    """
    scaled = coord / grid_size
    grid_coord = np.floor(scaled).astype(int)
    grid_coord -= grid_coord.min(0)
    key = fnv_hash_vec(grid_coord)
    idx_sort = np.argsort(key)
    key_sort = key[idx_sort]
    _, _, count = np.unique(key_sort, return_inverse=True, return_counts=True)
    base = np.cumsum(np.insert(count, 0, 0)[0:-1])
    fragments: List[Tuple[np.ndarray, np.ndarray]] = []
    for i in range(int(count.max())):
        idx_select = base + i % count
        idx_part = idx_sort[idx_select]
        fragments.append((idx_part, grid_coord[idx_part]))
    return fragments


def split_head_and_relabel(compose: Compose) -> Tuple[Compose, Any]:
    """`head` = steps before `Cat` (CenterShift + colour /255), run once/scene on
    raw points. `relabel` = the registered `Relabel` step, applied to the raw
    segment for scoring (Pointcept scores raw labels with ignore_index=-1)."""
    steps = list(compose.transforms)
    cat_idx = next(i for i, t in enumerate(steps) if type(t).__name__ == "Cat")
    relabel = next(t for t in steps if type(t).__name__ == "Relabel")
    return Compose(steps[:cat_idx]), relabel


def build_views() -> List[Compose]:
    """The 13 Pointcept ScanNet aug_transform votes (degrees; pi/2 -> 90).

    Rotation and flip act on `pos` AND `normal`; scale acts on `pos` only.
    """
    geom = (DataKeys.POS, DataKeys.NORMAL)
    pos = DataKeys.POS
    views: List[Compose] = []
    for deg in (0.0, 90.0, 180.0, 270.0):
        views.append(Compose([RandomRotate(keys=geom, angle_range=(deg, deg), axis=2, p=1.0)]))
    for scale in (0.95, 1.05):
        for deg in (0.0, 90.0, 180.0, 270.0):
            views.append(
                Compose(
                    [
                        RandomRotate(keys=geom, angle_range=(deg, deg), axis=2, p=1.0),
                        RandomScale(keys=pos, scale_range=(scale, scale), p=1.0),
                    ]
                )
            )
    views.append(Compose([RandomFlip(keys=geom, axes=(0, 1), p=1.0)]))
    return views


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataset: Dataset,
    head: Compose,
    relabel: Any,
    views: List[Compose],
    device: str,
    num_classes: int,
) -> Dict[str, float]:
    model.to(device).eval()
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)

    for i in tqdm(range(len(dataset)), desc="Pointcept TTA"):  # type: ignore[arg-type]
        s = head(dataset[i])
        seg = relabel({DataKeys.SEGMENT: s[DataKeys.SEGMENT].clone()})[DataKeys.SEGMENT].long()
        n_raw = int(s[DataKeys.POS].shape[0])
        pred = torch.zeros(n_raw, num_classes, device=device)

        for view in views:
            a = view(copy.copy(dict(s)))
            coord = a[DataKeys.POS].detach().cpu().numpy()
            color = a[DataKeys.COLOR].to(device).float()
            normal = a[DataKeys.NORMAL].to(device).float()
            for idx_np, gc_np in grid_sample_test_fragments(coord, GRID_SIZE):
                idx = torch.from_numpy(idx_np).to(device).long()
                feat = torch.cat([color[idx], normal[idx]], dim=1)
                grid_coord = torch.from_numpy(gc_np).to(device).int()
                batch = torch.zeros(idx.numel(), dtype=torch.long, device=device)
                logits = model(feat, grid_coord, batch)
                pred[idx] += torch.softmax(logits, dim=-1)

        preds = pred.argmax(dim=1)
        cm += confusion_matrix(preds.cpu(), seg, num_classes, ignore_index=-1)

    inter = cm.diag().float()
    union = cm.sum(1).float() + cm.sum(0).float() - inter
    valid = union > 0
    return {
        "mIoU": (inter[valid] / union[valid]).mean().item(),
        "oA": (cm.diag().sum().float() / cm.sum().float().clamp_min(1)).item(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce Pointcept SpUNet/ScanNet SemSegTester.")
    parser.add_argument("--model", default="spunet-v1m1.scannet20.pointcept")
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--root", default=DATA_DIR)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--seed", default=SEED, type=int)
    parser.add_argument("--num-workers", default=NUM_WORKERS, type=int)
    parser.add_argument("--limit", default=None, type=int, help="Evaluate at most this many scenes.")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--force-process", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    set_determinism(tf32=False)

    print(f"Reproducing Pointcept {args.model!r} ScanNet SemSegTester (split={args.split!r})")
    model, model_info = create_model(args.model, task="segmentation", pretrained=True, return_info=True)
    num_classes = int(model.num_classes)
    head, relabel = split_head_and_relabel(model_info["transform"])
    views = build_views()
    print(f"Head (once/scene): {[type(t).__name__ for t in head.transforms]}  |  {len(views)} views")
    print("Per view: GridSample(mode=test) -> count.max() fragments, pred accumulated at raw resolution")

    dataset: Dataset = ScanNet20(
        root=args.root,
        split=args.split,
        transform=None,
        download=args.download,
        force_process=args.force_process,
        num_workers=args.num_workers,
        use_axis_alignment=False,
    )
    if args.limit is not None:
        n = min(int(args.limit), len(dataset))  # type: ignore[arg-type]
        dataset = Subset(dataset, range(n))
        print(f"Evaluating on a subset of the first {n} scenes.")

    metrics = evaluate(model, dataset, head, relabel, views, args.device, num_classes)
    print("\nResults (Pointcept SemSegTester reproduction, 13-view x test-mode fragments):")
    for key, value in metrics.items():
        print(f"  {key:<8} {value:.4f}")


if __name__ == "__main__":
    main()
