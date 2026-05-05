"""Benchmark RandLA-Net semantic segmentation on SemanticKITTI.

The pretrained `randlanet.semantickitti` checkpoint is converted from
[tsunghan-wu/RandLA-Net-pytorch](https://github.com/tsunghan-wu/RandLA-Net-pytorch),
a faithful PyTorch port of the original
[QingyongHu/RandLA-Net](https://github.com/QingyongHu/RandLA-Net) Tensorflow code.

The model expects a 0.06 m grid sub-sampled cloud (only XYZ; intensity is dropped),
matching the upstream `helper_tool` convention. We do this sub-sampling **inside the
benchmark loop** so we can keep the cluster ids that map every original point back to
its voxel — predictions are projected to full-resolution before mIoU is computed,
matching the upstream `proj_inds = KDTree(sub_pc).query(pc)` evaluation protocol.

Usage:
    uv run --no-sync python examples/randlanet_benchmark_semantickitti.py --limit 5
"""

import argparse
import os
import time
from typing import Any, Dict, Optional

import torch
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

import torch_pointcloud.transforms as T
from torch_pointcloud.config import DATA_DIR
from torch_pointcloud.datasets import SemanticKITTI
from torch_pointcloud.models import create_model
from torch_pointcloud.utils.data import DataKeys, collate
from torch_pointcloud.utils.metrics import confusion_matrix
from torch_pointcloud.utils.random import seed_everything


CUDA_AVAILABLE = torch.cuda.is_available()
CPU_COUNT = os.cpu_count()
DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"
NUM_WORKERS = CPU_COUNT // 2 if CPU_COUNT is not None else 0
BATCH_SIZE = 1
SEED = 42
IGNORE_INDEX = 255
GRID_SIZE = 0.06


# --- 19-class learning_map (raw SemanticKITTI ids -> contiguous indices) ----
# `moving-*` are merged with their static counterpart; bus/on-rails/lane-marking/
# other-* fall through to `default=255`. Identical to what the registered model
# transform applies, but reused here at full resolution.
_LEARNING_MAP = {
    10: 0,
    252: 0,
    11: 1,
    15: 2,
    18: 3,
    258: 3,
    20: 4,
    259: 4,
    30: 5,
    254: 5,
    31: 6,
    253: 6,
    32: 7,
    255: 7,
    40: 8,
    44: 9,
    48: 10,
    49: 11,
    50: 12,
    51: 13,
    70: 14,
    71: 15,
    72: 16,
    80: 17,
    81: 18,
}


def _eval_transforms() -> Any:
    """Per-sample transform: label remap then 0.06 m grid sub-sample.

    Stores the inverse cluster mapping under ``cluster`` so the eval loop can
    project sub-resolution predictions back to full resolution (matches Open3D-ML's
    ``KDTree(sub_pc).query(pc)`` semantics). Assumes ``batch_size=1``.
    """
    return T.Compose(
        [
            T.Relabel(keys=DataKeys.SEGMENT, labels=_LEARNING_MAP, default=IGNORE_INDEX),
            T.VoxelGrid(
                pos_key=DataKeys.POS,
                pos_reduce="mean",
                size=GRID_SIZE,
                method="fnv",
                cluster_key="cluster",
            ),
        ]
    )


@torch.inference_mode()
def forward_once(
    model: torch.nn.Module,
    pos: torch.Tensor,
    batch: torch.Tensor,
    device: str,
) -> tuple[torch.Tensor, float]:
    if device.startswith("cuda"):
        torch.cuda.synchronize()

    start = time.perf_counter()
    logits = model(None, pos, batch)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - start) * 1000.0
    return logits, latency_ms


@torch.no_grad()
def evaluate(
    model: Module,
    dataloader: DataLoader,
    device: str,
    num_classes: int,
    max_iters: Optional[int] = None,
) -> Dict[str, Any]:
    model.to(device).eval()
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    total_latency_ms = 0.0
    total_points = 0
    n_iters = 0

    total = len(dataloader) if max_iters is None else min(max_iters, len(dataloader))
    pbar = tqdm(dataloader, total=total, desc="Testing")
    for data in pbar:
        data = {k: v.to(device) if torch.is_tensor(v) else v for k, v in data.items()}
        # Dataset transform already grid-sub-sampled `pos`/`batch` and produced the
        # inverse `cluster` mapping for full-resolution back-projection.
        target_full = data[DataKeys.SEGMENT]
        cluster_full = data["cluster"]
        n_full = int(target_full.shape[0])

        logits_sub, latency_ms = forward_once(model, data[DataKeys.POS], data[DataKeys.BATCH], device)
        preds_sub = logits_sub.argmax(dim=1)
        # Project predictions back to full resolution: every original point gets the label
        # of its enclosing voxel (matches upstream's `KDTree(sub_pc).query(pc)` semantics).
        preds_full = preds_sub[cluster_full]

        cm += confusion_matrix(preds_full.cpu(), target_full.cpu(), num_classes, ignore_index=IGNORE_INDEX)
        total_latency_ms += latency_ms
        total_points += n_full
        oa = cm.diag().sum().float() / cm.sum().float().clamp_min(1)
        pbar.set_postfix({"oa": f"{oa.item():.4f}"})

        n_iters += 1
        if max_iters is not None and n_iters >= max_iters:
            pbar.close()
            break

    intersection = cm.diag().float()
    union = cm.sum(dim=1).float() + cm.sum(dim=0).float() - intersection
    iou_per_class = intersection / union.clamp_min(1e-10)
    return {
        "test/mIoU": iou_per_class.mean().item(),
        "test/oa": (cm.diag().sum().float() / cm.sum().float().clamp_min(1)).item(),
        "test/latency_ms": total_latency_ms / max(n_iters, 1),
        "test/points_per_second": total_points / max(total_latency_ms / 1000.0, 1e-12),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark RandLA-Net semantic segmentation on SemanticKITTI.")
    parser.add_argument("--model", default="randlanet.semantickitti", help="Registered segmentation model name.")
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--root", default=DATA_DIR, help="Dataset root directory.")
    parser.add_argument("--split", default="val", choices=["train", "val", "trainval", "test"])
    parser.add_argument("--seed", default=SEED, type=int)
    parser.add_argument("--batch-size", default=BATCH_SIZE, type=int)
    parser.add_argument("--num-workers", default=NUM_WORKERS, type=int)
    parser.add_argument("--limit", default=None, type=int, help="Evaluate at most this many scans.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    print(f"Benchmarking model {args.model!r} on SemanticKITTI (split={args.split!r})!")
    model = create_model(args.model, task="segmentation", pretrained=True)
    num_classes = int(model.num_classes)

    dataset: Dataset = SemanticKITTI(root=args.root, split=args.split, transform=_eval_transforms())
    if args.limit is not None:
        n = min(int(args.limit), len(dataset))  # type: ignore[arg-type]
        dataset = Subset(dataset, range(n))
        print(f"Evaluating on a subset of the first {n} scans.")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
    )

    print(f"Test set: {len(dataset)} scans")  # type: ignore[arg-type]
    metrics = evaluate(model, dataloader, args.device, num_classes)
    print("\nResults:")
    for key, value in metrics.items():
        print(f"  {key:<24} {value:.4f}")


if __name__ == "__main__":
    main()
