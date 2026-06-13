"""Evaluate XCube VAE reconstruction on the XCube ShapeNet test split.

Reconstructs every test shape through the sparse structure VAE (posterior mean) and reports the
per-level structure prediction accuracy (as in the reference `test.py`) and the voxel IoU between the
reconstructed and the input grid. The reference column is the original NVIDIA implementation with the
released checkpoints on the same shapes (`notebooks/xcube/benchmark_reference_vae.py`); both match to
all reported digits.

| Coarse VAE ($128^3$) | struct acc 3 | struct acc 2 | struct acc 1 | struct acc 0 | IoU    |
| -------------------- | ------------ | ------------ | ------------ | ------------ | ------ |
| chair (= reference)  | 0.9994       | 0.9942       | 0.9858       | 0.9682       | 0.9307 |
| car (= reference)    | 0.9998       | 0.9911       | 0.9765       | 0.9402       | 0.8850 |
| plane (= reference)  | 0.9997       | 0.9911       | 0.9791       | 0.9603       | 0.9134 |

The chair fine VAE ($512^3$) reaches struct acc 1.0000 / 0.9930 / 0.9850 and IoU 0.9429, also equal to the
reference on every reported digit.

Usage:
    uv run --no-sync python examples/xcube_benchmark_shapenet.py --model xcube-vae-coarse-nvidia.shapenet-chair
"""

from argparse import ArgumentParser, Namespace
from typing import TYPE_CHECKING, Dict, List, Literal

import torch
from tqdm import tqdm

import torch_pointcloud.models.xcube  # noqa: F401
from torch_pointcloud.config import DATA_DIR
from torch_pointcloud.datasets import XCubeShapeNet
from torch_pointcloud.datasets.shapenet import XCubeShapeNetCategory
from torch_pointcloud.models import create_model
from torch_pointcloud.models.xcube import XCubeVAE
from torch_pointcloud.utils.data import DataKeys, PointCloudDataLoader
from torch_pointcloud.utils.random import seed_everything

if TYPE_CHECKING:
    from fvdb import GridBatch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def structure_accuracy(logits: torch.Tensor, grid: "GridBatch", gt_grid: "GridBatch") -> float:
    """Fraction of `grid` voxels whose predicted existence matches membership in `gt_grid`."""
    missing = gt_grid.ijk_to_index(grid.ijk).jdata == -1
    return float((logits.argmax(dim=1) == missing.long()).float().mean())


def grid_iou(grid: "GridBatch", gt_grid: "GridBatch") -> List[float]:
    """Per-sample voxel IoU between a predicted and a ground-truth grid batch."""
    in_gt = (gt_grid.ijk_to_index(grid.ijk).jdata >= 0).float()
    ious = []
    for b in range(grid.grid_count):
        start, end = int(grid.ijk.joffsets[b]), int(grid.ijk.joffsets[b + 1])
        intersection = float(in_gt[start:end].sum())
        union = (end - start) + int(gt_grid.num_voxels_at(b)) - intersection
        ious.append(intersection / (union + 1e-6))
    return ious


@torch.no_grad()
def evaluate(model: XCubeVAE, loader: PointCloudDataLoader, device: str) -> Dict[str, float]:
    sums: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for data in tqdm(loader, desc="Reconstructing"):
        out = model(
            data[DataKeys.POS].to(device),
            data[f"batch_{DataKeys.POS}"].to(device),
            normal=data[DataKeys.NORMAL].to(device),
        )
        gt_grids = out["hash_tree"]
        for depth, logits in out["structure_logits"].items():
            acc = structure_accuracy(logits, out["structure_logit_grids"][depth], gt_grids[depth])
            sums[f"struct-acc-{depth}"] = sums.get(f"struct-acc-{depth}", 0.0) + acc
            counts[f"struct-acc-{depth}"] = counts.get(f"struct-acc-{depth}", 0) + 1
        for iou in grid_iou(out["grid"], out["input_grid"]):
            sums["iou"] = sums.get("iou", 0.0) + iou
            counts["iou"] = counts.get("iou", 0) + 1
        # fvdb allocates grids outside the torch caching allocator; without releasing torch's cached
        # blocks every step, fvdb runs out of device memory on large 512^3 shapes.
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    return {name: sums[name] / counts[name] for name in sums}


def parse_args() -> Namespace:
    parser = ArgumentParser(description="XCube VAE ShapeNet reconstruction benchmark.")
    parser.add_argument("--model", type=str, default="xcube-vae-coarse-nvidia.shapenet-chair")
    parser.add_argument("--root", type=str, default=DATA_DIR, help="Parent directory containing XCubeShapeNet/.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default=DEVICE)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    model = create_model(args.model, task="base", pretrained=True)
    assert isinstance(model, XCubeVAE)
    model.to(args.device).eval()

    categories: Dict[str, XCubeShapeNetCategory] = {"chair": "Chair", "car": "Car", "plane": "Airplane"}
    category = categories[args.model.rsplit("-", 1)[-1]]
    resolution: Literal[128, 512] = 128 if "coarse" in args.model else 512
    dataset = XCubeShapeNet(args.root, split="test", resolution=resolution, categories=category)
    loader = PointCloudDataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        cat_keys=[DataKeys.POS, DataKeys.NORMAL],
    )

    print(f"Benchmarking {args.model!r} on XCube ShapeNet {category} ({len(dataset)} shapes)!")
    metrics = evaluate(model, loader, args.device)
    print("\nResults:")
    for name, value in metrics.items():
        print(f"  {name:<16} {value:.4f}")


if __name__ == "__main__":
    main()
