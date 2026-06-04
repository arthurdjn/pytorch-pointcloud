r"""Evaluate `votenet-fair-base.sunrgbd` on SUN RGB-D val with mAP@0.25 / mAP@0.5.

Uniform benchmark shape: dataset -> `PointCloudDataLoader` -> model -> 3D-NMS AP. The
`SunRGBD` dataset reconstructs the upright cloud and oriented boxes from the raw release
(`torch_pointcloud.datasets.SunRGBD`), the registered model transform adds the floor-relative
height feature and samples to 20k points, and the AP (oriented box decode, per-class 3D NMS,
empty-box removal) is the faithful NumPy port in `torch_pointcloud.utils.detection`.

| Model                       | This script (mAP@0.25 / @0.50) | Reference (@0.25 / @0.50) |
| --------------------------- | ------------------------------ | ------------------------- |
| `votenet-fair-base.sunrgbd` | 59.04 / 34.27                  | 57.7 / 32.0               |

Measured on the full 5050-scene val split, reproducing the reference. The `SunRGBD` dataset's
pure-python extraction matches votenet's box convention (oriented boxes stored as
$[l, w, h] = [\text{coeffs}[1], \text{coeffs}[0], \text{coeffs}[2]]$; see `parse_boxes`).

Usage:
    uv run --no-sync python examples/votenet_benchmark_sunrgbd.py --root /path/to/data
"""

import os
from argparse import ArgumentParser, Namespace
from typing import Dict, List

import torch
from tqdm import tqdm

from torch_pointcloud.config import DATA_DIR
from torch_pointcloud.datasets import SunRGBD
from torch_pointcloud.models import VoteNetDetection, create_model
from torch_pointcloud.utils.data import PointCloudDataLoader
from torch_pointcloud.utils.detection import APCalculator, DatasetConfig, corners_from_boxes, parse_predictions
from torch_pointcloud.utils.random import seed_everything

CUDA_AVAILABLE = torch.cuda.is_available()
CPU_COUNT = os.cpu_count()
DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"
NUM_WORKERS = CPU_COUNT // 2 if CPU_COUNT is not None else 0
SEED = 42


def main() -> None:
    args = parse_args()
    print(f"Seeding everything to {args.seed}!")
    seed_everything(args.seed)

    print(f"Loading model {args.model!r}!")
    model, info = create_model(args.model, task="detection", pretrained=True, return_info=True)
    assert isinstance(model, VoteNetDetection)

    print("Loading SUN RGB-D val!")
    dataset = SunRGBD(root=args.root, split="val", transform=info["transforms"], download=args.download)
    dataloader = PointCloudDataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, cat_keys=["box"]
    )

    config = DatasetConfig(
        num_class=int(model.num_classes),
        num_heading_bin=model.num_heading_bin,
        num_size_cluster=model.num_size_cluster,
        mean_size_arr=model.mean_size_arr.cpu().numpy(),
        oriented=True,
    )

    print(f"Test set: {len(dataset)} scenes")
    print("Evaluating...")
    metrics = evaluate(model, dataloader, config, args.device, ap_iou=args.ap_iou)

    print("\nResults:")
    for name, value in metrics.items():
        print(f"  {name:<14} {value * 100:.2f}")


@torch.no_grad()
def evaluate(
    model: VoteNetDetection,
    dataloader: PointCloudDataLoader,
    config: DatasetConfig,
    device: str,
    *,
    ap_iou: List[float],
) -> Dict[str, float]:
    model.to(device).eval()
    calculators = {t: APCalculator(t) for t in ap_iou}

    for data in tqdm(dataloader, desc="SUN RGB-D val"):
        pos = data["pos"].to(device)
        x = data["x"].to(device)
        batch = data["batch"].to(device)
        out = model(x, pos, batch)

        batch_size = int(batch.max().item()) + 1
        num_points = pos.shape[0] // batch_size
        pc_dense = torch.cat([pos, x], dim=1).view(batch_size, num_points, -1)
        preds = parse_predictions(out, pc_dense, config)

        box, box_batch = data["box"], data["box_batch"]
        gts = [corners_from_boxes(box[box_batch == i].numpy()) for i in range(batch_size)]
        for calc in calculators.values():
            calc.step(preds, gts)

    return {f"mAP@{t:.2f}": calc.compute()[0] for t, calc in calculators.items()}


def parse_args() -> Namespace:
    parser = ArgumentParser(description="VoteNet SUN RGB-D detection AP benchmark.")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--root", type=str, default=DATA_DIR, help="Dataset root directory.")
    parser.add_argument("--model", type=str, default="votenet-fair-base.sunrgbd")
    parser.add_argument("--download", action="store_true", help="Download SUN RGB-D if missing.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--ap-iou", type=float, nargs="+", default=[0.25, 0.5])
    parser.add_argument("--device", type=str, default=DEVICE)
    return parser.parse_args()


if __name__ == "__main__":
    main()
