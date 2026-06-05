r"""Evaluate `votenet-fair-base.scannet` on ScanNet-V2 val with mAP@0.25 / mAP@0.5.

The benchmark is `create_model` -> `model.predict` -> `mean_average_precision3d`. Detection ground
truth is read from facebookresearch/votenet's preprocessed export (`{scene}_vert.npy` xyz[+rgb] and
`{scene}_bbox.npy` axis-aligned $(K, 7)$). A native ScanNet detection dataset that derives boxes from
per-instance labels (as a transform) is the intended uniform path; this script consumes the
preprocessed export to keep the verified reproduction.

| Model                       | This script (mAP@0.25) | Reference |
| --------------------------- | ---------------------- | --------- |
| `votenet-fair-base.scannet` | 57.84                  | 58.6      |

Data preparation (one-time, in a clone of facebookresearch/votenet):
    follow `scannet/README.md` to produce `scannet/scannet_train_detection_data/` and the
    `scannet/meta_data/scannetv2_val.txt` split list.

Usage:
    uv run --no-sync python examples/votenet_benchmark_scannet.py \
        --data-root /path/to/votenet/scannet/scannet_train_detection_data \
        --split-file /path/to/votenet/scannet/meta_data/scannetv2_val.txt
"""

from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import torch
from tqdm import tqdm

from torch_pointcloud.models import VoteNetDetectionModel, create_model
from torch_pointcloud.utils.metrics import mean_average_precision3d
from torch_pointcloud.utils.random import seed_everything

# NYU40 ids of the 18 ScanNet detection classes (order = class index), from model_util_scannet.py.
NYU40_IDS = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main() -> None:
    args = parse_args()
    print(f"Seeding everything to {args.seed}!")
    seed_everything(args.seed)

    model, info = create_model(args.model, task="detection", pretrained=True, return_info=True)
    model = model.to(args.device).eval()
    transform = info["transforms"]

    data_root = Path(args.data_root)
    scans = [s.strip() for s in Path(args.split_file).read_text().splitlines() if s.strip()]
    scans = [s for s in scans if (data_root / f"{s}_vert.npy").exists()]
    if args.limit is not None:
        scans = scans[: args.limit]
    if not scans:
        raise FileNotFoundError(f"No `*_vert.npy` scenes found under {data_root}.")

    print(f"Benchmarking model {args.model!r} on ScanNet (split={args.split!r})!")
    metrics = evaluate(model, scans, data_root, transform, args.device, iou_thresholds=args.iou_thresholds)

    print("\nResults:")
    for name, value in metrics.items():
        print(f"  {name:<10} {value * 100:.2f}")


@torch.no_grad()
def evaluate(
    model: VoteNetDetectionModel,
    scans: List[str],
    data_root: Path,
    transform: Callable,
    device: str,
    *,
    iou_thresholds: List[float],
) -> None:
    metrics = {}
    all_preds, all_targets = [], []
    pbar = tqdm(scans, desc="ScanNet val")
    for scan in pbar:
        bbox = np.load(data_root / f"{scan}_bbox.npy")
        if bbox.ndim == 1:
            bbox = bbox.reshape(0, 7)

        pos = torch.from_numpy(np.load(data_root / f"{scan}_vert.npy")[:, 0:3].astype("float32"))
        sample = transform({"pos": pos})
        pos_s, x_s = sample["pos"].to(device), sample["x"].to(device)
        batch = torch.zeros(pos_s.shape[0], dtype=torch.long, device=device)

        pred = model.decode(model(x_s, pos_s, batch), pos_s, batch)
        target = encode_scannet_target(bbox)

        metrics = mean_average_precision3d(pred, target, iou_thresholds=iou_thresholds)
        pbar.set_postfix({name: f"{value * 100:.2f}" for name, value in metrics.items()})

        all_preds.append(pred)
        all_targets.append(target)

    return mean_average_precision3d(all_preds, all_targets, iou_thresholds=iou_thresholds)


def encode_scannet_target(bbox: np.ndarray) -> Dict[str, torch.Tensor]:
    """ScanNet GT `(K, 7)` = [center, full size, nyu40] -> `{boxes (K, 7), labels (K,)}` (axis-aligned)."""
    nyu40id2class = {int(nyu): i for i, nyu in enumerate(NYU40_IDS)}
    labels = torch.tensor([nyu40id2class[int(x)] for x in bbox[:, 6]], dtype=torch.long)
    boxes = np.concatenate([bbox[:, 0:3], bbox[:, 3:6], np.zeros((bbox.shape[0], 1))], axis=1)
    batch = torch.zeros(bbox.shape[0], dtype=torch.long)
    return {"boxes": torch.from_numpy(boxes.astype("float32")), "labels": labels, "batch": batch}


def parse_args() -> Namespace:
    parser = ArgumentParser(description="VoteNet ScanNet-V2 detection AP benchmark.")
    parser.add_argument("--model", type=str, default="votenet-fair-base.scannet")
    parser.add_argument("--data-root", type=str, required=True, help="Dir with {scene}_vert.npy / {scene}_bbox.npy.")
    parser.add_argument("--split-file", type=str, required=True, help="scannetv2_val.txt path.")
    parser.add_argument("--ap-iou", type=float, nargs="+", default=[0.25, 0.5])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", type=str, default=DEVICE)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    main()
