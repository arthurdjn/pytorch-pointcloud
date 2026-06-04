r"""Evaluate `votenet-fair-base.scannet` on ScanNet-V2 val with mAP@0.25 / mAP@0.5.

Reproduces the reference detection AP with the packed VoteNet port. The AP (axis-aligned box
decode, per-class 3D NMS, empty-box removal, per-class-proposal AP) is the faithful NumPy port in
`torch_pointcloud.utils.detection`.

| Model                  | This script (AP@0.25) | Reference |
| ---------------------- | --------------------- | --------- |
| `votenet-fair-base.scannet` | 57.84% mAP@0.25       | 58.6%     |

Detection ground truth is read from facebookresearch/votenet's preprocessed export
(`{scene}_vert.npy` xyz[+rgb] and `{scene}_bbox.npy` axis-aligned $(K, 7)$). A native ScanNet
detection dataset that derives boxes from per-instance labels (as a transform) is the intended
uniform path; this script consumes the preprocessed export to keep the verified reproduction.

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
from typing import List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from torch_pointcloud.models import VoteNetDetection, create_model
from torch_pointcloud.utils.detection import APCalculator, DatasetConfig, corners_from_boxes, parse_predictions

# NYU40 ids of the 18 ScanNet detection classes (order = class index), from model_util_scannet.py.
NYU40_IDS = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def scannet_ground_truth(bbox: np.ndarray) -> List[Tuple[int, np.ndarray]]:
    """ScanNet GT boxes `(K, 7)` = [center, full size, nyu40] -> `[(sem_cls, corners(8, 3))]`, heading 0."""
    nyu40id2class = {int(nyu): i for i, nyu in enumerate(NYU40_IDS)}
    classes = np.array([nyu40id2class[int(x)] for x in bbox[:, 6]], dtype=np.float32)
    boxes = np.concatenate([bbox[:, 0:3], bbox[:, 3:6] / 2.0, np.zeros((bbox.shape[0], 1)), classes[:, None]], axis=1)
    return corners_from_boxes(boxes, half_sizes=True)


@torch.no_grad()
def evaluate(args: Namespace) -> None:
    model, info = create_model(args.model, task="detection", pretrained=True, return_info=True)
    assert isinstance(model, VoteNetDetection)
    model = model.to(args.device).eval()
    transform = info["transforms"]
    config = DatasetConfig(
        num_class=int(model.num_classes),
        num_heading_bin=model.num_heading_bin,
        num_size_cluster=model.num_size_cluster,
        mean_size_arr=model.mean_size_arr.cpu().numpy(),
        oriented=False,
    )

    data_root = Path(args.data_root)
    scans = [s.strip() for s in Path(args.split_file).read_text().splitlines() if s.strip()]
    scans = [s for s in scans if (data_root / f"{s}_vert.npy").exists()]
    if args.limit is not None:
        scans = scans[: args.limit]
    if not scans:
        raise FileNotFoundError(f"No `*_vert.npy` scenes found under {data_root}.")

    calculators = {t: APCalculator(t) for t in args.ap_iou}
    for scan in tqdm(scans, desc="ScanNet val"):
        bbox = np.load(data_root / f"{scan}_bbox.npy")
        if bbox.ndim == 1:
            bbox = bbox.reshape(0, 7)

        pos = torch.from_numpy(np.load(data_root / f"{scan}_vert.npy")[:, 0:3].astype("float32"))
        sample = transform({"pos": pos})
        pos_s, x_s = sample["pos"].to(args.device), sample["x"].to(args.device)
        batch = torch.zeros(pos_s.shape[0], dtype=torch.long, device=args.device)
        out = model(x_s, pos_s, batch)

        pc_dense = torch.cat([pos_s, x_s], dim=1).unsqueeze(0)
        preds = parse_predictions(out, pc_dense, config)
        for calc in calculators.values():
            calc.step(preds, [scannet_ground_truth(bbox)])

    print("\nResults:")
    for t, calc in calculators.items():
        mean_ap, _ = calc.compute()
        print(f"  mAP@{t:.2f}: {mean_ap * 100:.2f}")


def parse_args() -> Namespace:
    parser = ArgumentParser(description="VoteNet ScanNet-V2 detection AP benchmark.")
    parser.add_argument("--model", type=str, default="votenet-fair-base.scannet")
    parser.add_argument("--data-root", type=str, required=True, help="Dir with {scene}_vert.npy / {scene}_bbox.npy.")
    parser.add_argument("--split-file", type=str, required=True, help="scannetv2_val.txt path.")
    parser.add_argument("--ap-iou", type=float, nargs="+", default=[0.25, 0.5])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", type=str, default=DEVICE)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
