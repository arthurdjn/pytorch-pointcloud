r"""Reproduce mit-han-lab's S3DIS Area-5 numbers for `pvcnn-mit-han-lab.s3dis-area5`.

Implements the PVCNN evaluation protocol from
https://github.com/mit-han-lab/pvcnn/blob/master/evaluate/s3dis/eval.py
and the matching preprocessing from
https://github.com/mit-han-lab/pvcnn/blob/master/data/s3dis/prepare_data.py
on top of the library's own `SlidingWindowInferer` (no custom tiling loop):

- Shift each room so $(\min_x, \min_y, \min_z)$ sits at the origin.
- 2D tile the room (`dims=(0, 1)`) with $1.5\,\text{m}$ blocks at
  $0.75\,\text{m}$ stride (`block_size=1.5`, `overlap=0.5`) and a $1\,\text{mm}$
  membership margin (`padding=1e-3`).
- Force every predictor call to exactly $4096$ points via `roi_num_points=4096`
  composed with a `DivisiblePad(pad_fill="random")` transform that pads each
  block to a multiple of $4096$ via uniform sampling with replacement.
- Per-block features built as a single `Compose` chain of atomic transforms
  (`BBoxCenter`, `CopyItems`, `DivideKey`, `SubtractKey`, `ToFloat`, `Divide`,
  `Cat`) that produces $[x - c_x,\; y - c_y,\; z,\; r/255,\; g/255,\; b/255,\;
  x/x_\text{max},\; y/y_\text{max},\; z/z_\text{max}]$, where $(c_x, c_y)$ are
  the inferer-supplied `block_bbox`'s $xy$ midpoints.
- Softmax votes accumulated per point across overlapping blocks
  (`softmax=True`).
- Multi-vote is a thin outer loop that reseeds the inferer per pass.

Reproduced performance on S3DIS Area-5 (default args: 1 vote, 1.5 m blocks, 0.75 m stride):

| Setting                             | Upstream paper | torch-pointcloud sliding-window |
| ----------------------------------- | -------------- | ------------------------------- |
| `pvcnn-mit-han-lab.s3dis-area5`     | 56.64 % mIoU   | 57.51 % mIoU / 86.63 % OA (seed=42, 1 vote) |

Usage:

    uv run --no-sync python examples/pvcnn_benchmark_s3dis_sw.py
    uv run --no-sync python examples/pvcnn_benchmark_s3dis_sw.py --limit 5
    uv run --no-sync python examples/pvcnn_benchmark_s3dis_sw.py --num-votes 3
"""

import os
from argparse import ArgumentParser, Namespace
from typing import Any, Dict, Union

import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from torch_pointcloud import transforms as T
from torch_pointcloud.config import DATA_DIR
from torch_pointcloud.datasets import S3DIS
from torch_pointcloud.inferers import SlidingWindowInferer
from torch_pointcloud.models._registry import create_model
from torch_pointcloud.utils.data import DataKeys, collate
from torch_pointcloud.utils.ensemble import mean_ensemble
from torch_pointcloud.utils.metrics import confusion_matrix
from torch_pointcloud.utils.random import seed_everything

CUDA_AVAILABLE = torch.cuda.is_available()
CPU_COUNT = os.cpu_count()
DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"
NUM_WORKERS = CPU_COUNT // 2 if CPU_COUNT is not None else 0
SEED = 42
COORD_MAX_KEY = "coord_max"
NORM_POS_KEY = "norm_pos"
BLOCK_CENTER_KEY = "block_center"


def predict(model: Module, data: Dict[str, Any], device: str) -> Tensor:
    x = data[DataKeys.X].to(device)
    pos = data[DataKeys.POS].to(device)
    batch = data[DataKeys.BATCH].to(device)
    return model(x, pos, batch).cpu()


@torch.no_grad()
def evaluate(
    model: Module,
    dataloader: DataLoader,
    device: str,
    *,
    num_classes: int,
    block_size: float,
    stride: float,
    block_points: int,
    num_votes: int,
    padding: float,
    seed: int,
) -> Dict[str, float]:
    model.to(device).eval()
    overlap = 1.0 - stride / block_size
    cm = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    pbar = tqdm(dataloader, total=len(dataloader), desc="Testing")
    for room_idx, room in enumerate(pbar):
        target = room[DataKeys.SEGMENT]
        room_seed = seed * 1_000_003 + room_idx

        outputs = []
        for vote_i in range(num_votes):
            pad_rng = torch.Generator(device=room[DataKeys.POS].device).manual_seed(room_seed + vote_i)
            transform = T.Compose(
                [
                    T.BBoxCenter(keys="block_bbox", dst_keys=BLOCK_CENTER_KEY),
                    T.CopyItems(keys=DataKeys.POS, names=NORM_POS_KEY),
                    T.DivideKey(keys=NORM_POS_KEY, div_keys=COORD_MAX_KEY),
                    T.SubtractKey(keys=DataKeys.POS, sub_keys=BLOCK_CENTER_KEY, axes=[0, 1]),
                    T.ToFloat(keys=DataKeys.COLOR),
                    T.Divide(keys=DataKeys.COLOR, divisor=255.0),
                    T.Cat(keys=[DataKeys.POS, DataKeys.COLOR, NORM_POS_KEY], dst_key=DataKeys.X, dim=1),
                    T.DivisiblePad(num_samples=block_points, pad_fill="random", generator=pad_rng),
                ]
            )
            inferer = SlidingWindowInferer(
                block_size=block_size,
                overlap=overlap,
                dims=(0, 1),
                padding=padding,
                roi_num_points=block_points,
                softmax=True,
                transform=transform,
                seed=room_seed + vote_i,
            )
            outputs.append(inferer(room, predictor=lambda window: predict(model, window, device)))
        probs = mean_ensemble(outputs)

        preds = probs.argmax(dim=1)
        cm += confusion_matrix(preds, target, num_classes, ignore_index=-1)

        diag = cm.diag().float()
        iou = diag / (cm.sum(0) + cm.sum(1) - cm.diag()).clamp_min(1).float()
        pbar.set_postfix({"mIoU": f"{iou.mean().item():.4f}", "oa": f"{diag.sum().item() / max(int(cm.sum()), 1):.4f}"})

    diag = cm.diag().float()
    iou = diag / (cm.sum(0) + cm.sum(1) - cm.diag()).clamp_min(1).float()
    return {
        "test/mIoU": iou.mean().item(),
        "test/overall_acc": diag.sum().item() / max(int(cm.sum()), 1),
    }


def parse_args() -> Namespace:
    parser = ArgumentParser(description="mit-han-lab PVCNN S3DIS Area-5 reproduction (sliding-window).")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--root", type=str, default=DATA_DIR)
    parser.add_argument("--model", type=str, default="pvcnn-mit-han-lab.s3dis-area5")
    parser.add_argument("--areas", nargs="+", default=["Area_5"])
    parser.add_argument("--block-size", type=float, default=1.5)
    parser.add_argument("--stride", type=float, default=0.75)
    parser.add_argument("--block-points", type=int, default=4096)
    parser.add_argument("--padding", type=float, default=0.001)
    parser.add_argument("--num-votes", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--device", type=str, default=DEVICE)
    parser.add_argument("--download", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    print(f"Loading model {args.model!r}!")
    model = create_model(args.model, task="segmentation", pretrained=True)
    num_classes = int(model.num_classes)

    print(f"Loading raw S3DIS rooms from {args.areas}!")
    transform = T.Compose(
        [
            T.Shift(keys=DataKeys.POS, method="min"),
            T.Reduce(keys=DataKeys.POS, op="max", dst_keys=COORD_MAX_KEY),
        ]
    )
    dataset: Union[S3DIS, "Subset[Any]"] = S3DIS(
        root=args.root,
        areas=list(args.areas),
        download=args.download,
        show_progress=False,
        num_workers=args.num_workers,
        transform=transform,
    )
    if args.limit is not None:
        dataset = Subset(dataset, range(min(int(args.limit), len(dataset))))
        print(f"Subset: {len(dataset)} rooms.")

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=collate)

    print(
        f"PVCNN protocol: block={args.block_size} m, stride={args.stride} m, "
        f"block_points={args.block_points}, votes={args.num_votes}"
    )
    metrics = evaluate(
        model,
        dataloader,
        args.device,
        num_classes=num_classes,
        block_size=args.block_size,
        stride=args.stride,
        block_points=args.block_points,
        num_votes=args.num_votes,
        padding=args.padding,
        seed=args.seed,
    )

    print("\nResults:")
    for k, v in metrics.items():
        print(f"  {k:<20} {v:.4f}")


if __name__ == "__main__":
    main()
