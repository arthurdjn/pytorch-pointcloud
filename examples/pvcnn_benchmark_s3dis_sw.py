r"""Reproduce mit-han-lab's PVCNN numbers on S3DIS Area-5.

Follows the upstream evaluation protocol:

- https://github.com/mit-han-lab/pvcnn/blob/master/evaluate/s3dis/eval.py
- https://github.com/mit-han-lab/pvcnn/blob/master/data/s3dis/prepare_data.py

| Model                             | Paper        | This script                                 |
| --------------------------------- | ------------ | ------------------------------------------- |
| `pvcnn.s3dis-area5.mit-han-lab`   | 56.64 % mIoU | 57.51 % mIoU / 86.63 % OA (seed=42, 1 vote) |

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
from torch_pointcloud.models import create_model
from torch_pointcloud.utils.data import DataKeys, collate
from torch_pointcloud.utils.ensemble import mean_ensemble
from torch_pointcloud.utils.metrics import confusion_matrix
from torch_pointcloud.utils.random import seed_everything, set_determinism

CUDA_AVAILABLE = torch.cuda.is_available()
CPU_COUNT = os.cpu_count()
DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"
NUM_WORKERS = CPU_COUNT // 2 if CPU_COUNT is not None else 0
SEED = 42
COORD_MAX_KEY = "coord_max"


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
                    T.BBoxCenter(keys="block_bbox", dst_keys=DataKeys.BLOCK_CENTER),
                    T.CopyItems(keys=DataKeys.POS, names=DataKeys.NORM_POS),
                    T.DivideKey(keys=DataKeys.NORM_POS, div_keys=COORD_MAX_KEY),
                    T.SubtractKey(keys=DataKeys.POS, sub_keys=DataKeys.BLOCK_CENTER, axes=[0, 1]),
                    T.ToFloat(keys=DataKeys.COLOR),
                    T.Divide(keys=DataKeys.COLOR, divisor=255.0),
                    T.Cat(keys=[DataKeys.POS, DataKeys.COLOR, DataKeys.NORM_POS], dst_key=DataKeys.X, dim=1),
                    T.DivisiblePad(
                        num_samples=block_points,
                        pad_fill="random",
                        generator=pad_rng,
                        dst_inverse_key=DataKeys.INVERSE,
                    ),
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
                inverse_key=DataKeys.INVERSE,
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
    parser.add_argument("--model", type=str, default="pvcnn.s3dis-area5.mit-han-lab")
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
    set_determinism(tf32=False)

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
