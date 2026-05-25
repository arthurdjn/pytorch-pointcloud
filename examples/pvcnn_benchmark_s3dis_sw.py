"""Reproduce mit-han-lab's S3DIS Area-5 numbers for `pvcnn-mit-han-lab.s3dis-area5`.

Implements the PVCNN evaluation protocol from
https://github.com/mit-han-lab/pvcnn/blob/master/evaluate/s3dis/eval.py
and the matching preprocessing from
https://github.com/mit-han-lab/pvcnn/blob/master/data/s3dis/prepare_data.py:

- Translate each room so its $(\\min_x, \\min_y, \\min_z)$ sits at the origin.
- Compute room maxima $(x_\\max, y_\\max, z_\\max)$ for the room-normalised features.
- Tile the room in $X$-$Y$ at offsets $0$ m and $0.75$ m (`zero` / `half`); blocks
  are $1.5\\,\\text{m} \\times 1.5\\,\\text{m}$ and span the full room height.
- Per block: $4096$ points; input is
  $[x - (x_\\min^{\\text{blk}} + 0.75),\\; y - (y_\\min^{\\text{blk}} + 0.75),\\; z,\\;
    r/255,\\; g/255,\\; b/255,\\; x/x_\\max,\\; y/y_\\max,\\; z/z_\\max]$.
- Forward softmax votes are accumulated per scene point across overlapping blocks.

Reproduced performance on S3DIS Area-5 (default args: 1 vote, 1.5 m blocks, 0.75 m stride):

| Setting                             | Upstream paper | torch-pointcloud sliding-window |
| ----------------------------------- | -------------- | ------------------------------- |
| `pvcnn-mit-han-lab.s3dis-area5`     | 56.64 % mIoU   | 57.71 % mIoU / 86.58 % OA (seed=42, 1 vote) |

Usage:

    uv run --no-sync python examples/pvcnn_benchmark_s3dis_sw.py
    uv run --no-sync python examples/pvcnn_benchmark_s3dis_sw.py --limit 5
    uv run --no-sync python examples/pvcnn_benchmark_s3dis_sw.py --num-votes 3
"""

import math
import os
from argparse import ArgumentParser, Namespace
from typing import Any, Dict, Union

import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from torch_pointcloud.config import DATA_DIR
from torch_pointcloud.datasets import S3DIS
from torch_pointcloud.models._registry import create_model
from torch_pointcloud.utils.data import DataKeys, collate
from torch_pointcloud.utils.random import seed_everything

CUDA_AVAILABLE = torch.cuda.is_available()
CPU_COUNT = os.cpu_count()
DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"
NUM_WORKERS = CPU_COUNT // 2 if CPU_COUNT is not None else 0
SEED = 42


@torch.no_grad()
def predict_room(
    model: Module,
    pos: Tensor,
    color: Tensor,
    *,
    block_size: float,
    stride: float,
    block_points: int,
    padding: float,
    num_classes: int,
    device: str,
    generator: torch.Generator,
) -> Tensor:
    """Run mit-han-lab-style 2D sliding-window prediction over a single room.

    Returns per-point softmax-summed logits across overlapping blocks of shape $(N, C)$.
    """
    pos = pos - pos.amin(dim=0)
    n = pos.size(0)
    coord_min = pos.amin(dim=0)
    coord_max = pos.amax(dim=0)

    grid_x = max(1, int(math.ceil(float(coord_max[0] - coord_min[0] - block_size) / stride)) + 1)
    grid_y = max(1, int(math.ceil(float(coord_max[1] - coord_min[1] - block_size) / stride)) + 1)

    votes = torch.zeros(n, num_classes, dtype=torch.float64)
    color_f = color.float() / 255.0

    for index_y in range(grid_y):
        for index_x in range(grid_x):
            s_x = float(coord_min[0]) + index_x * stride
            e_x = min(s_x + block_size, float(coord_max[0]))
            s_x = e_x - block_size
            s_y = float(coord_min[1]) + index_y * stride
            e_y = min(s_y + block_size, float(coord_max[1]))
            s_y = e_y - block_size

            in_block = (
                (pos[:, 0] >= s_x - padding)
                & (pos[:, 0] <= e_x + padding)
                & (pos[:, 1] >= s_y - padding)
                & (pos[:, 1] <= e_y + padding)
            )
            point_idxs = torch.nonzero(in_block, as_tuple=False).flatten()
            if point_idxs.numel() == 0:
                continue

            n_in = int(point_idxs.numel())
            num_sub = math.ceil(n_in / block_points)
            point_size = num_sub * block_points
            n_extra = point_size - n_in
            if n_extra > 0:
                replace = n_extra > n_in
                extra = (
                    torch.randint(high=n_in, size=(n_extra,), generator=generator)
                    if replace
                    else torch.randperm(n_in, generator=generator)[:n_extra]
                )
                pick = torch.cat([point_idxs, point_idxs[extra]])
            else:
                pick = point_idxs
            pick = pick[torch.randperm(pick.numel(), generator=generator)].view(num_sub, block_points)

            for sub_idx in range(num_sub):
                sub_pick = pick[sub_idx]
                block_pos = pos[sub_pick]
                block_color = color_f[sub_pick]

                # PVCNN's preprocessing: block-centered xy in [-block_size/2, block_size/2]; z is room-relative.
                centered = block_pos.clone()
                centered[:, 0] -= s_x + block_size / 2.0
                centered[:, 1] -= s_y + block_size / 2.0
                # ch 2 (z) stays as room-relative — pos already has room min at 0.

                norm_pos = block_pos / coord_max.clamp_min(1e-6)
                x = torch.cat([centered, block_color, norm_pos], dim=1)

                batch = torch.zeros(block_points, dtype=torch.long, device=device)
                logits = model(x.to(device), centered.to(device), batch)
                probs = torch.softmax(logits, dim=-1).double().cpu()
                votes.index_add_(0, sub_pick, probs)

    return votes


def make_room_seed_generator(seed: int, room_index: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed * 1_000_003 + room_index)
    return g


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
    intersection = torch.zeros(num_classes, dtype=torch.float64)
    union = torch.zeros(num_classes, dtype=torch.float64)
    correct = total = 0

    pbar = tqdm(dataloader, total=len(dataloader), desc="Testing")
    for room_idx, data in enumerate(pbar):
        pos = data[DataKeys.POS][0] if isinstance(data[DataKeys.POS], list) else data[DataKeys.POS]
        color = data[DataKeys.COLOR][0] if isinstance(data[DataKeys.COLOR], list) else data[DataKeys.COLOR]
        target = data[DataKeys.SEGMENT][0] if isinstance(data[DataKeys.SEGMENT], list) else data[DataKeys.SEGMENT]

        votes = torch.zeros(pos.size(0), num_classes, dtype=torch.float64)
        for vote_i in range(num_votes):
            gen = make_room_seed_generator(seed + vote_i, room_idx)
            votes = votes + predict_room(
                model,
                pos,
                color,
                block_size=block_size,
                stride=stride,
                block_points=block_points,
                padding=padding,
                num_classes=num_classes,
                device=device,
                generator=gen,
            )
        preds = votes.argmax(dim=1)

        valid = target >= 0
        preds_v, target_v = preds[valid], target[valid]
        correct += int(preds_v.eq(target_v).sum().item())
        total += int(target_v.numel())
        for c in range(num_classes):
            inter = (preds_v == c) & (target_v == c)
            uni = (preds_v == c) | (target_v == c)
            intersection[c] += int(inter.sum())
            union[c] += int(uni.sum())

        miou_running = (intersection / union.clamp_min(1)).mean().item()
        pbar.set_postfix({"mIoU": f"{miou_running:.4f}", "oa": f"{correct / max(total, 1):.4f}"})

    iou_per_class = (intersection / union.clamp_min(1)).float()
    return {
        "test/mIoU": iou_per_class.mean().item(),
        "test/overall_acc": correct / max(total, 1),
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
    dataset: Union[S3DIS, "Subset[Any]"] = S3DIS(
        root=args.root,
        areas=list(args.areas),
        download=args.download,
        show_progress=False,
        num_workers=args.num_workers,
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
