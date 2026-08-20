r"""Evaluate `pointnet2.s3dis-area5.openpoints` on S3DIS via the voxel-partition test protocol.

| Model                              | This script             | Reference  |
| ---------------------------------- | ----------------------- | ---------- |
| `pointnet2.s3dis-area5.openpoints` | 63.59% mIoU / 88.23% OA | 63.6% mIoU |

Usage:

    uv run --no-sync python examples/pointnet2_benchmark_s3dis_openpoints.py
    uv run --no-sync python examples/pointnet2_benchmark_s3dis_openpoints.py --limit 5
"""

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
from torch_pointcloud.inferers import VoxelPartitionInferer
from torch_pointcloud.models import create_model
from torch_pointcloud.utils.data import DataKeys, collate
from torch_pointcloud.utils.metrics import confusion_matrix
from torch_pointcloud.utils.random import seed_everything, set_determinism

CUDA_AVAILABLE = torch.cuda.is_available()
CPU_COUNT = os.cpu_count()
DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"
NUM_WORKERS = CPU_COUNT // 2 if CPU_COUNT is not None else 0
SEED = 42


def predict(model: Module, data: Dict[str, Any], device: str) -> Tensor:
    x = data[DataKeys.X].to(device)
    pos = data[DataKeys.POS].to(device)
    batch = data[DataKeys.BATCH].to(device)
    return model(x, pos, batch)


@torch.no_grad()
def evaluate(
    model: Module,
    model_transform: Any,
    dataloader: DataLoader,
    device: str,
    *,
    num_classes: int,
    voxel_size: float,
    seed: int,
    sub_batch_size: int,
) -> Dict[str, float]:
    model.to(device).eval()
    cm = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    pbar = tqdm(dataloader, total=len(dataloader), desc="Testing")
    for room_idx, room in enumerate(pbar):
        inferer = VoxelPartitionInferer(
            voxel_size=voxel_size,
            transform=model_transform,
            sub_batch_size=sub_batch_size,
            seed=seed * 1_000_003 + room_idx,
        )
        logits = inferer(room, predictor=lambda window: predict(model, window, device))
        cm += confusion_matrix(logits.argmax(dim=1).cpu(), room[DataKeys.SEGMENT], num_classes, ignore_index=-1)

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
    parser = ArgumentParser(description="openpoints PointNet++ S3DIS reproduction (Pointcept-style scatter-back).")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--root", type=str, default=DATA_DIR)
    parser.add_argument("--model", type=str, default="pointnet2.s3dis-area5.openpoints")
    parser.add_argument("--areas", nargs="+", default=["Area_5"])
    parser.add_argument("--voxel-size", type=float, default=0.04, help="FNV voxel partition size (m).")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--sub-batch-size",
        type=int,
        default=4,
        help="Number of voxel sub-clouds to pack into one model forward (memory / speed tradeoff).",
    )
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--device", type=str, default=DEVICE)
    parser.add_argument("--download", action="store_true")
    parser.add_argument(
        "--aligned",
        action="store_true",
        help="Use S3DIS dataset with `aligned=True` (Pointcept-style global rotation).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    set_determinism(tf32=False)

    print(f"Loading model {args.model!r}!")
    model, model_info = create_model(args.model, task="segmentation", pretrained=True, return_info=True)
    num_classes = int(model.num_classes)
    model_transform = model_info["transform"]
    if model_transform is None:
        raise RuntimeError(f"Model {args.model!r} has no registered transform; cannot run the multi-voxel protocol.")

    print(f"Loading raw S3DIS rooms from {args.areas}!")
    dataset: Union[S3DIS, "Subset[Any]"] = S3DIS(
        root=args.root,
        areas=list(args.areas),
        aligned=args.aligned,
        download=args.download,
        show_progress=False,
        num_workers=args.num_workers,
    )
    if args.limit is not None:
        dataset = Subset(dataset, range(min(int(args.limit), len(dataset))))
        print(f"Subset: {len(dataset)} rooms.")

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=collate)
    print(f"Voxel: {args.voxel_size} m (voxel-partition scatter-back testing)")

    metrics = evaluate(
        model,
        model_transform,
        dataloader,
        args.device,
        num_classes=num_classes,
        voxel_size=args.voxel_size,
        seed=args.seed,
        sub_batch_size=args.sub_batch_size,
    )
    print("\nResults:")
    for k, v in metrics.items():
        print(f"  {k:<20} {v:.4f}")


if __name__ == "__main__":
    main()
