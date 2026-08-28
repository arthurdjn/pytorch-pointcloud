"""Benchmark SPVCNN semantic segmentation on SemanticKITTI (single pass, full-resolution scoring).

Results (val sequence 08, mIoU):

    | Variant                                   | reference | torch-pointcloud |
    | ----------------------------------------- | --------- | ---------------- |
    | spvcnn-119gmacs.semantickitti.mit-han-lab | 63.8      |                  |
    | spvcnn-47gmacs.semantickitti.mit-han-lab  | 61.4      |                  |
    | spvcnn-30gmacs.semantickitti.mit-han-lab  | 60.7      |                  |

Usage:
    uv run --no-sync python examples/spvcnn_benchmark_segmentation.py --limit 5
"""

import argparse
import os
from typing import Any, Dict

import torch
import torchsparse
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from torch_pointcloud.config import DATA_DIR
from torch_pointcloud.datasets import SemanticKITTI
from torch_pointcloud.inferers import Inferer, SimpleInferer
from torch_pointcloud.models import create_model
from torch_pointcloud.utils.data import DataKeys, PointCloudDataLoader
from torch_pointcloud.utils.metrics import confusion_matrix
from torch_pointcloud.utils.random import seed_everything, set_determinism

CUDA_AVAILABLE = torch.cuda.is_available()
CPU_COUNT = os.cpu_count()
DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"
NUM_WORKERS = CPU_COUNT // 2 if CPU_COUNT is not None else 0
SEED = 42
IGNORE_INDEX = 255

# torchsparse's default conv mode / kernel-map mode do not run this network; see
# https://github.com/mit-han-lab/torchsparse/issues/347#issuecomment-2920272471
torchsparse.nn.functional.set_conv_mode(2)
torchsparse.nn.functional.set_kmap_mode("hashmap")
ts_config = torchsparse.nn.functional.conv_config.get_default_conv_config()
ts_config.kmap_mode = "hashmap"
torchsparse.nn.functional.conv_config.set_global_conv_config(ts_config)


def build_inferer(seed: int) -> Inferer:
    return SimpleInferer()


@torch.no_grad()
def evaluate(model: Module, dataloader: DataLoader, inferer: Inferer, device: str, num_classes: int) -> Dict[str, Any]:
    model.to(device).eval()
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)

    pbar = tqdm(dataloader, total=len(dataloader), desc="Testing")
    for data in pbar:
        data = {key: value.to(device) if torch.is_tensor(value) else value for key, value in data.items()}
        scores = inferer(data, predictor=lambda d: model(d[DataKeys.X], d[DataKeys.POS], d[DataKeys.BATCH]))
        preds = scores.argmax(dim=1)[data[DataKeys.INVERSE]]
        cm += confusion_matrix(preds.cpu(), data[DataKeys.ORIGIN_SEGMENT].cpu(), num_classes, ignore_index=IGNORE_INDEX)
        oa = cm.diag().sum().float() / cm.sum().float().clamp_min(1)
        pbar.set_postfix({"oa": f"{oa.item():.4f}"})

    intersection = cm.diag().float()
    union = cm.sum(dim=1).float() + cm.sum(dim=0).float() - intersection
    iou_per_class = intersection / union.clamp_min(1e-10)
    return {
        "test/mIoU": iou_per_class.mean().item(),
        "test/oa": (cm.diag().sum().float() / cm.sum().float().clamp_min(1)).item(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark SPVCNN semantic segmentation on SemanticKITTI.")
    parser.add_argument(
        "--model", default="spvcnn-119gmacs.semantickitti.mit-han-lab", help="Registered segmentation model name"
    )
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--root", default=DATA_DIR, help="Dataset root directory.")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--seed", default=SEED, type=int)
    parser.add_argument("--num-workers", default=NUM_WORKERS, type=int)
    parser.add_argument("--limit", default=None, type=int, help="Evaluate at most this many scans.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    set_determinism(tf32=False)

    print(f"Benchmarking model {args.model!r} on SemanticKITTI!")
    model, model_info = create_model(args.model, task="segmentation", pretrained=True, return_info=True)
    num_classes = int(model.num_classes)
    inferer = build_inferer(args.seed)

    dataset: Dataset = SemanticKITTI(root=args.root, split=args.split, transform=model_info["transform"])
    if args.limit is not None:
        n = min(int(args.limit), len(dataset))  # type: ignore[arg-type]
        dataset = Subset(dataset, range(n))
        print(f"Evaluating on a subset of the first {n} scans.")

    dataloader = PointCloudDataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    print(f"Test set: {len(dataset)} scans  (single pass, scored at full resolution)")  # type: ignore[arg-type]
    metrics = evaluate(model, dataloader, inferer, args.device, num_classes)
    print("\nResults:")
    for key, value in metrics.items():
        print(f"  {key:<24} {value:.4f}")


if __name__ == "__main__":
    main()
