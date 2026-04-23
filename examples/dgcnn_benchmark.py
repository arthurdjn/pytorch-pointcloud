"""Benchmark a DGCNN classification model on ModelNet40.

Usage (with converted pretrained weights):

    python examples/dgcnn_benchmark.py \
        --weights weights/dgcnn-antao.modelnet40.1024.pt

Usage (random init, smoke test):

    python examples/dgcnn_benchmark.py --no-weights
"""

import os
from argparse import ArgumentParser, Namespace
from typing import Dict

import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm

from torch_pointcloud.config import DATA_DIR
from torch_pointcloud.datasets import ModelNetNormalResampled
from torch_pointcloud.models._registry import create_model
from torch_pointcloud.utils.data import collate
from torch_pointcloud.utils.random import seed_everything

CUDA_AVAILABLE = torch.cuda.is_available()
CPU_COUNT = os.cpu_count()
DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"
NUM_WORKERS = CPU_COUNT // 2 if CPU_COUNT is not None else 0
BATCH_SIZE = 32
NUM_POINTS = 1024
SEED = 42


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    print(f"Device: {args.device}")
    model, info = create_model(args.model, task="classification", return_info=True, pretrained=True)
    transform = info["transforms"]

    test_dataset = ModelNetNormalResampled(
        root=args.root,
        variant="40",
        train=False,
        transform=transform,
        download=args.download,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
    )

    print(f"Test set: {len(test_dataset)} samples")
    print("Evaluating...")
    metrics = evaluate(model, test_dataloader, args.device)

    print("\nResults:", end=" ")
    print(" | ".join(f"{k}: {v:.4f}" for k, v in metrics.items()))


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Benchmark DGCNN classification on ModelNet40.")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--root", type=str, default=DATA_DIR, help="Dataset root directory.")
    parser.add_argument(
        "--model",
        type=str,
        default="dgcnn-antao.modelnet40.1024",
        choices=["dgcnn-antao.modelnet40.1024", "dgcnn-antao.modelnet40.2048"],
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--device", type=str, default=DEVICE)
    parser.add_argument("--download", action="store_true", help="Download dataset if missing.")
    return parser.parse_args()


@torch.no_grad()
def evaluate(model: Module, dataloader: DataLoader, device: str) -> Dict[str, float]:
    model.to(device).eval()
    correct = 0
    total = 0

    pbar = tqdm(dataloader, total=len(dataloader), desc="Testing")
    for data in pbar:
        pos: Tensor = data["pos"].to(device)
        target: Tensor = data["label"].to(device)
        batch: Tensor = data["batch"].to(device)

        logits = model(None, pos, batch)
        preds = logits.argmax(dim=1)

        correct += preds.eq(target).sum().item()
        total += target.size(0)
        pbar.set_postfix({"acc": f"{correct / total:.4f}"})

    acc = correct / total
    return {"test/acc": acc}


if __name__ == "__main__":
    main()
