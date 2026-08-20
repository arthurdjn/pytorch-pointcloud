"""Evaluate the OctFormer ModelNet40 classifier (single-pass, no voting).

Results vs reference (ModelNet40 overall accuracy):

    | Variant                             | reference |
    | ----------------------------------- | --------- |
    | octformer-base.modelnet40.octree-nn | 92.7      |

Usage:
    uv run --no-sync python examples/octformer_benchmark_modelnet.py
    uv run --no-sync python examples/octformer_benchmark_modelnet.py --fresh-sampling
"""

import os
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List

import torch
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import torch_pointcloud.transforms as T
from torch_pointcloud.config import DATA_DIR
from torch_pointcloud.datasets import ModelNet40
from torch_pointcloud.models import create_model
from torch_pointcloud.utils.data import DataKeys, collate
from torch_pointcloud.utils.imports import _OCNN_GITHUB_URL, optional_import
from torch_pointcloud.utils.metrics import confusion_matrix
from torch_pointcloud.utils.random import seed_everything, set_determinism

if TYPE_CHECKING:
    from ocnn.octree import Octree

Octree, _ = optional_import("ocnn.octree", "Octree", url=_OCNN_GITHUB_URL)

CUDA_AVAILABLE = torch.cuda.is_available()
CPU_COUNT = os.cpu_count()
DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"
NUM_WORKERS = CPU_COUNT // 2 if CPU_COUNT is not None else 0
BATCH_SIZE = 16
SEED = 42
NUM_SAMPLES = 8000

SAMPLE_TRANSFORM = T.Compose(
    [
        T.RandomSampleFaceVertices(
            keys=DataKeys.POS,
            face_key=DataKeys.FACE,
            normal_key=DataKeys.NORMAL,
            num_samples=NUM_SAMPLES,
        ),
        T.Shift(keys=DataKeys.POS, method="bbox"),
        T.Rescale(keys=DataKeys.POS, method="bbox"),
    ]
)

EVAL_TRANSFORM = T.Compose(
    [
        T.BoxMask(keys=DataKeys.POS, bbox=(-0.99, -0.99, -0.99, 0.99, 0.99, 0.99), dst_keys=DataKeys.BOX_MASK),
        T.ApplyMask(keys=[DataKeys.POS, DataKeys.NORMAL], mask_key=DataKeys.BOX_MASK),
        T.Abs(keys=DataKeys.NORMAL),
        T.ToTensor(keys=[DataKeys.POS, DataKeys.NORMAL], dtype=torch.float32),
        T.BuildOctree(
            pos_key=DataKeys.POS,
            octree_key=DataKeys.OCTREE,
            depth=6,
            full_depth=2,
            batch_size=1,
            normal_key=DataKeys.NORMAL,
        ),
        T.OctreeFeatures(
            keys=DataKeys.OCTREE,
            features_type="ND",
            nempty=False,
            dst_keys=DataKeys.X,
        ),
    ]
)


class FrozenEvalDataset(Dataset):
    def __init__(self, data: List[Dict[str, Any]], transform: T.Compose) -> None:
        self.data = data
        self.transform = transform

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.transform(dict(self.data[index]))

    def __len__(self) -> int:
        return len(self.data)


def load_frozen_eval_data(dataset: ModelNet40, force: bool = False) -> List[Dict[str, Any]]:
    split = "train" if dataset.train else "test"
    cache_path = Path(dataset.processed_dir, f"{split}_sampled_{NUM_SAMPLES}.pt")
    if cache_path.exists() and not force:
        with torch.serialization.safe_globals([DataKeys]):
            return torch.load(cache_path, weights_only=True)

    data_list: List[Dict[str, Any]] = []
    for data in tqdm(dataset, total=len(dataset), desc="Sampling eval set"):
        sampled = SAMPLE_TRANSFORM(data)
        data_list.append(
            {
                DataKeys.POS: sampled[DataKeys.POS],
                DataKeys.NORMAL: sampled[DataKeys.NORMAL],
                DataKeys.LABEL: sampled[DataKeys.LABEL],
            }
        )

    tmp_path = cache_path.with_name(cache_path.name + ".tmp")
    torch.save(data_list, tmp_path)
    tmp_path.replace(cache_path)
    return data_list


def main() -> None:
    args = parse_args()

    print(f"Seeding everything to {args.seed}!")
    seed_everything(args.seed)
    set_determinism(tf32=False)

    print(f"Loading model {args.model!r}!")
    model, model_info = create_model(
        args.model,
        task="classification",
        pretrained=True,
        return_info=True,
    )

    num_classes: int = int(model.num_classes)

    print("Loading ModelNet40 test dataset!")
    test_dataset: Dataset
    if args.fresh_sampling:
        test_dataset = ModelNet40(
            root=args.root,
            train=False,
            download=args.download,
            force_process=args.force_process,
            transform=model_info.get("transform"),
        )
    else:
        mesh_dataset = ModelNet40(
            root=args.root,
            train=False,
            download=args.download,
            force_process=args.force_process,
        )
        frozen_data = load_frozen_eval_data(mesh_dataset, force=args.force_process)
        test_dataset = FrozenEvalDataset(frozen_data, EVAL_TRANSFORM)

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
    )

    print(f"Test set: {len(test_dataset)} samples")
    print("Evaluating...")
    metrics = evaluate(model, test_dataloader, args.device, num_classes=num_classes)

    print("\nResults:")
    for k, v in metrics.items():
        print(f"  {k:<20} {v:.4f}")


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Benchmark OctFormer classification on ModelNet40.")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--root", type=str, default=DATA_DIR, help="Dataset root directory.")
    parser.add_argument(
        "--model",
        type=str,
        default="octformer-base.modelnet40.octree-nn",
        choices=["octformer-base.modelnet40.octree-nn"],
    )
    parser.add_argument("--download", action="store_true", help="Download ModelNet if missing.")
    parser.add_argument(
        "--fresh-sampling",
        action="store_true",
        help="Resample surface points from the meshes on every run without the box clip (non-reference variant).",
    )
    parser.add_argument(
        "--force-process",
        action="store_true",
        help="Regenerate the processed mesh cache and the frozen once-sampled eval set.",
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--device", type=str, default=DEVICE)
    return parser.parse_args()


@torch.no_grad()
def evaluate(model: Module, dataloader: DataLoader, device: str, *, num_classes: int) -> Dict[str, float]:
    model.to(device).eval()
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)

    pbar = tqdm(dataloader, total=len(dataloader), desc="Testing")
    for data in pbar:
        octree = data[DataKeys.OCTREE].to(device)
        x = data[DataKeys.X].to(device)
        label = data[DataKeys.LABEL].to(device)

        logits = model(x, octree, octree.depth)
        preds = logits.argmax(dim=1)

        cm += confusion_matrix(preds.cpu(), label.cpu(), num_classes)
        oa = cm.diag().sum().float() / cm.sum().float().clamp_min(1)
        pbar.set_postfix({"oa": f"{oa.item():.4f}"})

    oa = cm.diag().sum().float() / cm.sum().float()
    per_class_acc = cm.diag().float() / cm.sum(dim=1).float().clamp_min(1)
    mean_class_acc = per_class_acc.mean()

    return {
        "test/overall_acc": oa.item(),
        "test/mean_class_acc": mean_class_acc.item(),
    }


if __name__ == "__main__":
    main()
