"""Benchmark ForAINet and SegmentAnyTree on the FOR-instance test plots over 8 m cylinders, scored at full resolution.

NOTE: the references are the papers' FOR-instance test F1 (SegmentAnyTree 83.9, ForAINet 85.1), each scored by its
authors' own tree matching. SegmentAnyTree's `eval.py` (SmartForest-no/SegmentAnyTree at a3561ed8, the plots prepared
as its `run_paper_test.sh` does for `for_instance_no_outer`) scored with this script's metrics gives F1 72.42.
ForAINet's own evaluation (prs-eth/ForAINet) was not run.
"""

import os
from argparse import ArgumentParser, Namespace
from typing import Any, Dict, List, Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

import torch_pointcloud.transforms as T
from torch_pointcloud.config import DATA_DIR
from torch_pointcloud.datasets import FORinstance
from torch_pointcloud.datasets.forinstance import (
    FORINSTANCE_LEARNING_MAP,
    FORINSTANCE_OUT_POINTS,
    FORINSTANCE_TREE_LEARNING_MAP,
)
from torch_pointcloud.inferers import CylinderInstanceSegmentationInferer
from torch_pointcloud.inferers.cylinder import MergeCriterion
from torch_pointcloud.metrics import (
    accuracy,
    confusion_matrix,
    instance_coverage,
    instance_f1_score,
    instance_matches,
    instance_precision,
    instance_recall,
    intersection_over_union,
)
from torch_pointcloud.metrics.instance_segmentation import InstanceMatches
from torch_pointcloud.models import ForAINetInstanceSegmentation, create_model
from torch_pointcloud.ops.cluster import knn
from torch_pointcloud.ops.mask import mask_nms, mask_to_instance
from torch_pointcloud.utils.data import DataKeys, PointCloudDataLoader
from torch_pointcloud.utils.random import seed_everything, set_determinism
from torch_pointcloud.utils.types import InstanceSegmentation3D

CUDA_AVAILABLE = torch.cuda.is_available()
CPU_COUNT = os.cpu_count()
DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"
NUM_WORKERS = CPU_COUNT // 2 if CPU_COUNT is not None else 0
SEED = 2022
VOXEL_SIZE = 0.2
CYLINDER_RADIUS = 8.0
NMS_THRESHOLD = 0.3
MIN_POINTS = 10
MAX_DISTANCE = 1.0
IOU_THRESHOLD = 0.5
# FOR-instance class codes of each model's semantic classes; the out-points are removed before inference.
LEARNING_MAPS = {
    "segment-any-tree.for-instance.nibio": FORINSTANCE_TREE_LEARNING_MAP,
    "forainet.for-instance.prs-eth": FORINSTANCE_LEARNING_MAP,
}
# The fourth input feature: SegmentAnyTree reads z above the plot's lowest point, ForAINet the elevation itself.
HEIGHTS = {
    "segment-any-tree.for-instance.nibio": T.AxisMinOffset(keys=DataKeys.POS, axis=2, dst_keys=DataKeys.HEIGHT),
    "forainet.for-instance.prs-eth": T.Slice(keys=DataKeys.POS, start=2, stop=3, dim=1, dst_keys=DataKeys.HEIGHT),
}
# Each checkpoint's own proposal score threshold and block merging rule.
MIN_SCORES = {"segment-any-tree.for-instance.nibio": 0.6, "forainet.for-instance.prs-eth": 0.5}
MERGE_RULES: Dict[str, Tuple[MergeCriterion, float]] = {
    "segment-any-tree.for-instance.nibio": ("iou", 0.1),
    "forainet.for-instance.prs-eth": ("overlap", 0.3),
}

INFERER_TRANSFORM = T.Compose(
    [
        T.Shift(keys=DataKeys.POS, method="centroid"),
        T.Cat(keys=[DataKeys.POS, DataKeys.HEIGHT], dst_key=DataKeys.X, dim=1),
        T.Voxelize(
            pos_key=DataKeys.POS,
            pos_reduce="first",
            size=VOXEL_SIZE,
            reduce="first",
            keys=[DataKeys.X, DataKeys.HEIGHT],
            dst_pos_grid_key=DataKeys.POS_GRID,
            dst_inverse_key=DataKeys.INVERSE,
            random_first=True,
        ),
    ]
)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    set_determinism(tf32=False)

    print(f"Benchmarking model {args.model!r} on FOR-instance!")
    model = configure_model(args)
    inferer = configure_inferer(args)
    dataloader = configure_dataloader(args)

    print(f"Test set: {len(dataloader.dataset)} plots")  # type: ignore[arg-type]
    metrics = evaluate(model, dataloader, inferer, args)
    print("\nResults:")
    for key, value in metrics.items():
        print(f"  {key:<24} {value * 100:.2f}")


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Benchmark forest instance segmentation on FOR-instance.")
    parser.add_argument("--model", default="segment-any-tree.for-instance.nibio", choices=list(LEARNING_MAPS))
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--root", default=DATA_DIR, help="Dataset root directory.")
    parser.add_argument("--download", action="store_true", help="Download the FOR-instance archive (1.6 GB).")
    parser.add_argument("--seed", default=SEED, type=int)
    parser.add_argument("--num-workers", default=NUM_WORKERS, type=int)
    parser.add_argument("--sw-batch-size", default=8, type=int, help="Cylinders per forward pass.")
    parser.add_argument("--limit", default=None, type=int, help="Evaluate at most this many plots.")
    return parser.parse_args()


def configure_dataset(args: Namespace) -> Dataset:
    transform = T.Compose(
        [
            T.LabelMask(keys=DataKeys.SEGMENT, labels=[FORINSTANCE_OUT_POINTS], invert=True, dst_keys="inside"),
            T.ApplyMask(keys=[DataKeys.POS, DataKeys.SEGMENT, DataKeys.INSTANCE], mask_key="inside"),
            T.Relabel(keys=DataKeys.SEGMENT, labels=LEARNING_MAPS[args.model], default=-1),
            HEIGHTS[args.model],
            # UTM coordinates need float64; relative to the plot's corner they fit float32.
            T.Shift(keys=DataKeys.POS, method="min"),
            T.ToFloat(keys=[DataKeys.POS, DataKeys.HEIGHT]),
            T.CopyItems(
                keys=[DataKeys.POS, DataKeys.SEGMENT, DataKeys.INSTANCE],
                dst_keys=[DataKeys.ORIGIN_POS, DataKeys.ORIGIN_SEGMENT, DataKeys.ORIGIN_INSTANCE],
            ),
            T.Voxelize(
                pos_key=DataKeys.POS,
                pos_reduce="first",
                size=VOXEL_SIZE,
                reduce="first",
                keys=[DataKeys.HEIGHT],
                random_first=True,
            ),
        ]
    )
    dataset: Dataset
    dataset = FORinstance(root=args.root, train=False, transform=transform, download=args.download)
    if args.limit is not None:
        n = min(int(args.limit), len(dataset))
        dataset = Subset(dataset, range(n))
        print(f"Evaluating on a subset of the first {n} plots.")

    return dataset


def configure_dataloader(args: Namespace) -> DataLoader:
    dataset = configure_dataset(args)
    return PointCloudDataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)


def configure_model(args: Namespace) -> ForAINetInstanceSegmentation:
    model = create_model(args.model, task="instance-segmentation", pretrained=True)
    assert isinstance(model, ForAINetInstanceSegmentation)
    return model.to(args.device).eval()


def configure_inferer(args: Namespace) -> CylinderInstanceSegmentationInferer:
    merge_criterion, merge_threshold = MERGE_RULES[args.model]
    return CylinderInstanceSegmentationInferer(
        radius=CYLINDER_RADIUS,
        spacing=CYLINDER_RADIUS,
        align=True,
        merge_criterion=merge_criterion,
        merge_threshold=merge_threshold,
        transform=INFERER_TRANSFORM,
        inverse_key=DataKeys.INVERSE,
        sw_batch_size=args.sw_batch_size,
    )


@torch.no_grad()
def evaluate(
    model: ForAINetInstanceSegmentation,
    dataloader: DataLoader,
    inferer: CylinderInstanceSegmentationInferer,
    args: Namespace,
) -> Dict[str, float]:
    min_score = MIN_SCORES[args.model]
    thing_classes = torch.tensor(
        [c for c in range(model.num_semantic_classes) if c not in model.stuff_classes],
        device=args.device,
    )

    def predictor(batch: Dict[str, Any]) -> InstanceSegmentation3D:
        output = model(batch[DataKeys.X], batch[DataKeys.POS], batch[DataKeys.POS_GRID], batch[DataKeys.BATCH])
        proposals = model.decode(output)
        keep = mask_nms(proposals["mask_index"], proposals["scores"], NMS_THRESHOLD)
        sizes = torch.bincount(proposals["mask_index"][0], minlength=proposals["scores"].numel())
        keep = keep[(proposals["scores"][keep] > min_score) & (sizes[keep] > MIN_POINTS)]
        ignore_mask = torch.ones_like(proposals["scores"], dtype=torch.bool)
        ignore_mask[keep] = False
        return {
            "mask_index": proposals["mask_index"],
            "labels": proposals["labels"],
            "scores": proposals["scores"],
            "batch": proposals["batch"],
            "ignore_mask": ignore_mask,
            "semantic_logits": output["semantic_logits"].log_softmax(dim=1),
        }

    num_classes = model.num_semantic_classes
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    matches: List[InstanceMatches] = []
    for data in tqdm(dataloader, desc="Plots"):
        data = {key: value.to(args.device) if torch.is_tensor(value) else value for key, value in data.items()}
        scene: Dict[str, Any] = {key: data[key] for key in (DataKeys.POS, DataKeys.HEIGHT, DataKeys.BATCH)}
        instances = inferer(scene, predictor)
        semantic, instance = full_resolution_labels(
            data[DataKeys.POS],
            instances,
            data[DataKeys.ORIGIN_POS],
            thing_classes,
        )

        target, target_instance = data[DataKeys.ORIGIN_SEGMENT], data[DataKeys.ORIGIN_INSTANCE]
        cm += confusion_matrix(semantic, target, num_classes, ignore_index=-1).cpu()
        # Trees are scored on the points that the ground truth or the prediction puts in a tree class.
        tree = torch.isin(target, thing_classes) | torch.isin(semantic, thing_classes)
        labelled = (instance[tree] >= 0).nonzero().squeeze(1)
        ids, inverse = torch.unique(instance[tree][labelled], return_inverse=True)
        trees: InstanceSegmentation3D = {
            "mask_index": torch.stack([inverse, labelled]),
            "labels": torch.zeros_like(ids),
            "scores": instances["scores"][ids],
            "batch": torch.zeros_like(ids),
        }
        matches.append(instance_matches(trees, target_instance[tree], torch.zeros_like(target_instance[tree])))

    stuff, thing = list(model.stuff_classes), thing_classes.tolist()
    binary = torch.stack([cm[:, stuff].sum(dim=1), cm[:, thing].sum(dim=1)], dim=1)
    binary = torch.stack([binary[stuff].sum(dim=0), binary[thing].sum(dim=0)])
    return {
        "precision": instance_precision(matches, iou_threshold=IOU_THRESHOLD),
        "recall": instance_recall(matches, iou_threshold=IOU_THRESHOLD),
        "F1": instance_f1_score(matches, iou_threshold=IOU_THRESHOLD),
        "MUCov": instance_coverage(matches),
        "MWCov": instance_coverage(matches, weighted=True),
        "mIoU": intersection_over_union(cm),
        "OA": accuracy(cm, average="micro"),
        "mAcc": accuracy(cm, average="macro"),
        "tree IoU": float(intersection_over_union(binary, average="none")[1]),
        "binary mIoU": intersection_over_union(binary),
    }


def full_resolution_labels(
    pos: Tensor,
    instances: InstanceSegmentation3D,
    origin_pos: Tensor,
    thing_classes: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Semantic class and tree of every original point, from its nearest sampled point.

    A point takes the summed votes of its nearest sampled point and the tree of its nearest point in a tree, unless
    its class is not a tree class or that point lies more than `MAX_DISTANCE` away; trees left with `MIN_POINTS`
    points or fewer are dropped.
    """
    semantic = instances["semantic_logits"][nearest_point(pos, origin_pos)].argmax(dim=1)

    instance = mask_to_instance(instances["mask_index"], instances["scores"], pos.size(0))
    labelled = (instance >= 0).nonzero().squeeze(1)
    if labelled.numel() == 0:
        return semantic, torch.full_like(semantic, -1)

    nearest = labelled[nearest_point(pos[labelled], origin_pos)]
    full = instance[nearest]
    far = (pos[nearest] - origin_pos).norm(dim=1) > MAX_DISTANCE
    full[far | ~torch.isin(semantic, thing_classes)] = -1
    ids, inverse, counts = torch.unique(full, return_inverse=True, return_counts=True)
    full[(counts < MIN_POINTS)[inverse]] = -1
    return semantic, full


def nearest_point(source: Tensor, query: Tensor) -> Tensor:
    """Index of the nearest `source` point of every `query` point."""
    target, index = knn(source, query, 1)
    return torch.empty_like(index).scatter_(0, target, index)


if __name__ == "__main__":
    main()
