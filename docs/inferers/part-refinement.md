# PartRefinementInferer

`PartRefinementInferer` post-processes another inferer's part-segmentation output. It takes the base inferer's per-point argmax and, for every shape, re-assigns the labels that are implausible (a predicted part with fewer than `min_count` points, or a part the shape's category does not own) by a majority vote of each point's nearest neighbors. This is the post-processing of the :arxiv: [PointNeXt](https://arxiv.org/abs/2206.04670) ShapeNetPart protocol.

![A part label the category cannot own, and the neighbor vote that replaces it](../assets/animations/part_refinement.webp)

The checkpoint gets the committed object right, so the frames mislabel part of it first: what the
refinement then repairs is a label this category cannot own.

## The refinement pass

The base inferer runs first and its scores are reduced to per-point labels by argmax. Per shape:

1. Count the points of every predicted label. A label is *implausible* when it has fewer than `min_count` points, or when it is not in the category's `part_ids` entry.
2. Refine each implausible label in turn, in ascending label order: its points take the majority label of their `num_neighbors` nearest points of the same shape, with the label under refinement excluded from the vote. Already-refined labels feed the next votes.
3. A point whose neighbors all carry the label under refinement keeps its label instead of falling through to class 0.

The return value is one-hot of shape $(N, C)$, so a metric's argmax recovers the refined labels. An empty scene returns the base output unchanged.

## Usage

```{.python notest}
from torch_pointcloud.inferers import PartRefinementInferer, SimpleInferer

inferer = PartRefinementInferer(SimpleInferer())
scores = inferer(
    shapes,
    predictor=lambda d: model(d["x"], d["pos"], d["batch"], d["category"]),
)
labels = scores.argmax(dim=1)
```

## When to use

- **ShapeNetPart-style benchmarks** whose reference protocol cleans up rare or foreign part labels before computing instance mIoU.
- On top of **any base inferer**: the refinement only needs the base's per-point scores.
- Skip it when calibrated scores matter downstream: the output is one-hot labels, not probabilities.
