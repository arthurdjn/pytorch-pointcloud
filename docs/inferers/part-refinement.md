# PartRefinementInferer

`PartRefinementInferer` post-processes another inferer's part-segmentation output. It takes the base's per-point argmax and re-assigns the implausible labels by a majority vote of each point's nearest neighbors. This is the post-processing of the :arxiv: [PointNeXt](https://arxiv.org/abs/2206.04670) ShapeNetPart protocol.

## The refinement pass

The base inferer runs first and its scores are reduced to per-point labels by argmax. Then per shape:

1. Count the points of every predicted label. A label is implausible when it has fewer than `min_count` points, or when the shape's category does not own it.
2. Refine implausible labels in ascending order: their points take the majority label of their `num_neighbors` nearest points in the same shape, with the label under refinement excluded from the vote. Already-refined labels feed the next votes.
3. A point whose neighbors all carry the label under refinement keeps its own, rather than falling through to class 0.

The return value is one-hot of shape $(N, C)$, so a metric's argmax recovers the refined labels. An empty scene returns the base output unchanged.

```{.python notest}
from torch_pointcloud.inferers import PartRefinementInferer, SimpleInferer

inferer = PartRefinementInferer(SimpleInferer())
scores = inferer(shapes, predictor=lambda d: model(d["x"], d["pos"], d["batch"], d["category"]))
labels = scores.argmax(dim=1)
```

## When to use

ShapeNetPart-style benchmarks whose reference protocol cleans up rare or foreign part labels before computing instance mIoU. It only needs the base's per-point scores, so it goes on top of any inferer. The output is one-hot labels, so skip it when a downstream step needs calibrated scores.
