# SlidingWindowInferer

`SlidingWindowInferer` tiles the scene with axis-aligned cubic blocks of side `block_size` and calls the predictor once per non-empty block. Predictions from every block that contains a point are blended back into one output per point.

![A window sweeping a room block by block, filling in the points each pass covers](../assets/animations/sliding_window.webp)

## Blocks, overlap, blending

Block centers sit on a regular grid spaced $\text{block\_size} \cdot (1 - \text{overlap})$ apart. With `overlap=0.0` (default) the blocks form a strict partition and each point is predicted exactly once. Raising `overlap` makes adjacent blocks share points: those points collect several predictions, combined according to `aggregate`:

- `"mean"` (default): distance-weighted average. `mode="constant"` weights every prediction equally; `mode="gaussian"` down-weights predictions far from their block center, which softens seams at block boundaries. With `softmax=True` (default) the average is over probabilities rather than raw logits.
- `"max"`: winner-takes-all; each point keeps the prediction of the block that is most confident about it.
- `"vote"`: each block casts one hard vote (its argmax) per point and the output holds the per-class vote fractions, so `argmax` recovers the majority label.

`"max"` and `"vote"` always read confidences off the softmax, regardless of the `softmax` flag.

!!! warning "`block_size` is in the units of `pos`, not always meters"

    The inferer tiles in whatever coordinate space `pos` is in at call time. If positions are voxel indices after upstream voxelization, `block_size` is a voxel count: a scene voxelized at $2\,\text{cm}$ tiled with `block_size=200` gives $4\,\text{m}$ blocks.

## Usage

```{.python notest}
import torch_pointcloud as tp
from torch_pointcloud.inferers import SlidingWindowInferer

model = tp.create_model(
    "pointnet2.s3dis-area5.xu-yan", task="segmentation", pretrained=True
).eval()

inferer = SlidingWindowInferer(block_size=1.0)
probs = inferer(scene, predictor=lambda d: model(d["x"], d["pos"], d["batch"]))

# 25 % overlap with Gaussian blending at boundaries:
inferer = SlidingWindowInferer(block_size=1.0, overlap=0.25, mode="gaussian")
```

## When to use

- **Rooms and scenes too large** for one forward pass, where a fixed metric block size matches how the model was trained (block-based S3DIS / ScanNet protocols).
- You want **deterministic tiling**: the block grid depends only on the scene extent, not on random centers.
- **Seam artifacts** at block boundaries matter: turn on `overlap` with `mode="gaussian"`.
- Window size should track point **density** rather than metric size? Use [`KNNWindowInferer`](knn-window.md). No windowing at all? Use [`VoxelPartitionInferer`](voxel-partition.md).
