# SlidingWindowInferer

`SlidingWindowInferer` tiles the scene with axis-aligned cubic blocks of side `block_size` and calls the predictor once per non-empty block. Every block containing a point contributes to that point's output.

## Blocks, overlap, blending

Block centers sit on a grid spaced $\text{block\_size} \cdot (1 - \text{overlap})$ apart. At `overlap=0.0` (default) the blocks partition the scene and each point is predicted once. Raising `overlap` makes adjacent blocks share points, and `aggregate` decides how their predictions combine:

- `"mean"` (default): distance-weighted average. `mode="constant"` weights every prediction equally, `mode="gaussian"` down-weights predictions far from their block center, which softens seams. With `softmax=True` (default) the average is over probabilities rather than logits.
- `"max"`: each point keeps the prediction of the block most confident about it.
- `"vote"`: each block casts one hard vote per point, and the output holds per-class vote fractions, so `argmax` recovers the majority label.

`"max"` and `"vote"` always read confidences off the softmax, whatever `softmax` is set to.

!!! warning "`block_size` is in the units of `pos`, not always meters"
    The inferer tiles in whatever space `pos` is in at call time. If positions are voxel indices after an upstream voxelization, `block_size` is a voxel count: a scene voxelized at 2 cm and tiled with `block_size=200` gives 4 m blocks.

```{.python notest}
import torch_pointcloud as tp
from torch_pointcloud.inferers import SlidingWindowInferer

model = tp.create_model("pointnet2.s3dis-area5.xu-yan", task="segmentation", pretrained=True).eval()

inferer = SlidingWindowInferer(block_size=1.0)
probs = inferer(scene, predictor=lambda d: model(d["x"], d["pos"], d["batch"]))

# 25% overlap with Gaussian blending at boundaries
inferer = SlidingWindowInferer(block_size=1.0, overlap=0.25, mode="gaussian")
```

## When to use

Rooms too large for one pass, where a fixed metric block size matches how the model was trained. The block grid depends only on the scene extent, so tiling is deterministic. For a window that follows point density instead of metric size, use [`KNNWindowInferer`](knn-window.md); for no windowing at all, [`VoxelPartitionInferer`](voxel-partition.md).
