# VoxelPartitionInferer

`VoxelPartitionInferer` never crops. Each pass is a whole-extent uniform downsample, one point per voxel, repeated until every original point has been predicted, then scattered back to full resolution.

## K-pass interleaved downsampling

Points are bucketed into an FNV voxel grid at `voxel_size`. With $c_v$ points in voxel $v$, the inferer runs $K = \max_v c_v$ passes; pass $i$ picks the $(i \bmod c_v)$-th point of every bucket, so each sub-cloud spans the whole scene at roughly uniform density. Per-pass predictions are scatter-summed to the original point indices (in float64, for stable averaging) and, with `reduce="mean"`, divided by per-point participation counts. Each point is picked $\lfloor K / c_v \rfloor$ or $\lfloor K / c_v \rfloor + 1$ times across the $K$ passes, so every point gets a prediction. `sub_batch_size` packs several sub-clouds into one predictor call to amortise per-call costs like FPS and radius search.

## Usage

```{.python notest}
import torch_pointcloud as tp
from torch_pointcloud.inferers import VoxelPartitionInferer

model, info = tp.create_model(
    "pointnet2.s3dis-area5.openpoints", task="segmentation", pretrained=True, return_info=True
)
model = model.eval()

inferer = VoxelPartitionInferer(voxel_size=0.04, sub_batch_size=4, transform=info["transform"])
logits = inferer(room, predictor=lambda d: model(d["x"], d["pos"], d["batch"]))
```

## When to use

- The model wants **whole-scene context** in every forward pass, just at reduced density, with no block or window edges at all.
- Scenes are **dense but spatially compact**, so a uniform downsample preserves structure better than cropping.
- You pair it with [`TTAInferer`](tta.md): each augmentation pass triggers a fresh $K$-pass partition (use `reduce="sum"` for the un-normalised voting protocols).
- Memory is bounded by **points per voxel pass**, not scene extent; for very large extents where even one downsampled pass is too big, use [`SlidingWindowInferer`](sliding-window.md).
