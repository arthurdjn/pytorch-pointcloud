# TTAInferer

`TTAInferer` wraps any base [`Inferer`](overview.md) and runs it once per augmentation pass, aggregating the per-point predictions. Because outputs are indexed by point ID rather than spatial position, predictions from rotated or flipped views are already aligned; no inverse transform is needed.

## Wrap and compose

Test-time augmentation layers on top of whole-scene or windowed inference without changing the predictor:

```{.python notest}
from torch_pointcloud.inferers import SlidingWindowInferer, TTAInferer
from torch_pointcloud.transforms import Compose, RandomFlip, RandomRotate

base = SlidingWindowInferer(block_size=1.0)
inferer = TTAInferer(
    base=base,
    transforms=Compose([
        RandomRotate(keys="pos", angle_range=(-180.0, 180.0), axis=2, p=1.0),
        RandomFlip(keys="pos", axes=[0, 1], p=0.5),
    ]),
    num_passes=4,
    aggregate="mean",
)
probs = inferer(scene, predictor=lambda d: model(d["x"], d["pos"], d["batch"]))
```

## Random passes vs fixed views

`transforms` accepts two shapes:

- **Single callable**: re-sampled independently each pass. Use for random augmentations such as the uniformly random rotation above. `num_passes` is required.
- **Sequence of callables**: each element is applied to exactly one pass, in order. Use for a fixed view set; `num_passes` is inferred from the sequence length (a mismatched value is ignored with a warning).

```{.python notest}
# Enumerated 8-view TTA: one pass per fixed rotation.
views = [Compose([RandomRotate(keys="pos", angle_range=(a, a), axis=2, p=1.0)])
         for a in (0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0)]
inferer = TTAInferer(base=base, transforms=views, aggregate="mean")
```

## When to use

- Squeezing the **last points of mIoU** out of a trained segmentation model; 4-pass rotation/flip TTA is a standard benchmark protocol.
- **Ablating view sensitivity** with an enumerated rotation set.
- On top of any strategy: wrap [`SimpleInferer`](simple.md), [`SlidingWindowInferer`](sliding-window.md), [`KNNWindowInferer`](knn-window.md), or [`VoxelPartitionInferer`](voxel-partition.md); each augmentation pass reruns the full base strategy.
- Skip it when latency matters: cost scales linearly with `num_passes`.
