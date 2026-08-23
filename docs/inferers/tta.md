# TTAInferer

`TTAInferer` wraps any base [inferer](overview.md) and runs it once per augmentation pass, aggregating the per-point predictions. Outputs are indexed by point id rather than by position, so predictions from rotated or flipped views are already aligned and no inverse transform is needed.

```{.python notest}
from torch_pointcloud.inferers import SlidingWindowInferer, TTAInferer
from torch_pointcloud.transforms import Compose, RandomFlip, RandomRotate

inferer = TTAInferer(
    base=SlidingWindowInferer(block_size=1.0),
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

`transforms` accepts two shapes. A single callable is re-sampled independently each pass, for random augmentation as above, and `num_passes` is required. A sequence of callables applies one element per pass in order, for a fixed view set, and `num_passes` is inferred from its length (a mismatched value is ignored with a warning).

```{.python notest}
# Enumerated 8-view TTA: one pass per fixed rotation.
views = [
    Compose([RandomRotate(keys="pos", angle_range=(a, a), axis=2, p=1.0)])
    for a in (0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0)
]
inferer = TTAInferer(base=base, transforms=views, aggregate="mean")
```

## When to use

The last points of mIoU on a trained segmentation model, where 4-pass rotation and flip TTA is a standard benchmark protocol, and ablating view sensitivity with an enumerated rotation set. It wraps any strategy, and each pass reruns the full base. Cost scales linearly with `num_passes`, so skip it when latency matters.
