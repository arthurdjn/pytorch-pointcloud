# KNNWindowInferer

`KNNWindowInferer` covers a large scene with overlapping kNN windows of a fixed point budget instead of fixed metric blocks. Each window holds exactly `roi_num_points` points, so crops adapt to point density.

![Successive kNN crops, each seeded at the point covered least so far, until the room is covered](../assets/animations/knn_window.webp)

The animation walks the strategy step by step: seed at the point covered least so far (the star), take
its $k$ nearest points as the crop (orange), predict, and reseed. The windows walk themselves across the
scene.

## The coverage loop

Every point starts with a coverage score initialized to small random noise. Each iteration:

1. Pick the `sw_batch_size` least-covered points as window centers.
2. Crop the `roi_num_points` nearest neighbors of each center and run the predictor on the packed crop.
3. Accumulate per-point predictions, weighted by distance to the center.
4. Bump each cropped point's coverage by a distance-based increment; points near the center gain more.

The loop ends once every point's coverage exceeds the `overlap` threshold. Because windows overlap, most points are predicted several times; `aggregate` controls how those predictions are combined: `"weighted_mean"` averages distance-weighted logits (probabilities with `softmax=True`), `"ema"` keeps a softmax exponential moving average and outputs calibrated probabilities directly. This is the possibility-based crop voting protocol of RandLA-Net-style pipelines.

## Usage

```{.python notest}
import torch_pointcloud as tp
from torch_pointcloud.inferers import KNNWindowInferer

model = tp.create_model(
    "randlanet.semantickitti.tsung-han-wu",
    task="segmentation",
    pretrained=True,
).eval()

# EMA aggregation: outputs calibrated probabilities directly.
inferer = KNNWindowInferer(roi_num_points=65_536, overlap=0.5, aggregate="ema")
probs = inferer(
    scene, predictor=lambda d: model(d.get("x"), d["pos"], d["batch"])
)
```

## When to use

- **Non-uniform density** (outdoor LiDAR, terrestrial scans) where fixed metric blocks would hold wildly varying point counts.
- The model expects a **fixed point budget** per forward pass (RandLA-Net-style pipelines).
- You want to reproduce a **reference evaluation protocol** built on possibility-based coverage (use `aggregate="ema"`, `sw_batch_size=1`).
- Prefer deterministic metric tiling instead? Use [`SlidingWindowInferer`](sliding-window.md). Radius-defined spheres? Use [`PotentialSphereInferer`](potential-sphere.md).
