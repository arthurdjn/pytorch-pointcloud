# KNNWindowInferer

`KNNWindowInferer` covers a scene with overlapping kNN windows of a fixed point budget instead of fixed metric blocks. Each window holds exactly `roi_num_points` points, so crops adapt to point density.

## The coverage loop

Every point starts with a coverage score of small random noise. Each iteration:

1. Pick the `sw_batch_size` least-covered points as window centers.
2. Crop the `roi_num_points` nearest neighbors of each center and run the predictor on the packed crop.
3. Accumulate per-point predictions, weighted by distance to the center.
4. Raise each cropped point's coverage, most for points near the center.

The loop ends once every point's coverage passes the `overlap` threshold. Windows overlap, so most points are predicted several times, and `aggregate` combines those predictions: `"weighted_mean"` averages distance-weighted logits (probabilities with `softmax=True`), `"ema"` keeps a softmax exponential moving average and outputs calibrated probabilities directly.

```{.python notest}
import torch_pointcloud as tp
from torch_pointcloud.inferers import KNNWindowInferer

model = tp.create_model("randlanet.semantickitti.tsung-han-wu", task="segmentation", pretrained=True).eval()

inferer = KNNWindowInferer(roi_num_points=65_536, overlap=0.5, aggregate="ema")
probs = inferer(scene, predictor=lambda d: model(d.get("x"), d["pos"], d["batch"]))
```

## When to use

Non-uniform density (outdoor LiDAR, terrestrial scans) where fixed metric blocks would hold wildly varying point counts, and models trained on a fixed point budget per pass. To reproduce a possibility-based reference protocol, set `aggregate="ema"` and `sw_batch_size=1`. For deterministic metric tiling use [`SlidingWindowInferer`](sliding-window.md), for radius-defined spheres [`PotentialSphereInferer`](potential-sphere.md).
