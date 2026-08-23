# PotentialSphereInferer

`PotentialSphereInferer` covers the scene with spheres of a fixed `radius`, centered where the cloud has been seen the least, and blends each sphere's softmax into the running per-point scores until every region has been covered about `num_votes` times. This is the test protocol of :arxiv: [KPConv](https://arxiv.org/abs/1904.08889).

## The potential loop

Coverage is tracked on a coarse grid of potentials, one scalar per `potential_size` cell (default `radius / 10`), initialized with small random noise. Each step:

1. Center a sphere of `radius` on the lowest-potential cell, plus a Gaussian `jitter`.
2. Raise the potentials of the cells it covers by the Tukey window $(1 - d^2 / r^2)^2$.
3. Run the predictor on the sphere, with positions centered on it.
4. Blend the sphere's softmax into the per-point scores by an EMA, restricted to points within $\text{inner\_ratio} \cdot \text{radius}$ of the center, where its context is complete.

The loop stops once every cell reaches `num_votes`. The output is the EMA of softmax probabilities, so points no sphere reaches keep all-zero scores. If no sphere with at least two points can be drawn, the inferer raises `ValueError`: `radius` is too small for the scale of `pos`.

```{.python notest}
import torch_pointcloud as tp
from torch_pointcloud.inferers import PotentialSphereInferer

model = tp.create_model("kpfcnn-base.s3dis.hugues-thomas", task="segmentation", pretrained=True).eval()

inferer = PotentialSphereInferer(radius=1.5, num_votes=10.0)
probs = inferer(room, predictor=lambda d: model(d["x"], d["pos"], d["batch"]))
```

## When to use

Models trained on radius-defined input spheres, where each forward pass expects a fixed metric extent rather than a fixed point count. The output is probabilities already, with no final softmax. For a fixed point budget instead, use [`KNNWindowInferer`](knn-window.md).
