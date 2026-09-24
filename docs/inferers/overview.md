# Inferers

An inferer decides how a model is run at test time: on the whole scene, on fragments of it, or under several views. It returns one prediction per point.

```{.python notest}
import torch
from torch_pointcloud import create_model
from torch_pointcloud.inferers import SlidingWindowInferer

scene = {
    "pos": torch.rand(20_000, 3) * 10,
    "x": torch.rand(20_000, 3),
    "batch": torch.zeros(20_000, dtype=torch.long),
}
model = create_model(...)

inferer = SlidingWindowInferer(block_size=5.0, softmax=True)
scores = inferer(scene, predictor=lambda d: model(d["x"], d["pos"], d["batch"]))
```

| Inferer                                                         | Description                                                      |
| --------------------------------------------------------------- | ---------------------------------------------------------------- |
| [`SimpleInferer`](../api/inferers/simple.md)                    | Single forward pass on the whole scene                           |
| [`SlidingWindowInferer`](../api/inferers/sliding_window.md)     | Blocks on a regular grid, predicted one at a time                |
| [`VoxelPartitionInferer`](../api/inferers/voxel_partition.md)   | Passes of one point per voxel, each over the whole scene         |
| [`KNNWindowInferer`](../api/inferers/knn_window.md)             | Crops of the $k$ nearest points around a center                  |
| [`PotentialSphereInferer`](../api/inferers/potential_sphere.md) | Spheres of fixed radius around a center                          |
| [`TTAInferer`](../api/inferers/tta.md)                          | Another inferer run under several views, scores averaged         |
| [`PartRefinementInferer`](../api/inferers/part_refinement.md)   | Another inferer, then a neighbor vote on implausible part labels |
