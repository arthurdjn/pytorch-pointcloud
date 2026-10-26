# Inferers

An inferer decides how a model is run at test time: on the whole scene, on fragments of it, or under several views, and merges the partial outputs. Outputs merge differently per task, so each inferer is named after the task it serves. [`SimpleInferer`](../api/inferers/simple.md) runs one forward pass on the whole scene and serves every task.

## Segmentation

Segmentation inferers return one prediction per point, averaged over the fragments and views that cover it.

```python
import torch
from torch_pointcloud import create_model
from torch_pointcloud.inferers import SlidingWindowSegmentationInferer

scene = {
    "pos": torch.rand(20_000, 3) * 10,
    "x": torch.rand(20_000, 9),
    "batch": torch.zeros(20_000, dtype=torch.long),
}
model = create_model("pointnet.s3dis-area5").eval()

inferer = SlidingWindowSegmentationInferer(block_size=5.0, softmax=True)
scores = inferer(scene, predictor=lambda d: model(d["x"], d["pos"], d["batch"]))
```

| Inferer                                                                     | Description                                                      |
| --------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| [`SlidingWindowSegmentationInferer`](../api/inferers/sliding_window.md)     | Blocks on a regular grid, predicted one at a time                |
| [`VoxelPartitionSegmentationInferer`](../api/inferers/voxel_partition.md)   | Passes of one point per voxel, each over the whole scene         |
| [`KNNWindowSegmentationInferer`](../api/inferers/knn_window.md)             | Crops of the $k$ nearest points around a center                  |
| [`PotentialSphereSegmentationInferer`](../api/inferers/potential_sphere.md) | Spheres of fixed radius around a center                          |
| [`TTASegmentationInferer`](../api/inferers/tta.md)                          | Another inferer run under several views, scores averaged         |
| [`PartRefinementInferer`](../api/inferers/part_refinement.md)               | Another inferer, then a neighbor vote on implausible part labels |

## Detection

Detection inferers return the raw boxes of every pass, mapped back to the coordinates of the scene. Filter them with a score threshold and `nms3d` as you would a single pass: the NMS also removes the duplicates across passes.

```python
import torch
from torch_pointcloud import create_model
from torch_pointcloud.inferers import TTADetectionInferer
from torch_pointcloud.utils.data import DataKeys, select_inputs

scene = {
    "pos": torch.rand(20_000, 3) * torch.tensor([60.0, 60.0, 3.0]) - torch.tensor([0.0, 30.0, 2.0]),
    "intensity": torch.rand(20_000, 1),
    "batch": torch.zeros(20_000, dtype=torch.long),
}
model, info = create_model("pointpillars.kitti.openpcdet", task="detection", return_info=True)
model.eval()

inferer = TTADetectionInferer(flips=[(), (1,)], transform=info["transform"], cat_keys=[DataKeys.POS_VOXEL])
detections = inferer(scene, predictor=lambda d: model.decode(model(*select_inputs(d, info["input_keys"]))))
```

| Inferer                                          | Description                                                     |
| ------------------------------------------------ | --------------------------------------------------------------- |
| [`TTADetectionInferer`](../api/inferers/tta.md) | One pass per rotated, scaled or flipped view, boxes mapped back |
