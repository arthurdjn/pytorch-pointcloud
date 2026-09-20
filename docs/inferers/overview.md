# Inferers

Inferers run a model over a point cloud at inference time. Several of them split a large cloud into smaller chunks, to keep inference efficient and within memory.

```{.python notest}
from torch_pointcloud.inferers import SlidingWindowInferer, TTAInferer
from torch_pointcloud.transforms import Compose, RandomFlip, RandomRotate

inferer = TTAInferer(
    base=SlidingWindowInferer(block_size=6.0),
    transforms=Compose([
        RandomRotate(keys="pos", angle_range=(-180.0, 180.0), axis=2, p=1.0),
        RandomFlip(keys="pos", axes=[0, 1], p=0.5),
    ]),
    num_passes=4,
    aggregate="mean",
)

data = {"pos": torch.randn(1000, 3), "x": torch.randn(1000, 3), "batch": torch.zeros(1000)}
probs = inferer(data, predictor=lambda d: model(d["x"], d["pos"], d["batch"]))
```

## Per-fragment transforms

The inferers that split the scene take a `transform`, applied to every block, sub-cloud, sphere or crop after it is sliced out of the scene and before it reaches the predictor. This is where the preprocessing that depends on the fragment belongs: centering, feature stacking, padding, voxelization.

A transform that changes the number of rows records a source-to-predictor index map. Pass the key of that map as `inverse_key` and the inferer gathers the predictions back to the fragment's own points:

```python
import torch_pointcloud.transforms as T
from torch_pointcloud.inferers import VoxelPartitionInferer
from torch_pointcloud.utils.data import DataKeys

inferer = VoxelPartitionInferer(
    voxel_size=0.04,
    transform=T.DivisiblePad(num_samples=4096, dst_inverse_key=DataKeys.INVERSE),
    inverse_key=DataKeys.INVERSE,
)
```

`SlidingWindowInferer`, `VoxelPartitionInferer`, `KNNWindowInferer` and `PotentialSphereInferer` all take `inverse_key`. The map is removed from the fragment before the predictor is called, and a transform that changes the row count without recording one raises.

A step belongs on the dataset when it needs the whole scene: label remapping, scene statistics such as the room extent, a subsampling of the scene done once. It belongs in the inferer when it depends on the fragment: centering on the fragment, coordinates relative to the block, grid coordinates, padding, and the feature stack built from them. The inferer returns one prediction per point of the scene it was given, scored against that scene's labels.

The transform registered with a checkpoint is the single-pass pipeline, where the fragment is the whole scene. Depending on the protocol its steps go to the dataset, to the inferer, or are split between the two, and the sampling step (`Voxelize`, a crop) is the one the inferer replaces.

## Which one to use

| Inferer                                                         | Runs the predictor on                                 | Reproduces                                             |
| --------------------------------------------------------------- | ----------------------------------------------------- | ------------------------------------------------------ |
| [`SimpleInferer`](../api/inferers/simple.md)                    | the whole scene, once                                 | single-pass evaluation, the Lightning default          |
| [`SlidingWindowInferer`](../api/inferers/sliding_window.md)     | cubic blocks on a regular grid                        | block-based S3DIS / ScanNet protocols                  |
| [`VoxelPartitionInferer`](../api/inferers/voxel_partition.md)   | whole-extent downsamples, one point per voxel         | the fragment protocol of sparse and point transformers |
| [`KNNWindowInferer`](../api/inferers/knn_window.md)             | fixed-budget kNN crops around the least-covered point | possibility-driven crop voting (RandLA-Net)            |
| [`PotentialSphereInferer`](../api/inferers/potential_sphere.md) | radius spheres drawn from a potential grid            | potential sphere voting (KPConv)                       |
| [`TTAInferer`](../api/inferers/tta.md)                          | any base inferer, under several views                 | test-time augmentation and voting                      |
| [`PartRefinementInferer`](../api/inferers/part_refinement.md)   | any base inferer, then a neighbor vote                | part-segmentation label refinement                     |
